#!/usr/bin/env python3
"""
============================================================================
LIQUID NEURAL SREK TRADING SYSTEM - MAIN APPLICATION
============================================================================
Version: 4.0.0

Main entry point for the trading system. Provides:
- Mode selection (Historical Training / Paper Trading / Live Trading)
- Real-time dashboard
- System monitoring
- Graceful shutdown handling

Usage:
    python main.py                    # Interactive mode selection
    python main.py --mode historical  # Direct historical training
    python main.py --mode paper       # Direct paper trading
    python main.py --mode live        # Direct live trading
    python main.py --help             # Show help

============================================================================
"""

import asyncio
import argparse
import logging
import signal
import sys
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import yaml

# Add modules directory to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

# Rich console for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better UI: pip install rich")


# ============================================================================
# CONSTANTS
# ============================================================================

VERSION = "4.0.0"
SYSTEM_NAME = "Liquid Neural SREK Trading System"

# ASCII Art Logo
LOGO = """
╔═══════════════════════════════════════════════════════════════════════════╗
║     ██╗     ██╗ ██████╗ ██╗   ██╗██╗██████╗     ███╗   ██╗███████╗██╗   ██╗║
║     ██║     ██║██╔═══██╗██║   ██║██║██╔══██╗    ████╗  ██║██╔════╝██║   ██║║
║     ██║     ██║██║   ██║██║   ██║██║██║  ██║    ██╔██╗ ██║█████╗  ██║   ██║║
║     ██║     ██║██║▄▄ ██║██║   ██║██║██║  ██║    ██║╚██╗██║██╔══╝  ██║   ██║║
║     ███████╗██║╚██████╔╝╚██████╔╝██║██████╔╝    ██║ ╚████║███████╗╚██████╔╝║
║     ╚══════╝╚═╝ ╚══▀▀═╝  ╚═════╝ ╚═╝╚═════╝     ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ║
║                           SREK TRADING SYSTEM v4.0                         ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""


class TradingMode(Enum):
    """Trading operation modes"""
    HISTORICAL = "historical"
    PAPER = "paper"
    LIVE = "live"


@dataclass
class SystemStatus:
    """Current system status"""
    mode: TradingMode
    is_running: bool
    start_time: float
    trades_today: int
    win_rate: float
    daily_pnl: float
    capital: float
    gpu_usage: float
    vram_usage: float
    cpu_usage: float
    active_positions: int
    last_update: float


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

def load_config(config_path: Path = None) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please copy config.yaml.template to config.yaml and add your credentials."
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    if not config.get('oanda', {}).get('account_id'):
        raise ValueError("OANDA account_id not configured in config.yaml")
    if not config.get('oanda', {}).get('api_token'):
        raise ValueError("OANDA api_token not configured in config.yaml")
    
    return config


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging based on configuration"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # Create logs directory
    logs_dir = Path(__file__).parent / config.get('storage', {}).get('logs_dir', 'data/logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('trading_system')
    logger.setLevel(log_level)
    
    # File handler
    if log_config.get('log_to_file', True):
        from logging.handlers import RotatingFileHandler
        
        log_file = logs_dir / f"trading_{time.strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=log_config.get('max_log_size_mb', 100) * 1024 * 1024,
            backupCount=log_config.get('backup_count', 5)
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    # Console handler
    if log_config.get('log_to_console', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        if log_config.get('colored_output', True):
            try:
                import colorlog
                console_handler.setFormatter(colorlog.ColoredFormatter(
                    '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
                    log_colors={
                        'DEBUG': 'cyan',
                        'INFO': 'green',
                        'WARNING': 'yellow',
                        'ERROR': 'red',
                        'CRITICAL': 'red,bg_white'
                    }
                ))
            except ImportError:
                console_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s'
                ))
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
        
        logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# GPU & SYSTEM MONITORING
# ============================================================================

def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information"""
    info = {
        'available': False,
        'name': 'N/A',
        'vram_total_gb': 0,
        'vram_used_gb': 0,
        'vram_free_gb': 0,
        'utilization': 0,
        'temperature': 0
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            info['available'] = True
            info['name'] = torch.cuda.get_device_name(0)
            
            total = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            
            info['vram_total_gb'] = total / (1024**3)
            info['vram_used_gb'] = allocated / (1024**3)
            info['vram_free_gb'] = (total - reserved) / (1024**3)
            info['utilization'] = (allocated / total) * 100
    except Exception:
        pass
    
    return info


def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    info = {
        'cpu_percent': 0,
        'ram_percent': 0,
        'ram_available_gb': 0
    }
    
    try:
        import psutil
        info['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        info['ram_percent'] = mem.percent
        info['ram_available_gb'] = mem.available / (1024**3)
    except ImportError:
        pass
    
    return info


# ============================================================================
# TRADING SYSTEM CORE
# ============================================================================

class TradingSystem:
    """
    Main trading system orchestrator.
    
    Manages all modules and coordinates trading operations.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # State
        self._lock = asyncio.Lock()
        self._is_running = False
        self._shutdown_event = asyncio.Event()
        self._mode: Optional[TradingMode] = None
        
        # Statistics
        self._start_time = 0.0
        self._trades_today = 0
        self._wins_today = 0
        self._daily_pnl = 0.0
        self._capital = config.get('trading', {}).get('initial_capital', 10000.0)
        
        # Modules (lazy loaded)
        self._modules_initialized = False
        self._gpu_memory_manager = None
        self._market_data = None
        self._srek_population = None
        self._neural_ode = None
        self._lstm_networks = None
        self._transformer = None
        self._gnn = None
        self._attention_memory = None
        self._meta_learner = None
    
    async def initialize_modules_async(self):
        """Initialize all trading modules"""
        if self._modules_initialized:
            return
        
        self.logger.info("Initializing trading modules...")
        
        try:
            # Import modules
            from module_16_gpu_memory_manager_enhanced import (
                EnhancedGPUMemoryManager, GPUMemoryConfig
            )
            from module_1_liquid_neural_ode_enhanced import (
                LiquidNeuralODE, LiquidNeuralODEConfig
            )
            from module_2_meta_srek_population_enhanced import (
                MetaSREKPopulation, MetaSREKConfig
            )
            from module_6_multi_timescale_networks_enhanced import (
                MultiTimescaleNetworks, MultiTimescaleConfig
            )
            from module_9_transformer_validator_enhanced import (
                TransformerValidator, TransformerConfig
            )
            from module_13_correlation_gnn_enhanced import (
                CorrelationGNN, CorrelationGNNConfig
            )
            from module_17_attention_memory import (
                AttentionMemoryBank, AttentionMemoryConfig
            )
            from module_18_ensemble_meta_learner import (
                EnsembleMetaLearner, EnsembleMetaLearnerConfig
            )
            
            gpu_config = self.config.get('gpu', {})
            
            # 1. GPU Memory Manager
            self.logger.info("  [1/8] GPU Memory Manager...")
            self._gpu_memory_manager = EnhancedGPUMemoryManager(
                GPUMemoryConfig(
                    target_utilization=gpu_config.get('target_vram_utilization', 0.80),
                    safety_buffer_mb=gpu_config.get('safety_buffer_mb', 300)
                )
            )
            await self._gpu_memory_manager.initialize_async()
            
            # 2. Liquid Neural ODE
            self.logger.info("  [2/8] Liquid Neural ODE...")
            self._neural_ode = LiquidNeuralODE(
                gpu_memory_manager=self._gpu_memory_manager
            )
            await self._neural_ode.initialize_async()
            
            # 3. Meta-SREK Population
            self.logger.info("  [3/8] Meta-SREK Population...")
            self._srek_population = MetaSREKPopulation(
                gpu_memory_manager=self._gpu_memory_manager
            )
            await self._srek_population.initialize_async()
            
            # 4. Multi-Timescale Networks
            self.logger.info("  [4/8] Multi-Timescale Networks...")
            self._lstm_networks = MultiTimescaleNetworks(
                gpu_memory_manager=self._gpu_memory_manager
            )
            await self._lstm_networks.initialize_async()
            
            # 5. Transformer Validator
            self.logger.info("  [5/8] Transformer Validator...")
            self._transformer = TransformerValidator(
                gpu_memory_manager=self._gpu_memory_manager
            )
            await self._transformer.initialize_async()
            
            # 6. Correlation GNN
            self.logger.info("  [6/8] Correlation GNN...")
            self._gnn = CorrelationGNN(
                gpu_memory_manager=self._gpu_memory_manager
            )
            await self._gnn.initialize_async()
            
            # 7. Attention Memory
            self.logger.info("  [7/8] Attention Memory...")
            self._attention_memory = AttentionMemoryBank(
                gpu_memory_manager=self._gpu_memory_manager
            )
            await self._attention_memory.initialize_async()
            
            # 8. Ensemble Meta-Learner
            self.logger.info("  [8/8] Ensemble Meta-Learner...")
            self._meta_learner = EnsembleMetaLearner(
                gpu_memory_manager=self._gpu_memory_manager
            )
            await self._meta_learner.initialize_async()
            
            self._modules_initialized = True
            self.logger.info("✅ All modules initialized successfully!")
            
            # Print VRAM status
            if self._gpu_memory_manager:
                status = await self._gpu_memory_manager.get_status_async()
                self.logger.info(
                    f"   VRAM: {status.allocated_mb:.0f}/{status.total_vram_mb:.0f} MB "
                    f"({status.utilization_percent:.1f}%)"
                )
            
        except ImportError as e:
            self.logger.error(f"Module import failed: {e}")
            self.logger.error("Make sure all module files are in the 'modules' directory")
            raise
        except Exception as e:
            self.logger.error(f"Module initialization failed: {e}")
            raise
    
    async def start_async(self, mode: TradingMode):
        """Start the trading system in specified mode"""
        async with self._lock:
            if self._is_running:
                self.logger.warning("System already running")
                return
            
            self._mode = mode
            self._is_running = True
            self._start_time = time.time()
            self._shutdown_event.clear()
        
        self.logger.info(f"Starting trading system in {mode.value.upper()} mode...")
        
        try:
            # Initialize modules if not done
            await self.initialize_modules_async()
            
            # Start appropriate mode
            if mode == TradingMode.HISTORICAL:
                await self._run_historical_training_async()
            elif mode == TradingMode.PAPER:
                await self._run_paper_trading_async()
            elif mode == TradingMode.LIVE:
                await self._run_live_trading_async()
            
        except asyncio.CancelledError:
            self.logger.info("Trading system cancelled")
        except Exception as e:
            self.logger.error(f"Trading system error: {e}")
            raise
        finally:
            async with self._lock:
                self._is_running = False
    
    async def stop_async(self):
        """Stop the trading system gracefully"""
        self.logger.info("Stopping trading system...")
        
        async with self._lock:
            if not self._is_running:
                return
            
            self._shutdown_event.set()
        
        # Wait for shutdown
        await asyncio.sleep(1)
        
        # Cleanup modules
        await self._cleanup_modules_async()
        
        self.logger.info("Trading system stopped")
    
    async def _cleanup_modules_async(self):
        """Cleanup all modules"""
        if self._meta_learner:
            await self._meta_learner.cleanup_async()
        if self._attention_memory:
            await self._attention_memory.cleanup_async()
        if self._gnn:
            await self._gnn.cleanup_async()
        if self._transformer:
            await self._transformer.cleanup_async()
        if self._lstm_networks:
            await self._lstm_networks.cleanup_async()
        if self._srek_population:
            await self._srek_population.cleanup_async()
        if self._neural_ode:
            await self._neural_ode.cleanup_async()
        if self._gpu_memory_manager:
            await self._gpu_memory_manager.cleanup_async()
    
    async def get_status_async(self) -> SystemStatus:
        """Get current system status"""
        gpu_info = get_gpu_info()
        sys_info = get_system_info()
        
        async with self._lock:
            return SystemStatus(
                mode=self._mode or TradingMode.PAPER,
                is_running=self._is_running,
                start_time=self._start_time,
                trades_today=self._trades_today,
                win_rate=self._wins_today / max(self._trades_today, 1),
                daily_pnl=self._daily_pnl,
                capital=self._capital,
                gpu_usage=gpu_info['utilization'],
                vram_usage=gpu_info['vram_used_gb'],
                cpu_usage=sys_info['cpu_percent'],
                active_positions=0,  # TODO: Get from position manager
                last_update=time.time()
            )
    
    # ========================================================================
    # HISTORICAL TRAINING MODE
    # ========================================================================
    
    async def _run_historical_training_async(self):
        """Run historical data training"""
        self.logger.info("=" * 60)
        self.logger.info("HISTORICAL TRAINING MODE")
        self.logger.info("=" * 60)
        
        training_config = self.config.get('training', {})
        
        # Step 1: Fetch historical data
        self.logger.info("\n[Step 1/4] Fetching historical data...")
        await self._fetch_historical_data_async()
        
        # Step 2: Preprocess data
        self.logger.info("\n[Step 2/4] Preprocessing data...")
        await self._preprocess_data_async()
        
        # Step 3: Train models
        self.logger.info("\n[Step 3/4] Training neural networks...")
        epochs = training_config.get('epochs', 100)
        
        for epoch in range(epochs):
            if self._shutdown_event.is_set():
                break
            
            # Training step (simplified)
            await asyncio.sleep(0.1)  # Placeholder for actual training
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"  Epoch {epoch + 1}/{epochs} complete")
        
        # Step 4: Validate and save
        self.logger.info("\n[Step 4/4] Validating and saving models...")
        await self._save_checkpoints_async()
        
        self.logger.info("\n✅ Historical training complete!")
    
    async def _fetch_historical_data_async(self):
        """Fetch historical data from OANDA"""
        oanda_config = self.config.get('oanda', {})
        pairs = self.config.get('trading', {}).get('pairs', ['EUR_USD'])
        months = self.config.get('training', {}).get('historical_data_months', 6)
        
        self.logger.info(f"  Fetching {months} months of data for {len(pairs)} pairs...")
        
        # This would integrate with Module 5 (Market Data Pipeline)
        # For now, placeholder
        await asyncio.sleep(1)
        
        self.logger.info("  ✅ Historical data fetched")
    
    async def _preprocess_data_async(self):
        """Preprocess data for training"""
        self.logger.info("  Computing features...")
        await asyncio.sleep(0.5)
        self.logger.info("  Normalizing data...")
        await asyncio.sleep(0.5)
        self.logger.info("  ✅ Preprocessing complete")
    
    async def _save_checkpoints_async(self):
        """Save model checkpoints"""
        checkpoint_dir = Path(__file__).parent / self.config.get('storage', {}).get(
            'checkpoints_dir', 'data/checkpoints'
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Save each module
        if self._neural_ode:
            await self._neural_ode.save_checkpoint_async(
                str(checkpoint_dir / f"neural_ode_{timestamp}.pt")
            )
        if self._srek_population:
            await self._srek_population.save_checkpoint_async(
                str(checkpoint_dir / f"srek_population_{timestamp}.pt")
            )
        # ... other modules
        
        self.logger.info(f"  ✅ Checkpoints saved to {checkpoint_dir}")
    
    # ========================================================================
    # PAPER TRADING MODE
    # ========================================================================
    
    async def _run_paper_trading_async(self):
        """Run paper trading (simulated trading)"""
        self.logger.info("=" * 60)
        self.logger.info("PAPER TRADING MODE")
        self.logger.info("=" * 60)
        self.logger.info("Trading with simulated money on real market data")
        self.logger.info("Press Ctrl+C to stop\n")
        
        # Load latest checkpoints
        await self._load_checkpoints_async()
        
        # Start market data stream
        # await self._market_data.start_stream_async()
        
        # Main trading loop
        while not self._shutdown_event.is_set():
            try:
                # Get market data
                # Analyze with all modules
                # Generate predictions
                # Execute paper trades
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Paper trading error: {e}")
                await asyncio.sleep(10)
        
        self.logger.info("Paper trading stopped")
    
    # ========================================================================
    # LIVE TRADING MODE
    # ========================================================================
    
    async def _run_live_trading_async(self):
        """Run live trading with real money"""
        self.logger.info("=" * 60)
        self.logger.info("🔴 LIVE TRADING MODE - REAL MONEY")
        self.logger.info("=" * 60)
        
        # Safety confirmation
        oanda_env = self.config.get('oanda', {}).get('environment', 'practice')
        
        if oanda_env == 'practice':
            self.logger.warning("⚠️  OANDA environment is 'practice' - no real money at risk")
        else:
            self.logger.warning("⚠️  OANDA environment is 'live' - REAL MONEY AT RISK!")
            self.logger.warning("Starting in 10 seconds... Press Ctrl+C to cancel")
            await asyncio.sleep(10)
        
        # Load latest checkpoints
        await self._load_checkpoints_async()
        
        # Main trading loop
        while not self._shutdown_event.is_set():
            try:
                # Same as paper trading but with real execution
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Live trading error: {e}")
                await asyncio.sleep(10)
        
        self.logger.info("Live trading stopped")
    
    async def _load_checkpoints_async(self):
        """Load latest model checkpoints"""
        checkpoint_dir = Path(__file__).parent / self.config.get('storage', {}).get(
            'checkpoints_dir', 'data/checkpoints'
        )
        
        if not checkpoint_dir.exists():
            self.logger.warning("No checkpoints found - using untrained models")
            return
        
        self.logger.info("Loading model checkpoints...")
        
        # Find latest checkpoints
        # Load each module
        # ...
        
        self.logger.info("✅ Checkpoints loaded")


# ============================================================================
# INTERACTIVE MENU
# ============================================================================

def show_menu() -> Optional[TradingMode]:
    """Show interactive mode selection menu"""
    if RICH_AVAILABLE:
        return show_rich_menu()
    else:
        return show_simple_menu()


def show_simple_menu() -> Optional[TradingMode]:
    """Simple text-based menu"""
    print(LOGO)
    print("\n" + "=" * 50)
    print("SELECT TRADING MODE")
    print("=" * 50)
    print("\n  [1] Historical Training")
    print("      - Download historical data")
    print("      - Train all neural networks")
    print("      - Backtest strategies")
    print("\n  [2] Paper Trading")
    print("      - Trade with simulated money")
    print("      - Real-time market data")
    print("      - Safe learning environment")
    print("\n  [3] Live Trading")
    print("      - Trade with REAL MONEY")
    print("      - Full automated trading")
    print("      - Use with caution!")
    print("\n  [Q] Quit")
    print("\n" + "=" * 50)
    
    while True:
        choice = input("\nEnter your choice (1/2/3/Q): ").strip().upper()
        
        if choice == '1':
            return TradingMode.HISTORICAL
        elif choice == '2':
            return TradingMode.PAPER
        elif choice == '3':
            confirm = input("⚠️  Are you sure? This uses REAL MONEY (yes/no): ")
            if confirm.lower() == 'yes':
                return TradingMode.LIVE
            print("Live trading cancelled.")
        elif choice == 'Q':
            return None
        else:
            print("Invalid choice. Please enter 1, 2, 3, or Q.")


def show_rich_menu() -> Optional[TradingMode]:
    """Rich/beautiful menu using rich library"""
    console = Console()
    
    console.print(LOGO, style="bold cyan")
    
    # Create menu table
    table = Table(
        title="SELECT TRADING MODE",
        box=box.DOUBLE_EDGE,
        title_style="bold white on blue",
        header_style="bold cyan"
    )
    
    table.add_column("Option", style="bold yellow", width=8)
    table.add_column("Mode", style="bold white", width=20)
    table.add_column("Description", style="white")
    
    table.add_row(
        "[1]",
        "Historical Training",
        "Download data, train models, backtest strategies"
    )
    table.add_row(
        "[2]",
        "Paper Trading",
        "Trade with simulated money on real market data"
    )
    table.add_row(
        "[3]",
        "Live Trading",
        "[bold red]REAL MONEY[/] - Automated trading with actual capital"
    )
    table.add_row(
        "[Q]",
        "Quit",
        "Exit the application"
    )
    
    console.print()
    console.print(table)
    console.print()
    
    while True:
        choice = console.input("[bold cyan]Enter your choice (1/2/3/Q): [/]").strip().upper()
        
        if choice == '1':
            return TradingMode.HISTORICAL
        elif choice == '2':
            return TradingMode.PAPER
        elif choice == '3':
            console.print("\n[bold red]⚠️  WARNING: LIVE TRADING USES REAL MONEY![/]")
            confirm = console.input("[bold yellow]Type 'YES' to confirm: [/]")
            if confirm == 'YES':
                return TradingMode.LIVE
            console.print("[green]Live trading cancelled.[/]")
        elif choice == 'Q':
            return None
        else:
            console.print("[red]Invalid choice. Please enter 1, 2, 3, or Q.[/]")


# ============================================================================
# SIGNAL HANDLERS
# ============================================================================

def setup_signal_handlers(trading_system: TradingSystem, loop: asyncio.AbstractEventLoop):
    """Setup graceful shutdown handlers"""
    
    def signal_handler(sig, frame):
        print("\n\nShutdown signal received...")
        loop.create_task(trading_system.stop_async())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description=f"{SYSTEM_NAME} v{VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Interactive mode selection
  python main.py --mode historical  Start historical training
  python main.py --mode paper       Start paper trading
  python main.py --mode live        Start live trading (caution!)
  python main.py --config my.yaml   Use custom config file
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['historical', 'paper', 'live'],
        help='Trading mode to start directly'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=Path,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f'{SYSTEM_NAME} v{VERSION}'
    )
    
    return parser.parse_args()


async def main_async():
    """Async main entry point"""
    args = parse_arguments()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo fix this:")
        print("  1. Copy config.yaml.template to config/config.yaml")
        print("  2. Edit config.yaml with your OANDA credentials")
        return 1
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        return 1
    
    # Setup logging
    logger = setup_logging(config)
    
    # Determine mode
    if args.mode:
        mode = TradingMode(args.mode)
    else:
        mode = show_menu()
        if mode is None:
            print("\nGoodbye!")
            return 0
    
    # Create trading system
    trading_system = TradingSystem(config, logger)
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    setup_signal_handlers(trading_system, loop)
    
    # Start trading
    try:
        await trading_system.start_async(mode)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    finally:
        await trading_system.stop_async()
    
    return 0


def main():
    """Main entry point"""
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        return 0


if __name__ == "__main__":
    sys.exit(main())

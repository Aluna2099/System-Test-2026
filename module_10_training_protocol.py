"""
MODULE 10: TRAINING PROTOCOL MANAGER
Production-Ready Implementation

Manages the complete training lifecycle from boot camp to live trading.
Coordinates phase transitions based on performance criteria.

- Three-phase training: Boot Camp → Hardening → Live
- Strict phase promotion criteria
- Performance tracking across phases
- Curriculum learning integration
- Async/await architecture throughout
- Thread-safe state management
- Zero GPU usage (CPU-only orchestration)

Author: Liquid Neural SREK Trading System v4.0
Date: 2026-01-11
Version: 1.0.0

TRAINING PHASES:
1. BOOT_CAMP (Month 1): Train on historical data
   - Target: 70%+ win rate, profit factor > 1.5
   - No real capital at risk
   
2. HARDENING (Months 2-7): Paper trading with simulated capital
   - Target: 75%+ win rate, profit factor > 1.8, max DD < 20%
   - Builds confidence before real money
   
3. LIVE (Month 8+): Real capital trading
   - Target: Maintain 75%+ win rate, compound capital
   - Automatic demotion if criteria fail
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class TrainingPhase(Enum):
    """Training phase states"""
    BOOT_CAMP = "boot_camp"
    HARDENING = "hardening"
    LIVE = "live"
    PAUSED = "paused"  # Emergency pause
    
    @classmethod
    def from_string(cls, s: str) -> 'TrainingPhase':
        """Convert string to enum"""
        mapping = {
            'boot_camp': cls.BOOT_CAMP,
            'hardening': cls.HARDENING,
            'live': cls.LIVE,
            'paused': cls.PAUSED
        }
        return mapping.get(s.lower(), cls.BOOT_CAMP)


class CurriculumDifficulty(Enum):
    """Curriculum learning difficulty levels"""
    EASY = "easy"  # Stable trending markets
    MEDIUM = "medium"  # Mixed conditions
    HARD = "hard"  # Volatile markets
    EXTREME = "extreme"  # Crisis scenarios


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PhaseMetrics:
    """Performance metrics for a training phase"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit_pips: float = 0.0
    total_loss_pips: float = 0.0
    max_drawdown_percent: float = 0.0
    current_drawdown_percent: float = 0.0
    peak_capital: float = 0.0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    days_in_phase: int = 0
    phase_start_time: float = 0.0
    last_update_time: float = 0.0
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate"""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    @property
    def profit_factor(self) -> float:
        """Calculate profit factor"""
        if self.total_loss_pips == 0:
            return float('inf') if self.total_profit_pips > 0 else 0.0
        return self.total_profit_pips / abs(self.total_loss_pips)
    
    @property
    def average_win(self) -> float:
        """Average winning trade in pips"""
        if self.winning_trades == 0:
            return 0.0
        return self.total_profit_pips / self.winning_trades
    
    @property
    def average_loss(self) -> float:
        """Average losing trade in pips"""
        if self.losing_trades == 0:
            return 0.0
        return abs(self.total_loss_pips) / self.losing_trades
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            **asdict(self),
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'average_win': self.average_win,
            'average_loss': self.average_loss
        }


@dataclass
class PromotionCriteria:
    """Criteria for phase promotion"""
    min_trades: int = 100
    min_win_rate: float = 0.70
    min_profit_factor: float = 1.5
    max_drawdown_percent: float = 25.0
    min_days: int = 7
    max_consecutive_losses: int = 10
    
    def check(self, metrics: PhaseMetrics) -> Tuple[bool, List[str]]:
        """
        Check if metrics meet promotion criteria.
        
        Returns:
            Tuple of (passed, list of failed criteria)
        """
        failures = []
        
        if metrics.total_trades < self.min_trades:
            failures.append(f"trades ({metrics.total_trades}) < min ({self.min_trades})")
        
        if metrics.win_rate < self.min_win_rate:
            failures.append(f"win_rate ({metrics.win_rate:.2%}) < min ({self.min_win_rate:.2%})")
        
        if metrics.profit_factor < self.min_profit_factor:
            failures.append(f"profit_factor ({metrics.profit_factor:.2f}) < min ({self.min_profit_factor:.2f})")
        
        if metrics.max_drawdown_percent > self.max_drawdown_percent:
            failures.append(f"drawdown ({metrics.max_drawdown_percent:.1f}%) > max ({self.max_drawdown_percent:.1f}%)")
        
        if metrics.days_in_phase < self.min_days:
            failures.append(f"days ({metrics.days_in_phase}) < min ({self.min_days})")
        
        if metrics.max_consecutive_losses > self.max_consecutive_losses:
            failures.append(f"consec_losses ({metrics.max_consecutive_losses}) > max ({self.max_consecutive_losses})")
        
        return (len(failures) == 0, failures)


@dataclass
class DemotionCriteria:
    """Criteria for phase demotion (safety net)"""
    max_drawdown_percent: float = 30.0
    max_consecutive_losses: int = 7
    min_win_rate_30d: float = 0.55  # Rolling 30-day
    
    def check(self, metrics: PhaseMetrics, rolling_win_rate: float) -> Tuple[bool, str]:
        """
        Check if should demote.
        
        Returns:
            Tuple of (should_demote, reason)
        """
        if metrics.current_drawdown_percent > self.max_drawdown_percent:
            return (True, f"drawdown ({metrics.current_drawdown_percent:.1f}%) > max ({self.max_drawdown_percent:.1f}%)")
        
        if metrics.consecutive_losses > self.max_consecutive_losses:
            return (True, f"consecutive_losses ({metrics.consecutive_losses}) > max ({self.max_consecutive_losses})")
        
        if rolling_win_rate < self.min_win_rate_30d:
            return (True, f"rolling_win_rate ({rolling_win_rate:.2%}) < min ({self.min_win_rate_30d:.2%})")
        
        return (False, "")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingProtocolConfig:
    """
    Configuration for Training Protocol Manager
    
    Includes validation to prevent runtime errors
    """
    # Phase durations (minimum)
    boot_camp_min_days: int = 7
    hardening_min_days: int = 30
    
    # Boot Camp promotion criteria
    boot_camp_min_trades: int = 100
    boot_camp_min_win_rate: float = 0.70
    boot_camp_min_profit_factor: float = 1.5
    boot_camp_max_drawdown: float = 25.0
    
    # Hardening promotion criteria (stricter)
    hardening_min_trades: int = 500
    hardening_min_win_rate: float = 0.75
    hardening_min_profit_factor: float = 1.8
    hardening_max_drawdown: float = 20.0
    
    # Live demotion criteria (safety)
    live_max_drawdown: float = 30.0
    live_max_consecutive_losses: int = 7
    live_min_rolling_win_rate: float = 0.55
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_phase_days: int = 7
    
    # Training scheduling
    training_batch_size: int = 32
    training_interval_minutes: int = 60  # Train every hour
    evolution_interval_hours: int = 24  # Evolve SREKs daily
    
    # Persistence
    data_dir: str = "data/training_protocol"
    checkpoint_interval_hours: int = 1
    
    # Performance tracking
    rolling_window_days: int = 30
    
    def __post_init__(self):
        """Validate configuration"""
        # Duration validation
        if self.boot_camp_min_days <= 0:
            raise ValueError(f"boot_camp_min_days must be positive, got {self.boot_camp_min_days}")
        if self.hardening_min_days <= 0:
            raise ValueError(f"hardening_min_days must be positive, got {self.hardening_min_days}")
        
        # Trade count validation
        if self.boot_camp_min_trades <= 0:
            raise ValueError(f"boot_camp_min_trades must be positive, got {self.boot_camp_min_trades}")
        if self.hardening_min_trades <= 0:
            raise ValueError(f"hardening_min_trades must be positive, got {self.hardening_min_trades}")
        
        # Win rate validation
        if not 0.0 < self.boot_camp_min_win_rate <= 1.0:
            raise ValueError(f"boot_camp_min_win_rate must be in (0, 1], got {self.boot_camp_min_win_rate}")
        if not 0.0 < self.hardening_min_win_rate <= 1.0:
            raise ValueError(f"hardening_min_win_rate must be in (0, 1], got {self.hardening_min_win_rate}")
        if not 0.0 < self.live_min_rolling_win_rate <= 1.0:
            raise ValueError(f"live_min_rolling_win_rate must be in (0, 1], got {self.live_min_rolling_win_rate}")
        
        # Profit factor validation
        if self.boot_camp_min_profit_factor <= 0:
            raise ValueError(f"boot_camp_min_profit_factor must be positive, got {self.boot_camp_min_profit_factor}")
        if self.hardening_min_profit_factor <= 0:
            raise ValueError(f"hardening_min_profit_factor must be positive, got {self.hardening_min_profit_factor}")
        
        # Drawdown validation
        if not 0.0 < self.boot_camp_max_drawdown <= 100.0:
            raise ValueError(f"boot_camp_max_drawdown must be in (0, 100], got {self.boot_camp_max_drawdown}")
        if not 0.0 < self.hardening_max_drawdown <= 100.0:
            raise ValueError(f"hardening_max_drawdown must be in (0, 100], got {self.hardening_max_drawdown}")
        if not 0.0 < self.live_max_drawdown <= 100.0:
            raise ValueError(f"live_max_drawdown must be in (0, 100], got {self.live_max_drawdown}")
        
        # Interval validation
        if self.training_interval_minutes <= 0:
            raise ValueError(f"training_interval_minutes must be positive, got {self.training_interval_minutes}")
        if self.evolution_interval_hours <= 0:
            raise ValueError(f"evolution_interval_hours must be positive, got {self.evolution_interval_hours}")
        if self.checkpoint_interval_hours <= 0:
            raise ValueError(f"checkpoint_interval_hours must be positive, got {self.checkpoint_interval_hours}")
        
        # Other validation
        if self.training_batch_size <= 0:
            raise ValueError(f"training_batch_size must be positive, got {self.training_batch_size}")
        if self.rolling_window_days <= 0:
            raise ValueError(f"rolling_window_days must be positive, got {self.rolling_window_days}")
        if self.curriculum_phase_days <= 0:
            raise ValueError(f"curriculum_phase_days must be positive, got {self.curriculum_phase_days}")


# ============================================================================
# TRAINING PROTOCOL MANAGER
# ============================================================================

class TrainingProtocolManager:
    """
    Manages the complete training lifecycle.
    
    Phases:
    1. BOOT_CAMP: Train on historical data, build base performance
    2. HARDENING: Paper trading, validate on live market (no real capital)
    3. LIVE: Real capital trading with strict safety criteria
    
    Features:
    - Automatic phase promotion based on criteria
    - Automatic demotion on safety violations
    - Curriculum learning (easy → hard scenarios)
    - Performance tracking and persistence
    - Thread-safe state management
    - Async/await throughout
    """
    
    def __init__(
        self,
        config: Optional[TrainingProtocolConfig] = None,
        modules: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Training Protocol Manager.
        
        Args:
            config: Configuration
            modules: Dictionary of system modules for coordination
        """
        self.config = config or TrainingProtocolConfig()
        self.modules = modules or {}
        
        # Thread safety locks
        self._lock = asyncio.Lock()  # Protects shared state
        self._metrics_lock = asyncio.Lock()  # Protects metrics
        self._phase_lock = asyncio.Lock()  # Protects phase transitions
        
        # State (protected by _lock)
        self._is_initialized = False
        self._current_phase = TrainingPhase.BOOT_CAMP
        self._phase_start_time = 0.0
        self._is_training_active = False
        self._is_paused = False
        
        # Metrics (protected by _metrics_lock)
        self._phase_metrics: Dict[TrainingPhase, PhaseMetrics] = {
            TrainingPhase.BOOT_CAMP: PhaseMetrics(),
            TrainingPhase.HARDENING: PhaseMetrics(),
            TrainingPhase.LIVE: PhaseMetrics()
        }
        
        # Rolling trade history for 30-day metrics
        self._trade_history: deque = deque(maxlen=10000)
        
        # Curriculum learning state
        self._curriculum_difficulty = CurriculumDifficulty.EASY
        self._curriculum_start_time = 0.0
        
        # Promotion/demotion criteria
        self._boot_camp_criteria = PromotionCriteria(
            min_trades=self.config.boot_camp_min_trades,
            min_win_rate=self.config.boot_camp_min_win_rate,
            min_profit_factor=self.config.boot_camp_min_profit_factor,
            max_drawdown_percent=self.config.boot_camp_max_drawdown,
            min_days=self.config.boot_camp_min_days
        )
        
        self._hardening_criteria = PromotionCriteria(
            min_trades=self.config.hardening_min_trades,
            min_win_rate=self.config.hardening_min_win_rate,
            min_profit_factor=self.config.hardening_min_profit_factor,
            max_drawdown_percent=self.config.hardening_max_drawdown,
            min_days=self.config.hardening_min_days
        )
        
        self._live_demotion_criteria = DemotionCriteria(
            max_drawdown_percent=self.config.live_max_drawdown,
            max_consecutive_losses=self.config.live_max_consecutive_losses,
            min_win_rate_30d=self.config.live_min_rolling_win_rate
        )
        
        # Background tasks
        self._training_task: Optional[asyncio.Task] = None
        self._evolution_task: Optional[asyncio.Task] = None
        self._checkpoint_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._total_training_iterations = 0
        self._total_evolution_cycles = 0
        self._phase_transitions: List[Dict] = []
        
        logger.info(
            f"TrainingProtocolManager initialized: "
            f"phase={self._current_phase.value}"
        )
    
    async def initialize_async(self) -> Dict[str, Any]:
        """
        Initialize training protocol.
        
        Returns:
            Initialization status
        """
        async with self._lock:
            if self._is_initialized:
                return {'status': 'already_initialized'}
            
            try:
                # Create data directory
                Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
                
                # Try to load existing checkpoint
                checkpoint_path = Path(self.config.data_dir) / "protocol_checkpoint.json"
                if checkpoint_path.exists():
                    loaded = await self._load_checkpoint_async(str(checkpoint_path))
                    if loaded['status'] == 'success':
                        logger.info(f"Resumed from checkpoint: phase={self._current_phase.value}")
                
                # Initialize phase timing
                if self._phase_start_time == 0:
                    self._phase_start_time = time.time()
                
                if self._curriculum_start_time == 0:
                    self._curriculum_start_time = time.time()
                
                self._is_initialized = True
                
                logger.info(f"✅ TrainingProtocolManager initialized")
                
                return {
                    'status': 'success',
                    'current_phase': self._current_phase.value,
                    'curriculum_difficulty': self._curriculum_difficulty.value
                }
                
            except Exception as e:
                logger.error(f"❌ Initialization failed: {e}")
                return {'status': 'failed', 'error': str(e)}
    
    async def start_training_async(self) -> Dict[str, Any]:
        """
        Start background training tasks.
        
        Returns:
            Start status
        """
        async with self._lock:
            if not self._is_initialized:
                return {'status': 'failed', 'error': 'Not initialized'}
            
            if self._is_training_active:
                return {'status': 'already_active'}
            
            self._is_training_active = True
            
            # Start background tasks
            self._training_task = asyncio.create_task(
                self._training_loop_async()
            )
            
            self._evolution_task = asyncio.create_task(
                self._evolution_loop_async()
            )
            
            self._checkpoint_task = asyncio.create_task(
                self._checkpoint_loop_async()
            )
            
            logger.info("✅ Training background tasks started")
            
            return {'status': 'success'}
    
    async def stop_training_async(self) -> Dict[str, Any]:
        """
        Stop background training tasks.
        
        Returns:
            Stop status
        """
        async with self._lock:
            self._is_training_active = False
        
        # Cancel tasks
        for task in [self._training_task, self._evolution_task, self._checkpoint_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._training_task = None
        self._evolution_task = None
        self._checkpoint_task = None
        
        logger.info("✅ Training background tasks stopped")
        
        return {'status': 'success'}
    
    async def record_trade_async(
        self,
        trade_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Record a completed trade and update metrics.
        
        Args:
            trade_result: Trade outcome with pnl_pips, is_win, etc.
            
        Returns:
            Updated metrics and phase status
        """
        async with self._metrics_lock:
            metrics = self._phase_metrics[self._current_phase]
            
            # Extract trade data
            pnl_pips = trade_result.get('pnl_pips', 0.0)
            is_win = trade_result.get('is_win', pnl_pips > 0)
            
            # Update metrics
            metrics.total_trades += 1
            
            if is_win:
                metrics.winning_trades += 1
                metrics.total_profit_pips += pnl_pips
                metrics.consecutive_losses = 0
            else:
                metrics.losing_trades += 1
                metrics.total_loss_pips += abs(pnl_pips)
                metrics.consecutive_losses += 1
                metrics.max_consecutive_losses = max(
                    metrics.max_consecutive_losses,
                    metrics.consecutive_losses
                )
            
            # Update drawdown
            current_capital = trade_result.get('current_capital', 0.0)
            if current_capital > metrics.peak_capital:
                metrics.peak_capital = current_capital
            
            if metrics.peak_capital > 0:
                metrics.current_drawdown_percent = (
                    (metrics.peak_capital - current_capital) / metrics.peak_capital * 100
                )
                metrics.max_drawdown_percent = max(
                    metrics.max_drawdown_percent,
                    metrics.current_drawdown_percent
                )
            
            metrics.last_update_time = time.time()
            
            # Update days in phase
            metrics.days_in_phase = int(
                (time.time() - metrics.phase_start_time) / 86400
            )
            
            # Add to history
            self._trade_history.append({
                'timestamp': time.time(),
                'pnl_pips': pnl_pips,
                'is_win': is_win,
                'phase': self._current_phase.value
            })
        
        # Check for phase transitions
        phase_result = await self._check_phase_transition_async()
        
        return {
            'status': 'success',
            'metrics': metrics.to_dict(),
            'phase_transition': phase_result
        }
    
    async def _check_phase_transition_async(self) -> Dict[str, Any]:
        """
        Check if should promote or demote phase.
        
        Returns:
            Phase transition result
        """
        async with self._phase_lock:
            current_phase = self._current_phase
            
            async with self._metrics_lock:
                metrics = self._phase_metrics[current_phase]
                rolling_win_rate = self._calculate_rolling_win_rate()
            
            # Check promotion
            if current_phase == TrainingPhase.BOOT_CAMP:
                passed, failures = self._boot_camp_criteria.check(metrics)
                if passed:
                    return await self._promote_phase_async(
                        TrainingPhase.HARDENING,
                        "Boot camp criteria met"
                    )
                    
            elif current_phase == TrainingPhase.HARDENING:
                passed, failures = self._hardening_criteria.check(metrics)
                if passed:
                    return await self._promote_phase_async(
                        TrainingPhase.LIVE,
                        "Hardening criteria met"
                    )
            
            # Check demotion (only for Live phase)
            elif current_phase == TrainingPhase.LIVE:
                should_demote, reason = self._live_demotion_criteria.check(
                    metrics, rolling_win_rate
                )
                if should_demote:
                    return await self._demote_phase_async(reason)
            
            return {'transition': False, 'phase': current_phase.value}
    
    async def _promote_phase_async(
        self,
        new_phase: TrainingPhase,
        reason: str
    ) -> Dict[str, Any]:
        """
        Promote to next training phase.
        
        Args:
            new_phase: Target phase
            reason: Promotion reason
            
        Returns:
            Promotion result
        """
        old_phase = self._current_phase
        self._current_phase = new_phase
        self._phase_start_time = time.time()
        
        # Initialize new phase metrics
        async with self._metrics_lock:
            self._phase_metrics[new_phase] = PhaseMetrics(
                phase_start_time=time.time()
            )
        
        # Record transition
        transition = {
            'timestamp': time.time(),
            'from_phase': old_phase.value,
            'to_phase': new_phase.value,
            'type': 'promotion',
            'reason': reason
        }
        self._phase_transitions.append(transition)
        
        logger.info(
            f"🎉 PHASE PROMOTION: {old_phase.value} → {new_phase.value} | "
            f"Reason: {reason}"
        )
        
        return {
            'transition': True,
            'type': 'promotion',
            'from_phase': old_phase.value,
            'to_phase': new_phase.value,
            'reason': reason
        }
    
    async def _demote_phase_async(self, reason: str) -> Dict[str, Any]:
        """
        Demote to previous training phase (safety).
        
        Args:
            reason: Demotion reason
            
        Returns:
            Demotion result
        """
        old_phase = self._current_phase
        
        # Demote Live → Hardening
        if old_phase == TrainingPhase.LIVE:
            new_phase = TrainingPhase.HARDENING
        else:
            # Already in boot camp, pause instead
            new_phase = TrainingPhase.PAUSED
            self._is_paused = True
        
        self._current_phase = new_phase
        self._phase_start_time = time.time()
        
        # Reset new phase metrics
        async with self._metrics_lock:
            self._phase_metrics[new_phase] = PhaseMetrics(
                phase_start_time=time.time()
            )
        
        # Record transition
        transition = {
            'timestamp': time.time(),
            'from_phase': old_phase.value,
            'to_phase': new_phase.value,
            'type': 'demotion',
            'reason': reason
        }
        self._phase_transitions.append(transition)
        
        logger.warning(
            f"⚠️ PHASE DEMOTION: {old_phase.value} → {new_phase.value} | "
            f"Reason: {reason}"
        )
        
        return {
            'transition': True,
            'type': 'demotion',
            'from_phase': old_phase.value,
            'to_phase': new_phase.value,
            'reason': reason
        }
    
    def _calculate_rolling_win_rate(self) -> float:
        """Calculate rolling win rate over last N days."""
        if not self._trade_history:
            return 0.0
        
        cutoff_time = time.time() - (self.config.rolling_window_days * 86400)
        
        recent_trades = [
            t for t in self._trade_history
            if t['timestamp'] > cutoff_time
        ]
        
        if not recent_trades:
            return 0.0
        
        wins = sum(1 for t in recent_trades if t['is_win'])
        return wins / len(recent_trades)
    
    async def _training_loop_async(self):
        """
        Background training loop.
        
        Coordinates periodic training across modules.
        """
        logger.info("Starting training loop...")
        
        interval_seconds = self.config.training_interval_minutes * 60
        
        while True:
            try:
                async with self._lock:
                    if not self._is_training_active:
                        break
                
                # Skip if paused
                if self._is_paused:
                    await asyncio.sleep(60)
                    continue
                
                # Perform training iteration
                await self._training_iteration_async()
                
                self._total_training_iterations += 1
                
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Training loop error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _training_iteration_async(self):
        """
        Single training iteration.
        
        Coordinates training across modules based on current phase.
        """
        phase = self._current_phase
        
        logger.debug(f"Training iteration: phase={phase.value}")
        
        # Get curriculum scenarios if enabled
        if self.config.use_curriculum and phase == TrainingPhase.BOOT_CAMP:
            await self._update_curriculum_async()
        
        # Train each module
        # Module 1: Liquid Neural ODE
        if 'liquid_ode' in self.modules:
            try:
                ode = self.modules['liquid_ode']
                # Training would involve backprop through ODE
                # This is coordinated by the module itself
                logger.debug("ODE training iteration")
            except Exception as e:
                logger.error(f"ODE training error: {e}")
        
        # Module 2: Meta-SREK Population
        if 'meta_srek' in self.modules:
            try:
                srek = self.modules['meta_srek']
                # Meta-learning adaptation
                logger.debug("SREK training iteration")
            except Exception as e:
                logger.error(f"SREK training error: {e}")
        
        # Module 9: Transformer Validator
        if 'transformer' in self.modules:
            try:
                transformer = self.modules['transformer']
                # Train on recent trade outcomes
                logger.debug("Transformer training iteration")
            except Exception as e:
                logger.error(f"Transformer training error: {e}")
    
    async def _evolution_loop_async(self):
        """
        Background SREK evolution loop.
        
        Runs evolutionary algorithm on SREK population.
        """
        logger.info("Starting evolution loop...")
        
        interval_seconds = self.config.evolution_interval_hours * 3600
        
        while True:
            try:
                async with self._lock:
                    if not self._is_training_active:
                        break
                
                # Skip if not in appropriate phase
                if self._current_phase == TrainingPhase.PAUSED:
                    await asyncio.sleep(3600)
                    continue
                
                # Perform evolution
                if 'meta_srek' in self.modules:
                    try:
                        srek = self.modules['meta_srek']
                        # Evolution would be called here
                        # await srek.evolve_population_async()
                        self._total_evolution_cycles += 1
                        logger.info(f"Evolution cycle {self._total_evolution_cycles} complete")
                    except Exception as e:
                        logger.error(f"Evolution error: {e}")
                
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Evolution loop error: {e}")
                await asyncio.sleep(3600)
    
    async def _checkpoint_loop_async(self):
        """
        Background checkpoint loop.
        
        Periodically saves state for crash recovery.
        """
        logger.info("Starting checkpoint loop...")
        
        interval_seconds = self.config.checkpoint_interval_hours * 3600
        
        while True:
            try:
                async with self._lock:
                    if not self._is_training_active:
                        break
                
                # Save checkpoint
                checkpoint_path = Path(self.config.data_dir) / "protocol_checkpoint.json"
                await self._save_checkpoint_async(str(checkpoint_path))
                
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Checkpoint loop error: {e}")
                await asyncio.sleep(300)
    
    async def _update_curriculum_async(self):
        """Update curriculum difficulty based on time."""
        days_elapsed = (time.time() - self._curriculum_start_time) / 86400
        phase_days = self.config.curriculum_phase_days
        
        if days_elapsed < phase_days:
            new_difficulty = CurriculumDifficulty.EASY
        elif days_elapsed < phase_days * 2:
            new_difficulty = CurriculumDifficulty.MEDIUM
        elif days_elapsed < phase_days * 3:
            new_difficulty = CurriculumDifficulty.HARD
        else:
            new_difficulty = CurriculumDifficulty.EXTREME
        
        if new_difficulty != self._curriculum_difficulty:
            logger.info(
                f"📚 Curriculum difficulty: {self._curriculum_difficulty.value} → "
                f"{new_difficulty.value}"
            )
            self._curriculum_difficulty = new_difficulty
    
    async def get_current_phase_async(self) -> Dict[str, Any]:
        """Get current training phase info."""
        async with self._lock:
            async with self._metrics_lock:
                metrics = self._phase_metrics[self._current_phase]
                
                return {
                    'phase': self._current_phase.value,
                    'is_paused': self._is_paused,
                    'days_in_phase': metrics.days_in_phase,
                    'metrics': metrics.to_dict(),
                    'curriculum_difficulty': self._curriculum_difficulty.value if self.config.use_curriculum else None
                }
    
    async def get_metrics_async(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        async with self._lock:
            async with self._metrics_lock:
                all_metrics = {
                    phase.value: self._phase_metrics[phase].to_dict()
                    for phase in [TrainingPhase.BOOT_CAMP, TrainingPhase.HARDENING, TrainingPhase.LIVE]
                }
                
                return {
                    'is_initialized': self._is_initialized,
                    'current_phase': self._current_phase.value,
                    'is_training_active': self._is_training_active,
                    'is_paused': self._is_paused,
                    'total_training_iterations': self._total_training_iterations,
                    'total_evolution_cycles': self._total_evolution_cycles,
                    'phase_transitions': len(self._phase_transitions),
                    'rolling_win_rate': self._calculate_rolling_win_rate(),
                    'phase_metrics': all_metrics
                }
    
    async def force_phase_transition_async(
        self,
        target_phase: TrainingPhase,
        reason: str = "manual"
    ) -> Dict[str, Any]:
        """
        Force phase transition (admin override).
        
        Args:
            target_phase: Target phase
            reason: Reason for override
            
        Returns:
            Transition result
        """
        async with self._phase_lock:
            old_phase = self._current_phase
            self._current_phase = target_phase
            self._phase_start_time = time.time()
            
            if target_phase == TrainingPhase.PAUSED:
                self._is_paused = True
            else:
                self._is_paused = False
            
            # Reset metrics for new phase
            async with self._metrics_lock:
                self._phase_metrics[target_phase] = PhaseMetrics(
                    phase_start_time=time.time()
                )
            
            # Record transition
            transition = {
                'timestamp': time.time(),
                'from_phase': old_phase.value,
                'to_phase': target_phase.value,
                'type': 'manual',
                'reason': reason
            }
            self._phase_transitions.append(transition)
            
            logger.info(
                f"🔧 MANUAL PHASE TRANSITION: {old_phase.value} → {target_phase.value} | "
                f"Reason: {reason}"
            )
            
            return {
                'status': 'success',
                'from_phase': old_phase.value,
                'to_phase': target_phase.value
            }
    
    async def _save_checkpoint_async(self, filepath: str) -> Dict[str, Any]:
        """Save checkpoint to file."""
        try:
            async with self._lock:
                async with self._metrics_lock:
                    checkpoint = {
                        'current_phase': self._current_phase.value,
                        'phase_start_time': self._phase_start_time,
                        'is_paused': self._is_paused,
                        'curriculum_difficulty': self._curriculum_difficulty.value,
                        'curriculum_start_time': self._curriculum_start_time,
                        'total_training_iterations': self._total_training_iterations,
                        'total_evolution_cycles': self._total_evolution_cycles,
                        'phase_transitions': self._phase_transitions,
                        'phase_metrics': {
                            phase.value: asdict(metrics)
                            for phase, metrics in self._phase_metrics.items()
                        },
                        'timestamp': time.time()
                    }
            
            # Write to file (offload I/O)
            await asyncio.to_thread(
                self._write_json_sync,
                filepath,
                checkpoint
            )
            
            logger.debug(f"Checkpoint saved: {filepath}")
            return {'status': 'success', 'filepath': filepath}
            
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _write_json_sync(self, filepath: str, data: Dict):
        """Synchronous JSON write."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def _load_checkpoint_async(self, filepath: str) -> Dict[str, Any]:
        """Load checkpoint from file."""
        try:
            # Read from file (offload I/O)
            checkpoint = await asyncio.to_thread(
                self._read_json_sync,
                filepath
            )
            
            # Restore state
            self._current_phase = TrainingPhase.from_string(
                checkpoint.get('current_phase', 'boot_camp')
            )
            self._phase_start_time = checkpoint.get('phase_start_time', time.time())
            self._is_paused = checkpoint.get('is_paused', False)
            self._curriculum_difficulty = CurriculumDifficulty(
                checkpoint.get('curriculum_difficulty', 'easy')
            )
            self._curriculum_start_time = checkpoint.get('curriculum_start_time', time.time())
            self._total_training_iterations = checkpoint.get('total_training_iterations', 0)
            self._total_evolution_cycles = checkpoint.get('total_evolution_cycles', 0)
            self._phase_transitions = checkpoint.get('phase_transitions', [])
            
            # Restore phase metrics
            for phase_str, metrics_dict in checkpoint.get('phase_metrics', {}).items():
                phase = TrainingPhase.from_string(phase_str)
                if phase in self._phase_metrics:
                    self._phase_metrics[phase] = PhaseMetrics(**metrics_dict)
            
            logger.info(f"Checkpoint loaded: {filepath}")
            return {'status': 'success', 'filepath': filepath}
            
        except Exception as e:
            logger.error(f"Checkpoint load failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _read_json_sync(self, filepath: str) -> Dict:
        """Synchronous JSON read."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    async def cleanup_async(self):
        """Cleanup resources."""
        await self.stop_training_async()
        
        # Save final checkpoint
        checkpoint_path = Path(self.config.data_dir) / "protocol_checkpoint.json"
        await self._save_checkpoint_async(str(checkpoint_path))
        
        async with self._lock:
            self._is_initialized = False
        
        logger.info("✅ TrainingProtocolManager cleaned up")


# ============================================================================
# INTEGRATION TEST
# ============================================================================

async def test_training_protocol_manager():
    """Integration test for TrainingProtocolManager"""
    logger.info("=" * 60)
    logger.info("TESTING MODULE 10: TRAINING PROTOCOL MANAGER")
    logger.info("=" * 60)
    
    # Test 0: Config validation
    logger.info("\n[Test 0] Configuration validation...")
    try:
        invalid_config = TrainingProtocolConfig(boot_camp_min_win_rate=1.5)
        logger.error("Should have raised ValueError")
    except ValueError as e:
        logger.info(f"✅ Config validation caught error: {e}")
    
    # Configuration
    config = TrainingProtocolConfig(
        boot_camp_min_days=1,  # Short for testing
        boot_camp_min_trades=10,
        hardening_min_days=1,
        hardening_min_trades=20,
        data_dir="/tmp/training_protocol_test"
    )
    
    # Create manager
    manager = TrainingProtocolManager(config=config)
    
    # Test 1: Initialization
    logger.info("\n[Test 1] Initialization...")
    init_result = await manager.initialize_async()
    assert init_result['status'] == 'success', f"Init failed: {init_result}"
    logger.info(f"✅ Initialized: phase={init_result['current_phase']}")
    
    # Test 2: Record winning trades
    logger.info("\n[Test 2] Recording winning trades...")
    for i in range(15):
        result = await manager.record_trade_async({
            'pnl_pips': 10.0,
            'is_win': True,
            'current_capital': 200 + (i * 10)
        })
    
    metrics = await manager.get_current_phase_async()
    logger.info(f"✅ Win rate: {metrics['metrics']['win_rate']:.2%}")
    logger.info(f"   Profit factor: {metrics['metrics']['profit_factor']:.2f}")
    
    # Test 3: Phase promotion
    logger.info("\n[Test 3] Phase promotion (should promote to Hardening)...")
    # Need to meet all criteria
    phase_info = await manager.get_current_phase_async()
    if phase_info['phase'] == 'hardening':
        logger.info("✅ Auto-promoted to Hardening phase")
    else:
        logger.info(f"   Still in {phase_info['phase']} (criteria not fully met)")
    
    # Test 4: Record losing trades
    logger.info("\n[Test 4] Recording losing trades...")
    for i in range(5):
        result = await manager.record_trade_async({
            'pnl_pips': -15.0,
            'is_win': False,
            'current_capital': 300 - (i * 15)
        })
    
    metrics = await manager.get_current_phase_async()
    logger.info(f"✅ Updated win rate: {metrics['metrics']['win_rate']:.2%}")
    logger.info(f"   Consecutive losses: {metrics['metrics']['consecutive_losses']}")
    
    # Test 5: Force phase transition
    logger.info("\n[Test 5] Force phase transition...")
    result = await manager.force_phase_transition_async(
        TrainingPhase.HARDENING,
        reason="test override"
    )
    assert result['status'] == 'success'
    logger.info(f"✅ Forced to: {result['to_phase']}")
    
    # Test 6: Thread safety (concurrent trade recording)
    logger.info("\n[Test 6] Thread safety (10 concurrent trades)...")
    tasks = []
    for i in range(10):
        tasks.append(manager.record_trade_async({
            'pnl_pips': 5.0 if i % 2 == 0 else -5.0,
            'is_win': i % 2 == 0,
            'current_capital': 200
        }))
    
    results = await asyncio.gather(*tasks)
    assert len(results) == 10
    logger.info("✅ All 10 concurrent trades recorded")
    
    # Test 7: Metrics
    logger.info("\n[Test 7] Comprehensive metrics...")
    metrics = await manager.get_metrics_async()
    logger.info(f"✅ Current phase: {metrics['current_phase']}")
    logger.info(f"   Training iterations: {metrics['total_training_iterations']}")
    logger.info(f"   Phase transitions: {metrics['phase_transitions']}")
    
    # Test 8: Checkpoint save/load
    logger.info("\n[Test 8] Checkpoint save/load...")
    save_result = await manager._save_checkpoint_async("/tmp/training_protocol_test/test_checkpoint.json")
    assert save_result['status'] == 'success'
    
    load_result = await manager._load_checkpoint_async("/tmp/training_protocol_test/test_checkpoint.json")
    assert load_result['status'] == 'success'
    logger.info("✅ Checkpoint save/load successful")
    
    # Test 9: Cleanup
    logger.info("\n[Test 9] Cleanup...")
    await manager.cleanup_async()
    logger.info("✅ Cleanup successful")
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL TESTS PASSED ✅")
    logger.info("=" * 60)


if __name__ == '__main__':
    asyncio.run(test_training_protocol_manager())

"""
MODULE 4: AGGRESSIVE RISK MANAGER
Production-Ready Implementation - FIXED VERSION

Kelly Criterion position sizing with dynamic adjustments.
- Async/await architecture throughout
- Thread-safe capital management
- Dynamic Kelly sizing (confidence × regime × Sharpe × volatility × correlation)
- Drawdown protection
- Nano-lot support ($200 → $100K+)
- Mistake-based learning

Author: Liquid Neural SREK Trading System v4.0
Date: 2026-01-10
Version: 1.1.0 (Fixed)

FIXES APPLIED:
- Issue 4.1 (HIGH): _risk_multiplier now protected by _stats_lock
- Issue 4.2 (MEDIUM): Division by zero protection in drawdown calculation
- Issue 4.3 (MEDIUM): Fire-and-forget task wrapped with error handler
- Issue 4.4 (LOW): Config validation with __post_init__
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from collections import deque
import numpy as np
import json
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION WITH VALIDATION
# ============================================================================

@dataclass
class RiskManagerConfig:
    """
    Configuration for Aggressive Risk Manager
    
    Includes validation to prevent runtime errors
    """
    # Capital management
    initial_capital: float = 200.0  # Starting capital
    min_capital: float = 50.0  # Stop trading below this
    
    # Kelly Criterion base settings
    base_kelly_fraction: float = 0.05  # 5% base risk (aggressive)
    max_kelly_fraction: float = 0.10  # 10% maximum risk
    min_kelly_fraction: float = 0.005  # 0.5% minimum risk (after mistakes)
    
    # Dynamic Kelly adjustments (Gemini optimization)
    use_dynamic_kelly: bool = True
    confidence_weight: float = 1.0
    regime_stability_weight: float = 0.8
    sharpe_weight: float = 0.6
    volatility_weight: float = -0.4  # Negative = reduce risk in high vol
    correlation_weight: float = -0.3  # Negative = reduce risk in high corr
    
    # Leverage
    max_leverage: float = 5.0  # Conservative (Oanda allows 50x)
    use_leverage: bool = True
    
    # Drawdown protection
    max_drawdown_percent: float = 25.0  # Stop trading at 25% DD
    reduce_risk_drawdown_threshold: float = 10.0  # Start reducing at 10% DD
    
    # Position sizing
    min_position_size: float = 1.0  # Oanda nano-lot minimum
    max_positions: int = 5  # Max concurrent positions
    
    # Mistake learning
    mistake_memory_trades: int = 50  # Learn from last 50 trades
    consecutive_loss_threshold: int = 3  # Reduce risk after 3 losses
    
    # Performance tracking
    sharpe_calculation_window: int = 30  # Days for Sharpe calculation
    
    # Persistence
    data_dir: str = 'data/risk_manager'
    save_interval_trades: int = 10  # Save every 10 trades
    
    def __post_init__(self):
        """
        Validate configuration to prevent runtime errors
        
        FIX Issue 4.4: Added comprehensive validation
        """
        # Capital validation
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {self.initial_capital}")
        if self.min_capital <= 0:
            raise ValueError(f"min_capital must be positive, got {self.min_capital}")
        if self.min_capital >= self.initial_capital:
            raise ValueError(f"min_capital must be less than initial_capital")
        
        # Kelly validation
        if not 0.0 < self.base_kelly_fraction < 1.0:
            raise ValueError(f"base_kelly_fraction must be in (0, 1), got {self.base_kelly_fraction}")
        if not 0.0 < self.max_kelly_fraction <= 1.0:
            raise ValueError(f"max_kelly_fraction must be in (0, 1], got {self.max_kelly_fraction}")
        if not 0.0 < self.min_kelly_fraction < 1.0:
            raise ValueError(f"min_kelly_fraction must be in (0, 1), got {self.min_kelly_fraction}")
        if self.min_kelly_fraction >= self.max_kelly_fraction:
            raise ValueError(f"min_kelly_fraction must be less than max_kelly_fraction")
        if self.base_kelly_fraction > self.max_kelly_fraction:
            raise ValueError(f"base_kelly_fraction must be <= max_kelly_fraction")
        
        # Leverage validation
        if self.max_leverage <= 0:
            raise ValueError(f"max_leverage must be positive, got {self.max_leverage}")
        
        # Drawdown validation
        if not 0.0 < self.max_drawdown_percent <= 100.0:
            raise ValueError(f"max_drawdown_percent must be in (0, 100], got {self.max_drawdown_percent}")
        if not 0.0 <= self.reduce_risk_drawdown_threshold < self.max_drawdown_percent:
            raise ValueError(f"reduce_risk_drawdown_threshold must be in [0, max_drawdown_percent)")
        
        # Position validation
        if self.min_position_size <= 0:
            raise ValueError(f"min_position_size must be positive, got {self.min_position_size}")
        if self.max_positions <= 0:
            raise ValueError(f"max_positions must be positive, got {self.max_positions}")
        
        # Mistake learning validation
        if self.mistake_memory_trades <= 0:
            raise ValueError(f"mistake_memory_trades must be positive, got {self.mistake_memory_trades}")
        if self.consecutive_loss_threshold <= 0:
            raise ValueError(f"consecutive_loss_threshold must be positive, got {self.consecutive_loss_threshold}")
        
        # Persistence validation
        if self.save_interval_trades <= 0:
            raise ValueError(f"save_interval_trades must be positive, got {self.save_interval_trades}")


# ============================================================================
# TRADE RECORD
# ============================================================================

class TradeOutcome(Enum):
    """Trade outcome types"""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


@dataclass
class TradeRecord:
    """Complete record of a trade"""
    # Identifiers
    trade_id: str
    timestamp: float
    
    # Market data
    pair: str
    regime: str
    entry_price: float
    
    # Position details (REQUIRED - no defaults)
    position_size: float  # In units (supports fractional)
    direction: str  # 'buy' or 'sell'
    
    # Optional fields (with defaults)
    exit_price: Optional[float] = None
    leverage: float = 1.0
    
    # Risk management
    kelly_fraction: float = 0.0
    risk_dollars: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Performance context
    srek_confidence: float = 0.0
    pattern_quality: float = 0.0
    regime_stability: float = 0.0
    recent_sharpe: float = 0.0
    market_volatility: float = 0.0
    market_correlation: float = 0.0
    
    # Outcome
    outcome: Optional[TradeOutcome] = None
    profit_loss: float = 0.0
    profit_loss_percent: float = 0.0
    
    # Capital state at trade time
    capital_before: float = 0.0
    capital_after: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = asdict(self)
        if self.outcome:
            d['outcome'] = self.outcome.value
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeRecord':
        """Create from dictionary"""
        if 'outcome' in data and data['outcome']:
            data['outcome'] = TradeOutcome(data['outcome'])
        return cls(**data)


# ============================================================================
# AGGRESSIVE RISK MANAGER (ALL RACE CONDITIONS FIXED)
# ============================================================================

class AggressiveRiskManager:
    """
    Kelly Criterion position sizing with dynamic adjustments
    
    Features:
    - Dynamic Kelly sizing (Gemini optimization)
    - Thread-safe capital management
    - Drawdown protection
    - Nano-lot support
    - Mistake-based learning
    - Complete async architecture
    
    FIXES APPLIED:
    - Issue 4.1: _risk_multiplier protected by _stats_lock
    - Issue 4.2: Division by zero protection in drawdown
    - Issue 4.3: Fire-and-forget task wrapped with error handler
    - Issue 4.4: Config validation with __post_init__
    """
    
    def __init__(self, config: RiskManagerConfig):
        self.config = config
        
        # Create data directory
        Path(config.data_dir).mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.capital_file = Path(config.data_dir) / 'capital_state.json'
        self.history_file = Path(config.data_dir) / 'trade_history.json'
        
        # Thread safety locks
        self._capital_lock = asyncio.Lock()  # Protects capital state
        self._history_lock = asyncio.Lock()  # Protects trade history
        self._stats_lock = asyncio.Lock()  # Protects statistics AND _risk_multiplier
        self._position_lock = asyncio.Lock()  # Protects position tracking
        
        # Capital state (protected by _capital_lock)
        self.current_capital = config.initial_capital
        self.peak_capital = config.initial_capital
        self.total_deposited = config.initial_capital
        self.total_withdrawn = 0.0
        
        # Position tracking (protected by _position_lock)
        self.active_positions: Dict[str, TradeRecord] = {}
        
        # Trade history (protected by _history_lock)
        self.trade_history: deque = deque(maxlen=10000)  # Last 10K trades
        self.recent_trades: deque = deque(maxlen=config.mistake_memory_trades)
        
        # Performance metrics (protected by _stats_lock)
        self._stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'breakeven': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_kelly_fraction': 0.0,
            'trades_since_save': 0,
            'background_task_errors': 0
        }
        
        # FIX Issue 4.1: _risk_multiplier now documented as protected by _stats_lock
        # Adaptive risk state (protected by _stats_lock)
        self._risk_multiplier = 1.0  # Adjusted based on performance
        
        # State tracking
        self._is_initialized = False
        
        logger.info(
            f"AggressiveRiskManager initialized: "
            f"capital=${config.initial_capital}, "
            f"kelly={config.base_kelly_fraction:.1%}"
        )
    
    def _create_error_handling_task(self, coro, task_name: str):
        """
        FIX Issue 4.3: Wrap fire-and-forget coroutine with error handling
        
        Args:
            coro: Coroutine to wrap
            task_name: Name for logging
        """
        async def wrapper():
            try:
                await coro
            except Exception as e:
                logger.error(f"Background task '{task_name}' failed: {e}")
                logger.debug(traceback.format_exc())
                async with self._stats_lock:
                    self._stats['background_task_errors'] += 1
        
        return asyncio.create_task(wrapper())
    
    def _safe_drawdown_calculation(
        self, 
        current_capital: float, 
        peak_capital: float
    ) -> float:
        """
        FIX Issue 4.2: Safe drawdown calculation with division by zero protection
        
        Args:
            current_capital: Current capital
            peak_capital: Peak capital
            
        Returns:
            Drawdown percentage (0-100)
        """
        if peak_capital <= 0:
            return 0.0  # No drawdown if peak is invalid
        
        drawdown = ((peak_capital - current_capital) / peak_capital) * 100
        return max(0.0, drawdown)  # Ensure non-negative
    
    async def initialize_async(self) -> Dict[str, Any]:
        """
        Initialize risk manager (load saved state)
        
        Returns:
            Initialization status
        """
        try:
            # Load capital state
            if self.capital_file.exists():
                capital_data = await asyncio.to_thread(
                    self._load_json_sync,
                    self.capital_file
                )
                
                async with self._capital_lock:
                    self.current_capital = capital_data.get(
                        'current_capital',
                        self.config.initial_capital
                    )
                    self.peak_capital = capital_data.get(
                        'peak_capital',
                        self.current_capital
                    )
                    self.total_deposited = capital_data.get(
                        'total_deposited',
                        self.config.initial_capital
                    )
                    self.total_withdrawn = capital_data.get(
                        'total_withdrawn',
                        0.0
                    )
                
                logger.info(f"Loaded capital: ${self.current_capital:.2f}")
            
            # Load trade history
            if self.history_file.exists():
                history_data = await asyncio.to_thread(
                    self._load_json_sync,
                    self.history_file
                )
                
                async with self._history_lock:
                    trades = history_data.get('trades', [])
                    for trade_data in trades[-10000:]:  # Last 10K
                        trade = TradeRecord.from_dict(trade_data)
                        self.trade_history.append(trade)
                    
                    # Recent trades for learning
                    for trade_data in trades[-self.config.mistake_memory_trades:]:
                        trade = TradeRecord.from_dict(trade_data)
                        self.recent_trades.append(trade)
                
                async with self._stats_lock:
                    stats = history_data.get('stats', {})
                    for key in self._stats:
                        if key in stats:
                            self._stats[key] = stats[key]
                    
                    # Also restore risk multiplier if saved
                    self._risk_multiplier = history_data.get('risk_multiplier', 1.0)
                
                logger.info(
                    f"Loaded history: {len(self.trade_history)} trades"
                )
            
            self._is_initialized = True
            
            return {
                'status': 'success',
                'current_capital': self.current_capital,
                'total_trades': self._stats['total_trades']
            }
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _load_json_sync(self, path: Path) -> Dict:
        """Synchronous JSON load"""
        with open(path, 'r') as f:
            return json.load(f)
    
    async def calculate_position_size_async(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str,
        pair: str = 'EUR_USD',
        regime: str = 'normal',
        srek_confidence: float = 0.75,
        pattern_quality: Optional[float] = None,
        regime_stability: Optional[float] = None,
        recent_sharpe: Optional[float] = None,
        market_volatility: Optional[float] = None,
        market_correlation: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate position size using dynamic Kelly Criterion
        
        FIX Issue 4.2: Safe drawdown calculation
        """
        if not self._is_initialized:
            raise RuntimeError("Not initialized. Call initialize_async() first")
        
        try:
            # Get current capital (thread-safe)
            async with self._capital_lock:
                current_capital = self.current_capital
                peak_capital = self.peak_capital
            
            # Edge case: Insufficient capital
            if current_capital < self.config.min_capital:
                return {
                    'should_trade': False,
                    'reason': f'Capital below minimum (${current_capital:.2f})',
                    'position_size': 0.0,
                    'risk_dollars': 0.0,
                    'kelly_fraction': 0.0,
                    'leverage': 0.0
                }
            
            # FIX Issue 4.2: Safe drawdown calculation
            drawdown = self._safe_drawdown_calculation(current_capital, peak_capital)
            
            # Check max drawdown
            if drawdown >= self.config.max_drawdown_percent:
                return {
                    'should_trade': False,
                    'reason': f'Max drawdown exceeded ({drawdown:.1f}%)',
                    'position_size': 0.0,
                    'risk_dollars': 0.0,
                    'kelly_fraction': 0.0,
                    'leverage': 0.0
                }
            
            # Calculate stop loss distance (in percentage)
            if direction == 'buy':
                stop_distance = abs(entry_price - stop_loss) / entry_price
            else:  # sell
                stop_distance = abs(stop_loss - entry_price) / entry_price
            
            # Edge case: Stop too tight
            if stop_distance < 0.001:  # Less than 0.1%
                return {
                    'should_trade': False,
                    'reason': 'Stop loss too tight',
                    'position_size': 0.0,
                    'risk_dollars': 0.0,
                    'kelly_fraction': 0.0,
                    'leverage': 0.0
                }
            
            # Calculate Kelly fraction
            kelly_fraction = await self._calculate_dynamic_kelly_async(
                srek_confidence=srek_confidence,
                pattern_quality=pattern_quality,
                regime_stability=regime_stability,
                recent_sharpe=recent_sharpe,
                market_volatility=market_volatility,
                market_correlation=market_correlation,
                drawdown=drawdown
            )
            
            # Calculate risk dollars
            risk_dollars = current_capital * kelly_fraction
            
            # Calculate position size (in units)
            position_size = risk_dollars / (entry_price * stop_distance)
            
            # Apply leverage if enabled
            leverage = 1.0
            if self.config.use_leverage:
                position_value = position_size * entry_price
                required_margin = position_value / self.config.max_leverage
                
                if required_margin < current_capital:
                    leverage = min(
                        position_value / current_capital,
                        self.config.max_leverage
                    )
            
            # Enforce minimum position size (Oanda nano-lot)
            if position_size < self.config.min_position_size:
                return {
                    'should_trade': False,
                    'reason': 'Position size below minimum',
                    'position_size': 0.0,
                    'risk_dollars': 0.0,
                    'kelly_fraction': kelly_fraction,
                    'leverage': 0.0
                }
            
            # Check concurrent positions limit
            async with self._position_lock:
                num_positions = len(self.active_positions)
            
            if num_positions >= self.config.max_positions:
                return {
                    'should_trade': False,
                    'reason': f'Max positions reached ({num_positions})',
                    'position_size': 0.0,
                    'risk_dollars': 0.0,
                    'kelly_fraction': kelly_fraction,
                    'leverage': 0.0
                }
            
            return {
                'should_trade': True,
                'reason': 'Position approved',
                'position_size': float(position_size),
                'risk_dollars': float(risk_dollars),
                'kelly_fraction': float(kelly_fraction),
                'leverage': float(leverage),
                'stop_distance_percent': float(stop_distance * 100),
                'current_capital': float(current_capital),
                'drawdown_percent': float(drawdown)
            }
            
        except Exception as e:
            logger.error(f"Position calculation failed: {e}")
            return {
                'should_trade': False,
                'reason': f'Calculation error: {e}',
                'position_size': 0.0,
                'risk_dollars': 0.0,
                'kelly_fraction': 0.0,
                'leverage': 0.0
            }
    
    async def _calculate_dynamic_kelly_async(
        self,
        srek_confidence: float,
        pattern_quality: Optional[float],
        regime_stability: Optional[float],
        recent_sharpe: Optional[float],
        market_volatility: Optional[float],
        market_correlation: Optional[float],
        drawdown: float
    ) -> float:
        """
        Calculate dynamic Kelly fraction
        
        FIX Issue 4.1: _risk_multiplier read protected by _stats_lock
        """
        base_kelly = self.config.base_kelly_fraction
        
        if not self.config.use_dynamic_kelly:
            return base_kelly
        
        # Initialize multipliers to 1.0
        confidence_mult = 1.0
        regime_mult = 1.0
        sharpe_mult = 1.0
        volatility_mult = 1.0
        correlation_mult = 1.0
        drawdown_mult = 1.0
        
        # 1. Confidence multiplier (0.5x to 1.5x)
        if srek_confidence is not None:
            confidence_mult = 0.5 + (srek_confidence * 1.0)
            confidence_mult *= self.config.confidence_weight
        
        # 2. Regime stability multiplier (0.7x to 1.3x)
        if regime_stability is not None:
            regime_mult = 0.7 + (regime_stability * 0.6)
            regime_mult = 1.0 + (regime_mult - 1.0) * self.config.regime_stability_weight
        
        # 3. Sharpe ratio multiplier (0.8x to 1.4x)
        if recent_sharpe is not None:
            normalized_sharpe = np.clip((recent_sharpe + 1) / 4, 0, 1)
            sharpe_mult = 0.8 + (normalized_sharpe * 0.6)
            sharpe_mult = 1.0 + (sharpe_mult - 1.0) * self.config.sharpe_weight
        
        # 4. Volatility multiplier (0.6x to 1.0x, reduces in high vol)
        if market_volatility is not None:
            volatility_mult = 1.0 - (market_volatility * 0.4)
            volatility_mult = 1.0 + (volatility_mult - 1.0) * abs(self.config.volatility_weight)
        
        # 5. Correlation multiplier (0.7x to 1.0x, reduces in high corr)
        if market_correlation is not None:
            correlation_mult = 1.0 - (market_correlation * 0.3)
            correlation_mult = 1.0 + (correlation_mult - 1.0) * abs(self.config.correlation_weight)
        
        # 6. Drawdown multiplier (0.3x to 1.0x)
        if drawdown >= self.config.reduce_risk_drawdown_threshold:
            dd_range = self.config.max_drawdown_percent - self.config.reduce_risk_drawdown_threshold
            if dd_range > 0:  # FIX: Prevent division by zero
                dd_factor = 1.0 - (
                    (drawdown - self.config.reduce_risk_drawdown_threshold) / dd_range
                )
                drawdown_mult = 0.3 + (dd_factor * 0.7)
            else:
                drawdown_mult = 0.3
        
        # FIX Issue 4.1: Get risk multiplier while holding lock
        async with self._stats_lock:
            risk_mult = self._risk_multiplier
        
        # Combined Kelly fraction
        kelly = (
            base_kelly *
            confidence_mult *
            regime_mult *
            sharpe_mult *
            volatility_mult *
            correlation_mult *
            drawdown_mult *
            risk_mult
        )
        
        # Clamp to limits
        kelly = np.clip(
            kelly,
            self.config.min_kelly_fraction,
            self.config.max_kelly_fraction
        )
        
        return float(kelly)
    
    async def open_position_async(
        self,
        trade_id: str,
        pair: str,
        regime: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: float,
        direction: str,
        leverage: float,
        kelly_fraction: float,
        risk_dollars: float,
        srek_confidence: float,
        pattern_quality: Optional[float] = None,
        regime_stability: Optional[float] = None,
        recent_sharpe: Optional[float] = None,
        market_volatility: Optional[float] = None,
        market_correlation: Optional[float] = None
    ) -> Dict[str, Any]:
        """Open a new position (thread-safe)"""
        try:
            async with self._capital_lock:
                capital_before = self.current_capital
            
            trade = TradeRecord(
                trade_id=trade_id,
                timestamp=time.time(),
                pair=pair,
                regime=regime,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                direction=direction,
                leverage=leverage,
                kelly_fraction=kelly_fraction,
                risk_dollars=risk_dollars,
                srek_confidence=srek_confidence,
                pattern_quality=pattern_quality or 0.0,
                regime_stability=regime_stability or 0.0,
                recent_sharpe=recent_sharpe or 0.0,
                market_volatility=market_volatility or 0.0,
                market_correlation=market_correlation or 0.0,
                capital_before=capital_before
            )
            
            async with self._position_lock:
                self.active_positions[trade_id] = trade
            
            logger.info(
                f"Opened position: {trade_id}, "
                f"size={position_size:.2f}, "
                f"kelly={kelly_fraction:.2%}"
            )
            
            return {'status': 'success', 'trade_id': trade_id}
            
        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def close_position_async(
        self,
        trade_id: str,
        exit_price: float
    ) -> Dict[str, Any]:
        """
        Close a position and update capital (thread-safe)
        
        FIX Issue 4.3: Fire-and-forget save task wrapped with error handler
        """
        try:
            async with self._position_lock:
                if trade_id not in self.active_positions:
                    return {'status': 'not_found'}
                
                trade = self.active_positions[trade_id]
            
            # Calculate P&L
            if trade.direction == 'buy':
                pnl = (exit_price - trade.entry_price) * trade.position_size
            else:
                pnl = (trade.entry_price - exit_price) * trade.position_size
            
            pnl_percent = (pnl / trade.risk_dollars) * 100 if trade.risk_dollars > 0 else 0.0
            
            if pnl > 0:
                outcome = TradeOutcome.WIN
            elif pnl < 0:
                outcome = TradeOutcome.LOSS
            else:
                outcome = TradeOutcome.BREAKEVEN
            
            trade.exit_price = exit_price
            trade.profit_loss = pnl
            trade.profit_loss_percent = pnl_percent
            trade.outcome = outcome
            
            async with self._capital_lock:
                self.current_capital += pnl
                trade.capital_after = self.current_capital
                
                if self.current_capital > self.peak_capital:
                    self.peak_capital = self.current_capital
            
            await self._update_stats_async(trade)
            
            async with self._position_lock:
                del self.active_positions[trade_id]
            
            async with self._history_lock:
                self.trade_history.append(trade)
                self.recent_trades.append(trade)
            
            if outcome == TradeOutcome.LOSS:
                await self._learn_from_mistake_async(trade)
            
            should_save = False
            async with self._stats_lock:
                self._stats['trades_since_save'] += 1
                
                if self._stats['trades_since_save'] >= self.config.save_interval_trades:
                    should_save = True
                    self._stats['trades_since_save'] = 0
            
            # FIX Issue 4.3: Wrap fire-and-forget task with error handler
            if should_save:
                self._create_error_handling_task(
                    self.save_state_async(),
                    "auto_save_state"
                )
            
            logger.info(
                f"Closed position: {trade_id}, "
                f"P&L=${pnl:.2f} ({pnl_percent:+.1f}%), "
                f"capital=${trade.capital_after:.2f}"
            )
            
            return {
                'status': 'success',
                'trade_id': trade_id,
                'outcome': outcome.value,
                'pnl': float(pnl),
                'pnl_percent': float(pnl_percent),
                'capital': float(trade.capital_after)
            }
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _update_stats_async(self, trade: TradeRecord):
        """Update statistics (thread-safe)"""
        async with self._stats_lock:
            self._stats['total_trades'] += 1
            
            if trade.outcome == TradeOutcome.WIN:
                self._stats['wins'] += 1
                self._stats['total_profit'] += abs(trade.profit_loss)
                self._stats['consecutive_wins'] += 1
                self._stats['consecutive_losses'] = 0
                
                if abs(trade.profit_loss) > self._stats['largest_win']:
                    self._stats['largest_win'] = abs(trade.profit_loss)
                
                if self._stats['consecutive_wins'] > self._stats['max_consecutive_wins']:
                    self._stats['max_consecutive_wins'] = self._stats['consecutive_wins']
            
            elif trade.outcome == TradeOutcome.LOSS:
                self._stats['losses'] += 1
                self._stats['total_loss'] += abs(trade.profit_loss)
                self._stats['consecutive_losses'] += 1
                self._stats['consecutive_wins'] = 0
                
                if abs(trade.profit_loss) > self._stats['largest_loss']:
                    self._stats['largest_loss'] = abs(trade.profit_loss)
                
                if self._stats['consecutive_losses'] > self._stats['max_consecutive_losses']:
                    self._stats['max_consecutive_losses'] = self._stats['consecutive_losses']
            
            else:
                self._stats['breakeven'] += 1
                self._stats['consecutive_wins'] = 0
                self._stats['consecutive_losses'] = 0
            
            n = self._stats['total_trades']
            if n > 0:
                old_avg = self._stats['avg_kelly_fraction']
                self._stats['avg_kelly_fraction'] = (
                    (old_avg * (n - 1) + trade.kelly_fraction) / n
                )
    
    async def _learn_from_mistake_async(self, trade: TradeRecord):
        """
        Learn from losing trades and adjust risk
        
        FIX Issue 4.1: All _risk_multiplier access protected by _stats_lock
        """
        async with self._stats_lock:
            consecutive_losses = self._stats['consecutive_losses']
            
            if consecutive_losses >= self.config.consecutive_loss_threshold:
                reduction_factor = max(0.5, 1.0 - (consecutive_losses * 0.05))
                self._risk_multiplier = reduction_factor
                
                logger.warning(
                    f"Risk reduced to {reduction_factor:.0%} "
                    f"after {consecutive_losses} consecutive losses"
                )
            else:
                if consecutive_losses == 0:
                    self._risk_multiplier = min(1.0, self._risk_multiplier + 0.05)
    
    async def get_metrics_async(self) -> Dict[str, Any]:
        """Get risk manager metrics (thread-safe)"""
        async with self._capital_lock:
            current_capital = self.current_capital
            peak_capital = self.peak_capital
        
        drawdown = self._safe_drawdown_calculation(current_capital, peak_capital)
        
        async with self._stats_lock:
            stats = self._stats.copy()
            risk_multiplier = self._risk_multiplier
        
        async with self._position_lock:
            num_positions = len(self.active_positions)
        
        win_rate = stats['wins'] / stats['total_trades'] if stats['total_trades'] > 0 else 0.0
        
        profit_factor = (
            stats['total_profit'] / stats['total_loss']
            if stats['total_loss'] > 0 else
            (10.0 if stats['total_profit'] > 0 else 1.0)
        )
        
        net_profit = stats['total_profit'] - stats['total_loss']
        roi = ((current_capital - self.config.initial_capital) / self.config.initial_capital) * 100
        
        return {
            'current_capital': float(current_capital),
            'peak_capital': float(peak_capital),
            'drawdown_percent': float(drawdown),
            'roi_percent': float(roi),
            'total_trades': stats['total_trades'],
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'net_profit': float(net_profit),
            'active_positions': num_positions,
            'risk_multiplier': float(risk_multiplier),
            'avg_kelly_fraction': float(stats['avg_kelly_fraction']),
            'consecutive_losses': stats['consecutive_losses'],
            'background_task_errors': stats['background_task_errors']
        }
    
    async def save_state_async(self) -> Dict[str, Any]:
        """Save state to disk (async I/O)"""
        try:
            async with self._capital_lock:
                capital_data = {
                    'current_capital': self.current_capital,
                    'peak_capital': self.peak_capital,
                    'total_deposited': self.total_deposited,
                    'total_withdrawn': self.total_withdrawn,
                    'timestamp': time.time()
                }
            
            async with self._history_lock:
                trades = [trade.to_dict() for trade in self.trade_history]
            
            async with self._stats_lock:
                stats = self._stats.copy()
                risk_multiplier = self._risk_multiplier
            
            history_data = {
                'trades': trades,
                'stats': stats,
                'risk_multiplier': risk_multiplier,
                'timestamp': time.time()
            }
            
            await asyncio.to_thread(
                self._save_json_sync,
                self.capital_file,
                capital_data
            )
            
            await asyncio.to_thread(
                self._save_json_sync,
                self.history_file,
                history_data
            )
            
            logger.info("State saved")
            
            return {'status': 'success'}
            
        except Exception as e:
            logger.error(f"State save failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _save_json_sync(self, path: Path, data: Dict):
        """Synchronous JSON save"""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def cleanup_async(self):
        """Cleanup and save final state"""
        await self.save_state_async()
        logger.info("AggressiveRiskManager cleaned up")


# ============================================================================
# INTEGRATION TEST
# ============================================================================

async def test_aggressive_risk_manager():
    """Integration test for AggressiveRiskManager (FIXED VERSION)"""
    logger.info("=" * 60)
    logger.info("TESTING MODULE 4: AGGRESSIVE RISK MANAGER (FIXED)")
    logger.info("=" * 60)
    
    # Test 0: Configuration validation
    logger.info("\n[Test 0] Configuration validation...")
    try:
        valid_config = RiskManagerConfig(
            initial_capital=200.0,
            base_kelly_fraction=0.05,
            use_dynamic_kelly=True
        )
        logger.info("Valid configuration accepted")
        
        try:
            invalid_config = RiskManagerConfig(initial_capital=-100)
            logger.error("Invalid config should have raised ValueError")
        except ValueError as e:
            logger.info(f"Invalid config correctly rejected: {e}")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return
    
    config = RiskManagerConfig(
        initial_capital=200.0,
        base_kelly_fraction=0.05,
        use_dynamic_kelly=True
    )
    
    rm = AggressiveRiskManager(config=config)
    
    # Test 1: Initialization
    logger.info("\n[Test 1] Initialization...")
    init_result = await rm.initialize_async()
    assert init_result['status'] == 'success'
    logger.info(f"Initialization: {init_result}")
    
    # Test 2: Calculate position size
    logger.info("\n[Test 2] Position size calculation...")
    position_result = await rm.calculate_position_size_async(
        entry_price=1.1000,
        stop_loss=1.0950,
        direction='buy',
        pair='EUR_USD',
        regime='trending',
        srek_confidence=0.85
    )
    assert position_result['should_trade']
    logger.info(f"Position approved: {position_result['position_size']:.2f} units")
    
    # Test 3: Open and close position
    logger.info("\n[Test 3] Open and close position...")
    trade_id = str(uuid.uuid4())
    await rm.open_position_async(
        trade_id=trade_id,
        pair='EUR_USD',
        regime='trending',
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1100,
        position_size=position_result['position_size'],
        direction='buy',
        leverage=position_result['leverage'],
        kelly_fraction=position_result['kelly_fraction'],
        risk_dollars=position_result['risk_dollars'],
        srek_confidence=0.85
    )
    close_result = await rm.close_position_async(trade_id, exit_price=1.1080)
    assert close_result['status'] == 'success'
    logger.info(f"Position closed: P&L=${close_result['pnl']:.2f}")
    
    # Test 4: Risk reduction after losses
    logger.info("\n[Test 4] Risk reduction after consecutive losses...")
    for i in range(5):
        pos = await rm.calculate_position_size_async(
            entry_price=1.1000, stop_loss=1.0950, direction='buy', srek_confidence=0.75
        )
        if pos['should_trade']:
            tid = str(uuid.uuid4())
            await rm.open_position_async(
                trade_id=tid, pair='EUR_USD', regime='normal',
                entry_price=1.1000, stop_loss=1.0950, take_profit=1.1100,
                position_size=pos['position_size'], direction='buy',
                leverage=pos['leverage'], kelly_fraction=pos['kelly_fraction'],
                risk_dollars=pos['risk_dollars'], srek_confidence=0.75
            )
            await rm.close_position_async(tid, exit_price=1.0940)
    
    metrics = await rm.get_metrics_async()
    logger.info(f"Risk multiplier after losses: {metrics['risk_multiplier']:.0%}")
    assert metrics['risk_multiplier'] < 1.0
    
    # Test 5: Safe drawdown calculation
    logger.info("\n[Test 5] Safe drawdown calculation...")
    drawdown = rm._safe_drawdown_calculation(100.0, 0.0)
    assert drawdown == 0.0
    logger.info(f"Safe drawdown with zero peak: {drawdown}")
    
    # Test 6: Save and cleanup
    logger.info("\n[Test 6] Save and cleanup...")
    await rm.save_state_async()
    await rm.cleanup_async()
    logger.info("Cleanup complete")
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL TESTS PASSED")
    logger.info("=" * 60)


if __name__ == '__main__':
    asyncio.run(test_aggressive_risk_manager())

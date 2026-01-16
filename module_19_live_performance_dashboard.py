#!/usr/bin/env python3
"""
============================================================================
MODULE 19: LIVE PERFORMANCE DASHBOARD
============================================================================
Version: 1.0.0
Author: MIT PhD-Level AI Engineering Team
VRAM: 0 MB (CPU-based metrics computation)

PURPOSE:
    Real-time performance monitoring and display showing:
    - Win rate (24h, 7d, 30d rolling)
    - Compound growth rate (24h, 7d, 30d)
    - Live P&L tracking
    - Trade statistics
    - System health metrics

FEATURES:
    - Rolling window statistics (24h, 7d, 30d)
    - Real-time compound growth calculation
    - Live trade monitoring
    - Performance alerts
    - Database-backed persistence
    - Thread-safe async operations

METRICS COMPUTED:
    - Win Rate: successful_trades / total_trades
    - Compound Rate: (current_capital / starting_capital)^(365/days) - 1
    - Sharpe Ratio: (avg_return - risk_free) / std_return
    - Max Drawdown: max((peak - trough) / peak)
    - Profit Factor: gross_profit / gross_loss

============================================================================
"""

import asyncio
import logging
import time
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Deque
from collections import deque
from enum import Enum
import json

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class TimeWindow(Enum):
    """Time windows for rolling statistics"""
    HOUR_24 = "24h"
    WEEK_7 = "7d"
    MONTH_30 = "30d"
    ALL_TIME = "all"


@dataclass
class TradeRecord:
    """Record of a single trade"""
    trade_id: str
    timestamp: datetime
    pair: str
    direction: str  # "BUY" or "SELL"
    entry_price: float
    exit_price: Optional[float] = None
    units: int = 0
    profit_loss: float = 0.0
    profit_pct: float = 0.0
    confidence: float = 0.0
    duration_seconds: int = 0
    is_winner: bool = False
    is_closed: bool = False
    exit_reason: str = ""


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot"""
    timestamp: datetime
    capital: float
    daily_pnl: float
    daily_pnl_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float


@dataclass
class RollingMetrics:
    """Rolling window metrics"""
    window: TimeWindow
    start_capital: float
    current_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    gross_profit: float
    gross_loss: float
    win_rate: float
    compound_rate_annualized: float
    avg_trade_duration_sec: float
    best_trade_pct: float
    worst_trade_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'window': self.window.value,
            'capital_start': self.start_capital,
            'capital_current': self.current_capital,
            'total_trades': self.total_trades,
            'wins': self.winning_trades,
            'losses': self.losing_trades,
            'win_rate': f"{self.win_rate * 100:.2f}%",
            'compound_rate': f"{self.compound_rate_annualized * 100:.2f}%",
            'sharpe': f"{self.sharpe_ratio:.2f}",
            'max_drawdown': f"{self.max_drawdown_pct * 100:.2f}%"
        }


@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    # Update intervals
    metrics_update_interval_sec: float = 5.0
    display_update_interval_sec: float = 1.0
    snapshot_interval_sec: float = 60.0
    
    # History settings
    max_trade_history: int = 10000
    max_snapshots: int = 1440  # 24 hours at 1-min intervals
    
    # Alerts
    alert_on_drawdown_pct: float = 0.05  # Alert at 5% drawdown
    alert_on_loss_streak: int = 5  # Alert after 5 consecutive losses
    alert_on_win_rate_below: float = 0.40  # Alert if win rate < 40%
    
    # Display
    show_live_trades: bool = True
    show_charts: bool = True
    colored_output: bool = True
    
    # Risk-free rate for Sharpe calculation (annualized)
    risk_free_rate: float = 0.05  # 5% annual


@dataclass
class DashboardState:
    """Current dashboard state"""
    is_running: bool = False
    last_update: float = 0.0
    starting_capital: float = 10000.0
    current_capital: float = 10000.0
    peak_capital: float = 10000.0
    session_start: datetime = field(default_factory=datetime.utcnow)
    
    # Trade tracking
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    current_streak: int = 0  # Positive = wins, negative = losses
    
    # P&L tracking
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    
    # Active positions
    active_positions: int = 0


# ============================================================================
# LIVE PERFORMANCE DASHBOARD
# ============================================================================

class LivePerformanceDashboard:
    """
    Real-time performance monitoring dashboard.
    
    Provides live metrics on:
    - Win rate (24h, 7d, 30d rolling windows)
    - Compound growth rate (annualized)
    - Trade statistics
    - System health
    
    Thread-safe with async operations.
    """
    
    def __init__(
        self,
        config: DashboardConfig = None,
        initial_capital: float = 10000.0
    ):
        self.config = config or DashboardConfig()
        
        # State
        self._state = DashboardState(
            starting_capital=initial_capital,
            current_capital=initial_capital,
            peak_capital=initial_capital
        )
        
        # Trade history (thread-safe deque)
        self._trades: Deque[TradeRecord] = deque(maxlen=self.config.max_trade_history)
        self._snapshots: Deque[PerformanceSnapshot] = deque(maxlen=self.config.max_snapshots)
        
        # Capital history for drawdown calculation
        self._capital_history: Deque[tuple] = deque(maxlen=10000)  # (timestamp, capital)
        
        # Locks for thread safety
        self._state_lock = asyncio.Lock()
        self._trades_lock = asyncio.Lock()
        self._snapshot_lock = asyncio.Lock()
        
        # Background tasks
        self._update_task: Optional[asyncio.Task] = None
        self._display_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._alert_callbacks: List[callable] = []
        
        # Metrics cache
        self._cached_metrics: Dict[TimeWindow, RollingMetrics] = {}
        self._cache_time: float = 0.0
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    async def initialize_async(self, starting_capital: float = None):
        """Initialize the dashboard"""
        async with self._state_lock:
            if starting_capital:
                self._state.starting_capital = starting_capital
                self._state.current_capital = starting_capital
                self._state.peak_capital = starting_capital
            
            self._state.session_start = datetime.utcnow()
            self._state.is_running = True
        
        # Record initial capital
        await self._record_capital_async(self._state.current_capital)
        
        logger.info(f"Dashboard initialized with capital: ${starting_capital or self._state.starting_capital:,.2f}")
    
    async def start_async(self):
        """Start background monitoring tasks"""
        self._update_task = asyncio.create_task(self._metrics_update_loop_async())
        logger.info("Performance dashboard started")
    
    async def stop_async(self):
        """Stop the dashboard"""
        async with self._state_lock:
            self._state.is_running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance dashboard stopped")
    
    # ========================================================================
    # TRADE RECORDING
    # ========================================================================
    
    async def record_trade_async(self, trade: TradeRecord):
        """
        Record a completed trade and update metrics.
        
        This is the main entry point for trade data.
        """
        async with self._trades_lock:
            self._trades.append(trade)
        
        async with self._state_lock:
            self._state.total_trades += 1
            
            if trade.is_winner:
                self._state.winning_trades += 1
                self._state.gross_profit += trade.profit_loss
                self._state.current_streak = max(1, self._state.current_streak + 1)
            else:
                self._state.losing_trades += 1
                self._state.gross_loss += abs(trade.profit_loss)
                self._state.current_streak = min(-1, self._state.current_streak - 1)
            
            # Update capital
            self._state.current_capital += trade.profit_loss
            
            # Update peak for drawdown
            if self._state.current_capital > self._state.peak_capital:
                self._state.peak_capital = self._state.current_capital
        
        # Record capital change
        await self._record_capital_async(self._state.current_capital)
        
        # Check alerts
        await self._check_alerts_async(trade)
        
        # Invalidate cache
        self._cache_time = 0
        
        logger.debug(f"Trade recorded: {trade.pair} {trade.direction} P&L: ${trade.profit_loss:.2f}")
    
    async def record_trade_from_dict_async(self, trade_data: Dict[str, Any]):
        """Record trade from dictionary data"""
        trade = TradeRecord(
            trade_id=trade_data.get('trade_id', str(time.time())),
            timestamp=trade_data.get('timestamp', datetime.utcnow()),
            pair=trade_data.get('pair', 'UNKNOWN'),
            direction=trade_data.get('direction', 'BUY'),
            entry_price=trade_data.get('entry_price', 0.0),
            exit_price=trade_data.get('exit_price'),
            units=trade_data.get('units', 0),
            profit_loss=trade_data.get('profit_loss', 0.0),
            profit_pct=trade_data.get('profit_pct', 0.0),
            confidence=trade_data.get('confidence', 0.0),
            duration_seconds=trade_data.get('duration_seconds', 0),
            is_winner=trade_data.get('profit_loss', 0) > 0,
            is_closed=True,
            exit_reason=trade_data.get('exit_reason', '')
        )
        await self.record_trade_async(trade)
    
    async def update_capital_async(self, new_capital: float):
        """Update current capital (for unrealized P&L)"""
        async with self._state_lock:
            self._state.current_capital = new_capital
            if new_capital > self._state.peak_capital:
                self._state.peak_capital = new_capital
        
        await self._record_capital_async(new_capital)
    
    async def _record_capital_async(self, capital: float):
        """Record capital for history tracking"""
        self._capital_history.append((time.time(), capital))
    
    # ========================================================================
    # METRICS CALCULATION
    # ========================================================================
    
    async def get_metrics_async(self, window: TimeWindow = TimeWindow.ALL_TIME) -> RollingMetrics:
        """
        Get performance metrics for specified time window.
        
        Implements caching to avoid repeated computation.
        """
        # Check cache (5 second TTL)
        if time.time() - self._cache_time < 5.0 and window in self._cached_metrics:
            return self._cached_metrics[window]
        
        # Calculate fresh metrics
        metrics = await self._calculate_metrics_async(window)
        
        # Update cache
        self._cached_metrics[window] = metrics
        self._cache_time = time.time()
        
        return metrics
    
    async def _calculate_metrics_async(self, window: TimeWindow) -> RollingMetrics:
        """Calculate metrics for a specific time window"""
        now = datetime.utcnow()
        
        # Determine cutoff time
        if window == TimeWindow.HOUR_24:
            cutoff = now - timedelta(hours=24)
        elif window == TimeWindow.WEEK_7:
            cutoff = now - timedelta(days=7)
        elif window == TimeWindow.MONTH_30:
            cutoff = now - timedelta(days=30)
        else:
            cutoff = datetime.min
        
        # Filter trades
        async with self._trades_lock:
            window_trades = [t for t in self._trades if t.timestamp >= cutoff]
        
        # Calculate base stats
        total_trades = len(window_trades)
        winning_trades = sum(1 for t in window_trades if t.is_winner)
        losing_trades = total_trades - winning_trades
        
        gross_profit = sum(t.profit_loss for t in window_trades if t.profit_loss > 0)
        gross_loss = sum(abs(t.profit_loss) for t in window_trades if t.profit_loss < 0)
        
        # Win rate
        win_rate = winning_trades / max(total_trades, 1)
        
        # Get capital at window start
        async with self._state_lock:
            current_capital = self._state.current_capital
            starting_capital = self._state.starting_capital
        
        # Find capital at window start from history
        window_start_capital = starting_capital
        cutoff_ts = cutoff.timestamp() if cutoff != datetime.min else 0
        for ts, cap in self._capital_history:
            if ts >= cutoff_ts:
                window_start_capital = cap
                break
        
        # Compound rate (annualized)
        if window == TimeWindow.HOUR_24:
            days = 1
        elif window == TimeWindow.WEEK_7:
            days = 7
        elif window == TimeWindow.MONTH_30:
            days = 30
        else:
            days = max(1, (now - self._state.session_start).days)
        
        if window_start_capital > 0 and current_capital > 0:
            total_return = current_capital / window_start_capital
            compound_rate = (total_return ** (365 / max(days, 1))) - 1
        else:
            compound_rate = 0.0
        
        # Average trade duration
        durations = [t.duration_seconds for t in window_trades if t.duration_seconds > 0]
        avg_duration = sum(durations) / max(len(durations), 1)
        
        # Best/worst trades
        profits = [t.profit_pct for t in window_trades]
        best_trade = max(profits) if profits else 0.0
        worst_trade = min(profits) if profits else 0.0
        
        # Sharpe ratio
        sharpe = await self._calculate_sharpe_async(window_trades, days)
        
        # Max drawdown
        max_dd = await self._calculate_max_drawdown_async(cutoff_ts)
        
        return RollingMetrics(
            window=window,
            start_capital=window_start_capital,
            current_capital=current_capital,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            win_rate=win_rate,
            compound_rate_annualized=compound_rate,
            avg_trade_duration_sec=avg_duration,
            best_trade_pct=best_trade,
            worst_trade_pct=worst_trade,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd
        )
    
    async def _calculate_sharpe_async(self, trades: List[TradeRecord], days: int) -> float:
        """Calculate Sharpe ratio from trades"""
        if len(trades) < 2:
            return 0.0
        
        # Daily returns
        returns = [t.profit_pct for t in trades]
        
        if not returns:
            return 0.0
        
        avg_return = sum(returns) / len(returns)
        
        # Standard deviation
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_return = math.sqrt(variance) if variance > 0 else 0.001
        
        # Annualize
        daily_rf = self.config.risk_free_rate / 365
        trades_per_day = len(returns) / max(days, 1)
        
        # Sharpe = (avg_return - rf) / std * sqrt(252)
        sharpe = ((avg_return - daily_rf) / std_return) * math.sqrt(252 * trades_per_day)
        
        return sharpe
    
    async def _calculate_max_drawdown_async(self, since_ts: float) -> float:
        """Calculate maximum drawdown since timestamp"""
        relevant_history = [(ts, cap) for ts, cap in self._capital_history if ts >= since_ts]
        
        if len(relevant_history) < 2:
            return 0.0
        
        peak = relevant_history[0][1]
        max_drawdown = 0.0
        
        for _, capital in relevant_history:
            if capital > peak:
                peak = capital
            
            drawdown = (peak - capital) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    # ========================================================================
    # LIVE DISPLAY
    # ========================================================================
    
    async def get_live_display_async(self) -> str:
        """Generate live display string"""
        # Get all window metrics
        metrics_24h = await self.get_metrics_async(TimeWindow.HOUR_24)
        metrics_7d = await self.get_metrics_async(TimeWindow.WEEK_7)
        metrics_30d = await self.get_metrics_async(TimeWindow.MONTH_30)
        metrics_all = await self.get_metrics_async(TimeWindow.ALL_TIME)
        
        async with self._state_lock:
            state = self._state
        
        # Calculate current drawdown
        current_dd = (state.peak_capital - state.current_capital) / state.peak_capital if state.peak_capital > 0 else 0
        
        # Format display
        lines = [
            "",
            "╔═══════════════════════════════════════════════════════════════════════════════╗",
            "║                    LIVE PERFORMANCE DASHBOARD                                 ║",
            "╠═══════════════════════════════════════════════════════════════════════════════╣",
            f"║  💰 CAPITAL: ${state.current_capital:>12,.2f}    │    📈 PEAK: ${state.peak_capital:>12,.2f}           ║",
            f"║  📊 DRAWDOWN: {current_dd*100:>6.2f}%              │    🎯 POSITIONS: {state.active_positions:>3}                    ║",
            "╠═══════════════════════════════════════════════════════════════════════════════╣",
            "║                           ROLLING METRICS                                     ║",
            "╠════════════╦═══════════════╦═══════════════╦═══════════════╦═════════════════╣",
            "║   WINDOW   ║   WIN RATE    ║ COMPOUND RATE ║    SHARPE     ║   MAX DRAWDOWN  ║",
            "╠════════════╬═══════════════╬═══════════════╬═══════════════╬═════════════════╣",
        ]
        
        # Add metrics rows
        for label, m in [("24 HOURS", metrics_24h), ("7 DAYS", metrics_7d), 
                         ("30 DAYS", metrics_30d), ("ALL TIME", metrics_all)]:
            wr = f"{m.win_rate*100:.1f}%"
            cr = f"{m.compound_rate_annualized*100:+.1f}%"
            sr = f"{m.sharpe_ratio:.2f}"
            dd = f"{m.max_drawdown_pct*100:.1f}%"
            
            # Color coding (conceptual - actual colors depend on terminal)
            lines.append(f"║  {label:<8}  ║  {wr:>11}  ║  {cr:>11}  ║  {sr:>11}  ║  {dd:>13}  ║")
        
        lines.extend([
            "╠════════════╩═══════════════╩═══════════════╩═══════════════╩═════════════════╣",
            f"║  TRADES: {state.total_trades:>5} total │ {state.winning_trades:>5} wins │ {state.losing_trades:>5} losses │ Streak: {state.current_streak:>+4}       ║",
            "╚═══════════════════════════════════════════════════════════════════════════════╝",
            f"  Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            ""
        ])
        
        return "\n".join(lines)
    
    async def get_metrics_dict_async(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary (for JSON export/API)"""
        metrics = {}
        
        for window in TimeWindow:
            m = await self.get_metrics_async(window)
            metrics[window.value] = m.to_dict()
        
        async with self._state_lock:
            state = self._state
        
        metrics['current'] = {
            'capital': state.current_capital,
            'peak_capital': state.peak_capital,
            'starting_capital': state.starting_capital,
            'total_trades': state.total_trades,
            'winning_trades': state.winning_trades,
            'losing_trades': state.losing_trades,
            'gross_profit': state.gross_profit,
            'gross_loss': state.gross_loss,
            'current_streak': state.current_streak,
            'active_positions': state.active_positions,
            'session_start': state.session_start.isoformat(),
            'current_drawdown_pct': (state.peak_capital - state.current_capital) / state.peak_capital if state.peak_capital > 0 else 0
        }
        
        return metrics
    
    # ========================================================================
    # ALERTS
    # ========================================================================
    
    def register_alert_callback(self, callback: callable):
        """Register a callback for performance alerts"""
        self._alert_callbacks.append(callback)
    
    async def _check_alerts_async(self, trade: TradeRecord):
        """Check for alert conditions"""
        alerts = []
        
        async with self._state_lock:
            state = self._state
        
        # Check drawdown
        current_dd = (state.peak_capital - state.current_capital) / state.peak_capital if state.peak_capital > 0 else 0
        if current_dd >= self.config.alert_on_drawdown_pct:
            alerts.append(f"⚠️ DRAWDOWN ALERT: {current_dd*100:.1f}% drawdown exceeded threshold")
        
        # Check loss streak
        if state.current_streak <= -self.config.alert_on_loss_streak:
            alerts.append(f"⚠️ LOSS STREAK ALERT: {abs(state.current_streak)} consecutive losses")
        
        # Check win rate (only after sufficient trades)
        if state.total_trades >= 20:
            win_rate = state.winning_trades / state.total_trades
            if win_rate < self.config.alert_on_win_rate_below:
                alerts.append(f"⚠️ WIN RATE ALERT: {win_rate*100:.1f}% below threshold")
        
        # Fire callbacks
        for alert in alerts:
            logger.warning(alert)
            for callback in self._alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
    
    # ========================================================================
    # BACKGROUND TASKS
    # ========================================================================
    
    async def _metrics_update_loop_async(self):
        """Background loop for metrics updates"""
        last_snapshot = 0
        
        while self._state.is_running:
            try:
                # Update cache
                for window in TimeWindow:
                    await self.get_metrics_async(window)
                
                # Take snapshot periodically
                if time.time() - last_snapshot >= self.config.snapshot_interval_sec:
                    await self._take_snapshot_async()
                    last_snapshot = time.time()
                
                await asyncio.sleep(self.config.metrics_update_interval_sec)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics update error: {e}")
                await asyncio.sleep(10)
    
    async def _take_snapshot_async(self):
        """Take a performance snapshot for history"""
        async with self._state_lock:
            state = self._state
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            capital=state.current_capital,
            daily_pnl=state.current_capital - state.starting_capital,
            daily_pnl_pct=(state.current_capital - state.starting_capital) / state.starting_capital if state.starting_capital > 0 else 0,
            total_trades=state.total_trades,
            winning_trades=state.winning_trades,
            losing_trades=state.losing_trades,
            win_rate=state.winning_trades / max(state.total_trades, 1),
            avg_win=state.gross_profit / max(state.winning_trades, 1),
            avg_loss=state.gross_loss / max(state.losing_trades, 1),
            profit_factor=state.gross_profit / max(state.gross_loss, 0.01),
            sharpe_ratio=0,  # Calculated separately
            max_drawdown=state.peak_capital - state.current_capital,
            max_drawdown_pct=(state.peak_capital - state.current_capital) / state.peak_capital if state.peak_capital > 0 else 0
        )
        
        async with self._snapshot_lock:
            self._snapshots.append(snapshot)
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    async def save_to_database_async(self, db_connection) -> bool:
        """Save current metrics to database"""
        try:
            metrics = await self.get_metrics_dict_async()
            
            db_connection.execute("""
                INSERT INTO performance_log 
                (timestamp, capital, daily_pnl, win_rate, sharpe_ratio, max_drawdown, trades_today)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow(),
                metrics['current']['capital'],
                metrics['current']['gross_profit'] - metrics['current']['gross_loss'],
                metrics['current']['winning_trades'] / max(metrics['current']['total_trades'], 1),
                metrics['24h'].get('sharpe', 0) if '24h' in metrics else 0,
                metrics['current']['current_drawdown_pct'],
                metrics['current']['total_trades']
            ))
            
            return True
        except Exception as e:
            logger.error(f"Database save error: {e}")
            return False
    
    async def export_to_json_async(self, filepath: str) -> bool:
        """Export metrics to JSON file"""
        try:
            metrics = await self.get_metrics_dict_async()
            
            # Use to_thread for blocking file I/O
            def _write_json_sync():
                with open(filepath, 'w') as f:
                    json.dump(metrics, f, indent=2, default=str)
            
            await asyncio.to_thread(_write_json_sync)
            
            return True
        except Exception as e:
            logger.error(f"JSON export error: {e}")
            return False


# ============================================================================
# FACTORY AND HELPERS
# ============================================================================

def create_dashboard(
    initial_capital: float = 10000.0,
    alert_drawdown_pct: float = 0.05
) -> LivePerformanceDashboard:
    """Factory function to create a configured dashboard"""
    config = DashboardConfig(
        alert_on_drawdown_pct=alert_drawdown_pct
    )
    return LivePerformanceDashboard(config=config, initial_capital=initial_capital)


# ============================================================================
# STANDALONE TEST
# ============================================================================

async def _test_dashboard():
    """Test the dashboard functionality"""
    print("Testing Live Performance Dashboard...")
    
    # Create dashboard
    dashboard = create_dashboard(initial_capital=10000.0)
    await dashboard.initialize_async()
    
    # Simulate some trades
    import random
    
    for i in range(20):
        is_winner = random.random() > 0.4  # 60% win rate
        pnl = random.uniform(50, 200) if is_winner else -random.uniform(30, 150)
        
        trade = TradeRecord(
            trade_id=f"test_{i}",
            timestamp=datetime.utcnow() - timedelta(hours=random.randint(0, 48)),
            pair=random.choice(["EUR_USD", "GBP_USD", "USD_JPY"]),
            direction=random.choice(["BUY", "SELL"]),
            entry_price=1.1000,
            exit_price=1.1000 + (pnl / 10000),
            units=1000,
            profit_loss=pnl,
            profit_pct=pnl / 10000,
            confidence=random.uniform(0.6, 0.95),
            duration_seconds=random.randint(300, 7200),
            is_winner=is_winner,
            is_closed=True
        )
        
        await dashboard.record_trade_async(trade)
    
    # Display metrics
    display = await dashboard.get_live_display_async()
    print(display)
    
    # Get dict
    metrics = await dashboard.get_metrics_dict_async()
    print("\nMetrics Dictionary:")
    print(json.dumps(metrics, indent=2, default=str))
    
    print("\n✅ Dashboard test complete!")


if __name__ == "__main__":
    asyncio.run(_test_dashboard())

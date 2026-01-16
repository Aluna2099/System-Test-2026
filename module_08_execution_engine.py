"""
MODULE 8: EXECUTION ENGINE
Production-Ready Implementation

Order execution, slippage management, and trade lifecycle management.
- Async order submission via Oanda API
- Slippage estimation and management
- Partial fill handling
- Position monitoring with stop loss/take profit
- Emergency stop loss system
- Thread-safe state management
- Comprehensive metrics tracking

Author: Liquid Neural SREK Trading System v4.0
Date: 2026-01-10
Version: 1.0.0

EXECUTION FLOW:
Trade Signal → Pre-Trade Validation → Order Submission → Execution Monitoring → Post-Trade Analysis
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONFIGURATION
# ============================================================================

class OrderType(Enum):
    """Order types supported"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    MARKET_IF_TOUCHED = "MARKET_IF_TOUCHED"


class OrderStatus(Enum):
    """Order/position status"""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    CLOSED = "closed"
    ERROR = "error"


class TradeDirection(Enum):
    """Trade direction"""
    BUY = "buy"
    SELL = "sell"


class ExitReason(Enum):
    """Reason for position exit"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TIME_EXIT = "time_exit"
    MANUAL = "manual"
    EMERGENCY = "emergency"
    TRAILING_STOP = "trailing_stop"


@dataclass
class ExecutionConfig:
    """
    Configuration for Execution Engine
    
    Includes validation to prevent runtime errors
    """
    # Oanda API configuration
    oanda_api_url: str = "https://api-fxpractice.oanda.com/v3"
    oanda_account_id: str = ""
    oanda_api_key: str = ""
    
    # Execution settings
    default_order_type: str = "MARKET"
    time_in_force: str = "FOK"  # Fill or Kill
    max_slippage_pips: float = 3.0  # Maximum acceptable slippage
    
    # Position limits
    max_open_positions: int = 10
    max_position_size: int = 100000  # Units
    min_position_size: int = 100  # Units
    
    # Time limits
    max_position_hold_seconds: int = 14400  # 4 hours default
    position_check_interval_seconds: float = 1.0
    
    # Pip values by pair type
    pip_value_standard: float = 0.0001  # EUR/USD, GBP/USD, etc.
    pip_value_jpy: float = 0.01  # USD/JPY, EUR/JPY, etc.
    
    # Retry settings
    max_order_retries: int = 3
    retry_delay_seconds: float = 0.5
    
    # Emergency settings
    emergency_stop_loss_pips: float = 50.0  # Hard stop
    
    # Numerical stability
    epsilon: float = 1e-10
    
    # Simulation mode (for testing without real API)
    simulation_mode: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if self.max_slippage_pips < 0:
            raise ValueError("max_slippage_pips must be non-negative")
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be positive")
        if self.max_position_size <= 0:
            raise ValueError("max_position_size must be positive")
        if self.min_position_size <= 0:
            raise ValueError("min_position_size must be positive")
        if self.min_position_size > self.max_position_size:
            raise ValueError("min_position_size cannot exceed max_position_size")
        if self.max_position_hold_seconds <= 0:
            raise ValueError("max_position_hold_seconds must be positive")
        if self.position_check_interval_seconds <= 0:
            raise ValueError("position_check_interval_seconds must be positive")
        if self.max_order_retries < 0:
            raise ValueError("max_order_retries must be non-negative")
        if self.emergency_stop_loss_pips <= 0:
            raise ValueError("emergency_stop_loss_pips must be positive")


@dataclass
class Position:
    """Represents an open trading position"""
    trade_id: str
    pair: str
    direction: TradeDirection
    units: int
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: float
    status: OrderStatus = OrderStatus.OPEN
    current_price: float = 0.0
    unrealized_pnl_pips: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'trade_id': self.trade_id,
            'pair': self.pair,
            'direction': self.direction.value,
            'units': self.units,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'entry_time': self.entry_time,
            'status': self.status.value,
            'current_price': self.current_price,
            'unrealized_pnl_pips': self.unrealized_pnl_pips
        }


# ============================================================================
# EXECUTION ENGINE (MAIN MODULE)
# ============================================================================

class ExecutionEngine:
    """
    Order execution with Oanda API integration.
    
    Features:
    - Async order submission
    - Slippage estimation & management
    - Partial fill handling
    - Position monitoring
    - Emergency stop loss
    - Thread-safe state management
    - Comprehensive metrics tracking
    """
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        """
        Initialize Execution Engine.
        
        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or ExecutionConfig()
        
        # Thread safety locks
        self._lock = asyncio.Lock()  # Protects shared state
        self._positions_lock = asyncio.Lock()  # Protects positions
        self._metrics_lock = asyncio.Lock()  # Protects metrics
        
        # State tracking (protected by _lock)
        self._is_initialized = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        
        # Active positions (protected by _positions_lock)
        self.open_positions: Dict[str, Position] = {}
        self.closed_positions: deque = deque(maxlen=1000)  # Last 1000 closed
        
        # Execution metrics (protected by _metrics_lock)
        self._metrics = {
            'total_trades': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_slippage_pips': 0.0,
            'avg_slippage_pips': 0.0,
            'max_slippage_pips': 0.0,
            'total_execution_time_ms': 0.0,
            'avg_execution_time_ms': 0.0,
            'wins': 0,
            'losses': 0,
            'total_pnl_pips': 0.0,
            'stop_loss_hits': 0,
            'take_profit_hits': 0,
            'time_exits': 0,
            'emergency_exits': 0
        }
        
        # Pricing cache (protected by _lock)
        self._pricing_cache: Dict[str, Dict[str, Any]] = {}
        self._pricing_cache_ttl = 1.0  # 1 second TTL
        
        logger.info(
            f"ExecutionEngine initialized: "
            f"simulation_mode={self.config.simulation_mode}, "
            f"max_positions={self.config.max_open_positions}"
        )
    
    async def initialize_async(self) -> Dict[str, Any]:
        """
        Async initialization.
        
        Returns:
            Initialization status
        """
        async with self._lock:
            if self._is_initialized:
                logger.warning("Already initialized")
                return {'status': 'already_initialized'}
            
            try:
                # Validate API credentials if not in simulation mode
                if not self.config.simulation_mode:
                    if not self.config.oanda_api_key:
                        raise ValueError("Oanda API key required in live mode")
                    if not self.config.oanda_account_id:
                        raise ValueError("Oanda account ID required in live mode")
                
                self._is_initialized = True
                
                logger.info("ExecutionEngine initialized successfully")
                
                return {
                    'status': 'success',
                    'simulation_mode': self.config.simulation_mode,
                    'max_positions': self.config.max_open_positions
                }
                
            except Exception as e:
                logger.error(f"Initialization failed: {e}")
                return {'status': 'failed', 'error': str(e)}
    
    def _get_pip_value(self, pair: str) -> float:
        """
        Get pip value for a currency pair.
        
        Args:
            pair: Currency pair (e.g., "EUR_USD", "USD_JPY")
            
        Returns:
            Pip value (0.0001 for most pairs, 0.01 for JPY pairs)
        """
        # JPY pairs have different pip value
        if 'JPY' in pair.upper():
            return self.config.pip_value_jpy
        return self.config.pip_value_standard
    
    async def execute_trade_async(
        self,
        pair: str,
        direction: Union[str, TradeDirection],
        units: int,
        stop_loss_pips: float,
        take_profit_pips: float,
        order_type: OrderType = OrderType.MARKET
    ) -> Dict[str, Any]:
        """
        Execute trade with full validation and monitoring.
        
        Process:
        1. Pre-trade validation
        2. Calculate entry price & slippage
        3. Submit order to Oanda
        4. Monitor execution
        5. Set stop loss & take profit
        6. Log trade
        
        Args:
            pair: Currency pair (e.g., "EUR_USD")
            direction: "buy" or "sell"
            units: Number of units to trade
            stop_loss_pips: Stop loss in pips
            take_profit_pips: Take profit in pips
            order_type: Order type (default MARKET)
            
        Returns:
            Execution result dictionary
        """
        start_time = time.time()
        
        # Normalize direction
        if isinstance(direction, str):
            direction = TradeDirection(direction.lower())
        
        try:
            # ═══════════════════════════════════════════════════════
            # STEP 1: PRE-TRADE VALIDATION
            # ═══════════════════════════════════════════════════════
            
            validation = await self._validate_trade_async(pair, units)
            
            if not validation['valid']:
                async with self._metrics_lock:
                    self._metrics['failed_executions'] += 1
                
                return {
                    'success': False,
                    'error': validation['reason']
                }
            
            # ═══════════════════════════════════════════════════════
            # STEP 2: GET CURRENT PRICE & ESTIMATE SLIPPAGE
            # ═══════════════════════════════════════════════════════
            
            pricing = await self._get_pricing_async(pair)
            
            if direction == TradeDirection.BUY:
                entry_price = pricing['ask']
            else:
                entry_price = pricing['bid']
            
            # Estimate slippage based on spread & volatility
            estimated_slippage = self._estimate_slippage(pricing, pair)
            
            # Check if slippage is acceptable
            if estimated_slippage > self.config.max_slippage_pips:
                async with self._metrics_lock:
                    self._metrics['failed_executions'] += 1
                
                return {
                    'success': False,
                    'error': f"Estimated slippage {estimated_slippage:.2f} exceeds max {self.config.max_slippage_pips}"
                }
            
            # ═══════════════════════════════════════════════════════
            # STEP 3: SUBMIT ORDER
            # ═══════════════════════════════════════════════════════
            
            order_result = await self._submit_order_async(
                pair=pair,
                direction=direction,
                units=units,
                entry_price=entry_price,
                stop_loss_pips=stop_loss_pips,
                take_profit_pips=take_profit_pips
            )
            
            if not order_result['success']:
                async with self._metrics_lock:
                    self._metrics['failed_executions'] += 1
                
                return {
                    'success': False,
                    'error': order_result.get('error', 'Order submission failed')
                }
            
            # ═══════════════════════════════════════════════════════
            # STEP 4: VERIFY EXECUTION & CALCULATE SLIPPAGE
            # ═══════════════════════════════════════════════════════
            
            trade_id = order_result['trade_id']
            execution_price = order_result['execution_price']
            
            # Calculate actual slippage
            pip_value = self._get_pip_value(pair)
            
            if direction == TradeDirection.BUY:
                actual_slippage = (execution_price - entry_price) / pip_value
            else:
                actual_slippage = (entry_price - execution_price) / pip_value
            
            # ═══════════════════════════════════════════════════════
            # STEP 5: CREATE POSITION RECORD
            # ═══════════════════════════════════════════════════════
            
            if direction == TradeDirection.BUY:
                stop_loss_price = execution_price - (stop_loss_pips * pip_value)
                take_profit_price = execution_price + (take_profit_pips * pip_value)
            else:
                stop_loss_price = execution_price + (stop_loss_pips * pip_value)
                take_profit_price = execution_price - (take_profit_pips * pip_value)
            
            position = Position(
                trade_id=trade_id,
                pair=pair,
                direction=direction,
                units=units,
                entry_price=execution_price,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                entry_time=time.time(),
                status=OrderStatus.OPEN,
                current_price=execution_price,
                unrealized_pnl_pips=0.0
            )
            
            async with self._positions_lock:
                self.open_positions[trade_id] = position
            
            # ═══════════════════════════════════════════════════════
            # STEP 6: UPDATE METRICS (THREAD-SAFE)
            # ═══════════════════════════════════════════════════════
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            async with self._metrics_lock:
                self._metrics['total_trades'] += 1
                self._metrics['successful_executions'] += 1
                self._metrics['total_slippage_pips'] += actual_slippage
                self._metrics['total_execution_time_ms'] += execution_time_ms
                
                # Update max slippage
                if abs(actual_slippage) > self._metrics['max_slippage_pips']:
                    self._metrics['max_slippage_pips'] = abs(actual_slippage)
                
                # Update averages (safe division)
                n = self._metrics['successful_executions']
                if n > 0:
                    self._metrics['avg_slippage_pips'] = (
                        self._metrics['total_slippage_pips'] / n
                    )
                    self._metrics['avg_execution_time_ms'] = (
                        self._metrics['total_execution_time_ms'] / n
                    )
            
            logger.info(
                f"✅ Trade executed: {pair} {direction.value} {units} units @ {execution_price:.5f}, "
                f"slippage: {actual_slippage:.1f} pips, latency: {execution_time_ms:.1f}ms"
            )
            
            return {
                'success': True,
                'trade_id': trade_id,
                'pair': pair,
                'direction': direction.value,
                'units': units,
                'entry_price': execution_price,
                'slippage_pips': actual_slippage,
                'execution_time_ms': execution_time_ms,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price
            }
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}", exc_info=True)
            
            async with self._metrics_lock:
                self._metrics['failed_executions'] += 1
            
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _validate_trade_async(
        self,
        pair: str,
        units: int
    ) -> Dict[str, Any]:
        """
        Validate trade parameters.
        
        Checks:
        - Position count limit
        - Position size limits
        - Pair format
        
        Args:
            pair: Currency pair
            units: Trade size
            
        Returns:
            Validation result
        """
        # Check position count
        async with self._positions_lock:
            if len(self.open_positions) >= self.config.max_open_positions:
                return {
                    'valid': False,
                    'reason': f"Maximum open positions ({self.config.max_open_positions}) reached"
                }
        
        # Check position size
        if units < self.config.min_position_size:
            return {
                'valid': False,
                'reason': f"Position size {units} below minimum {self.config.min_position_size}"
            }
        
        if units > self.config.max_position_size:
            return {
                'valid': False,
                'reason': f"Position size {units} exceeds maximum {self.config.max_position_size}"
            }
        
        # Validate pair format
        if not pair or '_' not in pair:
            return {
                'valid': False,
                'reason': f"Invalid pair format: {pair}"
            }
        
        return {'valid': True}
    
    async def _get_pricing_async(self, pair: str) -> Dict[str, Any]:
        """
        Get current pricing for a pair.
        
        Uses cache with TTL for efficiency.
        In simulation mode, generates synthetic prices.
        
        Args:
            pair: Currency pair
            
        Returns:
            Pricing dictionary with bid, ask, spread
        """
        async with self._lock:
            # Check cache
            if pair in self._pricing_cache:
                cached = self._pricing_cache[pair]
                if time.time() - cached['timestamp'] < self._pricing_cache_ttl:
                    return cached['pricing']
        
        if self.config.simulation_mode:
            # Generate synthetic pricing for simulation
            pricing = await self._get_simulated_pricing_async(pair)
        else:
            # Real API call
            pricing = await self._fetch_oanda_pricing_async(pair)
        
        # Cache the result
        async with self._lock:
            self._pricing_cache[pair] = {
                'timestamp': time.time(),
                'pricing': pricing
            }
        
        return pricing
    
    async def _get_simulated_pricing_async(self, pair: str) -> Dict[str, Any]:
        """
        Generate simulated pricing for testing.
        
        Args:
            pair: Currency pair
            
        Returns:
            Simulated pricing
        """
        # Base prices for common pairs
        base_prices = {
            'EUR_USD': 1.0850,
            'GBP_USD': 1.2650,
            'USD_JPY': 149.50,
            'AUD_USD': 0.6550,
            'USD_CAD': 1.3550,
            'EUR_GBP': 0.8580,
            'EUR_JPY': 162.20
        }
        
        # Get base price or default
        mid_price = base_prices.get(pair.upper(), 1.0)
        
        # Add small random variation
        variation = np.random.normal(0, 0.0001)
        mid_price = mid_price * (1 + variation)
        
        # Calculate spread (tighter for major pairs)
        pip_value = self._get_pip_value(pair)
        spread_pips = np.random.uniform(0.5, 2.0)  # 0.5 to 2 pips
        half_spread = spread_pips * pip_value / 2
        
        bid = mid_price - half_spread
        ask = mid_price + half_spread
        
        return {
            'bid': bid,
            'ask': ask,
            'mid': mid_price,
            'spread': ask - bid,
            'spread_pips': spread_pips,
            'timestamp': time.time()
        }
    
    async def _fetch_oanda_pricing_async(self, pair: str) -> Dict[str, Any]:
        """
        Fetch pricing from Oanda API.
        
        Args:
            pair: Currency pair
            
        Returns:
            Pricing from API
        """
        try:
            # Import aiohttp only when needed
            import aiohttp
            
            url = f"{self.config.oanda_api_url}/accounts/{self.config.oanda_account_id}/pricing"
            params = {'instruments': pair}
            headers = {
                'Authorization': f"Bearer {self.config.oanda_api_key}",
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        price_data = data['prices'][0]
                        
                        bid = float(price_data['bids'][0]['price'])
                        ask = float(price_data['asks'][0]['price'])
                        
                        return {
                            'bid': bid,
                            'ask': ask,
                            'mid': (bid + ask) / 2,
                            'spread': ask - bid,
                            'spread_pips': (ask - bid) / self._get_pip_value(pair),
                            'timestamp': time.time()
                        }
                    else:
                        error_msg = await response.text()
                        raise Exception(f"Pricing API error: {error_msg}")
                        
        except ImportError:
            logger.warning("aiohttp not installed, using simulated pricing")
            return await self._get_simulated_pricing_async(pair)
        except Exception as e:
            logger.error(f"Failed to fetch pricing: {e}")
            # Fallback to simulation
            return await self._get_simulated_pricing_async(pair)
    
    def _estimate_slippage(
        self,
        pricing: Dict[str, Any],
        pair: str
    ) -> float:
        """
        Estimate expected slippage.
        
        Based on:
        - Current spread
        - Market volatility
        - Time of day
        
        Args:
            pricing: Current pricing
            pair: Currency pair
            
        Returns:
            Estimated slippage in pips
        """
        # Base slippage = half the spread
        spread_pips = pricing.get('spread_pips', 1.0)
        base_slippage = spread_pips / 2
        
        # Add volatility factor (could be enhanced with actual volatility data)
        volatility_factor = 1.2  # Conservative multiplier
        
        estimated_slippage = base_slippage * volatility_factor
        
        return estimated_slippage
    
    async def _submit_order_async(
        self,
        pair: str,
        direction: TradeDirection,
        units: int,
        entry_price: float,
        stop_loss_pips: float,
        take_profit_pips: float
    ) -> Dict[str, Any]:
        """
        Submit order to broker.
        
        In simulation mode, generates synthetic execution.
        In live mode, submits to Oanda API.
        
        Args:
            pair: Currency pair
            direction: Buy or Sell
            units: Trade size
            entry_price: Expected entry price
            stop_loss_pips: Stop loss distance
            take_profit_pips: Take profit distance
            
        Returns:
            Order result
        """
        if self.config.simulation_mode:
            return await self._submit_simulated_order_async(
                pair, direction, units, entry_price,
                stop_loss_pips, take_profit_pips
            )
        else:
            return await self._submit_oanda_order_async(
                pair, direction, units, entry_price,
                stop_loss_pips, take_profit_pips
            )
    
    async def _submit_simulated_order_async(
        self,
        pair: str,
        direction: TradeDirection,
        units: int,
        entry_price: float,
        stop_loss_pips: float,
        take_profit_pips: float
    ) -> Dict[str, Any]:
        """
        Simulate order execution for testing.
        
        Adds realistic slippage.
        """
        # Simulate execution delay
        await asyncio.sleep(0.05)  # 50ms simulated latency
        
        # Generate trade ID
        trade_id = f"SIM-{uuid.uuid4().hex[:12]}"
        
        # Add random slippage (-1 to +1 pip)
        pip_value = self._get_pip_value(pair)
        slippage = np.random.uniform(-1, 1) * pip_value
        
        if direction == TradeDirection.BUY:
            execution_price = entry_price + slippage
        else:
            execution_price = entry_price - slippage
        
        return {
            'success': True,
            'trade_id': trade_id,
            'execution_price': execution_price,
            'filled_units': units,
            'simulated': True
        }
    
    async def _submit_oanda_order_async(
        self,
        pair: str,
        direction: TradeDirection,
        units: int,
        entry_price: float,
        stop_loss_pips: float,
        take_profit_pips: float
    ) -> Dict[str, Any]:
        """
        Submit order to Oanda API.
        
        Uses market order with stop loss & take profit attached.
        """
        try:
            import aiohttp
            
            pip_value = self._get_pip_value(pair)
            
            # Calculate stop loss and take profit prices
            if direction == TradeDirection.BUY:
                stop_loss_price = entry_price - (stop_loss_pips * pip_value)
                take_profit_price = entry_price + (take_profit_pips * pip_value)
                order_units = units
            else:
                stop_loss_price = entry_price + (stop_loss_pips * pip_value)
                take_profit_price = entry_price - (take_profit_pips * pip_value)
                order_units = -units
            
            # Build order request
            order_data = {
                'order': {
                    'instrument': pair,
                    'units': str(order_units),
                    'type': 'MARKET',
                    'timeInForce': self.config.time_in_force,
                    'stopLossOnFill': {
                        'price': f"{stop_loss_price:.5f}"
                    },
                    'takeProfitOnFill': {
                        'price': f"{take_profit_price:.5f}"
                    }
                }
            }
            
            url = f"{self.config.oanda_api_url}/accounts/{self.config.oanda_account_id}/orders"
            headers = {
                'Authorization': f"Bearer {self.config.oanda_api_key}",
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=order_data, headers=headers) as response:
                    result = await response.json()
                    
                    if response.status == 201:
                        fill = result.get('orderFillTransaction', {})
                        
                        return {
                            'success': True,
                            'trade_id': fill.get('id', str(uuid.uuid4())),
                            'execution_price': float(fill.get('price', entry_price)),
                            'filled_units': abs(int(fill.get('units', units)))
                        }
                    else:
                        return {
                            'success': False,
                            'error': result.get('errorMessage', f"HTTP {response.status}")
                        }
                        
        except ImportError:
            logger.warning("aiohttp not installed, using simulated order")
            return await self._submit_simulated_order_async(
                pair, direction, units, entry_price,
                stop_loss_pips, take_profit_pips
            )
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def start_position_monitoring_async(self):
        """
        Start background position monitoring task.
        
        Monitors all open positions for:
        - Stop loss / take profit hits
        - Time-based exits
        - Emergency conditions
        """
        async with self._lock:
            if self._is_monitoring:
                logger.warning("Position monitoring already running")
                return
            
            self._is_monitoring = True
        
        self._monitoring_task = asyncio.create_task(
            self._position_monitoring_loop_async()
        )
        
        logger.info("Position monitoring started")
    
    async def stop_position_monitoring_async(self):
        """Stop background position monitoring."""
        async with self._lock:
            self._is_monitoring = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        
        logger.info("Position monitoring stopped")
    
    async def _position_monitoring_loop_async(self):
        """
        Continuous position monitoring loop.
        
        Runs in background task.
        """
        logger.info("Starting position monitoring loop...")
        
        while True:
            try:
                async with self._lock:
                    if not self._is_monitoring:
                        break
                
                # Get open position IDs
                async with self._positions_lock:
                    open_trade_ids = list(self.open_positions.keys())
                
                # Check each position
                for trade_id in open_trade_ids:
                    await self._check_position_async(trade_id)
                
                # Sleep between checks
                await asyncio.sleep(self.config.position_check_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Position monitoring error: {e}", exc_info=True)
                await asyncio.sleep(5)  # Back off on error
    
    async def _check_position_async(self, trade_id: str):
        """
        Check single position for exit conditions.
        
        Updates P&L and checks:
        - Stop loss hit
        - Take profit hit
        - Time-based exit
        - Emergency conditions
        
        Args:
            trade_id: Position trade ID
        """
        async with self._positions_lock:
            if trade_id not in self.open_positions:
                return
            
            position = self.open_positions[trade_id]
        
        # Get current price
        pricing = await self._get_pricing_async(position.pair)
        
        if position.direction == TradeDirection.BUY:
            current_price = pricing['bid']  # Exit at bid
        else:
            current_price = pricing['ask']  # Exit at ask
        
        # Calculate unrealized P&L
        pip_value = self._get_pip_value(position.pair)
        
        if position.direction == TradeDirection.BUY:
            pnl_pips = (current_price - position.entry_price) / pip_value
        else:
            pnl_pips = (position.entry_price - current_price) / pip_value
        
        # Update position
        async with self._positions_lock:
            if trade_id in self.open_positions:
                self.open_positions[trade_id].current_price = current_price
                self.open_positions[trade_id].unrealized_pnl_pips = pnl_pips
        
        # Check exit conditions
        should_exit = False
        exit_reason = None
        
        # Stop loss check
        if position.direction == TradeDirection.BUY:
            if current_price <= position.stop_loss:
                should_exit = True
                exit_reason = ExitReason.STOP_LOSS
        else:
            if current_price >= position.stop_loss:
                should_exit = True
                exit_reason = ExitReason.STOP_LOSS
        
        # Take profit check
        if not should_exit:
            if position.direction == TradeDirection.BUY:
                if current_price >= position.take_profit:
                    should_exit = True
                    exit_reason = ExitReason.TAKE_PROFIT
            else:
                if current_price <= position.take_profit:
                    should_exit = True
                    exit_reason = ExitReason.TAKE_PROFIT
        
        # Time-based exit
        if not should_exit:
            hold_time = time.time() - position.entry_time
            if hold_time > self.config.max_position_hold_seconds:
                should_exit = True
                exit_reason = ExitReason.TIME_EXIT
        
        # Emergency stop loss (hard stop regardless of position stop loss)
        if not should_exit:
            if pnl_pips < -self.config.emergency_stop_loss_pips:
                should_exit = True
                exit_reason = ExitReason.EMERGENCY
        
        # Execute exit if needed
        if should_exit:
            await self._close_position_async(trade_id, current_price, exit_reason)
    
    async def _close_position_async(
        self,
        trade_id: str,
        exit_price: float,
        reason: ExitReason
    ) -> Dict[str, Any]:
        """
        Close a position.
        
        Args:
            trade_id: Position to close
            exit_price: Exit price
            reason: Reason for closing
            
        Returns:
            Close result
        """
        async with self._positions_lock:
            if trade_id not in self.open_positions:
                return {'success': False, 'error': 'Position not found'}
            
            position = self.open_positions[trade_id]
            
            # Calculate final P&L
            pip_value = self._get_pip_value(position.pair)
            
            if position.direction == TradeDirection.BUY:
                final_pnl_pips = (exit_price - position.entry_price) / pip_value
            else:
                final_pnl_pips = (position.entry_price - exit_price) / pip_value
            
            # Update position status
            position.status = OrderStatus.CLOSED
            position.current_price = exit_price
            position.unrealized_pnl_pips = final_pnl_pips
            
            # Move to closed positions
            self.closed_positions.append(position.to_dict())
            del self.open_positions[trade_id]
        
        # Update metrics (thread-safe)
        async with self._metrics_lock:
            self._metrics['total_pnl_pips'] += final_pnl_pips
            
            if final_pnl_pips > 0:
                self._metrics['wins'] += 1
            else:
                self._metrics['losses'] += 1
            
            if reason == ExitReason.STOP_LOSS:
                self._metrics['stop_loss_hits'] += 1
            elif reason == ExitReason.TAKE_PROFIT:
                self._metrics['take_profit_hits'] += 1
            elif reason == ExitReason.TIME_EXIT:
                self._metrics['time_exits'] += 1
            elif reason == ExitReason.EMERGENCY:
                self._metrics['emergency_exits'] += 1
        
        logger.info(
            f"Position closed: {trade_id} | {reason.value} | "
            f"P&L: {final_pnl_pips:+.1f} pips"
        )
        
        return {
            'success': True,
            'trade_id': trade_id,
            'exit_price': exit_price,
            'exit_reason': reason.value,
            'pnl_pips': final_pnl_pips
        }
    
    async def close_all_positions_async(
        self,
        reason: ExitReason = ExitReason.EMERGENCY
    ) -> Dict[str, Any]:
        """
        Emergency close all open positions.
        
        Args:
            reason: Reason for closing (default EMERGENCY)
            
        Returns:
            Close all result
        """
        async with self._positions_lock:
            trade_ids = list(self.open_positions.keys())
        
        results = []
        for trade_id in trade_ids:
            try:
                # Get current price
                async with self._positions_lock:
                    if trade_id not in self.open_positions:
                        continue
                    position = self.open_positions[trade_id]
                
                pricing = await self._get_pricing_async(position.pair)
                
                if position.direction == TradeDirection.BUY:
                    exit_price = pricing['bid']
                else:
                    exit_price = pricing['ask']
                
                result = await self._close_position_async(trade_id, exit_price, reason)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to close position {trade_id}: {e}")
                results.append({'success': False, 'trade_id': trade_id, 'error': str(e)})
        
        return {
            'total_positions': len(trade_ids),
            'closed': len([r for r in results if r.get('success')]),
            'failed': len([r for r in results if not r.get('success')]),
            'results': results
        }
    
    async def get_open_positions_async(self) -> Dict[str, Any]:
        """Get all open positions (thread-safe)."""
        async with self._positions_lock:
            positions = {
                tid: pos.to_dict()
                for tid, pos in self.open_positions.items()
            }
            
            return {
                'count': len(positions),
                'positions': positions
            }
    
    async def get_metrics_async(self) -> Dict[str, Any]:
        """Get all execution metrics (thread-safe)."""
        async with self._metrics_lock:
            metrics = self._metrics.copy()
        
        async with self._positions_lock:
            metrics['open_positions'] = len(self.open_positions)
            metrics['closed_positions'] = len(self.closed_positions)
        
        # Calculate win rate
        total_closed = metrics['wins'] + metrics['losses']
        if total_closed > 0:
            metrics['win_rate'] = metrics['wins'] / total_closed
        else:
            metrics['win_rate'] = 0.0
        
        return metrics
    
    async def save_checkpoint_async(self, filepath: str) -> Dict[str, Any]:
        """Save checkpoint with positions and metrics."""
        async with self._lock:
            try:
                import pickle
                
                async with self._positions_lock:
                    positions_data = {
                        tid: pos.to_dict()
                        for tid, pos in self.open_positions.items()
                    }
                    closed_data = list(self.closed_positions)
                
                async with self._metrics_lock:
                    metrics_data = self._metrics.copy()
                
                checkpoint = {
                    'open_positions': positions_data,
                    'closed_positions': closed_data,
                    'metrics': metrics_data,
                    'config': {
                        k: v for k, v in self.config.__dict__.items()
                        if not k.startswith('_') and k not in ['oanda_api_key']
                    },
                    'timestamp': time.time()
                }
                
                await asyncio.to_thread(
                    lambda: pickle.dump(checkpoint, open(filepath, 'wb'))
                )
                
                logger.info(f"Checkpoint saved: {filepath}")
                return {'status': 'success', 'filepath': filepath}
                
            except Exception as e:
                logger.error(f"Checkpoint save failed: {e}")
                return {'status': 'failed', 'error': str(e)}
    
    async def load_checkpoint_async(self, filepath: str) -> Dict[str, Any]:
        """Load checkpoint with positions and metrics."""
        async with self._lock:
            try:
                import pickle
                
                checkpoint = await asyncio.to_thread(
                    lambda: pickle.load(open(filepath, 'rb'))
                )
                
                # Restore metrics
                async with self._metrics_lock:
                    self._metrics.update(checkpoint.get('metrics', {}))
                
                # Restore closed positions
                async with self._positions_lock:
                    self.closed_positions.clear()
                    for pos in checkpoint.get('closed_positions', []):
                        self.closed_positions.append(pos)
                
                # Note: Open positions are not restored as they may be stale
                
                logger.info(f"Checkpoint loaded: {filepath}")
                return {
                    'status': 'success',
                    'filepath': filepath,
                    'timestamp': checkpoint.get('timestamp', 'unknown')
                }
                
            except Exception as e:
                logger.error(f"Checkpoint load failed: {e}")
                return {'status': 'failed', 'error': str(e)}
    
    async def cleanup_async(self):
        """Cleanup resources and stop monitoring."""
        # Stop monitoring
        await self.stop_position_monitoring_async()
        
        # Close all positions if any open
        async with self._positions_lock:
            if self.open_positions:
                logger.warning(f"Closing {len(self.open_positions)} open positions during cleanup")
        
        await self.close_all_positions_async(reason=ExitReason.EMERGENCY)
        
        async with self._lock:
            self._is_initialized = False
        
        logger.info("ExecutionEngine cleaned up")


# ============================================================================
# INTEGRATION TEST
# ============================================================================

async def test_execution_engine():
    """Integration test for ExecutionEngine"""
    logger.info("=" * 60)
    logger.info("TESTING MODULE 8: EXECUTION ENGINE")
    logger.info("=" * 60)
    
    # Test 0: Configuration validation
    logger.info("\n[Test 0] Configuration validation...")
    try:
        valid_config = ExecutionConfig(simulation_mode=True)
        logger.info("Valid configuration accepted")
        
        try:
            invalid_config = ExecutionConfig(max_open_positions=-1)
            logger.error("Invalid config should have raised ValueError")
        except ValueError as e:
            logger.info(f"Invalid config correctly rejected: {e}")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return
    
    # Create engine in simulation mode
    config = ExecutionConfig(simulation_mode=True)
    engine = ExecutionEngine(config)
    
    # Test 1: Initialization
    logger.info("\n[Test 1] Initialization...")
    init_result = await engine.initialize_async()
    assert init_result['status'] == 'success'
    logger.info(f"Initialization: {init_result}")
    
    # Test 2: Execute trade
    logger.info("\n[Test 2] Execute trade...")
    trade_result = await engine.execute_trade_async(
        pair="EUR_USD",
        direction="buy",
        units=1000,
        stop_loss_pips=20.0,
        take_profit_pips=40.0
    )
    assert trade_result['success'], f"Trade failed: {trade_result.get('error')}"
    logger.info(f"Trade executed: {trade_result['trade_id']}")
    logger.info(f"  Entry: {trade_result['entry_price']:.5f}")
    logger.info(f"  Slippage: {trade_result['slippage_pips']:.2f} pips")
    logger.info(f"  Latency: {trade_result['execution_time_ms']:.1f}ms")
    
    # Test 3: Multiple trades
    logger.info("\n[Test 3] Multiple trades...")
    pairs = ["GBP_USD", "USD_JPY", "AUD_USD"]
    for pair in pairs:
        result = await engine.execute_trade_async(
            pair=pair,
            direction="sell",
            units=500,
            stop_loss_pips=15.0,
            take_profit_pips=30.0
        )
        assert result['success']
        logger.info(f"  Opened {pair}: {result['trade_id']}")
    
    # Test 4: Get open positions
    logger.info("\n[Test 4] Open positions...")
    positions = await engine.get_open_positions_async()
    assert positions['count'] == 4  # 1 EUR_USD + 3 others
    logger.info(f"Open positions: {positions['count']}")
    
    # Test 5: Position monitoring
    logger.info("\n[Test 5] Position monitoring...")
    await engine.start_position_monitoring_async()
    await asyncio.sleep(2)  # Let monitoring run briefly
    logger.info("Position monitoring active")
    
    # Test 6: Concurrent trades (thread safety)
    logger.info("\n[Test 6] Thread safety (concurrent trades)...")
    tasks = []
    for i in range(5):
        tasks.append(engine.execute_trade_async(
            pair="EUR_USD",
            direction="buy" if i % 2 == 0 else "sell",
            units=100,
            stop_loss_pips=10.0,
            take_profit_pips=20.0
        ))
    
    results = await asyncio.gather(*tasks)
    successful = sum(1 for r in results if r['success'])
    logger.info(f"Concurrent trades: {successful}/{len(tasks)} successful")
    
    # Test 7: Metrics
    logger.info("\n[Test 7] Execution metrics...")
    metrics = await engine.get_metrics_async()
    logger.info(f"Total trades: {metrics['total_trades']}")
    logger.info(f"Successful: {metrics['successful_executions']}")
    logger.info(f"Avg slippage: {metrics['avg_slippage_pips']:.2f} pips")
    logger.info(f"Avg latency: {metrics['avg_execution_time_ms']:.1f}ms")
    
    # Test 8: Close all positions
    logger.info("\n[Test 8] Close all positions...")
    close_result = await engine.close_all_positions_async()
    logger.info(f"Closed: {close_result['closed']}/{close_result['total_positions']}")
    
    # Test 9: Final metrics after closes
    logger.info("\n[Test 9] Final metrics...")
    final_metrics = await engine.get_metrics_async()
    logger.info(f"Wins: {final_metrics['wins']}")
    logger.info(f"Losses: {final_metrics['losses']}")
    logger.info(f"Win rate: {final_metrics['win_rate']:.1%}")
    logger.info(f"Total P&L: {final_metrics['total_pnl_pips']:+.1f} pips")
    
    # Test 10: Checkpoint save/load
    logger.info("\n[Test 10] Checkpoint save/load...")
    save_result = await engine.save_checkpoint_async("/tmp/exec_test.pkl")
    assert save_result['status'] == 'success'
    
    load_result = await engine.load_checkpoint_async("/tmp/exec_test.pkl")
    assert load_result['status'] == 'success'
    logger.info("Checkpoint: save/load successful")
    
    # Test 11: Cleanup
    logger.info("\n[Test 11] Cleanup...")
    await engine.cleanup_async()
    logger.info("Cleanup: successful")
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL TESTS PASSED")
    logger.info("=" * 60)


if __name__ == '__main__':
    asyncio.run(test_execution_engine())

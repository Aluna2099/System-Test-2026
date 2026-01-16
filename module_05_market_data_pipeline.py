#!/usr/bin/env python3
"""
============================================================================
MODULE 5: MARKET DATA PIPELINE - PRODUCTION FIXED VERSION
============================================================================
Version: 3.0.0 - PRODUCTION READY (NO SIMULATION)
Author: MIT PhD-Level AI Engineering Team

CRITICAL: This version has been FIXED to:
1. Actually fetch from OANDA API (no more placeholders)
2. BLOCK all simulation code (raises errors instead)
3. Integrate with Module 21 (Data Integrity Guardian)
4. Validate all data before use

CHANGES FROM v2.0.0:
- REMOVED: All _simulate_* methods
- ADDED: Real OANDA REST API implementation
- ADDED: aiohttp-based async HTTP client
- ADDED: Data Integrity Guardian integration
- ADDED: Strict mode that refuses to start without valid API credentials
- FIXED: _fetch_current_prices_async now actually fetches
- FIXED: _fetch_candles_async now actually fetches

REQUIREMENTS:
- Valid OANDA API credentials (account_id + api_token)
- Internet connection to OANDA servers
- aiohttp library installed

============================================================================
"""

import asyncio
import logging
import time
import traceback
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Deque
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

# External dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# EXCEPTIONS
# ============================================================================

class MarketDataError(Exception):
    """Base exception for market data errors"""
    pass


class OandaAPIError(MarketDataError):
    """OANDA API specific error"""
    pass


class NoCredentialsError(MarketDataError):
    """Raised when OANDA credentials are missing"""
    pass


class SimulationBlockedError(MarketDataError):
    """Raised when simulation is attempted (BLOCKED IN PRODUCTION)"""
    pass


class DataValidationError(MarketDataError):
    """Raised when data fails validation"""
    pass


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Candle:
    """OHLCV candle data"""
    timestamp: float
    pair: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    # OANDA-specific fields
    complete: bool = True
    bid_open: Optional[float] = None
    bid_close: Optional[float] = None
    ask_open: Optional[float] = None
    ask_close: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'pair': self.pair,
            'timeframe': self.timeframe,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'complete': self.complete
        }
    
    @classmethod
    def from_oanda(cls, data: Dict, pair: str, timeframe: str) -> 'Candle':
        """Create Candle from OANDA API response"""
        # Parse timestamp
        ts_str = data.get('time', '')
        if ts_str:
            dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            timestamp = dt.timestamp()
        else:
            timestamp = time.time()
        
        # Get mid prices (or bid/ask if mid not available)
        mid = data.get('mid', {})
        bid = data.get('bid', {})
        ask = data.get('ask', {})
        
        return cls(
            timestamp=timestamp,
            pair=pair,
            timeframe=timeframe,
            open=float(mid.get('o', bid.get('o', 0))),
            high=float(mid.get('h', bid.get('h', 0))),
            low=float(mid.get('l', bid.get('l', 0))),
            close=float(mid.get('c', bid.get('c', 0))),
            volume=int(data.get('volume', 0)),
            complete=data.get('complete', True),
            bid_open=float(bid.get('o', 0)) if bid else None,
            bid_close=float(bid.get('c', 0)) if bid else None,
            ask_open=float(ask.get('o', 0)) if ask else None,
            ask_close=float(ask.get('c', 0)) if ask else None
        )


@dataclass
class PriceQuote:
    """Real-time price quote"""
    pair: str
    bid: float
    ask: float
    timestamp: float
    tradeable: bool = True
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2
    
    @classmethod
    def from_oanda(cls, data: Dict) -> 'PriceQuote':
        """Create PriceQuote from OANDA API response"""
        bids = data.get('bids', [{}])
        asks = data.get('asks', [{}])
        
        return cls(
            pair=data.get('instrument', ''),
            bid=float(bids[0].get('price', 0)) if bids else 0,
            ask=float(asks[0].get('price', 0)) if asks else 0,
            timestamp=time.time(),
            tradeable=data.get('tradeable', True)
        )


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MarketDataConfig:
    """
    Configuration for Production Market Data Pipeline
    
    CRITICAL: Requires valid OANDA credentials. Will NOT fall back to simulation.
    """
    # OANDA API credentials (REQUIRED)
    oanda_account_id: str = ""
    oanda_api_token: str = ""
    oanda_environment: str = "practice"  # "practice" or "live"
    
    # Currency pairs
    pairs: List[str] = field(default_factory=lambda: [
        "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
        "USD_CAD", "NZD_USD", "EUR_GBP", "EUR_JPY"
    ])
    
    # Timeframes
    primary_timeframe: str = "M5"
    secondary_timeframes: List[str] = field(default_factory=lambda: [
        "M1", "M15", "H1", "H4", "D"
    ])
    
    # Buffer sizes per timeframe
    timeframe_buffer_sizes: Dict[str, int] = field(default_factory=lambda: {
        "M1": 150,
        "M5": 500,
        "M15": 200,
        "H1": 168,
        "H4": 120,
        "D": 100
    })
    
    # Feature engineering
    feature_dimension: int = 50
    max_candles_per_pair: int = 500
    
    # Update intervals
    price_update_interval_sec: float = 1.0
    candle_check_interval_sec: float = 5.0
    
    # Rate limiting (OANDA allows ~120 req/sec)
    max_requests_per_second: int = 100
    
    # HTTP settings
    request_timeout_sec: float = 30.0
    max_retries: int = 3
    retry_delay_sec: float = 1.0
    
    # Data storage
    data_dir: str = "data/market_data"
    save_interval_sec: float = 300.0
    
    # STRICT MODE: Block all simulation (PRODUCTION DEFAULT)
    strict_mode: bool = True
    require_valid_credentials: bool = True
    
    def get_buffer_size(self, timeframe: str) -> int:
        return self.timeframe_buffer_sizes.get(timeframe, self.max_candles_per_pair)
    
    @property
    def all_timeframes(self) -> List[str]:
        return [self.primary_timeframe] + self.secondary_timeframes
    
    @property
    def api_url(self) -> str:
        if self.oanda_environment == "live":
            return "https://api-fxtrade.oanda.com/v3"
        return "https://api-fxpractice.oanda.com/v3"
    
    @property
    def stream_url(self) -> str:
        if self.oanda_environment == "live":
            return "https://stream-fxtrade.oanda.com/v3"
        return "https://stream-fxpractice.oanda.com/v3"
    
    def __post_init__(self):
        """Validate configuration"""
        # CRITICAL: Require credentials in production
        if self.require_valid_credentials:
            if not self.oanda_account_id:
                raise NoCredentialsError(
                    "OANDA account_id is REQUIRED. "
                    "This system does NOT support simulation mode."
                )
            if not self.oanda_api_token:
                raise NoCredentialsError(
                    "OANDA api_token is REQUIRED. "
                    "This system does NOT support simulation mode."
                )
        
        if self.oanda_environment not in ["practice", "live"]:
            raise ValueError("oanda_environment must be 'practice' or 'live'")
        
        if not self.pairs:
            raise ValueError("pairs list cannot be empty")
        
        valid_timeframes = ["S5", "S10", "S15", "S30", "M1", "M2", "M4", "M5",
                          "M10", "M15", "M30", "H1", "H2", "H3", "H4", "H6",
                          "H8", "H12", "D", "W", "M"]
        if self.primary_timeframe not in valid_timeframes:
            raise ValueError(f"Invalid primary_timeframe: {self.primary_timeframe}")


# ============================================================================
# FEATURE ENGINEERING (Unchanged from v2.0.0)
# ============================================================================

class FeatureEngineer:
    """
    Calculates 50+ technical features from candle data.
    
    Features include:
    - Price-based (returns, log returns, momentum)
    - Volatility (ATR, Bollinger width, historical vol)
    - Trend indicators (SMA, EMA, MACD)
    - Oscillators (RSI, Stochastic)
    - Volume features
    """
    
    @staticmethod
    def calculate_features(candles: List[Candle]) -> np.ndarray:
        """
        Calculate feature vector from candle data.
        
        Args:
            candles: List of recent candles (min 100 for full indicators)
            
        Returns:
            np.ndarray of shape (50,) containing normalized features
        """
        if len(candles) < 20:
            return np.zeros(50)
        
        # Extract OHLCV arrays
        closes = np.array([c.close for c in candles], dtype=np.float64)
        highs = np.array([c.high for c in candles], dtype=np.float64)
        lows = np.array([c.low for c in candles], dtype=np.float64)
        opens = np.array([c.open for c in candles], dtype=np.float64)
        volumes = np.array([c.volume for c in candles], dtype=np.float64)
        
        features = []
        
        # 1-5: Returns (1, 5, 10, 20, 50 periods)
        for period in [1, 5, 10, 20, min(50, len(closes) - 1)]:
            if len(closes) > period:
                ret = (closes[-1] - closes[-period - 1]) / max(closes[-period - 1], 1e-8)
                features.append(np.clip(ret, -0.1, 0.1))
            else:
                features.append(0.0)
        
        # 6-10: Log returns
        for period in [1, 5, 10, 20, min(50, len(closes) - 1)]:
            if len(closes) > period and closes[-period - 1] > 0:
                log_ret = np.log(closes[-1] / max(closes[-period - 1], 1e-8))
                features.append(np.clip(log_ret, -0.1, 0.1))
            else:
                features.append(0.0)
        
        # 11-15: Momentum
        for period in [5, 10, 14, 20, 50]:
            if len(closes) > period:
                momentum = closes[-1] - closes[-period]
                features.append(np.clip(momentum / max(abs(closes[-1]), 1e-8), -1, 1))
            else:
                features.append(0.0)
        
        # 16-20: Volatility (rolling std)
        for period in [5, 10, 20, 50, 100]:
            if len(closes) >= period:
                vol = np.std(closes[-period:]) / max(np.mean(closes[-period:]), 1e-8)
                features.append(np.clip(vol, 0, 0.5))
            else:
                features.append(0.0)
        
        # 21-25: ATR (Average True Range)
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        for period in [5, 10, 14, 20, 50]:
            if len(tr) >= period:
                atr = np.mean(tr[-period:]) / max(closes[-1], 1e-8)
                features.append(np.clip(atr, 0, 0.1))
            else:
                features.append(0.0)
        
        # 26-30: SMA ratios
        for period in [5, 10, 20, 50, 100]:
            if len(closes) >= period:
                sma = np.mean(closes[-period:])
                ratio = (closes[-1] - sma) / max(sma, 1e-8)
                features.append(np.clip(ratio, -0.1, 0.1))
            else:
                features.append(0.0)
        
        # 31-35: EMA ratios
        for period in [5, 10, 20, 50, 100]:
            if len(closes) >= period:
                alpha = 2 / (period + 1)
                ema = closes[-period]
                for price in closes[-period + 1:]:
                    ema = alpha * price + (1 - alpha) * ema
                ratio = (closes[-1] - ema) / max(ema, 1e-8)
                features.append(np.clip(ratio, -0.1, 0.1))
            else:
                features.append(0.0)
        
        # 36-38: RSI (14, 28, 50)
        for period in [14, 28, 50]:
            if len(closes) > period:
                deltas = np.diff(closes[-(period + 1):])
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100 if avg_gain > 0 else 50
                features.append(rsi / 100)  # Normalize to 0-1
            else:
                features.append(0.5)
        
        # 39-41: Bollinger Band position
        for period in [10, 20, 50]:
            if len(closes) >= period:
                sma = np.mean(closes[-period:])
                std = np.std(closes[-period:])
                if std > 0:
                    bb_pos = (closes[-1] - sma) / (2 * std)
                    features.append(np.clip(bb_pos, -1, 1))
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
        
        # 42-44: Stochastic %K
        for period in [5, 14, 28]:
            if len(closes) >= period:
                low_min = np.min(lows[-period:])
                high_max = np.max(highs[-period:])
                if high_max > low_min:
                    k = (closes[-1] - low_min) / (high_max - low_min)
                    features.append(k)
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
        
        # 45-47: Volume features
        if len(volumes) >= 20 and np.mean(volumes) > 0:
            vol_ratio = volumes[-1] / max(np.mean(volumes[-20:]), 1)
            features.append(np.clip(vol_ratio, 0, 5) / 5)
            
            vol_sma5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
            vol_sma20 = np.mean(volumes[-20:])
            vol_trend = vol_sma5 / max(vol_sma20, 1)
            features.append(np.clip(vol_trend, 0, 3) / 3)
            
            vol_std = np.std(volumes[-20:])
            vol_zscore = (volumes[-1] - np.mean(volumes[-20:])) / max(vol_std, 1)
            features.append(np.clip(vol_zscore, -3, 3) / 3)
        else:
            features.extend([0.5, 0.5, 0.0])
        
        # 48-50: Price range features
        if len(closes) >= 20:
            # Daily range
            daily_range = (highs[-1] - lows[-1]) / max(closes[-1], 1e-8)
            features.append(np.clip(daily_range, 0, 0.1) * 10)
            
            # Range expansion
            avg_range = np.mean(highs[-20:] - lows[-20:])
            curr_range = highs[-1] - lows[-1]
            range_exp = curr_range / max(avg_range, 1e-8)
            features.append(np.clip(range_exp, 0, 3) / 3)
            
            # Close position in range
            if highs[-1] > lows[-1]:
                close_pos = (closes[-1] - lows[-1]) / (highs[-1] - lows[-1])
                features.append(close_pos)
            else:
                features.append(0.5)
        else:
            features.extend([0.0, 0.5, 0.5])
        
        # Ensure exactly 50 features
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50], dtype=np.float32)


# ============================================================================
# OANDA HTTP CLIENT
# ============================================================================

class OandaHTTPClient:
    """
    Async HTTP client for OANDA REST API.
    
    Handles:
    - Authentication
    - Rate limiting
    - Retries
    - Error handling
    """
    
    def __init__(self, config: MarketDataConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_lock = asyncio.Lock()
        self._request_times: Deque[float] = deque(maxlen=200)
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_sec)
            self._session = aiohttp.ClientSession(timeout=timeout)
    
    async def _check_rate_limit(self):
        """Enforce rate limiting"""
        async with self._rate_limit_lock:
            now = time.time()
            
            # Remove old requests
            while self._request_times and now - self._request_times[0] > 1.0:
                self._request_times.popleft()
            
            # Wait if at limit
            if len(self._request_times) >= self.config.max_requests_per_second:
                wait_time = 1.0 - (now - self._request_times[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            self._request_times.append(time.time())
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        return {
            "Authorization": f"Bearer {self.config.oanda_api_token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339"
        }
    
    async def get_async(
        self,
        endpoint: str,
        params: Dict = None
    ) -> Tuple[int, Dict]:
        """
        Make authenticated GET request to OANDA API.
        
        Returns:
            Tuple of (status_code, response_data)
            
        Raises:
            OandaAPIError: On API errors
        """
        await self._ensure_session()
        await self._check_rate_limit()
        
        url = f"{self.config.api_url}{endpoint}"
        
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                async with self._session.get(
                    url,
                    headers=self._get_headers(),
                    params=params
                ) as response:
                    status = response.status
                    
                    try:
                        data = await response.json()
                    except Exception:
                        data = {'error': await response.text()}
                    
                    if status == 200:
                        return status, data
                    elif status == 429:  # Rate limited
                        wait_time = int(response.headers.get('Retry-After', 5))
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    elif status >= 400:
                        error_msg = data.get('errorMessage', str(data))
                        raise OandaAPIError(f"OANDA API error {status}: {error_msg}")
                    
            except aiohttp.ClientError as e:
                last_error = e
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                await asyncio.sleep(self.config.retry_delay_sec * (attempt + 1))
        
        raise OandaAPIError(f"Failed after {self.config.max_retries} retries: {last_error}")
    
    async def close_async(self):
        """Close the HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()


# ============================================================================
# MARKET DATA PIPELINE - PRODUCTION VERSION
# ============================================================================

class MarketDataPipeline:
    """
    Production-ready market data pipeline.
    
    CRITICAL: This version ONLY uses real OANDA API data.
    Simulation is BLOCKED and will raise SimulationBlockedError.
    
    Features:
    - Real-time price streaming
    - Multi-timeframe candle data
    - 50+ technical features
    - Thread-safe async operations
    - Data validation and integrity checks
    """
    
    def __init__(
        self,
        config: MarketDataConfig = None,
        data_guardian = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
            data_guardian: Optional Data Integrity Guardian (Module 21)
        """
        self.config = config or MarketDataConfig()
        self.data_guardian = data_guardian
        
        # HTTP client
        self._http_client = OandaHTTPClient(self.config)
        
        # Data storage
        self.current_prices: Dict[str, PriceQuote] = {}
        self.candle_buffers: Dict[str, Dict[str, Deque[Candle]]] = {}
        self.feature_cache: Dict[str, np.ndarray] = {}
        
        # State
        self._is_initialized = False
        self._is_running = False
        
        # Locks for thread safety
        self._price_lock = asyncio.Lock()
        self._candle_lock = asyncio.Lock()
        self._feature_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        
        # Background tasks
        self._price_task: Optional[asyncio.Task] = None
        self._candle_task: Optional[asyncio.Task] = None
        self._feature_task: Optional[asyncio.Task] = None
        self._save_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = {
            'price_updates': 0,
            'candle_updates': 0,
            'api_calls': 0,
            'api_errors': 0,
            'last_price_update': 0.0,
            'last_candle_update': 0.0
        }
        self._stats_lock = asyncio.Lock()
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    async def initialize_async(self) -> Dict[str, Any]:
        """
        Initialize the pipeline.
        
        CRITICAL: Will raise NoCredentialsError if OANDA credentials missing.
        """
        async with self._state_lock:
            if self._is_initialized:
                return {'status': 'already_initialized'}
        
        logger.info("Initializing Market Data Pipeline (PRODUCTION MODE)...")
        
        # Verify credentials
        if not self.config.oanda_api_token or not self.config.oanda_account_id:
            raise NoCredentialsError(
                "OANDA credentials are REQUIRED. "
                "This system does NOT support simulation mode. "
                "Please configure oanda_account_id and oanda_api_token."
            )
        
        # Test API connection
        logger.info("  Testing OANDA API connection...")
        await self._verify_api_connection_async()
        
        # Initialize candle buffers
        async with self._candle_lock:
            for pair in self.config.pairs:
                self.candle_buffers[pair] = {}
                for tf in self.config.all_timeframes:
                    buffer_size = self.config.get_buffer_size(tf)
                    self.candle_buffers[pair][tf] = deque(maxlen=buffer_size)
        
        # Load historical data for initial buffer
        logger.info("  Loading historical candle data...")
        await self._load_initial_candles_async()
        
        # Create data directory
        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
        
        async with self._state_lock:
            self._is_initialized = True
        
        logger.info("✅ Market Data Pipeline initialized (REAL DATA ONLY)")
        
        return {
            'status': 'success',
            'pairs': len(self.config.pairs),
            'timeframes': len(self.config.all_timeframes),
            'api_connected': True,
            'simulation_mode': False  # ALWAYS False in production
        }
    
    async def _verify_api_connection_async(self):
        """Verify OANDA API is accessible"""
        try:
            status, data = await self._http_client.get_async(
                f"/accounts/{self.config.oanda_account_id}/summary"
            )
            
            if status != 200:
                raise OandaAPIError(f"API verification failed: {data}")
            
            account = data.get('account', {})
            logger.info(f"  ✅ Connected to OANDA ({self.config.oanda_environment})")
            logger.info(f"     Account: {account.get('id', 'Unknown')}")
            logger.info(f"     Balance: {account.get('balance', 'Unknown')}")
            
        except Exception as e:
            raise OandaAPIError(f"Failed to connect to OANDA API: {e}")
    
    async def _load_initial_candles_async(self):
        """Load initial historical candles for all pairs/timeframes"""
        for pair in self.config.pairs:
            for tf in self.config.all_timeframes:
                try:
                    candles = await self._fetch_candles_from_api_async(
                        pair=pair,
                        timeframe=tf,
                        count=self.config.get_buffer_size(tf)
                    )
                    
                    async with self._candle_lock:
                        for candle in candles:
                            self.candle_buffers[pair][tf].append(candle)
                    
                    logger.info(f"    Loaded {len(candles)} {tf} candles for {pair}")
                    
                except Exception as e:
                    logger.error(f"    Failed to load {tf} candles for {pair}: {e}")
    
    # ========================================================================
    # START/STOP
    # ========================================================================
    
    async def start_async(self):
        """Start background data collection"""
        async with self._state_lock:
            if not self._is_initialized:
                raise MarketDataError("Pipeline not initialized")
            if self._is_running:
                return
            self._is_running = True
        
        logger.info("Starting market data collection...")
        
        # Start background tasks
        self._price_task = asyncio.create_task(
            self._price_update_loop_async()
        )
        self._candle_task = asyncio.create_task(
            self._candle_update_loop_async()
        )
        self._feature_task = asyncio.create_task(
            self._feature_update_loop_async()
        )
        self._save_task = asyncio.create_task(
            self._auto_save_loop_async()
        )
        
        logger.info("✅ Market data collection started")
    
    async def stop_async(self):
        """Stop background data collection"""
        async with self._state_lock:
            self._is_running = False
        
        # Cancel tasks
        for task in [self._price_task, self._candle_task, 
                     self._feature_task, self._save_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close HTTP client
        await self._http_client.close_async()
        
        logger.info("Market data collection stopped")
    
    # ========================================================================
    # REAL OANDA API FETCHING (NOT SIMULATION!)
    # ========================================================================
    
    async def _fetch_candles_from_api_async(
        self,
        pair: str,
        timeframe: str,
        count: int = 100
    ) -> List[Candle]:
        """
        Fetch candles from REAL OANDA API.
        
        This is the PRODUCTION implementation that actually calls OANDA.
        """
        endpoint = f"/instruments/{pair}/candles"
        
        params = {
            "granularity": timeframe,
            "count": min(count, 5000),  # OANDA max is 5000
            "price": "MBA"  # Mid, Bid, Ask
        }
        
        try:
            status, data = await self._http_client.get_async(endpoint, params)
            
            async with self._stats_lock:
                self._stats['api_calls'] += 1
            
            if status != 200:
                raise OandaAPIError(f"Candle fetch failed: {data}")
            
            # Validate data with guardian if available
            if self.data_guardian:
                data, report = await self.data_guardian.validate_async(
                    data,
                    source="oanda",
                    provenance=None
                )
                if report.overall_result.value == "failed":
                    raise DataValidationError(f"Data validation failed: {report.issues}")
            
            # Parse candles
            raw_candles = data.get('candles', [])
            candles = []
            
            for raw in raw_candles:
                if raw.get('complete', True):  # Only complete candles
                    candle = Candle.from_oanda(raw, pair, timeframe)
                    candles.append(candle)
            
            return candles
            
        except Exception as e:
            async with self._stats_lock:
                self._stats['api_errors'] += 1
            raise
    
    async def _fetch_prices_from_api_async(self) -> Dict[str, PriceQuote]:
        """
        Fetch current prices from REAL OANDA API.
        
        This is the PRODUCTION implementation that actually calls OANDA.
        """
        pairs_str = ",".join(self.config.pairs)
        endpoint = f"/accounts/{self.config.oanda_account_id}/pricing"
        
        params = {"instruments": pairs_str}
        
        try:
            status, data = await self._http_client.get_async(endpoint, params)
            
            async with self._stats_lock:
                self._stats['api_calls'] += 1
            
            if status != 200:
                raise OandaAPIError(f"Price fetch failed: {data}")
            
            # Parse prices
            prices = {}
            for price_data in data.get('prices', []):
                quote = PriceQuote.from_oanda(price_data)
                if quote.pair:
                    prices[quote.pair] = quote
            
            return prices
            
        except Exception as e:
            async with self._stats_lock:
                self._stats['api_errors'] += 1
            raise
    
    # ========================================================================
    # SIMULATION BLOCKED
    # ========================================================================
    
    async def _simulate_prices_async(self):
        """BLOCKED: Simulation is not allowed in production"""
        raise SimulationBlockedError(
            "SIMULATION IS BLOCKED IN PRODUCTION! "
            "This system only uses real OANDA API data. "
            "Please ensure your OANDA credentials are configured."
        )
    
    async def _simulate_candles_async(self):
        """BLOCKED: Simulation is not allowed in production"""
        raise SimulationBlockedError(
            "SIMULATION IS BLOCKED IN PRODUCTION! "
            "This system only uses real OANDA API data. "
            "Please ensure your OANDA credentials are configured."
        )
    
    # ========================================================================
    # BACKGROUND UPDATE LOOPS
    # ========================================================================
    
    async def _price_update_loop_async(self):
        """Background loop for price updates"""
        while self._is_running:
            try:
                # Fetch REAL prices from OANDA
                prices = await self._fetch_prices_from_api_async()
                
                # Update storage
                async with self._price_lock:
                    for pair, quote in prices.items():
                        self.current_prices[pair] = quote
                
                async with self._stats_lock:
                    self._stats['price_updates'] += 1
                    self._stats['last_price_update'] = time.time()
                
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Price update error: {e}")
                await asyncio.sleep(5)  # Back off on error
            
            await asyncio.sleep(self.config.price_update_interval_sec)
    
    async def _candle_update_loop_async(self):
        """Background loop for candle updates"""
        while self._is_running:
            try:
                for pair in self.config.pairs:
                    for tf in self.config.all_timeframes:
                        # Fetch latest candles
                        candles = await self._fetch_candles_from_api_async(
                            pair=pair,
                            timeframe=tf,
                            count=5  # Just get recent ones
                        )
                        
                        # Update buffer
                        async with self._candle_lock:
                            buffer = self.candle_buffers[pair][tf]
                            for candle in candles:
                                # Avoid duplicates
                                if not buffer or candle.timestamp > buffer[-1].timestamp:
                                    buffer.append(candle)
                
                async with self._stats_lock:
                    self._stats['candle_updates'] += 1
                    self._stats['last_candle_update'] = time.time()
                
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Candle update error: {e}")
                await asyncio.sleep(10)  # Back off on error
            
            await asyncio.sleep(self.config.candle_check_interval_sec)
    
    async def _feature_update_loop_async(self):
        """Background loop for feature calculation"""
        while self._is_running:
            try:
                for pair in self.config.pairs:
                    async with self._candle_lock:
                        if pair not in self.candle_buffers:
                            continue
                        tf = self.config.primary_timeframe
                        if tf not in self.candle_buffers[pair]:
                            continue
                        candles = list(self.candle_buffers[pair][tf])
                    
                    if len(candles) >= 20:
                        # Calculate features (CPU-bound)
                        features = await asyncio.to_thread(
                            FeatureEngineer.calculate_features,
                            candles
                        )
                        
                        async with self._feature_lock:
                            self.feature_cache[pair] = features
                
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Feature update error: {e}")
            
            await asyncio.sleep(1.0)  # Update features every second
    
    async def _auto_save_loop_async(self):
        """Background loop for auto-saving data"""
        while self._is_running:
            try:
                await asyncio.sleep(self.config.save_interval_sec)
                await self.save_data_async()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Auto-save error: {e}")
    
    # ========================================================================
    # DATA ACCESS METHODS
    # ========================================================================
    
    async def get_current_price_async(self, pair: str) -> Optional[PriceQuote]:
        """Get current price for a pair"""
        async with self._price_lock:
            return self.current_prices.get(pair)
    
    async def get_all_prices_async(self) -> Dict[str, PriceQuote]:
        """Get all current prices"""
        async with self._price_lock:
            return dict(self.current_prices)
    
    async def get_candles_async(
        self,
        pair: str,
        timeframe: str = None,
        count: int = None
    ) -> List[Candle]:
        """Get candles for a pair/timeframe"""
        timeframe = timeframe or self.config.primary_timeframe
        
        async with self._candle_lock:
            if pair not in self.candle_buffers:
                return []
            if timeframe not in self.candle_buffers[pair]:
                return []
            
            candles = list(self.candle_buffers[pair][timeframe])
        
        if count:
            candles = candles[-count:]
        
        return candles
    
    async def get_features_async(self, pair: str) -> Optional[np.ndarray]:
        """Get calculated features for a pair"""
        async with self._feature_lock:
            return self.feature_cache.get(pair)
    
    async def get_all_features_async(self) -> Dict[str, np.ndarray]:
        """Get features for all pairs"""
        async with self._feature_lock:
            return dict(self.feature_cache)
    
    async def get_multi_timeframe_candles_async(
        self,
        pair: str
    ) -> Dict[str, List[Candle]]:
        """Get candles for all timeframes"""
        async with self._candle_lock:
            if pair not in self.candle_buffers:
                return {}
            
            return {
                tf: list(candles)
                for tf, candles in self.candle_buffers[pair].items()
            }
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    async def save_data_async(self):
        """Save current data to disk"""
        data_path = Path(self.config.data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Save candles
        async with self._candle_lock:
            candle_data = {}
            for pair, timeframes in self.candle_buffers.items():
                candle_data[pair] = {}
                for tf, candles in timeframes.items():
                    candle_data[pair][tf] = [c.to_dict() for c in candles]
        
        candle_file = data_path / "candles.json"
        await asyncio.to_thread(
            lambda: candle_file.write_text(json.dumps(candle_data, indent=2))
        )
        
        logger.debug(f"Saved candle data to {candle_file}")
    
    async def load_data_async(self):
        """Load data from disk"""
        candle_file = Path(self.config.data_dir) / "candles.json"
        
        if not candle_file.exists():
            return
        
        try:
            content = await asyncio.to_thread(candle_file.read_text)
            candle_data = json.loads(content)
            
            async with self._candle_lock:
                for pair, timeframes in candle_data.items():
                    if pair not in self.candle_buffers:
                        continue
                    for tf, candles in timeframes.items():
                        if tf not in self.candle_buffers[pair]:
                            continue
                        for c in candles:
                            candle = Candle(
                                timestamp=c['timestamp'],
                                pair=c['pair'],
                                timeframe=c['timeframe'],
                                open=c['open'],
                                high=c['high'],
                                low=c['low'],
                                close=c['close'],
                                volume=c['volume'],
                                complete=c.get('complete', True)
                            )
                            self.candle_buffers[pair][tf].append(candle)
            
            logger.info(f"Loaded candle data from {candle_file}")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    async def get_stats_async(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        async with self._stats_lock:
            stats = dict(self._stats)
        
        async with self._candle_lock:
            candle_counts = {}
            for pair, tfs in self.candle_buffers.items():
                candle_counts[pair] = {tf: len(candles) for tf, candles in tfs.items()}
        
        stats['candle_counts'] = candle_counts
        stats['is_running'] = self._is_running
        stats['simulation_mode'] = False  # ALWAYS False
        
        return stats
    
    # ========================================================================
    # CLEANUP
    # ========================================================================
    
    async def cleanup_async(self):
        """Cleanup resources"""
        await self.stop_async()
        logger.info("Market Data Pipeline cleaned up")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_market_data_pipeline(
    account_id: str,
    api_token: str,
    environment: str = "practice",
    pairs: List[str] = None,
    data_guardian = None
) -> MarketDataPipeline:
    """
    Factory function to create configured pipeline.
    
    Args:
        account_id: OANDA account ID (REQUIRED)
        api_token: OANDA API token (REQUIRED)
        environment: "practice" or "live"
        pairs: List of currency pairs
        data_guardian: Optional Data Integrity Guardian (Module 21)
        
    Returns:
        Configured MarketDataPipeline
        
    Raises:
        NoCredentialsError: If credentials missing
    """
    if not account_id or not api_token:
        raise NoCredentialsError(
            "OANDA credentials are REQUIRED. "
            "This system does NOT support simulation."
        )
    
    config = MarketDataConfig(
        oanda_account_id=account_id,
        oanda_api_token=api_token,
        oanda_environment=environment,
        pairs=pairs or [
            "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
            "USD_CAD", "NZD_USD", "EUR_GBP", "EUR_JPY"
        ],
        strict_mode=True,
        require_valid_credentials=True
    )
    
    return MarketDataPipeline(config=config, data_guardian=data_guardian)


# ============================================================================
# STANDALONE TEST
# ============================================================================

async def _test_pipeline():
    """Test the pipeline (requires real credentials)"""
    import os
    
    account_id = os.environ.get('OANDA_ACCOUNT_ID', '')
    api_token = os.environ.get('OANDA_API_TOKEN', '')
    
    if not account_id or not api_token:
        print("⚠️  Set OANDA_ACCOUNT_ID and OANDA_API_TOKEN environment variables")
        print("   This system does NOT support simulation mode.")
        return
    
    print("Testing Market Data Pipeline (PRODUCTION MODE)...")
    
    pipeline = create_market_data_pipeline(
        account_id=account_id,
        api_token=api_token,
        environment="practice"
    )
    
    # Initialize
    result = await pipeline.initialize_async()
    print(f"Initialize: {result}")
    
    # Start
    await pipeline.start_async()
    
    # Wait for some data
    await asyncio.sleep(10)
    
    # Get stats
    stats = await pipeline.get_stats_async()
    print(f"Stats: {stats}")
    
    # Get prices
    prices = await pipeline.get_all_prices_async()
    print(f"Prices: {list(prices.keys())}")
    
    # Stop
    await pipeline.stop_async()
    
    print("✅ Pipeline test complete!")


if __name__ == "__main__":
    asyncio.run(_test_pipeline())

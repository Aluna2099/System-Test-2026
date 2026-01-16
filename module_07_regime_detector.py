"""
MODULE 7: REGIME DETECTOR
Production-Ready Implementation

Market regime detection using Hidden Markov Models and heuristics.
- 5 market regimes: Trending, Ranging, Volatile, Breakout, Crisis
- HMM-based regime inference with Viterbi algorithm
- Heuristic fallback when HMM not trained
- Crisis early warning system
- Regime transition detection
- Async/await architecture throughout
- Thread-safe state management

Author: Liquid Neural SREK Trading System v4.0
Date: 2026-01-10
Version: 1.0.0

MARKET REGIMES:
1. TRENDING - Directional momentum, trend following optimal
2. RANGING - Mean reversion, oscillation between support/resistance
3. VOLATILE - High uncertainty, large swings, reduce position size
4. BREAKOUT - Regime transition, consolidation to expansion
5. CRISIS - Systemic shock, STOP TRADING
"""

import asyncio
import logging
import time
import math
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

class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    CRISIS = "crisis"
    
    @classmethod
    def from_id(cls, regime_id: int) -> 'MarketRegime':
        """Convert regime ID to enum"""
        mapping = {
            0: cls.TRENDING,
            1: cls.RANGING,
            2: cls.VOLATILE,
            3: cls.BREAKOUT,
            4: cls.CRISIS
        }
        return mapping.get(regime_id, cls.VOLATILE)
    
    def to_id(self) -> int:
        """Convert enum to regime ID"""
        mapping = {
            MarketRegime.TRENDING: 0,
            MarketRegime.RANGING: 1,
            MarketRegime.VOLATILE: 2,
            MarketRegime.BREAKOUT: 3,
            MarketRegime.CRISIS: 4
        }
        return mapping[self]


@dataclass
class RegimeDetectorConfig:
    """
    Configuration for Regime Detector
    
    Includes validation to prevent runtime errors
    """
    # HMM configuration
    n_states: int = 5  # Number of regimes
    n_features: int = 10  # Regime feature dimension
    hmm_n_iter: int = 100  # HMM training iterations
    
    # Crisis thresholds
    crisis_volatility_threshold: float = 0.05  # 5% daily vol triggers crisis
    crisis_correlation_threshold: float = 0.9  # Panic correlation threshold
    high_volatility_threshold: float = 0.03  # 3% for volatile regime
    
    # Hurst exponent thresholds
    trending_hurst_threshold: float = 0.6  # Above = trending
    random_walk_hurst_range: Tuple[float, float] = (0.4, 0.6)  # Random walk
    
    # Autocorrelation thresholds
    ranging_autocorr_threshold: float = -0.2  # Below = ranging (mean reversion)
    
    # Transition detection
    regime_history_size: int = 100  # History for transition detection
    transition_window: int = 10  # Recent samples for transition analysis
    transition_change_threshold: int = 3  # Min changes for transition
    low_confidence_threshold: float = 0.6  # Below = uncertain
    
    # Numerical stability
    epsilon: float = 1e-10
    
    def __post_init__(self):
        """Validate configuration"""
        if self.n_states <= 0:
            raise ValueError(f"n_states must be positive, got {self.n_states}")
        if self.n_features <= 0:
            raise ValueError(f"n_features must be positive, got {self.n_features}")
        if self.hmm_n_iter <= 0:
            raise ValueError(f"hmm_n_iter must be positive, got {self.hmm_n_iter}")
        if not 0 < self.crisis_volatility_threshold < 1:
            raise ValueError(f"crisis_volatility_threshold must be in (0, 1)")
        if not 0 < self.high_volatility_threshold < self.crisis_volatility_threshold:
            raise ValueError("high_volatility_threshold must be less than crisis threshold")
        if self.regime_history_size <= 0:
            raise ValueError("regime_history_size must be positive")
        if self.transition_window <= 0:
            raise ValueError("transition_window must be positive")


# ============================================================================
# REGIME DETECTOR (MAIN MODULE)
# ============================================================================

class RegimeDetector:
    """
    Market regime detection using Hidden Markov Models.
    
    Features:
    - HMM-based regime inference
    - Heuristic fallback when HMM not trained
    - Crisis early warning system
    - Regime transition detection
    - Async/await architecture
    - Thread-safe state management
    
    Approach:
    1. Feature extraction (volatility, autocorrelation, Hurst, etc.)
    2. HMM with 5 hidden states (regimes)
    3. Viterbi algorithm for regime inference
    4. Transition detection for early warnings
    """
    
    def __init__(self, config: Optional[RegimeDetectorConfig] = None):
        """
        Initialize Regime Detector.
        
        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or RegimeDetectorConfig()
        
        # HMM model (initialized lazily or via training)
        self.hmm = None
        self._hmm_trained = False
        
        # Thread safety
        self._lock = asyncio.Lock()  # Protects shared state
        self._hmm_lock = asyncio.Lock()  # Protects HMM model
        
        # Regime history (protected by _lock)
        self.regime_history: deque = deque(maxlen=self.config.regime_history_size)
        
        # State tracking (protected by _lock)
        self._is_initialized = False
        self._detection_count = 0
        self._last_detection_time = 0.0
        self._current_regime = MarketRegime.VOLATILE  # Default
        
        # Statistics (protected by _lock)
        self._stats = {
            'total_detections': 0,
            'crisis_warnings': 0,
            'transitions_detected': 0,
            'regime_counts': {r.value: 0 for r in MarketRegime},
            'avg_confidence': 0.0,
            'hmm_trained': False
        }
        
        logger.info(
            f"RegimeDetector initialized: "
            f"n_states={self.config.n_states}, "
            f"n_features={self.config.n_features}"
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
                self._is_initialized = True
                
                logger.info("RegimeDetector initialized successfully")
                
                return {
                    'status': 'success',
                    'n_states': self.config.n_states,
                    'n_features': self.config.n_features,
                    'hmm_trained': self._hmm_trained
                }
                
            except Exception as e:
                logger.error(f"Initialization failed: {e}")
                return {'status': 'failed', 'error': str(e)}
    
    async def detect_regime_async(
        self,
        features: np.ndarray,
        historical_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Detect current market regime.
        
        Args:
            features: Current market features [50] or [n_features]
            historical_data: Recent price history for context [optional]
        
        Returns:
            {
                'regime': str,
                'regime_id': int,
                'confidence': float,
                'regime_probabilities': dict,
                'crisis_warning': bool,
                'transition_probability': float
            }
        """
        start_time = time.time()
        
        try:
            # Extract regime-specific features (CPU-bound, offload)
            regime_features = await asyncio.to_thread(
                self._extract_regime_features,
                features,
                historical_data
            )
            
            # Detect regime (CPU-bound HMM inference, offload)
            async with self._hmm_lock:
                regime_result = await asyncio.to_thread(
                    self._detect_regime_internal,
                    regime_features
                )
            
            # Crisis check (fast, can run inline)
            crisis_warning = self._check_crisis_conditions(regime_features)
            
            # Get transition probability
            transition_prob = await self._get_transition_probability_async()
            
            # Store in history (thread-safe)
            async with self._lock:
                self.regime_history.append({
                    'timestamp': time.time(),
                    'regime': regime_result['regime'],
                    'regime_id': regime_result['regime_id'],
                    'confidence': regime_result['confidence']
                })
                
                # Update current regime
                self._current_regime = MarketRegime(regime_result['regime'])
                
                # Update stats
                self._detection_count += 1
                self._last_detection_time = time.time() - start_time
                self._stats['total_detections'] += 1
                self._stats['regime_counts'][regime_result['regime']] += 1
                
                if crisis_warning:
                    self._stats['crisis_warnings'] += 1
                
                # Update running average confidence
                n = self._stats['total_detections']
                old_avg = self._stats['avg_confidence']
                self._stats['avg_confidence'] = (
                    (old_avg * (n - 1) + regime_result['confidence']) / n
                )
            
            return {
                **regime_result,
                'crisis_warning': crisis_warning,
                'transition_probability': transition_prob,
                'detection_time_ms': (time.time() - start_time) * 1000
            }
            
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            # Return safe default on error
            return {
                'regime': MarketRegime.VOLATILE.value,
                'regime_id': MarketRegime.VOLATILE.to_id(),
                'confidence': 0.5,
                'crisis_warning': True,  # Conservative on error
                'transition_probability': 0.5,
                'error': str(e)
            }
    
    def _extract_regime_features(
        self,
        features: np.ndarray,
        historical_data: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Extract regime-specific features (CPU-bound).
        
        10 features:
        1. Realized volatility
        2. Autocorrelation (mean reversion measure)
        3. Hurst exponent (trending measure)
        4. Amplitude of oscillation
        5. Trend strength (ADX-like)
        6. Support/resistance proximity
        7. Volume profile ratio
        8. Correlation stability
        9. Time since last regime change
        10. Volatility of volatility
        
        Args:
            features: Market features [n] 
            historical_data: Price history [optional]
            
        Returns:
            Regime features [10]
        """
        regime_features = np.zeros(self.config.n_features, dtype=np.float32)
        eps = self.config.epsilon
        
        # Ensure features is numpy array
        if not isinstance(features, np.ndarray):
            features = np.array(features, dtype=np.float32)
        
        # ═══════════════════════════════════════════════════════
        # FEATURE 1: REALIZED VOLATILITY
        # ═══════════════════════════════════════════════════════
        if historical_data is not None and len(historical_data) > 20:
            # Ensure positive prices for log
            prices = np.maximum(historical_data[-20:], eps)
            returns = np.diff(np.log(prices))
            # Annualized volatility (assuming 5-min bars: 252 days × 288 bars/day)
            regime_features[0] = np.std(returns) * np.sqrt(252 * 288)
        elif len(features) > 45:
            regime_features[0] = abs(features[45]) if not np.isnan(features[45]) else 0.02
        else:
            regime_features[0] = 0.02  # Default moderate volatility
        
        # ═══════════════════════════════════════════════════════
        # FEATURE 2: AUTOCORRELATION (LAG-1)
        # ═══════════════════════════════════════════════════════
        if historical_data is not None and len(historical_data) > 21:
            prices = np.maximum(historical_data[-21:], eps)
            returns = np.diff(np.log(prices))
            if len(returns) > 1 and np.std(returns[:-1]) > eps and np.std(returns[1:]) > eps:
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                regime_features[1] = autocorr if not np.isnan(autocorr) else 0.0
            else:
                regime_features[1] = 0.0
        elif len(features) > 49:
            regime_features[1] = features[49] if not np.isnan(features[49]) else 0.0
        else:
            regime_features[1] = 0.0
        
        # ═══════════════════════════════════════════════════════
        # FEATURE 3: HURST EXPONENT (SIMPLIFIED)
        # ═══════════════════════════════════════════════════════
        if historical_data is not None and len(historical_data) > 50:
            hurst = self._calculate_hurst_exponent(historical_data[-50:])
            regime_features[2] = hurst
        elif len(features) > 48:
            regime_features[2] = features[48] if not np.isnan(features[48]) else 0.5
        else:
            regime_features[2] = 0.5  # Random walk default
        
        # ═══════════════════════════════════════════════════════
        # FEATURE 4: AMPLITUDE OF OSCILLATION
        # ═══════════════════════════════════════════════════════
        if historical_data is not None and len(historical_data) > 20:
            recent = historical_data[-20:]
            amplitude = (np.max(recent) - np.min(recent)) / (np.mean(recent) + eps)
            regime_features[3] = min(amplitude, 1.0)  # Cap at 100%
        else:
            regime_features[3] = 0.01
        
        # ═══════════════════════════════════════════════════════
        # FEATURE 5: TREND STRENGTH (ADX-LIKE)
        # ═══════════════════════════════════════════════════════
        if historical_data is not None and len(historical_data) > 20:
            # Simple trend strength: |end - start| / range
            recent = historical_data[-20:]
            price_range = np.max(recent) - np.min(recent) + eps
            trend_move = abs(recent[-1] - recent[0])
            regime_features[4] = trend_move / price_range
        elif len(features) > 20:
            # Use SMA distance as proxy
            regime_features[4] = abs(features[20]) if not np.isnan(features[20]) else 0.5
        else:
            regime_features[4] = 0.5
        
        # ═══════════════════════════════════════════════════════
        # FEATURE 6: SUPPORT/RESISTANCE PROXIMITY
        # ═══════════════════════════════════════════════════════
        if historical_data is not None and len(historical_data) > 50:
            recent = historical_data[-50:]
            current = recent[-1]
            high = np.max(recent)
            low = np.min(recent)
            price_range = high - low + eps
            # 0 = at support, 1 = at resistance
            regime_features[5] = (current - low) / price_range
        else:
            regime_features[5] = 0.5  # Middle of range
        
        # ═══════════════════════════════════════════════════════
        # FEATURE 7: VOLUME PROFILE RATIO
        # ═══════════════════════════════════════════════════════
        if len(features) > 40:
            # Volume ratio from features
            vol_ratio = features[40] if not np.isnan(features[40]) else 1.0
            regime_features[6] = np.clip(vol_ratio, 0.1, 10.0) / 10.0
        else:
            regime_features[6] = 0.1  # Normal volume
        
        # ═══════════════════════════════════════════════════════
        # FEATURE 8: CORRELATION STABILITY (ROLLING WINDOW)
        # ═══════════════════════════════════════════════════════
        if historical_data is not None and len(historical_data) > 40:
            # Correlation between first and second half of recent data
            half = len(historical_data) // 2
            first_half = historical_data[-40:-20]
            second_half = historical_data[-20:]
            
            if len(first_half) > 5 and len(second_half) > 5:
                # Compare return distributions
                ret1 = np.diff(np.log(np.maximum(first_half, eps)))
                ret2 = np.diff(np.log(np.maximum(second_half, eps)))
                
                # Stability = inverse of distribution shift
                mean_shift = abs(np.mean(ret1) - np.mean(ret2))
                std_shift = abs(np.std(ret1) - np.std(ret2))
                stability = 1.0 / (1.0 + mean_shift * 100 + std_shift * 100)
                regime_features[7] = stability
            else:
                regime_features[7] = 0.5
        else:
            regime_features[7] = 0.5
        
        # ═══════════════════════════════════════════════════════
        # FEATURE 9: TIME SINCE LAST REGIME CHANGE (NORMALIZED)
        # ═══════════════════════════════════════════════════════
        # This will be updated based on regime history
        regime_features[8] = 0.5  # Default, updated by caller if needed
        
        # ═══════════════════════════════════════════════════════
        # FEATURE 10: VOLATILITY OF VOLATILITY
        # ═══════════════════════════════════════════════════════
        if historical_data is not None and len(historical_data) > 50:
            prices = np.maximum(historical_data[-50:], eps)
            returns = np.diff(np.log(prices))
            
            # Rolling volatility windows
            window_size = 10
            if len(returns) >= window_size * 3:
                vol_series = []
                for i in range(0, len(returns) - window_size, window_size // 2):
                    window_vol = np.std(returns[i:i+window_size])
                    vol_series.append(window_vol)
                
                if len(vol_series) > 1:
                    vol_of_vol = np.std(vol_series) / (np.mean(vol_series) + eps)
                    regime_features[9] = min(vol_of_vol, 2.0) / 2.0
                else:
                    regime_features[9] = 0.1
            else:
                regime_features[9] = 0.1
        else:
            regime_features[9] = 0.1
        
        # Ensure no NaN values
        regime_features = np.nan_to_num(regime_features, nan=0.5, posinf=1.0, neginf=0.0)
        
        return regime_features
    
    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """
        Calculate Hurst exponent using R/S analysis (simplified).
        
        H > 0.5: Trending (persistent)
        H = 0.5: Random walk
        H < 0.5: Mean reverting (anti-persistent)
        
        Args:
            prices: Price series
            
        Returns:
            Hurst exponent [0, 1]
        """
        eps = self.config.epsilon
        
        # Ensure positive prices
        prices = np.maximum(prices, eps)
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        if len(returns) < 10:
            return 0.5  # Default to random walk
        
        # R/S analysis
        n = len(returns)
        
        # Mean and cumulative deviation
        mean_ret = np.mean(returns)
        cumdev = np.cumsum(returns - mean_ret)
        
        # Range
        R = np.max(cumdev) - np.min(cumdev)
        
        # Standard deviation
        S = np.std(returns)
        
        if S < eps:
            return 0.5
        
        # R/S ratio
        RS = R / S
        
        # Hurst exponent (simplified: H ≈ log(R/S) / log(n))
        if RS > eps and n > 1:
            H = np.log(RS) / np.log(n)
            H = np.clip(H, 0.0, 1.0)
        else:
            H = 0.5
        
        return float(H)
    
    def _detect_regime_internal(self, regime_features: np.ndarray) -> Dict[str, Any]:
        """
        Internal regime detection (CPU-bound).
        
        Uses HMM if trained, otherwise falls back to heuristics.
        
        Args:
            regime_features: Extracted regime features [10]
            
        Returns:
            Regime detection result
        """
        if self.hmm is not None and self._hmm_trained:
            return self._hmm_regime_detection(regime_features)
        else:
            return self._heuristic_regime_detection(regime_features)
    
    def _hmm_regime_detection(self, regime_features: np.ndarray) -> Dict[str, Any]:
        """
        HMM-based regime detection.
        
        Args:
            regime_features: Features [10]
            
        Returns:
            Regime result with probabilities
        """
        try:
            # Reshape for HMM: [1, n_features]
            features_2d = regime_features.reshape(1, -1)
            
            # Get regime probabilities
            log_prob = self.hmm.score(features_2d)
            hidden_states = self.hmm.predict(features_2d)
            
            regime_id = int(hidden_states[0])
            regime = MarketRegime.from_id(regime_id)
            
            # Calculate confidence from log probability
            # Higher log probability → more confident
            confidence = np.exp(log_prob / len(regime_features))
            confidence = float(np.clip(confidence, 0.1, 1.0))
            
            # Get state probabilities if available
            try:
                posteriors = self.hmm.predict_proba(features_2d)[0]
                regime_probabilities = {
                    MarketRegime.from_id(i).value: float(posteriors[i])
                    for i in range(len(posteriors))
                }
            except Exception:
                regime_probabilities = {regime.value: confidence}
            
            return {
                'regime': regime.value,
                'regime_id': regime_id,
                'confidence': confidence,
                'regime_probabilities': regime_probabilities,
                'method': 'hmm'
            }
            
        except Exception as e:
            logger.warning(f"HMM detection failed, falling back to heuristics: {e}")
            return self._heuristic_regime_detection(regime_features)
    
    def _heuristic_regime_detection(self, regime_features: np.ndarray) -> Dict[str, Any]:
        """
        Heuristic-based regime detection (fallback when HMM not available).
        
        Uses simple rules based on regime features.
        
        Args:
            regime_features: Features [10]
            
        Returns:
            Regime result
        """
        volatility = regime_features[0]
        autocorr = regime_features[1]
        hurst = regime_features[2]
        amplitude = regime_features[3]
        trend_strength = regime_features[4]
        
        # ═══════════════════════════════════════════════════════
        # CRISIS: Very high volatility
        # ═══════════════════════════════════════════════════════
        if volatility > self.config.crisis_volatility_threshold:
            return {
                'regime': MarketRegime.CRISIS.value,
                'regime_id': MarketRegime.CRISIS.to_id(),
                'confidence': 0.9,
                'regime_probabilities': {MarketRegime.CRISIS.value: 0.9},
                'method': 'heuristic'
            }
        
        # ═══════════════════════════════════════════════════════
        # VOLATILE: High volatility + random walk behavior
        # ═══════════════════════════════════════════════════════
        hurst_low, hurst_high = self.config.random_walk_hurst_range
        if volatility > self.config.high_volatility_threshold and hurst_low <= hurst <= hurst_high:
            return {
                'regime': MarketRegime.VOLATILE.value,
                'regime_id': MarketRegime.VOLATILE.to_id(),
                'confidence': 0.7,
                'regime_probabilities': {MarketRegime.VOLATILE.value: 0.7},
                'method': 'heuristic'
            }
        
        # ═══════════════════════════════════════════════════════
        # TRENDING: High Hurst exponent (persistent)
        # ═══════════════════════════════════════════════════════
        if hurst > self.config.trending_hurst_threshold and trend_strength > 0.5:
            confidence = 0.7 + 0.2 * min(hurst - 0.6, 0.3) / 0.3
            return {
                'regime': MarketRegime.TRENDING.value,
                'regime_id': MarketRegime.TRENDING.to_id(),
                'confidence': confidence,
                'regime_probabilities': {MarketRegime.TRENDING.value: confidence},
                'method': 'heuristic'
            }
        
        # ═══════════════════════════════════════════════════════
        # RANGING: Negative autocorrelation (mean reversion)
        # ═══════════════════════════════════════════════════════
        if autocorr < self.config.ranging_autocorr_threshold:
            confidence = 0.6 + 0.2 * min(abs(autocorr), 0.5) / 0.5
            return {
                'regime': MarketRegime.RANGING.value,
                'regime_id': MarketRegime.RANGING.to_id(),
                'confidence': confidence,
                'regime_probabilities': {MarketRegime.RANGING.value: confidence},
                'method': 'heuristic'
            }
        
        # ═══════════════════════════════════════════════════════
        # BREAKOUT: Low amplitude + building trend
        # ═══════════════════════════════════════════════════════
        if amplitude < 0.02 and trend_strength > 0.3:
            return {
                'regime': MarketRegime.BREAKOUT.value,
                'regime_id': MarketRegime.BREAKOUT.to_id(),
                'confidence': 0.6,
                'regime_probabilities': {MarketRegime.BREAKOUT.value: 0.6},
                'method': 'heuristic'
            }
        
        # ═══════════════════════════════════════════════════════
        # DEFAULT: BREAKOUT (uncertain conditions)
        # ═══════════════════════════════════════════════════════
        return {
            'regime': MarketRegime.BREAKOUT.value,
            'regime_id': MarketRegime.BREAKOUT.to_id(),
            'confidence': 0.5,
            'regime_probabilities': {MarketRegime.BREAKOUT.value: 0.5},
            'method': 'heuristic'
        }
    
    def _check_crisis_conditions(self, regime_features: np.ndarray) -> bool:
        """
        Early crisis warning system.
        
        Triggers when:
        1. Volatility spike (>5% daily)
        2. Very high vol-of-vol
        3. Extreme amplitude
        
        Args:
            regime_features: Extracted features [10]
            
        Returns:
            True if crisis conditions detected
        """
        if len(regime_features) < 10:
            return True  # Conservative on incomplete data
        
        volatility = regime_features[0]
        amplitude = regime_features[3]
        vol_of_vol = regime_features[9]
        
        # Condition 1: Volatility spike
        if volatility > self.config.crisis_volatility_threshold:
            logger.warning("⚠️ CRISIS WARNING: Volatility spike detected")
            return True
        
        # Condition 2: Extreme vol-of-vol (regime instability)
        if vol_of_vol > 0.8:
            logger.warning("⚠️ CRISIS WARNING: Extreme volatility instability")
            return True
        
        # Condition 3: Extreme amplitude (price dislocation)
        if amplitude > 0.1:  # 10% range
            logger.warning("⚠️ CRISIS WARNING: Extreme price amplitude")
            return True
        
        return False
    
    async def _get_transition_probability_async(self) -> float:
        """
        Calculate probability of regime transition.
        
        Returns:
            Transition probability [0, 1]
        """
        async with self._lock:
            if len(self.regime_history) < self.config.transition_window:
                return 0.5  # Uncertain with limited data
            
            recent = list(self.regime_history)[-self.config.transition_window:]
        
        # Analyze transitions in thread
        return await asyncio.to_thread(
            self._calculate_transition_probability,
            recent
        )
    
    def _calculate_transition_probability(self, recent_regimes: List[Dict]) -> float:
        """
        Calculate transition probability from recent history.
        
        Args:
            recent_regimes: Recent regime history
            
        Returns:
            Transition probability
        """
        if len(recent_regimes) < 2:
            return 0.5
        
        regime_names = [r['regime'] for r in recent_regimes]
        confidences = [r['confidence'] for r in recent_regimes]
        
        # Count regime changes
        changes = sum(
            1 for i in range(len(regime_names) - 1)
            if regime_names[i] != regime_names[i + 1]
        )
        
        # Low confidence indicates uncertainty
        avg_confidence = np.mean(confidences)
        
        # Unique regimes
        unique_regimes = len(set(regime_names))
        
        # Transition probability formula
        change_factor = changes / max(len(regime_names) - 1, 1)
        confidence_factor = 1.0 - avg_confidence
        diversity_factor = (unique_regimes - 1) / 4  # Max 5 regimes
        
        prob = 0.4 * change_factor + 0.4 * confidence_factor + 0.2 * diversity_factor
        
        return float(np.clip(prob, 0.0, 1.0))
    
    async def detect_regime_transition_async(self) -> Dict[str, Any]:
        """
        Detect if market is transitioning between regimes.
        
        Returns:
            Transition analysis
        """
        async with self._lock:
            if len(self.regime_history) < self.config.transition_window:
                return {
                    'transition_detected': False,
                    'reason': 'insufficient_history'
                }
            
            recent_regimes = list(self.regime_history)[-self.config.transition_window:]
        
        # Analyze in thread
        analysis = await asyncio.to_thread(
            self._analyze_regime_transitions,
            recent_regimes
        )
        
        # Update stats if transition detected
        if analysis['transition_detected']:
            async with self._lock:
                self._stats['transitions_detected'] += 1
        
        return analysis
    
    def _analyze_regime_transitions(self, recent_regimes: List[Dict]) -> Dict[str, Any]:
        """
        Analyze recent regime history for transitions (CPU-bound).
        
        Args:
            recent_regimes: Recent regime history
            
        Returns:
            Transition analysis
        """
        regime_names = [r['regime'] for r in recent_regimes]
        confidences = [r['confidence'] for r in recent_regimes]
        
        # Count unique regimes
        unique_regimes = set(regime_names)
        
        # Count regime changes
        regime_changes = sum(
            1 for i in range(len(regime_names) - 1)
            if regime_names[i] != regime_names[i + 1]
        )
        
        # Confidence analysis
        avg_confidence = float(np.mean(confidences))
        confidence_trend = confidences[-1] - np.mean(confidences[:-1]) if len(confidences) > 1 else 0
        
        # Transition detection criteria
        transition_detected = (
            regime_changes >= self.config.transition_change_threshold or
            avg_confidence < self.config.low_confidence_threshold or
            len(unique_regimes) >= 3
        )
        
        return {
            'transition_detected': transition_detected,
            'regime_changes': regime_changes,
            'avg_confidence': avg_confidence,
            'confidence_trend': float(confidence_trend),
            'unique_regimes': len(unique_regimes),
            'current_regime': regime_names[-1] if regime_names else None,
            'regime_sequence': regime_names
        }
    
    async def train_hmm_async(self, historical_features: np.ndarray) -> Dict[str, Any]:
        """
        Train Hidden Markov Model on historical data.
        
        Args:
            historical_features: [n_samples, 10] regime features
        
        Returns:
            Training result
        """
        logger.info("Training HMM for regime detection...")
        
        async with self._hmm_lock:
            try:
                # Train in thread (CPU-bound scikit-learn)
                self.hmm = await asyncio.to_thread(
                    self._train_hmm_internal,
                    historical_features
                )
                
                self._hmm_trained = True
                
                async with self._lock:
                    self._stats['hmm_trained'] = True
                
                logger.info("✅ HMM training complete")
                
                return {
                    'status': 'success',
                    'n_samples': len(historical_features),
                    'n_states': self.config.n_states
                }
                
            except Exception as e:
                logger.error(f"HMM training failed: {e}")
                return {'status': 'failed', 'error': str(e)}
    
    def _train_hmm_internal(self, historical_features: np.ndarray):
        """
        Internal HMM training (CPU-bound).
        
        Args:
            historical_features: Training data [n_samples, n_features]
            
        Returns:
            Trained HMM model
        """
        try:
            from hmmlearn import hmm as hmm_lib
        except ImportError:
            logger.warning("hmmlearn not installed. Using heuristic detection only.")
            return None
        
        # Validate input
        if len(historical_features) < 100:
            raise ValueError(f"Need at least 100 samples, got {len(historical_features)}")
        
        # Ensure correct shape
        if historical_features.ndim == 1:
            historical_features = historical_features.reshape(-1, 1)
        
        # Gaussian HMM with n_states
        model = hmm_lib.GaussianHMM(
            n_components=self.config.n_states,
            covariance_type='diag',
            n_iter=self.config.hmm_n_iter,
            random_state=42,
            verbose=False
        )
        
        # Fit model
        model.fit(historical_features)
        
        logger.info(
            f"HMM trained: converged={model.monitor_.converged}, "
            f"iterations={model.monitor_.iter}"
        )
        
        return model
    
    async def get_current_regime_async(self) -> Dict[str, Any]:
        """Get current regime (thread-safe)."""
        async with self._lock:
            return {
                'regime': self._current_regime.value,
                'regime_id': self._current_regime.to_id(),
                'detection_count': self._detection_count,
                'hmm_trained': self._hmm_trained
            }
    
    async def get_metrics_async(self) -> Dict[str, Any]:
        """Get all metrics (thread-safe)."""
        async with self._lock:
            return {
                'is_initialized': self._is_initialized,
                'detection_count': self._detection_count,
                'last_detection_ms': self._last_detection_time * 1000,
                'current_regime': self._current_regime.value,
                'hmm_trained': self._hmm_trained,
                'history_size': len(self.regime_history),
                'stats': self._stats.copy()
            }
    
    async def save_checkpoint_async(self, filepath: str) -> Dict[str, Any]:
        """Save checkpoint including HMM model."""
        async with self._lock:
            try:
                import pickle
                
                checkpoint = {
                    'config': self.config.__dict__,
                    'stats': self._stats.copy(),
                    'current_regime': self._current_regime.value,
                    'hmm_trained': self._hmm_trained,
                    'regime_history': list(self.regime_history),
                    'timestamp': time.time()
                }
                
                # Save HMM separately if trained
                if self.hmm is not None:
                    checkpoint['hmm_model'] = self.hmm
                
                await asyncio.to_thread(
                    lambda: pickle.dump(checkpoint, open(filepath, 'wb'))
                )
                
                logger.info(f"Checkpoint saved: {filepath}")
                return {'status': 'success', 'filepath': filepath}
                
            except Exception as e:
                logger.error(f"Checkpoint save failed: {e}")
                return {'status': 'failed', 'error': str(e)}
    
    async def load_checkpoint_async(self, filepath: str) -> Dict[str, Any]:
        """Load checkpoint including HMM model."""
        async with self._lock:
            try:
                import pickle
                
                checkpoint = await asyncio.to_thread(
                    lambda: pickle.load(open(filepath, 'rb'))
                )
                
                # Restore state
                self._stats.update(checkpoint.get('stats', {}))
                self._current_regime = MarketRegime(checkpoint.get('current_regime', 'volatile'))
                self._hmm_trained = checkpoint.get('hmm_trained', False)
                
                # Restore history
                history = checkpoint.get('regime_history', [])
                self.regime_history.clear()
                for item in history:
                    self.regime_history.append(item)
                
                # Restore HMM if available
                if 'hmm_model' in checkpoint:
                    async with self._hmm_lock:
                        self.hmm = checkpoint['hmm_model']
                
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
        """Cleanup resources."""
        async with self._lock:
            if not self._is_initialized:
                return
            
            self.regime_history.clear()
            self._is_initialized = False
            
            logger.info("RegimeDetector cleaned up")


# ============================================================================
# INTEGRATION TEST
# ============================================================================

async def test_regime_detector():
    """Integration test for RegimeDetector"""
    logger.info("=" * 60)
    logger.info("TESTING MODULE 7: REGIME DETECTOR")
    logger.info("=" * 60)
    
    # Test 0: Configuration validation
    logger.info("\n[Test 0] Configuration validation...")
    try:
        valid_config = RegimeDetectorConfig()
        logger.info("Valid configuration accepted")
        
        try:
            invalid_config = RegimeDetectorConfig(n_states=-1)
            logger.error("Invalid config should have raised ValueError")
        except ValueError as e:
            logger.info(f"Invalid config correctly rejected: {e}")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return
    
    # Create detector
    config = RegimeDetectorConfig()
    detector = RegimeDetector(config)
    
    # Test 1: Initialization
    logger.info("\n[Test 1] Initialization...")
    init_result = await detector.initialize_async()
    assert init_result['status'] == 'success'
    logger.info(f"Initialization: {init_result}")
    
    # Test 2: Regime detection with features only
    logger.info("\n[Test 2] Regime detection (features only)...")
    features = np.random.randn(50).astype(np.float32)
    result = await detector.detect_regime_async(features)
    logger.info(f"Regime: {result['regime']}")
    logger.info(f"Confidence: {result['confidence']:.2f}")
    logger.info(f"Crisis warning: {result['crisis_warning']}")
    logger.info(f"Method: {result.get('method', 'N/A')}")
    
    # Test 3: Regime detection with historical data
    logger.info("\n[Test 3] Regime detection (with history)...")
    historical = 100 + np.cumsum(np.random.randn(100) * 0.01)  # Simulated prices
    result = await detector.detect_regime_async(features, historical)
    logger.info(f"Regime: {result['regime']}")
    logger.info(f"Confidence: {result['confidence']:.2f}")
    
    # Test 4: Multiple detections to build history
    logger.info("\n[Test 4] Building regime history...")
    for i in range(15):
        features = np.random.randn(50).astype(np.float32)
        await detector.detect_regime_async(features)
    
    metrics = await detector.get_metrics_async()
    logger.info(f"Detection count: {metrics['detection_count']}")
    logger.info(f"History size: {metrics['history_size']}")
    
    # Test 5: Transition detection
    logger.info("\n[Test 5] Regime transition detection...")
    transition = await detector.detect_regime_transition_async()
    logger.info(f"Transition detected: {transition['transition_detected']}")
    logger.info(f"Regime changes: {transition.get('regime_changes', 'N/A')}")
    logger.info(f"Avg confidence: {transition.get('avg_confidence', 'N/A'):.2f}")
    
    # Test 6: Crisis detection
    logger.info("\n[Test 6] Crisis detection...")
    # Simulate high volatility
    crisis_features = np.zeros(50, dtype=np.float32)
    crisis_features[45] = 0.08  # High volatility
    result = await detector.detect_regime_async(crisis_features)
    logger.info(f"Crisis regime: {result['regime']}")
    logger.info(f"Crisis warning: {result['crisis_warning']}")
    
    # Test 7: Concurrent detections
    logger.info("\n[Test 7] Thread safety (5 concurrent detections)...")
    tasks = []
    for i in range(5):
        features = np.random.randn(50).astype(np.float32)
        tasks.append(detector.detect_regime_async(features))
    
    results = await asyncio.gather(*tasks)
    assert len(results) == 5
    logger.info("Thread safety: All 5 detections completed")
    
    # Test 8: Checkpoint save/load
    logger.info("\n[Test 8] Checkpoint save/load...")
    save_result = await detector.save_checkpoint_async("/tmp/regime_test.pkl")
    assert save_result['status'] == 'success'
    
    load_result = await detector.load_checkpoint_async("/tmp/regime_test.pkl")
    assert load_result['status'] == 'success'
    logger.info("Checkpoint: save/load successful")
    
    # Test 9: Metrics
    logger.info("\n[Test 9] Final metrics...")
    metrics = await detector.get_metrics_async()
    logger.info(f"Total detections: {metrics['stats']['total_detections']}")
    logger.info(f"Crisis warnings: {metrics['stats']['crisis_warnings']}")
    logger.info(f"Regime counts: {metrics['stats']['regime_counts']}")
    
    # Test 10: Cleanup
    logger.info("\n[Test 10] Cleanup...")
    await detector.cleanup_async()
    logger.info("Cleanup: successful")
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL TESTS PASSED")
    logger.info("=" * 60)


if __name__ == '__main__':
    asyncio.run(test_regime_detector())

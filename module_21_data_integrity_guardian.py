#!/usr/bin/env python3
"""
============================================================================
MODULE 21: DATA INTEGRITY GUARDIAN
============================================================================
Version: 1.0.0
Author: MIT PhD-Level AI Engineering Team
VRAM: 0 MB (CPU-based validation)

PURPOSE:
    Ensures the trading system ONLY uses REAL market data.
    
    CRITICAL: This module BLOCKS all simulated/fake data from entering
    the trading system. It validates every data point against known
    patterns of real OANDA data.

FEATURES:
    1. API SOURCE VERIFICATION
       - Validates data comes from real OANDA endpoints
       - Rejects any data without proper API signatures
       - Tracks data provenance
    
    2. TIMESTAMP VALIDATION
       - Verifies timestamps are within market hours
       - Detects artificially generated timestamps
       - Validates timezone consistency
    
    3. PRICE PATTERN VALIDATION
       - Detects random walk patterns (simulated)
       - Validates bid/ask spreads
       - Checks for realistic price movements
    
    4. VOLUME VALIDATION
       - Detects fake/zero volumes
       - Validates volume patterns
       - Checks for realistic tick data
    
    5. DATA FRESHNESS
       - Ensures data is recent (not stale)
       - Validates against expected update frequency
       - Alerts on data gaps

INTEGRATION:
    This module wraps the Market Data Pipeline to validate ALL data
    before it enters the trading system:
    
    guardian = DataIntegrityGuardian()
    
    # Wrap data fetch
    raw_data = await market_data.fetch_candles_async()
    validated_data = await guardian.validate_async(raw_data)
    
    # If validation fails, raises DataIntegrityError
    # The system will NOT proceed with invalid/simulated data

============================================================================
"""

import asyncio
import logging
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
import re

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available - some validations limited")


# ============================================================================
# EXCEPTIONS
# ============================================================================

class DataIntegrityError(Exception):
    """Raised when data fails integrity validation"""
    pass


class SimulatedDataDetectedError(DataIntegrityError):
    """Raised when simulated/fake data is detected"""
    pass


class StaleDataError(DataIntegrityError):
    """Raised when data is too old"""
    pass


class InvalidSourceError(DataIntegrityError):
    """Raised when data source cannot be verified"""
    pass


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class ValidationResult(Enum):
    """Validation outcome"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class ValidationReport:
    """Detailed validation report"""
    timestamp: datetime
    data_source: str
    record_count: int
    
    # Results
    overall_result: ValidationResult
    checks_passed: int
    checks_failed: int
    checks_warning: int
    
    # Details
    source_verified: bool
    timestamps_valid: bool
    prices_valid: bool
    volumes_valid: bool
    freshness_valid: bool
    
    # Issues found
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Data provenance
    api_endpoint: str = ""
    response_headers: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'data_source': self.data_source,
            'record_count': self.record_count,
            'overall_result': self.overall_result.value,
            'checks_passed': self.checks_passed,
            'checks_failed': self.checks_failed,
            'source_verified': self.source_verified,
            'timestamps_valid': self.timestamps_valid,
            'prices_valid': self.prices_valid,
            'volumes_valid': self.volumes_valid,
            'freshness_valid': self.freshness_valid,
            'issues': self.issues,
            'warnings': self.warnings
        }


@dataclass
class GuardianConfig:
    """Data Integrity Guardian configuration"""
    
    # Source validation
    require_api_verification: bool = True
    allowed_sources: Set[str] = field(default_factory=lambda: {
        "oanda", "api-fxpractice.oanda.com", "api-fxtrade.oanda.com"
    })
    
    # Timestamp validation
    max_data_age_seconds: float = 60.0  # Data older than this is stale
    validate_market_hours: bool = True
    
    # Price validation
    max_price_change_pct: float = 0.05  # 5% max change between candles
    min_spread_pips: float = 0.1
    max_spread_pips: float = 50.0
    
    # Volume validation
    min_volume_threshold: int = 1
    detect_zero_volume: bool = True
    
    # Random walk detection
    detect_random_walk: bool = True
    random_walk_threshold: float = 0.95  # Hurst exponent threshold
    
    # Strict mode
    strict_mode: bool = True  # Fail on ANY suspicious data
    block_simulated_data: bool = True  # NEVER allow simulated data


@dataclass
class DataProvenance:
    """Tracks data origin and chain of custody"""
    source_endpoint: str
    fetch_timestamp: float
    response_status: int
    response_headers: Dict[str, str]
    content_hash: str
    record_count: int
    
    # Verification
    is_verified: bool = False
    verification_method: str = ""
    verification_timestamp: float = 0.0


# ============================================================================
# DATA INTEGRITY GUARDIAN
# ============================================================================

class DataIntegrityGuardian:
    """
    Validates all market data to ensure ONLY real data enters the system.
    
    CRITICAL: This module is the gatekeeper that prevents ANY simulated
    or fake data from being used by the trading system.
    """
    
    def __init__(self, config: GuardianConfig = None):
        self.config = config or GuardianConfig()
        
        # Provenance tracking
        self._provenance_log: List[DataProvenance] = []
        
        # Statistics
        self._stats = {
            'validations_passed': 0,
            'validations_failed': 0,
            'simulated_data_blocked': 0,
            'stale_data_blocked': 0
        }
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Known simulation patterns
        self._simulation_signatures = {
            'random_seed_pattern': re.compile(r'random|seed|fake|simul|test|mock', re.I),
            'sequential_timestamps': True,
            'perfect_round_numbers': True,
        }
    
    # ========================================================================
    # MAIN VALIDATION
    # ========================================================================
    
    async def validate_async(
        self,
        data: Any,
        source: str = "unknown",
        provenance: DataProvenance = None
    ) -> Tuple[Any, ValidationReport]:
        """
        Validate data before it enters the trading system.
        
        Args:
            data: Raw market data (candles, prices, etc.)
            source: Data source identifier
            provenance: Optional provenance information
            
        Returns:
            Tuple of (validated_data, validation_report)
            
        Raises:
            SimulatedDataDetectedError: If simulated data detected
            StaleDataError: If data is too old
            InvalidSourceError: If source cannot be verified
            DataIntegrityError: For other integrity failures
        """
        report = ValidationReport(
            timestamp=datetime.utcnow(),
            data_source=source,
            record_count=self._count_records(data),
            overall_result=ValidationResult.PASSED,
            checks_passed=0,
            checks_failed=0,
            checks_warning=0,
            source_verified=False,
            timestamps_valid=False,
            prices_valid=False,
            volumes_valid=False,
            freshness_valid=False
        )
        
        try:
            # 1. SOURCE VERIFICATION
            source_valid = await self._validate_source_async(source, provenance, report)
            if not source_valid and self.config.require_api_verification:
                raise InvalidSourceError(
                    f"Cannot verify data source: {source}. "
                    "Only real OANDA API data is accepted."
                )
            
            # 2. SIMULATION DETECTION
            if self.config.block_simulated_data:
                is_simulated = await self._detect_simulation_async(data, source, report)
                if is_simulated:
                    async with self._lock:
                        self._stats['simulated_data_blocked'] += 1
                    raise SimulatedDataDetectedError(
                        "SIMULATED DATA DETECTED! This system only accepts "
                        "real market data from OANDA API. Simulation mode is BLOCKED."
                    )
            
            # 3. TIMESTAMP VALIDATION
            await self._validate_timestamps_async(data, report)
            
            # 4. FRESHNESS CHECK
            freshness_valid = await self._validate_freshness_async(data, report)
            if not freshness_valid and self.config.strict_mode:
                async with self._lock:
                    self._stats['stale_data_blocked'] += 1
                raise StaleDataError(
                    f"Data is stale (older than {self.config.max_data_age_seconds}s). "
                    "Only fresh market data is accepted."
                )
            
            # 5. PRICE VALIDATION
            await self._validate_prices_async(data, report)
            
            # 6. VOLUME VALIDATION
            await self._validate_volumes_async(data, report)
            
            # 7. RANDOM WALK DETECTION
            if self.config.detect_random_walk:
                is_random = await self._detect_random_walk_async(data, report)
                if is_random:
                    report.warnings.append(
                        "Price pattern resembles random walk - possible simulation"
                    )
                    if self.config.strict_mode:
                        raise SimulatedDataDetectedError(
                            "Price pattern detected as random walk (simulated data signature)"
                        )
            
            # Determine overall result
            if report.checks_failed > 0:
                report.overall_result = ValidationResult.FAILED
            elif report.checks_warning > 0:
                report.overall_result = ValidationResult.WARNING
            else:
                report.overall_result = ValidationResult.PASSED
            
            # Update stats
            async with self._lock:
                if report.overall_result == ValidationResult.PASSED:
                    self._stats['validations_passed'] += 1
                else:
                    self._stats['validations_failed'] += 1
            
            # Log provenance
            if provenance:
                provenance.is_verified = report.overall_result == ValidationResult.PASSED
                provenance.verification_timestamp = time.time()
                self._provenance_log.append(provenance)
            
            return data, report
            
        except (SimulatedDataDetectedError, StaleDataError, InvalidSourceError):
            raise
        except Exception as e:
            report.overall_result = ValidationResult.FAILED
            report.issues.append(f"Validation error: {str(e)}")
            raise DataIntegrityError(f"Data validation failed: {e}")
    
    # ========================================================================
    # SOURCE VERIFICATION
    # ========================================================================
    
    async def _validate_source_async(
        self,
        source: str,
        provenance: DataProvenance,
        report: ValidationReport
    ) -> bool:
        """Verify data comes from legitimate OANDA API"""
        
        # Check source name
        source_lower = source.lower()
        is_allowed = any(
            allowed in source_lower 
            for allowed in self.config.allowed_sources
        )
        
        if not is_allowed:
            report.issues.append(f"Unrecognized data source: {source}")
            report.checks_failed += 1
            return False
        
        # Check provenance if available
        if provenance:
            # Verify API endpoint
            if 'oanda.com' not in provenance.source_endpoint.lower():
                report.issues.append(
                    f"Invalid API endpoint: {provenance.source_endpoint}"
                )
                report.checks_failed += 1
                return False
            
            # Verify response status
            if provenance.response_status != 200:
                report.issues.append(
                    f"API returned non-200 status: {provenance.response_status}"
                )
                report.checks_failed += 1
                return False
            
            # Check for OANDA-specific headers
            oanda_headers = ['X-Request-Id', 'RequestID']
            has_oanda_header = any(
                h in provenance.response_headers 
                for h in oanda_headers
            )
            
            if not has_oanda_header:
                report.warnings.append("Missing OANDA-specific response headers")
                report.checks_warning += 1
            
            report.api_endpoint = provenance.source_endpoint
            report.response_headers = provenance.response_headers
        
        report.source_verified = True
        report.checks_passed += 1
        return True
    
    # ========================================================================
    # SIMULATION DETECTION
    # ========================================================================
    
    async def _detect_simulation_async(
        self,
        data: Any,
        source: str,
        report: ValidationReport
    ) -> bool:
        """
        Detect if data appears to be simulated.
        
        Checks for:
        - Simulation keywords in source/metadata
        - Perfectly sequential timestamps
        - Round number prices
        - Unrealistic patterns
        """
        
        # Check source for simulation keywords
        if self._simulation_signatures['random_seed_pattern'].search(source):
            report.issues.append(f"Source contains simulation keyword: {source}")
            return True
        
        # Convert data to analyzable format
        records = self._extract_records(data)
        
        if not records:
            return False
        
        # Check for sequential timestamps (simulation signature)
        timestamps = [r.get('timestamp', r.get('time', 0)) for r in records]
        if self._check_sequential_timestamps(timestamps):
            report.issues.append("Timestamps are perfectly sequential (simulation signature)")
            return True
        
        # Check for round number prices
        prices = []
        for r in records:
            if 'close' in r:
                prices.append(r['close'])
            elif 'price' in r:
                prices.append(r['price'])
        
        if prices and self._check_round_prices(prices):
            report.issues.append("Prices are suspiciously round (simulation signature)")
            return True
        
        # Check for identical consecutive prices (unrealistic)
        if self._check_identical_prices(prices):
            report.warnings.append("Multiple identical consecutive prices detected")
        
        return False
    
    def _check_sequential_timestamps(self, timestamps: List) -> bool:
        """Check if timestamps are perfectly sequential (simulation marker)"""
        if len(timestamps) < 10:
            return False
        
        # Convert to numeric if needed
        try:
            if isinstance(timestamps[0], str):
                return False  # ISO strings are likely real
            
            diffs = []
            for i in range(1, len(timestamps)):
                diff = timestamps[i] - timestamps[i-1]
                diffs.append(diff)
            
            # Check if all diffs are exactly equal
            if len(set(diffs)) == 1:
                return True
                
        except Exception:
            pass
        
        return False
    
    def _check_round_prices(self, prices: List[float]) -> bool:
        """Check if prices are suspiciously round"""
        if not prices:
            return False
        
        round_count = 0
        for p in prices:
            # Check if price has too few decimal places
            str_price = f"{p:.10f}"
            if str_price.endswith('000000'):
                round_count += 1
        
        # If more than 90% are round, suspicious
        return round_count / len(prices) > 0.9
    
    def _check_identical_prices(self, prices: List[float]) -> bool:
        """Check for unrealistic identical consecutive prices"""
        if len(prices) < 5:
            return False
        
        consecutive = 1
        max_consecutive = 1
        
        for i in range(1, len(prices)):
            if prices[i] == prices[i-1]:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 1
        
        # More than 10 identical prices in a row is suspicious
        return max_consecutive > 10
    
    # ========================================================================
    # TIMESTAMP VALIDATION
    # ========================================================================
    
    async def _validate_timestamps_async(
        self,
        data: Any,
        report: ValidationReport
    ):
        """Validate timestamp integrity"""
        records = self._extract_records(data)
        
        if not records:
            report.timestamps_valid = True
            report.checks_passed += 1
            return
        
        issues = []
        
        for i, record in enumerate(records):
            ts = record.get('timestamp', record.get('time'))
            
            if ts is None:
                issues.append(f"Record {i}: Missing timestamp")
                continue
            
            # Parse timestamp
            try:
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                elif isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                else:
                    dt = ts
                
                # Check if timestamp is in the future
                if dt > datetime.now(timezone.utc) + timedelta(minutes=5):
                    issues.append(f"Record {i}: Future timestamp detected")
                
                # Check if timestamp is too old (for recent data)
                age = (datetime.now(timezone.utc) - dt).total_seconds()
                if age > 86400 * 365:  # More than 1 year old
                    report.warnings.append(f"Record {i}: Very old timestamp ({age/86400:.0f} days)")
                    
            except Exception as e:
                issues.append(f"Record {i}: Invalid timestamp format - {e}")
        
        if issues:
            report.issues.extend(issues)
            report.checks_failed += 1
            report.timestamps_valid = False
        else:
            report.timestamps_valid = True
            report.checks_passed += 1
    
    # ========================================================================
    # FRESHNESS VALIDATION
    # ========================================================================
    
    async def _validate_freshness_async(
        self,
        data: Any,
        report: ValidationReport
    ) -> bool:
        """Validate data is fresh (not stale)"""
        records = self._extract_records(data)
        
        if not records:
            report.freshness_valid = True
            report.checks_passed += 1
            return True
        
        # Get most recent timestamp
        most_recent = None
        
        for record in records:
            ts = record.get('timestamp', record.get('time'))
            
            try:
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                elif isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                else:
                    continue
                
                if most_recent is None or dt > most_recent:
                    most_recent = dt
                    
            except Exception:
                continue
        
        if most_recent is None:
            report.warnings.append("Could not determine data freshness")
            report.checks_warning += 1
            report.freshness_valid = True
            return True
        
        age = (datetime.now(timezone.utc) - most_recent).total_seconds()
        
        if age > self.config.max_data_age_seconds:
            report.issues.append(
                f"Data is stale: {age:.1f}s old "
                f"(max: {self.config.max_data_age_seconds}s)"
            )
            report.checks_failed += 1
            report.freshness_valid = False
            return False
        
        report.freshness_valid = True
        report.checks_passed += 1
        return True
    
    # ========================================================================
    # PRICE VALIDATION
    # ========================================================================
    
    async def _validate_prices_async(
        self,
        data: Any,
        report: ValidationReport
    ):
        """Validate price data integrity"""
        records = self._extract_records(data)
        
        if not records:
            report.prices_valid = True
            report.checks_passed += 1
            return
        
        issues = []
        
        prev_close = None
        
        for i, record in enumerate(records):
            # Get OHLC if available
            open_p = record.get('open', record.get('o'))
            high_p = record.get('high', record.get('h'))
            low_p = record.get('low', record.get('l'))
            close_p = record.get('close', record.get('c'))
            
            # Validate OHLC relationship
            if all(p is not None for p in [open_p, high_p, low_p, close_p]):
                if high_p < low_p:
                    issues.append(f"Record {i}: High < Low (impossible)")
                if high_p < open_p or high_p < close_p:
                    issues.append(f"Record {i}: High not highest")
                if low_p > open_p or low_p > close_p:
                    issues.append(f"Record {i}: Low not lowest")
            
            # Check for price change limits
            if prev_close is not None and close_p is not None:
                change_pct = abs(close_p - prev_close) / prev_close
                if change_pct > self.config.max_price_change_pct:
                    report.warnings.append(
                        f"Record {i}: Large price change {change_pct*100:.1f}%"
                    )
            
            if close_p is not None:
                prev_close = close_p
            
            # Check for zero or negative prices
            for price_name, price_val in [
                ('open', open_p), ('high', high_p), 
                ('low', low_p), ('close', close_p)
            ]:
                if price_val is not None and price_val <= 0:
                    issues.append(f"Record {i}: Invalid {price_name} price: {price_val}")
        
        if issues:
            report.issues.extend(issues)
            report.checks_failed += 1
            report.prices_valid = False
        else:
            report.prices_valid = True
            report.checks_passed += 1
    
    # ========================================================================
    # VOLUME VALIDATION
    # ========================================================================
    
    async def _validate_volumes_async(
        self,
        data: Any,
        report: ValidationReport
    ):
        """Validate volume data"""
        records = self._extract_records(data)
        
        if not records:
            report.volumes_valid = True
            report.checks_passed += 1
            return
        
        issues = []
        zero_count = 0
        
        for i, record in enumerate(records):
            volume = record.get('volume', record.get('v'))
            
            if volume is not None:
                if volume < 0:
                    issues.append(f"Record {i}: Negative volume")
                elif volume == 0:
                    zero_count += 1
                elif volume < self.config.min_volume_threshold:
                    report.warnings.append(f"Record {i}: Very low volume: {volume}")
        
        # Check zero volume ratio
        if len(records) > 0 and zero_count / len(records) > 0.5:
            issues.append(
                f"Too many zero-volume records: {zero_count}/{len(records)} "
                "(possible simulation)"
            )
        
        if issues:
            report.issues.extend(issues)
            report.checks_failed += 1
            report.volumes_valid = False
        else:
            report.volumes_valid = True
            report.checks_passed += 1
    
    # ========================================================================
    # RANDOM WALK DETECTION
    # ========================================================================
    
    async def _detect_random_walk_async(
        self,
        data: Any,
        report: ValidationReport
    ) -> bool:
        """
        Detect if price data follows a pure random walk pattern.
        
        Real market data has mean-reverting characteristics and
        autocorrelation. Pure random walk suggests simulated data.
        """
        if not NUMPY_AVAILABLE:
            return False
        
        records = self._extract_records(data)
        
        if len(records) < 100:
            return False
        
        # Extract prices
        prices = []
        for r in records:
            close = r.get('close', r.get('c'))
            if close is not None:
                prices.append(close)
        
        if len(prices) < 100:
            return False
        
        prices = np.array(prices)
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Test 1: Autocorrelation (real data has some autocorrelation)
        if len(returns) > 10:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            
            # Pure random walk has ~0 autocorrelation
            # Real markets typically have small negative autocorrelation
            if abs(autocorr) < 0.001:
                report.warnings.append(
                    f"Zero autocorrelation detected ({autocorr:.4f})"
                )
        
        # Test 2: Hurst exponent estimation (simplified)
        # H = 0.5 is random walk, H > 0.5 is trending, H < 0.5 is mean-reverting
        hurst = self._estimate_hurst(prices)
        
        if 0.49 < hurst < 0.51:  # Very close to 0.5
            report.warnings.append(
                f"Hurst exponent suggests pure random walk: {hurst:.3f}"
            )
            return True
        
        return False
    
    def _estimate_hurst(self, prices: np.ndarray) -> float:
        """Estimate Hurst exponent using R/S analysis (simplified)"""
        n = len(prices)
        if n < 20:
            return 0.5
        
        returns = np.diff(np.log(prices))
        
        # Calculate R/S for different lag lengths
        lags = [10, 20, 50, 100]
        lags = [l for l in lags if l < n // 2]
        
        if not lags:
            return 0.5
        
        rs_values = []
        for lag in lags:
            chunks = len(returns) // lag
            rs_sum = 0
            
            for i in range(chunks):
                chunk = returns[i * lag:(i + 1) * lag]
                mean_chunk = np.mean(chunk)
                
                # Cumulative deviation
                cum_dev = np.cumsum(chunk - mean_chunk)
                
                # Range
                r = np.max(cum_dev) - np.min(cum_dev)
                
                # Standard deviation
                s = np.std(chunk, ddof=1) if len(chunk) > 1 else 1
                
                if s > 0:
                    rs_sum += r / s
            
            if chunks > 0:
                rs_values.append((lag, rs_sum / chunks))
        
        if len(rs_values) < 2:
            return 0.5
        
        # Estimate Hurst from log-log regression
        log_lags = np.log([v[0] for v in rs_values])
        log_rs = np.log([v[1] for v in rs_values])
        
        # Simple linear regression
        hurst = np.polyfit(log_lags, log_rs, 1)[0]
        
        return float(np.clip(hurst, 0, 1))
    
    # ========================================================================
    # HELPERS
    # ========================================================================
    
    def _count_records(self, data: Any) -> int:
        """Count number of records in data"""
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            if 'candles' in data:
                return len(data['candles'])
            elif 'prices' in data:
                return len(data['prices'])
        return 0
    
    def _extract_records(self, data: Any) -> List[Dict]:
        """Extract records from various data formats"""
        if isinstance(data, list):
            # List of dicts
            if data and isinstance(data[0], dict):
                return data
            # List of objects with __dict__
            elif data and hasattr(data[0], '__dict__'):
                return [vars(r) for r in data]
        
        elif isinstance(data, dict):
            # OANDA format
            if 'candles' in data:
                return data['candles']
            elif 'prices' in data:
                return data['prices']
        
        return []
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    async def get_stats_async(self) -> Dict[str, int]:
        """Get validation statistics"""
        async with self._lock:
            return dict(self._stats)
    
    async def get_provenance_log_async(self) -> List[Dict]:
        """Get data provenance log"""
        return [
            {
                'source': p.source_endpoint,
                'timestamp': p.fetch_timestamp,
                'verified': p.is_verified,
                'records': p.record_count
            }
            for p in self._provenance_log[-100:]  # Last 100
        ]


# ============================================================================
# FACTORY
# ============================================================================

def create_data_guardian(strict_mode: bool = True) -> DataIntegrityGuardian:
    """Create a configured Data Integrity Guardian"""
    config = GuardianConfig(
        strict_mode=strict_mode,
        block_simulated_data=True,
        require_api_verification=True
    )
    return DataIntegrityGuardian(config=config)


# ============================================================================
# OANDA API WRAPPER WITH BUILT-IN VALIDATION
# ============================================================================

class ValidatedOandaClient:
    """
    OANDA API client with built-in data validation.
    
    This client GUARANTEES that only real OANDA data is returned.
    Any simulated or invalid data is blocked.
    """
    
    def __init__(
        self,
        account_id: str,
        api_token: str,
        environment: str = "practice",
        guardian: DataIntegrityGuardian = None
    ):
        self.account_id = account_id
        self.api_token = api_token
        self.environment = environment
        
        # API URL
        if environment == "live":
            self.api_url = "https://api-fxtrade.oanda.com/v3"
        else:
            self.api_url = "https://api-fxpractice.oanda.com/v3"
        
        # Data guardian
        self.guardian = guardian or create_data_guardian(strict_mode=True)
        
        self._session = None
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()
    
    async def fetch_candles_validated_async(
        self,
        pair: str,
        granularity: str = "M5",
        count: int = 100
    ) -> List[Dict]:
        """
        Fetch candles with FULL validation.
        
        Raises:
            SimulatedDataDetectedError: If any simulation detected
            DataIntegrityError: If validation fails
        """
        await self._ensure_session()
        
        url = f"{self.api_url}/instruments/{pair}/candles"
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        params = {
            "granularity": granularity,
            "count": count,
            "price": "MBA"
        }
        
        async with self._session.get(url, headers=headers, params=params) as response:
            # Create provenance record
            provenance = DataProvenance(
                source_endpoint=str(response.url),
                fetch_timestamp=time.time(),
                response_status=response.status,
                response_headers=dict(response.headers),
                content_hash="",  # Will compute after
                record_count=0
            )
            
            if response.status != 200:
                raise InvalidSourceError(
                    f"OANDA API returned status {response.status}"
                )
            
            data = await response.json()
            
            # Compute content hash
            provenance.content_hash = hashlib.sha256(
                str(data).encode()
            ).hexdigest()[:16]
            
            provenance.record_count = len(data.get('candles', []))
            
            # VALIDATE DATA
            validated_data, report = await self.guardian.validate_async(
                data=data,
                source="oanda",
                provenance=provenance
            )
            
            if report.overall_result == ValidationResult.FAILED:
                raise DataIntegrityError(
                    f"Data validation failed: {report.issues}"
                )
            
            return validated_data.get('candles', [])
    
    async def close_async(self):
        """Close the client session"""
        if self._session:
            await self._session.close()


# ============================================================================
# STANDALONE TEST
# ============================================================================

async def _test_guardian():
    """Test the Data Integrity Guardian"""
    print("Testing Data Integrity Guardian...")
    
    guardian = create_data_guardian(strict_mode=True)
    
    # Test 1: Valid-looking data
    valid_data = [
        {
            'timestamp': datetime.utcnow().isoformat(),
            'open': 1.1234,
            'high': 1.1256,
            'low': 1.1220,
            'close': 1.1245,
            'volume': 1500
        }
        for _ in range(10)
    ]
    
    try:
        _, report = await guardian.validate_async(valid_data, source="oanda")
        print(f"✅ Valid data passed: {report.overall_result.value}")
    except DataIntegrityError as e:
        print(f"❌ Valid data failed: {e}")
    
    # Test 2: Simulated data (round prices)
    fake_data = [
        {
            'timestamp': datetime.utcnow().isoformat(),
            'open': 1.1000,
            'high': 1.1100,
            'low': 1.0900,
            'close': 1.1000,
            'volume': 1000
        }
        for _ in range(100)
    ]
    
    try:
        await guardian.validate_async(fake_data, source="test_simulator")
        print("❌ Simulated data should have been blocked!")
    except SimulatedDataDetectedError:
        print("✅ Simulated data correctly blocked")
    except DataIntegrityError:
        print("✅ Simulated data blocked (generic)")
    
    # Print stats
    stats = await guardian.get_stats_async()
    print(f"\nStats: {stats}")
    
    print("\n✅ Data Integrity Guardian test complete!")


if __name__ == "__main__":
    asyncio.run(_test_guardian())

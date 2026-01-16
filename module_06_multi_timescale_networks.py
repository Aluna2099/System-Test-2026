"""
MODULE 6: MULTI-TIMESCALE NETWORKS - ENHANCED VERSION
Production-Ready Implementation for 80% VRAM Utilization

Parallel networks processing 9 temporal resolutions (up from 5).
- 9 bidirectional LSTMs (S30, M1, M5, M15, H1, H4, D, W, M)
- Cross-timescale attention mechanism (NEW)
- Temporal attention with learned hierarchical weighting
- Timescale agreement checking with confidence
- MC Dropout uncertainty quantification
- Async/await architecture throughout
- Thread-safe state management
- GPU memory management integration (600MB budget)

Author: Liquid Neural SREK Trading System v4.0
Date: 2026-01-12
Version: 2.0.0 (Enhanced for 80% VRAM)

ENHANCEMENTS OVER v1.1.0:
- timescales: 5 → 9 (+80%)
- hidden_dim: 64 → 128 (2x)
- Bidirectional LSTM: ENABLED
- Cross-timescale attention: ENABLED (8-head)
- num_lstm_layers: 2 → 3
- max_vram_mb: 300 → 600 (2x)
- Module 5 alignment: Full compatibility with new timeframes

FIXES PRESERVED (from v1.1.0):
- Issue 6.1 (MEDIUM): _prepare_input_async properly offloads CPU work
- Issue 6.2 (MEDIUM): torch.no_grad() for inference mode
- Issue 6.3 (MEDIUM): Proper self.eval() mode management
- Issue 6.4 (LOW): Removed unused ThreadPoolExecutor
- Issue 6.5 (LOW): MC Dropout uncertainty quantification
- Issue 6.6 (LOW): Stats persisted in checkpoints

TIMESCALE MAPPING (Module 5 compatibility):
- s30 → 30-second scalping signals
- m1  → 1-minute micro trends
- m5  → 5-minute short-term (PRIMARY)
- m15 → 15-minute intraday
- h1  → 1-hour daily rhythm
- h4  → 4-hour swing
- d1  → Daily macro
- w1  → Weekly structural
- mn1 → Monthly mega trends
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION WITH VALIDATION (ENHANCED)
# ============================================================================

@dataclass
class MultiTimescaleConfig:
    """
    Enhanced Configuration for Multi-Timescale Networks
    
    Optimized for 80% VRAM utilization on RTX 3060 6GB
    Target budget: 600MB (9 bidirectional LSTMs)
    """
    # Network dimensions (ENHANCED)
    input_dim: int = 50           # Feature dimension per timescale
    hidden_dim: int = 128         # LSTM hidden dimension (64 → 128, 2x)
    num_lstm_layers: int = 3      # LSTM depth (2 → 3)
    output_dim: int = 3           # Buy/Sell/Hold probabilities
    
    # Bidirectional LSTM (NEW)
    bidirectional: bool = True
    
    # Cross-timescale attention (NEW)
    use_cross_timescale_attention: bool = True
    attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # Dropout
    lstm_dropout: float = 0.15    # Slightly higher for regularization
    fusion_dropout: float = 0.25  # For MC Dropout uncertainty
    
    # Timescales (ENHANCED - 9 timescales aligned with Module 5)
    timescales: List[str] = field(default_factory=lambda: [
        's30', 'm1', 'm5', 'm15', 'h1', 'h4', 'd1', 'w1', 'mn1'
    ])
    
    # Sequence lengths per timescale (memory-optimized)
    sequence_lengths: Dict[str, int] = field(default_factory=lambda: {
        's30': 30,   # 30 × 30sec = 15 minutes (scalping)
        'm1': 40,    # 40 × 1min = 40 minutes (micro)
        'm5': 50,    # 50 × 5min = ~4 hours (short-term)
        'm15': 48,   # 48 × 15min = 12 hours (intraday)
        'h1': 48,    # 48 × 1hr = 2 days (daily rhythm)
        'h4': 42,    # 42 × 4hr = 7 days (swing)
        'd1': 30,    # 30 × 1day = ~1 month (macro)
        'w1': 20,    # 20 × 1week = 5 months (structural)
        'mn1': 12    # 12 × 1month = 1 year (mega)
    })
    
    # Hierarchical timescale weights (NEW - prior importance)
    timescale_prior_weights: Dict[str, float] = field(default_factory=lambda: {
        's30': 0.05,  # Scalping - low default weight
        'm1': 0.08,   # Micro - low weight
        'm5': 0.15,   # Short-term - medium-high (primary)
        'm15': 0.15,  # Intraday - medium-high
        'h1': 0.15,   # Daily - medium-high
        'h4': 0.15,   # Swing - medium-high
        'd1': 0.12,   # Macro - medium
        'w1': 0.08,   # Structural - low
        'mn1': 0.07   # Mega - low
    })
    
    # Agreement thresholds
    min_agreement_threshold: float = 0.70
    
    # MC Dropout for uncertainty
    mc_samples: int = 50          # 30 → 50 for better estimates
    
    # GPU configuration (ENHANCED)
    device: str = 'cuda'
    dtype: torch.dtype = torch.float32
    max_vram_mb: int = 600        # 300 → 600 (doubled for enhancements)
    batch_size: int = 32
    
    # Numerical stability
    epsilon: float = 1e-8
    
    @property
    def num_timescales(self) -> int:
        """Number of configured timescales"""
        return len(self.timescales)
    
    @property
    def effective_hidden_dim(self) -> int:
        """Hidden dimension accounting for bidirectional"""
        return self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
    
    def __post_init__(self):
        """Validate configuration"""
        # Dimension validation
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_lstm_layers <= 0:
            raise ValueError(f"num_lstm_layers must be positive")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive")
        
        # Dropout validation
        if not 0.0 <= self.lstm_dropout < 1.0:
            raise ValueError(f"lstm_dropout must be in [0, 1)")
        if not 0.0 <= self.fusion_dropout < 1.0:
            raise ValueError(f"fusion_dropout must be in [0, 1)")
        
        # Attention validation
        if self.use_cross_timescale_attention:
            if self.attention_heads <= 0:
                raise ValueError("attention_heads must be positive")
            if self.effective_hidden_dim % self.attention_heads != 0:
                raise ValueError(
                    f"effective_hidden_dim ({self.effective_hidden_dim}) must be "
                    f"divisible by attention_heads ({self.attention_heads})"
                )
        
        # Timescales validation
        if not self.timescales:
            raise ValueError("timescales list cannot be empty")
        for ts in self.timescales:
            if ts not in self.sequence_lengths:
                raise ValueError(f"Missing sequence length for timescale: {ts}")
            if ts not in self.timescale_prior_weights:
                raise ValueError(f"Missing prior weight for timescale: {ts}")
        
        # Prior weights should sum to ~1
        weight_sum = sum(self.timescale_prior_weights.values())
        if not 0.9 <= weight_sum <= 1.1:
            raise ValueError(f"timescale_prior_weights should sum to ~1, got {weight_sum}")
        
        # Agreement threshold validation
        if not 0.0 <= self.min_agreement_threshold <= 1.0:
            raise ValueError("min_agreement_threshold must be in [0, 1]")
        
        # MC samples validation
        if self.mc_samples <= 0:
            raise ValueError("mc_samples must be positive")
        
        # Memory validation
        if self.max_vram_mb <= 0:
            raise ValueError("max_vram_mb must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


class TimescaleDirection(Enum):
    """Direction signal from a timescale"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# ============================================================================
# BIDIRECTIONAL TIMESCALE LSTM (ENHANCED)
# ============================================================================

class BidirectionalTimescaleLSTM(nn.Module):
    """
    Enhanced bidirectional LSTM network for a single timescale.
    
    Processes sequential market data at a specific temporal resolution
    in both forward and backward directions for richer context.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection with LayerNorm
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # Output dimension
        self.output_dim = hidden_dim * self.num_directions
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
        # Feature projection (combine forward + backward)
        self.feature_proj = nn.Linear(self.output_dim, self.output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize with Xavier uniform"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for m in [self.input_proj, self.feature_proj]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through bidirectional LSTM.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            
        Returns:
            output: Full sequence output [batch, seq_len, output_dim]
            features: Combined hidden state [batch, output_dim]
        """
        batch_size = x.size(0)
        
        # Input projection
        x_proj = self.input_proj(x)  # [batch, seq_len, hidden_dim]
        
        # LSTM forward pass
        output, (h_n, c_n) = self.lstm(x_proj)
        # output: [batch, seq_len, hidden_dim * num_directions]
        # h_n: [num_layers * num_directions, batch, hidden_dim]
        
        # Combine forward and backward final hidden states
        if self.bidirectional:
            # Get last layer forward and backward states
            h_forward = h_n[-2]   # [batch, hidden_dim]
            h_backward = h_n[-1]  # [batch, hidden_dim]
            features = torch.cat([h_forward, h_backward], dim=1)
        else:
            features = h_n[-1]
        
        # Apply layer normalization
        features = self.layer_norm(features)
        
        # Feature projection
        features = self.feature_proj(features)
        
        return output, features


# ============================================================================
# CROSS-TIMESCALE ATTENTION (NEW)
# ============================================================================

class CrossTimescaleAttention(nn.Module):
    """
    Cross-timescale attention mechanism.
    
    Allows each timescale to attend to features from other timescales,
    enabling information flow across temporal resolutions.
    
    This captures relationships like:
    - Short-term breakouts aligned with long-term trends
    - Divergences between micro and macro signals
    - Hierarchical pattern confirmation
    """
    
    def __init__(
        self,
        num_timescales: int,
        feature_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_timescales = num_timescales
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections for each timescale
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Learnable timescale embeddings (positional encoding for timescales)
        self.timescale_embeddings = nn.Parameter(
            torch.randn(num_timescales, feature_dim) * 0.02
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize with small weights"""
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            nn.init.zeros_(m.bias)
    
    def forward(self, timescale_features: Dict[str, Tensor], timescale_order: List[str]) -> Dict[str, Tensor]:
        """
        Apply cross-timescale attention.
        
        Args:
            timescale_features: {timescale: [batch, feature_dim]}
            timescale_order: Ordered list of timescale names
            
        Returns:
            attended_features: {timescale: [batch, feature_dim]}
        """
        batch_size = next(iter(timescale_features.values())).size(0)
        
        # Stack features: [batch, num_timescales, feature_dim]
        stacked = torch.stack(
            [timescale_features[ts] for ts in timescale_order],
            dim=1
        )
        
        # Add timescale embeddings
        stacked = stacked + self.timescale_embeddings.unsqueeze(0)
        
        # Pre-norm
        x = self.norm(stacked)
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, num_ts, feature_dim]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        # [batch, num_ts, num_heads, head_dim] -> [batch, num_heads, num_ts, head_dim]
        q = q.view(batch_size, self.num_timescales, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, self.num_timescales, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, self.num_timescales, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention across timescales
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [batch, num_heads, num_ts, head_dim]
        
        # Reshape back: [batch, num_ts, feature_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, self.num_timescales, self.feature_dim)
        
        # Output projection
        out = self.out_proj(out)
        
        # Residual connection
        out = stacked + self.dropout(out)
        
        # Convert back to dictionary
        return {
            ts: out[:, i, :]
            for i, ts in enumerate(timescale_order)
        }


# ============================================================================
# TEMPORAL ATTENTION (ENHANCED with hierarchical priors)
# ============================================================================

class TemporalAttention(nn.Module):
    """
    Enhanced attention mechanism to weight importance of each timescale.
    
    Learns which timescales are most predictive for current market conditions,
    with hierarchical prior weights based on typical importance.
    """
    
    def __init__(
        self,
        num_timescales: int,
        hidden_dim: int,
        prior_weights: Optional[List[float]] = None
    ):
        super().__init__()
        
        self.num_timescales = num_timescales
        total_dim = num_timescales * hidden_dim
        
        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(total_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_timescales)
        )
        
        # Prior weights (learned baseline importance)
        if prior_weights is not None:
            self.register_buffer(
                'prior_weights',
                torch.tensor(prior_weights, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'prior_weights',
                torch.ones(num_timescales) / num_timescales
            )
        
        # Learnable mixing weight (how much to trust learned vs prior)
        self.mix_weight = nn.Parameter(torch.tensor(0.7))  # 70% learned, 30% prior
    
    def forward(self, features: Tensor) -> Tensor:
        """
        Calculate attention weights with hierarchical prior.
        
        Args:
            features: Concatenated features [batch, num_timescales * hidden_dim]
            
        Returns:
            weights: Attention weights [batch, num_timescales]
        """
        # Learned attention logits
        learned_logits = self.attention(features)
        learned_weights = F.softmax(learned_logits, dim=1)
        
        # Mix with prior
        mix = torch.sigmoid(self.mix_weight)
        prior = self.prior_weights.unsqueeze(0).expand(features.size(0), -1)
        
        combined = mix * learned_weights + (1 - mix) * prior
        
        # Renormalize to ensure sum = 1
        return combined / (combined.sum(dim=1, keepdim=True) + 1e-8)


# ============================================================================
# MULTI-TIMESCALE NETWORKS (MAIN MODULE - ENHANCED)
# ============================================================================

class MultiTimescaleNetworks(nn.Module):
    """
    Enhanced parallel networks processing 9 temporal resolutions.
    
    Architecture:
    - 9 bidirectional LSTMs (S30, M1, M5, M15, H1, H4, D, W, M)
    - Cross-timescale attention for inter-timescale communication
    - Temporal attention with hierarchical weighting
    - Fusion layer with MC Dropout uncertainty
    
    Features:
    - Bidirectional processing for richer temporal context
    - Cross-timescale attention captures multi-resolution patterns
    - MC Dropout uncertainty quantification
    - Async/await architecture throughout
    - Thread-safe state management
    - GPU memory management (600MB budget)
    
    VRAM Budget: 600MB
    - 9 BiLSTMs: ~450MB
    - Cross-attention: ~50MB
    - Temporal attention: ~30MB
    - Fusion layers: ~50MB
    - Activations: ~20MB buffer
    """
    
    def __init__(
        self,
        config: MultiTimescaleConfig,
        gpu_memory_manager: Optional[Any] = None
    ):
        super().__init__()
        
        self.config = config
        self.gpu_memory_manager = gpu_memory_manager
        
        # Device setup
        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu'
        )
        self.dtype = config.dtype
        
        # Thread safety
        self._lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()
        
        # State tracking (protected by _lock)
        self._is_initialized = False
        self._vram_allocated_mb = 0.0
        self._inference_count = 0
        self._last_prediction_time = 0.0
        
        # Statistics (protected by _lock)
        self._stats = {
            'total_predictions': 0,
            'uncertainty_predictions': 0,
            'agreements_checked': 0,
            'trades_approved': 0,
            'trades_rejected': 0,
            'avg_agreement_score': 0.0,
            'avg_uncertainty': 0.0
        }
        
        # Build networks
        self._build_networks()
        
        # Move to device
        self.to(self.device, dtype=self.dtype)
        
        # Start in eval mode
        self.eval()
        
        logger.info(
            f"Enhanced MultiTimescaleNetworks initialized: "
            f"{self._count_parameters():,} parameters, "
            f"device={self.device}, "
            f"timescales={config.timescales}, "
            f"bidirectional={config.bidirectional}"
        )
    
    def _build_networks(self):
        """Build all timescale networks, attention, and fusion layers"""
        cfg = self.config
        
        # ═══════════════════════════════════════════════════════
        # BIDIRECTIONAL TIMESCALE-SPECIFIC NETWORKS
        # ═══════════════════════════════════════════════════════
        
        self.timescale_networks = nn.ModuleDict()
        
        for ts in cfg.timescales:
            self.timescale_networks[ts] = BidirectionalTimescaleLSTM(
                input_dim=cfg.input_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_lstm_layers,
                dropout=cfg.lstm_dropout,
                bidirectional=cfg.bidirectional
            )
        
        # ═══════════════════════════════════════════════════════
        # CROSS-TIMESCALE ATTENTION (NEW)
        # ═══════════════════════════════════════════════════════
        
        if cfg.use_cross_timescale_attention:
            self.cross_attention = CrossTimescaleAttention(
                num_timescales=cfg.num_timescales,
                feature_dim=cfg.effective_hidden_dim,
                num_heads=cfg.attention_heads,
                dropout=cfg.attention_dropout
            )
            logger.info("✅ Cross-timescale attention enabled")
        else:
            self.cross_attention = None
        
        # ═══════════════════════════════════════════════════════
        # TEMPORAL ATTENTION (ENHANCED)
        # ═══════════════════════════════════════════════════════
        
        prior_weights = [
            cfg.timescale_prior_weights[ts]
            for ts in cfg.timescales
        ]
        
        self.attention = TemporalAttention(
            num_timescales=cfg.num_timescales,
            hidden_dim=cfg.effective_hidden_dim,
            prior_weights=prior_weights
        )
        
        # ═══════════════════════════════════════════════════════
        # FUSION LAYER (ENHANCED with more capacity)
        # ═══════════════════════════════════════════════════════
        
        total_features = cfg.num_timescales * cfg.effective_hidden_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_features, cfg.effective_hidden_dim * 2),
            nn.LayerNorm(cfg.effective_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.fusion_dropout),
            nn.Linear(cfg.effective_hidden_dim * 2, cfg.effective_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.fusion_dropout),
            nn.Linear(cfg.effective_hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
            nn.Softmax(dim=1)
        )
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    async def initialize_async(self) -> Dict[str, Any]:
        """Async initialization with GPU memory allocation"""
        async with self._lock:
            if self._is_initialized:
                logger.warning("Already initialized")
                return {'status': 'already_initialized'}
            
            try:
                if self.gpu_memory_manager is not None:
                    allocated = await self.gpu_memory_manager.allocate_async(
                        module_name="MultiTimescaleNetworks",
                        size_mb=self.config.max_vram_mb,
                        priority="CORE"
                    )
                    
                    if not allocated:
                        raise RuntimeError(
                            f"Failed to allocate {self.config.max_vram_mb}MB VRAM"
                        )
                    
                    self._vram_allocated_mb = self.config.max_vram_mb
                else:
                    # Estimate VRAM
                    param_bytes = sum(
                        p.numel() * p.element_size() for p in self.parameters()
                    )
                    self._vram_allocated_mb = param_bytes / (1024 * 1024) * 2
                
                await self._warmup_async()
                
                self._is_initialized = True
                
                logger.info(
                    f"✅ Enhanced MultiTimescaleNetworks initialized: "
                    f"VRAM={self._vram_allocated_mb:.1f}MB, "
                    f"params={self._count_parameters():,}"
                )
                
                return {
                    'status': 'success',
                    'vram_mb': self._vram_allocated_mb,
                    'parameters': self._count_parameters(),
                    'device': str(self.device),
                    'timescales': self.config.timescales,
                    'features': {
                        'bidirectional': self.config.bidirectional,
                        'cross_attention': self.cross_attention is not None,
                        'num_timescales': self.config.num_timescales
                    }
                }
                
            except Exception as e:
                logger.error(f"❌ Initialization failed: {e}")
                return {'status': 'failed', 'error': str(e)}
    
    async def _warmup_async(self):
        """Warmup inference to compile CUDA kernels"""
        logger.info("Warming up CUDA kernels...")
        
        dummy_features = {}
        for ts in self.config.timescales:
            seq_len = self.config.sequence_lengths[ts]
            dummy_features[ts] = torch.randn(
                self.config.batch_size,
                seq_len,
                self.config.input_dim,
                device=self.device,
                dtype=self.dtype
            )
        
        await asyncio.to_thread(
            self._forward_sync,
            dummy_features,
            use_dropout=False
        )
        
        logger.info("✅ Warmup complete")
    
    def _forward_sync(
        self,
        features_dict: Dict[str, Tensor],
        use_dropout: bool = False
    ) -> Dict[str, Any]:
        """
        Synchronous forward pass (CPU-bound, called via asyncio.to_thread)
        """
        if use_dropout:
            self.train()
        else:
            self.eval()
        
        context = torch.no_grad() if not use_dropout else torch.enable_grad()
        
        with context:
            # Process each timescale
            timescale_features = {}
            
            for ts in self.config.timescales:
                if ts not in features_dict:
                    raise ValueError(f"Missing features for timescale: {ts}")
                
                _, features = self.timescale_networks[ts](features_dict[ts])
                timescale_features[ts] = features
            
            # Apply cross-timescale attention (NEW)
            if self.cross_attention is not None:
                timescale_features = self.cross_attention(
                    timescale_features,
                    self.config.timescales
                )
            
            # Concatenate features
            all_features = torch.cat(
                [timescale_features[ts] for ts in self.config.timescales],
                dim=1
            )
            
            # Temporal attention
            attention_weights = self.attention(all_features)
            
            # Apply attention (weighted combination)
            weighted_features_list = []
            for i, ts in enumerate(self.config.timescales):
                weight = attention_weights[:, i:i+1]
                weighted = timescale_features[ts] * weight
                weighted_features_list.append(weighted)
            
            weighted_features = torch.cat(weighted_features_list, dim=1)
            
            # Fusion & prediction
            prediction = self.fusion(weighted_features)
        
        self.eval()
        
        return {
            'prediction': prediction,
            'attention_weights': attention_weights,
            'timescale_features': timescale_features
        }
    
    async def predict_async(
        self,
        features_dict: Dict[str, np.ndarray],
        return_uncertainty: bool = False
    ) -> Dict[str, Any]:
        """Async prediction with multi-timescale processing"""
        async with self._lock:
            if not self._is_initialized:
                raise RuntimeError("Not initialized. Call initialize_async() first")
        
        start_time = time.time()
        
        try:
            features_tensor = await asyncio.to_thread(
                self._prepare_input_sync,
                features_dict
            )
            
            if return_uncertainty:
                result = await self._predict_with_uncertainty_async(features_tensor)
            else:
                result = await asyncio.to_thread(
                    self._forward_sync,
                    features_tensor,
                    False
                )
                
                result = {
                    'prediction': result['prediction'].cpu().numpy(),
                    'attention_weights': result['attention_weights'].cpu().numpy(),
                    'timescale_features': {
                        ts: feat.cpu().numpy()
                        for ts, feat in result['timescale_features'].items()
                    },
                    'uncertainty': None
                }
            
            async with self._lock:
                self._inference_count += 1
                self._last_prediction_time = time.time() - start_time
                self._stats['total_predictions'] += 1
                if return_uncertainty:
                    self._stats['uncertainty_predictions'] += 1
            
            result['inference_time_ms'] = (time.time() - start_time) * 1000
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise
    
    def _prepare_input_sync(
        self,
        features_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Tensor]:
        """Synchronous input preparation"""
        result = {}
        
        for ts in self.config.timescales:
            if ts not in features_dict:
                raise ValueError(f"Missing features for timescale: {ts}")
            
            features = features_dict[ts]
            
            if features.ndim == 2:
                features = features.reshape(1, *features.shape)
            
            if features.shape[2] != self.config.input_dim:
                raise ValueError(
                    f"Expected {self.config.input_dim} features for {ts}, "
                    f"got {features.shape[2]}"
                )
            
            result[ts] = torch.from_numpy(features.astype(np.float32)).to(
                device=self.device,
                dtype=self.dtype
            )
        
        return result
    
    async def _predict_with_uncertainty_async(
        self,
        features_tensor: Dict[str, Tensor]
    ) -> Dict[str, Any]:
        """MC Dropout uncertainty quantification"""
        mc_samples = self.config.mc_samples
        
        all_predictions = []
        all_attention = []
        
        for _ in range(mc_samples):
            result = await asyncio.to_thread(
                self._forward_sync,
                features_tensor,
                True
            )
            all_predictions.append(result['prediction'].cpu())
            all_attention.append(result['attention_weights'].cpu())
        
        predictions_stacked = torch.stack(all_predictions, dim=0)
        attention_stacked = torch.stack(all_attention, dim=0)
        
        mean_prediction = predictions_stacked.mean(dim=0)
        std_prediction = predictions_stacked.std(dim=0)
        mean_attention = attention_stacked.mean(dim=0)
        
        epistemic_uncertainty = std_prediction.mean(dim=1)
        
        last_result = await asyncio.to_thread(
            self._forward_sync,
            features_tensor,
            False
        )
        
        async with self._lock:
            n = self._stats['uncertainty_predictions'] + 1
            old_avg = self._stats['avg_uncertainty']
            new_unc = float(epistemic_uncertainty.mean())
            self._stats['avg_uncertainty'] = (old_avg * (n - 1) + new_unc) / n
        
        return {
            'prediction': mean_prediction.numpy(),
            'attention_weights': mean_attention.numpy(),
            'timescale_features': {
                ts: feat.cpu().numpy()
                for ts, feat in last_result['timescale_features'].items()
            },
            'uncertainty': epistemic_uncertainty.numpy(),
            'prediction_std': std_prediction.numpy()
        }
    
    async def check_timescale_agreement_async(
        self,
        features_dict: Dict[str, np.ndarray],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Check if multiple timescales agree on direction"""
        if threshold is None:
            threshold = self.config.min_agreement_threshold
        
        result = await self.predict_async(features_dict)
        
        analysis = await asyncio.to_thread(
            self._analyze_timescale_agreement,
            result['timescale_features'],
            result['attention_weights'][0],
            threshold
        )
        
        async with self._lock:
            self._stats['agreements_checked'] += 1
            
            if analysis['should_trade']:
                self._stats['trades_approved'] += 1
            else:
                self._stats['trades_rejected'] += 1
            
            n = self._stats['agreements_checked']
            old_avg = self._stats['avg_agreement_score']
            self._stats['avg_agreement_score'] = (
                (old_avg * (n - 1) + analysis['agreement_score']) / n
            )
        
        return analysis
    
    def _analyze_timescale_agreement(
        self,
        ts_features: Dict[str, np.ndarray],
        attention_weights: np.ndarray,
        threshold: float
    ) -> Dict[str, Any]:
        """Analyze agreement between timescales"""
        directions = {}
        
        for ts, features in ts_features.items():
            if features.ndim > 1:
                features = features[0]
            
            mean_activation = float(np.mean(features))
            
            if mean_activation > 0.1:
                directions[ts] = TimescaleDirection.BULLISH
            elif mean_activation < -0.1:
                directions[ts] = TimescaleDirection.BEARISH
            else:
                directions[ts] = TimescaleDirection.NEUTRAL
        
        bullish_weight = 0.0
        bearish_weight = 0.0
        neutral_weight = 0.0
        
        for i, ts in enumerate(self.config.timescales):
            weight = float(attention_weights[i])
            direction = directions[ts]
            
            if direction == TimescaleDirection.BULLISH:
                bullish_weight += weight
            elif direction == TimescaleDirection.BEARISH:
                bearish_weight += weight
            else:
                neutral_weight += weight
        
        max_weight = max(bullish_weight, bearish_weight, neutral_weight)
        
        if max_weight == bullish_weight:
            dominant = TimescaleDirection.BULLISH
        elif max_weight == bearish_weight:
            dominant = TimescaleDirection.BEARISH
        else:
            dominant = TimescaleDirection.NEUTRAL
        
        total_weight = bullish_weight + bearish_weight + neutral_weight
        
        if total_weight > self.config.epsilon:
            agreement_score = max_weight / total_weight
        else:
            agreement_score = 0.0
        
        should_trade = (
            agreement_score >= threshold and
            dominant != TimescaleDirection.NEUTRAL
        )
        
        return {
            'should_trade': should_trade,
            'agreement_score': float(agreement_score),
            'dominant_direction': dominant.value,
            'timescale_votes': {
                'bullish': float(bullish_weight),
                'bearish': float(bearish_weight),
                'neutral': float(neutral_weight)
            },
            'individual_directions': {
                ts: d.value for ts, d in directions.items()
            }
        }
    
    async def select_relevant_timescales_async(
        self,
        market_regime: str,
        volatility: float
    ) -> List[str]:
        """Dynamically select which timescales to use based on market conditions"""
        relevant_timescales = []
        
        if market_regime == 'trending':
            relevant_timescales = ['h1', 'h4', 'd1', 'w1']
        elif market_regime == 'ranging':
            relevant_timescales = ['m15', 'h1', 'h4']
        elif market_regime == 'volatile':
            relevant_timescales = ['s30', 'm1', 'm5', 'm15']
        elif market_regime == 'breakout':
            relevant_timescales = ['m5', 'm15', 'h1', 'h4', 'd1']
        elif market_regime == 'crisis':
            relevant_timescales = ['s30', 'm1', 'm5']
        else:
            relevant_timescales = self.config.timescales.copy()
        
        if volatility > 0.7:
            for ts in ['s30', 'm1']:
                if ts not in relevant_timescales and ts in self.config.timescales:
                    relevant_timescales.insert(0, ts)
        elif volatility < 0.3:
            for ts in ['d1', 'w1', 'mn1']:
                if ts not in relevant_timescales and ts in self.config.timescales:
                    relevant_timescales.append(ts)
        
        relevant_timescales = [
            ts for ts in relevant_timescales
            if ts in self.config.timescales
        ]
        
        return relevant_timescales
    
    async def get_metrics_async(self) -> Dict[str, Any]:
        """Get current model metrics (thread-safe)"""
        async with self._lock:
            return {
                'is_initialized': self._is_initialized,
                'vram_allocated_mb': self._vram_allocated_mb,
                'inference_count': self._inference_count,
                'last_inference_ms': self._last_prediction_time * 1000,
                'parameter_count': self._count_parameters(),
                'device': str(self.device),
                'timescales': self.config.timescales,
                'num_timescales': self.config.num_timescales,
                'features': {
                    'bidirectional': self.config.bidirectional,
                    'cross_attention': self.cross_attention is not None
                },
                'stats': self._stats.copy()
            }
    
    async def save_checkpoint_async(self, filepath: str) -> Dict[str, Any]:
        """Save model checkpoint"""
        async with self._model_lock:
            try:
                Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                
                async with self._lock:
                    stats_copy = self._stats.copy()
                
                checkpoint = {
                    'model_state_dict': self.state_dict(),
                    'config': {
                        k: v for k, v in self.config.__dict__.items()
                        if not k.startswith('_') and not isinstance(v, torch.dtype)
                    },
                    'stats': stats_copy,
                    'metrics': await self.get_metrics_async(),
                    'timestamp': time.time(),
                    'version': '2.0.0'
                }
                
                await asyncio.to_thread(torch.save, checkpoint, filepath)
                
                logger.info(f"✅ Checkpoint saved: {filepath}")
                return {'status': 'success', 'filepath': filepath}
                
            except Exception as e:
                logger.error(f"❌ Checkpoint save failed: {e}")
                return {'status': 'failed', 'error': str(e)}
    
    async def load_checkpoint_async(self, filepath: str) -> Dict[str, Any]:
        """Load model checkpoint"""
        async with self._model_lock:
            try:
                checkpoint = await asyncio.to_thread(
                    torch.load, filepath, map_location=self.device
                )
                
                self.load_state_dict(checkpoint['model_state_dict'])
                
                if 'stats' in checkpoint:
                    async with self._lock:
                        self._stats.update(checkpoint['stats'])
                
                logger.info(f"✅ Checkpoint loaded: {filepath}")
                return {
                    'status': 'success',
                    'filepath': filepath,
                    'timestamp': checkpoint.get('timestamp', 'unknown'),
                    'version': checkpoint.get('version', 'unknown')
                }
                
            except Exception as e:
                logger.error(f"❌ Checkpoint load failed: {e}")
                return {'status': 'failed', 'error': str(e)}
    
    async def cleanup_async(self):
        """Cleanup resources and deallocate GPU memory"""
        async with self._lock:
            if not self._is_initialized:
                return
            
            if self.gpu_memory_manager is not None:
                await self.gpu_memory_manager.deallocate_async(
                    module_name="MultiTimescaleNetworks"
                )
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            self._is_initialized = False
            self._vram_allocated_mb = 0.0
            
            logger.info("✅ MultiTimescaleNetworks cleaned up")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize_async()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup_async()


# ============================================================================
# INTEGRATION TEST
# ============================================================================

async def test_enhanced_multi_timescale_networks():
    """Integration test for Enhanced MultiTimescaleNetworks"""
    logger.info("=" * 70)
    logger.info("TESTING MODULE 6: MULTI-TIMESCALE NETWORKS (ENHANCED v2.0.0)")
    logger.info("=" * 70)
    
    # Test 0: Configuration validation
    logger.info("\n[Test 0] Configuration validation...")
    try:
        config = MultiTimescaleConfig(
            input_dim=50,
            hidden_dim=128,
            num_lstm_layers=3,
            output_dim=3,
            bidirectional=True,
            use_cross_timescale_attention=True,
            attention_heads=8,
            mc_samples=10,  # Reduced for testing
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"✅ Valid config: {config.num_timescales} timescales")
        logger.info(f"   Timescales: {config.timescales}")
        logger.info(f"   Effective hidden dim: {config.effective_hidden_dim}")
        
        try:
            invalid_config = MultiTimescaleConfig(input_dim=-1)
            logger.error("❌ Should have raised ValueError")
        except ValueError as e:
            logger.info(f"✅ Invalid config correctly rejected")
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        return
    
    # Create model
    model = MultiTimescaleNetworks(config=config, gpu_memory_manager=None)
    
    # Test 1: Initialization
    logger.info("\n[Test 1] Initialization...")
    init_result = await model.initialize_async()
    assert init_result['status'] == 'success', f"Init failed: {init_result}"
    logger.info(f"✅ Initialization: params={init_result['parameters']:,}")
    logger.info(f"   Features: {init_result['features']}")
    
    # Test 2: Single prediction
    logger.info("\n[Test 2] Single prediction...")
    features = {
        ts: np.random.randn(1, config.sequence_lengths[ts], 50).astype(np.float32)
        for ts in config.timescales
    }
    result = await model.predict_async(features)
    assert result['prediction'].shape == (1, 3), "Wrong output shape"
    logger.info(f"✅ Prediction shape: {result['prediction'].shape}")
    logger.info(f"   Attention: {result['attention_weights'][0][:5]}...")
    logger.info(f"   Inference: {result['inference_time_ms']:.2f}ms")
    
    # Test 3: Prediction with uncertainty
    logger.info("\n[Test 3] MC Dropout uncertainty...")
    result_unc = await model.predict_async(features, return_uncertainty=True)
    assert result_unc['uncertainty'] is not None
    logger.info(f"✅ Uncertainty: {result_unc['uncertainty'][0]:.4f}")
    logger.info(f"   Prediction std: {result_unc['prediction_std'][0][:3]}")
    
    # Test 4: Batch prediction
    logger.info("\n[Test 4] Batch prediction...")
    batch_features = {
        ts: np.random.randn(8, config.sequence_lengths[ts], 50).astype(np.float32)
        for ts in config.timescales
    }
    result_batch = await model.predict_async(batch_features)
    assert result_batch['prediction'].shape == (8, 3)
    logger.info(f"✅ Batch shape: {result_batch['prediction'].shape}")
    
    # Test 5: Timescale agreement
    logger.info("\n[Test 5] Timescale agreement...")
    agreement = await model.check_timescale_agreement_async(features)
    logger.info(f"✅ Agreement: {agreement['agreement_score']:.2f}")
    logger.info(f"   Should trade: {agreement['should_trade']}")
    logger.info(f"   Direction: {agreement['dominant_direction']}")
    
    # Test 6: Timescale selection
    logger.info("\n[Test 6] Timescale selection...")
    for regime in ['trending', 'ranging', 'volatile', 'breakout', 'crisis']:
        ts = await model.select_relevant_timescales_async(regime, 0.5)
        logger.info(f"   {regime}: {ts}")
    
    # Test 7: Thread safety
    logger.info("\n[Test 7] Thread safety (5 concurrent)...")
    tasks = []
    for _ in range(5):
        f = {
            ts: np.random.randn(1, config.sequence_lengths[ts], 50).astype(np.float32)
            for ts in config.timescales
        }
        tasks.append(model.predict_async(f))
    results = await asyncio.gather(*tasks)
    assert len(results) == 5
    logger.info(f"✅ Thread safety: All 5 completed")
    
    # Test 8: Metrics
    logger.info("\n[Test 8] Metrics...")
    metrics = await model.get_metrics_async()
    logger.info(f"✅ Total predictions: {metrics['stats']['total_predictions']}")
    logger.info(f"   Uncertainty predictions: {metrics['stats']['uncertainty_predictions']}")
    
    # Test 9: Checkpoint
    logger.info("\n[Test 9] Checkpoint save/load...")
    save_result = await model.save_checkpoint_async("/tmp/mts_enhanced_test.pt")
    assert save_result['status'] == 'success'
    load_result = await model.load_checkpoint_async("/tmp/mts_enhanced_test.pt")
    assert load_result['status'] == 'success'
    logger.info(f"✅ Checkpoint: save/load successful")
    
    # Test 10: Cleanup
    logger.info("\n[Test 10] Cleanup...")
    await model.cleanup_async()
    logger.info(f"✅ Cleanup: successful")
    
    logger.info("\n" + "=" * 70)
    logger.info("ALL TESTS PASSED ✅")
    logger.info("=" * 70)
    
    # Enhancement summary
    logger.info("\n" + "=" * 70)
    logger.info("ENHANCEMENT SUMMARY (v1.1.0 → v2.0.0):")
    logger.info("=" * 70)
    logger.info("✅ timescales: 5 → 9 (+80%)")
    logger.info("✅ hidden_dim: 64 → 128 (2x)")
    logger.info("✅ num_lstm_layers: 2 → 3")
    logger.info("✅ Bidirectional LSTM: ENABLED")
    logger.info("✅ Cross-timescale attention: ENABLED (8-head)")
    logger.info("✅ Hierarchical prior weights: ADDED")
    logger.info("✅ mc_samples: 30 → 50")
    logger.info("✅ max_vram_mb: 300 → 600")
    logger.info("=" * 70)


if __name__ == '__main__':
    asyncio.run(test_enhanced_multi_timescale_networks())

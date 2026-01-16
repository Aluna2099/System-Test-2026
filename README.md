# 🚀 Liquid Neural SREK Trading System v4.0

**AI-Powered Autonomous Forex Trading System**

An advanced neural network ensemble system for automated forex trading, featuring liquid neural ODEs, self-refining evolutionary knowledge agents (SREKs), and meta-learning for rapid adaptation.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Quick Start (5 Minutes)](#quick-start)
4. [Detailed Installation](#detailed-installation)
5. [Configuration Guide](#configuration-guide)
6. [Getting OANDA API Access](#getting-oanda-api-access)
7. [Running the System](#running-the-system)
8. [Trading Modes](#trading-modes)
9. [Understanding the Modules](#understanding-the-modules)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)

---

## 🎯 Overview

The Liquid Neural SREK Trading System is a state-of-the-art AI trading platform that combines:

- **400 SREK Agents**: Self-refining evolutionary knowledge agents
- **Liquid Neural ODEs**: Continuous-time neural networks
- **Multi-Timescale LSTMs**: Capturing patterns from minutes to weeks
- **Transformer Validators**: Second-opinion validation for high-confidence trades
- **Graph Neural Networks**: Currency correlation analysis
- **Attention Memory**: Long-term pattern storage with neural retrieval
- **Meta-Learning**: Rapid adaptation to new market conditions

### Expected Performance
- Target Win Rate: 65-75%
- Expected Sharpe Ratio: 1.5-2.5
- Daily Returns: 0.1-0.5%
- Max Drawdown: <10%

> ⚠️ **DISCLAIMER**: Past performance does not guarantee future results. Trading forex involves significant risk. Only trade with money you can afford to lose.

---

## 💻 System Requirements

### Minimum Requirements
| Component | Requirement |
|-----------|-------------|
| **OS** | Windows 10/11, Linux (Ubuntu 20.04+), macOS 12+ |
| **CPU** | 4 cores, 3.0 GHz |
| **RAM** | 8 GB |
| **GPU** | NVIDIA GTX 1060 6GB (CUDA capable) |
| **Storage** | 20 GB SSD |
| **Python** | 3.10 or higher |
| **Internet** | Stable connection |

### Recommended (Optimal Performance)
| Component | Requirement |
|-----------|-------------|
| **CPU** | 8+ cores, 3.5 GHz |
| **RAM** | 16+ GB |
| **GPU** | NVIDIA RTX 3060 6GB or better |
| **Storage** | 50+ GB NVMe SSD |

---

## ⚡ Quick Start

### 1. Clone/Download the System
```bash
# Clone from repository (if applicable)
git clone https://github.com/yourusername/liquid-neural-trading.git
cd liquid-neural-trading

# OR just extract the downloaded files to a folder
```

### 2. Run the Installer
```bash
python install.py
```

### 3. Configure OANDA Credentials
Edit `config/config.yaml`:
```yaml
oanda:
  account_id: "YOUR_ACCOUNT_ID"
  api_token: "YOUR_API_TOKEN"
  environment: "practice"  # Start with practice!
```

### 4. Start the System
```bash
python main.py
```

### 5. Select a Mode
- **[1] Historical Training** - First time? Start here!
- **[2] Paper Trading** - Practice with fake money
- **[3] Live Trading** - Real money (after practicing!)

---

## 📦 Detailed Installation

### Step 1: Install Python 3.10+

#### Windows
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run installer, **CHECK "Add Python to PATH"**
3. Click "Install Now"

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

#### macOS
```bash
# Using Homebrew
brew install python@3.10
```

### Step 2: Install NVIDIA CUDA (GPU Support)

#### Windows
1. Download CUDA Toolkit 11.8 from [NVIDIA](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Run installer with default options
3. Restart computer

#### Linux
```bash
# CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

### Step 3: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv trading_env

# Activate it
# Windows:
trading_env\Scripts\activate
# Linux/macOS:
source trading_env/bin/activate
```

### Step 4: Run Installer

```bash
python install.py
```

The installer will:
- ✅ Check system requirements
- ✅ Create directory structure
- ✅ Install Python dependencies
- ✅ Initialize database
- ✅ Create desktop shortcut

### Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA: True
```

---

## ⚙️ Configuration Guide

### Configuration File Location
```
trading_system/
└── config/
    └── config.yaml  ← Edit this file!
```

### Essential Settings

```yaml
# OANDA API (REQUIRED)
oanda:
  account_id: "101-001-XXXXXXXX-001"  # Your account ID
  api_token: "your-token-here"        # Your API token
  environment: "practice"              # "practice" or "live"

# Trading Parameters
trading:
  pairs:                              # Currency pairs to trade
    - "EUR_USD"
    - "GBP_USD"
    - "USD_JPY"
  max_risk_per_trade: 0.02            # 2% risk per trade
  max_daily_loss: 0.05                # 5% daily loss limit
  min_entry_confidence: 0.75          # Minimum confidence to trade

# GPU Settings
gpu:
  device: "cuda"                      # "cuda" or "cpu"
  target_vram_utilization: 0.80       # 80% VRAM usage
```

### Advanced Settings (Optional)

```yaml
# Training Configuration
training:
  historical_data_months: 6           # Months of data to download
  epochs: 100                         # Training epochs
  batch_size: 32                      # Batch size

# Logging
logging:
  level: "INFO"                       # DEBUG, INFO, WARNING, ERROR
  log_to_file: true
```

---

## 🔑 Getting OANDA API Access

### Step 1: Create OANDA Account

1. Go to [www.oanda.com](https://www.oanda.com)
2. Click "Open Account"
3. Select "Demo Account" (free, no deposit needed)
4. Complete registration

### Step 2: Get API Credentials

1. Log into your OANDA account
2. Go to **Account** → **Manage API Access**
3. Click **Generate Token**
4. Copy the following:
   - **Account ID**: Found in account settings (format: 101-001-XXXXXXXX-001)
   - **API Token**: The generated token (keep this SECRET!)

### Step 3: Add to Configuration

Edit `config/config.yaml`:
```yaml
oanda:
  account_id: "101-001-12345678-001"  # Your actual account ID
  api_token: "abcd1234-efgh5678-xxxx"  # Your actual token
  environment: "practice"              # Keep as "practice" initially!
```

### ⚠️ Security Notes
- NEVER share your API token
- NEVER commit config.yaml to version control
- Add `config.yaml` to `.gitignore`

---

## ▶️ Running the System

### Method 1: Command Line (Recommended)

```bash
# Interactive mode selection
python main.py

# Direct mode selection
python main.py --mode historical
python main.py --mode paper
python main.py --mode live
```

### Method 2: Desktop Shortcut

Double-click the desktop shortcut created during installation:
- **Windows**: "Liquid Neural Trading.lnk"
- **Linux**: "liquid-neural-trading.desktop"
- **macOS**: "Liquid Neural Trading.command"

### Method 3: Scripts

```bash
# Windows
scripts\start_trading_system.bat

# Linux/macOS
./scripts/start_trading_system.sh
```

---

## 📊 Trading Modes

### 1. Historical Training Mode

**Purpose**: Download data and train the AI models

**When to use**: 
- First time setup
- After major market changes
- Weekly/monthly retraining

**What it does**:
1. Downloads 6 months of historical data from OANDA
2. Calculates 50 technical features
3. Trains all neural network modules
4. Backtests strategies
5. Saves model checkpoints

**Command**:
```bash
python main.py --mode historical
```

**Duration**: 2-8 hours depending on hardware

---

### 2. Paper Trading Mode

**Purpose**: Practice trading with simulated money

**When to use**:
- After initial training
- Testing new strategies
- Learning the system

**What it does**:
1. Connects to real-time OANDA data
2. Makes predictions using trained models
3. Simulates trades (no real money)
4. Tracks performance metrics
5. Continues learning from results

**Command**:
```bash
python main.py --mode paper
```

**Duration**: Runs continuously until stopped (Ctrl+C)

---

### 3. Live Trading Mode

**Purpose**: Trade with real money

**When to use**:
- ONLY after extensive paper trading
- When confident in system performance
- With money you can afford to lose

**What it does**:
1. Connects to OANDA live API
2. Executes real trades
3. Manages positions automatically
4. Implements risk controls

**Command**:
```bash
python main.py --mode live
```

**⚠️ WARNING**: This mode uses REAL MONEY. Start with small amounts!

---

## 🧠 Understanding the Modules

### Module Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRADING SYSTEM                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │  Module 1   │   │  Module 2   │   │  Module 6   │       │
│  │ Neural ODE  │   │ 400 SREKs   │   │   LSTMs     │       │
│  │  1000 MB    │   │  1200 MB    │   │   500 MB    │       │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘       │
│         │                 │                 │               │
│         └────────────┬────┴────────────────┘               │
│                      │                                      │
│              ┌───────▼───────┐                             │
│              │   Module 18   │                             │
│              │ Meta-Learner  │                             │
│              │   400 MB      │                             │
│              └───────┬───────┘                             │
│                      │                                      │
│              ┌───────▼───────┐                             │
│              │   Module 9    │                             │
│              │  Transformer  │                             │
│              │    700 MB     │                             │
│              └───────┬───────┘                             │
│                      │                                      │
│              ┌───────▼───────┐                             │
│              │   DECISION    │                             │
│              │  BUY/SELL/    │                             │
│              │    HOLD       │                             │
│              └───────────────┘                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Module Descriptions

| Module | Purpose | VRAM |
|--------|---------|------|
| **Module 1**: Liquid Neural ODE | Continuous-time market dynamics | 1000 MB |
| **Module 2**: Meta-SREK Population | 400 evolutionary trading agents | 1200 MB |
| **Module 6**: Multi-Timescale LSTM | M1, M5, M15, H1, H4, D, W patterns | 500 MB |
| **Module 9**: Transformer Validator | Validates high-confidence trades | 700 MB |
| **Module 13**: Correlation GNN | Currency pair relationships | 150 MB |
| **Module 17**: Attention Memory | Long-term pattern storage | 700 MB |
| **Module 18**: Ensemble Meta-Learner | Dynamic ensemble weighting | 400 MB |
| **Module 16**: GPU Memory Manager | Coordinates VRAM allocation | - |

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "CUDA not available"

**Problem**: PyTorch can't access GPU

**Solutions**:
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### 2. "Out of memory"

**Problem**: GPU VRAM exhausted

**Solutions**:
- Close other GPU applications
- Reduce batch_size in config.yaml
- Lower model dimensions in advanced settings

#### 3. "OANDA API error"

**Problem**: Can't connect to OANDA

**Solutions**:
- Verify account_id and api_token in config.yaml
- Check internet connection
- Ensure OANDA servers aren't in maintenance

#### 4. "Module not found"

**Problem**: Missing module files

**Solutions**:
```bash
# Check modules directory
ls modules/

# Run installer again
python install.py
```

### Getting Help

1. Check the logs: `data/logs/trading_YYYYMMDD.log`
2. Run diagnostics: `python -c "import main; main.check_system()"`
3. Open an issue on GitHub

---

## ❓ FAQ

### Q: How much money do I need to start?
**A**: Start with $1,000-$5,000 minimum on a demo account. Only move to live trading after consistent paper trading profits.

### Q: How long does training take?
**A**: Initial training takes 2-8 hours. The system continues learning during paper/live trading.

### Q: Can I use CPU instead of GPU?
**A**: Yes, set `device: "cpu"` in config.yaml, but expect 10-20x slower performance.

### Q: Is this guaranteed to make money?
**A**: NO. Trading involves risk. Past performance doesn't guarantee future results.

### Q: What currency pairs are supported?
**A**: Any forex pair available on OANDA. Default pairs are major pairs (EUR/USD, GBP/USD, etc.)

### Q: Can I modify the trading strategy?
**A**: Yes, all parameters are configurable in config.yaml. The neural networks adapt automatically.

### Q: How do I stop the system?
**A**: Press Ctrl+C. The system will gracefully close positions and save state.

---

## 📄 License

This software is provided for educational and research purposes. Use at your own risk.

---

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- OANDA for the trading API
- torchdiffeq for neural ODE implementations
- The quantitative finance research community

---

**Happy Trading! 📈**

*Remember: Always start with paper trading and never risk more than you can afford to lose.*

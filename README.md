# 🧠 Advanced Multi-Pair & Hyperopt-Ready Crypto Trading Bot

A production-grade algorithmic trading bot built on **Freqtrade** for Perpetual Futures. Designed with **non-colinear indicators (StochRSI + CVD)**, **dynamic ATR-based risk management**, and **Bayesian Hyperopt optimization** per individual asset.

---

## Architecture Overview

```
 ┌──────────────────────────────┐
 │   Freqtrade Core Orchestrator│
 └──────────────┬───────────────┘
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
 ┌──────────┐ ┌──────────┐ ┌──────────┐
 │BTC/USDT  │ │SOL/USDT  │ │PEPE/USDT │
 │ Loop     │ │ Loop     │ │ Loop     │
 ├──────────┤ ├──────────┤ ├──────────┤
 │TF: 15m   │ │TF: 5m    │ │TF: 1m    │
 │Stoch: 9  │ │Stoch: 14 │ │Stoch: 7  │
 │ATR: 1.5x │ │ATR: 2.3x │ │ATR: 3.1x │
 └──────────┘ └──────────┘ └──────────┘
```

### Asset Profile Adaptation

| Profile | Regime | Logic |
|---|---|---|
| **High Liquid / Low Beta** (BTC) | Trend Following | Breakout mechanisms, wider ATR, longer CVD horizon |
| **High Vol / Mid-Cap** (SOL, AVAX) | Mean Reversion | Shorter timeframes, reactive StochRSI, tighter ATR |

### Indicator Set (Non-Colinear)

1. **Stochastic RSI** — Momentum / exhaustion detection (per-pair tuned 5–25 periods)
2. **CVD (Cumulative Volume Delta)** — Order-flow approximation, prevents traps via divergence detection
3. **ATR** — Dynamic trailing stop and position sizing

### Regime Filter: $\Delta CVD_{1H}$

- **$|\Delta CVD| > \text{threshold}$** → **Trend Regime** → trade structural momentum
- **$|\Delta CVD| \leq \text{threshold}$** → **Range Regime** → trade StochRSI mean reversion

---

## Project Structure

```
freqtrade-adaptive-bot/
├── config.json                      # Exchange & runtime config
├── pair_params.default.json         # Default per-pair parameters (fallback)
├── user_data/
│   └── strategies/
│       └── MultiPairAdaptiveStrategy.py   # Core strategy
├── scripts/
│   ├── download_data.sh             # Historical data ingestion
│   ├── run_hyperopt.sh              # Bayesian optimization runner
│   └── deploy_params.sh             # Deploy hyperopt output → pair_params.json
├── docs/
│   └── architecture.md              # Full system design doc
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Freqtrade](https://www.freqtrade.io/en/stable/) installed
- Binance (or compatible exchange) account

### Quick Start

```bash
# 1. Clone
git clone https://github.com/DylanAlberto/freqtrade-adaptive-bot.git
cd freqtrade-adaptive-bot

# 2. Symlink strategy into your freqtrade user_data
ln -s "$(pwd)/user_data/strategies" ~/freqtrade/user_data/strategies

# 3. Download historical data
bash scripts/download_data.sh

# 4. Run hyperopt (per pair recommendation)
bash scripts/run_hyperopt.sh

# 5. Run live / dry-run
freqtrade trade --config config.json --strategy MultiPairAdaptiveStrategy
```

---

## Hyperopt Workflow

```
1. Data Ingestion      →  freqtrade download-data ...
2. Bayesian Optim.     →  freqtrade hyperopt --epochs 500
3. Deploy Params       →  Copy JSON results → pair_params.json
4. Strategy loads      →  Reads pair_params.json at init
```

The Bayesian optimizer (via `scikit-optimize`) targets **Sharpe Ratio** / **Sortino Ratio** and writes optimal variables per asset into `pair_params.json`.

---

## Risk Management

- **Margin**: Isolated per position
- **Stop Loss**: Dynamic ATR-based (per-pair optimized multiplier)
- **Orders**: Limit entry/exit with order book pricing
- **Trailing Stop**: Enabled globally, per-pair ATR distance

---

## License

MIT

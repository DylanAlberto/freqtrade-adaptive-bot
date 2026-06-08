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
| **High Liquid / Low Beta** (BTC, ETH) | Trend Following | Breakout, wider ATR, longer CVD horizon |
| **High Vol / Mid-Cap** (SOL, AVAX) | Mean Reversion | Shorter timeframes, reactive StochRSI, tighter ATR |
| **Meme / Micro-Cap** (PEPE) | Mean Reversion | 1m base, very tight StochRSI, wide ATR buffer |

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
├── config.json                      # Live / dry-run config
├── config.backtest.json             # Backtest-specific config (5 pairs)
├── pair_params.default.json         # Default per-pair parameters (fallback)
│
├── user_data/
│   └── strategies/
│       └── MultiPairAdaptiveStrategy.py   # Core strategy (v3 interface)
│
├── scripts/
│   ├── download_data.sh             # Historical data ingestion
│   ├── run_backtest.sh              # Multi-mode backtest runner
│   ├── backtest_sweep.sh            # Parameter sensitivity sweeps
│   ├── run_hyperopt.sh              # Bayesian optimization runner
│   └── deploy_params.sh             # Deploy hyperopt output → pair_params.json
│
├── docs/
│   └── architecture.md              # Full system design doc
│
└── README.md
```

---

## Backtesting Workflow

### 1. Download Historical Data

```bash
bash scripts/download_data.sh
```

Downloads 180 days of futures data (1m, 5m, 15m, 1h) for all configured pairs.

### 2. Run Backtest

```bash
# Quick check (last ~500 candles)
MODE=quick bash scripts/run_backtest.sh

# Monthly
MODE=month bash scripts/run_backtest.sh

# Full 180-day
MODE=full bash scripts/run_backtest.sh

# Custom date range
MODE=range RANGE=20260101-20260607 bash scripts/run_backtest.sh
```

### 3. Parameter Sensitivity Sweep

```bash
bash scripts/backtest_sweep.sh
```

Tests variations of `max_open_trades` and `trailing_stop_positive` to measure sensitivity.

### 4. Hyperopt (per-pair)

```bash
# Optimise all pairs
bash scripts/run_hyperopt.sh

# Optimise a single pair
PAIR=BTC/USDT:USDT bash scripts/run_hyperopt.sh
```

### 5. Deploy Optimised Parameters

```bash
bash scripts/deploy_params.sh
```

Copies defaults → `pair_params.json`. Edit this file with your hyperopt results.

---

## Pairs Currently Configured

| Pair | Timeframe | Profile | Stop ATR Mult |
|---|---|---|---|
| BTC/USDT:USDT | 15m | Trend | 1.5× |
| ETH/USDT:USDT | 15m | Trend | 1.8× |
| SOL/USDT:USDT | 5m | Mean Reversion | 2.3× |
| AVAX/USDT:USDT | 5m | Mean Reversion | 2.5× |
| PEPE/USDT:USDT | 1m | Mean Reversion | 3.1× |

---

## Risk Management

- **Margin**: Isolated per position
- **Stop Loss**: Dynamic ATR-based (per-pair optimized multiplier, capped 15 %)
- **Orders**: Limit entry/exit with order book pricing
- **Trailing Stop**: Enabled, activates after 1 % profit, trails at 0.5 %
- **Max Open Trades**: 3 (configurable)

---

## Strategy Entry Logic Details

| Regime | Direction | Condition |
|---|---|---|
| **Trend** | Long | StochRSI crosses above oversold + CVD delta > 0 |
| **Trend** | Short | StochRSI crosses below overbought + CVD delta < 0 |
| **Range** | Long | StochRSI crosses above oversold (mean reversion) |
| **Range** | Short | StochRSI crosses below overbought (mean reversion) |

Exit triggers: StochRSI reaches opposite extreme OR CVD-close divergence detected.

---

## License

MIT

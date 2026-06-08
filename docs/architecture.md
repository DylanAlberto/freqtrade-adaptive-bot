# System Architecture

## Overview

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

Each pair runs with its own optimised parameter set. The strategy reads
`pair_params.json` at initialisation and applies the correct parameters
to each data stream.

---

## Asset Profile Classification

### High Liquidity / Low Beta (e.g., BTC/USDT)

- **Regime:** Trend Following / Continuation
- **Characteristics:** Thick order book, low relative volatility
- **Strategy:** Breakout mechanisms, wider ATR trailing stops, longer-horizon CVD trend tracking
- **Timeframe:** 15m base

### High Volatility / Mid-Cap (e.g., SOL/USDT)

- **Regime:** Mean Reversion / Range Trading
- **Characteristics:** Thinner book, wider swings
- **Strategy:** Shorter timeframes (5m), highly reactive StochRSI boundaries, tighter ATR buffers
- **Timeframe:** 5m base

---

## Indicator Mathematics

### 1. Stochastic RSI (Momentum)

$$
\text{RSI} = 100 - \frac{100}{1 + \frac{\text{AvgGain}}{\text{AvgLoss}}}
$$

$$
\text{StochRSI} = \frac{\text{RSI} - \min(\text{RSI}, n)}{\max(\text{RSI}, n) - \min(\text{RSI}, n)}
$$

Smoothed with a 3-period SMA (%K) and another 3-period SMA (%D).

### 2. Cumulative Volume Delta (Order Flow)

$$
\delta_i = (\text{close}_i - \text{open}_i) \times \text{volume}_i
$$

$$
\text{CVD}_i = \sum_{j=0}^{i} \delta_j
$$

**Divergence detection:**
- **Bullish divergence:** Price ↓, CVD ↑ → absorption, potential reversal up
- **Bearish divergence:** Price ↑, CVD ↓ → distribution, potential reversal down

### 3. Regime Filter ($\Delta$CVD)

$$
\Delta\text{CVD}_{1H} = \text{CVD}_{\text{current}} - \text{CVD}_{t-n}
$$

- $|\Delta\text{CVD}| > \text{threshold}$ → **Trend Regime**
- $|\Delta\text{CVD}| \leq \text{threshold}$ → **Range Regime**

### 4. Dynamic ATR Stop Loss

$$
\text{ATR} = \text{SMA}\left(\max(H-L, |H-C_{\text{prev}}|, |L-C_{\text{prev}}|), n\right)
$$

$$
\text{StopLoss}_\% = -\min\left(\frac{\text{ATR} \times \text{multiplier}}{\text{entry\_price}},\ 0.15\right)
$$

---

## Hyperopt Workflow

```
1. Data Ingestion
   └── freqtrade download-data --pairs ... --timeframes 1m 5m 15m 1h --days 180

2. Bayesian Optimization
   └── freqtrade hyperopt --epochs 500 --hyperopt-loss SharpeHyperOptLoss

3. Parameter Deployment
   └── Extract best params → write to pair_params.json

4. Live Trading
   └── Strategy loads pair_params.json → per-pipe parameters applied
```

### Optimization Space

| Parameter | Range | Step |
|---|---|---|
| StochRSI Period | 5 – 25 | 1 |
| StochRSI Oversold | 10 – 35 | 5 |
| StochRSI Overbought | 65 – 90 | 5 |
| CVD Threshold | 0.05 – 0.50 | 0.01 |
| ATR Stop Multiplier | 1.0 – 4.0 | 0.1 |

---

## Risk Management

- **Margin:** Isolated per position (no cross-margin risk)
- **Stoploss:** Dynamic ATR-based, capped at 15%
- **Entry orders:** Limit (bid-side) with order book pricing
- **Exit orders:** Limit (ask-side) with order book pricing
- **Emergency exit:** Market order
- **Trailing stop:** Yes, activates after 1% profit, trails at 0.5%

---

## Config Reference

See `config.json` for the full configuration. Key fields:

| Field | Value | Notes |
|---|---|---|
| `trading_mode` | `futures` | Perpetual swaps |
| `margin_mode` | `isolated` | Per-position isolation |
| `dry_run` | `true` | Paper trading by default |
| `stake_amount` | `unlimited` | Dynamic per position size |
| `max_open_trades` | `3` | Max concurrent positions |

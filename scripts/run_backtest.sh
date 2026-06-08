#!/usr/bin/env bash
# ==============================================================
# Backtest Runner — Freqtrade Multi-Pair Adaptive Strategy
# ==============================================================
#
# Runs backtests with optional parameter overrides and produces
# JSON + HTML reports.
#
# Usage:
#   bash scripts/run_backtest.sh                          # default
#   MODE=quick    bash scripts/run_backtest.sh            # fast (~100 candles)
#   MODE=full     bash scripts/run_backtest.sh            # full 180-day
#   MODE=range    RANGE=20260301-20260607 bash scripts/run_backtest.sh
#   EXPORT=signals  bash scripts/run_backtest.sh          # incl. signal export
# ==============================================================

set -euo pipefail

# ---- Config ---------------------------------------------------
STRATEGY="${STRATEGY:-MultiPairAdaptiveStrategy}"
CONFIG="${CONFIG:-config.backtest.json}"
EXCHANGE="${EXCHANGE:-binance}"
MODE="${MODE:-default}"
RANGE="${RANGE:-20250901-20260607}"
DATA_DIR="${DATA_DIR:-user_data/data}"
EXPORT="${EXPORT:-trades}"
TIMEFRAMES="${TIMEFRAMES:-5m 15m}"

# ---- Mode presets ---------------------------------------------
case "${MODE}" in
    quick)
        echo "[MODE] Quick — last 500 candles (~1.7 days at 5m)"
        EXTRA="--timerange=20260601-"
        ;;
    week)
        echo "[MODE] Week"
        EXTRA="--timerange=20260531-"
        ;;
    month)
        echo "[MODE] Month"
        EXTRA="--timerange=20260507-"
        ;;
    full)
        echo "[MODE] Full — 180 days"
        EXTRA="--timerange=${RANGE}"
        ;;
    range)
        echo "[MODE] Custom range: ${RANGE}"
        EXTRA="--timerange=${RANGE}"
        ;;
    *)
        echo "[MODE] Default — 3 months"
        EXTRA="--timerange=20260307-"
        ;;
esac

# ---- Export flags ---------------------------------------------
case "${EXPORT}" in
    signals)
        EXPORT_FLAGS="--export signals --exportfilename user_data/backtest_results/backtest_signals.json"
        ;;
    trades)
        EXPORT_FLAGS="--export trades --exportfilename user_data/backtest_results/backtest_trades.json"
        ;;
    none)
        EXPORT_FLAGS=""
        ;;
    *)
        EXPORT_FLAGS="--export trades --exportfilename user_data/backtest_results/backtest_trades.json"
        ;;
esac

# ---- Ensure output dirs exist ---------------------------------
mkdir -p user_data/backtest_results
mkdir -p user_data/plot

# ---- Print banner ---------------------------------------------
echo ""
echo "=============================================="
echo "  Backtest — ${STRATEGY}"
echo "  Mode:     ${MODE}"
echo "  Timerange:${EXTRA}"
echo "  Export:   ${EXPORT}"
echo "  Pairs:    see ${CONFIG}"
echo "=============================================="
echo ""

# ---- Run backtest --------------------------------------------
CMD="freqtrade backtesting \
    --config ${CONFIG} \
    --strategy ${STRATEGY} \
    --datadir ${DATA_DIR} \
    --timeframe-detail 1m \
    --breakdown week \
    ${EXTRA} \
    ${EXPORT_FLAGS}"

echo "${CMD}"
echo ""
eval "${CMD}"

echo ""
echo "=== Backtest complete ==="
echo "Results saved to user_data/backtest_results/"

#!/usr/bin/env bash
# ==============================================================
# Bayesian Hyperopt — Multi-Pair Parameter Optimization
# ==============================================================
# Runs Freqtrade's hyperopt engine with Bayesian optimization
# (scikit-optimize) targeting Sharpe Ratio.
#
# Usage:
#   # Single pair
#   PAIR=BTC/USDT:USDT bash scripts/run_hyperopt.sh
#
#   # All pairs (sequential)
#   bash scripts/run_hyperopt.sh
# ==============================================================

set -euo pipefail

STRATEGY="${STRATEGY:-MultiPairAdaptiveStrategy}"
EXCHANGE="${EXCHANGE:-binance}"
EPOCHS="${EPOCHS:-500}"
LOSS="${LOSS:-SharpeHyperOptLoss}"
SPACES="${SPACES:-all}"
TIMERANGE="${TIMERANGE:-20251001-20260607}"
DATA_DIR="${DATA_DIR:-user_data/data}"
CONFIG="${CONFIG:-config.json}"

# Default pairs to iterate
ALL_PAIRS=("BTC/USDT:USDT" "SOL/USDT:USDT" "ETH/USDT:USDT")

# If a specific PAIR is set, only optimize that one
if [ -n "${PAIR:-}" ]; then
    TARGET_PAIRS=("${PAIR}")
else
    TARGET_PAIRS=("${ALL_PAIRS[@]}")
fi

echo "=============================================="
echo " Hyperopt Engine — Bayesian Optimization"
echo " Strategy:  ${STRATEGY}"
echo " Loss func: ${LOSS}"
echo " Epochs:    ${EPOCHS}"
echo " Spaces:    ${SPACES}"
echo " Pairs:     ${TARGET_PAIRS[*]}"
echo " Timerange: ${TIMERANGE}"
echo "=============================================="

for pair in "${TARGET_PAIRS[@]}"; do
    echo ""
    echo "--- Optimizing ${pair} ---"

    freqtrade hyperopt \
        --config "${CONFIG}" \
        --strategy "${STRATEGY}" \
        --hyperopt-loss "${LOSS}" \
        --spaces "${SPACES}" \
        --epochs "${EPOCHS}" \
        --timerange "${TIMERANGE}" \
        --datadir "${DATA_DIR}" \
        --export-filename "hyperopt_results_$(echo ${pair} | tr '/' '_' | tr ':' '_').json"
done

echo ""
echo "Hyperopt complete for all pairs."
echo "Next step: run 'bash scripts/deploy_params.sh' to copy results."

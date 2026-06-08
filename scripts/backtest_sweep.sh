#!/usr/bin/env bash
# ==============================================================
# Batch Backtest — Iterate over parameter variations
# ==============================================================
#
# Runs multiple backtests sweeping one parameter at a time to
# measure sensitivity. Useful before launching a full hyperopt.
#
# Usage:
#   bash scripts/backtest_sweep.sh
# ==============================================================

set -euo pipefail

STRATEGY="${STRATEGY:-MultiPairAdaptiveStrategy}"
CONFIG="${CONFIG:-config.backtest.json}"
DATA_DIR="${DATA_DIR:-user_data/data}"
TIMERANGE="${TIMERANGE:-20260301-20260607}"

mkdir -p user_data/backtest_results/sweep

run_sweep() {
    local label="$1"
    shift
    echo ""
    echo "--- Sweep: ${label} ---"
    echo "Params: $@"
    echo ""

    freqtrade backtesting \
        --config "${CONFIG}" \
        --strategy "${STRATEGY}" \
        --datadir "${DATA_DIR}" \
        --timerange "${TIMERANGE}" \
        --export trades \
        --exportfilename "user_data/backtest_results/sweep/${label// /_}.json" \
        --breakdown week \
        "$@"

    echo ""
}

# ---- Example: Sweep max_open_trades ---------------------------
for n in 1 2 3 5; do
    run_sweep "max_${n}trades" --max-open-trades "${n}"
done

# ---- Example: Sweep trailing_stop_positive --------------------
for ts in 0.003 0.005 0.008 0.01; do
    run_sweep "trail_${ts}" \
        --custom "trailing_stop_positive=${ts}"
done

echo ""
echo "=== All sweeps complete ==="
echo "Aggregate results:"
for f in user_data/backtest_results/sweep/*.json; do
    echo "  - $(basename ${f})"
done

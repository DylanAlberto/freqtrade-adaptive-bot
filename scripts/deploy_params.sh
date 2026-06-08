#!/usr/bin/env bash
# ==============================================================
# Deploy Hyperopt Results → pair_params.json
# ==============================================================
#
# This script is a TEMPLATE — hyperopt output structures vary
# between Freqtrade versions. After running hyperopt:
#
#   1. Inspect the generated JSON (in user_data/ or --export-filename)
#   2. Extract the optimal parameters per pair
#   3. Write them into pair_params.json (at project root)
#
# A future version of this bot may include an automated
# parser. For now, follow the manual steps below.
#
# Usage:
#   bash scripts/deploy_params.sh
# ==============================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_PARAMS="${ROOT_DIR}/pair_params.default.json"
TARGET_PARAMS="${ROOT_DIR}/pair_params.json"

echo "=============================================="
echo " Deploy Hyperopt Parameters"
echo "=============================================="

if [ ! -f "${DEFAULT_PARAMS}" ]; then
    echo "ERROR: ${DEFAULT_PARAMS} not found."
    exit 1
fi

# --- Manual workflow reminder ---
cat <<INSTRUCTIONS

To deploy optimised parameters:

  1. Run hyperopt (bash scripts/run_hyperopt.sh)
  2. Check the output file for best parameters per pair, e.g.:

     Best result:
       StochRSI Period   → 14
       StochRSI Oversold → 23
       CVD Threshold     → 0.18
       ATR Stop Mult     → 2.3x

  3. Update the corresponding entry in ${TARGET_PARAMS}:

     "SOL/USDT:USDT": {
       "stochrsi": { "period": 14, "oversold": 23, ... },
       "cvd": { "threshold": 0.18, ... },
       "risk": { "atr_multiplier_stop": 2.3, ... }
     }

  4. Run the bot — the strategy reads ${TARGET_PARAMS} automatically

INSTRUCTIONS

# If pair_params.json doesn't exist, copy the defaults
if [ ! -f "${TARGET_PARAMS}" ]; then
    echo "Creating ${TARGET_PARAMS} from defaults..."
    cp "${DEFAULT_PARAMS}" "${TARGET_PARAMS}"
    echo "Done. Edit ${TARGET_PARAMS} with your hyperopt results."
else
    echo "${TARGET_PARAMS} already exists. Edit it directly with hyperopt results."
fi

#!/usr/bin/env bash
# ==============================================================
# Historical Data Ingestion — Multi-Timeframe + Multi-Pair
# ==============================================================
# Downloads 180 days of 1m, 5m, 15m, 1h futures data for
# the configured pair whitelist.
#
# Usage:  bash scripts/download_data.sh
# ==============================================================

set -euo pipefail

EXCHANGE="${EXCHANGE:-binance}"
DAYS="${DAYS:-180}"
TIMEFRAMES="${TIMEFRAMES:-1m 5m 15m 1h}"
PAIRS="${PAIRS:-BTC/USDT:USDT SOL/USDT:USDT ETH/USDT:USDT}"
DATA_DIR="${DATA_DIR:-user_data/data}"

echo "=============================================="
echo " Downloading ${DAYS} days of futures data"
echo " Exchange:  ${EXCHANGE}"
echo " Pairs:     ${PAIRS}"
echo " TFs:       ${TIMEFRAMES}"
echo " Data dir:  ${DATA_DIR}"
echo "=============================================="

freqtrade download-data \
    --exchange "${EXCHANGE}" \
    --pairs ${PAIRS} \
    --timeframes ${TIMEFRAMES} \
    --days "${DAYS}" \
    --datadir "${DATA_DIR}" \
    --trading-mode futures \
    --erase

echo "Done."

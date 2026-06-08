#!/usr/bin/env bash
# Download Open Interest data from Binance for all configured pairs
# Usage: bash scripts/download_oi.sh
# Stores OI as feather files alongside OHLCV data

PAIRS="BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT AVAX/USDT:USDT PEPE/USDT:USDT TAO/USDT:USDT VIRTUAL/USDT:USDT ARC/USDT:USDT PENGU/USDT:USDT IP/USDT:USDT ASTER/USDT:USDT WLD/USDT:USDT ZEC/USDT:USDT DASH/USDT:USDT"
LIMIT=500  # Binance max per request (500 candles)

echo "OI data will be fetched live by the strategy."
echo "For backtesting, OI data must be pre-downloaded."
echo ""
echo "This script is a placeholder for future automated OI ingestion."
echo "For now, run: python scripts/download_oi_data.py"

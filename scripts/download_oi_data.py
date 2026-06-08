"""
Download Open Interest data from Binance and store as feather.
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

EXCHANGE = "binance"
DATA_DIR = Path("user_data/data/futures")
PAIRS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
    "AVAX/USDT:USDT", "PEPE/USDT:USDT", "TAO/USDT:USDT",
    "VIRTUAL/USDT:USDT", "ARC/USDT:USDT", "PENGU/USDT:USDT",
    "IP/USDT:USDT", "ASTER/USDT:USDT", "WLD/USDT:USDT",
    "ZEC/USDT:USDT", "DASH/USDT:USDT",
]

# Binance OI endpoint format:
# https://fapi.binance.com/fapi/v1/openInterestHist?symbol=BTCUSDT&period=15m&limit=500

BINANCE_SYMBOL_MAP = {p: p.replace("/", "").replace(":USDT", "USDT") for p in PAIRS}

DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_oi(symbol: str, period: str = "15m", limit: int = 500) -> pd.DataFrame:
    """Fetch historical OI from Binance."""
    url = f"https://fapi.binance.com/fapi/v1/openInterestHist"
    params = {"symbol": symbol, "period": period, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    df.columns = [f"oi_{c}" for c in df.columns]
    return df


def main():
    for pair in PAIRS:
        bsym = BINANCE_SYMBOL_MAP[pair]
        safe_name = pair.replace("/", "_").replace(":", "_")
        out_path = DATA_DIR / f"{safe_name}-oi-15m.feather"

        print(f"Fetching OI for {pair} ({bsym})...")
        try:
            df = fetch_oi(bsym, period="15m", limit=500)
            if df.empty:
                print(f"  No data for {pair}")
                continue
            df.to_feather(out_path)
            print(f"  Saved {len(df)} candles to {out_path.name}")
        except Exception as e:
            print(f"  Error: {e}")

        time.sleep(0.5)  # rate limit


if __name__ == "__main__":
    main()

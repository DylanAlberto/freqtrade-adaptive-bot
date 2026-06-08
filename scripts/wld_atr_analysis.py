"""Analyze WLD ATR on 15m timeframe."""
import pandas as pd
import talib.abstract as ta
from pathlib import Path

data_dir = Path('user_data/data/futures')
for f in sorted(data_dir.glob('*WLD*')):
    if '15m' not in f.name and '1h' not in f.name:
        continue
    if 'mark' in f.name or 'funding' in f.name:
        continue

    df = pd.read_feather(f)
    atr = ta.ATR(df, timeperiod=14)
    close = df['close']

    atr_pct = (atr / close * 100)
    
    print(f'=== {f.name} ===')
    print(f'  Candles: {len(df)}')
    print(f'  Close range: {close.min():.2f} - {close.max():.2f}')
    print(f'  ATR mean: {atr.mean():.4f}')
    print(f'  ATR % of price: mean={atr_pct.mean():.2f}%, median={atr_pct.median():.2f}%')
    print(f'  ATR % percentile: 10th={atr_pct.quantile(0.10):.2f}%  90th={atr_pct.quantile(0.90):.2f}%')
    print()
    print(f'  Stop distance at 2x ATR: mean {atr_pct.mean()*2:.2f}%')
    print(f'  Stop distance at 3x ATR: mean {atr_pct.mean()*3:.2f}%')
    print(f'  Stop distance at 4x ATR: mean {atr_pct.mean()*4:.2f}%')
    print(f'  Stop distance at 5x ATR: mean {atr_pct.mean()*5:.2f}%')
    print(f'  Stop distance at 6x ATR: mean {atr_pct.mean()*6:.2f}%')
    print()

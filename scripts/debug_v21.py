"""Debug: check what's in the strategy file."""
with open('user_data/strategies/MultiPairAdaptiveStrategy.py', 'r') as f:
    c = f.read()

checks = {
    "adjust_trade_position docstring": 'Sell 50% of position at 1:1' in c,
    "custom_exit docstring": 'Stage 2 exit: ATR take-profit' in c,
    "custom_stoploss": 'def custom_stoploss' in c,
    "breakeven in file": 'breakeven' in c,
    "trail_offset in file": 'trail_offset' in c,
    "tp_remaining in file": 'tp_remaining' in c,
    "stoch_exit in file": 'stoch_exit' in c,
    "CVD momentum sections": c.count('CVD momentum'),
    "populate_exit_trend simplified": 'only CVD divergence' in c,
}
for k, v in checks.items():
    print(f"  {k}: {v}")

"""Fix ema50 column name only."""
with open('user_data/strategies/MultiPairAdaptiveStrategy.py', 'r') as f:
    content = f.read()

old = 'dataframe["close"] > dataframe["ema50"]'
new = 'dataframe["close"] > dataframe["ema50_4h"]'

if old in content:
    content = content.replace(old, new)
    with open('user_data/strategies/MultiPairAdaptiveStrategy.py', 'w') as f:
        f.write(content)
    print('Fixed ema50 -> ema50_4h')
else:
    print('Pattern not found')
    # Find what's around ema50
    idx = content.find('ema50')
    if idx > 0:
        print(content[max(0,idx-100):idx+50])

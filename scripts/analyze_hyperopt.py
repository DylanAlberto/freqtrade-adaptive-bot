"""Analyze hyperopt results."""
import json

with open('user_data/hyperopt_results/strategy_MultiPairAdaptiveStrategy_2026-06-07_21-34-06.fthypt') as f:
    header = json.loads(f.readline())
    data = [json.loads(l) for l in f if l.strip()]

total = len(data)
pos = [r for r in data if r.get('total_profit', 0) > 0]

print(f"Total epochs: {total}")
print(f"Profitable epochs: {len(pos)}")
print()

if pos:
    pos.sort(key=lambda r: r['total_profit'], reverse=True)
    print("Top 5 profitable:")
    for r in pos[:5]:
        ep = r['current_epoch']
        prof = r['total_profit']
        rm = r.get('results_metrics', {})
        wr = rm.get('winrate', 0) * 100
        tc = rm.get('trade_count', 0)
        loss = r.get('loss', 0)
        p = r['params_dict']
        print(f"  Epoch {ep:>3}: profit=${prof:>8.2f} | WR={wr:>5.1f}% | Trades={tc:>4} | Obj={loss:>8.4f}")
        print(f"    Params: stoch_period={p.get('stochrsi_period')}  oversold={p.get('stochrsi_oversold')}  overbought={p.get('stochrsi_overbought')}  cvd={p.get('cvd_threshold')}  atr={p.get('atr_stop_mult')}")
else:
    data.sort(key=lambda r: r.get('total_profit', 0), reverse=True)
    print("Top 5 least negative:")
    for r in data[:5]:
        ep = r['current_epoch']
        prof = r['total_profit']
        rm = r.get('results_metrics', {})
        wr = rm.get('winrate', 0) * 100
        tc = rm.get('trade_count', 0)
        loss = r.get('loss', 0)
        p = r['params_dict']
        print(f"  Epoch {ep:>3}: profit=${prof:>8.2f} | WR={wr:>5.1f}% | Trades={tc:>4} | Obj={loss:>8.4f}")
        print(f"    Params: stoch_period={p.get('stochrsi_period')}  oversold={p.get('stochrsi_oversold')}  overbought={p.get('stochrsi_overbought')}  cvd={p.get('cvd_threshold')}  atr={p.get('atr_stop_mult')}")
        
# Also check epoch 1 specifically (the "best" reported)
print(f"\n\nBest params per hyperopt (epoch 1):")
print(json.dumps(header.get('params_details', {}), indent=2))

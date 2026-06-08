"""Check which epochs produced trades."""
import json

with open('user_data/hyperopt_results/strategy_MultiPairAdaptiveStrategy_2026-06-07_21-34-06.fthypt') as f:
    f.readline()  # skip header
    data = [json.loads(l) for l in f if l.strip()]

count = 0
for r in data:
    tc = r.get('results_metrics', {}).get('trade_count', 0)
    if tc > 0:
        ep = r['current_epoch']
        prof = r['total_profit']
        p = r['params_dict']
        print(f"Epoch {ep:>3}: trades={tc:>4}  profit=${prof:>8.2f}")
        print(f"  stoch_period={p['stochrsi_period']}  oversold={p['stochrsi_oversold']}  overbought={p['stochrsi_overbought']}  cvd={p['cvd_threshold']:.3f}  atr={p['atr_stop_mult']:.3f}")
        count += 1
        if count >= 5:
            break

if count == 0:
    print("No epochs with trades found. Checking all epochs...")
    for r in data[:5]:
        tc = r.get('results_metrics', {}).get('trade_count', 0)
        ep = r['current_epoch']
        prof = r['total_profit']
        p = r['params_dict']
        print(f"Epoch {ep:>3}: trades={tc:>4}  profit=${prof:>8.2f}")
        print(f"  {p}")

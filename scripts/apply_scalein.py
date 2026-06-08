"""Apply scale-in + dynamic TP to strategy file."""
with open('user_data/strategies/MultiPairAdaptiveStrategy.py', 'r') as f:
    content = f.read()

# 1. Replace docstring of adjust_trade_position
content = content.replace(
    'Sell 50% of position at 1:1 risk/reward.',
    'Scale-in during strong trends.'
)

content = content.replace(
    'Remaining 50% runs until take_profit or exit_signal.',
    'Adds same-sized entries at ATR intervals.'
)

# 2. Remove early return and partial exit tracking
content = content.replace(
    '''        if current_profit < 0:
            return None  # not profitable yet

        # Track which trades already did the 50% exit
        partial_key = f"partial_{trade.id}"
        if partial_key in self._partial_exits:
            return None  # already reduced

        ''',
    ''
)

# 3. Add max_entries check
content = content.replace(
    'p = self._params_for(trade.pair)\n\n        try:',
    'p = self._params_for(trade.pair)\n        max_entries = p.get("max_entries", 3)\n        if trade.nr_of_successful_entries >= max_entries:\n            return None\n\n        try:'
)

# 4. Replace 1:1 R:R with scale-in distance
content = content.replace(
    '''        # 1:1 R:R level = initial stop distance
        stop_distance = (atr * p["atr_multiplier_stop"]) / trade.open_rate

        if current_profit > stop_distance:
            # Sell 50% — negative amount reduces position
            self._partial_exits.add(partial_key)
            logger.info(
                f"Partial exit 50% at {current_profit:.2%} profit for {trade.pair}"
            )
            return -(trade.stake_amount / 2)''',
    '''        scale_distance = p.get("scale_distance_atr", 1.0)
        if trade.is_short:
            price_diff = trade.open_rate - current_rate
        else:
            price_diff = current_rate - trade.open_rate
        min_distance = atr * scale_distance
        if price_diff >= min_distance:
            logger.info(
                f"Scale-in #{trade.nr_of_successful_entries + 1} for {trade.pair}"
            )
            return trade.stake_amount / trade.nr_of_successful_entries'''
)

# 5. Replace custom_exit docstring
content = content.replace(
    'Stage 2 exit: ATR take-profit + Time-Based Exit for remaining 50%.',
    'Dynamic take-profit decreasing with each scale-in entry.'
)
content = content.replace(
    'Stage 1 (50% at 1:1) is handled by adjust_trade_position().',
    'Entry 1 -> 4x ATR, Entry 2 -> 2x ATR, Entry 3 -> 1x ATR.'
)

# 6. Add tp_scale and entries in custom_exit
content = content.replace(
    'tp_mult = p.get("atr_multiplier_tp", 4.0)',
    'tp_mult = p.get("atr_multiplier_tp", 4.0)\n        tp_scale = p.get("tp_scale_factor", 0.5)\n        entries = trade.nr_of_successful_entries'
)

# 7. Make tp_target dynamic
content = content.replace(
    'tp_target = (atr * tp_mult) / trade.open_rate',
    'effective_tp = tp_mult * (tp_scale ** (entries - 1))\n        tp_target = (atr * effective_tp) / trade.open_rate'
)

# 8. Remove stale _partial_exits reference
content = content.replace(
    'self._partial_exits: set = set()\n\n        # Cache',
    '# Cache'
)

with open('user_data/strategies/MultiPairAdaptiveStrategy.py', 'w') as f:
    f.write(content)

print('Scale-in + TP applied')
print(f'  scale-in: {"Scale-in" in content}')
print(f'  tp_scale: {"tp_scale_factor" in content}')
print(f'  _partial_exits: {"_partial_exits" not in content}')

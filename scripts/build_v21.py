"""Build v21 step by step, checking each replacement."""
with open('user_data/strategies/MultiPairAdaptiveStrategy.py', 'r') as f:
    content = f.read()

step = 0

# Step 1: Fix ema50 column name
step += 1
content = content.replace(
    'dataframe["close"] > dataframe["ema50"]',
    'dataframe["close"] > dataframe["ema50_4h"]'
)
print(f'Step {step}: ema50_4h fix -> {"ema50_4h" in content}')

# Step 2: Add Dict import
step += 1
content = content.replace(
    'from typing import Any, Dict, Optional',
    'from typing import Any, Dict, Optional'
)
print(f'Step {step}: imports OK')

# Step 3: Add _partial_exits to __init__
step += 1
old = 'self.pair_params: Dict[str, Any] = _load_pair_params(config)\n\n        # Cache'
new = 'self.pair_params: Dict[str, Any] = _load_pair_params(config)\n\n        self._partial_exits: Dict[int, bool] = {}\n\n        # Cache'
content = content.replace(old, new)
print(f'Step {step}: _partial_exits init -> {"_partial_exits" in content[:content.find("def _params_for")]}')

# Step 4: Add trail_offset to _params_for (pair_params branch)
step += 1
old = '"atr_multiplier_tp":    float(params["risk"].get("atr_multiplier_tp", 4.0)),'
new = '"atr_multiplier_tp":    float(params["risk"].get("atr_multiplier_tp", 4.0)),\n                "atr_multiplier_trail_offset": float(params["risk"].get("atr_multiplier_trail_offset", 2.0)),'
content = content.replace(old, new)
print(f'Step {4}a: trail_offset pair_params -> {"trail_offset" in content}')

step += 1
old = '"atr_multiplier_tp":    4.0,\n                "position_size_pct":    0.10,'
new = '"atr_multiplier_tp":    4.0,\n                "atr_multiplier_trail_offset": 2.0,\n                "position_size_pct":    0.10,'
content = content.replace(old, new)
print(f'Step {4}b: trail_offset hyperopt -> {content.count("trail_offset") >= 2}')

# Step 5: Replace populate_exit_trend
step += 1
old_exit_trend = '''    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        p    = self._params_for(pair)

        dataframe.loc[:, "exit_long"]  = 0
        dataframe.loc[:, "exit_short"] = 0

        # Exit LONG: Fast StochRSI crosses BACK below overbought
        #             OR bearish CVD divergence
        dataframe.loc[
            crossed_below(dataframe["stochrsi_k_fast"], p["stochrsi_overbought"]) |
            (dataframe["cvd_div_neg"] == 1),
            "exit_long",
        ] = 1

        # Exit SHORT: Fast StochRSI crosses BACK above oversold
        #             OR bullish CVD divergence
        dataframe.loc[
            crossed_above(dataframe["stochrsi_k_fast"], p["stochrsi_oversold"]) |
            (dataframe["cvd_div_pos"] == 1),
            "exit_short",
        ] = 1

        return dataframe'''

new_exit_trend = '''    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Simplified: only CVD divergence exits.
        StochRSI exits handled in custom_exit for per-trade control.
        """
        dataframe.loc[:, "exit_long"]  = 0
        dataframe.loc[:, "exit_short"] = 0

        dataframe.loc[dataframe["cvd_div_neg"] == 1, "exit_long"] = 1
        dataframe.loc[dataframe["cvd_div_pos"] == 1, "exit_short"] = 1

        return dataframe'''

if old_exit_trend in content:
    content = content.replace(old_exit_trend, new_exit_trend)
    print(f'Step {step}: exit_trend replaced -> {"only CVD divergence" in content}')
else:
    print(f'Step {step}: EXIT TREND NOT FOUND - checking...')
    idx = content.find('def populate_exit_trend')
    if idx > 0:
        print(content[idx:idx+400])

# Step 6: Replace custom_stoploss
step += 1
old_stoploss = '''    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: pd.Timestamp,
        current_rate: float,
        current_profit: float,
        after_fill: bool = False,
        **kwargs,
    ) -> Optional[float]:
        p = self._params_for(pair)

        # Get the most recent analysed candle for this pair
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        except Exception:
            return self.stoploss  # fallback during warmup

        if dataframe is None or dataframe.empty:
            return self.stoploss

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get("atr")

        if atr is None or np.isnan(atr) or atr == 0:
            return self.stoploss

        # --- ATR-based trailing stop ---
        # stop = current_price - (ATR * multiplier)
        # As price rises, stop trails up automatically
        # As volatility expands/contracts, stop widens/tightens dynamically
        stop_price = current_rate - (atr * p["atr_multiplier_stop"])

        # Convert to Freqtrade's open_rate-relative format
        stoploss_rel = (stop_price - trade.open_rate) / trade.open_rate

        # Cap at 15% max loss from entry
        return max(stoploss_rel, -0.15)'''

new_stoploss = '''    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: pd.Timestamp,
        current_rate: float,
        current_profit: float,
        after_fill: bool = False,
        **kwargs,
    ) -> Optional[float]:
        p = self._params_for(pair)
        partial = self._partial_exits.get(trade.id, False)

        # --- Breakeven after partial exit ---
        if partial:
            be_stop = trade.open_rate * 0.999  # 0.1% below entry for fees
            stoploss_rel = (be_stop - trade.open_rate) / trade.open_rate
            return max(stoploss_rel, -0.15)

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        except Exception:
            return self.stoploss

        if dataframe is None or dataframe.empty:
            return self.stoploss

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get("atr")

        if atr is None or np.isnan(atr) or atr == 0:
            return self.stoploss

        stop_mult = p["atr_multiplier_stop"]
        trail_offset = p.get("atr_multiplier_trail_offset", 2.0)

        # --- Offset: don't trail until offset is reached ---
        offset_level = (atr * trail_offset) / trade.open_rate

        if current_profit < offset_level:
            # Stay at initial stop, don't chase price
            stop_price = trade.open_rate - (atr * stop_mult)
            stoploss_rel = (stop_price - trade.open_rate) / trade.open_rate
        else:
            # ATR-based trailing stop
            stop_price = current_rate - (atr * stop_mult)
            stoploss_rel = (stop_price - trade.open_rate) / trade.open_rate

        return max(stoploss_rel, -0.15)'''

if old_stoploss in content:
    content = content.replace(old_stoploss, new_stoploss)
    print(f'Step {step}: stoploss replaced -> {"trail_offset" in content[content.find("def custom_stoploss"):]}')
else:
    print(f'Step {step}: STOPLOSS NOT FOUND - checking...')
    idx = content.find('def custom_stoploss')
    if idx > 0:
        print(content[idx:idx+300])

# Write result
with open('user_data/strategies/MultiPairAdaptiveStrategy.py', 'w') as f:
    f.write(content)

print('\n=== FINAL CHECK ===')
with open('user_data/strategies/MultiPairAdaptiveStrategy.py') as f:
    c = f.read()
print(f'  ema50_4h: {"ema50_4h" in c}')
print(f'  _partial_exits init: {"_partial_exits: Dict[int, bool]" in c}')
print(f'  trail_offset in params: {c.count("trail_offset")}')
print(f'  exit_trend simplified: {"only CVD divergence" in c}')
print(f'  stoploss breakeven: {"be_stop" in c}')

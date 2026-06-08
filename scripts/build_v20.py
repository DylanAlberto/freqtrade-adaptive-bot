"""Build complete v20 strategy with all improvements."""
with open('user_data/strategies/MultiPairAdaptiveStrategy.py', 'r') as f:
    content = f.read()

# 1. Replace partial exit with scale-in average
old_exit = '''    # ------------------------------------------------------------------
    # Partial exit at 1:1 R:R + trailing the rest (Positive Asymmetry)
    # ------------------------------------------------------------------

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: pd.Timestamp,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        **kwargs,
    ) -> Optional[float]:
        """
        Sell 50% of position at 1:1 risk/reward.
        Remaining 50% runs until take_profit or exit_signal.
        """
        if current_profit < 0:
            return None  # not profitable yet

        # Track which trades already did the 50% exit
        partial_key = f"partial_{trade.id}"
        if partial_key in self._partial_exits:
            return None  # already reduced

        p = self._params_for(trade.pair)

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            if dataframe is None or dataframe.empty:
                return None
            atr = dataframe.iloc[-1].get("atr")
        except Exception:
            return None

        if atr is None or np.isnan(atr) or atr == 0:
            return None

        # 1:1 R:R level = initial stop distance
        stop_distance = (atr * p["atr_multiplier_stop"]) / trade.open_rate

        if current_profit > stop_distance:
            # Sell 50% — negative amount reduces position
            self._partial_exits.add(partial_key)
            logger.info(
                f"Partial exit 50% at {current_profit:.2%} profit for {trade.pair}"
            )
            return -(trade.stake_amount / 2)

        return None

    # ------------------------------------------------------------------
    # Multi-stage exits: 1:1 partial, take-profit, time-based
    # ------------------------------------------------------------------

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: pd.Timestamp,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        """
        Stage 2 exit: ATR take-profit + Time-Based Exit for remaining 50%.
        Stage 1 (50% at 1:1) is handled by adjust_trade_position().
        """
        p = self._params_for(pair)
        tp_mult = p.get("atr_multiplier_tp", 4.0)
        max_minutes = p.get("max_trade_minutes", 240)

        # --- Time-Based Exit: kill zombie trades -----------------------
        trade_minutes = (current_time - trade.open_date_utc).total_seconds() / 60
        if trade_minutes > max_minutes:
            return "time_exit"

        # --- ATR-based take profit for remaining 50% -------------------
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or dataframe.empty:
                return None
            atr = dataframe.iloc[-1].get("atr")
        except Exception:
            return None

        if atr is None or np.isnan(atr) or atr == 0:
            return None

        tp_target = (atr * tp_mult) / trade.open_rate
        if current_profit > tp_target:
            return "take_profit"

        return None'''

new_exit = '''    # ------------------------------------------------------------------
    # Scale-in: add positions in trend direction (up to max_entries)
    # ------------------------------------------------------------------

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: pd.Timestamp,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        **kwargs,
    ) -> Optional[float]:
        """
        Scale-in during strong trends.
        Adds same-sized entries at ATR intervals.
        """
        p = self._params_for(trade.pair)
        max_entries = p.get("max_entries", 3)
        if trade.nr_of_successful_entries >= max_entries:
            return None

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            if dataframe is None or dataframe.empty:
                return None
            atr = dataframe.iloc[-1].get("atr")
        except Exception:
            return None

        if atr is None or np.isnan(atr) or atr == 0:
            return None

        scale_distance = p.get("scale_distance_atr", 1.0)
        if trade.is_short:
            price_diff = trade.open_rate - current_rate
        else:
            price_diff = current_rate - trade.open_rate

        min_distance = atr * scale_distance
        if price_diff >= min_distance:
            logger.info(
                f"Scale-in #{trade.nr_of_successful_entries + 1} for {trade.pair}"
            )
            return trade.stake_amount / trade.nr_of_successful_entries
        return None

    # ------------------------------------------------------------------
    # Dynamic take-profit + Time-Based Exit
    # TP decreases with each scale-in entry
    # ------------------------------------------------------------------

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: pd.Timestamp,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        p = self._params_for(pair)
        base_tp = p.get("atr_multiplier_tp", 4.0)
        tp_scale = p.get("tp_scale_factor", 0.5)
        max_minutes = p.get("max_trade_minutes", 240)
        entries = trade.nr_of_successful_entries

        # Time-based exit
        trade_minutes = (current_time - trade.open_date_utc).total_seconds() / 60
        if trade_minutes > max_minutes:
            return "time_exit"

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or dataframe.empty:
                return None
            atr = dataframe.iloc[-1].get("atr")
        except Exception:
            return None

        if atr is None or np.isnan(atr) or atr == 0:
            return None

        effective_tp = base_tp * (tp_scale ** (entries - 1))
        tp_target = (atr * effective_tp) / trade.open_rate
        if current_profit > tp_target:
            return "take_profit"
        return None'''

content = content.replace(old_exit, new_exit)

# 2. Add section 9 - CVD momentum filter (before return dataframe)
old_return = '''
        return dataframe'''

new_sections = '''
        # --- 9. CVD momentum + Time-in-Oversold (agotamiento) ----------
        try:
            dataframe["cvd_rising"] = (dataframe["cvd"].diff(1) > 0).astype(int)
            cvd_diff = dataframe["cvd"].diff(1)
            dataframe["cvd_not_falling"] = (cvd_diff >= cvd_diff.shift(1)).fillna(1).astype(int)
            dataframe["cvd_not_rising"] = (cvd_diff <= cvd_diff.shift(1)).fillna(1).astype(int)
            
            # Oscillator oversold/overbought candle counter (simplified)
            p_os = p["stochrsi_oversold"]
            p_ob = p["stochrsi_overbought"]
            
            # Rolling count - faster method without groupby
            os_count = 0
            ob_count = 0
            os_list, ob_list = [], []
            for val in dataframe["stochrsi_k_fast"].values:
                os_count = os_count + 1 if val < p_os else 0
                ob_count = ob_count + 1 if val > p_ob else 0
                os_list.append(os_count)
                ob_list.append(ob_count)
            
            dataframe["stochrsi_os_candles"] = os_list
            dataframe["stochrsi_ob_candles"] = ob_list
            
            dataframe["exhaustion_exit_short"] = (
                (dataframe["stochrsi_os_candles"] >= 3) &
                (dataframe["cvd_not_falling"] == 1)
            ).astype(int)
            
            dataframe["exhaustion_exit_long"] = (
                (dataframe["stochrsi_ob_candles"] >= 3) &
                (dataframe["cvd_not_rising"] == 1)
            ).astype(int)
        except Exception as exc:
            logger.warning(f"Section 9 failed: {exc}")
            dataframe["cvd_rising"] = 1
            dataframe["cvd_not_falling"] = 1
            dataframe["cvd_not_rising"] = 1
            dataframe["stochrsi_os_candles"] = 0
            dataframe["stochrsi_ob_candles"] = 0
            dataframe["exhaustion_exit_short"] = 0
            dataframe["exhaustion_exit_long"] = 0

        return dataframe'''

content = content.replace(old_return, new_sections)

# 3. Update exit logic in populate_exit_trend
old_exit_trend = '''        # Exit LONG: Fast StochRSI crosses BACK below overbought
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
        ] = 1'''

new_exit_trend = '''        # Exit LONG:
        # 1. StochRSI crosses below OB + CVD NOT rising (liquidity void)
        # 2. Exhaustion: 3+ candles above OB + CVD flat
        # 3. Bearish CVD divergence
        dataframe.loc[
            (
                crossed_below(dataframe["stochrsi_k_fast"], p["stochrsi_overbought"]) &
                (dataframe["cvd_rising"] == 0)
            ) |
            (dataframe["exhaustion_exit_long"] == 1) |
            (dataframe["cvd_div_neg"] == 1),
            "exit_long",
        ] = 1

        # Exit SHORT:
        # 1. StochRSI crosses above OS + CVD rising (real bounce)
        # 2. Exhaustion: 3+ candles below OS + CVD flat
        # 3. Bullish CVD divergence
        dataframe.loc[
            (
                crossed_above(dataframe["stochrsi_k_fast"], p["stochrsi_oversold"]) &
                (dataframe["cvd_rising"] == 1)
            ) |
            (dataframe["exhaustion_exit_short"] == 1) |
            (dataframe["cvd_div_pos"] == 1),
            "exit_short",
        ] = 1'''

content = content.replace(old_exit_trend, new_exit_trend)

# 4. Remove _partial_exits reference (was renamed)
content = content.replace(
    "        # Track which trades have done a 50% partial exit\n        self._partial_exits: set = set()\n\n        # Cache resolved params per pair",
    "        # Cache resolved params per pair"
)

# 5. Add default values for new params
content = content.replace(
    '        "position_size_pct": 0.12,',
    '        "position_size_pct": 0.12,\n        "max_entries": 3,\n        "scale_distance_atr": 1.0,\n        "tp_scale_factor": 0.5,'
)

# 6. Add new params to _params_for pair_params branch
content = content.replace(
    '                "max_trade_minutes":    int(params["risk"].get("max_trade_minutes", 240)),',
    '                "max_trade_minutes":    int(params["risk"].get("max_trade_minutes", 240)),\n                "max_entries":          int(params["risk"].get("max_entries", 3)),\n                "scale_distance_atr":   float(params["risk"].get("scale_distance_atr", 1.0)),\n                "tp_scale_factor":      float(params["risk"].get("tp_scale_factor", 0.5)),'
)

# 7. Add new params to _params_for hyperopt branch
content = content.replace(
    '                "max_trade_minutes":    240,',
    '                "max_trade_minutes":    240,\n                "max_entries":          3,\n                "scale_distance_atr":   1.0,\n                "tp_scale_factor":      0.5,'
)

with open('user_data/strategies/MultiPairAdaptiveStrategy.py', 'w') as f:
    f.write(content)

print("Strategy built successfully")
# Count key elements
print(f"  has adjust_trade_position: {'def adjust_trade_position' in content}")
print(f"  has scale-in: {'Scale-in during strong trends' in content}")
print(f"  has exhaustion: {'exhaustion_exit_short' in content}")
print(f"  has ema50_4h: {'ema50_4h' in content}")
print(f"  has _partial_exits: {'_partial_exits' not in content}")

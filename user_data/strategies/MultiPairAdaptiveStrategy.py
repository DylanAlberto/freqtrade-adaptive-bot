"""
MultiPairAdaptiveStrategy — Advanced Multi-Pair Crypto Trading Bot
====================================================================

Non-colinear indicator set:
  1. Stochastic RSI  → momentum/exhaustion (per-pair tuned 5–25 periods)
  2. CVD (approx.)   → order-flow, divergence detection
  3. ATR             → dynamic trailing stop

Regime filter via ΔCVD(1H):
  |ΔCVD| > threshold  → Trend Regime       → trade momentum
  |ΔCVD| ≤ threshold  → Range Regime       → trade mean-reversion via StochRSI

Per-pair parameters injected from pair_params.json (output of Bayesian Hyperopt).
Compatible with Freqtrade backtesting + live trading (INTERFACE_VERSION 3).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    merge_informative_pair,
)
from pandas import DataFrame

try:
    from freqtrade.vendor.qtpylib.indicators import crossed_above, crossed_below
except ImportError:
    # Fallback implementations if qtpylib not available
    def crossed_above(series1, series2):
        return (series1.shift(1) <= series2.shift(1)) & (series1 > series2)

    def crossed_below(series1, series2):
        return (series1.shift(1) >= series2.shift(1)) & (series1 < series2)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approx_cvd(close: pd.Series, open_: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Approximate Cumulative Volume Delta from OHLCV data.
    delta  = (close - open) * volume    — raw delta per candle
    cvd    = cumulative sum of delta
    """
    delta = (close - open_) * volume
    return delta.cumsum()


def _load_pair_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load per-pair parameter dictionary from pair_params.json.
    Search order:
      1. Absolute path from config key 'pair_params_path'
      2. Relative to config['user_data_dir']/../pair_params.json
      3. Relative to this file's directory: ../../pair_params.json
      4. Current working directory
    Falls back to empty dict so strategy defaults always apply.
    """
    search_paths = []

    # 1. Explicit config key
    if "pair_params_path" in config:
        search_paths.append(Path(config["pair_params_path"]))

    # 2. user_data_dir parallel
    if "user_data_dir" in config:
        search_paths.append(Path(config["user_data_dir"]).parent / "pair_params.json")

    # 3. Relative to strategy file
    search_paths.append(Path(__file__).resolve().parent.parent.parent / "pair_params.json")

    # 4. CWD
    search_paths.append(Path.cwd() / "pair_params.json")

    for path in search_paths:
        try:
            if path.exists():
                with path.open() as f:
                    raw = json.load(f)
                    params = {k: v for k, v in raw.items() if k != "metadata"}
                    logger.info(f"Loaded per-pair parameters from {path} — {list(params.keys())}")
                    return params
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to load pair_params from {path}: {exc}")

    logger.info("No pair_params.json found — using hardcoded defaults")
    return {}


# ---------------------------------------------------------------------------
# Default parameters per pair (used when pair_params.json is absent)
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "stochrsi": {
        "fast_period": 7,
        "slow_period": 21,
        "smooth": 3,
        "oversold": 20,
        "overbought": 80,
    },
    "cvd": {
        "delta_period": 10,
        "threshold": 0.20,
    },
    "risk": {
        "atr_multiplier_stop": 2.0,
        "atr_multiplier_entry": 1.0,
        "position_size_pct": 0.12,
    },
}


# ===================================================================
# Strategy
# ===================================================================

class MultiPairAdaptiveStrategy(IStrategy):
    """
    Adapts execution logic per trading pair using a combination of
    StochRSI (momentum), CVD (order-flow), and ATR (volatility).

    Parameters are loaded from **pair_params.json** at initialisation,
    normally produced by Freqtrade's Bayesian Hyperopt engine.
    """

    # --- Freqtrade required attributes ---------------------------------
    INTERFACE_VERSION = 3

    can_short = True

    # WLD/USDT focused: 15m base, 1h informative
    timeframe: str = "15m"
    informative_timeframe: str = "1h"

    # Hard stoploss fallback — overridden dynamically by custom_stoploss()
    stoploss = -0.05

    # Explicitly enable custom_stoploss (required for ATR dynamic stops)
    use_custom_stoploss = True

    # Enable position adjustment for partial exits
    position_adjustment_enable = True

    # ROI table disabled (exit is signal-based)
    minimal_roi = {"0": 100.0}

    # Warmup candles needed for all indicators
    startup_candle_count: int = 200

    # --- Freqtrade Protections (Global Loss Shield + Cooldown) --------
    @property
    def protections(self):
        return [
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 96,        # 24h en 15m
                "trade_limit": 1,
                "stop_duration_candles": 96,
                "max_allowed_drawdown": 0.05,          # 5% diario
            },
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 1,            # 15min post-trade
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 48,         # 12h
                "stop_duration_candles": 48,
                "trade_limit": 3,
                "only_per_pair": True,
            },
        ]

    # --- Hyperopt spaces (fallback when pair_params.json absent) ------
    stochrsi_period      = IntParameter(5, 25, default=10, space="buy")
    stochrsi_oversold    = IntParameter(10, 35, default=20, space="buy")
    stochrsi_overbought  = IntParameter(65, 90, default=80, space="sell")
    cvd_threshold        = DecimalParameter(0.05, 0.50, default=0.20, space="buy")
    atr_stop_mult        = DecimalParameter(1.0, 4.0, default=2.0, space="sell")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        # Load per-pair parameters
        self.pair_params: Dict[str, Any] = _load_pair_params(config)

        # Track which trades have done a 50% partial exit
        # Cache resolved params per pair
        self._params_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Parameter resolution
    # ------------------------------------------------------------------

    def _params_for(self, pair: str) -> Dict[str, Any]:
        """Return the resolved parameter dict for *pair* (cached).

        Priority:
          1. pair_params.json entry for this pair
          2. Hyperopt parameters (IntParameter / DecimalParameter values)
          3. Hardcoded _DEFAULTS
        """
        if pair in self._params_cache:
            return self._params_cache[pair]

        params = self.pair_params.get(pair, None)

        if params is not None:
            # Use pair_params.json values
            resolved = {
                "stochrsi_fast_period":  int(params["stochrsi"]["fast_period"]),
                "stochrsi_slow_period":  int(params["stochrsi"]["slow_period"]),
                "stochrsi_smooth":       int(params["stochrsi"].get("smooth", 3)),
                "stochrsi_oversold":     float(params["stochrsi"]["oversold"]),
                "stochrsi_overbought":   float(params["stochrsi"]["overbought"]),
                "cvd_delta_period":     int(params["cvd"]["delta_period"]),
                "cvd_threshold":        float(params["cvd"]["threshold"]),
                "atr_multiplier_stop":  float(params["risk"]["atr_multiplier_stop"]),
                "atr_multiplier_entry": float(params["risk"]["atr_multiplier_entry"]),
                "atr_multiplier_tp":    float(params["risk"].get("atr_multiplier_tp", 4.0)),
                "atr_multiplier_trail_offset": float(params["risk"].get("atr_multiplier_trail_offset", 2.0)),
                "position_size_pct":    float(params["risk"].get("position_size_pct", 0.10)),
                "max_trade_minutes":    int(params["risk"].get("max_trade_minutes", 240)),
            }
        else:
            # Use hyperopt-testable parameters (falls back to defaults)
            resolved = {
                "stochrsi_fast_period":  7,
                "stochrsi_slow_period":  21,
                "stochrsi_smooth":       3,
                "stochrsi_oversold":     float(self.stochrsi_oversold.value),
                "stochrsi_overbought":   float(self.stochrsi_overbought.value),
                "cvd_delta_period":     10,
                "cvd_threshold":        self.cvd_threshold.value,
                "atr_multiplier_stop":  self.atr_stop_mult.value,
                "atr_multiplier_entry": 1.0,
                "atr_multiplier_tp":    4.0,
                "atr_multiplier_trail_offset": 2.0,
                "position_size_pct":    0.10,
                "max_trade_minutes":    240,
            }

        self._params_cache[pair] = resolved
        return resolved

    # ------------------------------------------------------------------
    # Informative pairs
    # ------------------------------------------------------------------

    def informative_pairs(self) -> list:
        pairs = self.config["exchange"]["pair_whitelist"]
        # Register 1h (for CVD regime) and 4h (for EMA50 trend filter)
        return ([(pair, self.informative_timeframe) for pair in pairs] +
                [(pair, "4h") for pair in pairs])

    # ------------------------------------------------------------------
    # Indicator population
    # ------------------------------------------------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        p    = self._params_for(pair)

        # --- 1. CVD (raw delta + cumulative) on the base timeframe -----
        dataframe["cvd_raw"] = (dataframe["close"] - dataframe["open"]) * dataframe["volume"]
        dataframe["cvd"]     = dataframe["cvd_raw"].cumsum()

        # --- 2. Stochastic RSI ------------------------------------------
        self._populate_stochrsi(dataframe, p)

        # --- 3. ATR + Volatility Shield ------------------------------
        import talib.abstract as ta
        atr_period = max(p["stochrsi_fast_period"], 14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=atr_period)

        # ATR moving average (20 periods) — baseline para el shield
        dataframe["atr_ma"] = ta.SMA(dataframe["atr"], timeperiod=20)

        # ATR ratio: cuánto se desvía el ATR actual de su media
        # > 1.0 → volatilidad por encima de lo normal
        dataframe["atr_ratio"] = dataframe["atr"] / dataframe["atr_ma"]

        # Volatility Shield: no operar cuando ATR supera 1.5× su media
        # (movimientos caóticos / manipulación)
        dataframe["volatility_shield"] = (dataframe["atr_ratio"] < 1.5).astype(int)

        # --- 4. CVD rate-of-change on base TF (regime helper) -----------
        dataframe["cvd_delta"]    = dataframe["cvd"].diff(p["cvd_delta_period"])
        dataframe["cvd_delta_abs"] = dataframe["cvd_delta"].abs()
        dataframe["regime_trend"] = (dataframe["cvd_delta_abs"] > p["cvd_threshold"]).astype(int)

        # --- 5. CVD-close divergences -----------------------------------
        _fast = max(p["cvd_delta_period"] // 2, 2)
        dataframe["cvd_div_pos"] = (
            (dataframe["close"].diff(_fast) < 0) &
            (dataframe["cvd"].diff(_fast) > 0)
        ).astype(int)

        dataframe["cvd_div_neg"] = (
            (dataframe["close"].diff(_fast) > 0) &
            (dataframe["cvd"].diff(_fast) < 0)
        ).astype(int)

        # --- 6. Informative (1H) CVD for regime filter ------------------
        dataframe = self._attach_informative_cvd(dataframe, metadata)

        # --- 7. Open Interest (OI) — institutional flow filter ---------
        dataframe["oi_confirmed"] = 1
        try:
            oi_df = self.dp.get_pair_dataframe(pair, "15m")
            if oi_df is not None and not oi_df.empty and "open_interest" in oi_df.columns:
                oi_change = oi_df["open_interest"].diff(4)
                dataframe["oi_change"] = oi_change
                dataframe["oi_confirmed"] = (
                    (dataframe["cvd_delta"] > 0) & (oi_change > 0) |
                    (dataframe["cvd_delta"] < 0) & (oi_change < 0)
                ).astype(int)
        except Exception:
            pass

        # --- 8. EMA 50 en 4H — interruptor direccional estructural ------
        # Si precio < EMA50(4h) → solo SHORT
        # Si precio > EMA50(4h) → solo LONG
        try:
            df_4h = self.dp.get_pair_dataframe(pair, "4h")
            if df_4h is not None and not df_4h.empty:
                import talib.abstract as ta
                df_4h["ema50"] = ta.EMA(df_4h["close"], timeperiod=50)
                # Merge la EMA 50 al dataframe base
                dataframe = merge_informative_pair(
                    dataframe, df_4h[["date", "ema50"]],
                    self.timeframe, "4h", ffill=True,
                )
                # Dirección: 1 = LONG only, 0 = SHORT only
                dataframe["ema50_direction"] = (
                    dataframe["close"] > dataframe["ema50_4h"]
                ).astype(int)
            else:
                dataframe["ema50_direction"] = 1  # fallback: permitir ambos
        except Exception:
            dataframe["ema50_direction"] = 1

        return dataframe

    # ------------------------------------------------------------------
    # Entry signals
    # ------------------------------------------------------------------

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        p    = self._params_for(pair)

        dataframe.loc[:, "enter_long"]  = 0
        dataframe.loc[:, "enter_short"] = 0

        # --- Trend Regime -----------------------------------------------
        trend_mask = dataframe["regime_trend"] == 1

        # LONG: StochRSI_fast crosses above oversold + CVD delta positive
        #       + StochRSI_slow > 50 (medium-term bullish filter)
        #       + Volatility Shield + OI + EMA50(4H) trending UP → LONG only
        dataframe.loc[
            trend_mask &
            crossed_above(dataframe["stochrsi_k_fast"], p["stochrsi_oversold"]) &
            (dataframe["cvd_delta"] > 0) &
            (dataframe["stochrsi_k_slow"] > 50) &
            (dataframe["volatility_shield"] == 1) &
            (dataframe["oi_confirmed"] == 1) &
            (dataframe["ema50_direction"] == 1),
            "enter_long",
        ] = 1

        # SHORT: StochRSI_fast crosses below overbought + CVD delta negative
        #        + StochRSI_slow < 50 + EMA50(4H) trending DOWN → SHORT only
        dataframe.loc[
            trend_mask &
            crossed_below(dataframe["stochrsi_k_fast"], p["stochrsi_overbought"]) &
            (dataframe["cvd_delta"] < 0) &
            (dataframe["stochrsi_k_slow"] < 50) &
            (dataframe["volatility_shield"] == 1) &
            (dataframe["oi_confirmed"] == 1) &
            (dataframe["ema50_direction"] == 0),
            "enter_short",
        ] = 1

        # --- Range Regime -----------------------------------------------
        range_mask = dataframe["regime_trend"] == 0

        # LONG: StochRSI_fast oversold bounce + StochRSI_slow bullish
        #       + Volatility + OI + EMA50 UP
        dataframe.loc[
            range_mask &
            crossed_above(dataframe["stochrsi_k_fast"], p["stochrsi_oversold"]) &
            (dataframe["stochrsi_k_slow"] > 50) &
            (dataframe["volatility_shield"] == 1) &
            (dataframe["oi_confirmed"] == 1) &
            (dataframe["ema50_direction"] == 1),
            "enter_long",
        ] = 1

        # SHORT: StochRSI_fast overbought rejection + StochRSI_slow bearish
        #        + Volatility + OI + EMA50 DOWN
        dataframe.loc[
            range_mask &
            crossed_below(dataframe["stochrsi_k_fast"], p["stochrsi_overbought"]) &
            (dataframe["stochrsi_k_slow"] < 50) &
            (dataframe["volatility_shield"] == 1) &
            (dataframe["oi_confirmed"] == 1) &
            (dataframe["ema50_direction"] == 0),
            "enter_short",
        ] = 1

        return dataframe

    # ------------------------------------------------------------------
    # Exit signals
    # ------------------------------------------------------------------

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Simplified: only CVD divergence exits.
        StochRSI exits handled in custom_exit for per-trade control.
        """
        dataframe.loc[:, "exit_long"]  = 0
        dataframe.loc[:, "exit_short"] = 0

        dataframe.loc[dataframe["cvd_div_neg"] == 1, "exit_long"] = 1
        dataframe.loc[dataframe["cvd_div_pos"] == 1, "exit_short"] = 1

        return dataframe

    # ------------------------------------------------------------------
    # Dynamic ATR-based stoploss
    # ------------------------------------------------------------------

    def custom_stoploss(
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

        return max(stoploss_rel, -0.15)

    # ------------------------------------------------------------------
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

        return None

    # ------------------------------------------------------------------
    # Kelly-informed position sizing — risk a fixed % of wallet per trade
    # ------------------------------------------------------------------

    def custom_stake_amount(
        self,
        pair: str,
        current_time: pd.Timestamp,
        current_rate: float,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """
        Z-Score modulated position sizing (Half-Kelly 2%-6%).

        Base: 5% of wallet (Half-Kelly).
        Modulation: ATR Z-Score + signal confluence.
        - ATR anómalo (> 1.5× media) → 2% (mínimo)
        - ATR normal (0.8-1.2× media) → 5% (base)
        - ATR comprimido (< 0.8× media) → 6% (máximo — calma = señal fiable)
        """
        p = self._params_for(pair)
        base_pct = p.get("position_size_pct", 0.05)

        # --- Z-Score modulation via ATR ratio --------------------------
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is not None and not dataframe.empty:
                atr_ratio = dataframe.iloc[-1].get("atr_ratio", 1.0)
            else:
                atr_ratio = 1.0
        except Exception:
            atr_ratio = 1.0

        atr_ratio = atr_ratio if atr_ratio > 0 else 1.0

        # Z-Score inspired bands (ATR ratio is a proxy for Z-Score)
        if atr_ratio < 0.8:
            mult = 1.2    # 6% — calma, señal fiable
        elif atr_ratio < 1.0:
            mult = 1.0    # 5% — normal
        elif atr_ratio < 1.2:
            mult = 0.8    # 4% — ligeramente volátil
        elif atr_ratio < 1.5:
            mult = 0.6    # 3% — volátil
        else:
            mult = 0.4    # 2% — shield activo, mínimo riesgo

        pct = base_pct * mult
        pct = max(0.02, min(0.06, pct))  # clamp 2%-6%

        wallet = self.wallets.get_total_stake_amount()
        stake = wallet * pct

        return max(min(stake, max_stake), min_stake)

    # ================================================================
    # Private helpers
    # ================================================================

    def _populate_stochrsi(self, dataframe: DataFrame, p: Dict[str, Any]) -> None:
        """Add fast and slow StochRSI columns.

        Fast  → trigger (short-period, sensitive)
        Slow  → filter (long-period, trend bias)
        """
        import talib.abstract as ta

        rsi_fast = ta.RSI(dataframe, timeperiod=p["stochrsi_fast_period"])
        rsi_slow = ta.RSI(dataframe, timeperiod=p["stochrsi_slow_period"])

        smooth = p["stochrsi_smooth"]

        # Fast StochRSI (%K) — entry trigger
        fast_stoch = ta.STOCHF(
            pd.DataFrame({"high": rsi_fast, "low": rsi_fast, "close": rsi_fast}),
            fastk_period=p["stochrsi_fast_period"],
            fastd_period=smooth,
            fastk_matype=0,
            fastd_matype=0,
        )
        dataframe["stochrsi_k_fast"] = fast_stoch["fastk"]
        dataframe["stochrsi_d_fast"] = fast_stoch["fastd"]

        # Slow StochRSI (%K) — trend filter (>50 bullish, <50 bearish)
        slow_stoch = ta.STOCHF(
            pd.DataFrame({"high": rsi_slow, "low": rsi_slow, "close": rsi_slow}),
            fastk_period=p["stochrsi_slow_period"],
            fastd_period=smooth,
            fastk_matype=0,
            fastd_matype=0,
        )
        dataframe["stochrsi_k_slow"] = slow_stoch["fastk"]
        dataframe["stochrsi_d_slow"] = slow_stoch["fastd"]

    def _attach_informative_cvd(
        self,
        dataframe: DataFrame,
        metadata: dict,
    ) -> DataFrame:
        """
        Fetch 1H data for the current pair, compute CVD, and merge
        the regime-relevant columns onto the base dataframe.

        This approach works identically in backtesting and live trading.
        """
        pair = metadata["pair"]
        tf   = self.informative_timeframe

        # Fetch the 1H dataframe
        try:
            informative = self.dp.get_pair_dataframe(pair, tf)
        except Exception as exc:
            logger.debug("Could not fetch informative data for %s: %s", pair, exc)
            return dataframe

        if informative is None or informative.empty:
            return dataframe

        # --- Compute CVD on 1H ------------------------------------------
        informative = informative.copy()
        informative["cvd_raw_1h"] = (
            (informative["close"] - informative["open"]) * informative["volume"]
        )
        informative["cvd_1h"] = informative["cvd_raw_1h"].cumsum()

        # ΔCVD over a longer window on 1H (triple the base period)
        p = self._params_for(pair)
        delta_window = p["cvd_delta_period"] * 3
        informative["cvd_1h_delta"] = informative["cvd_1h"].diff(delta_window)
        informative["cvd_1h_delta_abs"] = informative["cvd_1h_delta"].abs()

        # Keep only what we need + date for merge
        keep_cols = ["date", "cvd_1h", "cvd_1h_delta", "cvd_1h_delta_abs"]

        # Merge onto base dataframe
        dataframe = merge_informative_pair(
            dataframe, informative[keep_cols],
            self.timeframe, tf,
            ffill=True,
        )

        return dataframe

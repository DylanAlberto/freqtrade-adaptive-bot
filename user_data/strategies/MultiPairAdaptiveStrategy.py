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
    # V24: parámetros Hyperopt para salidas — optimización 100% matemática
    atr_trail_offset     = DecimalParameter(1.0, 3.0, default=2.0, space="sell")
    max_trade_minutes_p  = IntParameter(120, 480, default=240, space="sell")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        # Load per-pair parameters
        self.pair_params: Dict[str, Any] = _load_pair_params(config)

        # Track which trades have done a 50% partial exit (set of trade IDs)
        # Track which trades have done a 50% partial exit (set of trade IDs)
        self._partial_exits: set = set()

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
            # Use pair_params.json values (with graceful fallbacks)
            stoch = params.get("stochrsi", {})
            cvd   = params.get("cvd", {})
            risk  = params.get("risk", {})
            resolved = {
                "stochrsi_fast_period":  int(stoch.get("fast_period", 7)),
                "stochrsi_slow_period":  int(stoch.get("slow_period", 21)),
                "stochrsi_smooth":       int(stoch.get("smooth", 3)),
                "stochrsi_oversold":     float(stoch.get("oversold", 20)),
                "stochrsi_overbought":   float(stoch.get("overbought", 80)),
                "cvd_delta_period":     int(cvd.get("delta_period", 10)),
                "cvd_threshold":        float(cvd.get("threshold", 0.20)),
                "atr_multiplier_stop":  float(risk.get("atr_multiplier_stop", self.atr_stop_mult.value)),
                "atr_multiplier_entry": float(risk.get("atr_multiplier_entry", 1.0)),
                "atr_multiplier_tp":    float(risk.get("atr_multiplier_tp", 4.0)),
                "atr_multiplier_trail_offset": float(risk.get("atr_multiplier_trail_offset", self.atr_trail_offset.value)),
                "position_size_pct":    float(risk.get("position_size_pct", 0.10)),
                "max_trade_minutes":    int(risk.get("max_trade_minutes", self.max_trade_minutes_p.value)),
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
                "atr_multiplier_trail_offset": float(self.atr_trail_offset.value),
                "position_size_pct":    0.10,
                "max_trade_minutes":    int(self.max_trade_minutes_p.value),
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

        # --- 3b. EMA20 — distance-to-mean filter ------------------------
        dataframe["ema20"] = ta.EMA(dataframe["close"], timeperiod=20)

        # Distance from price to EMA20 in ATR units
        dataframe["dist_to_mean"] = (
            (dataframe["close"] - dataframe["ema20"]).abs() / dataframe["atr"]
        )
        # 1 = valid entry (price within 1.5 ATR of EMA20), 0 = bloqueado
        dataframe["dist_mean_filter"] = (dataframe["dist_to_mean"] <= 1.5).astype(int)

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
        # Rollback V23 → V22: EMA50(4H) es más estable que EMA20(1H).
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

        # --- 9. Volume Cascade Shield ------------------------------------
        # Bloquea entradas si volumen actual > 3× media 20 velas.
        # Típico de liquidaciones en cascada (cuchillos cayendo).
        dataframe["vol_sma20"] = ta.SMA(dataframe["volume"], timeperiod=20)
        dataframe["vol_ratio"] = dataframe["volume"] / dataframe["vol_sma20"]
        dataframe["vol_cascade_shield"] = (dataframe["vol_ratio"] <= 3.0).astype(int)

        # --- 10. Price Action filter — V24 ------------------------------
        # Prohíbe comprar solo por oscilador. Exige que la vela de 15m
        # cierre con cuerpo o con mecha de absorción en sobreventa/sobrecompra.
        # --- LONG: close > open (bullish body) OR long lower wick in oversold
        dataframe["pa_bullish_body"] = (dataframe["close"] > dataframe["open"]).astype(int)
        dataframe["pa_lower_wick"] = dataframe["close"] - dataframe["low"]
        dataframe["pa_upper_wick"] = dataframe["high"] - dataframe["close"]
        # Lower wick > 2× upper wick AND StochRSI in oversold = absorption candle
        dataframe["pa_absorption_long"] = (
            (dataframe["stochrsi_k_fast"] < p["stochrsi_oversold"]) &
            (dataframe["pa_lower_wick"] > 2 * dataframe["pa_upper_wick"])
        ).astype(int)
        # LONG confirmed: bullish body OR absorption wick
        dataframe["pa_confirmed_long"] = (
            (dataframe["pa_bullish_body"] == 1) |
            (dataframe["pa_absorption_long"] == 1)
        ).astype(int)

        # --- SHORT: close < open (bearish body) OR long upper wick in overbought
        dataframe["pa_bearish_body"] = (dataframe["close"] < dataframe["open"]).astype(int)
        # Upper wick > 2× lower wick AND StochRSI in overbought = rejection candle
        dataframe["pa_absorption_short"] = (
            (dataframe["stochrsi_k_fast"] > p["stochrsi_overbought"]) &
            (dataframe["pa_upper_wick"] > 2 * dataframe["pa_lower_wick"])
        ).astype(int)
        # SHORT confirmed: bearish body OR rejection wick
        dataframe["pa_confirmed_short"] = (
            (dataframe["pa_bearish_body"] == 1) |
            (dataframe["pa_absorption_short"] == 1)
        ).astype(int)

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
            (dataframe["ema50_direction"] == 1) &
            (dataframe["vol_cascade_shield"] == 1) &
            (dataframe["pa_confirmed_long"] == 1) &
            (dataframe["dist_mean_filter"] == 1),
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
            (dataframe["ema50_direction"] == 0) &
            (dataframe["vol_cascade_shield"] == 1) &
            (dataframe["pa_confirmed_short"] == 1) &
            (dataframe["dist_mean_filter"] == 1),
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
            (dataframe["ema50_direction"] == 1) &
            (dataframe["vol_cascade_shield"] == 1) &
            (dataframe["pa_confirmed_long"] == 1) &
            (dataframe["dist_mean_filter"] == 1),
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
            (dataframe["ema50_direction"] == 0) &
            (dataframe["vol_cascade_shield"] == 1) &
            (dataframe["pa_confirmed_short"] == 1) &
            (dataframe["dist_mean_filter"] == 1),
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

        # --- Breakeven Hard-Lock after partial exit ---
        # Lock remaining 50% at +0.1% profit the instant partial exit fires
        # Neutraliza las pérdidas de -$116 que causaron los trailing stops en v21
        if trade.id in self._partial_exits:
            if trade.is_short:
                be_stop = trade.open_rate * 0.999  # short: price -0.1% = profit
            else:
                be_stop = trade.open_rate * 1.001  # long: price +0.1% = profit
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

        if trade.id in self._partial_exits:
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
            self._partial_exits.add(trade.id)
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
        Full-Kelly dynamic position sizing (4%-12% of wallet per trade).

        Aprovecha el bajísimo DD del 0.8% para escalar el tamaño nominal
        sin aumentar significativamente el riesgo de ruina.
        - ATR comprimido (< 0.8× media) → 12% (máxima confianza)
        - ATR normal (0.8-1.0× media) → 10%
        - ATR ligeramente elevado (1.0-1.2× media) → 8%
        - ATR elevado (1.2-1.5× media) → 6%
        - ATR anómalo (> 1.5× media) → 4% (mínimo — shield activo)
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

        # Full-Kelly bands (ATR ratio proxies Z-Score)
        if atr_ratio < 0.8:
            mult = 2.4    # 12% — ATR comprimido, señal fiable
        elif atr_ratio < 1.0:
            mult = 2.0    # 10% — normal
        elif atr_ratio < 1.2:
            mult = 1.6    # 8%  — ligeramente volátil
        elif atr_ratio < 1.5:
            mult = 1.2    # 6%  — volátil
        else:
            mult = 0.8    # 4%  — shield activo, mínimo riesgo

        pct = base_pct * mult
        pct = max(0.04, min(0.12, pct))  # clamp 4%-12% (Full-Kelly)

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

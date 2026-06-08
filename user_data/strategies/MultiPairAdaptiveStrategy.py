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
        "period": 10,
        "smooth_k": 3,
        "smooth_d": 3,
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

    # These are overridden per-pair at runtime once we know the pair's
    # dedicated timeframe. Keep 5m as the default for startup.
    timeframe: str = "5m"
    informative_timeframe: str = "1h"

    # Hard stoploss fallback — overridden dynamically by custom_stoploss()
    stoploss = -0.05

    # ROI table disabled (exit is signal-based)
    minimal_roi = {"0": 100.0}

    # Warmup candles needed for all indicators
    startup_candle_count: int = 100

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

        # Cache resolved params per pair
        self._params_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Parameter resolution
    # ------------------------------------------------------------------

    def _params_for(self, pair: str) -> Dict[str, Any]:
        """Return the resolved parameter dict for *pair* (cached)."""
        if pair in self._params_cache:
            return self._params_cache[pair]

        params = self.pair_params.get(pair, _DEFAULTS)

        resolved = {
            "stochrsi_period":      int(params["stochrsi"]["period"]),
            "stochrsi_smooth_k":    int(params["stochrsi"]["smooth_k"]),
            "stochrsi_smooth_d":    int(params["stochrsi"]["smooth_d"]),
            "stochrsi_oversold":    float(params["stochrsi"]["oversold"]),
            "stochrsi_overbought":  float(params["stochrsi"]["overbought"]),
            "cvd_delta_period":     int(params["cvd"]["delta_period"]),
            "cvd_threshold":        float(params["cvd"]["threshold"]),
            "atr_multiplier_stop":  float(params["risk"]["atr_multiplier_stop"]),
            "atr_multiplier_entry": float(params["risk"]["atr_multiplier_entry"]),
        }

        self._params_cache[pair] = resolved
        return resolved

    # ------------------------------------------------------------------
    # Informative pairs
    # ------------------------------------------------------------------

    def informative_pairs(self) -> list:
        pairs = self.config["exchange"]["pair_whitelist"]
        return [(pair, self.informative_timeframe) for pair in pairs]

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

        # --- 3. ATR -----------------------------------------------------
        import talib.abstract as ta
        atr_period = max(p["stochrsi_period"], 14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=atr_period)

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
        # This merges CVD columns from the 1H timeframe onto the base df.
        dataframe = self._attach_informative_cvd(dataframe, metadata)

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

        # LONG: StochRSI_k crosses above oversold + CVD delta positive
        dataframe.loc[
            trend_mask &
            crossed_above(dataframe["stochrsi_k"], p["stochrsi_oversold"]) &
            (dataframe["cvd_delta"] > 0),
            "enter_long",
        ] = 1

        # SHORT: StochRSI_k crosses below overbought + CVD delta negative
        dataframe.loc[
            trend_mask &
            crossed_below(dataframe["stochrsi_k"], p["stochrsi_overbought"]) &
            (dataframe["cvd_delta"] < 0),
            "enter_short",
        ] = 1

        # --- Range Regime -----------------------------------------------
        range_mask = dataframe["regime_trend"] == 0

        # LONG: oversold bounce
        dataframe.loc[
            range_mask &
            crossed_above(dataframe["stochrsi_k"], p["stochrsi_oversold"]),
            "enter_long",
        ] = 1

        # SHORT: overbought rejection
        dataframe.loc[
            range_mask &
            crossed_below(dataframe["stochrsi_k"], p["stochrsi_overbought"]),
            "enter_short",
        ] = 1

        return dataframe

    # ------------------------------------------------------------------
    # Exit signals
    # ------------------------------------------------------------------

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        p    = self._params_for(pair)

        dataframe.loc[:, "exit_long"]  = 0
        dataframe.loc[:, "exit_short"] = 0

        # Exit LONG: StochRSI reaches overbought OR bearish CVD divergence
        dataframe.loc[
            (dataframe["stochrsi_k"] >= p["stochrsi_overbought"]) |
            (dataframe["cvd_div_neg"] == 1),
            "exit_long",
        ] = 1

        # Exit SHORT: StochRSI reaches oversold OR bullish CVD divergence
        dataframe.loc[
            (dataframe["stochrsi_k"] <= p["stochrsi_oversold"]) |
            (dataframe["cvd_div_pos"] == 1),
            "exit_short",
        ] = 1

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

        # Dynamic stop = ATR * multiplier, as fraction of entry price
        atr_stop_pct = (atr * p["atr_multiplier_stop"]) / trade.open_rate
        return -min(atr_stop_pct, 0.15)  # cap at 15%

    # ================================================================
    # Private helpers
    # ================================================================

    def _populate_stochrsi(self, dataframe: DataFrame, p: Dict[str, Any]) -> None:
        """Add StochRSI %K and %D columns."""
        import talib.abstract as ta

        rsi = ta.RSI(dataframe, timeperiod=p["stochrsi_period"])

        # STOCHF (Fast Stochastic) of RSI
        stoch = ta.STOCHF(
            pd.DataFrame({
                "high":  rsi,
                "low":   rsi,
                "close": rsi,
            }),
            fastk_period=p["stochrsi_period"],
            fastd_period=p["stochrsi_smooth_d"],
            fastk_matype=0,
            fastd_matype=0,
        )
        dataframe["stochrsi_k"] = stoch["fastk"]
        dataframe["stochrsi_d"] = stoch["fastd"]

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

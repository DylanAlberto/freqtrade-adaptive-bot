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
"""

from pathlib     import Path
from typing      import Dict, Any, Optional

import numpy as np
import pandas as pd

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    CategoricalParameter,
    informative,
)
from freqtrade.persistence import Trade
from pandas import DataFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approx_cvd(dataframe: DataFrame, period: int = 1) -> pd.Series:
    """
    Approximate Cumulative Volume Delta from OHLCV data.
    delta  = (close - open) * volume    — raw delta per candle
    cvd    = cumulative sum of delta
    """
    delta = (dataframe["close"] - dataframe["open"]) * dataframe["volume"]
    return delta.cumsum()


def _load_pair_params(path: str) -> Dict[str, Any]:
    """
    Load per-pair parameter dictionary from a JSON file.
    Falls back to empty dict so the strategy defaults always apply.
    """
    import json
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r") as f:
        raw = json.load(f)
        # strip metadata key if present
        return {k: v for k, v in raw.items() if k != "metadata"}


# ---------------------------------------------------------------------------
# Default parameters (used when pair_params.json entry is missing)
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

    Parameters are loaded from ``pair_params.json`` at the strategy root,
    normally produced by Freqtrade's Bayesian Hyperopt engine.
    """

    # --- Freqtrade required attributes ---------------------------------
    INTERFACE_VERSION = 3

    can_short = True

    # These timeframes are overridden per-pair at runtime once we know the
    # pair-specific parameters. We set sensible defaults here.
    timeframe: str = "5m"
    informative_timeframe: str = "1h"
    minimum_informative_tf: str = "1h"

    # Stoploss is dynamic (custom_stoploss), but Freqtrade needs a hard
    # default as fallback.
    stoploss = -0.05

    # ROI table — disabled; exit is signal-based
    minimal_roi = {"0": 100.0}

    # Tracer
    startup_candle_count: int = 100

    # Hyperopt spaces (used when pair_params.json is absent)
    stochrsi_period   = IntParameter(5, 25, default=10, space="buy")
    stochrsi_oversold = IntParameter(10, 35, default=20, space="buy")
    stochrsi_overbought = IntParameter(65, 90, default=80, space="sell")
    cvd_threshold     = DecimalParameter(0.05, 0.50, default=0.20, space="buy")
    atr_stop_mult     = DecimalParameter(1.0, 4.0, default=2.0, space="sell")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        # Locate the parameter file relative to the strategy file
        strategy_dir = Path(__file__).resolve().parent
        params_file   = strategy_dir.parent.parent / "pair_params.json"

        self.pair_params: Dict[str, Any] = _load_pair_params(str(strategy_dir.parent.parent / "pair_params.json"))

        if self.pair_params:
            self.log.info(f"Loaded per-pair parameters for {list(self.pair_params.keys())}")
        else:
            self.log.info("No pair_params.json found — using defaults + hyperopt parameters")

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

        # Merge with global hyperopt parameters when *not* per-pair
        resolved = {
            "stochrsi_period":       params["stochrsi"]["period"],
            "stochrsi_smooth_k":     params["stochrsi"]["smooth_k"],
            "stochrsi_smooth_d":     params["stochrsi"]["smooth_d"],
            "stochrsi_oversold":     params["stochrsi"]["oversold"],
            "stochrsi_overbought":   params["stochrsi"]["overbought"],
            "cvd_delta_period":      params["cvd"]["delta_period"],
            "cvd_threshold":         params["cvd"]["threshold"],
            "atr_multiplier_stop":   params["risk"]["atr_multiplier_stop"],
            "atr_multiplier_entry":  params["risk"]["atr_multiplier_entry"],
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

        # --- 1. CVD (raw + approximated) on current timeframe ---------
        dataframe["cvd_raw"]  = (dataframe["close"] - dataframe["open"]) * dataframe["volume"]
        dataframe["cvd"]      = _approx_cvd(dataframe)

        # --- 2. Stochastic RSI -----------------------------------------
        self._populate_stochrsi(dataframe, p)

        # --- 3. ATR ----------------------------------------------------
        import talib.abstract as ta
        dataframe["atr"] = ta.ATR(
            dataframe,
            timeperiod=max(p["stochrsi_period"], 14),
        )

        # --- 4. CVD rate-of-change (regime detection) ------------------
        delta_series = dataframe["cvd"].diff(p["cvd_delta_period"])
        dataframe["cvd_delta"] = delta_series
        dataframe["cvd_delta_abs"] = delta_series.abs()
        dataframe["regime_trend"] = (dataframe["cvd_delta_abs"] > p["cvd_threshold"]).astype(int)

        # --- 5. Divergence signal --------------------------------------
        # Price dropping but CVD increasing → bullish absorption
        dataframe["cvd_div_pos"] = (
            (dataframe["close"].diff(3) < 0) &
            (dataframe["cvd"].diff(3) > 0)
        ).astype(int)

        # Price rising but CVD decreasing → bearish distribution
        dataframe["cvd_div_neg"] = (
            (dataframe["close"].diff(3) > 0) &
            (dataframe["cvd"].diff(3) < 0)
        ).astype(int)

        # --- 6. CVD on 1H informative (via merge) ---------------------- 
        # Attach 1H CVD for regime detection on the *current* candle
        dataframe = self._attach_informative_cvd(dataframe, metadata)

        return dataframe

    # ------------------------------------------------------------------
    # Buy / Short signals
    # ------------------------------------------------------------------

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        p    = self._params_for(pair)

        # Conditions shared across regimes
        dataframe.loc[:, "enter_long"]  = 0
        dataframe.loc[:, "enter_short"] = 0

        # --- Trend Regime ----------------------------------------------
        trend_mask = dataframe["regime_trend"] == 1

        # LONG: StochRSI_k crossing above oversold + CVD delta positive
        dataframe.loc[
            trend_mask &
            qtpylib.crossed_above(dataframe["stochrsi_k"], p["stochrsi_oversold"]) &
            (dataframe["cvd_delta"] > 0),
            "enter_long",
        ] = 1

        # SHORT: StochRSI_k crossing below overbought + CVD delta negative
        dataframe.loc[
            trend_mask &
            qtpylib.crossed_below(dataframe["stochrsi_k"], p["stochrsi_overbought"]) &
            (dataframe["cvd_delta"] < 0),
            "enter_short",
        ] = 1

        # --- Range Regime ----------------------------------------------
        range_mask = dataframe["regime_trend"] == 0

        # LONG: oversold bounce
        dataframe.loc[
            range_mask &
            qtpylib.crossed_above(dataframe["stochrsi_k"], p["stochrsi_oversold"]),
            "enter_long",
        ] = 1

        # SHORT: overbought rejection
        dataframe.loc[
            range_mask &
            qtpylib.crossed_below(dataframe["stochrsi_k"], p["stochrsi_overbought"]),
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

        # Exit LONG when StochRSI reaches overbought in range regime
        # or when CVD divergence turns bearish
        dataframe.loc[
            (dataframe["stochrsi_k"] >= p["stochrsi_overbought"]) |
            (dataframe["cvd_div_neg"] == 1),
            "exit_long",
        ] = 1

        # Exit SHORT when StochRSI reaches oversold in range regime
        # or when CVD divergence turns bullish
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

        # ATR comes from the last available row for this pair
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss  # fallback

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get("atr")

        if atr is None or np.isnan(atr) or atr == 0:
            return self.stoploss

        # Dynamic stop = ATR * multiplier, as fraction of entry price
        atr_stop_pct = (atr * p["atr_multiplier_stop"]) / trade.open_rate
        return -min(atr_stop_pct, 0.15)  # cap at 15 %

    # ================================================================
    # Private helpers
    # ================================================================

    def _populate_stochrsi(self, dataframe: DataFrame, p: Dict[str, Any]) -> None:
        """Add StochRSI columns to *dataframe*."""
        import talib.abstract as ta

        # RSI first
        rsi = ta.RSI(dataframe, timeperiod=p["stochrsi_period"])

        # Stochastic of RSI
        stoch = ta.STOCHF(
            pd.DataFrame({
                "high":   rsi,
                "low":    rsi,
                "close":  rsi,
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
        Merge 1H CVD delta into the current dataframe for regime detection.
        Uses Freqtrade's built-in informative pair helper.
        """
        pair = metadata["pair"]
        p    = self._params_for(pair)

        # We rely on the informative caching mechanism of Freqtrade.
        # If informative data is not ready (first candles), fall through.
        try:
            informative = self.dp.get_pair_dataframe(pair, self.informative_timeframe)
            if informative is None or informative.empty:
                return dataframe

            # Compute CVD on the 1H dataframe
            informative["cvd_1h_raw"] = (
                (informative["close"] - informative["open"]) * informative["volume"]
            )
            informative["cvd_1h"] = informative["cvd_1h_raw"].cumsum()

            informative["cvd_1h_delta"] = informative["cvd_1h"].diff(p["cvd_delta_period"] * 3)
            informative["cvd_1h_delta_abs"] = informative["cvd_1h_delta"].abs()

            informative.rename(
                columns={
                    "cvd_1h":           "cvd_1h",
                    "cvd_1h_delta":     "cvd_1h_delta",
                    "cvd_1h_delta_abs": "cvd_1h_delta_abs",
                },
                inplace=True,
            )

            # Merge informative columns onto the base dataframe
            columns = ["date", "cvd_1h", "cvd_1h_delta", "cvd_1h_delta_abs"]
            dataframe = merge_informative_pair(
                dataframe, informative[columns],
                self.timeframe, self.informative_timeframe,
                ffill=True,
            )

        except Exception as exc:
            self.log.warning(f"CVD informative merge failed for {pair}: {exc}")

        return dataframe


# ===================================================================
# Required for Freqtrade discovery
# ===================================================================
try:
    import freqtrade.vendor.qtpylib.indicators as qtpylib
except ImportError:
    qtpylib = None

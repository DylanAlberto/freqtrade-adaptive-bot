"""
MultiPairHyperoptLoss — Custom loss function for V24 exit optimization
========================================================================
Optimiza 3 variables de salida de forma 100% matemática:

  1. ATR trailing stop distance     (atr_stop_mult:       1.0 - 4.0)
  2. Breakeven activation threshold (atr_trail_offset:    1.0 - 3.0)
  3. Time-Exit minute               (max_trade_minutes_p: 120 - 480)

Métrica objetivo: Sortino Ratio (castiga volatilidad negativa).
La función busca el máximo Sortino combinado con un penalty por
drawdown excesivo (> 5%) y un bonus por número de trades (> 30).
"""

from freqtrade.optimize.hyperopt import IHyperOptLoss
from pandas import DataFrame
import numpy as np


class MultiPairHyperoptLoss(IHyperOptLoss):
    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: int,
        max_date: int,
        config: dict,
        processed: dict[str, DataFrame],
        backtest_stats: dict,
        *args,
        **kwargs,
    ) -> float:
        """
        Sortino maximization with drawdown penalty + trade count bonus.
        Returns a value to MINIMIZE. Higher Sortino → lower return value.
        """
        # --- Sortino Ratio from backtest stats -------------------------
        sortino = backtest_stats.get("sortino", 0.0)
        if sortino is None or np.isnan(sortino) or np.isinf(sortino):
            sortino = 0.0

        # --- Max drawdown penalty --------------------------------------
        max_dd = backtest_stats.get("max_drawdown_account", 0.0)
        if max_dd is None or np.isnan(max_dd):
            max_dd = 0.0
        # absolute value: 0.05 = 5% drawdown
        dd_penalty = 0.0
        if max_dd > 0.05:
            dd_penalty = (max_dd - 0.05) * 10  # 1% extra DD → 0.1 penalty

        # --- Trade count bonus (avoid overfitting on < 30 trades) ------
        trade_bonus = 0.0
        if trade_count < 30:
            trade_bonus = (30 - trade_count) * 0.01

        # --- Loss value: higher Sortino → lower loss  ------------------
        # Sortino típico entre 0 y 2. Convertimos a loss:
        #   Sortino 2.0 → loss = -0.5 + penalty
        #   Sortino 0.5 → loss =  0.5 + penalty
        #   Sortino 0.0 → loss =  1.0 + penalty
        base_loss = max(1.0 - sortino, 0.01)

        total_loss = base_loss + dd_penalty + trade_bonus

        # Ensure positive (freqtrade expects positive loss to minimize)
        return max(total_loss, 0.001)

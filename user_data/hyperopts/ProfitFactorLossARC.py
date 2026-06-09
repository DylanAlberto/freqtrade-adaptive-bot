"""
ProfitFactorLossARC — Loss function relajada para pares volátiles
==================================================================
Versión de ProfitFactorLoss con penalización de stops ajustada:
  - Límite soft:  3.0× ATR  (≈6% de pérdida) en vez de 2.5×
  - Límite hard:  3.5× ATR  (≈7% de pérdida) en vez de 3.0×

Esto permite que pares como ARC, con mayor volatilidad intrínseca
que WLD, tengan más respiro sin ser penalizados injustamente.

Uso:
  freqtrade hyperopt --hyperopt-loss ProfitFactorLossARC --spaces buy sell
"""

from freqtrade.optimize.hyperopt import IHyperOptLoss
from pandas import DataFrame
import numpy as np


class ProfitFactorLossARC(IHyperOptLoss):
    """
    Profit Factor maximization with RELAXED ATR stop-distance penalty.
    Soft limit: 3.0× ATR | Hard limit: 3.5× ATR
    """

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
        # ================================================================
        # 1. PROFIT FACTOR — métrica primaria
        # ================================================================
        gross_profit = results.loc[results["profit_abs"] > 0, "profit_abs"].sum()
        gross_loss = abs(results.loc[results["profit_abs"] < 0, "profit_abs"].sum())

        if gross_loss > 0 and gross_profit > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0 and gross_loss == 0:
            profit_factor = 999.0
        else:
            profit_factor = 0.0

        base_loss = 1.0 / max(profit_factor, 0.01) if profit_factor > 0 else 10.0

        # ================================================================
        # 2. STOP PENALTY — RELAJADA para pares volátiles
        #    Soft: stop > 6%  (≈3.0× ATR)
        #    Hard: stop > 7%  (≈3.5× ATR)
        # ================================================================
        stop_loss_mask = results["exit_reason"].str.contains(
            "stop_loss", case=False, na=False
        )
        stop_trades = results[stop_loss_mask]

        stop_penalty = 0.0
        if len(stop_trades) > 0:
            max_stop_loss = stop_trades["profit_ratio"].min()

            # Soft: stop entre -6% y -7%
            if max_stop_loss < -0.06:
                soft_excess = abs(max_stop_loss) - 0.06
                soft_penalty = soft_excess * 40  # 1% extra → 0.4 penalty

                # Hard: stop > -7%
                if max_stop_loss < -0.07:
                    hard_excess = abs(max_stop_loss) - 0.07
                    hard_penalty = hard_excess * 150  # 1% extra → 1.5 penalty
                else:
                    hard_penalty = 0.0

                stop_penalty = soft_penalty + hard_penalty
            else:
                stop_penalty = -0.05  # bonus por stops contenidos

        # Asimetría: si pérdidas por stop > 60% de ganancias → penaliza
        if len(stop_trades) > 0 and gross_profit > 0:
            stop_total_loss = abs(stop_trades["profit_abs"].sum())
            if stop_total_loss > gross_profit * 0.6:
                asym_penalty = (stop_total_loss / gross_profit) - 0.6
                stop_penalty += asym_penalty * 2

        # ================================================================
        # 3. DRAWDOWN PENALTY
        # ================================================================
        max_dd = backtest_stats.get("max_drawdown_account", 0.0)
        if max_dd is None or np.isnan(max_dd):
            max_dd = 0.0

        if max_dd > 0.06:
            dd_penalty = (max_dd - 0.06) * 15 + 0.4
        elif max_dd > 0.03:
            dd_penalty = (max_dd - 0.03) * 8
        else:
            dd_penalty = -0.05

        # ================================================================
        # 4. TRADE FLOOR
        # ================================================================
        trade_penalty = (20 - trade_count) * 0.1 if trade_count < 20 else -0.05

        # ================================================================
        # 5. LOSS TOTAL
        # ================================================================
        total_loss = base_loss + stop_penalty + dd_penalty + trade_penalty

        epoch = kwargs.get("current_epoch", 0)
        if epoch % 15 == 0:
            print(
                f"  [ARC-HO] PF={profit_factor:.2f} base={base_loss:.4f} "
                f"stop={stop_penalty:.4f} dd={dd_penalty:.4f} "
                f"trade={trade_penalty:.4f} total={total_loss:.4f}"
            )

        return max(total_loss, 0.001)

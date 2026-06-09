"""
ProfitFactorLoss — Custom Hyperopt Loss para V24+
=====================================================
Reemplaza la métrica Sortino por defecto y reorienta el espacio
de búsqueda hacia la maximización del Profit Factor con una
penalización algorítmica severa sobre stops excesivos.

Matemática de la función:
--------------------------
  loss = 1/PF + stop_penalty + dd_penalty + trade_floor

Donde:
  - PF = gross_profit / gross_loss  (Profit Factor)
  - stop_penalty: escala progresivamente si el stop loss individual
    supera el 5% (≈2.5× ATR) o el 6% (≈3.0× ATR)
  - dd_penalty: castiga drawdown > 2%
  - trade_floor: fuerza mínimo 20 trades para evitar sobreajuste

Así, el algoritmo busca el balance ideal entre:
  ✅ Alta tasa de acierto (PF > 1.5)
  ✅ Stops ajustados (pérdida individual < 5%)
  ✅ Drawdown mínimo (< 2%)
  ✅ Suficientes trades para significancia estadística

Uso:
  freqtrade hyperopt --hyperopt-loss ProfitFactorLoss --spaces sell
"""

from freqtrade.optimize.hyperopt import IHyperOptLoss
from pandas import DataFrame
import numpy as np


class ProfitFactorLoss(IHyperOptLoss):
    """
    Profit Factor maximization with ATR stop-distance penalty.
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
            profit_factor = 999.0  # all winners
        else:
            profit_factor = 0.0  # all losers or no trades

        # Convert PF to loss: PF=2.0→loss=0.5, PF=1.0→loss=1.0, PF=0.5→loss=2.0
        if profit_factor > 0:
            base_loss = 1.0 / max(profit_factor, 0.01)
        else:
            base_loss = 10.0  # catastrophic

        # ================================================================
        # 2. STOP PENALTY — castiga stops más allá de 2.5× / 3.0× ATR
        # ================================================================
        # Identificamos trades cerrados por trailing_stop_loss
        stop_loss_mask = results["exit_reason"].str.contains(
            "stop_loss", case=False, na=False
        )
        stop_trades = results[stop_loss_mask]

        stop_penalty = 0.0
        if len(stop_trades) > 0:
            # profit_ratio negativo = pérdida (ej: -0.05 = -5%)
            max_stop_loss = stop_trades["profit_ratio"].min()

            # Soft penalty: stop entre -5% y -6% (≈2.5× a 3.0× ATR)
            if max_stop_loss < -0.05:
                # Escala lineal: a -5.5% → penalty = 0.5
                soft_excess = abs(max_stop_loss) - 0.05
                soft_penalty = soft_excess * 50  # 1% extra → 0.5 penalty

                # Hard penalty: stop > -6.0% (> 3.0× ATR) — severa
                if max_stop_loss < -0.06:
                    hard_excess = abs(max_stop_loss) - 0.06
                    hard_penalty = hard_excess * 200  # 1% extra → 2.0 penalty
                else:
                    hard_penalty = 0.0

                stop_penalty = soft_penalty + hard_penalty

            # Bonus: si NO hay stops > -5%, reducimos loss (premio)
            else:
                stop_penalty = -0.05  # pequeño bonus por stops ajustados

        # También penalizamos si el stop-loss total es desproporcionado
        # respecto al profit bruto (mala asimetría)
        if len(stop_trades) > 0 and gross_profit > 0:
            stop_total_loss = abs(stop_trades["profit_abs"].sum())
            if stop_total_loss > gross_profit * 0.5:
                # Las pérdidas por stop son > 50% de las ganancias → mala asimetría
                asymmetry_penalty = (stop_total_loss / gross_profit) - 0.5
                stop_penalty += asymmetry_penalty * 2

        # ================================================================
        # 3. DRAWDOWN PENALTY — protege el bajísimo DD del baseline
        # ================================================================
        max_dd = backtest_stats.get("max_drawdown_account", 0.0)
        if max_dd is None or np.isnan(max_dd):
            max_dd = 0.0

        # Penaliza suavemente DD > 2%, severamente DD > 5%
        if max_dd > 0.05:
            dd_penalty = (max_dd - 0.05) * 20 + 0.3  # 5%→0.3, 10%→1.3
        elif max_dd > 0.02:
            dd_penalty = (max_dd - 0.02) * 10  # 2%→0.0, 5%→0.3
        else:
            dd_penalty = -0.05  # bonus por DD excelente

        # ================================================================
        # 4. TRADE FLOOR — evita overfitting en pocos trades
        # ================================================================
        if trade_count < 20:
            trade_penalty = (20 - trade_count) * 0.1
        else:
            trade_penalty = -0.05  # pequeño bonus por suficientes trades

        # ================================================================
        # 5. LOSS TOTAL
        # ================================================================
        total_loss = base_loss + stop_penalty + dd_penalty + trade_penalty

        logger_msg = (
            f"PF={profit_factor:.2f} base={base_loss:.4f} "
            f"stop_pen={stop_penalty:.4f} dd_pen={dd_penalty:.4f} "
            f"trade_pen={trade_penalty:.4f} total={total_loss:.4f}"
        )

        # Solo log cada 10 épocas para no saturar
        epoch = kwargs.get("current_epoch", 0)
        if epoch % 10 == 0:
            print(f"  [Hyperopt] {logger_msg}")

        return max(total_loss, 0.001)

"""Metrics contract — Hydro-Alpha project.

Standard regression metrics are complemented with financial signal metrics:

  IC  (Information Coefficient) — Spearman rank correlation between predicted
      and realised returns. The industry standard for evaluating alpha signals.
      IC > 0.05 is considered useful in practice; IC > 0.10 is strong.

  ICIR (IC Information Ratio) — IC / std(IC) measured on a rolling basis.
      A measure of signal consistency, not just average strength.

  Hit Rate — proportion of correctly predicted directions (sign of return).
      0.5 = random; anything above 0.55 is practically significant.

  Sharpe (signal) — annualised Sharpe ratio of a long/short portfolio that
      goes long IDA when predicted excess return > 0, short otherwise.
      Assumes daily rebalancing, no transaction costs (upper bound).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return signal quality and regression metrics.

    Args:
        y_true: realised IDA excess returns over XLU (forward 20 days)
        y_pred: model predictions of those excess returns

    Returns:
        Dictionary of floats, written to results/model_metrics.csv.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Remove any remaining NaNs (should be none after data.py cleaning)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    # ── Signal quality (financial) ──────────────────────────────────────────
    ic, _   = stats.spearmanr(y_true, y_pred)
    hit_rate = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

    # Long/short signal: long when pred > 0, short when pred < 0
    ls_returns = np.where(y_pred > 0, y_true, -y_true)
    sharpe = (
        float(ls_returns.mean() / ls_returns.std() * np.sqrt(252 / 20))
        if ls_returns.std() > 0 else 0.0
    )

    # ── Regression quality ──────────────────────────────────────────────────
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    return {
        "ic":       round(float(ic), 6),
        "hit_rate": round(hit_rate, 6),
        "sharpe":   round(sharpe, 6),
        "rmse":     round(rmse, 6),
        "mae":      round(mae, 6),
        "r2":       round(r2, 6),
    }

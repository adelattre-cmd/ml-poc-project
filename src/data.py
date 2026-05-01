"""Dataset loading contract — Hydro-Alpha project.

Builds a supervised ML dataset from:
  - USGS daily streamflow (Columbia, Snake, Willamette, Deschutes rivers)
  - IDACORP (IDA) and XLU adjusted close prices

Target:
  IDA excess return over XLU, forward FORWARD_DAYS trading days.
  A positive target means IDA outperformed the utilities sector.

Feature engineering:
  For each river gauge, we compute:
    - flow_zscore   : (current flow − weekly_mean) / weekly_std
                      removes seasonality, isolates the anomaly
    - flow_pct      : percentile rank within the calendar-week distribution
                      (0 = historically driest, 1 = historically wettest)
    - flow_trend    : 30-day rolling slope (momentum in flow)
    - flow_deficit  : 90-day cumulative z-score (measures sustained drought)
  Plus:
    - sin/cos week-of-year encoding (residual seasonality signal)
    - IDA momentum  : 20-day return, captures stock trend
    - rel_momentum  : IDA 20d return − XLU 20d return (relative strength)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from config import DATA_DIR, FORWARD_DAYS, TARGET_TICKER, BENCH_TICKER

HYDRO_DIR    = DATA_DIR / "raw" / "hydro"
FLOW_FILE    = HYDRO_DIR / "usgs_streamflow_daily.csv"
STOCKS_FILE  = HYDRO_DIR / "stock_prices_daily.csv"

RIVERS = ["columbia", "snake", "willamette", "deschutes"]

# Chronological split — never shuffle a time series
TRAIN_END = "2018-12-31"
TEST_START = "2019-01-01"


# ── Feature engineering helpers ────────────────────────────────────────────────

def _weekly_zscore(series: pd.Series) -> pd.Series:
    """Subtract the historical mean for that calendar week, divide by std.

    This removes the strong seasonal cycle (snowmelt peaks in May-June)
    so the model sees pure anomalies, not just 'it is spring'.
    """
    week = series.index.isocalendar().week.astype(int)
    z = series.copy() * np.nan
    for w in range(1, 54):
        mask = week == w
        if mask.sum() < 10:
            continue
        mu  = series[mask].mean()
        sig = series[mask].std()
        if sig > 0:
            z[mask] = (series[mask] - mu) / sig
    return z


def _weekly_percentile(series: pd.Series) -> pd.Series:
    """Percentile rank within the calendar-week historical distribution."""
    week = series.index.isocalendar().week.astype(int)
    pct = series.copy() * np.nan
    for w in range(1, 54):
        mask = week == w
        vals = series[mask].dropna()
        if len(vals) < 10:
            continue
        pct[mask] = series[mask].apply(
            lambda x: float(stats.percentileofscore(vals, x, kind="rank")) / 100
            if pd.notna(x) else np.nan
        )
    return pct


def _rolling_trend(series: pd.Series, window: int = 30) -> pd.Series:
    """Rolling OLS slope (normalised by mean) — direction of flow momentum."""
    slopes = series.rolling(window, min_periods=window // 2).apply(
        lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if np.isfinite(y).sum() > 3 else np.nan,
        raw=True,
    )
    mean_val = series.mean()
    return slopes / mean_val if mean_val != 0 else slopes


def _cumulative_deficit(zscore: pd.Series, window: int = 90) -> pd.Series:
    """Rolling sum of z-scores — captures sustained drought vs. single-day dips."""
    return zscore.rolling(window, min_periods=30).sum()


# ── Main feature builder ───────────────────────────────────────────────────────

def build_features(flow: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full feature matrix aligned to trading days."""
    features = {}

    for river in RIVERS:
        col = f"discharge_cfs_{river}"
        if col not in flow.columns:
            continue
        s = flow[col].astype(float)

        features[f"{river}_zscore"]  = _weekly_zscore(s)
        features[f"{river}_pct"]     = _weekly_percentile(s)
        features[f"{river}_trend"]   = _rolling_trend(s, window=30)
        features[f"{river}_deficit"] = _cumulative_deficit(features[f"{river}_zscore"], window=90)

    feat_df = pd.DataFrame(features)

    # Seasonal encoding (residual after z-scoring)
    week = feat_df.index.isocalendar().week.astype(int)
    feat_df["sin_week"] = np.sin(2 * np.pi * week / 52)
    feat_df["cos_week"] = np.cos(2 * np.pi * week / 52)

    # Stock-side features (momentum)
    ida = stocks[TARGET_TICKER].resample("D").last().ffill()
    xlu = stocks[BENCH_TICKER].resample("D").last().ffill()

    feat_df["ida_mom_20d"] = ida.pct_change(20)
    feat_df["rel_mom_20d"] = ida.pct_change(20) - xlu.pct_change(20)

    return feat_df.sort_index()


def build_target(stocks: pd.DataFrame, forward_days: int) -> pd.Series:
    """IDA excess return over XLU, shifted forward by `forward_days` trading days.

    We use trading-day forward returns to avoid weekend/holiday distortions.
    The target is computed on the stock DataFrame (trading days only) then
    reindexed to daily so it aligns with the flow features.
    """
    ida_td = stocks[TARGET_TICKER].dropna()
    xlu_td = stocks[BENCH_TICKER].dropna()

    ida_fwd = ida_td.shift(-forward_days) / ida_td - 1
    xlu_fwd = xlu_td.shift(-forward_days) / xlu_td - 1
    excess  = ida_fwd - xlu_fwd

    # Reindex to calendar days for alignment with flow data
    return excess.reindex(
        pd.date_range(excess.index.min(), excess.index.max(), freq="D")
    ).ffill(limit=3)


# ── Public contract ────────────────────────────────────────────────────────────

def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return (X_train, X_test, y_train, y_test) for the Hydro-Alpha project.

    Split is strictly chronological: train ≤ 2018, test ≥ 2019.
    No shuffling — shuffling a time series causes look-ahead bias.
    """
    flow   = pd.read_csv(FLOW_FILE,   index_col=0, parse_dates=True)
    stocks = pd.read_csv(STOCKS_FILE, index_col=0, parse_dates=True)

    X = build_features(flow, stocks)
    y = build_target(stocks, forward_days=FORWARD_DAYS)

    # Align on common index, drop rows with any NaN in X or y
    common = X.index.intersection(y.dropna().index)
    X = X.loc[common].dropna()
    y = y.loc[X.index]

    train_mask = X.index <= TRAIN_END
    test_mask  = X.index >= TEST_START

    return (
        X[train_mask], X[test_mask],
        y[train_mask], y[test_mask],
    )

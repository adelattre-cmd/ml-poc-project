"""Dataset loading contract — Hydro-Alpha project.

Builds a supervised ML dataset from:
  - USGS daily streamflow (Columbia, Snake, Willamette, Deschutes rivers)
  - IDACORP (IDA) and XLU adjusted close prices
  - ICE MID-C electricity spot prices (optional — see USE_ICE_FEATURES)
  - SNOTEL snowpack / SWE data (Idaho headwaters — leading indicator of flow)
  - Henry Hub natural gas prices (replacement cost signal for IDACORP)

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
  ICE electricity features:
    - midc_zscore   : seasonally-adjusted MID-C price anomaly
    - midc_vol_30d  : 30-day rolling volatility of MID-C prices
    - midc_trend    : 30-day rolling slope of MID-C prices
    - midc_spike    : binary flag for price > 90th percentile within calendar week
  Cross-river interaction features:
    - snake_columbia_ratio : Snake/Columbia flow ratio (hydropower capacity proxy)
    - min_zscore           : worst drought signal across all 4 rivers
    - mean_deficit         : average sustained drought across all rivers
    - drought_breadth      : fraction of rivers with z-score < -1 (systemic stress)
  Non-linear threshold features:
    - snake_drought_flag   : 1 when Snake z-score < -1 (critical level for IDA)
    - deficit_extreme      : 1 when Snake 90d deficit < -20 (severe sustained drought)
  Plus:
    - sin/cos week-of-year encoding (residual seasonality signal)
    - IDA momentum  : 20-day return, captures stock trend
    - rel_momentum  : IDA 20d return − XLU 20d return (relative strength)
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

import os

from config import DATA_DIR, FORWARD_DAYS, TARGET_TICKER, BENCH_TICKER, RESULTS_DIR

HYDRO_DIR    = DATA_DIR / "raw" / "hydro"
FLOW_FILE    = HYDRO_DIR / "usgs_streamflow_daily.csv"
STOCKS_FILE  = HYDRO_DIR / "stock_prices_daily.csv"
ICE_FILE     = HYDRO_DIR / "ice_midc_daily.csv"
SNOTEL_FILE  = HYDRO_DIR / "snotel_swe_daily.csv"
GAS_FILE     = HYDRO_DIR / "henry_hub_gas_daily.csv"

# ICE features are available but disabled by default: walk-forward CV showed
# they degrade IC in 4/5 folds (signal is redundant with streamflow z-scores).
# Set USE_ICE_FEATURES=1 to enable.
USE_ICE_FEATURES = os.environ.get("USE_ICE_FEATURES", "0") == "1"

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


# ── ICE electricity feature helpers ───────────────────────────────────────────

def _build_ice_features(ice: pd.DataFrame) -> pd.DataFrame:
    """Build electricity price features from ICE MID-C data."""
    price = ice["midc_price"].astype(float)
    feats: dict[str, pd.Series] = {}

    feats["midc_zscore"] = _weekly_zscore(price)
    feats["midc_vol_30d"] = price.pct_change(fill_method=None).rolling(30, min_periods=15).std() * np.sqrt(252)
    feats["midc_trend"] = _rolling_trend(price, window=30)

    week = price.index.isocalendar().week.astype(int)
    spike = price.copy() * np.nan
    for w in range(1, 54):
        mask = week == w
        vals = price[mask].dropna()
        if len(vals) < 10:
            continue
        p90 = vals.quantile(0.90)
        spike[mask] = (price[mask] > p90).astype(float)
    feats["midc_spike"] = spike

    return pd.DataFrame(feats)


# ── SNOTEL snowpack feature helpers ──────────────────────────────────────────

def _build_snotel_features(snotel: pd.DataFrame) -> pd.DataFrame:
    """Build snowpack features from SNOTEL SWE data."""
    swe = snotel["swe_mean"].astype(float)
    feats: dict[str, pd.Series] = {}

    feats["swe_zscore"] = _weekly_zscore(swe)
    feats["swe_pct"] = _weekly_percentile(swe)
    feats["swe_trend"] = _rolling_trend(swe, window=30)
    feats["swe_deficit"] = _cumulative_deficit(_weekly_zscore(swe), window=90)

    return pd.DataFrame(feats)


# ── Natural gas feature helpers ──────────────────────────────────────────────

def _build_gas_features(gas: pd.DataFrame) -> pd.DataFrame:
    """Build natural gas price features from Henry Hub data."""
    price = gas["gas_price"].astype(float)
    feats: dict[str, pd.Series] = {}

    feats["gas_zscore"] = _weekly_zscore(price)
    feats["gas_vol_30d"] = price.pct_change(fill_method=None).rolling(30, min_periods=15).std() * np.sqrt(252)
    feats["gas_trend"] = _rolling_trend(price, window=30)

    return pd.DataFrame(feats)


# ── Main feature builder ───────────────────────────────────────────────────────

def build_features(
    flow: pd.DataFrame,
    stocks: pd.DataFrame,
    ice: pd.DataFrame | None = None,
    snotel: pd.DataFrame | None = None,
    gas: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the full feature matrix aligned to trading days."""
    features = {}

    for river in RIVERS:
        col = f"discharge_cfs_{river}"
        if col not in flow.columns:
            continue
        s = flow[col].astype(float)

        z = _weekly_zscore(s)
        features[f"{river}_zscore"]  = z
        features[f"{river}_pct"]     = _weekly_percentile(s)
        features[f"{river}_trend"]   = _rolling_trend(s, window=30)
        features[f"{river}_deficit"] = _cumulative_deficit(z, window=90)

        # Lagged z-scores capture the causal delay (flow today → margin impact weeks later)
        features[f"{river}_zscore_7d"]  = z.shift(7)
        features[f"{river}_zscore_14d"] = z.shift(14)

    feat_df = pd.DataFrame(features)

    # ── Cross-river interactions ──────────────────────────────────────────
    if "discharge_cfs_snake" in flow.columns and "discharge_cfs_columbia" in flow.columns:
        snake_raw = flow["discharge_cfs_snake"].astype(float)
        columbia_raw = flow["discharge_cfs_columbia"].astype(float)
        ratio = snake_raw / columbia_raw.replace(0, np.nan)
        feat_df["snake_columbia_ratio"] = _weekly_zscore(ratio)

    zscore_cols = [c for c in feat_df.columns if c.endswith("_zscore") and not c.startswith("midc")]
    if zscore_cols:
        feat_df["min_zscore"] = feat_df[zscore_cols].min(axis=1)

    deficit_cols = [c for c in feat_df.columns if c.endswith("_deficit")]
    if deficit_cols:
        feat_df["mean_deficit"] = feat_df[deficit_cols].mean(axis=1)

    if zscore_cols:
        feat_df["drought_breadth"] = (feat_df[zscore_cols] < -1).sum(axis=1) / len(zscore_cols)

    # ── Non-linear threshold features ─────────────────────────────────────
    if "snake_zscore" in feat_df.columns:
        feat_df["snake_drought_flag"] = (feat_df["snake_zscore"] < -1).astype(float)

    if "snake_deficit" in feat_df.columns:
        feat_df["deficit_extreme"] = (feat_df["snake_deficit"] < -20).astype(float)

    # ICE MID-C electricity features
    if ice is not None and not ice.empty:
        ice_feats = _build_ice_features(ice)
        feat_df = feat_df.join(ice_feats, how="left")

    # SNOTEL snowpack features
    if snotel is not None and not snotel.empty:
        snotel_feats = _build_snotel_features(snotel)
        feat_df = feat_df.join(snotel_feats, how="left")

    # Natural gas features
    if gas is not None and not gas.empty:
        gas_feats = _build_gas_features(gas)
        feat_df = feat_df.join(gas_feats, how="left")

        # Interaction: low snowpack + high gas = margin squeeze for IDACORP
        if snotel is not None and "swe_zscore" in feat_df.columns:
            feat_df["swe_gas_interact"] = feat_df["swe_zscore"] * feat_df["gas_zscore"] * -1

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


def build_target(
    stocks: pd.DataFrame,
    forward_days: int,
    winsorize_std: float = 2.0,
) -> pd.Series:
    """IDA excess return over XLU, shifted forward by `forward_days` trading days.

    We use trading-day forward returns to avoid weekend/holiday distortions.
    The target is computed on the stock DataFrame (trading days only) then
    reindexed to daily so it aligns with the flow features.

    Winsorization clips extreme returns to ±winsorize_std standard deviations,
    reducing the influence of outlier events on model training.
    """
    ida_td = stocks[TARGET_TICKER].dropna()
    xlu_td = stocks[BENCH_TICKER].dropna()

    ida_fwd = ida_td.shift(-forward_days) / ida_td - 1
    xlu_fwd = xlu_td.shift(-forward_days) / xlu_td - 1
    excess  = ida_fwd - xlu_fwd

    if winsorize_std > 0:
        mu = excess.mean()
        sigma = excess.std()
        excess = excess.clip(mu - winsorize_std * sigma, mu + winsorize_std * sigma)

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

    ice = None
    if USE_ICE_FEATURES and ICE_FILE.exists():
        ice = pd.read_csv(ICE_FILE, index_col=0, parse_dates=True)

    snotel = None
    if SNOTEL_FILE.exists():
        snotel = pd.read_csv(SNOTEL_FILE, index_col=0, parse_dates=True)

    gas = None
    if GAS_FILE.exists():
        gas = pd.read_csv(GAS_FILE, index_col=0, parse_dates=True)

    X = build_features(flow, stocks, ice=ice, snotel=snotel, gas=gas)
    y = build_target(stocks, forward_days=FORWARD_DAYS)

    # Align on common index, drop rows with any NaN in X or y
    common = X.index.intersection(y.dropna().index)
    X = X.loc[common].dropna()
    y = y.loc[X.index]

    # Apply feature pruning if a trained selection exists
    selected_features_file = RESULTS_DIR / "selected_features.json"
    if selected_features_file.exists():
        selected = json.loads(selected_features_file.read_text())
        available = [f for f in selected if f in X.columns]
        if available:
            X = X[available]

    train_mask = X.index <= TRAIN_END
    test_mask  = X.index >= TEST_START

    return (
        X[train_mask], X[test_mask],
        y[train_mask], y[test_mask],
    )

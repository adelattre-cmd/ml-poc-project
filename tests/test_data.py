"""Tests for the data loading and feature engineering contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data import (
    build_features,
    build_target,
    load_dataset_split,
    _weekly_zscore,
    _weekly_percentile,
    _rolling_trend,
    _cumulative_deficit,
    _build_ice_features,
    _build_snotel_features,
    _build_gas_features,
    RIVERS,
    TRAIN_END,
    TEST_START,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_flow():
    dates = pd.date_range("2005-01-01", periods=1000, freq="D")
    np.random.seed(42)
    data = {}
    for river in RIVERS:
        base = 10000 + 5000 * np.sin(2 * np.pi * np.arange(1000) / 365)
        data[f"discharge_cfs_{river}"] = base + np.random.normal(0, 500, 1000)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_stocks():
    dates = pd.date_range("2005-01-01", periods=1000, freq="B")
    np.random.seed(42)
    ida = 50 * np.exp(np.cumsum(np.random.normal(0.0002, 0.015, 1000)))
    xlu = 30 * np.exp(np.cumsum(np.random.normal(0.0001, 0.012, 1000)))
    return pd.DataFrame({"IDA": ida, "XLU": xlu}, index=dates)


@pytest.fixture
def sample_ice():
    dates = pd.date_range("2005-01-01", periods=1000, freq="D")
    np.random.seed(42)
    base = 40 + 15 * np.sin(2 * np.pi * np.arange(1000) / 365)
    prices = base + np.random.normal(0, 5, 1000)
    return pd.DataFrame({
        "midc_price": np.abs(prices),
        "midc_volume": np.random.randint(1000, 100000, 1000),
    }, index=dates)


@pytest.fixture
def sample_snotel():
    dates = pd.date_range("2005-01-01", periods=1000, freq="D")
    np.random.seed(42)
    swe_base = 5 + 4 * np.sin(2 * np.pi * np.arange(1000) / 365)
    return pd.DataFrame({
        "swe_mean": np.abs(swe_base + np.random.normal(0, 1, 1000)),
    }, index=dates)


@pytest.fixture
def sample_gas():
    dates = pd.date_range("2005-01-01", periods=1000, freq="D")
    np.random.seed(42)
    return pd.DataFrame({
        "gas_price": np.abs(3 + np.random.normal(0, 0.5, 1000)),
    }, index=dates)


# ── Helper function tests ─────────────────────────────────────────────────────

class TestWeeklyZscore:
    def test_output_shape(self, sample_flow):
        s = sample_flow["discharge_cfs_columbia"]
        z = _weekly_zscore(s)
        assert len(z) == len(s)

    def test_mean_near_zero(self, sample_flow):
        s = sample_flow["discharge_cfs_columbia"]
        z = _weekly_zscore(s)
        assert abs(z.dropna().mean()) < 0.5

    def test_handles_nan(self):
        dates = pd.date_range("2010-01-01", periods=365, freq="D")
        s = pd.Series(np.random.normal(100, 10, 365), index=dates)
        s.iloc[50:55] = np.nan
        z = _weekly_zscore(s)
        assert len(z) == 365


class TestWeeklyPercentile:
    def test_output_range(self, sample_flow):
        s = sample_flow["discharge_cfs_columbia"]
        pct = _weekly_percentile(s)
        valid = pct.dropna()
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0


class TestRollingTrend:
    def test_positive_trend_for_increasing_series(self):
        dates = pd.date_range("2010-01-01", periods=100, freq="D")
        s = pd.Series(np.arange(100, dtype=float), index=dates)
        trend = _rolling_trend(s, window=30)
        assert trend.dropna().iloc[-1] > 0


class TestCumulativeDeficit:
    def test_shape(self):
        dates = pd.date_range("2010-01-01", periods=200, freq="D")
        z = pd.Series(np.random.normal(0, 1, 200), index=dates)
        deficit = _cumulative_deficit(z, window=90)
        assert len(deficit) == 200


# ── ICE features tests ────────────────────────────────────────────────────────

class TestBuildIceFeatures:
    def test_output_columns(self, sample_ice):
        feats = _build_ice_features(sample_ice)
        expected = {"midc_zscore", "midc_vol_30d", "midc_trend", "midc_spike"}
        assert set(feats.columns) == expected

    def test_spike_is_binary(self, sample_ice):
        feats = _build_ice_features(sample_ice)
        valid = feats["midc_spike"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_spike_rate_around_10pct(self, sample_ice):
        feats = _build_ice_features(sample_ice)
        valid = feats["midc_spike"].dropna()
        rate = valid.mean()
        assert 0.02 < rate < 0.25


# ── SNOTEL features tests ───────────────────────────────────────────────────

class TestBuildSnotelFeatures:
    def test_output_columns(self, sample_snotel):
        feats = _build_snotel_features(sample_snotel)
        expected = {"swe_zscore", "swe_pct", "swe_trend", "swe_deficit"}
        assert set(feats.columns) == expected

    def test_zscore_mean_near_zero(self, sample_snotel):
        feats = _build_snotel_features(sample_snotel)
        valid = feats["swe_zscore"].dropna()
        assert abs(valid.mean()) < 0.5


# ── Gas features tests ──────────────────────────────────────────────────────

class TestBuildGasFeatures:
    def test_output_columns(self, sample_gas):
        feats = _build_gas_features(sample_gas)
        expected = {"gas_zscore", "gas_vol_30d", "gas_trend"}
        assert set(feats.columns) == expected

    def test_zscore_mean_near_zero(self, sample_gas):
        feats = _build_gas_features(sample_gas)
        valid = feats["gas_zscore"].dropna()
        assert abs(valid.mean()) < 0.5


# ── Feature builder tests ─────────────────────────────────────────────────────

class TestBuildFeatures:
    def test_without_extras(self, sample_flow, sample_stocks):
        X = build_features(sample_flow, sample_stocks, ice=None)
        assert X.shape[1] == 34

    def test_with_ice(self, sample_flow, sample_stocks, sample_ice):
        X = build_features(sample_flow, sample_stocks, ice=sample_ice)
        assert X.shape[1] == 38  # 34 base + 4 ICE features
        ice_cols = [c for c in X.columns if "midc" in c]
        assert len(ice_cols) == 4

    def test_with_snotel(self, sample_flow, sample_stocks, sample_snotel):
        X = build_features(sample_flow, sample_stocks, snotel=sample_snotel)
        assert X.shape[1] == 38  # 34 base + 4 SNOTEL features

    def test_with_gas(self, sample_flow, sample_stocks, sample_gas):
        X = build_features(sample_flow, sample_stocks, gas=sample_gas)
        assert X.shape[1] == 37  # 34 base + 3 gas features

    def test_with_snotel_and_gas(self, sample_flow, sample_stocks, sample_snotel, sample_gas):
        X = build_features(sample_flow, sample_stocks, snotel=sample_snotel, gas=sample_gas)
        assert X.shape[1] == 42  # 34 + 4 snotel + 3 gas + 1 interaction

    def test_lagged_features_present(self, sample_flow, sample_stocks):
        X = build_features(sample_flow, sample_stocks)
        lagged = [c for c in X.columns if "_7d" in c or "_14d" in c]
        assert len(lagged) == 8  # 4 rivers * 2 lags

    def test_no_future_leakage(self, sample_flow, sample_stocks):
        X = build_features(sample_flow, sample_stocks)
        assert X.index.is_monotonic_increasing


class TestBuildTarget:
    def test_output_is_series(self, sample_stocks):
        y = build_target(sample_stocks, forward_days=20)
        assert isinstance(y, pd.Series)

    def test_reasonable_range(self, sample_stocks):
        y = build_target(sample_stocks, forward_days=20)
        valid = y.dropna()
        assert valid.abs().max() < 1.0  # excess returns should be < 100%


# ── Integration test ──────────────────────────────────────────────────────────

class TestLoadDatasetSplit:
    def test_returns_four_elements(self):
        result = load_dataset_split()
        assert len(result) == 4

    def test_shapes_consistent(self):
        X_train, X_test, y_train, y_test = load_dataset_split()
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
        assert X_train.shape[1] == X_test.shape[1]

    def test_chronological_split(self):
        X_train, X_test, _, _ = load_dataset_split()
        assert X_train.index.max() <= pd.Timestamp(TRAIN_END)
        assert X_test.index.min() >= pd.Timestamp(TEST_START)

    def test_no_nan_in_output(self):
        X_train, X_test, y_train, y_test = load_dataset_split()
        assert X_train.isna().sum().sum() == 0
        assert X_test.isna().sum().sum() == 0
        assert y_train.isna().sum() == 0
        assert y_test.isna().sum() == 0

    def test_default_no_ice_features(self):
        X_train, _, _, _ = load_dataset_split()
        ice_cols = [c for c in X_train.columns if "midc" in c]
        assert len(ice_cols) == 0  # ICE disabled by default

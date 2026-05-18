"""Tests for the metrics computation contract."""

from __future__ import annotations

import numpy as np
import pytest

from metrics import compute_metrics


class TestComputeMetrics:
    def test_returns_dict(self):
        y_true = np.random.normal(0, 0.05, 100)
        y_pred = y_true + np.random.normal(0, 0.01, 100)
        result = compute_metrics(y_true, y_pred)
        assert isinstance(result, dict)

    def test_required_keys(self):
        y_true = np.random.normal(0, 0.05, 100)
        y_pred = y_true + np.random.normal(0, 0.01, 100)
        result = compute_metrics(y_true, y_pred)
        expected_keys = {"ic", "hit_rate", "sharpe", "rmse", "mae", "r2"}
        assert set(result.keys()) == expected_keys

    def test_all_values_are_float(self):
        y_true = np.random.normal(0, 0.05, 100)
        y_pred = y_true * 0.5
        result = compute_metrics(y_true, y_pred)
        for v in result.values():
            assert isinstance(v, float)

    def test_perfect_prediction(self):
        y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.05])
        result = compute_metrics(y_true, y_true)
        assert result["ic"] > 0.99
        assert result["hit_rate"] == 1.0
        assert result["rmse"] < 1e-10

    def test_random_prediction_low_ic(self):
        np.random.seed(42)
        y_true = np.random.normal(0, 0.05, 500)
        y_pred = np.random.normal(0, 0.05, 500)
        result = compute_metrics(y_true, y_pred)
        assert abs(result["ic"]) < 0.2

    def test_hit_rate_range(self):
        np.random.seed(42)
        y_true = np.random.normal(0, 0.05, 200)
        y_pred = y_true + np.random.normal(0, 0.02, 200)
        result = compute_metrics(y_true, y_pred)
        assert 0.0 <= result["hit_rate"] <= 1.0

    def test_handles_nan(self):
        y_true = np.array([0.01, np.nan, 0.03, -0.01, 0.05])
        y_pred = np.array([0.02, 0.01, np.nan, -0.02, 0.04])
        result = compute_metrics(y_true, y_pred)
        assert all(np.isfinite(v) for v in result.values())

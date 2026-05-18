"""Helpers for loading serialized models."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class EnsembleModel:
    """Wrapper that gives an IC-weighted ensemble dict a .predict() interface."""

    def __init__(self, ensemble_dict: dict) -> None:
        self.base_models = ensemble_dict["base_models"]
        self.weights = ensemble_dict["weights"]
        self.feature_names = ensemble_dict.get("feature_names")

    def predict(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and self.feature_names:
            available = [f for f in self.feature_names if f in X.columns]
            X = X[available]

        result = np.zeros(len(X) if not isinstance(X, pd.DataFrame) else X.shape[0])
        for name, model in self.base_models.items():
            w = self.weights.get(name, 0)
            if w == 0:
                continue
            if name == "xgboost":
                pipe = model
                imp = pipe.named_steps["imputer"]
                sc = pipe.named_steps["scaler"]
                reg = pipe.named_steps["reg"]
                X_np = sc.transform(imp.transform(X))
                result += w * reg.predict(X_np)
            else:
                result += w * model.predict(X)
        return result


def load_model(model_path: Path) -> Any:
    """Load a serialized model from disk.

    Supported formats are `.joblib`, `.pkl`, and `.pickle`.
    Stacking ensemble dicts are wrapped in EnsembleModel automatically.
    """

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    suffix = model_path.suffix.lower()

    if suffix == ".joblib":
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                "Loading `.joblib` files requires the `joblib` package. "
                "Add it to requirements.txt if needed."
            ) from exc

        obj = joblib.load(model_path)
        if isinstance(obj, dict) and "base_models" in obj and "meta_learner" in obj:
            return EnsembleModel(obj)
        return obj

    if suffix in {".pkl", ".pickle"}:
        with model_path.open("rb") as file_handle:
            obj = pickle.load(file_handle)
        if isinstance(obj, dict) and "base_models" in obj and "meta_learner" in obj:
            return EnsembleModel(obj)
        return obj

    raise ValueError(
        f"Unsupported model format for {model_path}. Use .joblib, .pkl, or .pickle."
    )

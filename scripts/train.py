"""Training script — Hydro-Alpha project.

Trains 3 regressors on the streamflow → excess return dataset and saves
them as sklearn Pipelines (imputer + scaler + model) to models/.

Run once before scripts/main.py:
    python scripts/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from data import load_dataset_split
from metrics import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def make_pipeline(regressor) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("reg",     regressor),
    ])


def train() -> None:
    print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_dataset_split()

    print(f"  Train : {len(X_train):,} obs  ({X_train.index.min().date()} → {X_train.index.max().date()})")
    print(f"  Test  : {len(X_test):,}  obs  ({X_test.index.min().date()} → {X_test.index.max().date()})")
    print(f"  Features : {X_train.shape[1]}")
    print(f"  Target mean (train) : {y_train.mean():.4f}  std : {y_train.std():.4f}")
    print()

    models = {
        "ridge": make_pipeline(
            Ridge(alpha=1.0)
        ),
        "random_forest": make_pipeline(
            RandomForestRegressor(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=30,   # avoid overfitting on financial data
                max_features=0.6,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "xgboost": make_pipeline(
            XGBRegressor(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_lambda=2.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )
        ),
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)

        m_train = compute_metrics(y_train, y_pred_train)
        m_test  = compute_metrics(y_test, y_pred_test)

        print(f"  Train  IC={m_train['ic']:+.4f}  Hit={m_train['hit_rate']:.3f}  Sharpe={m_train['sharpe']:+.3f}")
        print(f"  Test   IC={m_test['ic']:+.4f}  Hit={m_test['hit_rate']:.3f}  Sharpe={m_test['sharpe']:+.3f}")

        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        print(f"  Saved → {path}")
        print()

    print("Done.")


if __name__ == "__main__":
    train()

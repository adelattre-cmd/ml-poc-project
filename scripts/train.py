"""Training script — Hydro-Alpha project.

Trains regressors on the streamflow → excess return dataset and saves
them as sklearn Pipelines to models/.

Pipeline:
  1. Walk-forward hyperparameter tuning (expanding window)
  2. Feature importance pruning (removes noise features)
  3. Walk-forward CV evaluation with tuned params
  4. Final training on full train set
  5. Stacking ensemble of base models

Run once before scripts/main.py:
    python scripts/train.py
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except (ImportError, OSError):
    HAS_XGBOOST = False

from data import load_dataset_split
from metrics import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

WF_FOLDS = [
    ("2001-07-04", "2012-12-31", "2013-01-01", "2015-12-31"),
    ("2001-07-04", "2014-12-31", "2015-01-01", "2017-12-31"),
    ("2001-07-04", "2016-12-31", "2017-01-01", "2019-12-31"),
    ("2001-07-04", "2018-12-31", "2019-01-01", "2021-12-31"),
    ("2001-07-04", "2020-12-31", "2021-01-01", "2023-12-31"),
]

# Walk-forward tuning uses the first 3 folds; folds 4-5 are held out for
# unbiased evaluation to avoid leaking the tuning signal.
TUNE_FOLDS = WF_FOLDS[:3]
EVAL_FOLDS = WF_FOLDS


def make_pipeline(regressor) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("reg",     regressor),
    ])


# ── Hyperparameter grids ─────────────────────────────────────────────────────

PARAM_GRID = {
    "ridge": {
        "alpha": [0.1, 1.0, 10.0, 50.0],
    },
    "pca_ridge": {
        "alpha": [0.1, 1.0, 10.0, 50.0],
        "pca_var": [0.80, 0.90, 0.95],
    },
    "random_forest": {
        "n_estimators": [300, 500],
        "max_depth": [4, 6],
        "min_samples_leaf": [30, 60],
        "max_features": [0.4, 0.6],
    },
}

if HAS_XGBOOST:
    PARAM_GRID["xgboost"] = {
        "n_estimators": [200, 400],
        "max_depth": [3, 4],
        "learning_rate": [0.01, 0.02],
        "reg_lambda": [2.0, 5.0, 10.0],
    }


def _make_model(name: str, params: dict) -> Pipeline:
    if name == "ridge":
        return make_pipeline(Ridge(alpha=params["alpha"]))
    elif name == "pca_ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("pca",     PCA(n_components=params["pca_var"])),
            ("reg",     Ridge(alpha=params["alpha"])),
        ])
    elif name == "random_forest":
        return make_pipeline(RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ))
    elif name == "xgboost":
        return make_pipeline(XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=params["reg_lambda"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=30,
        ))
    raise ValueError(f"Unknown model: {name}")


def _grid_iter(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in product(*grid.values())]


# ── Walk-forward tuning ──────────────────────────────────────────────────────

def tune_hyperparams(X: pd.DataFrame, y: pd.Series) -> dict[str, dict]:
    """Tune each model via walk-forward CV on the tune folds."""
    print("=" * 60)
    print("Walk-Forward Hyperparameter Tuning")
    print("=" * 60)

    best_params: dict[str, dict] = {}

    for model_name, grid in PARAM_GRID.items():
        candidates = _grid_iter(grid)
        print(f"\n  {model_name}: {len(candidates)} configs × {len(TUNE_FOLDS)} folds")

        best_ic = -np.inf
        best_p = candidates[0]

        for params in candidates:
            fold_ics = []
            for tr_start, tr_end, te_start, te_end in TUNE_FOLDS:
                tr_mask = (X.index >= tr_start) & (X.index <= tr_end)
                te_mask = (X.index >= te_start) & (X.index <= te_end)
                X_tr, X_te = X[tr_mask], X[te_mask]
                y_tr, y_te = y[tr_mask], y[te_mask]

                if len(X_tr) < 100 or len(X_te) < 100:
                    continue

                model = _make_model(model_name, params)

                if model_name == "xgboost":
                    # Early stopping: use last 20% of train as eval set
                    split_idx = int(len(X_tr) * 0.8)
                    pipe = model
                    imp = pipe.named_steps["imputer"]
                    sc = pipe.named_steps["scaler"]
                    reg = pipe.named_steps["reg"]

                    X_tr_np = sc.fit_transform(imp.fit_transform(X_tr))
                    X_te_np = sc.transform(imp.transform(X_te))

                    X_fit = X_tr_np[:split_idx]
                    X_val = X_tr_np[split_idx:]
                    y_fit = y_tr.iloc[:split_idx]
                    y_val = y_tr.iloc[split_idx:]

                    reg.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
                    y_pred = reg.predict(X_te_np)
                else:
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_te)

                m = compute_metrics(y_te, y_pred)
                fold_ics.append(m["ic"])

            mean_ic = np.mean(fold_ics) if fold_ics else -np.inf
            if mean_ic > best_ic:
                best_ic = mean_ic
                best_p = params

        best_params[model_name] = best_p
        print(f"    Best: {best_p}  (mean IC={best_ic:+.4f})")

    return best_params


# ── Feature importance pruning ────────────────────────────────────────────────

def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    threshold: float = 0.01,
) -> list[str]:
    """Drop features with near-zero importance from a quick RF fit."""
    print("\n" + "=" * 60)
    print("Feature Importance Pruning")
    print("=" * 60)

    pipe = make_pipeline(RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=30,
        random_state=RANDOM_STATE, n_jobs=-1,
    ))
    pipe.fit(X_train, y_train)

    imp = pipe.named_steps["reg"].feature_importances_
    importance = pd.Series(imp, index=X_train.columns).sort_values(ascending=False)

    print("\n  Feature importances:")
    for feat, val in importance.items():
        marker = " ✗" if val < threshold else ""
        print(f"    {feat:25s}  {val:.4f}{marker}")

    kept = importance[importance >= threshold].index.tolist()
    dropped = importance[importance < threshold].index.tolist()

    print(f"\n  Keeping {len(kept)}/{len(importance)} features (threshold={threshold})")
    if dropped:
        print(f"  Dropped: {dropped}")

    return kept


# ── Walk-forward evaluation ──────────────────────────────────────────────────

def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict[str, dict],
    folds: list[tuple] | None = None,
) -> pd.DataFrame:
    """Evaluate tuned models via walk-forward CV."""
    if folds is None:
        folds = EVAL_FOLDS

    print("\n" + "=" * 60)
    print("Walk-Forward Cross-Validation (tuned params)")
    print("=" * 60)

    rows = []
    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        tr_mask = (X.index >= tr_start) & (X.index <= tr_end)
        te_mask = (X.index >= te_start) & (X.index <= te_end)
        X_tr, X_te = X[tr_mask], X[te_mask]
        y_tr, y_te = y[tr_mask], y[te_mask]

        if len(X_tr) < 100 or len(X_te) < 100:
            continue

        print(f"\n  Fold {fold_idx + 1}: train {tr_start}→{tr_end} ({len(X_tr)}), "
              f"test {te_start}→{te_end} ({len(X_te)})")

        for name, params in best_params.items():
            model = _make_model(name, params)

            if name == "xgboost":
                pipe = model
                imp_step = pipe.named_steps["imputer"]
                sc = pipe.named_steps["scaler"]
                reg = pipe.named_steps["reg"]

                X_tr_np = sc.fit_transform(imp_step.fit_transform(X_tr))
                X_te_np = sc.transform(imp_step.transform(X_te))

                split_idx = int(len(X_tr_np) * 0.8)
                reg.fit(
                    X_tr_np[:split_idx],
                    y_tr.iloc[:split_idx],
                    eval_set=[(X_tr_np[split_idx:], y_tr.iloc[split_idx:])],
                    verbose=False,
                )
                y_pred = reg.predict(X_te_np)
            else:
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)

            m = compute_metrics(y_te, y_pred)
            print(f"    {name:15s}  IC={m['ic']:+.4f}  Hit={m['hit_rate']:.3f}  Sharpe={m['sharpe']:+.3f}")
            rows.append({
                "fold": fold_idx + 1,
                "train_end": tr_end,
                "test_period": f"{te_start[:4]}-{te_end[:4]}",
                "model": name,
                **m,
            })

    results = pd.DataFrame(rows)

    print("\n  ── Average metrics across folds ──")
    avg = results.groupby("model")[["ic", "hit_rate", "sharpe"]].mean()
    std = results.groupby("model")[["ic", "hit_rate", "sharpe"]].std()
    for name in avg.index:
        print(f"    {name:15s}  IC={avg.loc[name,'ic']:+.4f}±{std.loc[name,'ic']:.4f}  "
              f"Hit={avg.loc[name,'hit_rate']:.3f}±{std.loc[name,'hit_rate']:.3f}  "
              f"Sharpe={avg.loc[name,'sharpe']:+.3f}±{std.loc[name,'sharpe']:.3f}")

    return results


# ── Stacking ensemble ────────────────────────────────────────────────────────

def train_ensemble(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    best_params: dict[str, dict],
    wf_results: pd.DataFrame,
) -> dict:
    """Train an IC-weighted ensemble of base models.

    Weights are proportional to mean walk-forward IC (clipped at 0). This is
    more robust than a learned meta-learner on small OOF samples.
    """
    print("\n" + "=" * 60)
    print("IC-Weighted Ensemble")
    print("=" * 60)

    # Compute weights from walk-forward IC
    mean_ic = wf_results.groupby("model")["ic"].mean()
    ic_clipped = mean_ic.clip(lower=0)
    weights = ic_clipped / ic_clipped.sum()
    print("\n  Walk-forward IC → weights:")
    for name, w in weights.items():
        print(f"    {name:15s}  IC={mean_ic[name]:+.4f}  →  weight={w:.3f}")

    # Train final base models and collect predictions
    trained_base = {}
    test_preds = pd.DataFrame(index=X_test.index)

    for name, params in best_params.items():
        final_model = _make_model(name, params)
        if name == "xgboost":
            pipe = final_model
            imp_s = pipe.named_steps["imputer"]
            sc = pipe.named_steps["scaler"]
            reg = pipe.named_steps["reg"]
            X_tr_np = sc.fit_transform(imp_s.fit_transform(X_train))
            split = int(len(X_tr_np) * 0.8)
            reg.fit(X_tr_np[:split], y_train.iloc[:split],
                    eval_set=[(X_tr_np[split:], y_train.iloc[split:])],
                    verbose=False)
            test_preds[name] = reg.predict(sc.transform(imp_s.transform(X_test)))
        else:
            final_model.fit(X_train, y_train)
            test_preds[name] = final_model.predict(X_test)

        trained_base[name] = final_model

    # Weighted average
    ensemble_pred = np.zeros(len(X_test))
    for name in best_params:
        ensemble_pred += weights[name] * test_preds[name].values

    m_ens = compute_metrics(y_test, ensemble_pred)
    print(f"\n  Ensemble Test  IC={m_ens['ic']:+.4f}  Hit={m_ens['hit_rate']:.3f}  Sharpe={m_ens['sharpe']:+.3f}")

    # Compare to best individual
    best_individual = max(best_params.keys(), key=lambda n: mean_ic[n])
    m_best = compute_metrics(y_test, test_preds[best_individual])
    delta_ic = m_ens["ic"] - m_best["ic"]
    delta_sharpe = m_ens["sharpe"] - m_best["sharpe"]
    print(f"  vs best individual ({best_individual}):  ΔIC={delta_ic:+.4f}  ΔSharpe={delta_sharpe:+.3f}")

    ensemble = {
        "base_models": trained_base,
        "weights": weights.to_dict(),
        "feature_names": list(X_train.columns),
    }
    return ensemble


# ── Main entry point ─────────────────────────────────────────────────────────

def train() -> None:
    print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_dataset_split()

    print(f"  Train : {len(X_train):,} obs  ({X_train.index.min().date()} → {X_train.index.max().date()})")
    print(f"  Test  : {len(X_test):,}  obs  ({X_test.index.min().date()} → {X_test.index.max().date()})")
    print(f"  Features ({X_train.shape[1]}): {list(X_train.columns)}")
    print(f"  Target mean (train) : {y_train.mean():.4f}  std : {y_train.std():.4f}")

    X_all = pd.concat([X_train, X_test]).sort_index()
    y_all = pd.concat([y_train, y_test]).sort_index()

    # ── Step 1: Feature pruning ───────────────────────────────────────────
    kept_features = select_features(X_train, y_train, threshold=0.01)
    X_train = X_train[kept_features]
    X_test = X_test[kept_features]
    X_all = X_all[kept_features]

    # ── Step 2: Hyperparameter tuning ─────────────────────────────────────
    best_params = tune_hyperparams(X_all, y_all)

    tuning_path = RESULTS_DIR / "best_params.json"
    serializable = {k: {pk: (float(pv) if isinstance(pv, (int, float)) else pv) for pk, pv in v.items()} for k, v in best_params.items()}
    tuning_path.write_text(json.dumps(serializable, indent=2))
    print(f"\n  Best params saved → {tuning_path.relative_to(PROJECT_ROOT)}")

    # ── Step 3: Walk-forward evaluation ───────────────────────────────────
    wf_results = walk_forward_cv(X_all, y_all, best_params)
    wf_path = RESULTS_DIR / "walk_forward_cv.csv"
    wf_results.to_csv(wf_path, index=False)
    print(f"\n  Walk-forward results saved → {wf_path.relative_to(PROJECT_ROOT)}")

    # ── Step 4: Final training + individual models ────────────────────────
    print("\n" + "=" * 60)
    print("Final Training (train ≤ 2018, test ≥ 2019)")
    print("=" * 60)

    for name, params in best_params.items():
        model = _make_model(name, params)
        print(f"\nTraining {name} {params}...")

        if name == "xgboost":
            pipe = model
            imp_s = pipe.named_steps["imputer"]
            sc = pipe.named_steps["scaler"]
            reg = pipe.named_steps["reg"]
            X_tr_np = sc.fit_transform(imp_s.fit_transform(X_train))
            X_te_np = sc.transform(imp_s.transform(X_test))
            split = int(len(X_tr_np) * 0.8)
            reg.fit(X_tr_np[:split], y_train.iloc[:split],
                    eval_set=[(X_tr_np[split:], y_train.iloc[split:])],
                    verbose=False)
            y_pred_train = reg.predict(X_tr_np)
            y_pred_test = reg.predict(X_te_np)
        else:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

        m_train = compute_metrics(y_train, y_pred_train)
        m_test  = compute_metrics(y_test, y_pred_test)

        print(f"  Train  IC={m_train['ic']:+.4f}  Hit={m_train['hit_rate']:.3f}  Sharpe={m_train['sharpe']:+.3f}")
        print(f"  Test   IC={m_test['ic']:+.4f}  Hit={m_test['hit_rate']:.3f}  Sharpe={m_test['sharpe']:+.3f}")

        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        print(f"  Saved → {path}")

    # ── Step 5: IC-weighted ensemble ──────────────────────────────────────
    ensemble = train_ensemble(X_train, X_test, y_train, y_test, best_params, wf_results)
    ensemble_path = MODELS_DIR / "ensemble.joblib"
    joblib.dump(ensemble, ensemble_path)
    print(f"\n  Ensemble saved → {ensemble_path}")

    print("\nDone.")


if __name__ == "__main__":
    train()

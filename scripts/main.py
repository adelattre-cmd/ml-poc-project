from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _load_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module `{module_name}` from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT_DIR = Path(__file__).resolve().parent

config = _load_module("project_config", SCRIPT_DIR.parent / "src" / "config.py")
sys.modules["config"] = config
load_dotenv(config.ENV_FILE)
PROJECT_ROOT = config.PROJECT_ROOT
SRC_DIR = config.SRC_DIR
APP_ENTRYPOINT = config.APP_ENTRYPOINT
MODELS = config.MODELS
STREAMLIT_HOST = config.STREAMLIT_HOST
STREAMLIT_PORT = config.STREAMLIT_PORT

data_module = _load_module("project_data", SRC_DIR / "data.py")
metrics_module = _load_module("project_metrics", SRC_DIR / "metrics.py")
model_io_module = _load_module("project_model_io", SRC_DIR / "model_io.py")
results_module = _load_module("project_results", SRC_DIR / "results.py")

sys.modules["data"] = data_module
sys.modules["metrics"] = metrics_module
sys.modules["model_io"] = model_io_module
sys.modules["results"] = results_module

load_dataset_split = data_module.load_dataset_split
compute_metrics = metrics_module.compute_metrics
load_model = model_io_module.load_model
write_metrics = results_module.write_metrics


def _validate_models_config() -> None:
    if not MODELS:
        raise ValueError("config.MODELS is empty. Add your trained models first.")

    for model_key, model_config in MODELS.items():
        if "path" not in model_config:
            raise ValueError(
                f"Missing `path` for model `{model_key}` in config.MODELS."
            )


def _validate_app_entrypoint() -> None:
    app_module = _load_module("project_app", APP_ENTRYPOINT)
    if not hasattr(app_module, "build_app") or not callable(app_module.build_app):
        raise TypeError("app.build_app must be a callable Streamlit entry point.")


def _streamlit_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [str(SRC_DIR)]
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def _load_dataset() -> tuple[Any, Any, Any, Any]:
    dataset_split = load_dataset_split()
    if not isinstance(dataset_split, tuple) or len(dataset_split) != 4:
        raise ValueError(
            "data.load_dataset_split() must return exactly four values: "
            "(X_train, X_test, y_train, y_test)."
        )

    return dataset_split


def _evaluate_models(X_test: Any, y_test: Any) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for model_key, model_config in MODELS.items():
        model = load_model(Path(model_config["path"]))

        if not hasattr(model, "predict"):
            raise TypeError(
                f"Loaded object for model `{model_key}` does not expose a `predict` method."
            )

        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)

        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(
                "metrics.compute_metrics() must return a non-empty dictionary."
            )

        row: dict[str, object] = {
            "model_key": model_key,
            "model_name": model_config.get("name", model_key),
            "model_path": str(model_config["path"]),
        }

        for metric_name, metric_value in metrics.items():
            row[metric_name] = float(metric_value)

        rows.append(row)

    return rows


def _launch_streamlit() -> None:
    if not APP_ENTRYPOINT.exists():
        raise FileNotFoundError(f"Streamlit entry point not found: {APP_ENTRYPOINT}")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_ENTRYPOINT),
            "--server.address",
            STREAMLIT_HOST,
            "--server.port",
            str(STREAMLIT_PORT),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=_streamlit_env(),
    )


def main() -> None:
    _validate_app_entrypoint()
    _validate_models_config()

    try:
        _, X_test, _, y_test = _load_dataset()
    except NotImplementedError as exc:
        raise NotImplementedError(
            "Dataset loading is still a template placeholder. "
            "Implement data.load_dataset_split()."
        ) from exc

    try:
        metrics_rows = _evaluate_models(X_test, y_test)
    except NotImplementedError as exc:
        raise NotImplementedError(
            "Metric computation is still a template placeholder. "
            "Implement metrics.compute_metrics()."
        ) from exc

    metrics_df = write_metrics(metrics_rows)

    print("Model evaluation completed. Metrics saved to results/model_metrics.csv")
    print(metrics_df.to_string(index=False))
    print(f"\nLaunching Streamlit on http://{STREAMLIT_HOST}:{STREAMLIT_PORT} ...")

    _launch_streamlit()


if __name__ == "__main__":
    main()

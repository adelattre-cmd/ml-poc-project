# ML Project Template

This repository is the base template that each student will fork and adapt for the final machine learning proof-of-concept project.

The template already defines the project structure and the main execution workflow. Your job as a student is to plug your own dataset loading logic, trained models, evaluation metrics, and Streamlit presentation into the fixed contracts described below.

## Repository Structure

- `deliverables/`: markdown files containing all assignements
- `deliverables/assignement1.md`: first assignement due (5 in total)
- `data/`: raw and processed data files
- `logs/`: log files generated during execution
- `models/`: trained machine learning models saved to disk
- `notebooks/`: Jupyter notebooks for analysis and experimentation
- `plots/`: generated visualizations
- `results/`: evaluation outputs, including model comparison tables
- `scripts/`: executable project scripts
- `scripts/main.py`: main entry point for evaluating models and launching the app
- `src/`: project source code
- `src/config.py`: project paths, model registry, and Streamlit configuration
- `src/data.py`: student-implemented dataset loading function
- `src/metrics.py`: student-implemented metric computation function
- `src/app.py`: fixed Streamlit entry point that students must customize
- `tests/`: optional tests
- `.env`: environment variables if your project needs them

## Expected Workflow

When you run:

```bash
python scripts/main.py
```

the template will do the following:

1. read the list of trained models from `src/config.py`,
2. call your dataset loading function from `src/data.py`,
3. load each serialized model from `models/`,
4. run predictions on the test split,
5. call your metric computation function from `src/metrics.py`,
6. save the results to `results/model_metrics.csv`,
7. print the metrics in the terminal,
8. launch the Streamlit app on `localhost`.

## What You Must Update

### 1. Register your trained models in `src/config.py`

Replace the example `MODELS` dictionary with your own trained models.

Each entry must define at least:

- `name`
- `description`
- `path`

Example:

```python
MODELS = {
    "log_reg": {
        "name": "Logistic Regression",
        "description": "Baseline classifier with standardized features.",
        "path": MODELS_DIR / "log_reg.joblib",
    },
    "rf": {
        "name": "Random Forest",
        "description": "Tree ensemble tuned on the validation split.",
        "path": MODELS_DIR / "random_forest.pkl",
    },
}
```

Supported model formats are:

- `.joblib`
- `.pkl`
- `.pickle`

Each saved object must expose a `.predict(X)` method.

### 2. Implement the dataset loading function in `src/data.py`

The file already exists and must keep this function name and signature:

```python
def load_dataset_split() -> tuple[Any, Any, Any, Any]:
```

It must return:

```python
(X_train, X_test, y_train, y_test)
```

Constraints:

- `X_train` and `X_test` must be in a format accepted by every model in `MODELS`
- `y_train` and `y_test` must contain the matching targets
- `X_test` and `y_test` will be used by `scripts/main.py` for evaluation
- Typical return types are `pandas.DataFrame`, `pandas.Series`, and/or `numpy.ndarray`

Minimal example:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR


def load_dataset_split():
    df = pd.read_csv(DATA_DIR / "processed_dataset.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3. Implement the metric computation function in `src/metrics.py`

The file already exists and must keep this function name and signature:

```python
def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
```

It must return a dictionary mapping metric names to numeric values.

Example:

```python
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }
```

Constraints:

- Use the same metric names for all evaluated models
- Every metric value must be numeric and convertible to `float`
- The returned dictionary is written directly to `results/model_metrics.csv`

### 4. Customize the Streamlit application in `src/app.py`

The file `src/app.py` is the fixed Streamlit entry point used by `scripts/main.py`.

Keep this function name:

```python
def build_app() -> None:
```

You should update the placeholder app to present:

- the business objective,
- the dataset and key insights,
- your visualizations,
- model comparison results,
- any prediction demo or interactive workflow relevant to your project.

The template app already tries to display `results/model_metrics.csv` if it exists.

## Recommended Student Workflow

1. Fork this repository.
2. Create and activate your virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

The template also reads `project-repo/.env` with `python-dotenv`. By default it contains:

```text
PYTHONPATH=./src
```

This is used when `scripts/main.py` launches Streamlit so modules inside `src/` resolve as top-level imports such as `from config import ...` or `from app import build_app`.

4. Add your data files to `data/`.
5. Train and save your models into `models/`.
6. Update `src/config.py`.
7. Implement `src/data.py`.
8. Implement `src/metrics.py`.
9. Customize `src/app.py`.
10. Run the full project:

```bash
python scripts/main.py
```

Optional flags:

```bash
python scripts/main.py --force-train   # retrain before evaluation
python scripts/main.py --skip-train    # fail if model files are missing
```

## Output Produced by the Template

After a successful run, you should have:

- printed metrics in the terminal,
- a CSV file at `results/model_metrics.csv`,
- a Streamlit app running locally, by default at:

```text
http://localhost:8501
```

## Common Errors

### `NotImplementedError` from `data`

You have not implemented `load_dataset_split()` yet.

### `NotImplementedError` from `metrics`

You have not implemented `compute_metrics()` yet.

### `FileNotFoundError` for a model path

One of the model files declared in `src/config.py` does not exist in `models/`.

### Model has no `predict` method

The object loaded from disk is not a trained model compatible with the template evaluation flow.

### Streamlit starts but shows only the placeholder page

You still need to customize `src/app.py` with your project content.

## Notes

- Keep `scripts/main.py` as the main orchestration entry point.
- Keep the function names and signatures in `src/data.py`, `src/metrics.py`, and `src/app.py` unchanged.
- Save your trained models before running the template.
- Use the same evaluation logic for all registered models so the comparison remains fair.

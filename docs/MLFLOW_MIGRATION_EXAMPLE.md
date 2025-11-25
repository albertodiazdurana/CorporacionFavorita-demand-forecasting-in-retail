# MLflow Migration Example

## Before & After: Updating Notebooks

### Example 1: Simple MLflow Usage

**❌ BEFORE (Old Pattern - FULL_02 notebook):**
```python
import mlflow
import mlflow.sklearn
from pathlib import Path

# Manual setup - easy to get wrong!
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
MLFLOW_DIR = PROJECT_ROOT / 'mlflow_results'
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
mlflow.set_experiment("full_pipeline_model_comparison")

print(f"MLflow tracking: file://{MLFLOW_DIR}")
```

**✅ AFTER (New Pattern - Centralized Config):**
```python
import mlflow
import mlflow.sklearn
from config.mlflow_config import setup_mlflow

# One line - always correct!
setup_mlflow("full_pipeline_model_comparison")
```

### Example 2: Week 3 Notebooks (Using Default Location)

**❌ BEFORE (Old Pattern - uses default mlruns/):**
```python
import mlflow

# No explicit setup - uses ./mlruns by default
mlflow.set_experiment("favorita-forecasting")

with mlflow.start_run(run_name="xgboost_baseline"):
    # Training code...
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
```

**✅ AFTER (New Pattern):**
```python
import mlflow
from config.mlflow_config import setup_mlflow

# Explicit setup - uses mlflow_results/
setup_mlflow("favorita-forecasting")

with mlflow.start_run(run_name="xgboost_baseline"):
    # Training code...
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
```

## Benefits of New Approach

### 1. Consistency
- All notebooks use the same tracking location
- No more "where did my models go?"

### 2. Less Code
- One import, one function call
- No path manipulation needed

### 3. Maintainability
- Change tracking location in ONE place (config/mlflow_config.py)
- All notebooks automatically updated

### 4. No Confusion
- Always use `mlflow_results/`
- MLflow UI startup is standardized

## Quick Migration Checklist

For each notebook that uses MLflow:

- [ ] Add import: `from config.mlflow_config import setup_mlflow`
- [ ] Replace manual `mlflow.set_tracking_uri()` with `setup_mlflow(experiment_name)`
- [ ] Remove manual MLFLOW_DIR path construction
- [ ] Test the notebook runs correctly
- [ ] Verify runs appear in MLflow UI at http://127.0.0.1:5000

## Which Notebooks Need Updates?

Based on your project, these notebooks likely use MLflow:

- `w03_d01_MODEL_baseline.ipynb` - Baseline models
- `w03_d02_MODEL_mlflow-features.ipynb` - MLflow + features
- `w03_d03_MODEL_tuning.ipynb` - Hyperparameter tuning
- `w03_d04_MODEL_lstm.ipynb` - LSTM experiments
- `w03_d05_MODEL_artifacts-export.ipynb` - Artifact export
- `FULL_02_train_final_model.ipynb` - Final models (already updated pattern)

Update these to use the centralized config!

# MLflow Configuration Guide

## Overview

This project uses a **centralized MLflow tracking setup** to avoid confusion with multiple tracking directories.

**Single tracking location:** `mlflow_results/`

## Quick Start

### 1. Use MLflow in Notebooks

Replace this OLD pattern:
```python
# ❌ DON'T DO THIS (inconsistent setup)
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
mlflow.set_experiment("my_experiment")
```

With this NEW pattern:
```python
# ✅ DO THIS (centralized config)
from config.mlflow_config import setup_mlflow

setup_mlflow("my_experiment_name")
```

### 2. Start MLflow UI

**Option A: Use the startup script (easiest)**
```bash
./scripts/start_mlflow_ui.sh
```

**Option B: Manual command - works in PowerShell**
```bash
cd /home/berto/Demand-forecasting-in-retail
source .venv/bin/activate
mlflow ui --backend-store-uri file://$(pwd)/mlflow_results --port 5000
```

**Option C: From Python**
```bash
python config/mlflow_config.py  # Prints the command to run
```

Then open: http://127.0.0.1:5000

## Project Structure

```
Demand-forecasting-in-retail/
├── mlflow_results/          # ✅ Single source of truth for MLflow
│   ├── <experiment_id>/
│   │   ├── <run_id>/
│   │   │   ├── artifacts/
│   │   │   ├── metrics/
│   │   │   ├── params/
│   │   │   └── meta.yaml
│   │   └── meta.yaml
│   └── models/
├── mlruns/                  # ⚠️ Ignore this (kept for backward compatibility)
├── config/
│   └── mlflow_config.py    # Centralized MLflow configuration
└── scripts/
    └── start_mlflow_ui.sh  # MLflow UI startup script
```

## Migration Guide

If you have old notebooks using the default `mlruns` directory, update them:

### Before:
```python
import mlflow
mlflow.set_experiment("my_experiment")
# Uses default ./mlruns directory
```

### After:
```python
from config.mlflow_config import setup_mlflow
setup_mlflow("my_experiment")
# Uses centralized mlflow_results/ directory
```

## Best Practices

### 1. **Always use the config module**
- Import `setup_mlflow()` at the start of every notebook
- This ensures consistency across the project

### 2. **Use descriptive experiment names**
```python
# ✅ Good experiment names
setup_mlflow("feature_engineering_v2")
setup_mlflow("xgboost_hyperparameter_tuning")
setup_mlflow("full_pipeline_model_comparison")

# ❌ Bad experiment names
setup_mlflow("test")
setup_mlflow("experiment1")
```

### 3. **Use descriptive run names**
```python
with mlflow.start_run(run_name="xgboost_full_q4q1"):
    # Your training code
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
```

### 4. **Starting MLflow UI**
- **Always** use `--backend-store-uri` pointing to `mlflow_results/`
- Use the startup script to avoid mistakes
- If you see unexpected experiments, check your browser cache

## Troubleshooting

### Problem: Can't see my models in MLflow UI

**Cause:** You're pointing MLflow UI to the wrong directory

**Solution:**
1. Kill any running MLflow UI: `pkill -f "mlflow ui"`
2. Start with correct backend: `./scripts/start_mlflow_ui.sh`
3. Hard refresh browser: `Ctrl+Shift+R` (or open incognito)

### Problem: Seeing old experiments from "mlruns"

**Cause:** Browser cache or pointing UI to wrong directory

**Solution:**
1. Verify MLflow UI command includes: `--backend-store-uri file://.../mlflow_results`
2. Clear browser cache or use incognito mode
3. Check you're accessing the right port (default: 5000)

### Problem: "favorita-forecasting" experiment appears

**Cause:** This was from an old MLflow instance or cached data

**Solution:**
1. Hard refresh browser: `Ctrl+Shift+R`
2. Verify correct tracking URI via API:
   ```bash
   curl "http://127.0.0.1:5000/api/2.0/mlflow/experiments/search?max_results=100"
   ```

### Problem: Models in different locations

**Cause:** Some notebooks using default `mlruns`, others using `mlflow_results`

**Solution:**
1. Update all notebooks to use `config.mlflow_config.setup_mlflow()`
2. Optionally migrate old runs (see below)

## Optional: Migrate Old Runs

If you have important runs in the old `mlruns` directory:

```bash
# List experiments in old location
ls mlruns/

# If needed, you can manually copy experiment directories
# But it's usually better to just re-run notebooks with new config
```

For most cases, just **re-run your notebooks** with the new configuration. The old `mlruns` directory is in `.gitignore` anyway.

## Summary

✅ **One tracking location:** `mlflow_results/`
✅ **One config module:** `config/mlflow_config.py`
✅ **One startup script:** `scripts/start_mlflow_ui.sh`
✅ **All notebooks use:** `setup_mlflow(experiment_name)`

This prevents the "mixed setup" situation from happening again!

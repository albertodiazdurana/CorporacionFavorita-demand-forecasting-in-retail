# MLflow Configuration Changes - Summary for Context Sharing

**Date:** 2025-11-25
**Issue:** MLflow models not appearing in UI due to mixed tracking directories
**Resolution:** Centralized configuration with unified tracking location

---

## Problem Identified

1. **Two MLflow tracking directories existed:**
   - `mlruns/` - Default MLflow location (mostly empty)
   - `mlflow_results/` - Used by FULL_02 notebook only

2. **MLflow UI was showing wrong experiments:**
   - Browser cache showing old "favorita-forecasting" experiment
   - MLflow UI not started with correct `--backend-store-uri` parameter

3. **Root cause:**
   - Inconsistent MLflow setup across notebooks
   - Manual tracking URI configuration in each notebook
   - No standardized way to start MLflow UI

---

## Solution Implemented

### 1. Centralized Configuration Module

**File:** `config/mlflow_config.py`

```python
from config.mlflow_config import setup_mlflow

# Single function call replaces manual setup
setup_mlflow("experiment_name")
# Automatically configures tracking_uri and experiment
```

**Benefits:**
- All notebooks use same tracking location (`mlflow_results/`)
- No manual path construction needed
- Change config in one place, affects all notebooks

### 2. MLflow UI Startup Script

**File:** `scripts/start_mlflow_ui.sh` (executable)

```bash
./scripts/start_mlflow_ui.sh
# Automatically uses correct backend-store-uri
# Opens on http://127.0.0.1:5000
```

**Benefits:**
- Always starts with correct tracking URI
- No more forgetting `--backend-store-uri` parameter
- Port conflict detection

### 3. Comprehensive Documentation

**Files created:**
- `README_MLFLOW.md` - Quick reference guide
- `docs/MLFLOW_SETUP.md` - Detailed setup and troubleshooting
- `docs/MLFLOW_MIGRATION_EXAMPLE.md` - Before/after migration examples
- `MLFLOW_SUMMARY.txt` - Visual summary

---

## Changes Made to Project

### New Files (5)

```
config/
└── mlflow_config.py          # Centralized MLflow configuration

scripts/
└── start_mlflow_ui.sh        # MLflow UI startup script (executable)

docs/
├── MLFLOW_SETUP.md           # Comprehensive setup guide
└── MLFLOW_MIGRATION_EXAMPLE.md  # Migration examples

README_MLFLOW.md              # Quick reference
MLFLOW_SUMMARY.txt            # Visual summary
```

### Modified Files (1)

```
.gitignore
├── Added comment: "# MLflow - ignore both old and new tracking directories"
└── Already had: mlruns/, mlartifacts/, mlflow_results/
```

---

## Migration Pattern for Existing Notebooks

### Before (Old Pattern)
```python
import mlflow
from pathlib import Path

# Manual setup - inconsistent across notebooks
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
MLFLOW_DIR = PROJECT_ROOT / 'mlflow_results'
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
mlflow.set_experiment("full_pipeline_model_comparison")
```

### After (New Pattern)
```python
import mlflow
from config.mlflow_config import setup_mlflow

# Centralized setup - consistent everywhere
setup_mlflow("full_pipeline_model_comparison")
```

**Reduction:** 7 lines → 2 lines, always correct

---

## Current State of MLflow Data

### Tracking Directory: `mlflow_results/`

**Experiments:**
1. `full_pipeline_model_comparison` (ID: 425292371833343006)
   - Run 1: `xgboost_full_q4q1` (FINISHED)
     - RMSE: 6.4008
     - MAE: 1.7480
     - Training samples: 3,798,720
   - Run 2: `lstm_full_q4q1` (FAILED)
     - Did not converge at scale
     - Training samples: 3,798,720

### Legacy Directory: `mlruns/`
- Contains only "Default" experiment (empty)
- Kept for backward compatibility
- Not used going forward

---

## How to Start Using New Setup

### For New Work

1. **In notebooks:**
   ```python
   from config.mlflow_config import setup_mlflow
   setup_mlflow("my_experiment_name")
   ```

2. **Start MLflow UI:**
   ```bash
   ./scripts/start_mlflow_ui.sh
   ```

3. **Access UI:**
   http://127.0.0.1:5000

### For Existing Notebooks

**Update these notebooks to use new pattern:**
- `w03_d01_MODEL_baseline.ipynb`
- `w03_d02_MODEL_mlflow-features.ipynb`
- `w03_d03_MODEL_tuning.ipynb`
- `w03_d04_MODEL_lstm.ipynb`
- `w03_d05_MODEL_artifacts-export.ipynb`
- `FULL_02_train_final_model.ipynb` (already uses custom setup)

**See:** `docs/MLFLOW_MIGRATION_EXAMPLE.md` for step-by-step examples

---

## Technical Details

### MLflow Tracking URI
```
file:///home/berto/Demand-forecasting-in-retail/mlflow_results
```

### Directory Structure
```
mlflow_results/
├── <experiment_id>/              # e.g., 425292371833343006
│   ├── <run_id>/                 # e.g., 5155a95a23e4477c...
│   │   ├── artifacts/            # Model files, plots
│   │   ├── metrics/              # RMSE, MAE, etc.
│   │   ├── params/               # Hyperparameters
│   │   ├── tags/                 # Metadata
│   │   └── meta.yaml             # Run metadata
│   ├── meta.yaml                 # Experiment metadata
│   └── models/                   # Registered models
└── .trash/                       # Deleted experiments
```

### MLflow UI Command
```bash
/home/berto/Demand-forecasting-in-retail/.venv/bin/mlflow ui \
  --backend-store-uri file:///home/berto/Demand-forecasting-in-retail/mlflow_results \
  --port 5000
```

---

## Key Takeaways

### What Changed
- ✅ Centralized MLflow configuration
- ✅ Unified tracking directory (`mlflow_results/`)
- ✅ Automated UI startup script
- ✅ Comprehensive documentation

### What Stayed Same
- ✅ MLflow logging API unchanged
- ✅ Existing experiment data preserved
- ✅ Model artifacts location unchanged
- ✅ All runs still accessible

### What's Better
- ✅ No more "where did my models go?"
- ✅ Consistent setup across all notebooks
- ✅ Easy to start MLflow UI correctly
- ✅ Single source of truth for configuration

---

## Troubleshooting Reference

### Issue: Models not visible in UI
**Solution:** Hard refresh browser (Ctrl+Shift+R) or use incognito mode

### Issue: Port 5000 already in use
**Solution:** `pkill -f "mlflow ui"` or use different port

### Issue: Seeing old experiments
**Solution:** Verify MLflow UI uses `--backend-store-uri file://.../mlflow_results`

### Issue: Import error in notebooks
**Solution:** Ensure you're running from project root or notebooks/ directory

---

## References

- **Quick Start:** `README_MLFLOW.md`
- **Full Guide:** `docs/MLFLOW_SETUP.md`
- **Migration Examples:** `docs/MLFLOW_MIGRATION_EXAMPLE.md`
- **Visual Summary:** `MLFLOW_SUMMARY.txt`

---

## Next Steps Recommended

1. **Update Week 3 notebooks** to use `setup_mlflow()`
2. **Test migration** on one notebook first
3. **Standardize UI startup** using `./scripts/start_mlflow_ui.sh`
4. **Document any issues** for future reference

---

## Git Status

```
Changes not staged for commit:
  modified:   .gitignore

Untracked files:
  MLFLOW_SUMMARY.txt
  README_MLFLOW.md
  config/mlflow_config.py
  docs/MLFLOW_MIGRATION_EXAMPLE.md
  docs/MLFLOW_SETUP.md
  scripts/start_mlflow_ui.sh
```

**Note:** `mlflow_results/` is in `.gitignore` (tracking data not committed to repo)

---

**Summary:** All MLflow models are now centralized in `mlflow_results/` with standardized configuration. Use `setup_mlflow()` in notebooks and `./scripts/start_mlflow_ui.sh` to start UI. Problem solved! 🎉

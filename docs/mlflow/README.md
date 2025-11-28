# MLflow Setup - Quick Reference

## 🎯 TL;DR - What You Need to Know

**Problem Solved:** Multiple MLflow tracking directories causing confusion
**Solution:** One centralized location (`mlflow_results/`) with helper scripts

## 🚀 Quick Start (3 Steps)

### 1. In Your Notebooks
```python
from config.mlflow_config import setup_mlflow

# Replace any manual mlflow.set_tracking_uri() calls with:
setup_mlflow("your_experiment_name")
```

### Starting the UI
```bash
# Option 1: Use the script (easiest)
./scripts/start_mlflow_ui.sh

# Option 2: Manual (works in PowerShell)
source .venv/bin/activate
mlflow ui --backend-store-uri file://$(pwd)/mlflow_results --port 5000
```

### 3. Open Browser
http://127.0.0.1:5000

**That's it!** All your models will be in one place.

---

## 📁 Files Created

| File                                                           | Purpose                          |
| -------------------------------------------------------------- | -------------------------------- |
| [config/mlflow_config.py](../../config/mlflow_config.py)       | Centralized MLflow configuration |
| [scripts/start_mlflow_ui.sh](../../scripts/start_mlflow_ui.sh) | Easy MLflow UI startup           |
| [SETUP.md](SETUP.md)                                           | Detailed setup guide             |

---

## 🔧 What Changed?

### Before (Messy)
- ❌ Different notebooks using different tracking locations
- ❌ Manual path construction in each notebook
- ❌ Forgetting `--backend-store-uri` when starting UI
- ❌ Models scattered across `mlruns/` and `mlflow_results/`

### After (Clean)
- ✅ One tracking location: `mlflow_results/`
- ✅ One import: `from config.mlflow_config import setup_mlflow`
- ✅ One startup script: `./scripts/start_mlflow_ui.sh`
- ✅ All models in one place, always visible

---

## 📝 Example Usage

### In a Notebook
```python
import mlflow
from config.mlflow_config import setup_mlflow

# Setup (one line)
setup_mlflow("feature_engineering_v2")

# Use MLflow normally
with mlflow.start_run(run_name="xgboost_baseline"):
    mlflow.log_params({"max_depth": 6, "n_estimators": 500})
    mlflow.log_metrics({"rmse": 6.40, "mae": 1.75})
    mlflow.sklearn.log_model(model, "model")
```

---

## 🐛 Troubleshooting

**Q: I don't see my models in MLflow UI**
- Hard refresh browser: `Ctrl+Shift+R`
- Or open incognito/private window
- Verify you used `setup_mlflow()` in your notebook

**Q: I see old experiments from "favorita-forecasting"**
- Browser cache issue - hard refresh or use incognito
- Verify startup script shows: `Tracking URI: file://.../mlflow_results`

**Q: Port 5000 already in use**
- Kill old instance: `pkill -f "mlflow ui"`
- Or use different port: `./scripts/start_mlflow_ui.sh 5001`

---

## ✨ Benefits

1. **Consistency** - All notebooks use same tracking location
2. **Simplicity** - One import, one function call
3. **Maintainability** - Change config in one place
4. **No More Confusion** - Models always in the right place

**Now you'll never lose your MLflow models again! 🎉**

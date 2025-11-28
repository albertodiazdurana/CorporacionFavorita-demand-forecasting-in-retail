# 🚀 Start Here - MLflow Documentation

**Welcome to the MLflow configuration documentation!**

All MLflow-related documentation is now centralized in this directory for easy access.

---

## ⚡ Essential Info

### In Your Notebooks
```python
from config.mlflow_config import setup_mlflow
setup_mlflow("experiment_name")
```

### Starting the UI
```bash
# Option 1: Use the script (easiest)
./scripts/start_mlflow_ui.sh

# Option 2: Manual (works in PowerShell)
source .venv/bin/activate
mlflow ui --backend-store-uri file://$(pwd)/mlflow_results --port 5000
```

### Access UI
http://127.0.0.1:5000

---

## 🎯 The Main Idea

**Problem:** MLflow models scattered across different tracking directories  
**Solution:** One centralized location with unified configuration

- **One tracking directory:** `mlflow_results/`
- **One config module:** `config/mlflow_config.py`
- **One startup script:** `scripts/start_mlflow_ui.sh`

**Result:** All your models in one place, always! 🎉

---

**👉 Start with [README.md](README.md) for the quickest overview!**

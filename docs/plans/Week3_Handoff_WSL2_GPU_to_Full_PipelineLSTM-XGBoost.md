# FULL_02 Handoff: Train Final Models (Production Pipeline)

**From:** FULL_01_data_to_features.ipynb (Complete)  
**To:** FULL_02_train_final_model.ipynb  
**Date:** 2025-11-24  
**Status:** Ready to Execute

---

## Executive Summary

**What Was Accomplished (FULL_01):**
- Processed full Guayas dataset: 4.8M rows (16x more than Week 3)
- Created 33 features per DEC-014
- Output: full_featured_data.pkl (1.3 GB)
- All families included (32 vs top-3 in sample)

**What's Next (FULL_02):**
- Train XGBoost on full dataset (CPU)
- Train LSTM on full dataset (GPU)
- Compare both models with MLflow tracking
- Validate Week 3 finding: Does LSTM still beat XGBoost at scale?
- Export production artifacts

**Key Question to Answer:**
Week 3 showed LSTM beat XGBoost by 4.5% on 300K sample. Does this hold at 4.8M rows?

---

## 1. Input Data

**File:** data/processed/full_featured_data.pkl

| Metric | Value |
|--------|-------|
| Rows | 4,801,160 |
| Columns | 42 |
| Features | 33 |
| Period | Oct 1, 2013 - Mar 31, 2014 |
| Stores | 10 |
| Items | 2,638 |
| Families | 32 |
| File size | 1.3 GB |

---

## 2. Data Split Strategy

### Per DEC-016 + DEC-013

```
Training: Oct 1, 2013 - Feb 21, 2014 (144 days)
Gap:      Feb 22 - Feb 28, 2014 (7 days per DEC-013)
Test:     Mar 1 - Mar 31, 2014 (31 days)
```

### Expected Sample Sizes

| Split | Days | Est. Rows |
|-------|------|-----------|
| Training | 144 | ~3.8M |
| Gap | 7 | ~180K (excluded) |
| Test | 31 | ~800K |

---

## 3. Models to Train

### Model 1: XGBoost (Baseline)

**Purpose:** Tree-based benchmark, typically strong on tabular data

**Hyperparameters (from Week 3 tuned model):**
```python
xgb_params = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 50
}
```

**Training Notes:**
- CPU-bound (GPU doesn't help XGBoost significantly)
- Expected time: 5-15 minutes on 3.8M rows
- Monitor memory usage (~8-16 GB RAM)

---

### Model 2: LSTM (Challenger)

**Purpose:** Neural approach that won Week 3 comparison

**Architecture (from Week 3):**
```python
Input: (1 timestep, 33 features)
├── LSTM(64 units)
├── Dropout(0.2)
├── Dense(32 units, relu)
├── Dropout(0.2)
└── Dense(1 unit)
Total parameters: 27,201
```

**Training Configuration:**
```python
lstm_config = {
    'optimizer': 'adam',
    'loss': 'mse',
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 10,
    'validation_split': 0.2
}
```

**Training Notes:**
- GPU-accelerated (Quadro T1000)
- Expected time: 1-5 minutes
- Enable memory growth for TensorFlow

---

## 4. MLflow Experiment Tracking

### Experiment Name
```
full_pipeline_model_comparison
```

### Runs to Create

| Run Name | Model | Description |
|----------|-------|-------------|
| xgboost_full_q4q1 | XGBoost | Full dataset, Week 3 hyperparameters |
| lstm_full_q4q1 | LSTM | Full dataset, Week 3 architecture |

### Metrics to Log

| Metric | Description |
|--------|-------------|
| rmse | Root Mean Squared Error (primary) |
| mae | Mean Absolute Error |
| mape_nonzero | MAPE on non-zero actuals only |
| bias | Mean signed error |
| training_time_sec | Wall clock training time |
| training_samples | Number of training rows |
| test_samples | Number of test rows |

### Parameters to Log

| Parameter | Description |
|-----------|-------------|
| model_type | 'xgboost' or 'lstm' |
| n_features | 33 |
| train_start | '2013-10-01' |
| train_end | '2014-02-21' |
| test_start | '2014-03-01' |
| test_end | '2014-03-31' |
| gap_days | 7 |

---

## 5. Evaluation Metrics

### Primary Metrics (all test data)

| Metric | Formula | Purpose |
|--------|---------|---------|
| RMSE | sqrt(mean((y_pred - y_true)^2)) | Primary comparison, penalizes large errors |
| MAE | mean(abs(y_pred - y_true)) | Interpretable average error |
| Bias | mean(y_pred - y_true) | Systematic over/under prediction |

### Conditional Metric (non-zero only)

| Metric | Formula | Purpose |
|--------|---------|---------|
| MAPE | mean(abs((y_pred - y_true) / y_true)) * 100 | Percentage error, filter y_true > 0 |

### Implementation

```python
def calculate_metrics(y_true, y_pred):
    """Calculate all evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    
    # MAPE on non-zero only
    mask = y_true > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'mape_nonzero': mape
    }
```

---

## 6. Comparison Framework

### Week 3 Baseline (300K Sample)

| Model | RMSE | MAE | Improvement |
|-------|------|-----|-------------|
| XGBoost Tuned | 6.4860 | ~3.2 | +10.08% vs baseline |
| LSTM | 6.2552 | ~3.05 | +13.28% vs baseline |
| **Delta** | **4.5%** | - | LSTM wins |

### Full Pipeline Target

| Model | Target RMSE | Notes |
|-------|-------------|-------|
| XGBoost | Document actual | May improve with more data |
| LSTM | <6.20 | Target improvement |
| **Comparison** | Document delta | Validate Week 3 finding |

### Comparison Table Template

```markdown
## Full Pipeline Results vs Week 3 Sample

| Metric | XGBoost (Sample) | XGBoost (Full) | LSTM (Sample) | LSTM (Full) |
|--------|------------------|----------------|---------------|-------------|
| Rows | 300K | 4.8M | 300K | 4.8M |
| Training | 18,905 | ~3.8M | 18,905 | ~3.8M |
| Test | 4,686 | ~800K | 4,686 | ~800K |
| RMSE | 6.4860 | [TBD] | 6.2552 | [TBD] |
| MAE | ~3.2 | [TBD] | ~3.05 | [TBD] |
| Bias | [TBD] | [TBD] | [TBD] | [TBD] |
| MAPE (non-zero) | [TBD] | [TBD] | [TBD] | [TBD] |
| Training time | ~30 sec | [TBD] | 36 sec (CPU) | [TBD] (GPU) |
| Winner | - | [TBD] | LSTM | [TBD] |
```

---

## 7. Notebook Structure

### FULL_02_train_final_model.ipynb

**Section 1: Setup and Data Loading**
- Imports and path configuration
- Load full_featured_data.pkl
- Verify data dimensions

**Section 2: Data Splitting (DEC-016 + DEC-013)**
- Define date ranges
- Create train/test splits
- Apply 7-day gap
- Validate sample sizes

**Section 3: Feature Preparation**
- Select 33 features (DEC-014)
- Handle categorical encoding
- Initialize StandardScaler

**Section 4: MLflow Initialization**
- Set experiment name
- Configure tracking

**Section 5: XGBoost Training**
- Fit model on training data
- Predict on test set
- Calculate metrics
- Log to MLflow

**Section 6: LSTM Training**
- Scale features
- Reshape for LSTM input
- Configure GPU memory growth
- Train with early stopping
- Predict on test set
- Calculate metrics
- Log to MLflow

**Section 7: Model Comparison**
- Side-by-side metrics table
- Comparison to Week 3 results
- Determine winner at scale
- Document findings

**Section 8: Feature Importance Stability**
- Permutation importance for both models
- Compare to Week 3 rankings
- Validate DEC-014 holds at scale

**Section 9: Production Artifacts Export**
- Save best model (winner)
- Export scaler, feature columns, config
- Create model_usage.md

**Section 10: Summary and Next Steps**
- Final comparison table
- Key findings
- Week 4 integration notes

---

## 8. Time Estimates

| Section | Estimated Time |
|---------|----------------|
| Setup and data loading | 2 min |
| Data splitting | 2 min |
| Feature preparation | 3 min |
| MLflow initialization | 1 min |
| XGBoost training | 5-15 min |
| LSTM training (GPU) | 1-5 min |
| Model comparison | 5 min |
| Feature importance | 5-10 min |
| Artifacts export | 5 min |
| Summary | 5 min |
| **Total** | **35-50 min** |

---

## 9. Success Criteria

### Minimum Success (Must Have)
- [ ] Both models train without errors
- [ ] All 4 metrics calculated for both models
- [ ] MLflow runs logged successfully
- [ ] Comparison table created
- [ ] Winner determined at scale

### Target Success (Should Have)
- [ ] LSTM RMSE < 6.20 (improve on sample)
- [ ] GPU speedup documented for LSTM
- [ ] Feature importance validates DEC-014
- [ ] Production artifacts exported
- [ ] Week 3 finding validated or updated

### Stretch Success (Nice to Have)
- [ ] Error analysis by product family
- [ ] Bias analysis (which model over/under predicts)
- [ ] Confidence intervals on predictions

---

## 10. Potential Outcomes

### Scenario A: LSTM Still Wins
- Validates Week 3 finding at scale
- Export LSTM as production model
- Document consistent performance

### Scenario B: XGBoost Wins at Scale
- Interesting finding: more data favors tree models
- Document as DEC-017
- Export XGBoost as production model
- Strong portfolio story (scale matters)

### Scenario C: Results Very Close (<1% difference)
- Consider ensemble approach
- Document trade-offs (speed vs accuracy)
- Export both models

---

## 11. Artifact Naming Convention

### If LSTM Wins
```
artifacts/
├── lstm_model_full.keras
├── scaler_full.pkl
├── feature_columns.json
├── model_config_full.json
└── model_usage_full.md
```

### If XGBoost Wins
```
artifacts/
├── xgboost_model_full.pkl
├── scaler_full.pkl
├── feature_columns.json
├── model_config_full.json
└── model_usage_full.md
```

---

## 12. Environment Reminders

### GPU Verification
```bash
nvidia-smi
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Memory Management
```python
# TensorFlow GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### XGBoost Memory
```python
# Monitor during training
import psutil
print(f"RAM used: {psutil.Process().memory_info().rss / 1e9:.1f} GB")
```

---

## 13. Key Decisions Reference

| Decision | Application in FULL_02 |
|----------|------------------------|
| DEC-013 | 7-day gap (Feb 22-28) between train and test |
| DEC-014 | 33 features (same for both models) |
| DEC-016 | Q4+Q1 training (Oct 1 - Feb 21) |

**New Decision Potential:**
- DEC-017: If XGBoost wins at scale, document scale-dependent model selection

---

## 14. Execution Checklist

### Pre-Flight
- [ ] VSCode connected to WSL2
- [ ] GPU verified
- [ ] full_featured_data.pkl exists (1.3 GB)
- [ ] MLflow installed and working

### During Execution
- [ ] Monitor GPU with `nvidia-smi` during LSTM training
- [ ] Monitor RAM during XGBoost training
- [ ] Save intermediate results

### Post-Execution
- [ ] Both MLflow runs logged
- [ ] Comparison table complete
- [ ] Winner determined
- [ ] Artifacts exported
- [ ] Checkpoint document created

---

**Handoff Complete. Ready for FULL_02_train_final_model.ipynb**

---

**END OF HANDOFF DOCUMENT**

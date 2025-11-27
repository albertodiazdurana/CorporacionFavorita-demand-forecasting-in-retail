
# Corporación Favorita Grocery Sales Forecasting

**Time Series Analysis & Forecasting Project**

A comprehensive machine learning project analyzing Ecuadorian grocery sales data to develop forecasting models for retail inventory optimization. This project implements advanced feature engineering, multi-method outlier detection, and temporal pattern analysis using XGBoost and LSTM approaches.

---

## Live Demo

🚀 **[Launch Forecasting App](https://demand-forecasting-in-retail-app.streamlit.app)** - Interactive demand forecasting with store/item selection

**App Repository:** [Demand-forecasting-in-retail-app](https://github.com/albertodiazdurana/Demand-forecasting-in-retail-app)

---

## Project Overview

### Objective
Forecast daily unit sales for Corporación Favorita stores in the Guayas region (January-March 2014) to optimize inventory planning and reduce waste through accurate demand prediction.

**Data Source**: [Kaggle - Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)

### Business Context
- **Domain**: Retail time series forecasting
- **Scope**: Guayas region, all product families
- **Dataset**: 4.8M transactions across 10 stores, 2,638 items, 183 days
- **Evaluation Metric**: RMSE (Root Mean Squared Error)

### Production Model Performance

| Metric | Value |
|--------|-------|
| Model | XGBoost |
| RMSE | 6.4008 |
| MAE | 1.7480 |
| Training Samples | 3,798,720 |
| Test Samples | 817,780 |
| Features | 33 |

---

## Project Status: COMPLETE ✓

### Phase Summary

| Week | Focus | Status |
|------|-------|--------|
| Week 1 | Exploratory Data Analysis | ✅ Complete |
| Week 2 | Feature Engineering | ✅ Complete |
| Week 3 | Modeling & Analysis | ✅ Complete |
| Week 4 | Production & Deployment | ✅ Complete |

---

## Key Findings

### Model Comparison: Scale Reverses Advantage

| Dataset | XGBoost | LSTM | Winner |
|---------|---------|------|--------|
| Sample (300K) | 6.4860 | 6.2552 | LSTM (-4.5%) |
| Full (4.8M) | 6.4008 | ~9.37 | XGBoost (+46%) |

**Key Insight:** LSTM advantage on small sample disappeared at production scale. Tree models handle large tabular datasets better.

### Top 5 Features (Permutation Importance)

1. **unit_sales_7d_avg** (6.43) - 7-day rolling average
2. **unit_sales_lag1_7d_corr** (1.96) - Lag-rolling correlation
3. **unit_sales_lag1** (1.64) - Previous day sales
4. **item_avg_sales** (0.30) - Item-level average
5. **unit_sales_14d_avg** (0.23) - 14-day rolling average

### Business Insights

- **Weekend Effect**: 33.9% higher sales on weekends
- **Autocorrelation**: Strong weekly pattern (lag7: r=0.40)
- **Pareto Distribution**: 34% of items generate 80% of sales
- **Sparsity**: 99.1% is normal retail behavior (not data quality issue)

---

## Technical Decisions

| ID | Decision | Rationale | Impact |
|----|----------|-----------|--------|
| DEC-011 | Keep NaN in lag features | XGBoost handles natively | Preserved 9-13% NaN |
| DEC-012 | Include oil features | Tree models find non-linear patterns | Tested in ablation |
| DEC-013 | 7-day train/test gap | Prevents lag feature leakage | Critical for validity |
| DEC-014 | 33 optimized features | Ablation showed improvement | +4.5% RMSE gain |
| DEC-015 | Reject full 2013 training | Seasonal mismatch | Avoided +106% worse RMSE |
| DEC-016 | Q4+Q1 temporal consistency | Seasonal alignment | Better than more data |
| DEC-017 | XGBoost over LSTM at scale | LSTM failed to converge | Production model selected |

---

## Repository Structure
```
Demand-forecasting-in-retail/
├── notebooks/
│   ├── w01_d01 through w01_d05     # Week 1: EDA (5 notebooks)
│   ├── w02_d01 through w02_d05     # Week 2: Feature Engineering (5 notebooks)
│   ├── w03_d01 through w03_d05     # Week 3: Modeling (5 notebooks)
│   ├── FULL_01_data_to_features.ipynb   # Production pipeline: Features
│   └── FULL_02_train_final_model.ipynb  # Production pipeline: Training
├── data/
│   ├── raw/                        # Original Kaggle CSVs
│   └── processed/                  # Feature-engineered datasets
│       ├── full_featured_data.pkl  # 4.8M rows, 33 features (1.3 GB)
│       └── sample_forecast_data.pkl # App sample (1.1 MB)
├── artifacts/                      # Production model files
│   ├── xgboost_model_full.pkl      # Trained model (2.1 MB)
│   ├── scaler_full.pkl             # StandardScaler
│   ├── feature_columns.json        # 33 feature names
│   └── model_config_full.json      # Metrics and hyperparameters
├── outputs/figures/                # Visualizations
│   ├── eda/
│   ├── features/
│   ├── models/
│   └── full_pipeline/
├── config/
│   └── mlflow_config.py            # Centralized MLflow configuration
├── scripts/
│   ├── start_mlflow_ui.sh          # MLflow UI launcher
│   └── export_sample_data.py       # Sample data for Streamlit app
├── docs/
│   ├── decisions/                  # DEC-011 through DEC-017
│   └── checkpoints/                # Daily progress documentation
├── mlflow_results/                 # Experiment tracking (not in git)
├── README.md
├── README_MLFLOW.md                # MLflow quick reference
├── requirements.txt
└── FULL_02_checkpoint.md           # Production pipeline summary
```

---

## Weekly Accomplishments

### Week 1: Exploratory Data Analysis
- Three-method outlier detection (IQR, Z-score, Isolation Forest)
- Complete temporal pattern analysis
- Discovered 99.1% sparsity is normal retail behavior
- **Notebooks**: w01_d01 through w01_d05

### Week 2: Feature Engineering
- Created 45 features across 5 categories (reduced to 33 after ablation)
- Lag features, rolling statistics, calendar, holidays, aggregations
- **Output**: w02_d05_FE_final.pkl (300K sample)
- **Notebooks**: w02_d01 through w02_d05

### Week 3: Modeling & Analysis
- XGBoost baseline and hyperparameter tuning
- LSTM model comparison (won on sample data)
- MLflow experiment tracking (6 experiments)
- Feature importance validation (permutation, SHAP, ablation)
- **Best Sample Model**: LSTM (RMSE 6.2552)
- **Notebooks**: w03_d01 through w03_d05

### Week 4: Production & Deployment
- Full pipeline on 4.8M rows (FULL_01 + FULL_02)
- Discovered LSTM doesn't scale (DEC-017)
- Production model: XGBoost (RMSE 6.4008)
- Streamlit app deployed to cloud
- Interactive forecasting with store/item selection

---

## Production Pipeline

### FULL_01: Data to Features
- **Input**: Raw Kaggle CSVs
- **Output**: full_featured_data.pkl (4.8M rows × 42 columns, 1.3 GB)
- **Processing Time**: ~15 minutes
- **Key Step**: Vectorized feature engineering for 33 features

### FULL_02: Train Final Model
- **Input**: full_featured_data.pkl
- **Split**: Q4 2013 + Q1 2014 training, March 2014 test, 7-day gap
- **Models Tested**: XGBoost (✓), LSTM (failed to converge)
- **Output**: Production artifacts in /artifacts
- **Training Time**: 92 seconds (XGBoost)

### Environment
- **OS**: WSL2 Ubuntu 22.04
- **CPU**: Intel Core i7-10875H @ 2.30GHz
- **GPU**: NVIDIA Quadro T1000 (4GB VRAM)
- **Python**: 3.11
- **Key Libraries**: TensorFlow 2.20.0, XGBoost 3.1.2, MLflow 3.6.0

---

## Quick Start

### Prerequisites
- Python 3.11+
- ~16GB RAM for full pipeline
- Kaggle account for data download

### Setup
```bash
# Clone repository
git clone https://github.com/albertodiazdurana/Demand-forecasting-in-retail.git
cd Demand-forecasting-in-retail

# Create environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download data from Kaggle
# Place files in data/raw/: train.csv, stores.csv, items.csv, oil.csv, holidays_events.csv
```

### Run Production Pipeline
```bash
# Option 1: Run notebooks in Jupyter
jupyter notebook notebooks/FULL_01_data_to_features.ipynb
jupyter notebook notebooks/FULL_02_train_final_model.ipynb

# Option 2: Use existing artifacts (skip training)
# Artifacts already in /artifacts - ready for deployment
```

### View MLflow Experiments
```bash
./scripts/start_mlflow_ui.sh
# Open http://127.0.0.1:5000
```

---

## Dataset Specifications

### Raw Data (Kaggle)
- **train.csv**: 125M rows, 479 MB
- **stores.csv**: Store metadata (54 stores)
- **items.csv**: Item metadata (4,100 items)
- **oil.csv**: Daily oil prices
- **holidays_events.csv**: Ecuador holidays

### Processed Data
- **Full Pipeline**: 4,801,160 rows × 42 columns (Guayas, all families)
- **Sample Pipeline**: 300,896 rows (Guayas, top-3 families)
- **Date Range**: Oct 1, 2013 - Mar 31, 2014 (183 days)
- **Stores**: 10 Guayas stores
- **Items**: 2,638 unique items

### Feature Categories (33 features)

| Category | Count | Examples |
|----------|-------|----------|
| Temporal Lags | 4 | lag1, lag7, lag14, lag30 |
| Rolling Stats | 4 | 7d_avg, 14d_avg, 30d_avg, lag1_7d_corr |
| Calendar | 7 | year, month, day, dayofweek, weekend |
| Holiday | 4 | is_holiday, holiday_proximity, days_to_next |
| Promotion | 2 | onpromotion, promo_item_interaction |
| Store/Item | 7 | store_avg, item_avg, cluster_avg, family_avg |
| Derived | 5 | perishable, month_start, month_end, is_payday |

---

## MLflow Experiment Tracking

This project includes comprehensive experiment tracking with **16 MLflow experiments** documenting the full modeling journey.

### Experiments Summary

| Phase | Experiments | Key Findings |
|-------|-------------|--------------|
| Week 3 Baseline | 2 | Initial XGBoost performance |
| Week 3 Tuning | 4 | Hyperparameter optimization |
| Week 3 Ablation | 6 | Feature reduction (+4.5% RMSE gain) |
| Week 3 LSTM | 2 | Neural network comparison |
| Full Pipeline | 2 | Production scale validation |

### View Experiments
```bash
./scripts/start_mlflow_ui.sh
# Open http://127.0.0.1:5000
```

### Tracked Metrics
- RMSE, MAE, MAPE, Bias
- Training time, samples
- Feature importance rankings
- Model hyperparameters

---

## Methodology

This project follows the **Data Science Collaboration Methodology v1.1**, a structured framework I developed for systematic project execution with:
- Weekly planning and daily checkpoints
- Decision logging (DEC-XXX format)
- Handoff documentation between phases
- Progressive validation and traceability

See `docs/` for methodology documentation.

---

## Acknowledgments

- **Data Source**: [Kaggle Corporación Favorita Competition](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)
- **Course**: MasterSchool Time Series Forecasting

---

## Author

**Alberto Diaz Durana**  
[GitHub](https://github.com/albertodiazdurana) | [LinkedIn](https://www.linkedin.com/in/albertodiazdurana/)

---

## License

MIT License - Academic project using Kaggle competition data under competition terms of use.
---



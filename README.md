# Corporación Favorita Grocery Sales Forecasting

**Time Series Analysis & Forecasting Project**

A comprehensive machine learning project analyzing Ecuadorian grocery sales data to develop forecasting models for retail inventory optimization. This project implements advanced feature engineering, multi-method outlier detection, and temporal pattern analysis using XGBoost and LSTM approaches.

---

## Project Overview

### Objective
Forecast daily unit sales for Corporación Favorita stores in the Guayas region (January-March 2014) to optimize inventory planning and reduce waste through accurate demand prediction.

**Data Source**: [Kaggle - Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)

### Business Context
- **Domain**: Retail time series forecasting
- **Scope**: Guayas region, top-3 product families (GROCERY I, BEVERAGES, CLEANING)
- **Dataset**: 300,896 transactions across 54 stores, 683 items, 91 days
- **Evaluation Metric**: NWRMSLE (Normalized Weighted Root Mean Squared Logarithmic Error)

### Success Criteria
- **Quantitative**: NWRMSLE improvement over naive baseline, forecast accuracy within business tolerance
- **Qualitative**: Interpretable models with actionable insights for inventory planners
- **Technical**: Reproducible pipeline with end-to-end execution

## Web Application

🚀 **[Live Forecasting App 🚀](https://github.com/albertodiazdurana/Demand-forecasting-in-retail-app)** - Streamlit deployment (separate repository)

Live demo of the production XGBoost model with interactive forecasting capabilities. Users can select stores, products, and generate single-day or multi-day forecasts with visualizations.

**Repository**: [Demand-forecasting-in-retail-app](https://github.com/albertodiazdurana/Demand-forecasting-in-retail-app)

---

## Current Status

**Phase**: Week 3 - Modeling & Analysis (IN PROGRESS)

### Completed Phases

#### Week 1: Exploratory Data Analysis
- **Key Finding**: 99.1% data sparsity is normal retail reality (0.9% of store-item-date combinations have sales)
- **Business Insights**:
  - 33.9% weekend sales lift over weekdays
  - Strong autocorrelation across all lag intervals
  - Pareto distribution: 34% of items generate 80% of sales
- **Technical Accomplishments**:
  - Three-method outlier detection (IQR, Z-score, Isolation Forest)
  - Complete temporal pattern analysis
  - Strategic decision to maintain sparse format

**Notebooks**: `w01_d01` through `w01_d05` (5 notebooks)  
**Documentation**: `Week1_ProjectPlan_v2_Expanded.md`, `Week1_to_Week2_Handoff.md`

#### Week 2: Feature Engineering
- **Features Engineered**: 29 features across 5 categories
- **Quality Status**: All validated (no data leakage, temporal order preserved)
- **Key Decisions**:
  - DEC-011: Keep NaN in lag features (tree models handle natively)
  - DEC-012: Include oil features despite weak linear correlation (tree models may find non-linear patterns)

**Feature Categories**:
1. **Temporal Features (10)**: Lags (4), rolling statistics (6)
2. **Oil Features (6)**: Price levels, lags, momentum derivatives
3. **Aggregation Features (11)**: Store, cluster, and item-level statistics
4. **Promotion Features (2)**: Interaction terms with item/cluster averages

**Notebooks**: `w02_d01` through `w02_d05` (5 notebooks)  
**Documentation**: `Week2_ProjectPlan_v2_Expanded.md`, `Week2_to_Week3_Handoff.md`

#### Week 3: Modeling & Analysis (Current)
**Planned Activities**:
- XGBoost baseline training and comprehensive evaluation
- MLflow experiment tracking setup
- Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- Feature importance validation (permutation, SHAP, ablation studies)
- Optional LSTM model comparison
- Preprocessing artifacts export for deployment

**Documentation**: `Week3_ProjectPlan_v1.md`

---

## Repository Structure

```
retail_demand_analysis/
├── notebooks/
│   ├── w01_d01_SETUP_data_inventory.ipynb        # Environment setup, data inventory
│   ├── w01_d02_EDA_data_loading_filtering.ipynb  # Data loading, Guayas filtering
│   ├── w01_d03_EDA_quality_preprocessing.ipynb   # Quality analysis, outlier detection
│   ├── w01_d04_EDA_temporal_patterns.ipynb       # Time series patterns, seasonality
│   ├── w01_d05_EDA_context_export.ipynb          # Store/product analysis, final export
│   ├── w02_d01_FE_lags.ipynb                     # Lag features (1/7/14/30 days)
│   ├── w02_d02_FE_rolling.ipynb                  # Rolling statistics (7/14/30 windows)
│   ├── w02_d03_FE_oil.ipynb                      # Oil price features and derivatives
│   ├── w02_d04_FE_aggregations.ipynb             # Store/cluster/item aggregations
│   └── w02_d05_FE_final.ipynb                    # Promotion interactions, final dataset
├── data/
│   ├── raw/                                       # Original Kaggle CSVs
│   ├── processed/                                 # Filtered, cleaned datasets
│   └── results/                                   # Model outputs, forecasts
├── outputs/
│   └── figures/                                   # Visualizations by phase
│       ├── eda/
│       ├── features/
│       └── models/
├── docs/
│   ├── plans/                                     # Weekly project plans
│   │   ├── Week1_ProjectPlan_v2_Expanded.md
│   │   ├── Week2_ProjectPlan_v2_Expanded.md
│   │   └── Week3_ProjectPlan_v1.md
│   ├── decisions/                                 # Decision log
│   │   ├── DEC-011_Lag_NaN_Strategy.md
│   │   └── DEC-012_Oil_Features_Inclusion.md
│   └── handoffs/                                  # Phase handoff documents
│       ├── Week1_to_Week2_Handoff.md
│       └── Week2_to_Week3_Handoff.md
└── README.md
```

---

## Key Technical Decisions

### DEC-011: Lag NaN Strategy
**Decision**: Keep NaN values in lag features instead of imputing  
**Rationale**: Tree-based models (XGBoost) handle missing values natively. Imputation could introduce bias by assuming zero demand when no historical data exists.  
**Impact**: 27K-40K NaN per lag feature (9-13% of dataset), preserved for model training

### DEC-012: Oil Features Inclusion
**Decision**: Include oil price features despite weak linear correlation (r=+0.01)  
**Rationale**:
- Tree models can capture non-linear relationships
- Oil prices may interact with other features (store type, product category)
- Multiple time scales (lag7/14/30, change7/14) capture different economic dynamics
- Feature importance analysis in Week 3 will validate utility

---

## Dataset Specifications

### Source Data
- **Origin**: [Kaggle Competition - Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)
- **Competition Files**: train.csv, stores.csv, items.csv, oil.csv, holidays_events.csv, transactions.csv
- **Region Filter**: Guayas (Ecuador's largest commercial region)
- **Product Filter**: Top-3 families by item count (58.4% of all items)
- **Time Period**: January 1 - March 31, 2014 (91 days)

### Final Feature-Engineered Dataset
- **File**: `w02_d05_FE_final.pkl`
- **Dimensions**: 300,896 rows × 57 columns
- **Size**: 110.4 MB
- **Sparsity**: 99.1% (normal retail characteristic)
- **Features**: 29 engineered + 28 original/derived

### Feature Categories Breakdown

| Category               | Count | Examples                                                   |
| ---------------------- | ----- | ---------------------------------------------------------- |
| Temporal (Lags)        | 4     | unit_sales_lag1, lag7, lag14, lag30                        |
| Temporal (Rolling)     | 6     | unit_sales_7d_avg, 14d_std, 30d_avg                        |
| Oil Economics          | 6     | oil_price, oil_price_lag7, oil_price_change7               |
| Store Aggregations     | 3     | store_avg_sales, store_median_sales, store_std_sales       |
| Cluster Aggregations   | 3     | cluster_avg_sales, cluster_median_sales, cluster_std_sales |
| Item Aggregations      | 5     | item_avg_sales, item_total_sales, item_transaction_count   |
| Promotion Interactions | 2     | promo_item_avg_interaction, promo_cluster_interaction      |

---

## Quick Start

### Data Download

Download the competition data from Kaggle:
1. Visit the [competition data page](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)
2. Download all CSV files to `data/raw/` directory
3. Required files: train.csv, stores.csv, items.csv, oil.csv, holidays_events.csv, transactions.csv

### Environment Setup

```bash
# Clone repository
git clone https://github.com/albertodiazdurana/retail_demand_analysis.git
cd retail_demand_analysis

# Create environment (requires Python 3.8+)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run base environment setup
python setup_base_environment.py
```

### Notebook Execution Order

**Week 1 - EDA**:
1. `w01_d01_SETUP_data_inventory.ipynb` - Environment validation
2. `w01_d02_EDA_data_loading_filtering.ipynb` - Data loading and filtering
3. `w01_d03_EDA_quality_preprocessing.ipynb` - Quality analysis
4. `w01_d04_EDA_temporal_patterns.ipynb` - Temporal pattern discovery
5. `w01_d05_EDA_context_export.ipynb` - Business context analysis

**Week 2 - Feature Engineering**:
1. `w02_d01_FE_lags.ipynb` - Lag feature creation
2. `w02_d02_FE_rolling.ipynb` - Rolling statistics
3. `w02_d03_FE_oil.ipynb` - Oil price features
4. `w02_d04_FE_aggregations.ipynb` - Aggregation features
5. `w02_d05_FE_final.ipynb` - Promotion features and final dataset

**Week 3 - Modeling** (In Progress):
- TBD: Baseline, tuning, validation notebooks

---

## Key Findings

### Business Insights

**Sales Patterns**:
- **Weekend Effect**: 33.9% higher sales on weekends vs weekdays
- **Autocorrelation**: Strong persistence (lag7: r=0.40, lag14: r=0.32)
- **Product Distribution**: Classic Pareto (34% of items generate 80% of sales)

**Promotion Impact**:
- **Coverage**: Only 4.6% of transactions promoted
- **Lift**: +74% mean sales increase, +66.7% median increase
- **Category Variation**: CLEANING (+71.7%), GROCERY I (+70.9%), BEVERAGES (+63.3%)

**Sparsity Reality**:
- **Finding**: 99.1% sparsity in store-item-date combinations
- **Interpretation**: Normal retail behavior (most items don't sell daily at each store)
- **Strategy**: Maintain sparse format, avoid expanding to dense matrix

### Technical Insights

**Feature Engineering**:
- **Lag Correlations**: lag7 strongest (0.40), capturing weekly seasonality
- **Oil Correlation Shift**: Aggregate level (r=-0.55) vs granular level (r=+0.01)
- **Rolling Windows**: 7d/14d/30d capture different smoothing levels
- **Promotion Interactions**: 2.4x better correlation than raw promotion flag

**Data Quality**:
- **Negative Sales**: Returns/adjustments clipped to zero for forecasting
- **Missing Dates**: Complete daily calendar filled (true zeros)
- **Outliers**: Multi-method detection (IQR + Z-score + Isolation Forest)

---

## Methodology & Standards

### Time Series Principles
- **Temporal Order**: Maintained throughout (no row shuffling)
- **No Data Leakage**: Features use only past information
- **Validation Strategy**: Time-series split (train: Jan-Feb, test: March)
- **Complete Calendar**: No gaps in date index per store-item

### Code Standards
- **File Naming**: `wYY_dXX_PHASE_description` format
- **Notebook Structure**: 5-6 sections, ~400-500 lines each
- **Output Requirements**: Every cell shows visible results (shapes, metrics, correlations)
- **Documentation**: Markdown cells precede all code sections

### Evaluation Approach
- **Primary Metric**: NWRMSLE (perishable items weighted 1.25x)
- **Cross-Validation**: Expanding window (respects temporal order)
- **Interpretability**: SHAP values, permutation importance, ablation studies

---

## Next Steps (Week 3)

1. **Baseline Model**: XGBoost with default parameters, comprehensive evaluation
2. **MLflow Setup**: Experiment tracking for reproducibility
3. **Hyperparameter Tuning**: GridSearchCV/RandomizedSearchCV
4. **Feature Validation**: Permutation importance, SHAP, ablation studies
5. **LSTM Comparison**: Optional deep learning approach
6. **Artifacts Export**: Scaler, feature columns, best model for deployment

---

## Documentation References

### Project Plans
- `Week1_ProjectPlan_v2_Expanded.md` - EDA phase planning
- `Week2_ProjectPlan_v2_Expanded.md` - Feature engineering roadmap
- `Week3_ProjectPlan_v1.md` - Modeling strategy (current)

### Handoff Documents
- `Week1_to_Week2_Handoff.md` - EDA to FE transition
- `Week2_to_Week3_Handoff.md` - FE to Modeling transition

### Checkpoint Files
- `w01_d03_checkpoint.md` through `w01_d05_checkpoint.md` - Week 1 progress
- `w02_d01_checkpoint.md` through `w02_d05_checkpoint.md` - Week 2 progress

### Decision Logs
- `DEC-011_Lag_NaN_Strategy.md` - Handling missing lag values
- `DEC-012_Oil_Features_Inclusion.md` - Oil price feature justification

---

## Dependencies

### Core Libraries
- **Data Processing**: pandas, numpy, dask
- **Visualization**: matplotlib, seaborn
- **Modeling**: xgboost, tensorflow/keras (LSTM)
- **Experiment Tracking**: mlflow
- **Feature Engineering**: scikit-learn

### Environment
- Python 3.8+
- Jupyter Notebook/Lab
- See `requirements.txt` for complete dependency list

---

## Author

**Alberto Diaz Durana**  
[GitHub](https://github.com/albertodiazdurana) | [LinkedIn](https://www.linkedin.com/in/albertodiazdurana/)

MasterSchool's Program Project - Time Series Analysis & Forecasting

---

## License

This project is part of academic coursework. Data sourced from [Kaggle's Corporación Favorita Grocery Sales Forecasting competition](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting) under Kaggle's competition terms of use.

---

## Acknowledgments

- **Data Source**: [Kaggle - Corporación Favorita Grocery Sales Forecasting Competition](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)
- **Framework**: Data Science Collaboration Methodology
- **Guidance**: Academic advisor and peer reviewers

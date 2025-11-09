# Corporación Favorita Grocery Sales Forecasting

Time series forecasting project to predict daily unit sales for grocery stores in Guayas region, Ecuador.

**Author:** Alberto Diaz Durana  
**Duration:** 4 weeks (November 2025)  
**Domain:** Time Series Analysis  
**Dataset:** Kaggle - Corporación Favorita Grocery Sales Forecasting

---

## Project Overview

### Objective
Forecast daily unit_sales for products in Guayas stores (January-March 2014) to optimize inventory management and reduce waste in retail operations.

### Business Value
- Right-size inventory (avoid overstocking and stockouts)
- Identify seasonal patterns and promotional impacts
- Support data-driven procurement decisions
- Reduce waste for perishable items

### Scope
- **Region**: Guayas province only (filtered from national dataset)
- **Products**: Top-3 product families by item count
- **Sample**: 300,000 transactions for development speed
- **Forecast horizon**: 3 months (Jan-Mar 2014)
- **Evaluation metric**: NWRMSLE (Normalized Weighted Root Mean Squared Logarithmic Error)

---

## Project Structure

```
retail_demand_analysis/
├── notebooks/              # Jupyter notebooks (sequential execution)
│   ├── 01_SETUP_environment.ipynb
│   ├── 02_EDA_data_quality.ipynb
│   ├── 03_EDA_temporal_patterns.ipynb
│   ├── 04_FE_core_features.ipynb
│   ├── 05_MODELING_baseline.ipynb
│   └── 06_MODELING_advanced.ipynb
├── data/
│   ├── raw/                # Original Kaggle CSV files
│   ├── processed/          # Filtered and cleaned datasets
│   └── results/            # Model outputs organized by phase
│       ├── eda/
│       ├── features/
│       └── models/
├── outputs/
│   └── figures/            # All visualizations organized by phase
│       ├── eda/
│       ├── features/
│       └── models/
├── docs/
│   ├── plans/              # Weekly project plans
│   ├── decisions/          # Decision log (analytical choices)
│   └── reports/            # Final reports and summaries
├── presentation/           # Final stakeholder presentation
├── .venv/                  # Virtual environment (not in Git)
├── .gitignore
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

---

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- Git
- Kaggle account (for data download)
- 8GB+ RAM recommended (for Dask operations)

### Installation

1. **Clone repository** (if using Git remote):
```powershell
git clone <repository-url>
cd retail_demand_analysis
```

2. **Create virtual environment**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. **Install dependencies**:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Configure Kaggle API**:
   - Create Kaggle account at https://www.kaggle.com
   - Go to Account → API → Create New API Token
   - Save `kaggle.json` to `C:\Users\<username>\.kaggle\`
   - Set permissions: `kaggle.json` should be read-only

5. **Verify installation**:
```powershell
python -c "import pandas, dask, statsmodels; print('OK - Installation successful')"
```

---

## Data Sources

### Primary Dataset
- **Source**: [Kaggle Competition - Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)
- **Files**:
  - `train.csv` (479 MB) - Daily sales transactions
  - `stores.csv` - Store metadata (54 stores)
  - `items.csv` - Product metadata (4,100 items, 33 families)
  - `oil.csv` - Daily oil prices (economic indicator)
  - `holidays_events.csv` - Holiday calendar
  - `transactions.csv` - Daily transaction counts

### Data Characteristics
- **Temporal range**: 2013-01-01 to 2017-08-15
- **Target variable**: unit_sales (continuous, can be negative for returns)
- **Missing dates**: Represent zero sales (not missing data)
- **Negative sales**: Product returns (clipped to zero for forecasting)

---

## Project Phases

### Week 1: Exploration & Understanding
**Focus**: Data quality, temporal patterns, cohort definition

**Deliverables**:
- Guayas region filtered dataset (300K rows)
- Top-3 product families identified
- 8-step EDA: missing data, outliers, calendar gaps, features, visualizations
- Holiday, perishable, and oil price analysis
- `guayas_prepared.csv` (clean, featured dataset)

### Week 2: Feature Development
**Focus**: Advanced feature engineering for time series

**Deliverables**:
- Lag features (1/7/14/30 days)
- Rolling statistics (7/14/30-day windows)
- Holiday proximity features
- Store/item aggregations
- Feature dictionary

### Week 3: Analysis & Modeling
**Focus**: Model training, validation, forecasting

**Deliverables**:
- Baseline models (naive, ARIMA)
- Advanced models (Prophet, LSTM)
- Time-series cross-validation
- Multi-step forecasts with uncertainty quantification
- Model evaluation report (NWRMSLE)

### Week 4: Communication & Delivery
**Focus**: Code consolidation, stakeholder deliverables

**Deliverables**:
- Consolidated notebooks (60% reduction)
- Presentation (15-20 slides)
- Technical report (20-25 pages)
- Lightweight web app interface
- Video walkthrough

---

## Usage

### Running Notebooks
Execute notebooks sequentially:

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Start Jupyter
jupyter notebook

# Open notebooks in order: 01 → 02 → 03 → ...
```

### Key Conventions
- **Notebook naming**: `##_PHASE_description.ipynb`
- **Data paths**: Use relative paths from project root
- **Output format**: Save to appropriate `data/results/` or `outputs/figures/` subfolder
- **Text indicators**: WARNING/OK/ERROR (no emojis)

---

## Methodology

This project follows a structured 4-phase framework:

1. **Exploration & Understanding**: Data quality, cohort definition, temporal patterns
2. **Feature Development**: Lag features, rolling statistics, external factors
3. **Analysis & Modeling**: Baseline and advanced models, validation, forecasting
4. **Communication & Delivery**: Code consolidation, presentation, documentation

**Key practices**:
- Decision log for major analytical choices
- Progressive execution (cell-by-cell testing)
- Reproducible notebooks (~400 lines, 5-6 sections)
- Time-series validation (no shuffling, expanding window)
- Professional documentation (no emojis, clear headers)

---

## Evaluation Metric

**NWRMSLE**: Normalized Weighted Root Mean Squared Logarithmic Error

$$
NWRMSLE = \sqrt{\frac{\sum_{i=1}^n w_i \left(\ln(\hat{y}_i + 1) - \ln(y_i + 1)\right)^2}{\sum_{i=1}^n w_i}}
$$

**Weights**:
- Perishable items: 1.25
- Non-perishable items: 1.00

**Rationale**: Log-scale reduces penalty for large differences when both predicted and actual values are large.

---

## Key Challenges

1. **Large dataset**: train.csv (479 MB) requires Dask for streaming
2. **Missing dates**: Fill complete calendar per store-item (zero sales assumption)
3. **Negative sales**: Product returns (clip to zero for forecasting)
4. **Temporal dependencies**: Strong autocorrelation (lag features critical)
5. **Perishable items**: Higher accuracy required (waste cost)
6. **Oil prices**: Weak correlation expected (verify during EDA)

---

## Dependencies

### Core Libraries (Week 1)
```
pandas==2.1.4          # Data manipulation
numpy==1.26.2          # Numerical operations
dask[dataframe]==2023.12.1  # Large file streaming
matplotlib==3.8.2      # Visualization
seaborn==0.13.0        # Statistical plots
statsmodels==0.14.1    # Time series analysis
scipy==1.11.4          # Statistical functions
jupyter==1.0.0         # Notebook environment
kaggle==1.6.6          # Data download
```

### Additional Libraries (Weeks 2-3)
Install when needed:
```powershell
pip install prophet scikit-learn lightgbm tensorflow
```

---

## Documentation

### Key Documents
- `docs/plans/Week1_ProjectPlan.md` - Week 1 detailed plan
- `docs/decisions/decision_log.md` - Major analytical decisions
- `docs/data_inventory.md` - Dataset characteristics
- `docs/reports/` - Final reports and summaries

### Decision Log
All major analytical choices documented with:
- Context and options considered
- Decision made and rationale
- Impact and reversibility
- Follow-up actions

---

## Contributing

This is an individual academic project. For questions or feedback:
- **Author**: Alberto Diaz Durana
- **Advisor**: [Academic advisor name]

---

## License

Academic project - Data sourced from Kaggle competition (subject to competition rules).

---

## Acknowledgments

- **Data source**: Kaggle - Corporación Favorita Grocery Sales Forecasting
- **Framework**: Data Science Collaboration Methodology (Academic Edition v1.0)
- **Tools**: Python ecosystem (pandas, scikit-learn, statsmodels)

---

## Project Status

**Current Phase**: Week 1 - Exploration & Understanding  
**Last Updated**: November 2025  
**Status**: In Progress

---

**End of README**
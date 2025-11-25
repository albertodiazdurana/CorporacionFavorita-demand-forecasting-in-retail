# Corporación Favorita Grocery Sales Forecasting
## Week 4 Project Plan: Communication & Deployment (Version 1.0)

**Prepared by:** Alberto Diaz Durana  
**Timeline:** Week 4 (5 working days, 20 hours total)  
**Phase:** Phase 4 - Communication & Delivery  
**Previous Phase:** Week 3 - Analysis & Modeling (COMPLETE) + Full Pipeline (IN PROGRESS)  
**Plan Version:** 1.0  
**Date:** 2025-11-24

---

## 1. Purpose

**Objective:**  
Deploy production model as a live web application and create video walkthrough demonstrating the complete demand forecasting solution for Corporación Favorita.

**Business Value:**
- Transform ML model into accessible business tool for demand planners
- Demonstrate end-to-end data science capability (exploration → deployment)
- Create portfolio-ready project showcasing time series forecasting expertise
- Enable Guayas store managers to generate on-demand sales forecasts

**Deliverables:**
1. Production pipeline complete (FULL_02 with XGBoost vs LSTM comparison)
2. Publication-ready visualizations supporting findings
3. Two GitHub repositories (analysis + deployment)
4. Live Streamlit web application (deployed to cloud)
5. Video walkthrough (10-15 minutes)
6. Summary document complementing README

**Resources:**
- Time allocation: 20 hours (4 hours/day × 5 days)
- Time buffer: 20% included (16 hours core work + 4 hours buffer)
- Environment: WSL2 Ubuntu 22.04 with GPU (Quadro T1000)
- Deployment: Streamlit Community Cloud (free tier)

---

## 2. Inputs & Dependencies

### From Full Pipeline (FULL_01 Complete)

**File:** `data/processed/full_featured_data.pkl`

| Metric | Value |
|--------|-------|
| Rows | 4,801,160 |
| Features | 33 (per DEC-014) |
| Period | Oct 1, 2013 - Mar 31, 2014 |
| Stores | 10 (Guayas) |
| Items | 2,638 |
| Families | 32 |
| File size | 1.3 GB |

### From Week 3 (Reference)

**Sample Results (300K rows):**
- XGBoost Tuned: RMSE 6.4860
- LSTM: RMSE 6.2552 (13.28% improvement over baseline)
- Winner: LSTM by 4.5%

**Artifacts Available:**
- lstm_model.keras (0.34 MB)
- scaler.pkl (1.84 KB)
- feature_columns.json
- model_config.json
- model_usage.md

### Decision Log (Active)

| Decision | Status | Application |
|----------|--------|-------------|
| DEC-013 | APPLY | 7-day train/test gap |
| DEC-014 | APPLY | 33 features |
| DEC-015 | REJECTED | Full 2013 training failed |
| DEC-016 | APPLY | Q4+Q1 temporal consistency |
| DEC-017 | TBD | If XGBoost wins at scale |

### External Dependencies

- GitHub account (repositories)
- Streamlit Community Cloud account (deployment)
- Screen recording software (video)
- Google Drive or similar (video hosting)

---

## 3. Execution Timeline

| Day | Focus Area | Core Hours | Buffer | Total | Key Deliverables |
|-----|------------|------------|--------|-------|------------------|
| 1 | Production Pipeline + Visualizations | 3.2h | 0.8h | 4h | FULL_02 complete, 5-6 figures, artifacts |
| 2 | GitHub Repos + Streamlit Development | 3.2h | 0.8h | 4h | 2 repos, working local app |
| 3 | Streamlit Cloud Deployment | 3.2h | 0.8h | 4h | Live app at public URL |
| 4 | Video Recording | 3.2h | 0.8h | 4h | 10-15 min video walkthrough |
| 5 | Summary & Final Review | 3.2h | 0.8h | 4h | SUMMARY.md, final submission |
| **Total** | | **16h** | **4h** | **20h** | **Complete portfolio project** |

### Cumulative Buffer Tracking

| Checkpoint | Buffer Allocated | Buffer Used | Buffer Remaining | Notes |
|------------|------------------|-------------|------------------|-------|
| Week 4 Start | 4h (20%) | 0h | 4h | Starting position |
| End of Day 1 | 0.8h | TBD | TBD | Update after FULL_02 |
| End of Day 2 | 1.6h cumulative | TBD | TBD | Update after repos |
| End of Day 3 | 2.4h cumulative | TBD | TBD | Update after deployment |
| End of Day 4 | 3.2h cumulative | TBD | TBD | Update after video |
| End of Day 5 | 4h cumulative | TBD | TBD | Final buffer status |

---

## 4. Detailed Deliverables

### Day 1: Production Pipeline Finalization + Visualizations
**Goal:** Complete FULL_02, generate publication-ready plots, export production artifacts

**Total Time:** 4 hours (3.2h core + 0.8h buffer)

#### Part 1: Complete FULL_02_train_final_model.ipynb (2.5 hours)

**Activities:**
- Load full_featured_data.pkl (4.8M rows)
- Apply DEC-016 split (Q4 2013 + Q1 2014 training, March 2014 test)
- Apply DEC-013 gap (7 days: Feb 22-28)
- Train XGBoost with Week 3 hyperparameters
- Train LSTM with GPU acceleration
- Calculate 4 metrics for both: RMSE, MAE, MAPE (non-zero), Bias
- Log both runs to MLflow
- Determine winner at scale

**Expected Outputs:**
- XGBoost: RMSE [TBD], training time [TBD]
- LSTM: RMSE [TBD], training time [TBD] (GPU)
- Winner: [TBD]

**MLflow Runs:**
- xgboost_full_q4q1
- lstm_full_q4q1

#### Part 2: Generate Visualizations (1 hour)

**Figures to Create:**

1. **model_comparison_full.png**
   - Bar chart: XGBoost vs LSTM (RMSE, MAE)
   - Side-by-side with Week 3 sample results

2. **sample_vs_full_comparison.png**
   - Table visualization: 300K vs 4.8M results
   - Both models compared

3. **feature_importance_full.png**
   - Top 10 features (permutation importance)
   - Validate DEC-014 at scale

4. **actual_vs_predicted.png**
   - Scatter plot: y_true vs y_pred
   - Show model fit quality

5. **training_timeline.png**
   - Data split visualization
   - Training/gap/test periods marked

6. **model_progression.png**
   - Week 1 baseline → Week 3 → Full Pipeline
   - Show improvement journey

**Save Location:** `outputs/figures/full_pipeline/`

#### Part 3: Export Production Artifacts (30 min)

**Files to Export:**
- `artifacts/[winner]_model_full.keras` or `.pkl`
- `artifacts/scaler_full.pkl`
- `artifacts/feature_columns.json`
- `artifacts/model_config_full.json`

**Deliverables:**
- [x] FULL_02 notebook complete
- [x] MLflow runs logged (2 runs)
- [x] 5-6 publication-ready figures
- [x] Production artifacts exported
- [x] FULL_02_checkpoint.md created

---

### Day 2: GitHub Repositories + Streamlit Development
**Goal:** Prepare both repos, build working Streamlit app locally

**Total Time:** 4 hours (3.2h core + 0.8h buffer)

#### Part 1: Main Analysis Repository (1.5 hours)

**Repository:** `retail_demand_analysis` (existing)

**Structure:**
```
retail_demand_analysis/
├── notebooks/
│   ├── w01_d01 through w01_d05    # Week 1 EDA
│   ├── w02_d01 through w02_d05    # Week 2 Features
│   ├── w03_d01 through w03_d05    # Week 3 Modeling
│   ├── FULL_01_data_to_features.ipynb
│   └── FULL_02_train_final_model.ipynb
├── data/
│   ├── raw/                        # Original Kaggle CSVs
│   └── processed/                  # Featured datasets
├── artifacts/                      # Production model files
├── outputs/
│   └── figures/                    # All visualizations
├── docs/
│   ├── decisions/                  # DEC-013 through DEC-017
│   ├── checkpoints/                # Daily checkpoints
│   └── plans/                      # Weekly project plans
├── README.md
├── requirements.txt
└── .gitignore
```

**Activities:**
- Clean notebook outputs (clear unnecessary cells)
- Verify all notebooks run without errors
- Write comprehensive README:
  - Project overview
  - Key findings (DEC-016, model comparison)
  - Quick start instructions
  - Results summary table
  - Links to Streamlit app and video
- Create requirements.txt
- Create .gitignore (exclude data/raw/, large files)
- Commit and push

#### Part 2: Streamlit App Repository (2.5 hours)

**Repository:** `corporacion_favorita_app` (NEW - separate repo)

**Structure (per course guidelines):**
```
corporacion_favorita_app/
├── app/
│   ├── __init__.py
│   ├── main.py              # Streamlit UI
│   └── config.py            # Paths, URIs, constants
├── model/
│   ├── __init__.py
│   └── model_utils.py       # Load model, predict functions
├── data/
│   ├── __init__.py
│   └── data_utils.py        # Data loading, feature engineering
├── artifacts/               # Model, scaler, feature_columns
│   ├── [model].keras or .pkl
│   ├── scaler.pkl
│   └── feature_columns.json
├── mlflow_results/          # Local MLflow store (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

**Implement config.py:**
```python
# Paths and configuration
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Model configuration
MODEL_PATH = ARTIFACTS_DIR / "[model_file]"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns.json"

# Forecast settings
SEQ_LEN = 1  # LSTM input sequence length
FORECAST_START = "2014-01-01"
FORECAST_END = "2014-03-31"

# UI settings
STORES = [24, 26, 27, 28, 29, 30, 32, 34, 35, 36, 51]
MAX_FORECAST_DAYS = 30
```

**Implement model/model_utils.py:**
```python
def load_model():
    """Load production model from artifacts."""
    pass

def load_scaler_and_features():
    """Load scaler and feature column list."""
    pass

def predict_scaled(model, scaler, X):
    """Generate predictions with proper scaling."""
    pass
```

**Implement data/data_utils.py:**
```python
def load_historical_data():
    """Load filtered historical data for display."""
    pass

def engineer_features(df, forecast_date):
    """Create features for inference."""
    pass

def prepare_forecast_input(df, date, store, item):
    """Prepare input for single prediction."""
    pass
```

**Implement app/main.py:**
```python
import streamlit as st
# ... imports

st.title("Corporación Favorita Sales Forecast")
st.write("Demand forecasting for Guayas stores")

# Sidebar - Configuration
store = st.sidebar.selectbox("Store", STORES)
# ... item selector

# Date picker
forecast_date = st.date_input("Forecast Date", ...)

# Forecast mode
mode = st.radio("Forecast Mode", ["Single Day", "Next N Days"])
if mode == "Next N Days":
    n_days = st.slider("Days", 1, 30, 7)

# Generate forecast button
if st.button("Generate Forecast"):
    # ... prediction logic
    
    # Plot: history + forecast
    st.pyplot(fig)
    
    # Download CSV
    st.download_button("Download CSV", csv_data)
```

**Test locally:**
```bash
cd corporacion_favorita_app
streamlit run app/main.py
```

**Create .gitignore:**
```
# Large/local files
mlflow_results/
data/*.csv
*.pkl.bak

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# OS
.DS_Store
.vscode/
```

**Deliverables:**
- [x] Main repo cleaned and pushed
- [x] Streamlit repo created with full structure
- [x] App working locally (http://localhost:8501)
- [x] Both READMEs drafted
- [x] Screenshots captured

---

### Day 3: Streamlit Cloud Deployment
**Goal:** Deploy app to Streamlit Community Cloud with public URL

**Total Time:** 4 hours (3.2h core + 0.8h buffer)

#### Part 1: Deployment Preparation (1 hour)

**Pre-deployment Checklist:**
- [ ] requirements.txt complete and tested
- [ ] No hardcoded absolute paths in code
- [ ] Model artifacts in repo (or accessible via URL)
- [ ] Config uses relative paths / environment variables
- [ ] App runs locally without errors
- [ ] Repo pushed to GitHub (public)

**Model Artifact Strategy:**

| Size | Strategy |
|------|----------|
| <25MB | Include in repo directly |
| 25-100MB | Git LFS or separate download |
| >100MB | Host on Google Drive / Hugging Face |

**For our model (~0.5MB):** Include directly in `artifacts/` folder.

**Update config.py for cloud:**
```python
import os
from pathlib import Path

# Cloud-compatible paths
if os.environ.get("STREAMLIT_CLOUD"):
    BASE_DIR = Path(__file__).parent.parent
else:
    BASE_DIR = Path(__file__).parent.parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"
```

#### Part 2: Streamlit Community Cloud Deployment (1 hour)

**Step 1: Access Streamlit Cloud**
- Go to https://share.streamlit.io
- Sign in with GitHub account

**Step 2: Create New App**
- Click "New app"
- Select repository: `corporacion_favorita_app`
- Branch: `main`
- Main file path: `app/main.py`

**Step 3: Configure Settings**
- Python version: 3.11
- Advanced settings (if needed):
  - Secrets for API keys
  - Environment variables

**Step 4: Deploy**
- Click "Deploy"
- Wait for build (2-5 minutes)
- Monitor logs for errors

**Expected URL:** `https://[your-username]-corporacion-favorita-app-[hash].streamlit.app`

#### Part 3: Deployment Testing (1 hour)

**Functional Tests:**
- [ ] App loads without errors
- [ ] Date picker works
- [ ] Store/item selectors populate
- [ ] Single day forecast generates
- [ ] N-day forecast generates
- [ ] Plot displays correctly
- [ ] CSV download works
- [ ] No timeout errors

**Performance Tests:**
- [ ] Page loads in <5 seconds
- [ ] Forecast generates in <10 seconds
- [ ] No memory errors

**Edge Case Tests:**
- [ ] First day of range works
- [ ] Last day of range works
- [ ] Maximum N-days (30) works

#### Part 4: Documentation Update (1 hour)

**Update Streamlit README.md:**
```markdown
# Corporación Favorita Sales Forecast App

## Live Demo
🚀 **[Launch App](https://[your-url].streamlit.app)**

## Overview
Interactive demand forecasting for Guayas grocery stores.

## Features
- Select store and product
- Single day or N-day forecasts
- Historical + forecast visualization
- CSV download

## Local Development
```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## Model Performance
- Model: [XGBoost/LSTM]
- RMSE: [value]
- Training: 4.8M rows (Guayas, Oct 2013 - Feb 2014)

## Screenshots
![App Screenshot](docs/screenshot.png)
```

**Update Main Repo README:**
- Add link to deployed app
- Add deployment URL to project overview

**Deliverables:**
- [x] App deployed to Streamlit Community Cloud
- [x] Public URL working
- [x] All features tested
- [x] README updated with deployment URL
- [x] Screenshots added

---

### Day 4: Video Recording
**Goal:** Record complete project walkthrough with live deployed app

**Total Time:** 4 hours (3.2h core + 0.8h buffer)

#### Part 1: Video Preparation (1 hour)

**Script Outline (12-15 minutes):**

**1. Introduction (1-2 min)**
- "Hello, I'm Alberto Diaz Durana"
- Problem: Corporación Favorita needs accurate demand forecasts
- Objective: Predict daily sales for 2,600+ products in Guayas stores
- Business value: Reduce waste, optimize inventory

**2. Main Repository Walkthrough (3 min)**
- Show GitHub repo structure
- Highlight key notebooks:
  - Week 1: EDA findings (seasonality, autocorrelation)
  - Week 2: Feature engineering (33 features)
  - Week 3: Model comparison (XGBoost vs LSTM)
  - FULL_01/02: Production pipeline
- Show MLflow experiment tracking
- Display key visualizations

**3. Key Findings (3 min)**
- DEC-016: Temporal consistency principle
  - "Using 6 months of recent data beat using 12 months of older data"
- Model comparison at scale (4.8M rows)
  - XGBoost vs LSTM results
  - Winner and why
- Feature importance insights
  - Lag features dominate
  - Removed features (DEC-014)

**4. Live App Demo (3-4 min)**
- Show Streamlit app repo briefly
- Open deployed app (public URL)
- Demo workflow:
  - Select store
  - Select date
  - Generate single-day forecast
  - Generate N-day forecast
  - Show plot
  - Download CSV
- Explain what demand planners would see

**5. Conclusions (2 min)**
- Business impact: 13% improvement in forecast accuracy
- Technical achievements:
  - Full pipeline (4.8M rows with GPU)
  - Deployed web application
  - Reproducible methodology
- Lessons learned
- Future improvements (more stores, real-time data)

#### Part 2: Recording Setup (30 min)

**Tools:**
- Screen recording: OBS Studio / Loom / Zoom
- Resolution: 1920x1080
- Audio: Clear microphone, quiet environment

**Preparation:**
- Close unnecessary applications
- Open all tabs/windows needed:
  - GitHub (both repos)
  - Deployed Streamlit app
  - MLflow UI
  - Key figures
- Test audio levels
- Do one practice run

#### Part 3: Recording (1.5 hours)

**Recording Tips:**
- Speak clearly and at moderate pace
- Pause briefly between sections
- If mistake, pause and restart that section
- Keep energy and engagement

**Takes:**
- Aim for 1-2 complete takes
- Don't over-edit - authentic is better than polished

#### Part 4: Post-Production (1 hour)

**Editing (minimal):**
- Trim start/end
- Cut obvious mistakes
- Add title card (optional)
- Export as MP4

**Upload:**
- YouTube (unlisted) or
- Google Drive (anyone with link)
- Get shareable URL

**Deliverables:**
- [x] Video recorded (10-15 min)
- [x] Video edited and exported
- [x] Video uploaded with shareable link
- [x] Link added to README

---

### Day 5: Summary Document & Final Review
**Goal:** Create summary document, final checks, submit

**Total Time:** 4 hours (3.2h core + 0.8h buffer)

#### Part 1: Create SUMMARY.md (1.5 hours)

**Location:** Main repo root

**Content (2-3 pages):**

```markdown
# Corporación Favorita Demand Forecasting - Project Summary

## Executive Summary
Built end-to-end demand forecasting solution for Corporación Favorita 
grocery stores in Guayas, Ecuador. Achieved 13% improvement over 
baseline using LSTM neural network with temporal consistency strategy.

## Project Links
- **Analysis Repository:** [GitHub Link]
- **Streamlit App Repository:** [GitHub Link]
- **Live Demo:** [Streamlit Cloud URL]
- **Video Walkthrough:** [YouTube/Drive Link]

## Methodology
- **Week 1:** Exploratory Data Analysis (EDA)
- **Week 2:** Feature Engineering (33 features)
- **Week 3:** Model Development (XGBoost vs LSTM)
- **Week 4:** Production Pipeline + Deployment

## Key Results

| Model | Dataset | RMSE | MAE | Training Time |
|-------|---------|------|-----|---------------|
| XGBoost | 300K sample | 6.4860 | [TBD] | ~30 sec |
| LSTM | 300K sample | 6.2552 | ~3.05 | 36 sec (CPU) |
| XGBoost | 4.8M full | [TBD] | [TBD] | [TBD] |
| LSTM | 4.8M full | [TBD] | [TBD] | [TBD] (GPU) |

## Key Decisions

| ID | Decision | Impact |
|----|----------|--------|
| DEC-013 | 7-day train/test gap | Prevents data leakage |
| DEC-014 | 33 features (reduced from 45) | +4.5% improvement |
| DEC-015 | Full 2013 training REJECTED | Seasonal mismatch |
| DEC-016 | Q4+Q1 temporal consistency | Better seasonal alignment |

## Key Findings

### 1. Temporal Consistency > Data Volume
Using 6 months of seasonally-aligned data (Q4 2013 + Q1 2014) 
outperformed 12 months of full 2013 data by significant margin.

### 2. LSTM on Tabular Data (Unexpected)
LSTM beat XGBoost by 4.5% on engineered features, contrary to 
typical expectation that tree models dominate tabular data.

### 3. Feature Reduction Improves Generalization
Removing 12 noisy features (rolling std, oil, promotion interactions) 
improved RMSE by 4.5%.

## Lessons Learned
1. Hypothesis testing and willingness to reject (DEC-015)
2. Temporal relevance matters more than data volume
3. Always compare multiple model architectures
4. GPU acceleration enables production-scale experiments

## Future Work
- Expand to all Ecuador stores
- Real-time data integration
- Promotion impact modeling
- Ensemble methods

## Acknowledgments
Course: Time Series Forecasting
Data: Kaggle Corporación Favorita competition
```

#### Part 2: Final Repository Checks (1 hour)

**Main Repo Checklist:**
- [ ] All notebooks run without errors
- [ ] README complete with all links
- [ ] requirements.txt accurate
- [ ] .gitignore properly excludes large files
- [ ] SUMMARY.md added
- [ ] Figures accessible
- [ ] Decision logs complete

**Streamlit Repo Checklist:**
- [ ] App deploys without errors
- [ ] README has deployment URL
- [ ] requirements.txt complete
- [ ] .gitignore excludes mlflow_results/
- [ ] Screenshots included

**Cross-Links Verified:**
- [ ] Main repo links to Streamlit app
- [ ] Main repo links to video
- [ ] Streamlit repo links to main repo
- [ ] Video description has repo links

#### Part 3: Tag Releases (30 min)

**Main Repo:**
```bash
git tag -a v1.0-final -m "Week 4 complete: Production pipeline + deployment"
git push origin v1.0-final
```

**Streamlit Repo:**
```bash
git tag -a v1.0-deployed -m "Production deployment to Streamlit Cloud"
git push origin v1.0-deployed
```

#### Part 4: Final Submission Preparation (1 hour)

**Submission Package:**
1. GitHub link: `https://github.com/[user]/retail_demand_analysis`
2. Streamlit repo: `https://github.com/[user]/corporacion_favorita_app`
3. Live app: `https://[app-url].streamlit.app`
4. Video: `https://[video-url]`

**Final Verification:**
- [ ] Open all links in incognito browser
- [ ] Verify app loads for anonymous user
- [ ] Verify video plays
- [ ] Verify README renders correctly

**Deliverables:**
- [x] SUMMARY.md complete
- [x] All repos tagged
- [x] All links working
- [x] Submission ready

---

## 5. Success Criteria

### Quantitative

| Criteria | Target | Status |
|----------|--------|--------|
| FULL_02 complete | Both models trained | [ ] |
| MLflow runs | 2 runs logged | [ ] |
| Visualizations | 5-6 figures | [ ] |
| GitHub repos | 2 repos, properly structured | [ ] |
| Streamlit app | Deployed, public URL | [ ] |
| Video length | 10-15 minutes | [ ] |
| Summary document | 2-3 pages | [ ] |

### Qualitative

| Criteria | Target | Status |
|----------|--------|--------|
| Video clarity | Non-technical audience understands | [ ] |
| App usability | Forecast in <3 clicks | [ ] |
| README quality | Enables reproduction | [ ] |
| Portfolio readiness | Professional presentation | [ ] |

### Technical

| Criteria | Target | Status |
|----------|--------|--------|
| All code runs | No errors | [ ] |
| Deployment stable | No timeout/crashes | [ ] |
| Links work | All accessible | [ ] |
| Reproducible | Can clone and run | [ ] |

---

## 6. Communication Plan

### Daily Checkpoints

| Day | Checkpoint Focus |
|-----|------------------|
| 1 | FULL_02 metrics, winner determined, figures ready |
| 2 | Both repos pushed, app works locally |
| 3 | App deployed, URL working |
| 4 | Video recorded and uploaded |
| 5 | Summary complete, submission ready |

### End-of-Week Deliverables Review

**Before Submission:**
- All links verified
- Video reviewed for quality
- App tested by fresh user (if possible)
- README proofread

---

## 7. Risk Management

### Identified Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| FULL_02 training slow | Medium | Medium | Use GPU, reduce batch size if needed |
| Streamlit deployment fails | Medium | High | Test locally first, check requirements.txt |
| Model too large for repo | Low | Medium | Host on Drive, update config |
| Video recording issues | Low | Medium | Practice run, backup recording method |
| App timeout in cloud | Medium | Medium | Optimize data loading, cache results |

### Contingency Plans

**If FULL_02 takes too long (Day 1):**
- Use Week 3 LSTM model as production model
- Document scale comparison as future work

**If deployment fails (Day 3):**
- Demo locally in video
- Document deployment steps for future

**If video issues (Day 4):**
- Record in segments
- Simpler edit, focus on content

---

## 8. Two Repository Summary

| Aspect | retail_demand_analysis | corporacion_favorita_app |
|--------|------------------------|--------------------------|
| Purpose | Full project analysis | Deployment app |
| Content | Notebooks, data, docs | Streamlit app code |
| Audience | Technical reviewers | End users, demo |
| Size | Large (notebooks, figures) | Small (app code, model) |
| Deploy | GitHub only | GitHub + Streamlit Cloud |

---

## 9. Streamlit App Specifications

### User Interface

**Sidebar:**
- Store selector (dropdown, 10 stores)
- Product family selector
- Item selector (optional)

**Main Panel:**
- Date picker (Jan-Mar 2014 range)
- Forecast mode toggle (Single day / N-day)
- N-day slider (1-30) when multi-day selected
- "Generate Forecast" button
- Results: Plot + Table + Download

### Technical Requirements

**Python Packages:**
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tensorflow>=2.15.0  # or xgboost>=2.0.0
matplotlib>=3.7.0
plotly>=5.18.0
```

**Performance Targets:**
- Page load: <5 seconds
- Forecast generation: <10 seconds
- Memory: <1GB

---

## 10. Video Specifications

### Content Breakdown

| Section | Duration | Content |
|---------|----------|---------|
| Introduction | 1-2 min | Problem, objective, business value |
| Repository tour | 3 min | Structure, notebooks, MLflow |
| Key findings | 3 min | DEC-016, model comparison, insights |
| Live app demo | 3-4 min | Full workflow demonstration |
| Conclusions | 2 min | Impact, lessons, future work |
| **Total** | **12-15 min** | |

### Technical Specs

- Resolution: 1920x1080 (1080p)
- Format: MP4
- Audio: Clear narration
- Hosting: YouTube (unlisted) or Google Drive

---

## 11. Week 4 Deliverables Checklist

### Day 1
- [ ] FULL_02_train_final_model.ipynb complete
- [ ] XGBoost and LSTM both trained on full data
- [ ] 4 metrics calculated for both models
- [ ] MLflow runs logged
- [ ] 5-6 publication-ready figures
- [ ] Production artifacts exported
- [ ] FULL_02_checkpoint.md created

### Day 2
- [ ] Main repo cleaned and organized
- [ ] Main repo README complete
- [ ] Main repo requirements.txt
- [ ] Streamlit repo created
- [ ] App structure implemented (app/, model/, data/)
- [ ] App working locally
- [ ] Both repos pushed to GitHub

### Day 3
- [ ] Deployment preparation complete
- [ ] App deployed to Streamlit Community Cloud
- [ ] Public URL working
- [ ] All features tested in cloud
- [ ] README updated with deployment URL
- [ ] Screenshots captured

### Day 4
- [ ] Video script prepared
- [ ] Recording setup tested
- [ ] Video recorded (10-15 min)
- [ ] Video edited (minimal)
- [ ] Video uploaded
- [ ] Link added to README

### Day 5
- [ ] SUMMARY.md created (2-3 pages)
- [ ] All repo checks passed
- [ ] Releases tagged (v1.0-final)
- [ ] All links verified
- [ ] Submission package ready

---

## 12. Portfolio Value

### Technical Skills Demonstrated

- **Data Engineering:** 4.8M row pipeline, feature engineering
- **Machine Learning:** XGBoost, LSTM, hyperparameter tuning
- **MLOps:** MLflow tracking, experiment comparison
- **Deployment:** Streamlit, cloud hosting
- **Time Series:** Temporal consistency, lag features, forecasting

### Soft Skills Demonstrated

- **Scientific Method:** Hypothesis testing (DEC-015 rejection)
- **Decision Making:** Documented rationale (DEC-013 to DEC-017)
- **Communication:** Video, README, summary document
- **Project Management:** Weekly plans, checkpoints, handoffs

### Interview Talking Points

1. "I discovered that temporal consistency matters more than data volume"
2. "LSTM unexpectedly beat XGBoost on tabular data - here's why..."
3. "I scaled from 300K to 4.8M rows using GPU acceleration"
4. "I deployed a production model accessible to non-technical users"

---

**Week 4 Plan Complete. Ready to Execute.**

---

**Document Version:** 1.0  
**Created:** 2025-11-24  
**Status:** Ready for Day 1 execution

---

**END OF WEEK 4 PROJECT PLAN**

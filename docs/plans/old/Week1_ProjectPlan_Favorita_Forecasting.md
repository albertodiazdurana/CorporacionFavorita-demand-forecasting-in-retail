# Corporación Favorita Grocery Sales Forecasting
## Week 1 Project Plan: Exploration & Understanding

**Prepared by:** Alberto Diaz Durana  
**Timeline:** Week 1 (5 working days, 20 hours total)  
**Phase:** Phase 1 - Exploration & Understanding  
**Next Phase:** Week 2 - Feature Development

---

## 1. Purpose

**Objective:**  
Establish project infrastructure, load and filter the Corporación Favorita dataset to Guayas region focus, and conduct comprehensive exploratory data analysis to understand temporal patterns, data quality issues, and key drivers of sales variability.

**Business Value:**  
- Validate data quality for reliable forecasting
- Identify seasonal patterns and external factors affecting demand
- Define analytical cohort (Guayas stores, top-3 families) for manageable scope
- Build foundation for feature engineering and modeling phases

**Resources:**
- Time allocation: 20 hours (4 hours/day × 5 days)
- Time buffer: 20% included (16 hours core work + 4 hours buffer)
- Tools: Python, pandas, Dask, Jupyter, Git
- Computational: Local machine, RAM considerations for large dataset

---

## 2. Inputs & Dependencies

### Primary Dataset
- **Source**: Kaggle - Corporación Favorita Grocery Sales Forecasting
- **Files**:
  - `train.csv` (~479 MB, millions of rows - requires Dask)
  - `stores.csv` (54 stores metadata)
  - `items.csv` (4,100 products metadata)
  - `oil.csv` (1,218 daily oil prices)
  - `holidays_events.csv` (350 holiday records)
  - `transactions.csv` (83,488 transaction counts)

### Data Characteristics
- **Temporal range**: 2013-01-01 to 2017-08-15
- **Target variable**: unit_sales (continuous, can be negative for returns)
- **Spatial scope**: Ecuador (multiple states), filter to **Guayas**
- **Product scope**: 33 families, filter to **top-3 by item count**
- **Sample size**: 300,000 rows after filtering for speed

### Dependencies
- None (starting from raw data)
- Kaggle API credentials for download

### Data Quality Assumptions
- Missing dates in train.csv represent zero sales (not missing data)
- Negative sales are product returns (clip to zero)
- ~16% onpromotion values are NaN (assume False)
- Oil price gaps may exist (forward fill)

---

## 3. Execution Timeline

| Day | Focus Area | Core Hours | Buffer | Total | Key Deliverables |
|-----|-----------|------------|--------|-------|------------------|
| 1 | System Setup & Planning | 3.2h | 0.8h | 4h | Environment, repo structure, Week 1 plan |
| 2 | Data Download & Filtering | 3.2h | 0.8h | 4h | Guayas dataset, top-3 families, 300K sample |
| 3 | EDA Part 1: Quality & Preprocessing | 3.2h | 0.8h | 4h | Missing data handled, outliers detected, calendar filled |
| 4 | EDA Part 2: Feature Engineering & Visualization | 3.2h | 0.8h | 4h | Date features, rolling averages, time series plots |
| 5 | EDA Part 3: Context Analysis & Consolidation | 3.2h | 0.8h | 4h | Holiday/perishable/oil analysis, guayas_prepared.csv |
| **Total** | | **16h** | **4h** | **20h** | **6 outputs + decision log** |

---

## 4. Detailed Deliverables

### Day 1 - System Setup & Planning
**Goal:** Establish reproducible project environment and planning foundation

**Total Time:** 4 hours (3.2h core + 0.8h buffer)

#### Part 0: Environment Configuration (45 min)
**Objective:** Set up development environment and validate tools
**Activities:**
- Install Python libraries (pandas, dask, matplotlib, seaborn, statsmodels)
- Create Jupyter kernel
- Test Dask installation with sample data
- Validate Kaggle API credentials
**Deliverables:**
- Working Jupyter environment
- `requirements.txt` file
- Environment validation report

#### Part 1: Repository Structure (30 min)
**Objective:** Create organized project directory following methodology standards
**Activities:**
- Initialize Git repository
- Create folder structure (notebooks, data/raw, data/processed, data/results, outputs/figures, docs)
- Add .gitignore (exclude data files, checkpoints)
- Create README.md with project overview
**Deliverables:**
- Complete directory tree
- Initial README
- Git repository initialized

#### Part 2: Project Planning (1.5 hours)
**Objective:** Document Week 1 plan and establish decision logging framework
**Activities:**
- Review PM Guidelines structure
- Create Week 1 project plan (this document)
- Initialize decision log (docs/decisions/decision_log.md)
- Create data lineage template
**Deliverables:**
- `Week1_ProjectPlan.md`
- `decision_log.md` template
- `data_lineage.md` template

#### Part 3: Kaggle Setup & Data Inventory (1.25 hours)
**Objective:** Download datasets and create data inventory
**Activities:**
- Configure Kaggle API
- Download competition files
- Extract archives to data/raw/
- Create data inventory (file sizes, row counts, schemas)
- Document in `docs/data_inventory.md`
**Deliverables:**
- All 6 CSV files in data/raw/
- Data inventory report
- Initial schema documentation

---

### Day 2 - Data Download & Filtering
**Goal:** Load large dataset efficiently and filter to Guayas analytical cohort

**Total Time:** 4 hours (3.2h core + 0.8h buffer)

#### Part 0: Load Support Files (30 min)
**Objective:** Load and validate all small CSV files
**Activities:**
- Load stores.csv, items.csv, oil.csv, holidays_events.csv, transactions.csv into pandas
- Validate schemas match documentation
- Check for missing values per file
- Convert date columns to datetime
**Deliverables:**
- 5 DataFrames loaded and validated
- Data type validation report
- Support files summary statistics

#### Part 1: Guayas Store Identification (20 min)
**Objective:** Filter stores.csv to Guayas region
**Activities:**
- Query stores.csv WHERE state = 'Guayas'
- Count stores in Guayas
- Document store types and clusters in Guayas
**Deliverables:**
- List of Guayas store_nbr values
- Guayas store profile (types, clusters)

#### Part 2: Load train.csv with Dask (1 hour)
**Objective:** Stream large train.csv and filter to Guayas stores
**Activities:**
- Use Dask to read train.csv in chunks
- Filter by store_nbr IN (Guayas stores)
- Convert to pandas once filtered
- Validate row count reduction
**Deliverables:**
- Filtered train DataFrame (Guayas only)
- Memory usage comparison (before/after)
- Load time metrics

#### Part 3: Top-3 Families Selection (45 min)
**Objective:** Identify and filter to top-3 product families by item count
**Activities:**
- Count unique items per family in items.csv
- Select top-3 families
- Filter items.csv to top-3 families
- Filter train DataFrame to items in top-3 families
**Deliverables:**
- Top-3 families list with item counts
- Filtered items DataFrame
- Filtered train DataFrame (Guayas + top-3 families)

#### Part 4: Random Sampling & Export (45 min)
**Objective:** Sample 300K rows for speed and export checkpoint
**Activities:**
- Random sample 300,000 rows from filtered train
- Set random seed for reproducibility
- Merge with stores, items metadata
- Export to data/processed/guayas_sample_300k.csv
- Export to pickle for faster reload
**Deliverables:**
- `guayas_sample_300k.csv` (300K rows)
- `guayas_sample_300k.pkl` (faster loading)
- Sampling metadata (seed, sample_fraction)

---

### Day 3 - EDA Part 1: Quality & Preprocessing
**Goal:** Handle missing data, detect outliers, fill calendar gaps

**Total Time:** 4 hours (3.2h core + 0.8h buffer)

#### Part 0: Data Loading & Initial Checks (30 min)
**Objective:** Reload filtered data and perform initial quality checks
**Activities:**
- Load guayas_sample_300k.pkl
- Check shape, dtypes, memory usage
- Display first/last rows
- Summary statistics (describe)
**Deliverables:**
- Loaded DataFrame
- Initial quality report
- Baseline statistics

#### Part 1: Missing Value Analysis (45 min)
**Objective:** Detect and handle missing values across all columns
**Activities:**
- Count NaN per column
- Visualize missing patterns (heatmap)
- Fill onpromotion NaN with False
- Document handling decisions in decision log
**Deliverables:**
- Missing value report
- Cleaned onpromotion column
- Decision log entry (DEC-001)

#### Part 2: Outlier Detection (1 hour)
**Objective:** Identify and handle outliers in unit_sales
**Activities:**
- Detect negative sales (returns)
- Clip negative values to 0
- Calculate Z-scores by store-item groups
- Flag extreme outliers (Z > 3.0)
- Visualize outlier distribution
- Document outlier handling decision
**Deliverables:**
- Outlier analysis report
- Cleaned unit_sales (negatives → 0)
- Outlier flags (for potential removal later)
- Decision log entry (DEC-002)

#### Part 3: Calendar Gap Filling (1.75 hours)
**Objective:** Create complete daily index for each store-item pair
**Activities:**
- Convert date to datetime
- Write fill_calendar function (asfreq daily, fill_value=0)
- Apply to each (store_nbr, item_nbr) group
- Validate no missing dates remain
- Compare row count before/after
**Deliverables:**
- Complete daily calendar DataFrame
- Gap filling validation report
- Before/after row count comparison
- Updated DataFrame with zero-filled gaps

---

### Day 4 - EDA Part 2: Feature Engineering & Visualization
**Goal:** Engineer temporal features and visualize sales patterns

**Total Time:** 4 hours (3.2h core + 0.8h buffer)

#### Part 0: Date Feature Extraction (30 min)
**Objective:** Extract year, month, day, day_of_week from date
**Activities:**
- Create year, month, day columns
- Create day_of_week (0=Monday, 6=Sunday)
- Validate feature distributions
- Summary statistics per feature
**Deliverables:**
- 4 new date-based features
- Feature distribution plots
- Feature summary table

#### Part 1: Rolling Averages (1 hour)
**Objective:** Calculate 7-day moving average for smoothing
**Activities:**
- Sort by (item_nbr, store_nbr, date)
- Calculate 7-day rolling mean per group
- Handle edge cases (min_periods=1)
- Visualize raw vs smoothed for sample items
**Deliverables:**
- unit_sales_7d_avg column
- Smoothing visualization (3-5 example items)
- Explanation of rolling window behavior

#### Part 2: Sales Over Time Visualization (1.25 hours)
**Objective:** Visualize overall trends and seasonal patterns
**Activities:**
- Aggregate total sales by date
- Plot time series (2013-2017)
- Identify trend, seasonality, anomalies
- Create year-month heatmap
- Annotate major patterns
**Deliverables:**
- Total sales time series plot
- Year-month heatmap
- Pattern interpretation report
- Identified anomalies (e.g., August 2017 drop)

#### Part 3: Autocorrelation Analysis (1.25 hours)
**Objective:** Assess temporal dependence for lag feature guidance
**Activities:**
- Aggregate daily sales
- Plot autocorrelation (pandas.plotting.autocorrelation_plot)
- Interpret lag significance
- Document findings for Week 2 feature engineering
**Deliverables:**
- Autocorrelation plot
- Lag analysis interpretation
- Recommendations for lag features (Week 2)

---

### Day 5 - EDA Part 3: Context Analysis & Consolidation
**Goal:** Analyze holidays, perishables, oil prices; export final dataset

**Total Time:** 4 hours (3.2h core + 0.8h buffer)

#### Part 0: Holiday Impact Analysis (1 hour)
**Objective:** Assess how holidays affect sales
**Activities:**
- Merge holidays_events.csv with train on date
- Group by holiday type, calculate mean sales
- Plot bar chart (avg sales by holiday type)
- Identify significant impacts
**Deliverables:**
- Holiday analysis report
- Bar chart (sales by holiday type)
- Key findings (Work Day vs Holiday vs Transfer)

#### Part 1: Perishable Analysis (1 hour)
**Objective:** Compare perishable vs non-perishable sales patterns
**Activities:**
- Merge items.csv (perishable flag)
- Group by perishable, sum unit_sales
- Calculate perishable percentage
- Plot comparison bar chart
- Interpret business implications
**Deliverables:**
- Perishable vs non-perishable comparison
- Bar chart visualization
- Business interpretation (inventory risk)

#### Part 2: Oil Price Correlation (1 hour)
**Objective:** Investigate oil price impact on sales
**Activities:**
- Merge oil.csv with aggregated daily sales
- Plot dual-axis chart (oil price + sales over time)
- Calculate correlation coefficient
- Visual interpretation
- Document findings (likely weak correlation)
**Deliverables:**
- Oil price vs sales plot
- Correlation analysis
- Interpretation report
- Decision on oil feature inclusion (Week 2)

#### Part 3: Export & Documentation (1 hour)
**Objective:** Save cleaned dataset and summarize Week 1
**Activities:**
- Export final DataFrame to data/processed/guayas_prepared.csv
- Export to pickle for Week 2
- Update decision log with all Week 1 decisions
- Create Week 1 summary report
- Commit all notebooks and docs to Git
**Deliverables:**
- `guayas_prepared.csv` (cleaned, featured, calendar-filled)
- `guayas_prepared.pkl`
- Week 1 summary report
- Updated decision log
- Git commit with message "Week 1: EDA complete"

---

## 5. Readiness Checklist (for Week 2 Transition)

### Required Inputs
- [ ] Clean user dataset (guayas_prepared.csv, ~300K rows)
- [ ] Validated cohort definition (Guayas stores, top-3 families)
- [ ] Data quality report showing <5% missing values post-cleaning

### Completion Criteria
- [ ] All missing values handled (onpromotion filled, negatives clipped)
- [ ] Calendar gaps filled (complete daily index per store-item)
- [ ] Initial features created (date components, 7-day rolling avg)
- [ ] Temporal patterns documented (trend, seasonality, autocorrelation)
- [ ] External factors analyzed (holidays, perishables, oil)

### Quality Checks
- [ ] Data types validated (datetime for date, bool for onpromotion, int/float for numeric)
- [ ] Range checks passed (unit_sales ≥ 0)
- [ ] No duplicate (store_nbr, item_nbr, date) rows
- [ ] Row count matches expected (complete calendar × store-item combinations)

### Deliverables Ready
- [ ] All 5 days of notebooks run without errors
- [ ] `guayas_prepared.csv` validated and exported
- [ ] Decision log populated (≥3 entries)
- [ ] Week 1 summary report complete

### Next Phase Readiness
After completing Week 1, you will have:
- Clean, gap-filled, featured dataset ready for advanced feature engineering
- Understanding of temporal patterns (seasonality, trend, autocorrelation)
- Documented data quality decisions and outlier handling
- Identified external factors (holidays, perishables) for Week 2 feature creation
- Baseline visualizations for comparison in later phases

---

## 6. Success Criteria

### Quantitative
- Dataset reduced from millions to ~300K rows (manageable sample)
- Missing values <2% in critical columns post-cleaning
- Calendar completeness: 100% daily coverage per store-item
- Outliers flagged: document count and % of total
- ≥3 major decisions logged with rationale
- 5 notebooks created (~400 lines each, 5-6 sections per notebook)
- ≥8 visualizations (time series, heatmap, bar charts, autocorrelation)

### Qualitative
- Data quality issues understood and documented
- Temporal patterns clearly identified (trend, seasonality, anomalies)
- Holiday and perishable impacts interpretable by business stakeholders
- Decision rationale clear for future reference
- Notebooks readable by technical reviewer

### Technical
- All notebooks run end-to-end without errors
- Consistent file paths (relative, constants defined)
- Clean code (no debug/commented code in final version)
- Git repository organized (clear commit messages)
- Reproducibility: someone else can run notebooks and get same results

---

## 7. Documentation & Ownership

### Version Control
- **Repository**: retail_demand_analysis (GitHub)
- **Branch**: main
- **Commit frequency**: Daily (end of each day's work)
- **Commit message format**: "Day X: [accomplishment summary]"

### Key Documents
- `Week1_ProjectPlan.md` (this document)
- `decision_log.md` (track analytical choices)
- `data_inventory.md` (dataset characteristics)
- `data_lineage.md` (data transformations tracking)
- `README.md` (project overview, setup instructions)

### Assumptions
- 300K sample is representative of full Guayas dataset
- Top-3 families capture sufficient product diversity
- Zero-filling missing dates is appropriate (no stockouts, true zero demand)
- Clipping negative sales to zero is acceptable for forecasting (returns handled)
- Oil price correlation assumed weak (verify during EDA)

### Limitations
- Sample size may miss rare events (low-frequency items)
- Guayas focus excludes other regions (cannot generalize nationally)
- Top-3 families may not represent all product dynamics
- Historical data only (no forward-looking external signals)

### Ownership
- **Project Lead**: Alberto Diaz Durana
- **Stakeholders**: Academic advisor, peer reviewers
- **Timeline**: Week 1 of 4-week project

---

## 8. Risk Management

### Identified Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Dask loading fails (RAM limit) | Medium | High | Use chunking, smaller sample if needed |
| Missing dates excessive (>20% gaps) | Low | Medium | Document pattern, adjust fill strategy |
| Outliers dominate (>25% flagged) | Low | Medium | Review z-score threshold, consult literature |
| Oil correlation unexpectedly strong | Low | Low | Include in Week 2 features despite assumption |
| Time overrun (>20 hours) | Medium | Medium | Use buffer, deprioritize oil analysis if needed |

---

## 9. Expected Outcomes

### Expected Outcomes Table

| Metric | Before Week 1 | After Week 1 | Improvement | Target Met |
|--------|---------------|--------------|-------------|------------|
| Dataset size | Millions of rows | 300K rows | Filtered, manageable | Yes |
| Missing values | Unknown | <2% in critical columns | Production-ready | Yes |
| Calendar completeness | Gaps present | 100% daily coverage | Complete | Yes |
| Features | Raw columns only | +5 engineered features | Ready for Week 2 | Yes |
| Visualizations | None | 8 plots | Stakeholder-ready | Yes |
| Documentation | None | Decision log + lineage | Reproducible | Yes |
| **Summary** | **Raw data** | **Analysis-ready dataset** | **Foundation established** | **6/6 targets** |

### Key Benefits
- Validated dataset ready for advanced feature engineering
- Clear understanding of temporal dynamics (guides model selection)
- Documented decisions (avoid rework, enable peer review)
- Professional repository structure (portfolio-ready)

---

## 10. Communication Plan

### Daily Progress Updates
- **Frequency**: End of each day
- **Format**: Brief summary (5 minutes to write)
- **Content**: Accomplishments, blockers, next day plan
- **Audience**: Self (for tracking), advisor (if requested)

### Week-End Summary
- **Timing**: Friday end-of-day
- **Format**: Email or document (~1 page)
- **Content**: Week 1 accomplishments, key findings, Week 2 preview
- **Audience**: Advisor, peer reviewers

---

## Day 1 Summary Template

### Time Allocation (4 hours total):
| Task | Duration | Percentage |
|------|----------|------------|
| Part 0: Environment Configuration | 45 min | 18.75% |
| Part 1: Repository Structure | 30 min | 12.5% |
| Part 2: Project Planning | 1.5 hours | 37.5% |
| Part 3: Kaggle Setup & Data Inventory | 1.25 hours | 31.25% |
| **Total** | **4 hours** | **100%** |

### Key Achievements:
- (To be filled at end of Day 1)

### Outputs Created:
- (To be filled at end of Day 1)

### Issues Encountered:
- (To be filled at end of Day 1)

### Ready for Next Phase:
- (To be filled at end of Day 1)

---

## Summary

Week 1 establishes the foundation for reliable time series forecasting by:
1. Setting up reproducible project infrastructure
2. Filtering massive dataset to manageable, relevant scope (Guayas, top-3 families)
3. Conducting rigorous 8-step EDA (quality, preprocessing, features, patterns, context)
4. Documenting decisions and data transformations for transparency

Upon completion, Week 2 (Feature Development) can begin with confidence in data quality and clear direction for lag features, rolling statistics, and external factor engineering based on Week 1 insights.

**Week 1 deliverables enable Week 2-4 success.**

---

**End of Week 1 Project Plan**

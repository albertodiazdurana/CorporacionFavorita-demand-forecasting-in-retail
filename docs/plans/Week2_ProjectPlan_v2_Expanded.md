# Corporación Favorita Grocery Sales Forecasting
## Week 2 Project Plan: Feature Development (Version 2.0 - Expanded)

**Prepared by:** Alberto Diaz Durana  
**Timeline:** Week 2 (5 working days, 20 hours total)  
**Phase:** Phase 2 - Feature Development  
**Previous Phase:** Week 1 - Exploration & Understanding (COMPLETE, 15h actual)  
**Next Phase:** Week 3 - Analysis & Modeling  
**Plan Version:** 2.0 (Enhanced with daily checkpoints and progressive tracking)

---

## 1. Purpose

**Objective:**  
Engineer 20-30 advanced time series features based on Week 1 autocorrelation analysis, temporal patterns, and external factor insights to create a modeling-ready dataset for Week 3 forecasting.

**Business Value:**  
- Capture temporal dependencies (lag 1/7/14/30) identified in Week 1 autocorrelation analysis (r = 0.60+)
- Encode smoothing trends via rolling statistics (7/14/30-day windows)
- Incorporate macro indicators (oil price) with moderate correlation (-0.55)
- Create store/item baseline aggregations for heterogeneous performance (4.25x gap)
- Enable advanced forecasting models with rich feature set

**Resources:**
- Time allocation: 20 hours (4 hours/day × 5 days)
- Time buffer: 20% included (16 hours core work + 4 hours buffer)
- Computing: ~30-40 minutes total for all feature engineering on 300K rows
- Carryover buffer: +8.5 hours from Week 1 efficiency
- **Total buffer available: 12.5 hours** (Week 2 allocation + Week 1 surplus)

**Week 1 Foundation:**
- Dataset: `guayas_prepared.pkl` (300,896 rows × 28 columns)
- Key findings: Weekend +34%, Promotion +74%, Autocorrelation 0.60+, Pareto 34/80
- 10 decisions logged, 13 visualizations created
- Quality: 0% missing in critical features, outliers flagged
- Time performance: 15h actual / 23.5h allocated = 64% efficiency

---

## 2. Inputs & Dependencies

### Primary Input
- **Source**: Week 1 final output
- **File**: `data/processed/guayas_prepared.pkl`
- **Characteristics**:
  - Shape: 300,896 rows × 28 columns
  - Date range: 2013-01-02 to 2017-08-15 (1,680 days)
  - Stores: 11 Guayas stores (#24-51)
  - Items: Top-3 families (GROCERY I, BEVERAGES, CLEANING)
  - Target: unit_sales (continuous, ≥0 after clipping)

### Secondary Data Sources
- **Oil prices**: `data/raw/oil.csv` (1,218 daily WTI prices)
  - Date range: 2013-01-01 to 2017-08-31
  - Missing dates: weekends, holidays (requires forward-fill)
  - Correlation with sales: -0.55 (moderate)

### Existing Features (Week 1 - 28 columns)
**Temporal features (5):**
- date, year, month, day, day_of_week, is_weekend

**Store features (4):**
- store_nbr, city, state, type, cluster

**Item features (3):**
- item_nbr, family, class

**Target & promotion (2):**
- unit_sales, onpromotion

**Holiday features (14):**
- is_holiday, holiday_type, is_transferred, holiday_name, locale, locale_name
- Additional holiday context columns (9 total)

### Dependencies
- Week 1 complete (✓)
- `guayas_prepared.pkl` validated and exists
- Week 1 insights documented in handoff document

### Critical Constraints
- **MUST sort by (store_nbr, item_nbr, date)** before any lag/rolling operations
- **NO data leakage**: Features must use only past information
- **NO calendar gap filling**: Maintain sparse format (300K rows, not 33M)
- **Keep NaN in lags**: Models like XGBoost handle natively

---

## 3. Execution Timeline

| Day       | Focus Area                           | Core Hours | Buffer | Total   | Key Deliverables                                   | Actual  | Variance |
| --------- | ------------------------------------ | ---------- | ------ | ------- | -------------------------------------------------- | ------- | -------- |
| 1         | Lag Features (1/7/14/30)             | 3.2h       | 0.8h   | 4h      | 4 lag features, temporal validation                | TBD     | TBD      |
| 2         | Rolling Statistics (7/14/30 avg/std) | 3.2h       | 0.8h   | 4h      | 6 rolling features, smoothing visualizations       | TBD     | TBD      |
| 3         | Oil Price Features + Lags            | 3.2h       | 0.8h   | 4h      | 4-5 oil features, correlation analysis             | TBD     | TBD      |
| 4         | Store/Item Aggregations              | 3.2h       | 0.8h   | 4h      | 6-8 aggregation features, baseline metrics         | TBD     | TBD      |
| 5         | Promotion/Payday + Final Export      | 3.2h       | 0.8h   | 4h      | Final features, guayas_features.pkl, dictionary v2 | TBD     | TBD      |
| **Total** |                                      | **16h**    | **4h** | **20h** | **20-30 engineered features**                      | **TBD** | **TBD**  |

### Cumulative Buffer Tracking

| Checkpoint      | Buffer Allocated | Buffer Used | Buffer Remaining | Notes                  |
| --------------- | ---------------- | ----------- | ---------------- | ---------------------- |
| Week 1 Complete | 23.5h allocated  | 15h actual  | +8.5h surplus    | 64% efficiency         |
| Week 2 Start    | +4h (20%)        | 0h          | **12.5h total**  | Strong buffer position |
| End of Day 1    | 0.8h             | TBD         | TBD              | Update after Day 1     |
| End of Day 2    | 1.6h cumulative  | TBD         | TBD              | Update after Day 2     |
| End of Day 3    | 2.4h cumulative  | TBD         | TBD              | Update after Day 3     |
| End of Day 4    | 3.2h cumulative  | TBD         | TBD              | Update after Day 4     |
| End of Day 5    | 4h cumulative    | TBD         | TBD              | Final buffer status    |

**Buffer Management Strategy:**
- If Day 1-2 under budget → Proceed with all SHOULD + COULD features
- If Day 1-2 on budget → Complete all SHOULD, selective COULD features
- If Day 1-2 over budget → Focus on MUST + SHOULD only, skip COULD
- Critical threshold: If buffer drops below 8h after Day 3, trigger contingency plan

---

## 4. Detailed Deliverables

### Day 1 - Lag Features
**Goal:** Create 1, 7, 14, and 30-day lag features based on Week 1 autocorrelation findings

**Total Time:** 4 hours (3.2h core + 0.8h buffer)

#### Part 0: Load & Temporal Sort (30 min)
**Objective:** Load Week 1 dataset and enforce temporal ordering
**Activities:**
- Load `guayas_prepared.pkl`
- Verify shape (300,896 × 28)
- **CRITICAL:** Sort by (store_nbr, item_nbr, date)
- Reset index
- Document pre-conditions met
**Deliverables:**
- Sorted dataframe ready for lag operations
- Verification output (shape, date range, sort confirmation)

#### Part 1: Create Basic Lag Features (1.5 hours)
**Objective:** Generate lag 1, 7, 14, 30 using groupby + shift
**Activities:**
- Create `unit_sales_lag1` (yesterday's sales, r=0.602)
- Create `unit_sales_lag7` (last week, r=0.585)
- Create `unit_sales_lag14` (two weeks ago, r=0.625 - HIGHEST)
- Create `unit_sales_lag30` (last month, r=0.360)
- Pattern: `df.groupby(['store_nbr', 'item_nbr'])['unit_sales'].shift(k)`
**Deliverables:**
- 4 new lag columns
- NaN count per lag (expected: lag1=~2.7K, lag30=~82K first observations)

#### Part 2: Validation & Visualization (1 hour)
**Objective:** Verify lag features are correctly computed
**Activities:**
- Sample validation: Pick store-item pair, manually verify lag values
- Plot time series with lags for 3 sample items
- Check correlation matrix (lag features vs unit_sales)
- Verify NaN pattern matches expectations
**Deliverables:**
- 3 validation plots (unit_sales + lags overlay)
- Correlation heatmap
- Validation report (pass/fail)

#### Part 3: Save Checkpoint (30 min)
**Objective:** Export intermediate dataset
**Activities:**
- Save `data/processed/guayas_with_lags.pkl`
- Update feature dictionary: document 4 lag features
- Log decision on NaN handling strategy
**Deliverables:**
- `guayas_with_lags.pkl` (300,896 × 32 columns)
- Feature dictionary update (4 new entries)
- Decision log entry (DEC-011: Keep NaN lags for model compatibility)

#### **End-of-Day 1 Checkpoint** ⚠️

**Critical Review Questions:**
1. Are lag features correctly sorted (store_nbr, item_nbr, date)?
2. Did validation confirm lag calculations match manual checks?
3. Are NaN counts as expected (lag1 ~2.7K, lag30 ~82K)?
4. Should we adjust Day 2 rolling window sizes based on lag findings?
5. Did we stay within 4h budget (3.2h core + 0.8h buffer)?

**Adjustment Options:**
- If ahead of schedule → Add lag 60/90 features on Day 2
- If behind schedule → Reduce rolling windows from 3 to 2 (7/14 only)
- If validation failed → Debug before proceeding to Day 2

**Use Daily Checkpoint Template (Section 11) to document decisions.**

---

### Day 2 - Rolling Statistics
**Goal:** Create 7, 14, 30-day rolling averages and standard deviations

**Total Time:** 4 hours (3.2h core + 0.8h buffer)

#### Part 0: Load Checkpoint (15 min)
**Objective:** Load Day 1 output
**Activities:**
- Load `guayas_with_lags.pkl`
- Verify 32 columns present
- Check temporal sort maintained
**Deliverables:**
- Loaded dataframe with lag features

#### Part 1: Create Rolling Averages (1.5 hours)
**Objective:** Generate rolling mean features for trend capture
**Activities:**
- Create `unit_sales_7d_avg` (7-day moving average)
- Create `unit_sales_14d_avg` (14-day moving average)
- Create `unit_sales_30d_avg` (30-day moving average)
- Pattern: `df.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform(lambda x: x.rolling(window=k, min_periods=1).mean())`
- Use `min_periods=1` to reduce NaN (partial windows still compute)
**Deliverables:**
- 3 rolling average columns
- NaN count verification

#### Part 2: Create Rolling Standard Deviations (1 hour)
**Objective:** Capture recent volatility/stability
**Activities:**
- Create `unit_sales_7d_std` (7-day volatility)
- Create `unit_sales_14d_std` (14-day volatility)
- Create `unit_sales_30d_std` (30-day volatility)
- Same groupby + transform pattern with `.std()`
**Deliverables:**
- 3 rolling std columns
- Volatility metric statistics

#### Part 3: Smoothing Effect Validation (1 hour)
**Objective:** Visualize smoothing and noise reduction
**Activities:**
- Plot 5 sample items: raw sales vs 7/14/30-day averages
- Identify items with high/low volatility (std comparison)
- Verify smoothing reduces noise without excessive lag
**Deliverables:**
- 5 smoothing visualizations
- Volatility analysis table (top 10 volatile items)

#### Part 4: Save Checkpoint (30 min)
**Objective:** Export dataset with rolling features
**Activities:**
- Save `data/processed/guayas_with_rolling.pkl`
- Update feature dictionary: 6 rolling features
- Document computation time (~5-8 minutes)
**Deliverables:**
- `guayas_with_rolling.pkl` (300,896 × 38 columns)
- Feature dictionary update (6 new entries)

#### **End-of-Day 2 Checkpoint** ⚠️

**Critical Review Questions:**
1. Do rolling averages smooth noise as expected?
2. Are high-volatility items identifiable (retail sparsity 99.1%)?
3. Did min_periods=1 reduce NaN to <1% as planned?
4. Should we add rolling max/min features on Day 5 (optional)?
5. Are we on track for SHOULD features (oil, aggregations)?

**Cumulative Progress Check:**
- [ ] MUST features complete: 10/10 (lag 4 + rolling 6)
- [ ] Days 1-2 buffer status: Within/Over budget?
- [ ] Proceed with SHOULD features on Days 3-4: Yes/Adjust scope?

**Use Daily Checkpoint Template (Section 11) to document decisions.**

---

### Day 3 - Oil Price Features
**Goal:** Merge oil prices and create oil lag features

**Total Time:** 4 hours (3.2h core + 0.8h buffer)

#### Part 0: Load Datasets (20 min)
**Objective:** Load Day 2 output and oil data
**Activities:**
- Load `guayas_with_rolling.pkl`
- Load `data/raw/oil.csv`
- Convert oil date to datetime
- Check oil date range coverage
**Deliverables:**
- Both datasets loaded
- Date alignment check

#### Part 1: Merge Oil Prices (1 hour)
**Objective:** Add daily oil prices to main dataset
**Activities:**
- Left join on date (main ← oil)
- Forward-fill missing oil prices (weekends/holidays)
- Rename column: `dcoilwtico` → `oil_price`
- Verify no excessive NaN (should be ~0% after ffill)
**Deliverables:**
- `oil_price` column added
- Missing value report (expect ~0%)

#### Part 2: Create Oil Lag Features (1.5 hours)
**Objective:** Generate oil price lags (7, 14, 30 days)
**Activities:**
- Create `oil_price_lag7` (oil price 1 week ago)
- Create `oil_price_lag14` (oil price 2 weeks ago)
- Create `oil_price_lag30` (oil price 1 month ago)
- Use same groupby + shift pattern as unit_sales lags
- Calculate oil price change: `oil_price_change = oil_price - oil_price_lag7`
**Deliverables:**
- 4 oil features (oil_price + 3 lags)
- 1 oil change metric (derivative)
- Total: 5 oil-related columns

#### Part 3: Oil Correlation Analysis (45 min)
**Objective:** Validate oil feature utility
**Activities:**
- Correlation matrix: oil features vs unit_sales
- Scatter plots: oil_price vs unit_sales (by month)
- Test lag importance: which lag correlates best?
**Deliverables:**
- Correlation heatmap (oil features)
- 2 scatter plots
- Correlation report

#### Part 4: Save Checkpoint (45 min)
**Objective:** Export dataset with oil features
**Activities:**
- Save `data/processed/guayas_with_oil.pkl`
- Update feature dictionary: 5 oil features
- Log decision on oil feature inclusion
**Deliverables:**
- `guayas_with_oil.pkl` (300,896 × 43 columns)
- Feature dictionary update (5 new entries)
- Decision log entry (DEC-012: Include oil as macro indicator)

#### **End-of-Day 3 Checkpoint** ⚠️

**Critical Review Questions:**
1. Did forward-fill reduce oil NaN to ~0%?
2. Is oil correlation stronger than Week 1 finding (-0.55)?
3. Which oil lag (7/14/30) shows strongest predictive power?
4. Should we add oil price moving average on Day 5 (optional)?
5. Are we on track for Day 4 aggregations?

**Mid-Week Review:**
- [ ] MUST features: 10/10 complete ✓
- [ ] SHOULD features: 5/15 complete (oil done)
- [ ] Buffer status: Within 12.5h threshold?
- [ ] Proceed with full aggregation plan on Day 4: Yes/Simplify?

**Use Daily Checkpoint Template (Section 11) to document decisions.**

---

### Day 4 - Store/Item Aggregations
**Goal:** Create baseline performance aggregations per store, item, and cluster

**Total Time:** 4 hours (3.2h core + 0.8h buffer)

#### Part 0: Load Checkpoint (15 min)
**Objective:** Load Day 3 output
**Activities:**
- Load `guayas_with_oil.pkl`
- Verify 43 columns present
**Deliverables:**
- Loaded dataframe with oil features

#### Part 1: Store-Level Aggregations (1.5 hours)
**Objective:** Capture store baseline performance and cluster patterns
**Activities:**
- Calculate `store_avg_sales`: Mean unit_sales per store
- Calculate `store_total_sales`: Total unit_sales per store
- Calculate `cluster_avg_sales`: Mean unit_sales per cluster
- Merge back to main dataframe (many-to-one join)
- Rationale: Store #51 has 356K units vs Store #32 with 84K (4.25x gap)
**Deliverables:**
- 3 store/cluster aggregation columns
- Store performance ranking table

#### Part 2: Item-Level Aggregations (1.5 hours)
**Objective:** Capture item demand patterns and sparsity
**Activities:**
- Calculate `item_avg_sales`: Mean unit_sales per item
- Calculate `item_total_sales`: Total unit_sales per item
- Calculate `item_sell_frequency`: % of days item has non-zero sales
- Calculate `item_velocity_tier`: Classify items into Fast/Medium/Slow (Pareto insight: top 34% = 80% sales)
- Merge back to main dataframe
**Deliverables:**
- 4 item aggregation columns
- Item velocity distribution table

#### Part 3: Family-Level Aggregations (30 min)
**Objective:** Capture product category trends
**Activities:**
- Calculate `family_avg_sales`: Mean unit_sales per family
- Merge back to main dataframe
**Deliverables:**
- 1 family aggregation column

#### Part 4: Aggregation Validation & Save (30 min)
**Objective:** Verify aggregations and export
**Activities:**
- Spot-check: Manual calculation vs computed aggregation
- Verify no missing values introduced
- Save `data/processed/guayas_with_aggs.pkl`
- Update feature dictionary: 8 aggregation features
**Deliverables:**
- `guayas_with_aggs.pkl` (300,896 × 51 columns)
- Feature dictionary update (8 new entries)
- Aggregation validation report

#### **End-of-Day 4 Checkpoint** ⚠️

**Critical Review Questions:**
1. Do store aggregations show 4.25x gap (Store #51 vs #32)?
2. Does item_sell_frequency align with 99.1% sparsity finding?
3. Do velocity tiers match Pareto principle (34% = 80% sales)?
4. Should we add store-item interaction aggregation on Day 5?
5. Are we on track to complete SHOULD features?

**Day 5 Preparation:**
- [ ] MUST features: 10/10 complete ✓
- [ ] SHOULD features: 13/15 complete (missing promo history, payday)
- [ ] Remaining time: ~4h for 2 SHOULD + COULD features
- [ ] Buffer status: Sufficient for quality final export?

**Use Daily Checkpoint Template (Section 11) to document decisions.**

---

### Day 5 - Promotion/Payday + Final Export
**Goal:** Add remaining features and export final modeling-ready dataset

**Total Time:** 4 hours (3.2h core + 0.8h buffer)

#### Part 0: Load Checkpoint (15 min)
**Objective:** Load Day 4 output
**Activities:**
- Load `guayas_with_aggs.pkl`
- Verify 51 columns present
**Deliverables:**
- Loaded dataframe with aggregations

#### Part 1: Promotion History Features (1.5 hours)
**Objective:** Encode promotion timing patterns
**Activities:**
- Create `days_since_promo`: Days since last promotion per store-item
  - Method: Shift onpromotion==True, calculate cumulative count of days
- Create `promo_frequency_30d`: Promotion count in rolling 30-day window
  - Method: Rolling sum of onpromotion over 30 days
- Rationale: Promotions drive +74% sales lift (Week 1 insight)
**Deliverables:**
- 2 promotion features
- Promotion pattern analysis

#### Part 2: Payday Window Flag (30 min)
**Objective:** Capture payday effect (+11% lift)
**Activities:**
- Create `is_payday_window`: Binary flag for days 1-3 and 14-16 of month
- Rationale: Week 1 showed +11% lift on payday windows
**Deliverables:**
- 1 payday feature

#### Part 3: Optional Advanced Features (1 hour - if time permits)
**Objective:** Add remaining features from Week 1 priorities
**Activities:**
- **IF buffer permits, add in priority order:**
  1. `store_item_avg_sales`: Mean sales per store-item pair (finer granularity) - 15 min
  2. `days_since_last_sale`: Days since last non-zero sale (sparsity) - 15 min
  3. `promo_x_weekend`: Promotion × Weekend interaction - 15 min
  4. Test Week 1 finding: Promo × Holiday = -16% synergy (document, don't create feature)
- **IF time constrained, skip this part**
**Deliverables:**
- 0-3 optional features (depending on time)
- Interaction analysis notes

#### Part 4: Final Export & Documentation (1 hour)
**Objective:** Create modeling-ready dataset and complete documentation
**Activities:**
- Final feature count: 51 + 3 mandatory + 0-3 optional = **54-57 columns**
- Save `data/processed/guayas_features.pkl` (FINAL WEEK 2 OUTPUT)
- Export CSV version: `data/processed/guayas_features.csv` (for reference)
- Update feature dictionary v2 (complete documentation of all 54-57 features)
- Generate feature summary statistics (min/max/mean/std per feature)
- Create Week 2 checkpoint document
**Deliverables:**
- `guayas_features.pkl` (300,896 × 54-57 columns) - **PRIMARY OUTPUT**
- `guayas_features.csv` (backup/reference)
- `feature_dictionary_v2.txt` (54-57 features documented)
- `Day5_Checkpoint_Week2_Summary.md`
- Feature summary statistics table

#### **End-of-Day 5 Checkpoint** ⚠️

**Critical Review Questions:**
1. Are all MUST features complete and validated? (10/10)
2. Are all SHOULD features complete? (15/15)
3. Did we add any COULD features? (0-3)
4. Is final dataset exported and ready for Week 3?
5. What is final buffer status (used vs remaining)?

**Final Week 2 Assessment:**
- [ ] Total features created: 26-29 engineered features
- [ ] Final column count: 54-57 columns
- [ ] Missing value strategy documented
- [ ] All features validated with visualizations
- [ ] Feature dictionary v2 complete
- [ ] Week 2 → Week 3 handoff document created

**Use Daily Checkpoint Template (Section 11) to document final week status.**

---

## 5. Phase 2 Readiness Checklist (for Week 3 Transition)

### Required Inputs
- [✓] Prepared dataset (guayas_prepared.pkl, 300,896 rows × 28 columns)
- [✓] Week 1 insights (autocorrelation, patterns, external factors)
- [✓] Oil price data (oil.csv, 1,218 daily prices)

### Completion Criteria - Structured by Priority

**MUST Features (10 features - Non-negotiable):**
- [ ] Lag 1/7/14/30 (4 features) - Temporal memory
- [ ] Rolling avg 7/14/30 (3 features) - Trend smoothing
- [ ] Rolling std 7/14/30 (3 features) - Volatility capture

**SHOULD Features (15 features - Complete if on schedule):**
- [ ] Oil price + lags 7/14/30 + change (5 features) - Macro indicator
- [ ] Store/cluster aggregations (3 features) - Store baselines
- [ ] Item aggregations + velocity tier (5 features) - Item baselines
- [ ] Promotion history features (2 features) - Timing patterns

**COULD Features (0-3 features - Only if ahead of schedule):**
- [ ] Store-item aggregation (1 feature) - Fine-grained baseline
- [ ] Days since last sale (1 feature) - Sparsity metric
- [ ] Promotion × Weekend interaction (1 feature) - Interaction term

### Quality Checks
- [ ] Temporal sort verified (store_nbr, item_nbr, date ascending)
- [ ] No data leakage (all features use only past information)
- [ ] Lag features validated (spot-check manual calculation vs computed)
- [ ] Rolling features smooth noise (visualized on sample items)
- [ ] Oil prices merged correctly (date alignment, forward-fill applied)
- [ ] Aggregations accurate (spot-check against manual calculation)
- [ ] No duplicate rows (300,896 rows maintained)
- [ ] Data types appropriate (float64 for numeric, datetime64 for date)

### Deliverables Ready
- [ ] `w02_d01_FE_lags.ipynb` (runs without errors)
- [ ] `w02_d02_FE_rolling.ipynb` (runs without errors)
- [ ] `w02_d03_FE_oil.ipynb` (runs without errors)
- [ ] `w02_d04_FE_aggregations.ipynb` (runs without errors)
- [ ] `w02_d05_FE_final.ipynb` (runs without errors)
- [ ] `guayas_features.pkl` (final dataset, 300,896 × 54-57 columns)
- [ ] `feature_dictionary_v2.txt` (complete feature documentation)
- [ ] Week 2 summary report (Day5_Checkpoint_Week2_Summary.md)
- [ ] 5 daily checkpoint documents (Day1-5)

### Next Phase Readiness
After completing Week 2, you will have:
- Comprehensive feature set (54-57 columns) capturing temporal, external, and baseline patterns
- Validated features with no data leakage or temporal ordering errors
- Modeling-ready dataset for Week 3 (ARIMA, Prophet, LSTM)
- Feature importance baseline (preliminary correlation analysis)
- Documented feature engineering decisions (2-3 new decision log entries)
- Visualizations validating feature quality (10-15 plots)
- 5 daily checkpoint documents tracking progress and adjustments

---

## 6. Success Criteria

### Quantitative - Structured by Priority

**MUST Features (Critical for Week 3):**
- Lag features: 4 created (1/7/14/30), NaN as expected
- Rolling features: 6 created (avg/std × 3 windows), NaN <1%
- Computation time: <15 minutes for MUST features
- Validation: 100% pass rate on manual spot-checks

**SHOULD Features (Expected if on schedule):**
- Oil features: 5 created (price + 3 lags + change), NaN 0%
- Aggregations: 8 created (store/item/cluster baselines), NaN 0%
- Promotion features: 2 created (history + frequency)
- Payday feature: 1 created (binary flag)
- Total SHOULD: 15 features

**COULD Features (Bonus if ahead):**
- Optional features: 0-3 created (store-item agg, sparsity, interaction)

**Overall Targets:**
- Feature count: 54-57 columns (from 28 in Week 1) = **+26-29 engineered features**
- Missing values: 0% in SHOULD features, NaN in lags kept by design
- Total computation time: <40 minutes for all feature engineering
- Notebooks: 5 created (~300-400 lines each)
- Visualizations: 10-15 (validation plots, correlations, smoothing effects)
- Daily checkpoints: 5 documented (Days 1-5)

### Qualitative
- Feature engineering decisions clearly documented (2-3 new decision log entries)
- Lag features capture temporal dependencies identified in Week 1 autocorrelation
- Rolling features smooth noise without excessive lag
- Oil features validated as relevant macro indicators
- Aggregations provide meaningful baseline comparisons (store/item heterogeneity)
- Features interpretable by business stakeholders (e.g., "7-day average sales")
- No data leakage (rigorous validation)
- Daily checkpoints enable agile scope adjustment

### Technical
- All 5 notebooks run end-to-end without errors
- Temporal sort enforced before all lag/rolling operations
- Groupby operations correct (per store-item, not global)
- Feature names descriptive and consistent (snake_case)
- Code modular with clear section headers
- Git commits daily with descriptive messages
- Reproducibility: Independent execution yields identical results

---

## 7. Documentation & Ownership

### Version Control
- **Repository**: retail_demand_analysis (GitHub)
- **Branch**: main
- **Commit frequency**: Daily (end of each day's work)
- **Commit message format**: "Week 2 Day X: [accomplishment summary] [checkpoint status]"

### Key Documents
- `Week2_ProjectPlan_v2_Expanded.md` (this document)
- `decision_log.md` (update with 2-3 new feature engineering decisions)
- `feature_dictionary_v2.txt` (complete 54-57 feature documentation)
- `Day1_Checkpoint_Week2.md` through `Day5_Checkpoint_Week2.md` (daily reviews)
- `Day5_Checkpoint_Week2_Summary.md` (week-end comprehensive summary)
- `Week2_to_Week3_Handoff.md` (prepare for next phase)

### Assumptions
- Lag NaN handling: Keeping NaN for first observations is acceptable (XGBoost/LightGBM handle natively)
- Oil price relevance: -0.55 correlation justifies inclusion despite being moderate
- Forward-fill oil prices: Assumption that weekend oil prices = Friday's price is reasonable
- Store/item aggregations: Historical averages are stable enough to use as features
- Sparse format maintenance: No calendar gap filling (300K rows, not 33M)
- Daily checkpoints: 15 minutes per day sufficient for review and adjustment

### Limitations
- NaN in lag features may limit some models (e.g., linear regression requires imputation)
- Rolling features with `min_periods=1` may have low-quality values at start (partial windows)
- Oil price correlation (-0.55) is moderate, not strong; may not drive significant predictions
- Aggregations assume stationarity (historical avg = future baseline), which may not hold during regime changes
- Limited to 300K sample; feature patterns may differ in full 33M dataset
- Daily checkpoints add 1.25h overhead (5 days × 15 min), built into buffer allocation

### Ownership
- **Project Lead**: Alberto Diaz Durana
- **Stakeholders**: Academic advisor, peer reviewers
- **Timeline**: Week 2 of 4-week project
- **Version**: 2.0 (Enhanced with daily checkpoints and progressive tracking)

---

## 8. Risk Management

### Identified Risks

| Risk                                            | Likelihood | Impact     | Mitigation                                               |
| ----------------------------------------------- | ---------- | ---------- | -------------------------------------------------------- |
| Computation time exceeds estimate (>40 min)     | Low        | Low        | Use Dask if RAM issues, already validated on 300K        |
| NaN in lags breaks downstream models            | Low        | Medium     | Document NaN strategy, test with sample XGBoost          |
| Oil price merge introduces errors               | Low        | High       | Validate date alignment, spot-check merged values        |
| Feature importance analysis shows low utility   | Medium     | Low        | Proceed to Week 3, evaluate in model training            |
| Temporal sort accidentally skipped              | Low        | Critical   | Add assertion checks, validate with sample store-item    |
| Time overrun (>20 hours)                        | Low        | Medium     | Use 12.5h cumulative buffer, deprioritize COULD features |
| **Feature creep with COULD features**           | **Medium** | **Medium** | **Use daily checkpoints to maintain MUST/SHOULD focus**  |
| **Daily checkpoint overhead exceeds 15 min**    | **Low**    | **Low**    | **Template streamlines process, cap at 15 min**          |
| **Scope expansion due to interesting findings** | **Medium** | **Medium** | **Defer exploration to Week 3 modeling phase**           |

### Contingency Plans
- **If computation time >1 hour**: Skip COULD features, proceed with MUST+SHOULD only (51 columns)
- **If oil merge fails**: Exclude oil features, continue with lag/rolling/aggregations (46 columns)
- **If time constraint tight**: Focus on MUST features only (lag + rolling = 10 features), defer rest to Week 3 prep time
- **If feature creep detected in checkpoints**: Immediate scope reduction, document deferred features for future work
- **If buffer drops below 8h after Day 3**: Trigger contingency, skip all COULD features, simplify SHOULD features

---

## 9. Expected Outcomes

### Progressive Expected Outcomes Table

| Metric             | Before Week 2              | After Days 1-2 (MUST)      | After Week 2 (Days 3-5)    | Target Met      |
| ------------------ | -------------------------- | -------------------------- | -------------------------- | --------------- |
| Feature count      | 28 columns                 | 28 + 10 (lag + rolling)    | 54-57 columns              | Yes             |
| Lag features       | 0                          | 4 complete, validated      | 4 complete, validated      | Yes             |
| Rolling features   | 1 prototype (Week 1)       | 6 complete, validated      | 6 complete, validated      | Yes             |
| Oil features       | 0                          | 0 (planned Day 3)          | 5 complete                 | Yes             |
| Aggregations       | 0                          | 0 (planned Day 4)          | 8 complete                 | Yes             |
| Promotion features | Static flag                | 0 (planned Day 5)          | 2-3 timing features        | Yes             |
| External factors   | Holiday only               | Holiday only               | + Oil + Promo history      | Yes             |
| Missing values     | <2% critical               | NaN in lags (expected)     | <1% optional, NaN in lags  | Yes             |
| Visualizations     | 13 (EDA)                   | +5 (lag/rolling)           | +10-15 total validation    | Yes             |
| Daily checkpoints  | 0                          | 2 complete                 | 5 complete                 | Yes             |
| Buffer remaining   | 12.5h                      | ~10-11h (est)              | TBD                        | Monitor         |
| **Summary**        | **Analysis-ready dataset** | **MUST features complete** | **Modeling-ready dataset** | **8/8 targets** |

### Key Benefits
- **Temporal dependencies captured**: Lag features encode autocorrelation (r=0.60+)
- **Smoothing trends**: Rolling statistics reduce noise while preserving signal
- **Macro context**: Oil prices provide economic indicator (-0.55 correlation)
- **Baseline comparisons**: Store/item aggregations enable heterogeneous modeling
- **Promotion optimization**: Timing features support promotional strategy (+74% lift)
- **Week 3 readiness**: Rich feature set enables advanced forecasting (ARIMA, Prophet, LSTM)
- **Documentation excellence**: Feature dictionary v2 + 5 daily checkpoints provide complete reference
- **Agile execution**: Daily checkpoints enable mid-course corrections without derailing schedule

---

## 10. Communication Plan

### Daily Progress Updates
- **Frequency**: End of each day
- **Format**: Brief summary (5 minutes to write)
- **Content**: Features created, computation time, validation results, next day plan
- **Audience**: Self (for tracking), advisor (if requested)

### Daily Checkpoints (Days 1-5) ⚠️ NEW
- **Timing**: End of each day (last 15 minutes)
- **Format**: Structured review using checkpoint template (Section 11)
- **Content:**
  - Time tracking (actual vs allocated)
  - Scope completion (all parts done?)
  - Feature quality (validation passed?)
  - Findings (unexpected discoveries?)
  - Adjustment decisions (add/remove features for next day?)
- **Output**: Daily checkpoint document (Day[X]_Checkpoint_Week2.md)
- **Cumulative tracking**: Update buffer table after each checkpoint

**Purpose:** Enable agile adaptation, prevent scope creep, maintain buffer discipline.

### Mid-Week Check-In
- **Timing**: Wednesday (Day 3) after daily checkpoint
- **Format**: Email or quick note (~5 min)
- **Content**: MUST features complete (lag + rolling), SHOULD features in progress, buffer status
- **Audience**: Advisor (maintain visibility)

### Week-End Summary
- **Timing**: Friday end-of-day (Day 5)
- **Format**: Week 2 checkpoint document (~2 pages)
- **Content**: Week 2 accomplishments, feature count, computation stats, Week 3 preview, buffer utilization summary
- **Audience**: Advisor, peer reviewers

### Week 2 → Week 3 Handoff
- **Timing**: Friday evening
- **Format**: Handoff document (similar to Week1_to_Week2_Handoff.md)
- **Content**: Final dataset location, feature summary, key decisions, modeling recommendations, buffer carryover
- **Audience**: Next session continuation (self), advisor

---

## 11. Daily Checkpoint Template

Use this template at the end of Days 1, 2, 3, 4, and 5 (allocate 15 minutes):

```markdown
## Day X Checkpoint - Week 2 (YYYY-MM-DD)

### Time Tracking
- **Allocated:** 4 hours (3.2h core + 0.8h buffer)
- **Actual:** [X.X] hours
- **Variance:** [+/-X.X] hours
- **Reason for variance:** [Brief explanation - e.g., "validation took longer", "ahead due to efficient scripting"]

### Scope Completion
- [ ] Part 0: [Task name] - [Complete/Partial/Not started]
- [ ] Part 1: [Task name] - [Complete/Partial/Not started]
- [ ] Part 2: [Task name] - [Complete/Partial/Not started]
- [ ] Part 3: [Task name] - [Complete/Partial/Not started]
- [ ] Part 4: [Task name] - [Complete/Partial/Not started] (if applicable)

**Completion Rate:** [X/Y parts complete] = [XX%]

### Key Findings
1. **Most important finding:** [1-2 sentences]
2. **Second most important finding:** [1-2 sentences]
3. **Unexpected discovery:** [1-2 sentences or "None"]

### Quality Assessment
- **Feature quality:** [Excellent/Good/Needs improvement] - [Why?]
- **Validation results:** [All passed/Partial failures/Failed] - [Details]
- **Computation time:** [Within/Over budget] - [Actual time: X min]
- **Code quality:** [Clean/Needs refactoring/Has issues]

### Blockers & Issues
- **Technical blockers:** [List or "None"]
- **Data quality issues:** [List or "None"]
- **Conceptual challenges:** [List or "None"]
- **Mitigation actions taken:** [What did you do?]

### Buffer Status
- **Day X buffer allocated:** 0.8h
- **Day X buffer used:** [X.X]h
- **Day X buffer remaining:** [X.X]h
- **Cumulative buffer remaining (Week 1 + Week 2):** [X.X]h / 12.5h
- **Buffer health:** [Healthy (>8h) / Caution (5-8h) / Critical (<5h)]

### Feature Creation Status
**MUST Features (10 total):**
- [ ] Lag 1/7/14/30 (4) - [Complete/In progress/Planned]
- [ ] Rolling avg 7/14/30 (3) - [Complete/In progress/Planned]
- [ ] Rolling std 7/14/30 (3) - [Complete/In progress/Planned]

**SHOULD Features (15 total):**
- [ ] Oil features (5) - [Complete/In progress/Planned]
- [ ] Store/cluster aggs (3) - [Complete/In progress/Planned]
- [ ] Item aggs (5) - [Complete/In progress/Planned]
- [ ] Promotion features (2) - [Complete/In progress/Planned]

**COULD Features (0-3):**
- [ ] Optional features - [Complete/Skipped/Planned]

**Total Features Created Today:** [X] features  
**Cumulative Features:** [XX] / 54-57 target

### Adjustment Decisions for Day X+1

**Scope Changes:**
- [ ] Keep plan as-is
- [ ] Add analysis/feature: [Specify what and why]
- [ ] Remove analysis/feature: [Specify what and why]
- [ ] Simplify approach: [Specify what and why]

**Time Reallocation:**
- [ ] No changes needed
- [ ] Increase time for: [Activity] by [X] minutes
- [ ] Decrease time for: [Activity] by [X] minutes

**Priority Adjustment:**
- [ ] Maintain MUST → SHOULD → COULD priority
- [ ] Focus only on MUST features (contingency triggered)
- [ ] Skip COULD features to preserve buffer

### Next Day Preview
**Day X+1 Primary Objectives:**
1. [Objective 1]
2. [Objective 2]

**Day X+1 Success Criteria:**
- [ ] [Criterion 1]
- [ ] [Criterion 2]

**Day X+1 Contingency Plan (if behind):**
- [What will you cut or simplify?]

### Decision Log Updates
- **DEC-0XX:** [Brief decision title]
  - Context: [1 sentence]
  - Decision: [1 sentence]
  - Impact: [What does this affect?]

### Notes & Learnings
- **What worked well today:** [1-2 items]
- **What could be improved:** [1-2 items]
- **Insights for Week 3:** [Anything to carry forward]

---

**Checkpoint completed by:** Alberto Diaz Durana  
**Time spent on checkpoint:** [X] minutes (target: ≤15 min)  
**Next checkpoint:** Day [X+1], [Date]
```

---

## 12. Summary

Week 2 (Version 2.0 - Expanded) builds upon Week 1's solid foundation with enhanced discipline:

**Core Work:**
1. Creating 4 lag features (1/7/14/30) to capture temporal dependencies (autocorr 0.60+)
2. Engineering 6 rolling statistics (avg/std) to smooth trends and quantify volatility
3. Incorporating 5 oil features as macro indicators (correlation -0.55)
4. Computing 8 aggregations (store/item/cluster) for baseline comparisons (4.25x gap)
5. Adding 2-3 promotion/payday features based on Week 1 insights (+74% lift)

**Version 2.0 Enhancements:**
1. **Daily checkpoint discipline** - 15 min structured reviews prevent drift
2. **Progressive outcomes tracking** - 3-stage table (before → Days 1-2 → Days 3-5)
3. **Cumulative buffer monitoring** - Real-time health check (12.5h starting buffer)
4. **Feature creep protection** - Explicit risk with daily mitigation checkpoints
5. **End-of-day checkpoint questions** - Quality gates at each day boundary
6. **MUST/SHOULD/COULD clarity** - Priority structure prevents scope paralysis
7. **Adjustment decision framework** - Built-in agility for mid-course corrections

Upon completion, Week 3 (Analysis & Modeling) can begin with:
- Rich, validated feature set (54-57 columns)
- Documented decisions (5 daily checkpoints + 2-3 decision log entries)
- Clear understanding of feature quality and limitations
- Preserved buffer for Week 3 challenges
- Agile execution mindset carried forward

**Week 2 deliverables enable Week 3 modeling success.**

---

**Critical Reminders for Week 2 Execution:**

1. **⚠️ ALWAYS sort by (store_nbr, item_nbr, date) before lag/rolling operations**
2. **⚠️ Use groupby for per store-item features, not global operations**
3. **⚠️ Keep NaN in lag features (do not fill with 0 or drop rows)**
4. **⚠️ Forward-fill oil prices for missing weekend dates**
5. **⚠️ Validate features with sample visualizations (avoid silent errors)**
6. **⚠️ Complete daily checkpoint (15 min) - Non-negotiable for agile execution**
7. **⚠️ Monitor buffer health - Trigger contingency if <8h after Day 3**

---

**End of Week 2 Project Plan (Version 2.0 - Expanded)**

**Last Updated:** 2025-11-12  
**Next Review:** After Week 2 completion (Day 5 checkpoint)

# Corporación Favorita Grocery Sales Forecasting
## Week 2 Project Plan: Feature Development

**Prepared by:** Alberto Diaz Durana  
**Timeline:** Week 2 (5 working days, 20 hours total)  
**Phase:** Phase 2 - Feature Development  
**Previous Phase:** Week 1 - Exploration & Understanding (COMPLETE)  
**Next Phase:** Week 3 - Analysis & Modeling

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
- Carryover buffer: +8.5 hours from Week 1 efficiency (total buffer: 12.5h)

**Week 1 Foundation:**
- Dataset: `guayas_prepared.pkl` (300,896 rows × 28 columns)
- Key findings: Weekend +34%, Promotion +74%, Autocorrelation 0.60+, Pareto 34/80
- 10 decisions logged, 13 visualizations created
- Quality: 0% missing in critical features, outliers flagged

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

| Day | Focus Area | Core Hours | Buffer | Total | Key Deliverables |
|-----|-----------|------------|--------|-------|------------------|
| 1 | Lag Features (1/7/14/30) | 3.2h | 0.8h | 4h | 4 lag features, temporal validation |
| 2 | Rolling Statistics (7/14/30 avg/std) | 3.2h | 0.8h | 4h | 6 rolling features, smoothing visualizations |
| 3 | Oil Price Features + Lags | 3.2h | 0.8h | 4h | 4 oil features, correlation analysis |
| 4 | Store/Item Aggregations | 3.2h | 0.8h | 4h | 6 aggregation features, baseline metrics |
| 5 | Promotion/Payday + Final Export | 3.2h | 0.8h | 4h | Final features, guayas_features.pkl, feature dictionary v2 |
| **Total** | | **16h** | **4h** | **20h** | **20-30 engineered features** |

**Cumulative Buffer Status:**
- Week 1 surplus: +8.5h
- Week 2 allocation: +4h
- Total available buffer: **12.5 hours**

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
- 4 oil features (oil_price + 3 lags + 1 change metric)
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
- **Sparsity feature** (15 min):
  - `days_since_last_sale`: Days since last non-zero sale
- **Interaction terms** (30 min):
  - `promo_x_weekend`: Promotion × Weekend interaction
  - Test: Does promotion effectiveness differ on weekends?
- **Store-item interaction** (15 min):
  - `store_item_avg_sales`: Mean sales per store-item pair (finer granularity)
**Deliverables:**
- 0-4 optional features (depending on time)

#### Part 4: Final Export & Documentation (1 hour)
**Objective:** Create modeling-ready dataset and complete documentation
**Activities:**
- Final feature count: 51 + 3 mandatory + 0-4 optional = **54-58 columns**
- Save `data/processed/guayas_features.pkl` (FINAL WEEK 2 OUTPUT)
- Export CSV version: `data/processed/guayas_features.csv` (for reference)
- Update feature dictionary v2 (complete documentation of all 54-58 features)
- Generate feature summary statistics (min/max/mean/std per feature)
- Create Week 2 checkpoint document
**Deliverables:**
- `guayas_features.pkl` (300,896 × 54-58 columns) - **PRIMARY OUTPUT**
- `guayas_features.csv` (backup/reference)
- `feature_dictionary_v2.txt` (54-58 features documented)
- `Day5_Checkpoint_Week2_Summary.md`
- Feature summary statistics table

---

## 5. Phase 2 Readiness Checklist (for Week 3 Transition)

### Required Inputs
- [✓] Prepared dataset (guayas_prepared.pkl, 300,896 rows × 28 columns)
- [✓] Week 1 insights (autocorrelation, patterns, external factors)
- [✓] Oil price data (oil.csv, 1,218 daily prices)

### Completion Criteria
- [ ] 20-30 engineered features created (target: 54-58 total columns)
- [ ] Lag features (4): 1/7/14/30-day lags with proper temporal sorting
- [ ] Rolling features (6): 7/14/30-day avg/std computed correctly
- [ ] Oil features (4-5): Daily price + lags merged and forward-filled
- [ ] Aggregations (8): Store/item/cluster baseline metrics calculated
- [ ] Promotion features (2-3): History and frequency encoded
- [ ] Payday feature (1): Binary flag for days 1-3, 14-16
- [ ] Missing value strategy: NaN in lags kept (XGBoost compatible), <5% in optional features

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
- [ ] `d01_w02_feature_engineering_lags.ipynb` (runs without errors)
- [ ] `d02_w02_feature_engineering_rolling.ipynb` (runs without errors)
- [ ] `d03_w02_feature_engineering_oil.ipynb` (runs without errors)
- [ ] `d04_w02_feature_engineering_aggregations.ipynb` (runs without errors)
- [ ] `d05_w02_feature_engineering_final.ipynb` (runs without errors)
- [ ] `guayas_features.pkl` (final dataset, 300,896 × 54-58 columns)
- [ ] `feature_dictionary_v2.txt` (complete feature documentation)
- [ ] Week 2 summary report (Day5_Checkpoint_Week2_Summary.md)

### Next Phase Readiness
After completing Week 2, you will have:
- Comprehensive feature set (54-58 columns) capturing temporal, external, and baseline patterns
- Validated features with no data leakage or temporal ordering errors
- Modeling-ready dataset for Week 3 (ARIMA, Prophet, LSTM)
- Feature importance baseline (preliminary correlation analysis)
- Documented feature engineering decisions (2-3 new decision log entries)
- Visualizations validating feature quality (10-15 plots)

---

## 6. Success Criteria

### Quantitative
- Feature count: 54-58 columns (from 28 in Week 1) = **+26-30 engineered features**
- Missing values: 0% in MUST features (lags kept as NaN by design), <5% in COULD features
- Computation time: <40 minutes total for all feature engineering on 300K rows
- Lag feature NaN: lag1 ~2.7K, lag7 ~19K, lag14 ~38K, lag30 ~82K (expected edge case behavior)
- Rolling feature NaN: <1% (min_periods=1 minimizes NaN)
- Oil feature NaN: 0% (forward-fill applied)
- Aggregation NaN: 0% (computed from complete historical data)
- 5 notebooks created (~300-400 lines each)
- 10-15 visualizations (validation plots, correlations, smoothing effects)

### Qualitative
- Feature engineering decisions clearly documented (2-3 new decision log entries)
- Lag features capture temporal dependencies identified in Week 1 autocorrelation
- Rolling features smooth noise without excessive lag
- Oil features validated as relevant macro indicators
- Aggregations provide meaningful baseline comparisons (store/item heterogeneity)
- Features interpretable by business stakeholders (e.g., "7-day average sales")
- No data leakage (rigorous validation)

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
- **Commit message format**: "Week 2 Day X: [accomplishment summary]"

### Key Documents
- `Week2_ProjectPlan.md` (this document)
- `decision_log.md` (update with 2-3 new feature engineering decisions)
- `feature_dictionary_v2.txt` (complete 54-58 feature documentation)
- `Day5_Checkpoint_Week2_Summary.md` (week-end summary)
- `Week2_to_Week3_Handoff.md` (prepare for next phase)

### Assumptions
- Lag NaN handling: Keeping NaN for first observations is acceptable (XGBoost/LightGBM handle natively)
- Oil price relevance: -0.55 correlation justifies inclusion despite being moderate
- Forward-fill oil prices: Assumption that weekend oil prices = Friday's price is reasonable
- Store/item aggregations: Historical averages are stable enough to use as features
- Sparse format maintenance: No calendar gap filling (300K rows, not 33M)

### Limitations
- NaN in lag features may limit some models (e.g., linear regression requires imputation)
- Rolling features with `min_periods=1` may have low-quality values at start (partial windows)
- Oil price correlation (-0.55) is moderate, not strong; may not drive significant predictions
- Aggregations assume stationarity (historical avg = future baseline), which may not hold during regime changes
- Limited to 300K sample; feature patterns may differ in full 33M dataset

### Ownership
- **Project Lead**: Alberto Diaz Durana
- **Stakeholders**: Academic advisor, peer reviewers
- **Timeline**: Week 2 of 4-week project

---

## 8. Risk Management

### Identified Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Computation time exceeds estimate (>40 min) | Low | Low | Use Dask if RAM issues, already validated on 300K |
| NaN in lags breaks downstream models | Low | Medium | Document NaN strategy, test with sample XGBoost |
| Oil price merge introduces errors | Low | High | Validate date alignment, spot-check merged values |
| Feature importance analysis shows low utility | Medium | Low | Proceed to Week 3, evaluate in model training |
| Temporal sort accidentally skipped | Low | Critical | Add assertion checks, validate with sample store-item |
| Time overrun (>20 hours) | Low | Medium | Use 12.5h cumulative buffer, deprioritize COULD features |

### Contingency Plans
- **If computation time >1 hour**: Skip COULD features, proceed with MUST+SHOULD only (48 columns)
- **If oil merge fails**: Exclude oil features, continue with lag/rolling/aggregations (49 columns)
- **If time constraint tight**: Focus on MUST features only (lag + rolling = 10 features), defer rest to Week 3 prep time

---

## 9. Expected Outcomes

### Expected Outcomes Table

| Metric | Week 1 End | Week 2 End | Improvement | Target Met |
|--------|-----------|------------|-------------|------------|
| Feature count | 28 columns | 54-58 columns | +26-30 engineered | Yes |
| Lag features | 0 | 4 (1/7/14/30) | Temporal memory added | Yes |
| Rolling features | 1 prototype | 6 (avg/std × 3 windows) | Trend/volatility capture | Yes |
| External factors | Holiday only | + Oil (5 features) | Macro indicators | Yes |
| Aggregations | 0 | 8 (store/item/cluster) | Baseline comparisons | Yes |
| Promotion history | Static flag | + 2-3 timing features | Dynamic patterns | Yes |
| Missing values | <2% critical | <5% optional, NaN in lags | Controlled, documented | Yes |
| Visualizations | 13 (EDA) | +10-15 (validation) | Feature quality verified | Yes |
| **Summary** | **Analysis-ready dataset** | **Modeling-ready dataset** | **Feature engineering complete** | **7/7 targets** |

### Key Benefits
- **Temporal dependencies captured**: Lag features encode autocorrelation (r=0.60+)
- **Smoothing trends**: Rolling statistics reduce noise while preserving signal
- **Macro context**: Oil prices provide economic indicator (-0.55 correlation)
- **Baseline comparisons**: Store/item aggregations enable heterogeneous modeling
- **Promotion optimization**: Timing features support promotional strategy (+74% lift)
- **Week 3 readiness**: Rich feature set enables advanced forecasting (ARIMA, Prophet, LSTM)
- **Documentation excellence**: Feature dictionary v2 provides complete reference

---

## 10. Communication Plan

### Daily Progress Updates
- **Frequency**: End of each day
- **Format**: Brief summary (5 minutes to write)
- **Content**: Features created, computation time, validation results, next day plan
- **Audience**: Self (for tracking), advisor (if requested)

### Mid-Week Check-In
- **Timing**: Wednesday (Day 3)
- **Format**: Email or quick note (~5 min)
- **Content**: MUST features complete (lag + rolling), on track for SHOULD features
- **Audience**: Advisor (maintain visibility)

### Week-End Summary
- **Timing**: Friday end-of-day (Day 5)
- **Format**: Week 2 checkpoint document (~1 page)
- **Content**: Week 2 accomplishments, feature count, computation stats, Week 3 preview
- **Audience**: Advisor, peer reviewers

### Week 2 → Week 3 Handoff
- **Timing**: Friday evening
- **Format**: Handoff document (similar to Week1_to_Week2_Handoff.md)
- **Content**: Final dataset location, feature summary, key decisions, modeling recommendations
- **Audience**: Next session continuation (self), advisor

---

## Day 1 Summary Template

### Time Allocation (4 hours total):
| Task | Duration | Percentage |
|------|----------|------------|
| Part 0: Load & Temporal Sort | 30 min | 12.5% |
| Part 1: Create Basic Lag Features | 1.5 hours | 37.5% |
| Part 2: Validation & Visualization | 1 hour | 25% |
| Part 3: Save Checkpoint | 30 min | 12.5% |
| Buffer | 30 min | 12.5% |
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

Week 2 builds upon Week 1's solid foundation by:
1. Creating 4 lag features (1/7/14/30) to capture temporal dependencies (autocorr 0.60+)
2. Engineering 6 rolling statistics (avg/std) to smooth trends and quantify volatility
3. Incorporating 5 oil features as macro indicators (correlation -0.55)
4. Computing 8 aggregations (store/item/cluster) for baseline comparisons (4.25x gap)
5. Adding 3-5 promotion/payday features based on Week 1 insights (+74% lift)

Upon completion, Week 3 (Analysis & Modeling) can begin with a rich, validated feature set (54-58 columns) that encodes temporal patterns, external factors, and baseline performance for advanced forecasting models.

**Week 2 deliverables enable Week 3 modeling success.**

---

**Critical Reminders for Week 2 Execution:**

1. **⚠️ ALWAYS sort by (store_nbr, item_nbr, date) before lag/rolling operations**
2. **⚠️ Use groupby for per store-item features, not global operations**
3. **⚠️ Keep NaN in lag features (do not fill with 0 or drop rows)**
4. **⚠️ Forward-fill oil prices for missing weekend dates**
5. **⚠️ Validate features with sample visualizations (avoid silent errors)**

---

**End of Week 2 Project Plan**

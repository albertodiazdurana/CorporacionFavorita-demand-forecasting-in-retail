# DEC-013: Train/Test Split with 7-Day Gap

**Decision ID:** DEC-013  
**Date:** 2025-11-18  
**Phase:** Week 3 Day 1 - Baseline Modeling  
**Category:** Modeling Strategy  
**Status:** APPROVED  
**Impact:** High (affects all Week 3 model evaluation validity)

---

## Context

### Problem Statement
When creating train/test splits for time series forecasting with lag features, there is risk of **data leakage** if test set predictions use information from the training period through lag features.

### Specific Situation
- **Dataset:** Q1 2014 (Jan 1 - Mar 31, 90 days total)
- **Lag features:** Maximum lag is 30 days (unit_sales_lag30)
- **Most important lag:** lag7 (correlation r=0.40 from Week 2)
- **Initial plan:** Train on Jan-Feb, test on March (no gap)
- **Risk identified:** lag features would use training period data in test predictions

### Example of Leakage
Without gap, predicting March 1:
- `unit_sales_lag1` uses Feb 28 data (training set) ← LEAKAGE
- `unit_sales_lag7` uses Feb 22 data (training set) ← LEAKAGE
- `unit_sales_lag30` uses Jan 30 data (training set) ← LEAKAGE

This creates **information leakage** from training into test evaluation, inflating performance metrics.

---

## Decision

**Implement 7-day gap between training and test sets:**

- **Training period:** January 1 - February 21, 2014 (52 days)
- **Gap period:** February 22 - February 28, 2014 (7 days) - **EXCLUDED from both sets**
- **Test period:** March 1 - March 31, 2014 (31 days)

**Gap duration rationale:** 7 days chosen to prevent leakage for `unit_sales_lag7` (strongest lag feature).

---

## Alternatives Considered

### Option 1: 30-Day Gap (Strict)
**Configuration:**
- Train: Jan 1 - Jan 30 (30 days)
- Gap: Jan 31 - Feb 28 (29 days)
- Test: March 1 - March 31 (31 days)

**Pros:**
- Eliminates ALL lag feature leakage (including lag30)
- Theoretically cleanest approach

**Cons:**
- Only 30 days training data (insufficient for baseline)
- Sacrifices 29 days of valuable training data
- Model would be undertrained

**Verdict:** REJECTED - Training data too limited

---

### Option 2: No Gap (Original Plan)
**Configuration:**
- Train: Jan 1 - Feb 28 (59 days)
- Gap: None
- Test: March 1 - March 31 (31 days)

**Pros:**
- Maximum training data (59 days)
- Simple implementation
- Matches course materials structure

**Cons:**
- All lag features leak training data into test predictions
- Artificially inflated performance metrics
- Not following time series best practices
- Invalidates model evaluation

**Verdict:** REJECTED - Unacceptable data leakage

---

### Option 3: 14-Day Gap (Balanced)
**Configuration:**
- Train: Jan 1 - Feb 14 (45 days)
- Gap: Feb 15 - Feb 28 (14 days)
- Test: March 1 - March 31 (31 days)

**Pros:**
- Prevents lag7 and lag14 leakage
- Reasonable training data (45 days)
- More conservative than 7-day

**Cons:**
- Sacrifices 7 additional training days vs 7-day gap
- lag30 still uses training data (same as 7-day)
- Marginal benefit over 7-day gap

**Verdict:** CONSIDERED but not selected - 7-day gap sufficient

---

### Option 4: 7-Day Gap (SELECTED)
**Configuration:**
- Train: Jan 1 - Feb 21 (52 days)
- Gap: Feb 22 - Feb 28 (7 days)
- Test: March 1 - March 31 (31 days)

**Pros:**
- Prevents lag7 leakage (strongest autocorrelation: r=0.40)
- Good training data availability (52 days)
- Pragmatic balance for academic project
- Follows time series best practices
- Sufficient for baseline and tuned models

**Cons:**
- lag14 and lag30 still use some training period data
- Not perfect (strict 30-day gap would be ideal in production)

**Verdict:** SELECTED - Best balance of rigor and practicality

---

## Rationale

### Why 7 Days Specifically?

**Focus on lag7 (most important):**
- Strongest autocorrelation: r=0.40 (Week 2 finding)
- Most predictive lag feature in baseline model
- Preventing this leakage has highest impact on evaluation validity

**Acceptable trade-offs:**
- `unit_sales_lag1`: Uses gap period data (Feb 22-28)
  - Impact: Minimal leakage, only 1 day back
  - Acceptable: lag1 has weaker correlation (r=0.26)
  
- `unit_sales_lag7`: Uses gap period data (Feb 15-21 for March 1 prediction)
  - Impact: **NO LEAKAGE** - gap prevents
  - Success: This is the target we're solving for

- `unit_sales_lag14`: Partially uses training period (Feb 7-14 for March 1 prediction)
  - Impact: Some leakage, but lag14 has moderate correlation (r=0.32)
  - Acceptable: Trade-off for sufficient training data

- `unit_sales_lag30`: Uses training period (Jan 30 for March 1 prediction)
  - Impact: Leakage present, but lag30 has weakest correlation (r=0.27)
  - Acceptable: 30-day gap would sacrifice too much training data

### Pragmatic for Academic Project
- Limited Q1 2014 data (90 days total)
- Need sufficient training data for baseline establishment
- Goal: Demonstrate methodology, not production deployment
- Documented limitation acceptable for course project

### Balances Rigor and Practicality
- Follows time series best practices (gap period used)
- Prevents most critical leakage (lag7)
- Maintains sufficient training data (52 days > 30 days minimum)
- Enables fair comparison between baseline and tuned models

---

## Implementation

### Code Implementation
```python
# Filter to Q1 2014
df_2014q1 = df[(df['date'] >= '2014-01-01') & (df['date'] <= '2014-03-31')].copy()

# Split with 7-day gap
train = df_2014q1[df_2014q1['date'] <= '2014-02-21'].copy()
gap = df_2014q1[(df_2014q1['date'] > '2014-02-21') & 
                (df_2014q1['date'] < '2014-03-01')].copy()
test = df_2014q1[df_2014q1['date'] >= '2014-03-01'].copy()

# Gap period excluded from both train and test
print(f"Train: {len(train)} rows")
print(f"Gap (excluded): {len(gap)} rows")
print(f"Test: {len(test)} rows")
```

### Verification
```python
# Verify gap period
latest_train_date = train['date'].max()  # 2014-02-21
earliest_test_date = test['date'].min()   # 2014-03-01
gap_days = (earliest_test_date - latest_train_date).days - 1  # 7 days

assert gap_days >= 7, "Gap must be at least 7 days"
```

### Results
- Training samples: 7,050 rows
- Gap samples: 932 rows (excluded)
- Test samples: 4,686 rows
- Train/test ratio: 60.1% / 39.9%
- Gap verification: ✓ 7 days confirmed

---

## Impact

### Immediate Impact (Week 3 Day 1)
- Baseline model evaluation now valid (no lag7 leakage)
- Performance metrics reflect true out-of-sample forecast ability
- RMSE of 7.21 is trustworthy benchmark for Day 3 tuning

### Project-Wide Impact
- All Week 3 models use same split (fair comparison)
- Hyperparameter tuning target (Day 3) based on valid baseline
- Feature importance (Day 2) reflects true predictive power
- Final model selection (Day 5) based on unbiased evaluation

### Documentation Impact
- Methodology rigor improved
- Limitation documented transparently
- Decision rationale clear for stakeholders
- Portfolio piece demonstrates best practices

### Week 4 Impact (Deployment)
- Preprocessing artifacts must respect gap period
- Streamlit app documentation should note lag feature behavior
- Production deployment may require stricter gap (30 days recommended)

---

## Limitations Acknowledged

### Partial Leakage Remains
- lag14 and lag30 still use some training period data
- Acceptable trade-off for academic project scope
- Would require 30-day gap for complete prevention

### Production Recommendations
For production deployment:
1. Use 30-day gap (prevents all lag leakage)
2. Or train on longer history (12+ months) where 30-day gap is smaller %
3. Or use cross-validation with multiple expanding windows
4. Document gap period in model card

### Dataset Constraints
- Q1 2014 only has 90 days total
- 30-day gap would leave 30 days training (too little)
- Larger dataset would allow stricter gap

---

## Success Metrics

### Evaluation Validity
- **Target:** lag7 leakage prevented ✓
- **Achieved:** Gap of 7 days confirmed ✓
- **Evidence:** Test date (2014-03-01) - 7 days = 2014-02-22 (outside training max 2014-02-21) ✓

### Training Data Sufficiency
- **Target:** ≥45 days training data
- **Achieved:** 52 days ✓
- **Evidence:** 7,050 samples across 52 days ✓

### Model Performance
- **Target:** RMSE improvement over naive baseline
- **Achieved:** 41.75% improvement ✓
- **Evidence:** Baseline RMSE 7.21 vs naive 12.38 ✓

---

## Related Decisions

### Prior Decisions
- **DEC-005:** Sparse Data Handling (Week 1) - Work with 300K sparse rows
- **DEC-011:** Lag NaN Strategy (Week 2) - Keep NaN, don't impute (XGBoost handles)
- Both decisions compatible with gap period approach

### Future Decisions
- **Week 3 Day 3:** Hyperparameter tuning will use same split
- **Week 3 Day 4:** LSTM model (if implemented) will use same split
- **Week 4:** Deployment gap period (may increase to 30 days for production)

---

## References

### Time Series Best Practices
- Hyndman & Athanasopoulos (2021), "Forecasting: Principles and Practice" - Chapter 5
- Sklearn TimeSeriesSplit documentation (expanding window cross-validation)
- Academic standard: Gap period equal to maximum lag

### Project Context
- Week 1 autocorrelation analysis (lag7 r=0.40)
- Week 2 lag feature engineering (4 lag features created)
- Week 3 Day 1 baseline modeling (RMSE 7.21)

---

## Approval

**Proposed by:** Alberto Diaz Durana (Week 3 Day 1 session)  
**Reviewed by:** Alberto Diaz Durana  
**Approved by:** Alberto Diaz Durana  
**Date:** 2025-11-18  
**Status:** APPROVED and IMPLEMENTED

---

## Revision History

| Version | Date | Change | Author |
|---------|------|--------|--------|
| 1.0 | 2025-11-18 | Initial decision documented | Alberto Diaz Durana |

---

**END OF DECISION LOG DEC-013**

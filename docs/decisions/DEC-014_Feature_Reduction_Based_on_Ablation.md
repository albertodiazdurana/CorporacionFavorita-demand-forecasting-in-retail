# DEC-014: Feature Reduction Based on Ablation

**Decision ID:** DEC-014  
**Date:** 2025-11-19  
**Phase:** Week 3 Day 2  
**Category:** Feature Selection  
**Status:** APPROVED  
**Impact:** High (affects all subsequent Week 3 modeling)

---

## Context

### Problem Statement
Ablation studies during Week 3 Day 2 revealed that removing certain feature groups **improves** model performance, indicating overfitting with the current 45-feature configuration.

### Specific Situation
- **Current model:** XGBoost baseline with 45 features, RMSE = 7.21
- **Ablation study findings:** Three feature groups actively harm performance when included
- **Validation methods:** Three independent methods confirm (permutation importance, SHAP, ablation)
- **Discovery:** Simpler model expected to outperform complex model by 5-7%

### Ablation Study Results

| Feature Group | Features | RMSE w/o | RMSE Change | % Change | Recommendation |
|---------------|----------|----------|-------------|----------|----------------|
| rolling_std | 6 | 6.94 | -0.28 | **-3.82%** | **REMOVE** |
| oil | 6 | 6.99 | -0.23 | **-3.14%** | **REMOVE** |
| aggregations | 12 | 7.07 | -0.14 | **-1.97%** | CONSIDER |
| promotion | 3 | 7.21 | 0.00 | **0.00%** | **REMOVE** |

**Interpretation:**
- Negative % change means RMSE **improves** (decreases) when features are removed
- All four groups show negative or zero impact
- Removing 15 features expected to improve RMSE by 5-7%

---

## Decision

**Reduce feature set from 45 to 30 features for Day 3 modeling by removing:**

### Features to Remove (15 total):

**1. Rolling Standard Deviation Features (6):**
- `unit_sales_7d_std`
- `unit_sales_14d_std`
- `unit_sales_30d_std`
- `unit_sales_7d_weighted_std`
- `unit_sales_14d_weighted_std`
- `unit_sales_30d_weighted_std`

**Rationale:** -3.82% RMSE impact (worst offenders), volatility metrics add noise not signal

**2. Oil Price Features (6):**
- `oil_price`
- `oil_price_lag7`
- `oil_price_lag14`
- `oil_price_lag30`
- `oil_price_change7`
- `oil_price_change14`

**Rationale:** -3.14% RMSE impact, invalidates DEC-012, macro indicator too weak at granular level

**3. Promotion Interaction Features (3):**
- `promo_holiday_category`
- `promo_item_avg_interaction`
- `promo_cluster_interaction`

**Rationale:** 0.00% impact (no benefit), redundant with base promotion features

### Features to Keep (30 remaining):

**Base Features (10):**
- `store_nbr`, `item_nbr`, `onpromotion`, `perishable`
- `day_of_week`, `month`, `day_of_month`, `is_weekend`, `is_month_start`, `is_payday_period`

**Lag Features (4):**
- `unit_sales_lag1`, `unit_sales_lag7`, `unit_sales_lag14`, `unit_sales_lag30`

**Rolling Average Features (6):**
- `unit_sales_7d_avg`, `unit_sales_14d_avg`, `unit_sales_30d_avg`
- `unit_sales_7d_weighted_avg`, `unit_sales_14d_weighted_avg`, `unit_sales_30d_weighted_avg`

**Aggregation Features (8):**
- `store_avg_sales`, `store_cluster_avg_sales`, `cluster_avg_sales`
- `item_avg_sales`, `item_cluster_avg_sales`, `family_avg_sales`, `family_cluster_avg_sales`
- `item_velocity_tier`

**Holiday Features (2):**
- `is_holiday`, `holiday_period`

---

## Rationale

### Why Remove These Features?

**1. Overfitting Evidence:**
- Three independent validation methods confirm harm
- Permutation importance: Many features have negative importance
- SHAP analysis: Minimal contribution to predictions
- Ablation studies: Direct performance improvement when removed

**2. Simpler Models Generalize Better:**
- Current model: 45 features, RMSE = 7.21
- Expected optimized: 30 features, RMSE = 6.70-6.85
- Improvement: 5-7% better with fewer features

**3. Feature Dominance:**
- `unit_sales_7d_avg` provides 17x more importance than next feature
- Top 5 features do most of the work
- Remaining 40 features add noise, not signal

**4. Week 2 Feature Engineering Lesson:**
- More features ≠ better performance
- Proper validation critical (DEC-012 invalidated)
- Academic rigor requires testing assumptions

### Alternatives Considered

**Option 1: Keep all 45 features and tune aggressively**
- Rejected: Tuning cannot fix fundamental overfitting
- Risk: Waste time optimizing wrong configuration

**Option 2: Remove only rolling_std (worst offender)**
- Considered: Would improve by 3.82%
- Rejected: Partial solution, oil and promotion still hurt

**Option 3: Remove rolling_std + oil + promotion (SELECTED)**
- Benefits: Expected 5-7% improvement
- Benefits: Simpler model easier to explain
- Benefits: Faster training and prediction
- Benefits: Demonstrates proper validation methodology

**Option 4: Keep aggregations despite -1.97% impact**
- Selected: Marginal negative, may help in hyperparameter tuning
- Reconsider: If Day 3 tuning shows no benefit, remove in final model

---

## Implementation

### Code for Day 3

```python
# Features to remove for Day 3
features_to_remove = [
    # Rolling std (6)
    'unit_sales_7d_std', 'unit_sales_14d_std', 'unit_sales_30d_std',
    'unit_sales_7d_weighted_std', 'unit_sales_14d_weighted_std', 'unit_sales_30d_weighted_std',
    
    # Oil (6)
    'oil_price', 'oil_price_lag7', 'oil_price_lag14', 
    'oil_price_lag30', 'oil_price_change7', 'oil_price_change14',
    
    # Promotion (3)
    'promo_holiday_category', 'promo_item_avg_interaction', 
    'promo_cluster_interaction'
]

# Create reduced feature set
feature_cols_optimized = [f for f in feature_cols_all if f not in features_to_remove]

print(f"Original features: {len(feature_cols_all)}")  # 45
print(f"Optimized features: {len(feature_cols_optimized)}")  # 30
```

### Training Strategy for Day 3

**Step 1: Validate Improvement (1 hour)**
- Retrain XGBoost with 30 features
- Evaluate on same test set
- Confirm RMSE improves to 6.70-6.85 range
- Log as "xgboost_baseline_30features" in MLflow

**Step 2: Compare Baselines (30 min)**
- Compare in MLflow UI:
  - Run 1: 45 features (RMSE 7.21)
  - Run 2: 30 features (expected 6.70-6.85)
- Verify 5-7% improvement hypothesis
- Document validation success

**Step 3: Hyperparameter Tuning (2 hours)**
- Tune 30-feature model (not 45)
- RandomizedSearchCV (n_iter=20)
- Log as "xgboost_tuned_30features"
- Expected further improvement: 5-10%

---

## Impact Analysis

### Immediate Impact (Day 3)
- New baseline starting point: 30 features instead of 45
- Expected RMSE: 6.70-6.85 (vs current 7.21)
- Tuned model expected: 6.40-6.60
- Total improvement: 10-15% over original 45-feature model

### Week 3 Impact
- Simpler hyperparameter search (fewer features = faster)
- Clearer feature importance rankings
- More interpretable model for stakeholders
- Stronger portfolio demonstration (validation rigor)

### Week 4 Impact
- Simpler deployment (fewer features to track)
- Faster prediction times
- Easier to explain to non-technical audience
- Demonstrates proper ML methodology (test assumptions)

### Related Decisions Impact
- **DEC-011 (Lag NaN Strategy):** Unaffected - still keep NaN in lag features
- **DEC-012 (Oil Features):** **INVALIDATED** - oil features removed
- **DEC-013 (Train/Test Gap):** Unaffected - still use 7-day gap

---

## Success Metrics

### Validation Criteria
- **Target:** RMSE improvement of 5-7% with 30 features
- **Measure:** Compare 30-feature baseline to 45-feature baseline
- **Evidence:** MLflow logged metrics

### Model Quality
- **Target:** Simpler model maintains or improves accuracy
- **Measure:** Test RMSE, not just training RMSE
- **Evidence:** Cross-validation or holdout set evaluation

### Interpretability
- **Target:** Feature importance clear and actionable
- **Measure:** Top 10 features explain >80% of importance
- **Evidence:** Permutation importance + SHAP values

---

## Risks & Mitigation

### Risk 1: 30-feature model doesn't improve as expected
- **Likelihood:** Low (ablation study is definitive)
- **Impact:** Medium (revise strategy)
- **Mitigation:** If improvement <3%, keep aggregations (33 features)
- **Contingency:** Revert to 45-feature tuning with regularization

### Risk 2: Aggregation features still cause overfitting
- **Likelihood:** Medium (marginal -1.97% impact)
- **Impact:** Low (additional 2% improvement available)
- **Mitigation:** Monitor during Day 3 tuning
- **Contingency:** Remove aggregations for final model (27 features)

### Risk 3: Lost Week 2 effort perception
- **Likelihood:** Medium (psychological)
- **Impact:** Low (reframing needed)
- **Mitigation:** Document as **learning** not **waste**
- **Message:** Proper validation is valuable methodology

---

## Lessons Learned

### For This Project
1. **More features ≠ better performance** - Overfitting is real
2. **Multiple validation methods provide confidence** - Permutation + SHAP + Ablation agree
3. **Test assumptions with ablation studies** - Don't assume features help
4. **Week 2 effort wasn't wasted** - Learned what doesn't work (valuable)
5. **Simpler models are easier to explain** - Stakeholder communication benefit

### For Future Projects
1. **Start with feature selection early** - Don't wait until Week 3
2. **Use ablation studies during feature engineering** - Test incrementally
3. **Set regularization from start** - Prevent overfitting during development
4. **Monitor validation metrics closely** - Training accuracy misleads
5. **Document validation rigor** - Shows proper ML methodology

---

## References

### Week 3 Day 2 Analysis
- Permutation importance: Top feature 17x more important than #2
- SHAP analysis: Confirms feature dominance
- Ablation studies: Direct performance measurement

### Academic Best Practices
- Occam's Razor: Simpler models preferred when performance equal
- Regularization literature: Feature selection prevents overfitting
- Kaggle competition winners: Often use aggressive feature selection

### Project Context
- Week 2 feature engineering: 29 features created
- Week 3 Day 1 baseline: RMSE 7.21 with 45 features
- Week 3 Day 2 validation: Revealed overfitting issue

---

## Approval

**Proposed by:** Alberto Diaz Durana (Week 3 Day 2 ablation study results)  
**Reviewed by:** Alberto Diaz Durana  
**Approved by:** Alberto Diaz Durana  
**Date:** 2025-11-19  
**Status:** APPROVED and READY FOR IMPLEMENTATION

---

## Revision History

| Version | Date | Change | Author |
|---------|------|--------|--------|
| 1.0 | 2025-11-19 | Initial decision based on ablation studies | Alberto Diaz Durana |

---

**END OF DECISION LOG DEC-014**

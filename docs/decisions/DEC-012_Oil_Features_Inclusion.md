# Decision Log Entry: DEC-012

**Decision ID:** DEC-012  
**Date:** 2025-11-12  
**Phase:** Week 2 Day 3 - Oil Price Features  
**Author:** Alberto Diaz Durana

---

## Decision

**Include 6 oil features despite weak linear correlation at granular level; Add dual change derivatives (7-day and 14-day) for momentum capture**

---

## Context

During Week 2 Day 3 oil price feature engineering:

**Features Created:**
- `oil_price` (daily WTI crude oil price)
- `oil_price_lag7`, `lag14`, `lag30` (3 lagged values)
- `oil_price_change7`, `change14` (2 momentum derivatives)
- Total: 6 oil features (7 including duplicate to be cleaned)

**Correlation Findings:**

Week 1 expectation (aggregated time series):
- Oil vs total daily sales: r = -0.55 (moderate negative)
- Expected pattern: Higher oil → Lower consumer spending → Lower sales

Day 3 reality (granular store-item level):
- Oil vs unit_sales: r = +0.01 to +0.01 (near-zero, positive)
- Sign flip: Negative → Positive
- Magnitude drop: -0.55 → 0.01 (97% weaker)

**Why correlation changed:**
1. **Aggregation level:** Week 1 = 1,686 daily totals (aggregate), Day 3 = 300,896 store-item-date rows (granular)
2. **Sparse data effect:** 99.1% sparsity + item-level noise drowns out macro signal
3. **Non-linear relationships:** Linear correlation misses complex patterns tree models can learn
4. **Category heterogeneity:** Different product families may respond differently to oil prices

---

## Options Considered

### Option 1: Exclude all oil features (not worth the effort)
**Pros:**
- Simplifies feature space (39 columns → 38)
- Removes features with near-zero linear correlation
- Saves computation time

**Cons:**
- Discards Week 1 finding (r = -0.55 at aggregate level)
- Ignores potential non-linear relationships
- Misses category-specific effects (elastic vs inelastic goods)
- Models can't learn what they don't see

### Option 2: Include only oil_price (drop lags and derivatives)
**Pros:**
- Minimal feature addition (1 column)
- Provides macro context signal

**Cons:**
- Loses temporal dynamics (delayed effects)
- No momentum information (rate of change)
- Models can't learn lag-dependent patterns

### Option 3: Include oil_price + single change derivative (originally planned 5 features)
**Pros:**
- Moderate feature count (5 columns)
- Captures both level and momentum

**Cons:**
- Single derivative misses multi-scale momentum patterns
- May not distinguish short-term volatility from medium-term trends

### Option 4: Include full set with dual change derivatives (6 features)
**Pros:**
- Provides multiple perspectives for model learning
- Captures both levels (price, lags) and momentum (change7, change14)
- Dual derivatives distinguish short-term volatility from sustained trends
- Models can learn category-specific sensitivities
- Non-linear relationships may exist despite weak linear correlation
- Interaction effects with other features (promotions, holidays)
- Minimal cost: 6 columns, ~0% NaN, fast computation (0.2s)

**Cons:**
- Slightly larger feature space (39 → 45 columns)
- May be ignored by models if truly uninformative
- Duplicate feature (oil_price_change) to clean in Week 4

---

## Decision: Option 4 - Full Oil Feature Set (6 features)

---

## Rationale

### Technical Justification:

1. **Non-linear models can find patterns linear correlation misses**
   - XGBoost/LightGBM learn complex interactions via tree splits
   - Oil may interact with other features (e.g., oil × promotion)
   - Category-level patterns may emerge in feature importance

2. **Dual change derivatives capture multi-scale momentum**
   - `change7`: Short-term volatility, immediate market reactions
   - `change14`: Medium-term trends, smooths weekly noise
   - Models learn which matters for which products:
     - Elastic goods (BEVERAGES): Sensitive to short-term changes
     - Stable goods (CLEANING): Respond to long-term trends

3. **Momentum pattern framework provides interpretability**
   - change7 > 0, change14 > 0: Sustained upward (strong negative signal for sales)
   - change7 > 0, change14 < 0: Recent reversal (mixed signal)
   - change7 < 0, change14 < 0: Sustained downward (positive for consumer spending)
   - change7 fluctuates, change14 ≈ 0: High volatility, stable medium-term

4. **Week 1 aggregate correlation (-0.55) validates macro relevance**
   - Macro signal exists at aggregate level
   - Granular level dilution expected (similar to lag features: 0.60 → 0.26-0.40)
   - Models can aggregate upward during training

5. **Low cost, high upside**
   - 6 features, 0% NaN, 0.2s computation
   - If uninformative, models will learn low feature importance
   - If informative, provides critical macro context

### Business Justification:

1. **Ecuador is oil-dependent economy**
   - Oil exports drive national income
   - Transportation costs directly tied to oil prices
   - Consumer purchasing power influenced by oil-driven inflation

2. **Retail reality: Price sensitivity varies by category**
   - BEVERAGES (volatile, elastic): May respond to short-term oil changes
   - GROCERY I (stable): May respond to sustained trends
   - CLEANING (very stable): May be insensitive to oil

3. **Feature importance will guide interpretation**
   - Week 3 modeling will reveal if oil features rank high
   - Can analyze: Does oil_price_change7 predict BEVERAGES better than CLEANING?
   - Provides evidence for category-specific strategies

---

## Impact

### Week 3 Modeling:
- **XGBoost/LightGBM:** Will evaluate 6 oil features alongside 10 temporal features (lags + rolling)
- **Feature importance:** Oil ranking will determine actual utility vs theoretical concern
- **Interaction effects:** Models may find oil × promotion or oil × holiday patterns
- **Category analysis:** Can segment by family to assess differential sensitivity

### Feature Space:
- Total features: 45 columns (28 base + 17 engineered)
- Oil contribution: 6 features (13% of total)
- Complexity manageable for tree models

### Computational Cost:
- Merge + lag creation: 0.2 seconds
- Memory: +16 MB (192.4 MB total)
- Negligible impact on model training time

### Documentation:
- Feature dictionary: 6 new entries
- Week 4 cleanup: Remove duplicate `oil_price_change`
- Checkpoint: Justified scope enhancement (6 vs planned 5)

---

## Alternatives for Future Consideration

If oil features show zero importance in Week 3:

1. **Remove all oil features** for final model
   - Simplifies interpretation
   - Reduces feature space by 6 columns
   - Improves training speed marginally

2. **Keep only oil_price_change14** (if it alone shows importance)
   - Single momentum indicator
   - Medium-term trend capture
   - Simplest informative signal

3. **Create oil × category interactions** (if category effects detected)
   - oil_price_change7 × is_beverages
   - oil_price_change14 × is_cleaning
   - Explicit category-specific features

---

## Validation & Monitoring

**Week 2 Day 3 Validation:**
- Correlation analysis: COMPLETED (r ≈ 0.01, weak but non-zero)
- Merge integrity: VERIFIED (0% NaN after forward/back-fill)
- Date alignment: CONFIRMED (oil covers full main dataset range)
- Dual derivatives: VALIDATED (change7 std = $10.52, change14 std = $10.57)

**Week 3 Monitoring:**
- Track XGBoost feature importance (do oil features rank in top 20?)
- Compare performance with/without oil features (ablation study)
- Analyze prediction errors by oil volatility periods (high change vs stable)
- Segment by product family: BEVERAGES vs CLEANING sensitivity

**Week 4 Review:**
- Include oil feature importance in final report
- Document whether oil contributed to model performance
- Remove from final feature set if importance near-zero
- Provide recommendation for production deployment

---

## Approval

**Approved by:** Alberto Diaz Durana  
**Date:** 2025-11-12  
**Phase:** Week 2 Day 3 Complete  

**Next review:** Week 3 Day 1 (Model Training)  
**Review criteria:** Feature importance analysis, ablation study, category-level effects

---

## References

- Week 1 Day 4: Autocorrelation analysis (oil r = -0.55 for aggregated time series)
- Week 2 Day 3: Granular correlation (r ≈ +0.01, sign flip due to aggregation)
- Week 2 Project Plan v2: Section 5 (Oil Price Features)
- XGBoost documentation: Feature importance via gain metric
- Feature dictionary v2: Oil feature definitions with momentum framework

---

## Key Takeaways

1. **Weak linear correlation does NOT imply uninformative features** - Tree models can find non-linear patterns
2. **Dual change derivatives capture multi-scale momentum** - Different products respond to different timescales
3. **Aggregation level matters** - Macro signal (-0.55) diluted at granular level (+0.01) but still learnable
4. **Low-cost exploration justified** - 6 features, 0% NaN, 0.2s computation, models will reveal utility
5. **Week 3 validation critical** - Feature importance will determine final inclusion

---

**End of Decision Log Entry DEC-012**
# Data Inventory - Corporación Favorita Forecasting Project

**Project:** Corporación Favorita Grocery Sales Forecasting  
**Author:** Alberto Diaz Durana  
**Date Created:** 2025-11-XX  
**Purpose:** Document all input datasets, schemas, and data quality characteristics

---

## Overview

This inventory documents all raw data files from the Kaggle competition, including file sizes, schemas, row counts, and initial data quality observations.

**Data Source:** [Kaggle - Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)

**Download Date:** [YYYY-MM-DD]  
**Location:** `data/raw/`

---

## File Summary

| File | Size (MB) | Rows | Columns | Date Range | Purpose |
|------|-----------|------|---------|------------|---------|
| train.csv | ~479 | ~125M | 6 | 2013-01-01 to 2017-08-15 | Daily sales transactions |
| test.csv | TBD | TBD | 5 | 2017-08-16 to 2017-08-31 | Prediction targets |
| stores.csv | <1 | 54 | 5 | N/A | Store metadata |
| items.csv | <1 | 4,100 | 4 | N/A | Product metadata |
| oil.csv | <1 | 1,218 | 2 | 2013-01-01 to 2017-08-31 | Daily oil prices |
| holidays_events.csv | <1 | 350 | 6 | 2012-03-02 to 2017-12-26 | Holiday calendar |
| transactions.csv | ~4 | 83,488 | 3 | 2013-01-01 to 2017-08-15 | Daily transaction counts |
| sample_submission.csv | TBD | TBD | 2 | N/A | Submission format example |

---

## Detailed File Specifications

### 1. train.csv

**Description:** Training data with daily unit sales by store and item.

**Schema:**
| Column | Data Type | Description | Example Values | Missing % |
|--------|-----------|-------------|----------------|-----------|
| id | int64 | Unique row identifier | 0, 1, 2, ... | 0% |
| date | object (string) | Transaction date | "2013-01-01" | 0% |
| store_nbr | int64 | Store identifier (1-54) | 1, 2, 3, ... | 0% |
| item_nbr | int64 | Item identifier | 96995, 99197, ... | 0% |
| unit_sales | float64 | Number of units sold | 7.0, -2.0, 15.5 | 0% |
| onpromotion | object (bool/NaN) | Promotion flag | True, False, NaN | ~16% |

**Key Characteristics:**
- **Total rows**: ~125 million (exact count TBD)
- **Temporal range**: 2013-01-01 to 2017-08-15 (1,688 days)
- **Unique stores**: 54
- **Unique items**: ~4,100
- **Negative values**: unit_sales can be negative (product returns)
- **Float values**: unit_sales can be decimal (e.g., 1.5 kg of cheese)
- **Missing dates**: Some store-item-date combinations absent (represent zero sales)

**Data Quality Notes:**
- Approximately 16% of onpromotion values are NaN
- Negative unit_sales values represent returns (need handling decision)
- File size (~479 MB) requires Dask for efficient loading
- Date column stored as string (requires pd.to_datetime conversion)

**Example Rows:**
```
id,date,store_nbr,item_nbr,unit_sales,onpromotion
0,2013-01-01,25,103665,7.0,
1,2013-01-01,25,105574,1.0,
2,2013-01-01,25,105575,2.0,
```

---

### 2. test.csv

**Description:** Test data for prediction submission (same schema as train, no unit_sales).

**Schema:**
| Column | Data Type | Description | Example Values | Missing % |
|--------|-----------|-------------|----------------|-----------|
| id | int64 | Unique row identifier | TBD | 0% |
| date | object (string) | Prediction date | "2017-08-16", ... | 0% |
| store_nbr | int64 | Store identifier | 1, 2, 3, ... | 0% |
| item_nbr | int64 | Item identifier | TBD | 0% |
| onpromotion | object (bool/NaN) | Promotion flag | True, False, NaN | TBD% |

**Key Characteristics:**
- **Total rows**: TBD (to be counted)
- **Temporal range**: 2017-08-16 to 2017-08-31 (16 days)
- **Purpose**: Generate forecasts for this date range
- **Note**: Our project focuses on Jan-Mar 2014, not this test period

---

### 3. stores.csv

**Description:** Store metadata including location and type.

**Schema:**
| Column | Data Type | Description | Example Values | Missing % |
|--------|-----------|-------------|----------------|-----------|
| store_nbr | int64 | Store identifier (primary key) | 1, 2, 3, ... | 0% |
| city | object (string) | City name | "Quito", "Guayaquil" | 0% |
| state | object (string) | State/province name | "Pichincha", "Guayas" | 0% |
| type | object (string) | Store type | "A", "B", "C", "D", "E" | 0% |
| cluster | int64 | Store cluster (grouping) | 1, 2, 3, ..., 17 | 0% |

**Key Characteristics:**
- **Total rows**: 54 stores
- **Unique cities**: TBD (to be counted)
- **Unique states**: TBD (to be counted, focus on "Guayas")
- **Store types**: 5 types (A, B, C, D, E)
- **Clusters**: 17 clusters (grouping of similar stores)

**Guayas Region Filter:**
- **Guayas stores**: TBD (to be identified in Day 2)
- **Purpose**: Filter train.csv to Guayas stores only

**Example Rows:**
```
store_nbr,city,state,type,cluster
1,Quito,Pichincha,D,13
2,Quito,Pichincha,D,13
3,Quito,Pichincha,D,8
```

---

### 4. items.csv

**Description:** Product metadata including family and perishable flag.

**Schema:**
| Column | Data Type | Description | Example Values | Missing % |
|--------|-----------|-------------|----------------|-----------|
| item_nbr | int64 | Item identifier (primary key) | 96995, 99197, ... | 0% |
| family | object (string) | Product family/category | "GROCERY I", "BEVERAGES" | 0% |
| class | int64 | Item class | 1065, 1066, ... | 0% |
| perishable | int64 | Perishable flag (0 or 1) | 0, 1 | 0% |

**Key Characteristics:**
- **Total rows**: 4,100 items
- **Unique families**: 33 product families
- **Perishable items**: TBD% (weight 1.25 in NWRMSLE)
- **Non-perishable items**: TBD% (weight 1.0 in NWRMSLE)

**Top-3 Families Filter:**
- **Selection criteria**: Top-3 families by unique item count
- **Top-3 families**: TBD (to be identified in Day 2)
- **Purpose**: Reduce complexity to manageable scope

**Example Rows:**
```
item_nbr,family,class,perishable
96995,GROCERY I,1093,0
99197,GROCERY I,1067,0
103501,CLEANING,3008,0
```

---

### 5. oil.csv

**Description:** Daily oil price data (Ecuador's economy is oil-dependent).

**Schema:**
| Column | Data Type | Description | Example Values | Missing % |
|--------|-----------|-------------|----------------|-----------|
| date | object (string) | Date | "2013-01-01" | 0% |
| dcoilwtico | float64 | Daily oil price (WTI) | 93.14, 94.72, ... | TBD% |

**Key Characteristics:**
- **Total rows**: 1,218 days
- **Temporal range**: 2013-01-01 to 2017-08-31
- **Missing dates**: Some days may be missing (weekends, holidays)
- **Missing values**: dcoilwtico may have NaN (forward fill strategy)

**Usage Notes:**
- Investigate correlation with unit_sales
- Expected weak correlation (verify in Day 5)
- May exclude from features if correlation <0.1

**Example Rows:**
```
date,dcoilwtico
2013-01-01,93.14
2013-01-02,93.14
2013-01-03,92.97
```

---

### 6. holidays_events.csv

**Description:** Holiday and event calendar with transfer/bridge day handling.

**Schema:**
| Column | Data Type | Description | Example Values | Missing % |
|--------|-----------|-------------|----------------|-----------|
| date | object (string) | Holiday date | "2012-03-02" | 0% |
| type | object (string) | Holiday type | "Holiday", "Transfer", "Bridge", "Work Day", "Event", "Additional" | 0% |
| locale | object (string) | Geographic scope | "National", "Regional", "Local" | 0% |
| locale_name | object (string) | Specific region/city | "Ecuador", "Guayas", "Guayaquil" | 0% |
| description | object (string) | Holiday name | "Dia de la Madre", "Navidad" | 0% |
| transferred | bool | Transfer flag | True, False | 0% |

**Key Characteristics:**
- **Total rows**: 350 holiday records
- **Temporal range**: 2012-03-02 to 2017-12-26
- **Holiday types**: 6 types (see schema above)
- **Transfer logic**: Holidays moved to different dates by government

**Important Notes:**
- **Transfer days**: Official date vs actual celebration date differ
- **Bridge days**: Extra days added to extend holidays
- **Work days**: Makeup days for bridge days
- **Focus**: Filter to "Guayas" region holidays

**Example Rows:**
```
date,type,locale,locale_name,description,transferred
2012-03-02,Holiday,Local,Manta,Fundacion de Manta,False
2012-03-23,Holiday,National,Ecuador,Viernes Santo,False
```

---

### 7. transactions.csv

**Description:** Daily transaction counts per store (total number of sales transactions).

**Schema:**
| Column | Data Type | Description | Example Values | Missing % |
|--------|-----------|-------------|----------------|-----------|
| date | object (string) | Transaction date | "2013-01-01" | 0% |
| store_nbr | int64 | Store identifier | 1, 2, 3, ... | 0% |
| transactions | int64 | Number of transactions | 2111, 1507, ... | 0% |

**Key Characteristics:**
- **Total rows**: 83,488 rows
- **Temporal range**: 2013-01-01 to 2017-08-15 (matches train.csv)
- **Unique stores**: 54
- **Purpose**: Can be used as proxy for store traffic/activity

**Usage Notes:**
- Can create "transactions per unit_sales" ratio (traffic efficiency)
- May correlate with high-traffic days (weekends, paydays)

**Example Rows:**
```
date,store_nbr,transactions
2013-01-01,25,770
2013-01-02,1,2111
2013-01-02,2,2358
```

---

### 8. sample_submission.csv

**Description:** Example submission format for Kaggle competition.

**Schema:**
| Column | Data Type | Description | Example Values |
|--------|-----------|-------------|----------------|
| id | int64 | Row identifier (matches test.csv) | TBD |
| unit_sales | float64 | Predicted sales | TBD |

**Key Characteristics:**
- **Purpose**: Reference format only (not used in our project)
- **Our forecast format**: Will create custom CSV for Jan-Mar 2014

---

## Data Quality Summary

### Overall Assessment

| Aspect | Status | Notes |
|--------|--------|-------|
| Completeness | WARNING | train.csv missing 16% onpromotion values, oil.csv has gaps |
| Consistency | OK | Schemas align across files (store_nbr, item_nbr, date) |
| Temporal Alignment | OK | train.csv and transactions.csv cover same period |
| File Sizes | WARNING | train.csv (479 MB) requires Dask for loading |
| Data Types | OK | Mostly correct, need datetime conversion for date columns |

### Key Issues Identified

1. **train.csv onpromotion NaN (16%)**: Decision needed on imputation strategy
2. **train.csv negative unit_sales**: Returns vs sales handling decision needed
3. **train.csv missing dates**: Zero-fill strategy required for complete calendar
4. **oil.csv gaps**: Forward fill or interpolation needed
5. **Large file size**: Dask required for train.csv loading

### Next Steps

- [ ] Load all files and verify row counts
- [ ] Convert date columns to datetime
- [ ] Quantify exact missing value percentages
- [ ] Identify Guayas stores (filter stores.csv)
- [ ] Identify top-3 product families (count items per family)
- [ ] Document findings in this inventory
- [ ] Update decision log with data quality decisions

---

## Guayas Region Scope

### Filtering Criteria
- **Region**: Guayas state only
- **Stores**: TBD stores (to be identified from stores.csv WHERE state='Guayas')
- **Products**: Top-3 families by item count (to be calculated)
- **Sample size**: 300,000 rows (random sample from filtered train.csv)

### Expected Reduction
- **Before filtering**: ~125M rows (full train.csv)
- **After Guayas filter**: TBD rows
- **After top-3 families filter**: TBD rows
- **After 300K sampling**: 300,000 rows (final dataset)

---

## References

- **Kaggle Competition**: https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting
- **Data Description**: https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data
- **Discussion Forum**: https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/discussion

---

**Last Updated:** [YYYY-MM-DD]  
**Status:** Template created - to be filled during Day 1 Part 3

---

**End of Data Inventory**

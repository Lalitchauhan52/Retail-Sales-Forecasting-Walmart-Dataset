# Methodology & Approach

## Problem Framing

This is a **multi-step time-series regression** task. For each (Store, Dept, Date) triplet in the test set, we must predict `Weekly_Sales`. The evaluation metric is **Weighted MAE (WMAE)** where holiday weeks are penalised 5× more than regular weeks.

## Data

| File | Description |
|------|-------------|
| `train.csv` | 421,570 weekly sales records (Feb 2010 → Oct 2012) |
| `test.csv` | Store-dept pairs to predict (Nov 2012 → Jul 2013) |
| `features.csv` | External: Temperature, Fuel, 5 MarkDowns, CPI, Unemployment |
| `stores.csv` | Store type (A/B/C) and square footage |

## Cleaning Decisions

| Decision | Rationale |
|----------|-----------|
| Remove negative `Weekly_Sales` | Returns/adjustments (~0.3% of rows); distort regression |
| Fill MarkDown NaN → 0 | MarkDowns only exist from mid-2011; 0 = "no promotion" |
| Forward-fill CPI / Unemployment per store | Slowly-changing signal; only a few gaps |

## Feature Engineering

### Calendar
- Raw: year, month, week, quarter, day_of_year
- Cyclical encodings `sin/cos(2π × month / 12)` and `sin/cos(2π × week / 52)` to avoid the artificial discontinuity at month/week boundaries (e.g., Dec ↔ Jan treated as far apart in raw numerics).

### Named Holidays
Four Walmart-specific "Super Holidays" each receive a binary flag:
- Super Bowl, Labor Day, Thanksgiving, New Year's

### Lag & Rolling Features
These are the single most important group of features:
- Lags: 1w, 2w, 4w, 52w (same week last year — very powerful for seasonal capture)
- Rolling mean: 4w, 12w
- Rolling std: 4w (captures dept-level volatility)

All lags are computed within `(Store, Dept)` groups to avoid data leakage across stores.

### MarkDown Aggregation
- `total_markdown` = sum of all 5 MarkDown columns (post-zero-fill)
- `has_markdown` = binary flag (any promotion active this week)

### Store Metadata
- `store_type_enc`: label-encoded A/B/C
- `Size`: raw square footage

## Validation Strategy

**Time-based split** — last 12 weeks of training data held out as validation. This mimics the real-world deployment scenario where the model predicts future weeks it has never seen.

Note: `TimeSeriesSplit` from scikit-learn is used for cross-validation inside the notebook to get a more robust performance estimate.

## Model Comparison

| Model | Notes |
|-------|-------|
| Ridge | Linear baseline — establishes minimum bar |
| Random Forest | Non-linear, robust; limited by tree depth |
| **XGBoost** | **Best single model** — gradient boosting with regularisation |
| LightGBM | Near-XGBoost, ~3× faster on large data |
| Ensemble | Weighted avg of RF + XGB + LGB; marginal improvement over XGB alone |

## WMAE Interpretation

$$\text{WMAE} = \frac{\sum_i w_i |y_i - \hat{y}_i|}{\sum_i w_i}, \quad w_i = \begin{cases} 5 & \text{holiday week} \\ 1 & \text{otherwise} \end{cases}$$

A score of $2,000 means on average the model is off by $2,000/week per (Store, Dept) pair, with holiday-week errors weighted 5×.

## Limitations & Future Work

1. **Lag features for test set** — test weeks immediately follow training; we bridge using "last known" values. A production system needs a rolling inference loop.
2. **Global model** — a single model for all 45 × 81 store-dept pairs. Hierarchical models (per-store or per-type) may improve outlier depts.
3. **No external calendar** — school holidays, regional events, weather anomalies could be added.
4. **Deployment** — a Streamlit dashboard or FastAPI endpoint would make the model interactive for planners.

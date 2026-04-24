# Retail Sales Forecasting — Walmart Dataset

> **End-to-End Machine Learning Pipeline** | Predicting weekly department-level sales across 45 Walmart stores using XGBoost with advanced feature engineering, WMAE-weighted evaluation, and multi-model comparison.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Project Summary

| | |
|---|---|
| **Dataset** | [Walmart Store Sales Forecasting (Kaggle)](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) |
| **Records** | 421,570 training rows |
| **Stores** | 45 stores, 3 types (A / B / C) |
| **Departments** | 81 departments |
| **Target** | `Weekly_Sales` (USD) |
| **Key Metric** | Weighted MAE (holiday weeks × 5) |

---

## Project Structure

```
retail-sales-forecasting/
├── data/
│   ├── raw/                   # Original Kaggle CSVs (not tracked by git)
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── features.csv
│   │   └── stores.csv
│   └── processed/             # Cleaned & merged datasets
├── notebooks/
│   └── retail_sales_forecasting.ipynb   # Main end-to-end notebook
├── src/
│   ├── __init__.py
│   ├── features.py            # Feature engineering functions
│   ├── model.py               # Training & evaluation logic
│   └── utils.py               # WMAE metric, helpers
├── models/
│   └── xgboost_model.pkl      # Saved trained model
├── outputs/
│   ├── figures/               # EDA & evaluation plots
│   └── submissions/           # Kaggle submission CSVs
├── tests/
│   └── test_features.py       # Unit tests for feature pipeline
├── docs/
│   └── approach.md            # Detailed methodology notes
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Pipeline Overview

```
Raw CSVs  →  Data Cleaning  →  Feature Engineering  →  Model Training  →  Evaluation  →  Submission
```

### Feature Engineering Highlights
- **Calendar features** — year, month, week, quarter, day-of-year + sine/cosine cyclical encodings
- **Named holiday flags** — Super Bowl, Labor Day, Thanksgiving, New Year's
- **Lag features** — 1-week, 2-week, 4-week, 52-week lags per store-dept pair
- **Rolling statistics** — 4-week & 12-week rolling mean; 4-week rolling std
- **MarkDown aggregations** — total markdown, has-markdown binary flag
- **Store metadata** — type (A/B/C) label encoded, store size

### Models Benchmarked
| Model | Notes |
|-------|-------|
| Ridge Regression | Baseline linear model |
| Random Forest | Ensemble, non-linear baseline |
| **XGBoost** | **Best single model** |
| LightGBM | Near-XGBoost performance, faster |
| Ensemble | Weighted average of top models |

---

## Key Findings

| Finding | Implication |
|---------|------------|
| Lag & rolling features dominate importance | Strong autocorrelation — same-week-last-year is very powerful |
| Thanksgiving & Christmas drive ≥2× normal sales | Nov–Dec inventory planning is critical |
| Type A stores are easiest to predict | Smaller/niche stores may need specialist sub-models |
| Departments 94 & 39 remain hard to predict | High variance depts may benefit from separate models |

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/retail-sales-forecasting.git
cd retail-sales-forecasting
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the data
Download the dataset from [Kaggle](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data) and place the files in `data/raw/`:
- `train.csv`
- `test.csv`
- `features.csv`
- `stores.csv`

### 4. Run the notebook
```bash
jupyter notebook notebooks/retail_sales_forecasting.ipynb
```

---

## Running Tests

```bash
pytest tests/
```

---

## Next Steps

1. **Hierarchical forecasting** — model at store level, disaggregate to dept
2. **Prophet / NeuralProphet** for explicit seasonality decomposition
3. **Bayesian hyperparameter tuning** with Optuna
4. **Walk-forward cross-validation** with multiple time splits
5. **External data** — local weather, competitor promotions, regional events
6. **Streamlit / FastAPI deployment** — interactive forecasting dashboard

---

## Author

**Lalit Chauhan**

---

<p>Co-Author  <a href="https://github.com/Manthan-gupta-21906">Manthan Gupta</a></p>

---

⭐️ If you like this project, please give it a star on GitHub!

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

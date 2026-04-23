"""
features.py
-----------
All feature engineering for the Walmart retail sales forecasting pipeline.
Mirrors exactly what the notebook does, extracted into reusable functions.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils import SUPER_BOWL, LABOR_DAY, THANKSGIVING, NEW_YEAR

# ── Constants ──────────────────────────────────────────────────────────────────

MARKDOWN_COLS = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

FEATURE_COLS = [
    'Store', 'Dept', 'store_type_enc', 'Size',
    'year', 'month', 'week', 'quarter', 'day_of_year',
    'month_sin', 'month_cos', 'week_sin', 'week_cos',
    'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'IsHoliday', 'is_super_bowl', 'is_labor_day', 'is_thanksgiving', 'is_new_year',
    'lag_1w', 'lag_2w', 'lag_4w', 'lag_52w',
    'roll_mean_4w', 'roll_mean_12w', 'roll_std_4w',
    'total_markdown', 'has_markdown',
    *MARKDOWN_COLS,
]

# Validation split date (matches notebook)
SPLIT_DATE = pd.Timestamp('2012-09-01')


# ── Data loading ───────────────────────────────────────────────────────────────

def load_raw(data_dir) -> tuple:
    """
    Load the 4 raw Kaggle CSVs from data_dir.

    Parameters
    ----------
    data_dir : str or Path — folder containing train.csv, test.csv,
                             features.csv, stores.csv

    Returns
    -------
    train, test, features, stores : DataFrames with parsed dates
    """
    from pathlib import Path
    data_dir = Path(data_dir)

    train    = pd.read_csv(data_dir / 'train.csv')
    test     = pd.read_csv(data_dir / 'test.csv')
    features = pd.read_csv(data_dir / 'features.csv')
    stores   = pd.read_csv(data_dir / 'stores.csv')

    for df in [train, test, features]:
        df['Date'] = pd.to_datetime(df['Date'])

    return train, test, features, stores


def merge_all(train, test, features, stores) -> tuple:
    """
    Merge train/test with features and stores tables.
    Drops the duplicate IsHoliday column produced by the features join.

    Returns
    -------
    train_df, test_df : merged DataFrames
    """
    def _merge(df):
        out = df.merge(features, on=['Store', 'Date'], how='left', suffixes=('', '_feat'))
        out = out.merge(stores, on='Store', how='left')
        if 'IsHoliday_feat' in out.columns:
            out.drop(columns=['IsHoliday_feat'], inplace=True)
        return out

    return _merge(train), _merge(test)


# ── Cleaning ───────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame, remove_negative_sales: bool = True) -> pd.DataFrame:
    """
    Clean a merged train DataFrame:
      - Remove negative Weekly_Sales rows
      - Fill MarkDown NaNs with 0 and clip negatives
      - Forward-fill CPI and Unemployment per store
    """
    df = df.copy()

    if remove_negative_sales and 'Weekly_Sales' in df.columns:
        neg_mask = df['Weekly_Sales'] < 0
        df = df[~neg_mask].copy()

    # MarkDowns only available from mid-2011; 0 = no promotion
    df[MARKDOWN_COLS] = df[MARKDOWN_COLS].fillna(0).clip(lower=0)

    # Slowly-changing economic signals — forward fill per store
    df = df.sort_values(['Store', 'Date'])
    df[['CPI', 'Unemployment']] = (
        df.groupby('Store')[['CPI', 'Unemployment']]
        .transform(lambda s: s.ffill().bfill())
    )

    return df


# ── Feature engineering ────────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year, month, week, quarter, day-of-year and sin/cos cyclical encodings."""
    df = df.copy()
    df['year']        = df['Date'].dt.year
    df['month']       = df['Date'].dt.month
    df['week']        = df['Date'].dt.isocalendar().week.astype(int)
    df['quarter']     = df['Date'].dt.quarter
    df['day_of_year'] = df['Date'].dt.day_of_year

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin']  = np.sin(2 * np.pi * df['week']  / 52)
    df['week_cos']  = np.cos(2 * np.pi * df['week']  / 52)
    return df


def add_holiday_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary flags for each named Walmart super-holiday."""
    df = df.copy()
    date_str = df['Date'].dt.strftime('%Y-%m-%d')
    df['is_super_bowl']   = date_str.isin(SUPER_BOWL).astype(int)
    df['is_labor_day']    = date_str.isin(LABOR_DAY).astype(int)
    df['is_thanksgiving'] = date_str.isin(THANKSGIVING).astype(int)
    df['is_new_year']     = date_str.isin(NEW_YEAR).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag & rolling features per (Store, Dept) group.

    Lags    : 1w, 2w, 4w, 52w
    Rolling : 4w mean, 12w mean, 4w std  (all shifted by 1 to avoid leakage)
    """
    df = df.copy().sort_values(['Store', 'Dept', 'Date'])
    grp = df.groupby(['Store', 'Dept'])['Weekly_Sales']

    for lag in [1, 2, 4, 52]:
        df[f'lag_{lag}w'] = grp.shift(lag)

    shifted = grp.shift(1)
    df['roll_mean_4w']  = shifted.transform(lambda s: s.rolling(4,  min_periods=1).mean())
    df['roll_mean_12w'] = shifted.transform(lambda s: s.rolling(12, min_periods=1).mean())
    df['roll_std_4w']   = shifted.transform(lambda s: s.rolling(4,  min_periods=1).std().fillna(0))

    return df


def add_markdown_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate MarkDown columns into summary features."""
    df = df.copy()
    df['total_markdown'] = df[MARKDOWN_COLS].sum(axis=1)
    df['has_markdown']   = (df['total_markdown'] > 0).astype(int)
    return df


def encode_store_type(df: pd.DataFrame,
                      le: LabelEncoder = None) -> tuple:
    """
    Label-encode the store Type column (A/B/C → 0/1/2).

    Parameters
    ----------
    df : DataFrame with a 'Type' column
    le : fitted LabelEncoder — pass None to fit a new one on this data

    Returns
    -------
    df, le
    """
    df = df.copy()
    if le is None:
        le = LabelEncoder()
        df['store_type_enc'] = le.fit_transform(df['Type'])
    else:
        df['store_type_enc'] = le.transform(df['Type'])
    return df, le


# ── Full pipeline ──────────────────────────────────────────────────────────────

def build_train_features(train_df: pd.DataFrame) -> tuple:
    """
    Run the complete feature engineering pipeline on the merged train DataFrame.

    Steps: clean → sort → calendar → holidays → lags → rolling →
           markdowns → encode store type → drop NaN lags

    Returns
    -------
    df   : fully engineered DataFrame
    le   : fitted LabelEncoder (reuse on test set)
    """
    df = clean(train_df, remove_negative_sales=True)
    df = df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_holiday_flags(df)
    df = add_lag_features(df)
    df = add_markdown_features(df)
    df, le = encode_store_type(df, le=None)
    df.dropna(inplace=True)
    return df, le


def build_test_features(test_df: pd.DataFrame,
                        train_df_clean: pd.DataFrame,
                        le: LabelEncoder) -> pd.DataFrame:
    """
    Apply the same feature engineering to the test set.

    Lag features are bridged from the last known training values per
    (Store, Dept) — because test weeks immediately follow the training window.

    Parameters
    ----------
    test_df        : merged test DataFrame (test + features + stores)
    train_df_clean : the engineered training DataFrame (used to extract last-known lags)
    le             : fitted LabelEncoder from build_train_features()

    Returns
    -------
    test_df : engineered test DataFrame aligned to FEATURE_COLS
    """
    df = test_df.copy()

    # Economic indicators
    df[MARKDOWN_COLS] = df[MARKDOWN_COLS].fillna(0).clip(lower=0)
    df = df.sort_values(['Store', 'Date'])
    df[['CPI', 'Unemployment']] = (
        df.groupby('Store')[['CPI', 'Unemployment']]
        .transform(lambda s: s.ffill().bfill())
    )

    # Calendar & holidays
    df = add_calendar_features(df)
    df = add_holiday_flags(df)
    df = add_markdown_features(df)

    # Bridge lag features from last known training values
    last_known = (
        train_df_clean.groupby(['Store', 'Dept'])
        .agg(
            lag_1w  =('Weekly_Sales', lambda x: x.iloc[-1]),
            lag_2w  =('Weekly_Sales', lambda x: x.iloc[-2]  if len(x) >= 2  else x.iloc[-1]),
            lag_4w  =('Weekly_Sales', lambda x: x.iloc[-4]  if len(x) >= 4  else x.iloc[-1]),
            lag_52w =('Weekly_Sales', lambda x: x.iloc[-52] if len(x) >= 52 else x.iloc[-1]),
            roll_mean_4w  =('Weekly_Sales', lambda x: x.tail(4).mean()),
            roll_mean_12w =('Weekly_Sales', lambda x: x.tail(12).mean()),
            roll_std_4w   =('Weekly_Sales', lambda x: x.tail(4).std() if len(x) >= 2 else 0),
        )
        .reset_index()
    )
    df = df.merge(last_known, on=['Store', 'Dept'], how='left')

    lag_cols = ['lag_1w', 'lag_2w', 'lag_4w', 'lag_52w',
                'roll_mean_4w', 'roll_mean_12w', 'roll_std_4w']
    df[lag_cols] = df[lag_cols].fillna(0)

    # Encode store type using the fitted encoder
    df, _ = encode_store_type(df, le=le)

    # Ensure all feature columns exist (fill missing with 0)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    return df

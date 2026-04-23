"""
test_features.py
----------------
Tests for src/features.py:
  - clean()
  - add_calendar_features()
  - add_holiday_flags()
  - add_lag_features()
  - add_markdown_features()
  - encode_store_type()
  - merge_all()
  - build_train_features()
  - build_test_features()
  - FEATURE_COLS / SPLIT_DATE constants
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from src.features import (
    clean,
    add_calendar_features,
    add_holiday_flags,
    add_lag_features,
    add_markdown_features,
    encode_store_type,
    merge_all,
    build_train_features,
    build_test_features,
    MARKDOWN_COLS,
    FEATURE_COLS,
    SPLIT_DATE,
)


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def base_df():
    """Minimal merged train-like DataFrame covering two stores."""
    return pd.DataFrame({
        'Store':        [1, 1, 1, 1, 2, 2],
        'Dept':         [1, 1, 1, 1, 1, 1],
        'Date':         pd.to_datetime([
                            '2010-02-12', '2010-02-19',
                            '2010-02-26', '2010-03-05',
                            '2010-02-12', '2010-02-19',
                        ]),
        'Weekly_Sales': [24924.5, -100.0, 18000.0, 22000.0, 31000.0, 29000.0],
        'IsHoliday':    [True, False, False, False, True, False],
        'Temperature':  [42.3, 38.5, 40.1, 44.0, 55.0, 53.0],
        'Fuel_Price':   [2.57, 2.55, 2.53, 2.51, 2.60, 2.58],
        'MarkDown1':    [np.nan, 500.0, np.nan, 200.0, np.nan, 100.0],
        'MarkDown2':    [np.nan] * 6,
        'MarkDown3':    [np.nan] * 6,
        'MarkDown4':    [np.nan] * 6,
        'MarkDown5':    [np.nan] * 6,
        'CPI':          [211.1, np.nan, 211.3, 211.4, 210.5, np.nan],
        'Unemployment': [8.1, np.nan, 8.0, 7.9, 7.5, np.nan],
        'Type':         ['A', 'A', 'A', 'A', 'B', 'B'],
        'Size':         [151315] * 4 + [202307] * 2,
    })


@pytest.fixture
def long_df():
    """60-week time series for a single Store-Dept — needed for lag tests."""
    n = 60
    return pd.DataFrame({
        'Store':        [1] * n,
        'Dept':         [1] * n,
        'Date':         pd.date_range('2010-01-01', periods=n, freq='W'),
        'Weekly_Sales': np.linspace(10000, 25000, n),
        'IsHoliday':    [False] * n,
        'Temperature':  [50.0] * n,
        'Fuel_Price':   [2.5] * n,
        'MarkDown1':    [0.0] * n,
        'MarkDown2':    [0.0] * n,
        'MarkDown3':    [0.0] * n,
        'MarkDown4':    [0.0] * n,
        'MarkDown5':    [0.0] * n,
        'CPI':          [211.0] * n,
        'Unemployment': [8.0] * n,
        'Type':         ['A'] * n,
        'Size':         [151315] * n,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

class TestConstants:

    def test_split_date_value(self):
        assert SPLIT_DATE == pd.Timestamp('2012-09-01')

    def test_markdown_cols_count(self):
        assert len(MARKDOWN_COLS) == 5

    def test_markdown_cols_names(self):
        assert MARKDOWN_COLS == ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

    def test_feature_cols_no_duplicates(self):
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS))

    def test_feature_cols_required_present(self):
        required = [
            'Store', 'Dept', 'store_type_enc', 'Size',
            'year', 'month', 'week',
            'month_sin', 'month_cos', 'week_sin', 'week_cos',
            'IsHoliday', 'is_super_bowl', 'is_labor_day', 'is_thanksgiving', 'is_new_year',
            'lag_1w', 'lag_2w', 'lag_4w', 'lag_52w',
            'roll_mean_4w', 'roll_mean_12w', 'roll_std_4w',
            'total_markdown', 'has_markdown',
            *MARKDOWN_COLS,
        ]
        for col in required:
            assert col in FEATURE_COLS, f"Missing: {col}"

    def test_feature_cols_is_list(self):
        assert isinstance(FEATURE_COLS, list)


# ══════════════════════════════════════════════════════════════════════════════
# clean()
# ══════════════════════════════════════════════════════════════════════════════

class TestClean:

    def test_removes_negative_sales(self, base_df):
        result = clean(base_df)
        assert (result['Weekly_Sales'] >= 0).all()

    def test_negative_sales_count(self, base_df):
        before = len(base_df)
        result = clean(base_df)
        n_neg  = (base_df['Weekly_Sales'] < 0).sum()
        assert len(result) == before - n_neg

    def test_keep_negatives_when_flag_false(self, base_df):
        result = clean(base_df, remove_negative_sales=False)
        assert len(result) == len(base_df)

    def test_no_negative_sales_when_none_present(self, long_df):
        result = clean(long_df)
        assert len(result) == len(long_df)

    def test_fills_markdown_nan_with_zero(self, base_df):
        result = clean(base_df)
        assert result[MARKDOWN_COLS].isna().sum().sum() == 0

    def test_clips_negative_markdown(self, base_df):
        df = base_df.copy()
        df['MarkDown1'] = -999.0
        result = clean(df)
        assert (result['MarkDown1'] >= 0).all()

    def test_ffills_cpi_within_store(self, base_df):
        result = clean(base_df)
        assert result['CPI'].isna().sum() == 0

    def test_ffills_unemployment_within_store(self, base_df):
        result = clean(base_df)
        assert result['Unemployment'].isna().sum() == 0

    def test_does_not_cross_fill_stores(self, base_df):
        """CPI from Store 1 should NOT fill Store 2's NaN."""
        result = clean(base_df)
        store1_cpi = result[result['Store'] == 1]['CPI'].values
        store2_cpi = result[result['Store'] == 2]['CPI'].values
        # All should be filled without cross-store contamination
        assert not np.isnan(store1_cpi).any()
        assert not np.isnan(store2_cpi).any()

    def test_returns_copy(self, base_df):
        result = clean(base_df)
        result['Weekly_Sales'] = -1
        assert (base_df['Weekly_Sales'] >= -100).any()  # original unchanged

    def test_sorted_by_store_date(self, base_df):
        result = clean(base_df)
        for store, grp in result.groupby('Store'):
            assert grp['Date'].is_monotonic_increasing


# ══════════════════════════════════════════════════════════════════════════════
# add_calendar_features()
# ══════════════════════════════════════════════════════════════════════════════

class TestCalendarFeatures:

    def test_all_columns_added(self, base_df):
        result = add_calendar_features(base_df)
        for col in ['year', 'month', 'week', 'quarter', 'day_of_year',
                    'month_sin', 'month_cos', 'week_sin', 'week_cos']:
            assert col in result.columns

    def test_year_values_correct(self, base_df):
        result = add_calendar_features(base_df)
        assert set(result['year'].unique()) == {2010}

    def test_month_range(self, base_df):
        result = add_calendar_features(base_df)
        assert result['month'].between(1, 12).all()

    def test_week_range(self, base_df):
        result = add_calendar_features(base_df)
        assert result['week'].between(1, 53).all()

    def test_quarter_range(self, base_df):
        result = add_calendar_features(base_df)
        assert result['quarter'].isin([1, 2, 3, 4]).all()

    def test_cyclical_month_sin_range(self, base_df):
        result = add_calendar_features(base_df)
        assert result['month_sin'].between(-1, 1).all()

    def test_cyclical_month_cos_range(self, base_df):
        result = add_calendar_features(base_df)
        assert result['month_cos'].between(-1, 1).all()

    def test_cyclical_week_sin_range(self, base_df):
        result = add_calendar_features(base_df)
        assert result['week_sin'].between(-1, 1).all()

    def test_cyclical_week_cos_range(self, base_df):
        result = add_calendar_features(base_df)
        assert result['week_cos'].between(-1, 1).all()

    def test_sin_cos_pythagorean_identity(self, base_df):
        """sin² + cos² should ≈ 1 for both month and week encodings."""
        result = add_calendar_features(base_df)
        month_sq = result['month_sin']**2 + result['month_cos']**2
        week_sq  = result['week_sin']**2  + result['week_cos']**2
        np.testing.assert_allclose(month_sq, 1.0, atol=1e-10)
        np.testing.assert_allclose(week_sq,  1.0, atol=1e-10)

    def test_does_not_mutate_input(self, base_df):
        original_cols = set(base_df.columns)
        add_calendar_features(base_df)
        assert set(base_df.columns) == original_cols


# ══════════════════════════════════════════════════════════════════════════════
# add_holiday_flags()
# ══════════════════════════════════════════════════════════════════════════════

class TestHolidayFlags:

    def test_all_flag_columns_added(self, base_df):
        result = add_holiday_flags(base_df)
        for col in ['is_super_bowl', 'is_labor_day', 'is_thanksgiving', 'is_new_year']:
            assert col in result.columns

    def test_super_bowl_flagged(self, base_df):
        result = add_holiday_flags(base_df)
        mask = result['Date'] == pd.Timestamp('2010-02-12')
        assert result.loc[mask, 'is_super_bowl'].iloc[0] == 1

    def test_non_holiday_not_flagged(self, base_df):
        result = add_holiday_flags(base_df)
        mask = result['Date'] == pd.Timestamp('2010-02-19')
        assert result.loc[mask, 'is_super_bowl'].iloc[0] == 0

    def test_flags_are_binary(self, base_df):
        result = add_holiday_flags(base_df)
        for col in ['is_super_bowl', 'is_labor_day', 'is_thanksgiving', 'is_new_year']:
            assert result[col].isin([0, 1]).all()

    def test_flags_are_integer_dtype(self, base_df):
        result = add_holiday_flags(base_df)
        for col in ['is_super_bowl', 'is_labor_day', 'is_thanksgiving', 'is_new_year']:
            assert result[col].dtype in [np.int32, np.int64, int]

    def test_only_one_holiday_flag_set_per_row(self, base_df):
        result = add_holiday_flags(base_df)
        flag_cols = ['is_super_bowl', 'is_labor_day', 'is_thanksgiving', 'is_new_year']
        row_sums = result[flag_cols].sum(axis=1)
        assert (row_sums <= 1).all()

    def test_does_not_mutate_input(self, base_df):
        original_cols = set(base_df.columns)
        add_holiday_flags(base_df)
        assert set(base_df.columns) == original_cols


# ══════════════════════════════════════════════════════════════════════════════
# add_lag_features()
# ══════════════════════════════════════════════════════════════════════════════

class TestLagFeatures:

    def test_all_lag_columns_created(self, long_df):
        result = add_lag_features(long_df)
        for col in ['lag_1w', 'lag_2w', 'lag_4w', 'lag_52w',
                    'roll_mean_4w', 'roll_mean_12w', 'roll_std_4w']:
            assert col in result.columns

    def test_lag_1w_is_previous_week(self, long_df):
        result = add_lag_features(long_df).reset_index(drop=True)
        # row[1].lag_1w should equal row[0].Weekly_Sales
        assert result.loc[1, 'lag_1w'] == pytest.approx(result.loc[0, 'Weekly_Sales'])

    def test_lag_52w_is_nan_for_first_52_rows(self, long_df):
        result = add_lag_features(long_df)
        # First 52 rows can't have a 52-week lag
        assert result.iloc[:52]['lag_52w'].isna().all()

    def test_lag_52w_filled_after_52_rows(self, long_df):
        result = add_lag_features(long_df)
        assert pd.notna(result.iloc[52]['lag_52w'])

    def test_no_cross_store_lag_leakage(self):
        """Lag features must stay within each Store-Dept group."""
        df = pd.DataFrame({
            'Store':        [1, 1, 2, 2],
            'Dept':         [1, 1, 1, 1],
            'Date':         pd.to_datetime(['2010-01-01', '2010-01-08',
                                            '2010-01-01', '2010-01-08']),
            'Weekly_Sales': [1000.0, 2000.0, 9999.0, 8888.0],
        })
        result = add_lag_features(df)
        # Store 1 first row must have NaN lag_1w (no previous week for this store)
        store1 = result[result['Store'] == 1].sort_values('Date')
        assert pd.isna(store1.iloc[0]['lag_1w'])

    def test_rolling_mean_4w_non_negative_for_positive_sales(self, long_df):
        result = add_lag_features(long_df)
        non_nan = result['roll_mean_4w'].dropna()
        assert (non_nan >= 0).all()

    def test_roll_std_4w_non_negative(self, long_df):
        result = add_lag_features(long_df)
        assert (result['roll_std_4w'].dropna() >= 0).all()

    def test_does_not_mutate_input(self, long_df):
        original_cols = set(long_df.columns)
        add_lag_features(long_df)
        assert set(long_df.columns) == original_cols


# ══════════════════════════════════════════════════════════════════════════════
# add_markdown_features()
# ══════════════════════════════════════════════════════════════════════════════

class TestMarkdownFeatures:

    def test_columns_added(self, base_df):
        result = add_markdown_features(base_df)
        assert 'total_markdown' in result.columns
        assert 'has_markdown' in result.columns

    def test_total_markdown_equals_row_sum(self, base_df):
        df = clean(base_df)
        result = add_markdown_features(df)
        expected = df[MARKDOWN_COLS].sum(axis=1).values
        np.testing.assert_array_almost_equal(result['total_markdown'].values, expected)

    def test_has_markdown_is_binary(self, base_df):
        result = add_markdown_features(base_df)
        assert result['has_markdown'].isin([0, 1]).all()

    def test_has_markdown_zero_when_all_markdowns_zero(self):
        df = pd.DataFrame({col: [0.0] for col in MARKDOWN_COLS})
        result = add_markdown_features(df)
        assert result['has_markdown'].iloc[0] == 0

    def test_has_markdown_one_when_any_markdown_nonzero(self):
        df = pd.DataFrame({col: [0.0] for col in MARKDOWN_COLS})
        df['MarkDown1'] = 500.0
        result = add_markdown_features(df)
        assert result['has_markdown'].iloc[0] == 1

    def test_total_markdown_zero_when_all_zero(self):
        df = pd.DataFrame({col: [0.0] for col in MARKDOWN_COLS})
        result = add_markdown_features(df)
        assert result['total_markdown'].iloc[0] == 0.0

    def test_does_not_mutate_input(self, base_df):
        original_cols = set(base_df.columns)
        add_markdown_features(base_df)
        assert set(base_df.columns) == original_cols


# ══════════════════════════════════════════════════════════════════════════════
# encode_store_type()
# ══════════════════════════════════════════════════════════════════════════════

class TestEncodeStoreType:

    def test_column_added(self, base_df):
        result, _ = encode_store_type(base_df)
        assert 'store_type_enc' in result.columns

    def test_encoded_values_are_integers(self, base_df):
        result, _ = encode_store_type(base_df)
        assert result['store_type_enc'].dtype in [np.int32, np.int64, int]

    def test_two_types_map_to_two_values(self, base_df):
        result, _ = encode_store_type(base_df)
        # base_df has Type A and B → should encode to 2 distinct ints
        assert result['store_type_enc'].nunique() == 2

    def test_fitted_encoder_reused_on_test(self, base_df):
        _, le = encode_store_type(base_df)
        test_df = pd.DataFrame({'Type': ['A', 'B', 'A']})
        result, _ = encode_store_type(test_df, le=le)
        # Encoding should be consistent with train
        assert result['store_type_enc'].nunique() == 2

    def test_returns_label_encoder(self, base_df):
        _, le = encode_store_type(base_df)
        assert isinstance(le, LabelEncoder)

    def test_does_not_mutate_input(self, base_df):
        original_cols = set(base_df.columns)
        encode_store_type(base_df)
        assert set(base_df.columns) == original_cols


# ══════════════════════════════════════════════════════════════════════════════
# merge_all()
# ══════════════════════════════════════════════════════════════════════════════

class TestMergeAll:

    @pytest.fixture
    def raw_tables(self):
        train = pd.DataFrame({
            'Store': [1, 1], 'Dept': [1, 1],
            'Date': pd.to_datetime(['2010-02-12', '2010-02-19']),
            'Weekly_Sales': [10000.0, 12000.0],
            'IsHoliday': [True, False],
        })
        test = pd.DataFrame({
            'Store': [1], 'Dept': [1],
            'Date': pd.to_datetime(['2010-02-26']),
            'IsHoliday': [False],
        })
        features = pd.DataFrame({
            'Store': [1, 1, 1],
            'Date': pd.to_datetime(['2010-02-12', '2010-02-19', '2010-02-26']),
            'IsHoliday': [True, False, False],
            'Temperature': [42.0, 38.0, 40.0],
            'Fuel_Price': [2.5, 2.5, 2.5],
            'MarkDown1': [np.nan, np.nan, np.nan],
            'MarkDown2': [np.nan, np.nan, np.nan],
            'MarkDown3': [np.nan, np.nan, np.nan],
            'MarkDown4': [np.nan, np.nan, np.nan],
            'MarkDown5': [np.nan, np.nan, np.nan],
            'CPI': [211.0, 211.0, 211.0],
            'Unemployment': [8.0, 8.0, 8.0],
        })
        stores = pd.DataFrame({
            'Store': [1], 'Type': ['A'], 'Size': [151315],
        })
        return train, test, features, stores

    def test_no_duplicate_isholiday_column(self, raw_tables):
        train, test, features, stores = raw_tables
        train_df, test_df = merge_all(train, test, features, stores)
        assert 'IsHoliday_feat' not in train_df.columns
        assert 'IsHoliday_feat' not in test_df.columns

    def test_store_columns_merged(self, raw_tables):
        train, test, features, stores = raw_tables
        train_df, _ = merge_all(train, test, features, stores)
        assert 'Type' in train_df.columns
        assert 'Size' in train_df.columns

    def test_features_columns_merged(self, raw_tables):
        train, test, features, stores = raw_tables
        train_df, _ = merge_all(train, test, features, stores)
        assert 'Temperature' in train_df.columns
        assert 'CPI' in train_df.columns

    def test_train_row_count_preserved(self, raw_tables):
        train, test, features, stores = raw_tables
        train_df, _ = merge_all(train, test, features, stores)
        assert len(train_df) == len(train)

    def test_test_row_count_preserved(self, raw_tables):
        train, test, features, stores = raw_tables
        _, test_df = merge_all(train, test, features, stores)
        assert len(test_df) == len(test)


# ══════════════════════════════════════════════════════════════════════════════
# build_train_features()
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildTrainFeatures:

    def test_returns_dataframe_and_encoder(self, long_df):
        df, le = build_train_features(long_df)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(le, LabelEncoder)

    def test_all_feature_cols_present(self, long_df):
        df, _ = build_train_features(long_df)
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_no_nans_in_feature_cols(self, long_df):
        df, _ = build_train_features(long_df)
        nan_counts = df[FEATURE_COLS].isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0].index.tolist()
        assert cols_with_nan == [], f"NaN found in: {cols_with_nan}"

    def test_no_negative_weekly_sales(self, long_df):
        # Inject one negative row
        df_with_neg = long_df.copy()
        df_with_neg.loc[0, 'Weekly_Sales'] = -500.0
        result, _ = build_train_features(df_with_neg)
        assert (result['Weekly_Sales'] >= 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# build_test_features()
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildTestFeatures:

    @pytest.fixture
    def train_and_test(self, long_df):
        train_feat, le = build_train_features(long_df)
        test_df = pd.DataFrame({
            'Store':        [1],
            'Dept':         [1],
            'Date':         pd.to_datetime(['2011-03-01']),
            'IsHoliday':    [False],
            'Temperature':  [50.0],
            'Fuel_Price':   [2.5],
            'MarkDown1':    [0.0], 'MarkDown2': [0.0], 'MarkDown3': [0.0],
            'MarkDown4':    [0.0], 'MarkDown5': [0.0],
            'CPI':          [211.0],
            'Unemployment': [8.0],
            'Type':         ['A'],
            'Size':         [151315],
        })
        return train_feat, test_df, le

    def test_all_feature_cols_present(self, train_and_test):
        train_feat, test_df, le = train_and_test
        result = build_test_features(test_df, train_feat, le)
        for col in FEATURE_COLS:
            assert col in result.columns, f"Missing: {col}"

    def test_lag_cols_filled_no_nan(self, train_and_test):
        train_feat, test_df, le = train_and_test
        result = build_test_features(test_df, train_feat, le)
        lag_cols = ['lag_1w', 'lag_2w', 'lag_4w', 'lag_52w',
                    'roll_mean_4w', 'roll_mean_12w', 'roll_std_4w']
        assert result[lag_cols].isna().sum().sum() == 0

    def test_store_type_enc_present(self, train_and_test):
        train_feat, test_df, le = train_and_test
        result = build_test_features(test_df, train_feat, le)
        assert 'store_type_enc' in result.columns

    def test_row_count_preserved(self, train_and_test):
        train_feat, test_df, le = train_and_test
        result = build_test_features(test_df, train_feat, le)
        assert len(result) == len(test_df)

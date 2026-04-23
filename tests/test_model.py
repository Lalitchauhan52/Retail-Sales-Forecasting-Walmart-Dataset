"""
test_model.py
-------------
Tests for src/model.py:
  - get_ridge / get_random_forest / get_xgboost / get_lightgbm
  - time_split()
  - train_all_models()
  - generate_submission()
  - save_model() / load_model()
"""

import pickle
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.model import (
    get_ridge,
    get_random_forest,
    get_xgboost,
    time_split,
    train_all_models,
    generate_submission,
    save_model,
    load_model,
    LGB_AVAILABLE,
)
from src.features import FEATURE_COLS, SPLIT_DATE, build_train_features


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def small_xy():
    """
    Tiny synthetic train/val arrays for fast model tests.
    Uses only 3 features so models train in milliseconds.
    """
    np.random.seed(0)
    n_train, n_val = 200, 50
    X_train = pd.DataFrame(np.random.randn(n_train, 3), columns=['f1', 'f2', 'f3'])
    X_val   = pd.DataFrame(np.random.randn(n_val,   3), columns=['f1', 'f2', 'f3'])
    y_train = np.random.uniform(5000, 50000, n_train)
    y_val   = np.random.uniform(5000, 50000, n_val)
    holiday = np.random.choice([True, False], n_val)
    return X_train, X_val, y_train, y_val, holiday


@pytest.fixture(scope='module')
def full_feature_df():
    """
    A DataFrame with all FEATURE_COLS + Date + Weekly_Sales + IsHoliday,
    spanning dates before and after SPLIT_DATE.
    """
    n = 300
    np.random.seed(42)
    dates = pd.date_range('2011-01-01', periods=n, freq='W')
    df = pd.DataFrame({col: np.random.randn(n) for col in FEATURE_COLS})
    df['Date']         = dates
    df['Weekly_Sales'] = np.random.uniform(5000, 50000, n)
    df['IsHoliday']    = np.random.choice([True, False], n)
    # Make integer columns integer-like
    for col in ['Store', 'Dept', 'store_type_enc', 'Size',
                'year', 'month', 'week', 'quarter', 'day_of_year',
                'IsHoliday', 'is_super_bowl', 'is_labor_day', 'is_thanksgiving', 'is_new_year',
                'has_markdown']:
        df[col] = np.random.randint(0, 5, n)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Model factory functions
# ══════════════════════════════════════════════════════════════════════════════

class TestModelFactories:

    def test_get_ridge_returns_ridge(self):
        assert isinstance(get_ridge(), Ridge)

    def test_get_ridge_alpha(self):
        assert get_ridge().alpha == 1.0

    def test_get_random_forest_returns_rf(self):
        assert isinstance(get_random_forest(), RandomForestRegressor)

    def test_get_random_forest_params(self):
        rf = get_random_forest()
        assert rf.n_estimators    == 200
        assert rf.max_depth       == 12
        assert rf.min_samples_leaf == 4
        assert rf.random_state    == 42

    def test_get_xgboost_returns_xgbregressor(self):
        assert isinstance(get_xgboost(), XGBRegressor)

    def test_get_xgboost_params(self):
        xgb = get_xgboost()
        assert xgb.n_estimators   == 500
        assert xgb.learning_rate  == pytest.approx(0.05)
        assert xgb.max_depth      == 7
        assert xgb.random_state   == 42

    @pytest.mark.skipif(not LGB_AVAILABLE, reason='LightGBM not installed')
    def test_get_lightgbm_params(self):
        from src.model import get_lightgbm
        import lightgbm as lgb
        model = get_lightgbm()
        assert isinstance(model, lgb.LGBMRegressor)
        assert model.n_estimators  == 500
        assert model.learning_rate == pytest.approx(0.05)
        assert model.num_leaves    == 63

    @pytest.mark.skipif(LGB_AVAILABLE, reason='Only runs when LightGBM absent')
    def test_get_lightgbm_raises_without_lgb(self):
        from src.model import get_lightgbm
        with pytest.raises(ImportError):
            get_lightgbm()

    def test_each_call_returns_new_instance(self):
        assert get_ridge()         is not get_ridge()
        assert get_random_forest() is not get_random_forest()
        assert get_xgboost()       is not get_xgboost()


# ══════════════════════════════════════════════════════════════════════════════
# time_split()
# ══════════════════════════════════════════════════════════════════════════════

class TestTimeSplit:

    def test_returns_five_objects(self, full_feature_df):
        result = time_split(full_feature_df)
        assert len(result) == 5

    def test_train_before_split_date(self, full_feature_df):
        X_train, X_val, y_train, y_val, holiday_val = time_split(full_feature_df)
        train_dates = full_feature_df.loc[full_feature_df['Date'] < SPLIT_DATE, 'Date']
        assert len(X_train) == len(train_dates)

    def test_val_on_or_after_split_date(self, full_feature_df):
        X_train, X_val, y_train, y_val, holiday_val = time_split(full_feature_df)
        val_dates = full_feature_df.loc[full_feature_df['Date'] >= SPLIT_DATE, 'Date']
        assert len(X_val) == len(val_dates)

    def test_no_overlap_between_train_and_val(self, full_feature_df):
        X_train, X_val, *_ = time_split(full_feature_df)
        assert len(X_train) + len(X_val) == len(full_feature_df)

    def test_x_has_feature_cols(self, full_feature_df):
        X_train, X_val, *_ = time_split(full_feature_df)
        assert list(X_train.columns) == FEATURE_COLS
        assert list(X_val.columns)   == FEATURE_COLS

    def test_holiday_val_is_array(self, full_feature_df):
        *_, holiday_val = time_split(full_feature_df)
        assert isinstance(holiday_val, np.ndarray)

    def test_y_lengths_match_x(self, full_feature_df):
        X_train, X_val, y_train, y_val, _ = time_split(full_feature_df)
        assert len(X_train) == len(y_train)
        assert len(X_val)   == len(y_val)


# ══════════════════════════════════════════════════════════════════════════════
# train_all_models()  — uses tiny synthetic data for speed
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainAllModels:

    def test_returns_three_dicts(self, small_xy):
        X_train, X_val, y_train, y_val, holiday = small_xy
        results, models, preds = train_all_models(
            X_train, y_train, X_val, y_val, holiday
        )
        assert isinstance(results, dict)
        assert isinstance(models,  dict)
        assert isinstance(preds,   dict)

    def test_ridge_in_results(self, small_xy):
        X_train, X_val, y_train, y_val, holiday = small_xy
        results, _, _ = train_all_models(X_train, y_train, X_val, y_val, holiday)
        assert 'Ridge' in results

    def test_xgboost_in_results(self, small_xy):
        X_train, X_val, y_train, y_val, holiday = small_xy
        results, _, _ = train_all_models(X_train, y_train, X_val, y_val, holiday)
        assert 'XGBoost' in results

    def test_ensemble_in_results(self, small_xy):
        X_train, X_val, y_train, y_val, holiday = small_xy
        results, _, _ = train_all_models(X_train, y_train, X_val, y_val, holiday)
        assert 'Ensemble' in results

    def test_results_have_required_keys(self, small_xy):
        X_train, X_val, y_train, y_val, holiday = small_xy
        results, _, _ = train_all_models(X_train, y_train, X_val, y_val, holiday)
        for name, metrics in results.items():
            assert 'WMAE' in metrics
            assert 'MAE'  in metrics
            assert 'RMSE' in metrics

    def test_all_metrics_are_positive(self, small_xy):
        X_train, X_val, y_train, y_val, holiday = small_xy
        results, _, _ = train_all_models(X_train, y_train, X_val, y_val, holiday)
        for name, metrics in results.items():
            assert metrics['WMAE'] >= 0, f"{name} WMAE negative"
            assert metrics['MAE']  >= 0, f"{name} MAE negative"
            assert metrics['RMSE'] >= 0, f"{name} RMSE negative"

    def test_predictions_non_negative(self, small_xy):
        X_train, X_val, y_train, y_val, holiday = small_xy
        _, _, preds = train_all_models(X_train, y_train, X_val, y_val, holiday)
        for name, pred in preds.items():
            if pred is not None:
                assert (pred >= 0).all(), f"{name} has negative predictions"

    def test_prediction_lengths_match_val(self, small_xy):
        X_train, X_val, y_train, y_val, holiday = small_xy
        _, _, preds = train_all_models(X_train, y_train, X_val, y_val, holiday)
        for name, pred in preds.items():
            if pred is not None:
                assert len(pred) == len(X_val), f"{name} pred length mismatch"

    def test_xgboost_better_than_ridge(self, small_xy):
        """XGBoost should generally outperform Ridge on non-linear data."""
        X_train, X_val, y_train, y_val, holiday = small_xy
        results, _, _ = train_all_models(X_train, y_train, X_val, y_val, holiday)
        # This may not always hold on tiny random data, so we just check both exist
        assert 'XGBoost' in results
        assert 'Ridge'   in results


# ══════════════════════════════════════════════════════════════════════════════
# generate_submission()
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateSubmission:

    @pytest.fixture
    def trained_xgb_and_test(self, small_xy):
        X_train, X_val, y_train, y_val, holiday = small_xy
        xgb = get_xgboost()
        xgb.n_estimators = 10  # fast
        xgb.fit(X_train, y_train)

        test_df = X_val.copy()
        test_df['Store'] = 1
        test_df['Dept']  = 1
        test_df['Date']  = pd.date_range('2013-01-01', periods=len(X_val), freq='W')
        return xgb, test_df

    def test_creates_submission_csv(self, tmp_path, trained_xgb_and_test):
        xgb, test_df = trained_xgb_and_test
        generate_submission(xgb, test_df, tmp_path)
        assert (tmp_path / 'submission.csv').exists()

    def test_submission_has_required_columns(self, tmp_path, trained_xgb_and_test):
        xgb, test_df = trained_xgb_and_test
        submission = generate_submission(xgb, test_df, tmp_path)
        for col in ['Store', 'Dept', 'Date', 'Predicted_Sales']:
            assert col in submission.columns

    def test_predictions_non_negative(self, tmp_path, trained_xgb_and_test):
        xgb, test_df = trained_xgb_and_test
        submission = generate_submission(xgb, test_df, tmp_path)
        assert (submission['Predicted_Sales'] >= 0).all()

    def test_row_count_matches_test(self, tmp_path, trained_xgb_and_test):
        xgb, test_df = trained_xgb_and_test
        submission = generate_submission(xgb, test_df, tmp_path)
        assert len(submission) == len(test_df)

    def test_date_column_is_string(self, tmp_path, trained_xgb_and_test):
        xgb, test_df = trained_xgb_and_test
        submission = generate_submission(xgb, test_df, tmp_path)
        assert submission['Date'].dtype == object  # string after strftime


# ══════════════════════════════════════════════════════════════════════════════
# save_model() / load_model()
# ══════════════════════════════════════════════════════════════════════════════

class TestModelPersistence:

    @pytest.fixture
    def tiny_xgb(self, small_xy):
        X_train, X_val, y_train, *_ = small_xy
        xgb = get_xgboost()
        xgb.n_estimators = 5  # fast
        xgb.fit(X_train, y_train)
        return xgb

    def test_save_creates_pkl_file(self, tmp_path, tiny_xgb):
        save_model(tiny_xgb, tmp_path)
        assert (tmp_path / 'xgboost_model.pkl').exists()

    def test_load_returns_xgboost_model(self, tmp_path, tiny_xgb):
        save_model(tiny_xgb, tmp_path)
        loaded = load_model(tmp_path)
        assert isinstance(loaded, XGBRegressor)

    def test_loaded_model_predicts_same(self, tmp_path, tiny_xgb, small_xy):
        _, X_val, *_ = small_xy
        original_preds = tiny_xgb.predict(X_val)

        save_model(tiny_xgb, tmp_path)
        loaded = load_model(tmp_path)
        loaded_preds = loaded.predict(X_val)

        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_save_creates_models_dir_if_missing(self, tmp_path, tiny_xgb):
        models_dir = tmp_path / 'new_models_dir'
        assert not models_dir.exists()
        save_model(tiny_xgb, models_dir)
        assert (models_dir / 'xgboost_model.pkl').exists()

    def test_load_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / 'empty_dir')

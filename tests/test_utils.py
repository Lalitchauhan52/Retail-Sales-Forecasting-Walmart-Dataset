"""
test_utils.py
-------------
Tests for src/utils.py:
  - wmae()
  - label_holiday()
  - holiday date constants
  - get_project_root()
  - get_paths()
"""

import os
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from src.utils import (
    wmae,
    label_holiday,
    get_project_root,
    get_paths,
    SUPER_BOWL, LABOR_DAY, THANKSGIVING, NEW_YEAR,
    SUPER_BOWL_SET, LABOR_DAY_SET, THANKSGIVING_SET, NEW_YEAR_SET,
)


# ══════════════════════════════════════════════════════════════════════════════
# wmae()
# ══════════════════════════════════════════════════════════════════════════════

class TestWMAE:

    def test_perfect_predictions_return_zero(self):
        y = [100.0, 200.0, 300.0]
        assert wmae(y, y, [False, False, False]) == pytest.approx(0.0)

    def test_regular_weeks_weight_one(self):
        # single row, error=10, weight=1 → WMAE = 10/1 = 10
        assert wmae([100.0], [110.0], [False]) == pytest.approx(10.0)

    def test_holiday_weeks_weight_five(self):
        # single holiday row, error=10, weight=5 → WMAE = 50/5 = 10
        assert wmae([100.0], [110.0], [True]) == pytest.approx(10.0)

    def test_holiday_penalises_more_than_regular(self):
        y     = [100.0, 100.0]
        preds = [110.0, 110.0]
        regular = wmae(y, preds, [False, False])
        holiday = wmae(y, preds, [True,  False])
        assert holiday > regular

    def test_mixed_weights_formula(self):
        # y=[0,0], preds=[10,10], holidays=[True, False]
        # weights=[5,1], errors=[10,10]
        # WMAE = (5*10 + 1*10) / (5+1) = 60/6 = 10.0
        assert wmae([0.0, 0.0], [10.0, 10.0], [True, False]) == pytest.approx(10.0)

    def test_all_holiday_rows(self):
        # weights all 5 → equivalent to unweighted MAE
        y     = [100.0, 200.0]
        preds = [110.0, 220.0]
        result = wmae(y, preds, [True, True])
        assert result == pytest.approx(15.0)   # (10+20)/2 = 15

    def test_accepts_numpy_arrays(self):
        y     = np.array([100.0, 200.0])
        preds = np.array([100.0, 200.0])
        assert wmae(y, preds, np.array([False, False])) == pytest.approx(0.0)

    def test_accepts_pandas_series(self):
        y     = pd.Series([100.0, 200.0])
        preds = pd.Series([100.0, 200.0])
        assert wmae(y, preds, pd.Series([False, False])) == pytest.approx(0.0)

    def test_returns_float(self):
        result = wmae([100.0], [110.0], [False])
        assert isinstance(result, float)

    def test_single_row(self):
        assert wmae([500.0], [500.0], [True]) == pytest.approx(0.0)

    def test_large_dataset_consistent(self):
        np.random.seed(42)
        n = 10000
        y     = np.random.uniform(0, 100000, n)
        preds = y + np.random.normal(0, 1000, n)
        holiday = np.random.choice([True, False], n)
        result = wmae(y, preds, holiday)
        assert result > 0
        assert np.isfinite(result)


# ══════════════════════════════════════════════════════════════════════════════
# label_holiday()
# ══════════════════════════════════════════════════════════════════════════════

class TestLabelHoliday:

    # ── Super Bowl ────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("date_str", SUPER_BOWL)
    def test_all_super_bowl_dates(self, date_str):
        assert label_holiday(pd.Timestamp(date_str)) == 'Super Bowl'

    # ── Labor Day ─────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("date_str", LABOR_DAY)
    def test_all_labor_day_dates(self, date_str):
        assert label_holiday(pd.Timestamp(date_str)) == 'Labor Day'

    # ── Thanksgiving ─────────────────────────────────────────────────────────
    @pytest.mark.parametrize("date_str", THANKSGIVING)
    def test_all_thanksgiving_dates(self, date_str):
        assert label_holiday(pd.Timestamp(date_str)) == 'Thanksgiving'

    # ── New Year ─────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("date_str", NEW_YEAR)
    def test_all_new_year_dates(self, date_str):
        assert label_holiday(pd.Timestamp(date_str)) == "New Year's"

    # ── Regular ───────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("date_str", [
        '2010-01-01', '2010-03-15', '2010-06-20',
        '2011-04-01', '2011-08-08', '2012-05-05',
    ])
    def test_regular_dates(self, date_str):
        assert label_holiday(pd.Timestamp(date_str)) == 'Regular'

    def test_day_before_super_bowl_is_regular(self):
        assert label_holiday(pd.Timestamp('2010-02-11')) == 'Regular'

    def test_day_after_thanksgiving_is_regular(self):
        assert label_holiday(pd.Timestamp('2010-11-27')) == 'Regular'

    def test_returns_string(self):
        assert isinstance(label_holiday(pd.Timestamp('2010-02-12')), str)


# ══════════════════════════════════════════════════════════════════════════════
# Holiday constants
# ══════════════════════════════════════════════════════════════════════════════

class TestHolidayConstants:

    def test_super_bowl_has_three_years(self):
        assert len(SUPER_BOWL) == 3

    def test_labor_day_has_three_years(self):
        assert len(LABOR_DAY) == 3

    def test_thanksgiving_has_three_years(self):
        assert len(THANKSGIVING) == 3

    def test_new_year_has_three_years(self):
        assert len(NEW_YEAR) == 3

    def test_sets_match_lists(self):
        assert SUPER_BOWL_SET   == set(SUPER_BOWL)
        assert LABOR_DAY_SET    == set(LABOR_DAY)
        assert THANKSGIVING_SET == set(THANKSGIVING)
        assert NEW_YEAR_SET     == set(NEW_YEAR)

    def test_no_duplicate_dates_across_holidays(self):
        all_dates = SUPER_BOWL + LABOR_DAY + THANKSGIVING + NEW_YEAR
        assert len(all_dates) == len(set(all_dates)), "Duplicate holiday dates found"

    def test_dates_are_valid_strings(self):
        for date_str in SUPER_BOWL + LABOR_DAY + THANKSGIVING + NEW_YEAR:
            pd.Timestamp(date_str)   # raises if invalid


# ══════════════════════════════════════════════════════════════════════════════
# get_project_root() and get_paths()
# ══════════════════════════════════════════════════════════════════════════════

class TestPathHelpers:

    def test_get_project_root_returns_path(self):
        result = get_project_root()
        assert isinstance(result, Path)

    def test_get_project_root_from_notebooks_dir(self, tmp_path):
        notebooks_dir = tmp_path / 'notebooks'
        notebooks_dir.mkdir()
        with patch('os.getcwd', return_value=str(notebooks_dir)):
            root = get_project_root()
        assert root == tmp_path

    def test_get_project_root_from_project_root(self, tmp_path):
        with patch('os.getcwd', return_value=str(tmp_path)):
            root = get_project_root()
        assert root == tmp_path

    def test_get_paths_returns_all_keys(self, tmp_path):
        with patch('os.getcwd', return_value=str(tmp_path)):
            paths = get_paths()
        for key in ['root', 'data', 'figures', 'submissions', 'models']:
            assert key in paths

    def test_get_paths_creates_output_dirs(self, tmp_path):
        with patch('os.getcwd', return_value=str(tmp_path)):
            paths = get_paths()
        assert paths['figures'].exists()
        assert paths['submissions'].exists()
        assert paths['models'].exists()

    def test_get_paths_data_not_created(self, tmp_path):
        """data/raw/ should not be auto-created — user must download it."""
        with patch('os.getcwd', return_value=str(tmp_path)):
            paths = get_paths()
        assert not paths['data'].exists()

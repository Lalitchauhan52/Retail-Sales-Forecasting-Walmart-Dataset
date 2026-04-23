"""
utils.py
--------
Shared utilities: WMAE metric, holiday constants, path helpers.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


# ── Metric ─────────────────────────────────────────────────────────────────────

def wmae(y_true, y_pred, is_holiday) -> float:
    """
    Weighted Mean Absolute Error — Walmart competition metric.
    Holiday weeks are weighted 5×, regular weeks 1×.

    Parameters
    ----------
    y_true     : actual Weekly_Sales (array-like or Series)
    y_pred     : predicted values (array-like)
    is_holiday : boolean/int array (1 = holiday week)

    Returns
    -------
    float : WMAE score (lower is better)
    """
    weights = np.where(is_holiday, 5, 1)
    return float(np.sum(weights * np.abs(np.array(y_true) - np.array(y_pred))) / np.sum(weights))


# ── Holiday date constants ──────────────────────────────────────────────────────

SUPER_BOWL   = ['2010-02-12', '2011-02-11', '2012-02-10']
LABOR_DAY    = ['2010-09-10', '2011-09-09', '2012-09-07']
THANKSGIVING = ['2010-11-26', '2011-11-25', '2012-11-23']
NEW_YEAR     = ['2010-12-31', '2011-12-30', '2012-12-30']

SUPER_BOWL_SET   = set(SUPER_BOWL)
LABOR_DAY_SET    = set(LABOR_DAY)
THANKSGIVING_SET = set(THANKSGIVING)
NEW_YEAR_SET     = set(NEW_YEAR)


def label_holiday(date: pd.Timestamp) -> str:
    """Return a named holiday label for a given date, or 'Regular'."""
    ds = date.strftime('%Y-%m-%d')
    if ds in SUPER_BOWL_SET:    return 'Super Bowl'
    if ds in LABOR_DAY_SET:     return 'Labor Day'
    if ds in THANKSGIVING_SET:  return 'Thanksgiving'
    if ds in NEW_YEAR_SET:      return "New Year's"
    return 'Regular'


# ── Path helpers ───────────────────────────────────────────────────────────────

def get_project_root() -> Path:
    """
    Auto-detect project root whether running from project root or notebooks/.
    Works the same way as the path setup cell in the notebook.
    """
    cwd = Path(os.getcwd())
    return cwd.parent if cwd.name == 'notebooks' else cwd


def get_paths() -> dict:
    """
    Return a dict of all important project paths, creating output dirs if needed.

    Returns
    -------
    {
        'root':        Path,
        'data':        Path,   # data/raw/
        'figures':     Path,   # outputs/figures/
        'submissions': Path,   # outputs/submissions/
        'models':      Path,   # models/
    }
    """
    root = get_project_root()
    paths = {
        'root':        root,
        'data':        root / 'data' / 'raw',
        'figures':     root / 'outputs' / 'figures',
        'submissions': root / 'outputs' / 'submissions',
        'models':      root / 'models',
    }
    for key in ['figures', 'submissions', 'models']:
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths

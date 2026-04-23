"""
model.py
--------
Model definitions, training, evaluation, feature importance,
and persistence — matching the notebook exactly.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.utils import wmae
from src.features import FEATURE_COLS, SPLIT_DATE

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


# ── Model definitions ───────────────────────────────────────────────────────────
# Parameters match the notebook exactly.

def get_ridge() -> Ridge:
    return Ridge(alpha=1.0)


def get_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200, max_depth=12,
        min_samples_leaf=4, n_jobs=-1, random_state=42
    )


def get_xgboost() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def get_lightgbm():
    if not LGB_AVAILABLE:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")
    return lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


# ── Train / val split ───────────────────────────────────────────────────────────

def time_split(df: pd.DataFrame) -> tuple:
    """
    Split by SPLIT_DATE (2012-09-01) — matches notebook exactly.

    Returns
    -------
    X_train, X_val, y_train, y_val, holiday_val
    """
    train_mask = df['Date'] < SPLIT_DATE
    val_mask   = df['Date'] >= SPLIT_DATE

    X = df[FEATURE_COLS]
    y = df['Weekly_Sales']

    return (
        X[train_mask], X[val_mask],
        y[train_mask], y[val_mask],
        df.loc[val_mask, 'IsHoliday'].values,
    )


# ── Training & evaluation ───────────────────────────────────────────────────────

def train_all_models(X_train, y_train, X_val, y_val, holiday_val) -> tuple:
    """
    Train Ridge, RandomForest, XGBoost, LightGBM (if available), and Ensemble.
    Prints results as they finish, matching notebook output format.

    Returns
    -------
    results : {model_name: {'WMAE': float, 'MAE': float, 'RMSE': float}}
    models  : {model_name: fitted estimator}
    preds   : {model_name: np.ndarray}
    """
    results, models, preds = {}, {}, {}

    def _eval(name, model, pred):
        pred = np.maximum(pred, 0)
        score = wmae(y_val, pred, holiday_val)
        mae   = mean_absolute_error(y_val, pred)
        rmse  = np.sqrt(mean_squared_error(y_val, pred))
        results[name] = {'WMAE': score, 'MAE': mae, 'RMSE': rmse}
        models[name]  = model
        preds[name]   = pred
        print(f"{name:<14}→  WMAE: ${score:>9,.2f}  |  MAE: ${mae:>9,.2f}  |  RMSE: ${rmse:>9,.2f}")

    # Ridge
    ridge = get_ridge()
    ridge.fit(X_train, y_train)
    _eval('Ridge', ridge, ridge.predict(X_val))

    # Random Forest
    rf = get_random_forest()
    rf.fit(X_train, y_train)
    _eval('Random Forest', rf, rf.predict(X_val))

    # XGBoost
    xgb_model = get_xgboost()
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    _eval('XGBoost', xgb_model, xgb_model.predict(X_val))

    # LightGBM
    if LGB_AVAILABLE:
        lgb_model = get_lightgbm()
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(False)],
        )
        _eval('LightGBM', lgb_model, lgb_model.predict(X_val))

    # Ensemble
    blend_preds = [preds['Random Forest'], preds['XGBoost']]
    if LGB_AVAILABLE:
        blend_preds.append(preds['LightGBM'])
    pred_blend = np.mean(blend_preds, axis=0)
    _eval('Ensemble', None, pred_blend)
    preds['Ensemble'] = pred_blend

    return results, models, preds


# ── Feature importance plot ─────────────────────────────────────────────────────

def plot_feature_importance(xgb_model, figures_dir, palette):
    """
    Plot XGBoost built-in feature importance (weight, gain, cover).
    Matches the updated notebook Cell 8.2.
    Saves to figures_dir/shap_summary.png.
    """
    importance_types = ['weight', 'gain', 'cover']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, imp_type in zip(axes, importance_types):
        scores = xgb_model.get_booster().get_score(importance_type=imp_type)
        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15])
        ax.barh(list(scores.keys())[::-1], list(scores.values())[::-1], color=palette[0])
        ax.set_title(f'Feature Importance ({imp_type.capitalize()})', fontweight='bold')
        ax.set_xlabel(imp_type.capitalize())

    plt.tight_layout()
    out = Path(figures_dir) / 'shap_summary.png'
    plt.savefig(out, bbox_inches='tight')
    plt.show()
    print(f"Saved → {out}")
    print("weight=frequency, gain=avg split gain, cover=avg coverage")


# ── Submission ──────────────────────────────────────────────────────────────────

def generate_submission(xgb_model, test_df, submissions_dir) -> pd.DataFrame:
    """
    Run predictions on test_df and save submission.csv.

    Returns
    -------
    submission DataFrame with columns: Store, Dept, Date, Predicted_Sales
    """
    from src.features import FEATURE_COLS
    test_X = test_df[FEATURE_COLS]
    test_df = test_df.copy()
    test_df['Predicted_Sales'] = np.maximum(xgb_model.predict(test_X), 0)

    submission = test_df[['Store', 'Dept', 'Date', 'Predicted_Sales']].copy()
    submission['Date'] = submission['Date'].dt.strftime('%Y-%m-%d')

    out = Path(submissions_dir) / 'submission.csv'
    submission.to_csv(out, index=False)
    print(f"submission.csv saved → {out}")
    print(f"Prediction range: ${submission['Predicted_Sales'].min():>10,.2f} → ${submission['Predicted_Sales'].max():>10,.2f}")
    print(f"Mean prediction:  ${submission['Predicted_Sales'].mean():>10,.2f}")
    return submission


# ── Persistence ─────────────────────────────────────────────────────────────────

def save_model(model, models_dir) -> None:
    out = Path(models_dir) / 'xgboost_model.pkl'
    with open(out, 'wb') as f:
        pickle.dump(model, f)
    print(f"xgboost_model.pkl saved → {out}")


def load_model(models_dir):
    path = Path(models_dir) / 'xgboost_model.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)

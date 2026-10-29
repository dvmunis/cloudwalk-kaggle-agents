"""A small, declarative feature toolbox.

Agents choose operations by name instead of writing code, which keeps every
experiment reproducible and easy to audit.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .schemas import FeatureConfig


def _check_columns(requested: list[str], available: list[str], op: str) -> None:
    missing = [c for c in requested if c not in available]
    if missing:
        raise ValueError(f"{op}: unknown columns {missing}")


def build(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], cfg: FeatureConfig):
    """Return (X_train, X_test) with the configured transformations applied identically to both."""
    for op, cols in (("drop", cfg.drop), ("frequency_encode", cfg.frequency_encode), ("as_category", cfg.as_category),
                     ("target_encode", cfg.target_encode), ("digit_features", cfg.digit_features)):
        _check_columns(cols, feature_cols, op)
    dropped = [c for c in cfg.target_encode if c in cfg.drop]
    if dropped:
        raise ValueError(f"target_encode: columns {dropped} are also dropped")

    cols = [c for c in feature_cols if c not in cfg.drop]
    X_train, X_test = train[cols].copy(), test[cols].copy()
    combined = pd.concat([X_train, X_test], axis=0)

    for col in cfg.frequency_encode:
        counts = combined[col].value_counts()
        X_train[f"{col}_freq"] = X_train[col].map(counts).astype(float)
        X_test[f"{col}_freq"] = X_test[col].map(counts).astype(float)

    # Synthetic data generators leave traces in how numbers are written (rounding,
    # last digits); these features expose them to the model.
    for col in cfg.digit_features:
        if not pd.api.types.is_numeric_dtype(combined[col]):
            raise ValueError(f"digit_features: column '{col}' is not numeric")
        for frame in (X_train, X_test):
            value = frame[col].astype(float).abs()
            whole = np.floor(value)
            frame[f"{col}_d0"] = whole % 10
            frame[f"{col}_d1"] = (whole // 10) % 10
            frame[f"{col}_frac"] = np.round((value - whole) * 10)
            frame[f"{col}_round100"] = (whole % 100 == 0).astype(float).where(value.notna())

    if cfg.add_null_count:
        X_train["null_count"] = train[cols].isna().sum(axis=1)
        X_test["null_count"] = test[cols].isna().sum(axis=1)

    # Object columns and requested ones become pandas categoricals with a shared
    # vocabulary, which both LightGBM and XGBoost handle natively.
    categorical = [c for c in cols if combined[c].dtype == object] + cfg.as_category
    for col in categorical:
        dtype = pd.CategoricalDtype(combined[col].astype(str).unique())
        X_train[col] = X_train[col].astype(str).astype(dtype)
        X_test[col] = X_test[col].astype(str).astype(dtype)

    return X_train, X_test


TARGET_ENCODING_SMOOTHING = 20
INNER_FOLDS = 5


def _te_fit(keys: pd.Series, Y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Smoothed target mean per value: (sum + m * prior) / (count + m)."""
    prior = Y.mean(axis=0)
    frame = pd.DataFrame(Y, index=keys.index)
    grouped = frame.groupby(keys.astype(str).to_numpy())
    enc = (grouped.sum() + TARGET_ENCODING_SMOOTHING * prior).div(grouped.size() + TARGET_ENCODING_SMOOTHING, axis=0)
    return enc, prior


def _te_apply(keys: pd.Series, enc: pd.DataFrame, prior: np.ndarray) -> np.ndarray:
    values = enc.reindex(keys.astype(str).to_numpy()).to_numpy()
    return np.where(np.isnan(values), prior, values)


def add_target_encoding(X_fit: pd.DataFrame, Y_fit: np.ndarray, others: list[pd.DataFrame], cols: list[str],
                        seed: int) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Add <col>_te columns without target leakage.

    Y_fit is (n_rows, k): one column for binary or regression targets, one per class
    for multiclass. Rows of X_fit get values from an inner K-fold, so the model never
    sees an encoding built from its own target; `others` (validation and test rows)
    get the encoding fitted on all of X_fit.
    """
    X_fit = X_fit.copy()
    others = [o.copy() for o in others]
    suffixes = ["_te"] if Y_fit.shape[1] == 1 else [f"_te{i}" for i in range(Y_fit.shape[1])]
    for col in cols:
        inner = np.zeros(Y_fit.shape)
        for tr, va in KFold(INNER_FOLDS, shuffle=True, random_state=seed).split(X_fit):
            enc, prior = _te_fit(X_fit[col].iloc[tr], Y_fit[tr])
            inner[va] = _te_apply(X_fit[col].iloc[va], enc, prior)
        enc, prior = _te_fit(X_fit[col], Y_fit)
        for j, suffix in enumerate(suffixes):
            X_fit[f"{col}{suffix}"] = inner[:, j]
        for other in others:
            values = _te_apply(other[col], enc, prior)
            for j, suffix in enumerate(suffixes):
                other[f"{col}{suffix}"] = values[:, j]
    return X_fit, others

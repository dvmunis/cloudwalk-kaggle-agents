"""Load a tabular competition and infer its structure from the files alone.

Everything a new competition needs (id column, target, task type, submission
format) is inferred from train/test/sample_submission, so switching
competitions does not require code changes.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import settings

SUBMISSION_FORMATS = ("value", "label", "proba", "proba_per_class", "topk")
MAX_PROFILED_COLUMNS = 60


@dataclass
class CompetitionData:
    slug: str
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame
    id_col: str
    target: str
    task: str  # regression | binary | multiclass
    classes: np.ndarray | None
    submission_format: str
    topk: int | None

    @property
    def feature_cols(self) -> list[str]:
        return [c for c in self.train.columns if c not in (self.id_col, self.target)]


def _find_csv(directory: Path, keyword: str) -> Path:
    exact = directory / f"{keyword}.csv"
    if exact.exists():
        return exact
    candidates = sorted(directory.glob(f"*{keyword}*.csv"))
    if not candidates:
        raise ValueError(f"No '{keyword}' CSV found in {directory}. Download the competition first.")
    return candidates[0]


def _infer_task(target: pd.Series) -> str:
    n_unique = target.nunique()
    is_discrete = target.dtype == object or target.dtype == bool or (
        pd.api.types.is_integer_dtype(target) and n_unique <= 20
    )
    if not is_discrete:
        return "regression"
    return "binary" if n_unique == 2 else "multiclass"


def _infer_submission_format(sample: pd.DataFrame, task: str) -> tuple[str, int | None]:
    target_cols = list(sample.columns[1:])
    if len(target_cols) > 1:
        return "proba_per_class", None
    if task == "regression":
        return "value", None
    values = sample[target_cols[0]]
    if values.dtype == object and values.astype(str).str.contains(" ").any():
        return "topk", int(values.astype(str).str.split().str.len().max())
    if pd.api.types.is_float_dtype(values) and not np.all(np.mod(values, 1) == 0):
        return "proba", None
    return "label", None


def load(slug: str) -> CompetitionData:
    directory = settings.competition_data_dir(slug)
    train = pd.read_csv(_find_csv(directory, "train"))
    test = pd.read_csv(_find_csv(directory, "test"))
    sample = pd.read_csv(_find_csv(directory, "sample_submission"))

    id_col = sample.columns[0]
    only_in_train = [c for c in train.columns if c not in test.columns]
    if len(only_in_train) != 1:
        raise ValueError(f"Expected exactly one target column (in train, not in test); found {only_in_train}")
    target = only_in_train[0]

    task = _infer_task(train[target])
    classes = None if task == "regression" else np.sort(train[target].unique())
    submission_format, topk = _infer_submission_format(sample, task)

    return CompetitionData(slug, train, test, sample, id_col, target, task, classes, submission_format, topk)


def suggest_metric(data: CompetitionData) -> str:
    """A default guess. The planner should confirm it against the competition's Evaluation page."""
    if data.task == "regression":
        return "rmse"
    if data.submission_format == "topk":
        return f"map@{data.topk}"
    if data.submission_format == "proba_per_class":
        return "log_loss"
    if data.submission_format == "proba":
        return "roc_auc"
    return "accuracy"


def _column_summary(series: pd.Series) -> dict:
    summary = {
        "dtype": str(series.dtype),
        "null_pct": round(float(series.isna().mean()) * 100, 2),
        "n_unique": int(series.nunique()),
    }
    if pd.api.types.is_numeric_dtype(series):
        summary.update(min=float(series.min()), max=float(series.max()), mean=round(float(series.mean()), 4))
    else:
        summary["top_values"] = series.value_counts().head(5).index.astype(str).tolist()
    return summary


def profile(data: CompetitionData) -> dict:
    """A compact, LLM-friendly description of the competition data."""
    target = data.train[data.target]
    if data.task == "regression":
        target_summary = target.describe().round(4).to_dict()
    else:
        target_summary = target.value_counts(normalize=True).round(4).astype(float).to_dict()
        target_summary = {str(k): v for k, v in target_summary.items()}

    features = data.feature_cols
    return {
        "competition": data.slug,
        "train_rows": len(data.train),
        "test_rows": len(data.test),
        "id_col": data.id_col,
        "target": data.target,
        "task": data.task,
        "classes": None if data.classes is None else [str(c) for c in data.classes],
        "submission_format": data.submission_format,
        "topk": data.topk,
        "suggested_metric": suggest_metric(data),
        "target_distribution": target_summary,
        "n_features": len(features),
        "features": {c: _column_summary(data.train[c]) for c in features[:MAX_PROFILED_COLUMNS]},
    }

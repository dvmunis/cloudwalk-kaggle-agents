"""Competition metrics, computed from raw model output.

Raw output is a vector for regression and an (n_rows, n_classes) probability
matrix for classification; y_true is the class index for classification.
"""

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn import metrics as skm


@dataclass(frozen=True)
class Metric:
    name: str
    higher_is_better: bool
    tasks: tuple[str, ...]
    fn: Callable[[np.ndarray, np.ndarray], float]


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(y)) ** 2)))


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _roc_auc(y, p):
    if p.shape[1] == 2:
        return float(skm.roc_auc_score(y, p[:, 1]))
    return float(skm.roc_auc_score(y, p, multi_class="ovr"))


def _log_loss(y, p):
    return float(skm.log_loss(y, p, labels=list(range(p.shape[1]))))


def _accuracy(y, p):
    return float(skm.accuracy_score(y, p.argmax(axis=1)))


def _f1(y, p):
    average = "binary" if p.shape[1] == 2 else "macro"
    return float(skm.f1_score(y, p.argmax(axis=1), average=average))


def top_k(p: np.ndarray, k: int) -> np.ndarray:
    """Class indices of the k most likely classes per row, best first."""
    return np.argsort(-p, axis=1)[:, :k]


def _map_at_k(k: int):
    def fn(y, p):
        hits = top_k(p, k) == y[:, None]
        weights = 1.0 / np.arange(1, k + 1)
        return float((hits * weights).sum(axis=1).mean())

    return fn


CLASSIFICATION = ("binary", "multiclass")

_METRICS = {
    "rmse": Metric("rmse", False, ("regression",), _rmse),
    "rmsle": Metric("rmsle", False, ("regression",), _rmsle),
    "mae": Metric("mae", False, ("regression",), _mae),
    "roc_auc": Metric("roc_auc", True, CLASSIFICATION, _roc_auc),
    "log_loss": Metric("log_loss", False, CLASSIFICATION, _log_loss),
    "accuracy": Metric("accuracy", True, CLASSIFICATION, _accuracy),
    "f1": Metric("f1", True, CLASSIFICATION, _f1),
}

SUPPORTED = [*_METRICS, "map@k"]


def get(name: str) -> Metric:
    name = name.lower()
    if match := re.fullmatch(r"map@(\d+)", name):
        k = int(match.group(1))
        return Metric(name, True, ("multiclass",), _map_at_k(k))
    if name not in _METRICS:
        raise ValueError(f"Unsupported metric '{name}'. Supported: {SUPPORTED}")
    return _METRICS[name]


def check_compatible(metric: Metric, task: str) -> None:
    if task not in metric.tasks:
        raise ValueError(f"Metric '{metric.name}' does not apply to a {task} task")

"""Blends: average the predictions of earlier runs.

Every trained run keeps its out-of-fold (OOF) and test predictions with the rows
and folds they came from. A blend of runs that share rows and folds gets an
honest CV score from the averaged OOF predictions, with no retraining.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from . import metrics, settings, store, submission
from .profiling import CompetitionData
from .schemas import TrainConfig

FILES = ("rows", "folds", "oof", "test")


def save_predictions(run_path: Path, rows: np.ndarray, folds: np.ndarray, oof: np.ndarray, test: np.ndarray) -> None:
    np.savez_compressed(run_path / "predictions.npz", rows=rows, folds=folds, oof=oof, test=test)


def has_predictions(slug: str, run_id: str) -> bool:
    return (settings.competition_runs_dir(slug) / run_id / "predictions.npz").exists()


def _load(slug: str, run_id: str) -> dict:
    path = store.run_dir(slug, run_id) / "predictions.npz"
    if not path.exists():
        raise ValueError(f"Run '{run_id}' has no saved predictions (trained before blending existed); it cannot be blended")
    with np.load(path) as saved:
        return {name: saved[name] for name in FILES}


def _rank(p: np.ndarray) -> np.ndarray:
    """Positive-class probabilities replaced by normalized ranks: AUC only depends on order."""
    r = rankdata(p[:, 1]) / len(p)
    return np.column_stack([1 - r, r])


def blend(data: CompetitionData, cfg: TrainConfig, attempt_id: str | None = None) -> dict:
    run_ids = list(dict.fromkeys(cfg.blend_runs))
    if len(run_ids) < 2:
        raise ValueError("blend needs at least two different runs in blend_runs")
    metric = metrics.get(cfg.metric)
    started = time.monotonic()
    runs = [_load(data.slug, r) for r in run_ids]
    first = runs[0]
    for run_id, run in zip(run_ids[1:], runs[1:]):
        if not (np.array_equal(run["rows"], first["rows"]) and np.array_equal(run["folds"], first["folds"])):
            raise ValueError(f"Run '{run_id}' was trained on different rows or folds than '{run_ids[0]}'")

    rank_average = metric.name == "roc_auc" and first["oof"].ndim == 2 and first["oof"].shape[1] == 2
    combine = _rank if rank_average else (lambda p: p)
    oof = np.mean([combine(r["oof"]) for r in runs], axis=0)
    test = np.mean([combine(r["test"]) for r in runs], axis=0)

    target = data.train.loc[first["rows"], data.target]
    y = target.to_numpy(dtype=float) if data.task == "regression" else np.searchsorted(data.classes, target.to_numpy())
    folds = first["folds"]
    fold_scores = [metric.fn(y[folds == f], oof[folds == f]) for f in np.unique(folds)]

    run_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-blend-{uuid.uuid4().hex[:6]}"
    run_path = settings.competition_runs_dir(data.slug) / run_id
    run_path.mkdir(parents=True)
    submission.build(data, test).to_csv(run_path / "submission.csv", index=False)
    save_predictions(run_path, first["rows"], folds, oof, test)

    members = []
    for member in run_ids:
        result = json.loads((store.run_dir(data.slug, member) / "result.json").read_text())
        members.append({"run_id": member, "model": result["config"]["model"], "cv_score": result["cv_score"]})
    result = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "competition": data.slug,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric": metric.name,
        "higher_is_better": metric.higher_is_better,
        "cv_score": metric.fn(y, oof),
        "cv_std": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
        "best_iterations": [],
        "train_rows": len(y),
        "n_features": 0,
        "top_features": [],
        "blend": {"method": "rank average" if rank_average else "mean", "members": members},
        "duration_s": round(time.monotonic() - started, 1),
        "config": cfg.model_dump(),
    }
    (run_path / "result.json").write_text(json.dumps(result, indent=2))
    store.append_experiment(data.slug, result)
    return result

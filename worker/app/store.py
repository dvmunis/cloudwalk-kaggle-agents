"""Run artifacts and the experiment log, kept on disk under RUNS_DIR/<competition>/."""

import json
from pathlib import Path

from . import settings


def run_dir(slug: str, run_id: str) -> Path:
    path = settings.competition_runs_dir(slug) / run_id
    if not path.is_dir():
        raise ValueError(f"Unknown run_id '{run_id}' for competition '{slug}'")
    return path


def append_experiment(slug: str, record: dict) -> None:
    log = settings.competition_runs_dir(slug) / "experiments.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps(record) + "\n")


def experiments(slug: str) -> list[dict]:
    log = settings.competition_runs_dir(slug) / "experiments.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def history(slug: str, limit: int = 10) -> dict:
    """Compact, LLM-friendly view of past experiments: the best full-data run and the latest ones."""
    runs = [
        {
            "run_id": r["run_id"],
            "metric": r["metric"],
            "cv_score": round(r["cv_score"], 5),
            "cv_std": round(r["cv_std"], 5),
            "model": r["config"]["model"],
            "params": r["config"]["params"],
            "features": r["config"]["features"],
            "sample_frac": r["config"]["sample_frac"],
            "note": r["config"].get("note", ""),
            "external_data": r["config"].get("external_data", []),
            "blend_runs": r["config"].get("blend_runs", []),
            "blendable": (settings.competition_runs_dir(slug) / r["run_id"] / "predictions.npz").exists(),
        }
        for r in experiments(slug)
    ]
    full = [r for r in runs if r["sample_frac"] == 1]
    best = best_model = None
    if full:
        pick = max if experiments(slug)[0]["higher_is_better"] else min
        best = pick(full, key=lambda r: r["cv_score"])
        models = [r for r in full if r["model"] != "blend"]
        best_model = pick(models, key=lambda r: r["cv_score"]) if models else None
    return {
        "n_experiments": len(runs),
        "best_full_data_run": best,
        "best_full_data_model_run": best_model,
        "recent": runs[-limit:],
    }

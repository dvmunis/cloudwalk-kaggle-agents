import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import leaderboard, main, metrics, profiling

FAST = {"n_estimators": 50}


@pytest.fixture
def client(data_root):
    main._loaded.clear()
    return TestClient(main.app)


@pytest.mark.parametrize(
    "slug, task, fmt, metric",
    [
        ("binary-comp", "binary", "proba", "roc_auc"),
        ("regression-comp", "regression", "value", "rmse"),
        ("topk-comp", "multiclass", "topk", "map@3"),
    ],
)
def test_profile_infers_competition_structure(client, slug, task, fmt, metric):
    body = client.get(f"/competitions/{slug}/profile").json()
    assert (body["task"], body["submission_format"], body["suggested_metric"]) == (task, fmt, metric)
    assert body["id_col"] == "id"


@pytest.mark.parametrize(
    "slug, cfg",
    [
        ("binary-comp", {"metric": "roc_auc", "features": {"frequency_encode": ["cat"], "add_null_count": True}}),
        ("regression-comp", {"metric": "rmsle", "target_transform": "log1p", "model": "xgboost"}),
        ("topk-comp", {"metric": "map@3"}),
    ],
)
def test_train_writes_valid_submission(client, data_root, slug, cfg):
    result = client.post(f"/competitions/{slug}/train", json={**cfg, "params": FAST, "n_folds": 3}).json()
    assert np.isfinite(result["cv_score"]) and len(result["fold_scores"]) == 3

    submission = pd.read_csv(data_root / "runs" / slug / result["run_id"] / "submission.csv")
    sample = pd.read_csv(data_root / "data" / slug / "sample_submission.csv")
    assert list(submission.columns) == list(sample.columns)
    assert len(submission) == len(sample)

    assert client.get(f"/competitions/{slug}/experiments").json()[-1]["run_id"] == result["run_id"]


def test_history_reports_best_full_data_run(client):
    assert client.get("/competitions/binary-comp/history").json()["best_full_data_run"] is None

    base = {"metric": "roc_auc", "params": FAST, "n_folds": 2}
    sampled = client.post("/competitions/binary-comp/train", json={**base, "sample_frac": 0.5}).json()
    full = client.post("/competitions/binary-comp/train", json=base).json()

    history = client.get("/competitions/binary-comp/history").json()
    assert history["n_experiments"] == 2
    assert history["best_full_data_run"]["run_id"] == full["run_id"]
    assert [r["run_id"] for r in history["recent"]] == [sampled["run_id"], full["run_id"]]


def test_binary_signal_is_learned(client):
    result = client.post("/competitions/binary-comp/train", json={"metric": "roc_auc", "params": FAST}).json()
    assert result["cv_score"] > 0.75


def test_agent_mistakes_return_readable_errors(client):
    bad_metric = client.post("/competitions/binary-comp/train", json={"metric": "rmse"})
    assert bad_metric.status_code == 422 and "does not apply" in bad_metric.json()["error"]

    bad_column = client.post("/competitions/binary-comp/train",
                             json={"metric": "roc_auc", "features": {"drop": ["nope"]}})
    assert bad_column.status_code == 422 and "nope" in bad_column.json()["error"]


def test_map_at_k():
    y = np.array([0, 1, 2])
    p = np.array([[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.6, 0.3, 0.1]])
    # Row 1 hit at rank 1, row 2 at rank 2, row 3 at rank 3.
    assert metrics.get("map@3").fn(y, p) == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)


@pytest.mark.parametrize(
    "scores, mine, rank",
    [([0.95, 0.90, 0.85, 0.80, 0.70], 0.86, 3), ([0.10, 0.20, 0.30, 0.40, 0.50], 0.25, 3)],
)
def test_leaderboard_position_reads_direction_from_ordering(scores, mine, rank):
    pos = leaderboard.position(scores, mine)
    assert pos["rank"] == rank and pos["top_percent"] == 60.0
    # Score of the last team inside each cut: top 20% of 5 teams is the 1st, top 50% the 2nd.
    assert pos["thresholds"] == {"top_10": scores[0], "top_20": scores[0], "top_50": scores[1]}


def test_missing_data_is_a_clear_error(data_root):
    with pytest.raises(ValueError, match="Download the competition first"):
        profiling.load("never-downloaded")


def test_dashboard_and_competition_list(client):
    assert client.get("/competitions").json() == []
    client.post("/competitions/binary-comp/train", json={"metric": "roc_auc", "params": FAST, "n_folds": 2})
    assert client.get("/competitions").json() == ["binary-comp"]
    page = client.get("/dashboard")
    assert page.status_code == 200 and "<title>" in page.text


def test_leaderboard_summary_ties_public_scores_to_runs():
    rows = [{"Score": "0.9"}, {"Score": "0.8"}]
    own = [{"description": "20261030-031552-lightgbm-6339bf lr 0.02", "publicScore": "0.85"},
           {"description": "manual upload", "publicScore": ""}]
    summary = leaderboard.summarize(rows, own)
    assert summary["submissions"] == [{"run_id": "20261030-031552-lightgbm-6339bf", "public_score": 0.85}]

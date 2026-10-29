"""Target encoding and blending: the tools added for the improvement loop."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import features, main

FAST = {"n_estimators": 50}
BASE = {"metric": "roc_auc", "params": FAST, "n_folds": 3}


@pytest.fixture
def client(data_root):
    main._loaded.clear()
    return TestClient(main.app)


def test_target_encoding_never_uses_a_rows_own_target():
    # A unique key per row: an encoding that saw the row's own target would equal it.
    X = pd.DataFrame({"key": np.arange(200)})
    y = np.random.default_rng(0).integers(0, 2, size=(200, 1)).astype(float)
    encoded, _ = features.add_target_encoding(X, y, [], ["key"], seed=0)
    assert np.allclose(encoded["key_te"], y.mean(), atol=0.05)


def test_digit_features_expose_how_numbers_are_written(client):
    X, _ = features.build(pd.DataFrame({"n": [1234.5, 30000.0, None]}), pd.DataFrame({"n": [7.0]}), ["n"],
                          features.FeatureConfig(digit_features=["n"]))
    assert X["n_d0"].tolist()[:2] == [4, 0] and X["n_d1"].tolist()[:2] == [3, 0]
    assert X["n_frac"].tolist()[:2] == [5, 0] and X["n_round100"].tolist()[:2] == [0, 1]

    text_column = client.post("/competitions/binary-comp/train", json={**BASE, "features": {"digit_features": ["cat"]}})
    assert text_column.status_code == 422 and "not numeric" in text_column.json()["error"]


@pytest.mark.parametrize("slug, metric", [("binary-comp", "roc_auc"), ("regression-comp", "rmse"),
                                          ("topk-comp", "map@3")])
def test_target_encoding_trains_for_every_task(client, slug, metric):
    cfg = {**BASE, "metric": metric, "features": {"target_encode": ["cat"]}}
    result = client.post(f"/competitions/{slug}/train", json=cfg).json()
    assert np.isfinite(result["cv_score"])
    assert any(f["feature"].startswith("cat_te") for f in result["top_features"])


def test_blend_scores_averaged_out_of_fold_predictions(client, data_root):
    a = client.post("/competitions/binary-comp/train", json=BASE).json()
    b = client.post("/competitions/binary-comp/train", json={**BASE, "model": "xgboost"}).json()
    blend = client.post("/competitions/binary-comp/train",
                        json={**BASE, "model": "blend", "blend_runs": [a["run_id"], b["run_id"]]}).json()

    assert blend["blend"]["method"] == "rank average"
    assert [m["run_id"] for m in blend["blend"]["members"]] == [a["run_id"], b["run_id"]]
    assert len(blend["fold_scores"]) == 3
    assert blend["cv_score"] > min(a["cv_score"], b["cv_score"]) - 0.02
    submission = pd.read_csv(data_root / "runs" / "binary-comp" / blend["run_id"] / "submission.csv")
    assert submission["churn"].between(0, 1).all()

    history = client.get("/competitions/binary-comp/history").json()
    assert history["recent"][-1]["blend_runs"] == [a["run_id"], b["run_id"]]
    assert all(r["blendable"] for r in history["recent"])
    assert history["best_full_data_model_run"]["model"] != "blend"


def test_blend_rejects_runs_that_cannot_be_combined(client):
    full = client.post("/competitions/binary-comp/train", json=BASE).json()
    sampled = client.post("/competitions/binary-comp/train", json={**BASE, "sample_frac": 0.5}).json()

    one = client.post("/competitions/binary-comp/train",
                      json={**BASE, "model": "blend", "blend_runs": [full["run_id"]]})
    assert one.status_code == 422 and "at least two" in one.json()["error"]

    mixed = client.post("/competitions/binary-comp/train",
                        json={**BASE, "model": "blend", "blend_runs": [full["run_id"], sampled["run_id"]]})
    assert mixed.status_code == 422 and "different rows" in mixed.json()["error"]


@pytest.fixture
def fake_kaggle(data_root, monkeypatch):
    """Two Kaggle 'datasets': the real-looking original and an unrelated one."""
    rng = np.random.default_rng(1)
    original = pd.DataFrame({"Num": rng.normal(size=300), "CAT": rng.choice(["a", "b", "c"], size=300)})
    original["Churn"] = (original["Num"] > 0.3).astype(int)
    files = {
        "someone/churn-original": ("churn.csv", original),
        "someone/weather": ("weather.csv", pd.DataFrame({"temp": [1.0, 2.0], "rain": [0, 1]})),
        "someone/binary-comp-oof": ("oof.csv", original),
        "other/churn-reupload": ("copy.csv", original),
    }
    searches = []

    def search(query):
        searches.append(query)
        return [{"ref": ref, "title": ref, "size": "1000"} for ref in files]

    def download(ref, dest):
        dest.mkdir(parents=True, exist_ok=True)
        name, frame = files[ref]
        frame.to_csv(dest / name, index=False)
        return [dest / name]

    monkeypatch.setattr("app.kaggle_client.search_datasets", search)
    monkeypatch.setattr("app.kaggle_client.download_dataset", download)
    return searches


def test_discovery_accepts_only_datasets_with_the_competition_columns(client, fake_kaggle):
    found = client.post("/competitions/binary-comp/external/discover", json={"queries": ["churn data"]}).json()

    assert fake_kaggle == ["churn data", "churn"]
    assert [d["name"] for d in found["accepted"]] == ["someone__churn-original"]
    reasons = {r["ref"]: r["reason"] for r in found["rejected"]}
    assert "no 'churn' column" in reasons["someone/weather"]
    assert "built from this competition" in reasons["someone/binary-comp-oof"]
    assert reasons["other/churn-reupload"] == "same rows as someone/churn-original"
    assert client.get("/competitions/binary-comp/profile").json()["external_datasets"][0]["rows"] == 300


def test_external_rows_only_join_the_training_folds(client, fake_kaggle):
    client.post("/competitions/binary-comp/external/discover", json={"queries": []})
    plain = client.post("/competitions/binary-comp/train", json=BASE).json()
    extended = client.post("/competitions/binary-comp/train",
                           json={**BASE, "external_data": ["someone__churn-original"],
                                 "features": {"target_encode": ["cat"]}}).json()

    # Out-of-fold predictions still cover exactly the competition rows.
    assert extended["train_rows"] == plain["train_rows"] == 400
    assert np.isfinite(extended["cv_score"])

    unknown = client.post("/competitions/binary-comp/train", json={**BASE, "external_data": ["nope"]})
    assert unknown.status_code == 422 and "unknown datasets" in unknown.json()["error"]

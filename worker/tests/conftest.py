"""Synthetic competitions on disk, so tests never touch Kaggle."""

import numpy as np
import pandas as pd
import pytest


def _write(directory, train, test, sample):
    directory.mkdir(parents=True)
    train.to_csv(directory / "train.csv", index=False)
    test.to_csv(directory / "test.csv", index=False)
    sample.to_csv(directory / "sample_submission.csv", index=False)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    rng = np.random.default_rng(0)
    n, m = 400, 100

    def frame(rows, start):
        return pd.DataFrame({
            "id": np.arange(start, start + rows),
            "num": rng.normal(size=rows),
            "cat": rng.choice(["a", "b", "c"], size=rows),
        })

    # Binary classification scored on probabilities (AUC-style sample file).
    train = frame(n, 0)
    train["churn"] = (train["num"] + (train["cat"] == "a") + rng.normal(0, 0.5, n) > 0.5).astype(int)
    test = frame(m, n)
    _write(tmp_path / "data" / "binary-comp", train, test, pd.DataFrame({"id": test["id"], "churn": 0.5}))

    # Regression.
    train = frame(n, 0)
    train["price"] = np.exp(train["num"]) * 10
    test = frame(m, n)
    _write(tmp_path / "data" / "regression-comp", train, test, pd.DataFrame({"id": test["id"], "price": 1.0}))

    # Multiclass with a MAP@3 top-k sample file.
    train = frame(n, 0)
    train["label"] = rng.choice(["x", "y", "z", "w"], size=n)
    test = frame(m, n)
    _write(tmp_path / "data" / "topk-comp", train, test, pd.DataFrame({"id": test["id"], "label": "x y z"}))

    return tmp_path

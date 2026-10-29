"""Cross-validated training: out-of-fold score for the agents, fold-averaged test predictions for the submission."""

import json
import time
import uuid
from datetime import datetime, timezone

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

import pandas as pd

from . import blending, external, features, metrics, settings, store, submission
from .profiling import CompetitionData
from .schemas import TrainConfig

EARLY_STOPPING_ROUNDS = 100

DEFAULT_PARAMS = {
    "lightgbm": {"n_estimators": 3000, "learning_rate": 0.05, "num_leaves": 63, "subsample": 0.8,
                 "subsample_freq": 1, "colsample_bytree": 0.8},
    "xgboost": {"n_estimators": 3000, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
                "colsample_bytree": 0.8, "early_stopping_rounds": EARLY_STOPPING_ROUNDS},
}


def _make_model(cfg: TrainConfig, task: str):
    params = {**DEFAULT_PARAMS[cfg.model], **cfg.params, "random_state": cfg.seed}
    if cfg.model == "lightgbm":
        import lightgbm as lgb

        cls = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
        return cls(verbose=-1, **params)

    import xgboost as xgb

    cls = xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier
    return cls(enable_categorical=True, tree_method="hist", **params)


def _fit(model, cfg: TrainConfig, X_tr, y_tr, X_va, y_va):
    if cfg.model == "lightgbm":
        import lightgbm as lgb

        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    else:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def _predict(model, X, task: str) -> np.ndarray:
    return model.predict(X) if task == "regression" else model.predict_proba(X)


def _new_run_id(model: str) -> str:
    return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{model}-{uuid.uuid4().hex[:6]}"


def train(data: CompetitionData, cfg: TrainConfig, attempt_id: str | None = None) -> dict:
    metric = metrics.get(cfg.metric)
    metrics.check_compatible(metric, data.task)
    if cfg.model == "blend":
        return blending.blend(data, cfg, attempt_id)
    if cfg.target_transform == "log1p" and data.task != "regression":
        raise ValueError("target_transform 'log1p' only applies to regression")

    started = time.monotonic()
    train_df = data.train.sample(frac=cfg.sample_frac, random_state=cfg.seed) if cfg.sample_frac < 1 else data.train
    # External rows are built with the same features, then only ever added to the
    # training side of each fold, so the CV score is still measured on competition rows.
    ext = external.load(data, cfg.external_data)
    if ext is not None:
        ext.index = pd.RangeIndex(int(data.train.index.max()) + 1, int(data.train.index.max()) + 1 + len(ext))
    combined = train_df if ext is None else pd.concat([train_df, ext])
    X_all, X_test = features.build(combined, data.test, data.feature_cols, cfg.features)
    X, X_ext = X_all.iloc[:len(train_df)], X_all.iloc[len(train_df):]

    log_target = cfg.target_transform == "log1p"
    if data.task == "regression":
        y = train_df[data.target].to_numpy(dtype=float)
        y_fit = np.log1p(y) if log_target else y
        y_ext = np.array([]) if ext is None else ext[data.target].to_numpy(dtype=float)
        y_ext_fit = np.log1p(y_ext) if log_target else y_ext
        folds = KFold(cfg.n_folds, shuffle=True, random_state=cfg.seed).split(X)
        oof, test_pred = np.zeros(len(X)), np.zeros(len(X_test))
    else:
        y = np.searchsorted(data.classes, train_df[data.target].to_numpy())
        y_fit = y
        y_ext = y_ext_fit = np.array([], dtype=int) if ext is None else np.searchsorted(
            data.classes, ext[data.target].to_numpy())
        n_classes = len(data.classes)
        encode = lambda labels: labels.reshape(-1, 1).astype(float) if n_classes == 2 else np.eye(n_classes)[labels]  # noqa: E731
        folds = StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed).split(X, y)
        oof, test_pred = np.zeros((len(X), n_classes)), np.zeros((len(X_test), n_classes))

    def predict(model, frame):
        pred = _predict(model, frame, data.task)
        return np.expm1(pred) if log_target else pred

    fold_ids = np.zeros(len(X), dtype=int)
    fold_scores, importances, best_iterations = [], 0.0, []
    for fold, (train_idx, valid_idx) in enumerate(folds):
        X_tr, X_va, X_te = pd.concat([X.iloc[train_idx], X_ext]), X.iloc[valid_idx], X_test
        y_tr = np.concatenate([y_fit[train_idx], y_ext_fit])
        if cfg.features.target_encode:
            # Encodings are fitted inside each fold, so validation rows never leak into them.
            y_tr_encode = y_tr.reshape(-1, 1).astype(float) if data.task == "regression" else encode(
                np.concatenate([y[train_idx], y_ext]))
            X_tr, (X_va, X_te) = features.add_target_encoding(
                X_tr, y_tr_encode, [X_va, X_te], cfg.features.target_encode, cfg.seed)
        model = _make_model(cfg, data.task)
        _fit(model, cfg, X_tr, y_tr, X_va, y_fit[valid_idx])
        oof[valid_idx] = predict(model, X_va)
        test_pred += predict(model, X_te) / cfg.n_folds
        fold_ids[valid_idx] = fold
        fold_scores.append(metric.fn(y[valid_idx], oof[valid_idx]))
        importances = importances + model.feature_importances_ / cfg.n_folds
        feature_names = list(X_tr.columns)
        best_iterations.append(int(getattr(model, "best_iteration_", None) or getattr(model, "best_iteration", 0) or 0))

    run_id = _new_run_id(cfg.model)
    run_path = settings.competition_runs_dir(data.slug) / run_id
    run_path.mkdir(parents=True)
    submission.build(data, test_pred).to_csv(run_path / "submission.csv", index=False)
    blending.save_predictions(run_path, train_df.index.to_numpy(), fold_ids, oof, test_pred)

    top_features = sorted(zip(feature_names, importances), key=lambda item: -item[1])[:15]
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
        "best_iterations": best_iterations,
        "train_rows": len(X),
        "n_features": len(feature_names),
        "top_features": [{"feature": f, "importance": round(float(i), 2)} for f, i in top_features],
        "duration_s": round(time.monotonic() - started, 1),
        "config": cfg.model_dump(),
    }
    (run_path / "result.json").write_text(json.dumps(result, indent=2))
    store.append_experiment(data.slug, result)
    return result

"""Evaluation by nested cross-validation.

Outer loop: ``n_repeats`` x stratified ``n_splits``-fold CV (a new shuffle per
repeat) gives an out-of-fold prediction for every patient and repeat. Inner
loop: hyper-parameters (``grid``) are selected by GridSearchCV on the training
folds only, so the reported accuracy never sees the patients it is scored on.
Selecting and scoring on the same folds favours whichever configuration got
lucky on them (scripts/04_protocol_study.py measures it).

``X_fit`` / ``X_eval`` let the training and evaluated patients use different
features for the same patients, e.g. train on ground-truth-LV features and
evaluate on reconstructed-LV features, which is the situation of the test set.

With 100 patients one error = 1 point of accuracy (5 points on a fold), so
everything is reported as mean +- std over repeats.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import metrics as skm
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .config import CLASS_NAMES
from .models import build_model, grid_for_pipeline


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    recall = skm.recall_score(y, pred, labels=sorted(CLASS_NAMES), average=None, zero_division=0)
    return {
        "accuracy": skm.accuracy_score(y, pred),
        "balanced_accuracy": skm.balanced_accuracy_score(y, pred),
        "macro_f1": skm.f1_score(y, pred, average="macro"),
        **{f"recall_{CLASS_NAMES[c]}": r for c, r in zip(sorted(CLASS_NAMES), recall)},
    }


def make_estimator(spec: dict, seed: int, inner_splits: int = 5):
    est = build_model(spec["model"], spec.get("params"), seed)
    if spec.get("grid"):
        inner = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
        est = GridSearchCV(est, grid_for_pipeline(spec["grid"]), cv=inner, scoring="accuracy", n_jobs=1)
    return est


def chosen_params(est) -> dict:
    out = dict(getattr(est, "best_params_", {}))
    model = getattr(est, "best_estimator_", est)
    if hasattr(model, "alpha_"):
        out["alpha"] = model.alpha_
    return {k.removeprefix("clf__"): v for k, v in out.items()}


def _fit_fold(spec, X_fit, X_eval, y, tr, va, seed, inner_splits):
    import torch

    torch.set_num_threads(1)
    est = make_estimator(spec, seed, inner_splits).fit(X_fit[tr], y[tr])
    return va, est.predict_proba(X_eval[va]), chosen_params(est)


def nested_cv(spec: dict, X_fit: np.ndarray, X_eval: np.ndarray, y: np.ndarray, n_splits: int = 5,
              n_repeats: int = 10, inner_splits: int = 5, seed: int = 0, n_jobs: int = -1) -> dict:
    """Out-of-fold probabilities ``proba`` (repeats, patients, classes), per-repeat
    metrics and the hyper-parameters chosen in every outer fold."""
    y = np.asarray(y)
    classes = np.unique(y)
    jobs = []
    for r in range(n_repeats):
        outer = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        for f, (tr, va) in enumerate(outer.split(X_fit, y)):
            jobs.append((r, f, tr, va))
    out = Parallel(n_jobs=n_jobs)(delayed(_fit_fold)(spec, X_fit, X_eval, y, tr, va, seed + r, inner_splits)
                                  for r, f, tr, va in jobs)
    proba = np.zeros((n_repeats, len(y), len(classes)))
    params = []
    for (r, f, _, _), (va, p, chosen) in zip(jobs, out):
        proba[r, va] = p
        params.append({"repeat": r, "fold": f, **chosen})
    pred = classes[proba.argmax(-1)]
    per_repeat = pd.DataFrame([{"repeat": r, **classification_metrics(y, pred[r])} for r in range(n_repeats)])
    return {"proba": proba, "pred": pred, "per_repeat": per_repeat, "params": pd.DataFrame(params)}


def summarize(per_repeat: pd.DataFrame) -> dict:
    """mean / std over repeats of every metric."""
    cols = [c for c in per_repeat if c != "repeat"]
    return {**{c: per_repeat[c].mean() for c in cols}, **{f"{c}_std": per_repeat[c].std() for c in cols}}


def pooled_confusion(y: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """Row-normalised confusion matrix pooled over repeats (pred: repeats x patients)."""
    labels = sorted(CLASS_NAMES)
    cm = sum(skm.confusion_matrix(y, p, labels=labels) for p in np.atleast_2d(pred))
    names = [CLASS_NAMES[c] for c in labels]
    return pd.DataFrame(cm / cm.sum(1, keepdims=True), index=names, columns=names)


def params_summary(params: pd.DataFrame) -> str:
    """Most frequent value of each hyper-parameter over the outer folds, e.g. 'C=10 (34/50)'."""
    parts = []
    for c in params.columns.drop(["repeat", "fold"], errors="ignore"):
        counts = params[c].map(lambda v: "None" if v is None or v != v else str(v)).value_counts()  # NaN = None
        parts.append(f"{c}={counts.index[0]} ({counts.iloc[0]}/{len(params)})")
    return ", ".join(parts)

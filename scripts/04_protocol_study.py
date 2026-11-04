"""Effect of the evaluation protocol, on the 21 volumetric features (true LV).

    python scripts/04_protocol_study.py [--config protocol_study.yaml]

1. single-split protocol: features of all patients scaled together, one shuffle
   and one 5-fold split (random_state=42); every configuration of a grid is
   scored on these folds and the best score is reported;
2. MLP hyper-parameters scored on the patients the MLPs were trained on;
3. MLP / RF and MLP / SVM mixing weight alpha chosen on the evaluation folds,
   for several MLP seeds;
4. "top-16 features" selected on all patients before the cross-validation vs
   inside each training fold;
5. the same grids evaluated by nested cross-validation (scaler and grid search
   inside the training folds, repeated splits).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from cardiac import viz
from cardiac.evaluate import nested_cv, params_summary, pooled_confusion, summarize
from cardiac.experiment import load_config, load_xy, mean_std_text
from cardiac.features import VOLUMETRIC
from cardiac.models import MLPEnsembleClassifier, build_model
from cardiac.tracking import Run


class TopKImportance(TransformerMixin, BaseEstimator):
    """Keeps the k features with the largest random-forest importance (fitted on the training data only)."""

    def __init__(self, k=16, random_state=42):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state).fit(X, y)
        self.idx_ = np.argsort(-rf.feature_importances_)[: self.k]
        return self

    def transform(self, X):
        return X[:, self.idx_]


def single_split(run: Run, cfg: dict, X: np.ndarray, y: np.ndarray) -> dict:
    """Best grid score on a single split, scaler fitted on all patients."""
    Xs, ys = shuffle(X, y, random_state=42)
    Xs = StandardScaler().fit_transform(Xs)
    folds = StratifiedKFold(5, shuffle=True, random_state=42)
    best = {}
    for name, spec in cfg["grids"].items():
        if not spec.get("grid"):
            continue
        rows = []
        for params in ParameterGrid(spec["grid"]):
            clf = build_model(spec["model"], params, seed=42).named_steps["clf"]
            s = cross_val_score(clf, Xs, ys, cv=folds)
            rows.append({**params, "accuracy": s.mean(), "folds": " ".join(f"{v:.2f}" for v in s)})
        df = pd.DataFrame(rows)
        run.save_table(df, f"grid_single_split_{name}", latex=False)
        best[name] = df["accuracy"].max()
        run.log.info("single split %-8s best of %3d configs = %.3f (median %.3f)", name, len(df), best[name],
                     df["accuracy"].median())
    torch.set_num_threads(4)
    mlp = [cross_val_score(MLPEnsembleClassifier(random_state=s), Xs, ys, cv=folds).mean() for s in cfg["mlp_seeds"]]
    best["mlp_ensemble"] = float(np.mean(mlp))
    run.log.info("single split mlp_ensemble over seeds: %s", np.round(mlp, 3))
    return {"best": best, "mlp_by_seed": mlp, "Xs": Xs, "ys": ys, "folds": folds}


def mlp_training_accuracy(run: Run, Xs, ys, folds) -> dict:
    """MLPs scored on the patients they were trained on, vs cross-validation."""
    rows = []
    for epochs in (100, 200, 500):
        m = MLPEnsembleClassifier(n_models=10, epochs=epochs, random_state=0)
        train_acc = float(np.mean(clone(m).fit(Xs, ys).predict(Xs) == ys))
        cv_acc = float(cross_val_score(m, Xs, ys, cv=folds).mean())
        rows.append({"epochs": epochs, "training_accuracy": train_acc, "cv_accuracy": cv_acc})
    df = pd.DataFrame(rows)
    run.save_table(df, "mlp_training_vs_cv")
    run.log.info("MLP accuracy on its training patients vs CV:\n%s", df.to_string(index=False))
    return df.to_dict("records")


def alpha_curves(run: Run, cfg: dict, Xs, ys, folds) -> dict:
    """Accuracy of alpha * MLP + (1 - alpha) * other on the evaluation folds, one curve per MLP seed."""
    others = {"rf": build_model("rf", {"n_estimators": 200}, seed=42),
              "svm": build_model("svm_linear", {"C": 10}, seed=42)}
    p_other = {k: cross_val_predict(m, Xs, ys, cv=folds, method="predict_proba") for k, m in others.items()}
    alphas = np.round(np.linspace(0, 1, 11), 2)
    rows = []
    for s in cfg["mlp_seeds"]:
        p_mlp = cross_val_predict(MLPEnsembleClassifier(random_state=s), Xs, ys, cv=folds, method="predict_proba")
        for k, p in p_other.items():
            for a in alphas:
                rows.append({"other": k, "mlp_seed": s, "alpha": a,
                             "accuracy": float(np.mean((a * p_mlp + (1 - a) * p).argmax(1) == ys))})
    df = pd.DataFrame(rows)
    run.save_table(df, "alpha_curves", latex=False)
    out = {}
    for k in others:
        sub = df[df["other"] == k]
        run.save_figure(viz.line(sub.assign(seed=sub["mlp_seed"].map("MLP seed {}".format)), "alpha", "accuracy",
                                 hue="seed", title=f"alpha x MLP + (1 - alpha) x {k.upper()}, single split"),
                        f"alpha_curve_{k}")
        best = sub.loc[sub.groupby("mlp_seed")["accuracy"].idxmax()]
        out[k] = {"best_alpha_by_seed": best["alpha"].tolist(), "best_accuracy_by_seed": best["accuracy"].tolist()}
        run.log.info("MLP+%s: best alpha per MLP seed %s (accuracy %s)", k, best["alpha"].tolist(),
                     np.round(best["accuracy"].to_numpy(), 2).tolist())
    return out


def selection_leak(run: Run, cfg: dict, X: np.ndarray, y: np.ndarray) -> dict:
    """Top-k features chosen on all patients vs inside the training folds, 10 random splits."""
    k = cfg["top_k_features"]
    Xs = StandardScaler().fit_transform(X)
    top = TopKImportance(k).fit(Xs, y).idx_
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rows = []
    for seed in range(10):
        folds = StratifiedKFold(5, shuffle=True, random_state=seed)
        rows.append({"seed": seed,
                     "all_features": cross_val_score(rf, Xs, y, cv=folds).mean(),
                     f"top{k}_selected_on_all": cross_val_score(rf, Xs[:, top], y, cv=folds).mean(),
                     f"top{k}_selected_in_folds": cross_val_score(Pipeline([("sel", TopKImportance(k)), ("rf", rf)]),
                                                                  Xs, y, cv=folds).mean()})
    df = pd.DataFrame(rows)
    run.save_table(df, "feature_selection_leak", latex=False)
    out = df.drop(columns="seed").agg(["mean", "std"]).to_dict()
    run.log.info("feature selection (RF, 10 splits): %s",
                 {c: f"{v['mean']:.3f} ± {v['std']:.3f}" for c, v in out.items()})
    out["selected_features"] = [VOLUMETRIC[i] for i in top]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="protocol_study.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    with Run(cfg.get("name", "protocol_study"), cfg) as run:
        X, _, y, _ = load_xy(cfg["feature_set"], "gt")
        ss = single_split(run, cfg, X, y)
        summary = {"single_split_best": ss["best"], "mlp_single_split_by_seed": ss["mlp_by_seed"]}
        summary["mlp_training_vs_cv"] = mlp_training_accuracy(run, ss["Xs"], ss["ys"], ss["folds"])
        summary["alpha"] = alpha_curves(run, cfg, ss["Xs"], ss["ys"], ss["folds"])
        summary["feature_selection"] = selection_leak(run, cfg, X, y)

        rows = []
        for name, spec in cfg["grids"].items():
            res = nested_cv(spec, X, X, y, n_repeats=cfg["n_repeats"], seed=cfg["seed"])
            run.save_table(res["per_repeat"], f"nested_{name}_per_repeat", latex=False)
            run.save_arrays(f"nested_{name}", proba=res["proba"], y=y)
            s = summarize(res["per_repeat"])
            rows.append({"model": name, "single_split": ss["best"].get(name, np.nan), **s,
                         "chosen_params": params_summary(res["params"])})
            run.log_metrics(rows[-1])
            run.log.info("nested %-12s accuracy = %.3f ± %.3f   (single split %.3f)  %s", name,
                         s["accuracy"], s["accuracy_std"], rows[-1]["single_split"], rows[-1]["chosen_params"])
            run.save_figure(viz.confusion(pooled_confusion(y, res["pred"]), f"{name}, nested CV"), f"confusion_{name}")
        df = pd.DataFrame(rows)
        metrics = ["accuracy", "balanced_accuracy", "macro_f1"]
        table = mean_std_text(df, metrics)[["model", "single_split"] + metrics + ["chosen_params"]]
        run.save_table(table, "comparison")
        long = pd.concat([df[["model"]].assign(protocol="single split (best of grid)", accuracy=df["single_split"], std=0.0),
                          df[["model"]].assign(protocol="nested CV", accuracy=df["accuracy"], std=df["accuracy_std"])])
        run.save_figure(viz.bars(long, "model", "accuracy", err="std", hue="protocol", ylim=(0.7, 1.0),
                                 title="single-split vs nested-CV accuracy"), "single_split_vs_nested")
        summary["nested"] = df[["model", "accuracy", "accuracy_std"]].to_dict("records")
        run.finish(summary)


if __name__ == "__main__":
    main()

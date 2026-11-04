"""Final model: nested-CV estimate, fit on all training patients, test predictions.

    python scripts/06_final_model.py --config final.yaml

* nested CV of the final configuration (accuracy, confusion, patients that are
  often misclassified);
* permutation importance of the features on the held-out folds;
* fit on the 100 training patients (grid search on all of them if the config
  has a grid) and prediction of the 50 test patients -> submission.csv
  (format of SampleSubmission.csv) + class probabilities.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from cardiac import config, viz
from cardiac.evaluate import (chosen_params, make_estimator, nested_cv, params_summary, pooled_confusion,
                              summarize)
from cardiac.experiment import feature_columns, load_config, load_test, load_xy
from cardiac.tracking import Run


def cv_permutation_importance(spec: dict, X: np.ndarray, y: np.ndarray, cols: list[str], seed: int,
                              n_repeats: int = 3) -> pd.DataFrame:
    """Drop in held-out accuracy when a feature is shuffled, averaged over the folds of ``n_repeats`` 5-fold CVs."""
    rows = []
    for r in range(n_repeats):
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(X, y):
            est = make_estimator(spec, seed + r).fit(X[tr], y[tr])
            imp = permutation_importance(est, X[va], y[va], n_repeats=20, random_state=seed, scoring="accuracy")
            rows.append(imp.importances_mean)
    imp = np.array(rows)
    return (pd.DataFrame({"feature": cols, "importance": imp.mean(0), "std": imp.std(0)})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="final.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    spec, fs, lv, seed = cfg["model"], cfg["feature_set"], cfg["lv"], cfg.get("seed", 0)
    torch.set_num_threads(4)

    with Run(cfg.get("name", "final_model"), cfg) as run:
        cols = feature_columns(fs)
        X, _, y, ids = load_xy(fs, lv)

        res = nested_cv(spec, X, X, y, n_repeats=cfg.get("n_repeats", 10), seed=seed)
        s = summarize(res["per_repeat"])
        run.save_table(res["per_repeat"], "nested_per_repeat", latex=False)
        run.save_table(res["params"], "nested_chosen_params", latex=False)
        run.log.info("nested CV: accuracy %.3f ± %.3f, balanced %.3f, macro-F1 %.3f; %s", s["accuracy"],
                     s["accuracy_std"], s["balanced_accuracy"], s["macro_f1"], params_summary(res["params"]))
        cm = pooled_confusion(y, res["pred"])
        run.save_table(cm.reset_index().rename(columns={"index": "true"}), "confusion", float_format="%.2f")
        run.save_figure(viz.confusion(cm, f"final model, nested CV ({cfg.get('n_repeats', 10)} repeats)"), "confusion")

        err = (res["pred"] != y).mean(0)
        hard = pd.DataFrame({"id": ids, "true": [config.CLASS_NAMES[c] for c in y], "error_rate": err,
                             "most_predicted": [config.CLASS_NAMES[np.bincount(p, minlength=5).argmax()]
                                                for p in res["pred"].T]})
        hard = hard[hard["error_rate"] > 0].sort_values("error_rate", ascending=False)
        run.save_table(hard, "misclassified_patients", float_format="%.2f")
        run.log.info("patients misclassified in at least one repeat: %d (always: %d)", len(hard),
                     int((hard["error_rate"] == 1).sum()))

        imp = cv_permutation_importance(spec, X, y, cols, seed)
        run.save_table(imp, "permutation_importance")
        run.save_figure(viz.bars(imp.head(20), "feature", "importance", err="std",
                                 title="permutation importance on held-out folds (accuracy drop)"),
                        "permutation_importance")

        # ------------------------------------------------------------ fit on all patients, predict test
        final = make_estimator(spec, seed).fit(X, y)
        params = chosen_params(final)
        run.log.info("final fit on %d patients, chosen params: %s", len(y), params)
        X_test, test_ids = load_test(fs, lv)
        proba = final.predict_proba(X_test)
        classes = np.unique(y)
        pred = classes[proba.argmax(1)]
        sample = pd.read_csv(config.META_DIR / "SampleSubmission.csv")
        id_col, cls_col = sample.columns[:2]
        submission = pd.DataFrame({id_col: test_ids.astype(int), cls_col: pred})
        submission.to_csv(run.dir / "submission.csv", index=False)
        probs = pd.DataFrame(proba, columns=[config.CLASS_NAMES[c] for c in classes]).assign(
            id=test_ids, predicted=[config.CLASS_NAMES[c] for c in pred], confidence=proba.max(1))
        run.save_table(probs[["id", "predicted", "confidence"] + [config.CLASS_NAMES[c] for c in classes]],
                       "test_probabilities", latex=False)
        dist = pd.Series(pred).map(config.CLASS_NAMES).value_counts().reindex(config.CLASS_NAMES.values(), fill_value=0)
        run.log.info("test predictions per class: %s", dist.to_dict())

        run.finish({"nested_cv": {k: s[k] for k in ("accuracy", "accuracy_std", "balanced_accuracy", "macro_f1")},
                    "final_params": params, "test_class_counts": dist.to_dict(),
                    "test_mean_confidence": float(proba.max(1).mean()),
                    "top_features": imp.head(5)["feature"].tolist()})


if __name__ == "__main__":
    main()

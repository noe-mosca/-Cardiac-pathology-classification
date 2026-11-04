"""Nested cross-validation of a grid: LV protocols x feature sets x models.

    python scripts/05_benchmark.py --config benchmark.yaml
    python scripts/05_benchmark.py --config lv_protocol.yaml

A protocol is the LV source of the training patients (``fit``) and of the
evaluated patients (``eval``): ``gt -> fill`` trains on the true LV and
evaluates on reconstructed LVs, as a model trained on the training set and
applied to the test set would be.
Every configuration uses the same outer splits, so differences between
configurations are paired: ``delta_vs_best`` is the per-repeat difference with
the best configuration of the same protocol (mean and std over repeats).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from cardiac import viz
from cardiac.evaluate import nested_cv, params_summary, pooled_confusion, summarize
from cardiac.experiment import load_config, load_xy, mean_std_text
from cardiac.tracking import Run

METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "recall_NOR", "recall_MINF", "recall_DCM", "recall_HCM",
           "recall_ARV"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--repeats", type=int, default=None, help="override n_repeats (quick tests)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    n_repeats = args.repeats or cfg.get("n_repeats", 10)
    models = cfg["models"]
    if only := cfg.get("only_models"):
        models = {k: v for k, v in models.items() if k in only}

    with Run(cfg.get("name", Path(args.config).stem), {**cfg, "n_repeats": n_repeats}) as run:
        rows, per_repeat = [], {}
        for proto in cfg["protocols"]:
            pname = f"{proto['fit']}->{proto['eval']}"
            for fs in cfg["feature_sets"]:
                X_fit, X_eval, y, ids = load_xy(fs, proto["fit"], proto["eval"])
                for mname, spec in models.items():
                    res = nested_cv(spec, X_fit, X_eval, y, n_repeats=n_repeats, seed=cfg.get("seed", 0))
                    key = (pname, fs, mname)
                    per_repeat[key] = res["per_repeat"]["accuracy"].to_numpy()
                    tag = f"{pname}_{fs}_{mname}".replace(">", "").replace("+", "_")
                    run.save_arrays(tag, proba=res["proba"], y=y, ids=ids)
                    row = {"protocol": pname, "features": fs, "n_features": X_fit.shape[1], "model": mname,
                           **summarize(res["per_repeat"]), "chosen_params": params_summary(res["params"])}
                    rows.append(row)
                    run.log_metrics(row)
                    run.log.info("[%s | %s] %-12s acc=%.3f ± %.3f  bacc=%.3f  %s", pname, fs, mname, row["accuracy"],
                                 row["accuracy_std"], row["balanced_accuracy"], row["chosen_params"])
                    pd.DataFrame(res["pred"].T, index=ids, columns=[f"repeat{r}" for r in range(n_repeats)]) \
                        .assign(category=y).to_csv(run.dir / "arrays" / f"{tag}_pred.csv")

        df = pd.DataFrame(rows)
        for pname, g in df.groupby("protocol", sort=False):
            best = g.loc[g["accuracy"].idxmax()]
            ref = per_repeat[(pname, best["features"], best["model"])]
            for i in g.index:
                d = per_repeat[(pname, df.at[i, "features"], df.at[i, "model"])] - ref
                df.loc[i, ["delta_vs_best", "delta_vs_best_std"]] = d.mean(), d.std()
        df.to_csv(run.dir / "metrics_full.csv", index=False)
        text = mean_std_text(df, METRICS + ["delta_vs_best"])
        run.save_table(text[["protocol", "features", "model"] + METRICS[:3] + ["delta_vs_best", "chosen_params"]],
                       "summary")
        run.save_table(text[["protocol", "features", "model"] + METRICS[3:]], "recall_per_class")
        run.save_table(df.drop(columns="chosen_params"), "summary_mean", latex=False)

        for pname, g in df.groupby("protocol", sort=False):
            safe = pname.replace(">", "")
            if g["features"].nunique() > 1:
                pivot = g.pivot_table(index="model", columns="features", values="accuracy", sort=False)
                run.save_figure(viz.heatmap(pivot, title=f"nested-CV accuracy ({pname})"), f"heatmap_{safe}")
            run.save_figure(viz.bars(g, "model", "accuracy", err="accuracy_std", hue="features", ylim=(0.6, 1.0),
                                     title=f"nested-CV accuracy, mean ± std over {n_repeats} repeats ({pname})"),
                            f"accuracy_{safe}")
        if df["protocol"].nunique() > 1:
            for fs, g in df.groupby("features", sort=False):
                run.save_figure(viz.bars(g, "model", "accuracy", err="accuracy_std", hue="protocol", ylim=(0.6, 1.0),
                                         title=f"LV protocol (training -> evaluated), {fs}"), f"protocols_{fs}")
        top = df.sort_values("accuracy", ascending=False).groupby("protocol", sort=False).head(1)
        for r in top.itertuples():
            tag = f"{r.protocol}_{r.features}_{r.model}".replace(">", "").replace("+", "_")
            pred = pd.read_csv(run.dir / "arrays" / f"{tag}_pred.csv", index_col=0, dtype={0: str})
            cm = pooled_confusion(pred["category"].to_numpy(), pred.drop(columns="category").to_numpy().T)
            run.save_figure(viz.confusion(cm, f"{r.model}, {r.features} ({r.protocol})"),
                            f"confusion_best_{r.protocol.replace('>', '')}")
        run.finish({"best_per_protocol": top[["protocol", "features", "model", "accuracy", "accuracy_std",
                                             "balanced_accuracy"]].to_dict("records")})


if __name__ == "__main__":
    main()

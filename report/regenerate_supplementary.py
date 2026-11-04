"""Regenerate the appendix figures and tables from final/results.

    python3 report/regenerate_supplementary.py      # from the repository root (locally: overleaf/ from "IM05 Mosca Noé")

Writes figs/supp/*.png and tabs/supp/*.tex, included by sec_merge/8_appendix.tex.
Everything is read from the experiment outputs, so rerunning final/scripts then this script
keeps the appendix in sync.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
# The code is in "IM05 Mosca Noé/final" locally, and at the root of the GitHub repository (report/ next to it).
FINAL = ROOT / "final" if (ROOT / "final" / "cardiac").exists() else ROOT
RESULTS = FINAL / "results"
FIGS = Path(__file__).resolve().parent / "figs/supp"
TABS = Path(__file__).resolve().parent / "tabs/supp"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FINAL))

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "lines.markersize": 4, "savefig.bbox": "tight", "savefig.dpi": 300,
})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fc", "#2a78d6", "#0b2f66"])
FULL_W, COL_W = 5.5, 3.33  # ICLR text width, half-width figures

CLASSES = ["NOR", "MINF", "DCM", "HCM", "ARV"]
MODEL_NAMES = {"logreg": "Logistic regression", "svm_linear": "Linear SVM", "svm_rbf": "RBF SVM",
               "knn": "$k$-NN", "rf": "Random forest", "extra_trees": "Extra trees", "xgboost": "XGBoost",
               "mlp_ensemble": "MLP ensemble", "mlp+rf": "MLP + RF", "mlp+svm": "MLP + linear SVM",
               "svm": "SVM (linear or RBF)"}
FS_NAMES = {"volumetric": "Volumetric", "volumetric+thickness": "Volumetric + thickness", "compact": "Compact"}

# Definitions of the 32 features (LV = left-ventricular cavity, RV = right-ventricular cavity).
FEATURE_DEFS = {
    "ed_lv_vol": "LV volume at ED (mL)", "es_lv_vol": "LV volume at ES (mL)",
    "lv_ef": "LV ejection fraction, $100\\,(V_{ED}-V_{ES})/V_{ED}$ (\\%)",
    "ed_rv_vol": "RV volume at ED (mL)", "es_rv_vol": "RV volume at ES (mL)", "rv_ef": "RV ejection fraction (\\%)",
    "myocardium_mass": "Myocardial mass, $1.05$ g/mL $\\times$ volume, mean of ED and ES (g)",
    "lv_mass_index": "Myocardial mass / BSA (g/m$^2$)", "stroke_volume": "LV stroke volume, $V_{ED}-V_{ES}$ (mL)",
    "es_lv_index": "LV ES volume / BSA", "ed_rv_index": "RV ED volume / BSA",
    "ratio_rv_lv_es": "RV / LV volume at ES", "ratio_rv_lv_ed": "RV / LV volume at ED",
    "weight": "Weight (kg)", "height": "Height (cm)", "bsa": "Body surface area, Mosteller (m$^2$)",
    "ed_lv_index": "LV ED volume / BSA", "es_rv_index": "RV ES volume / BSA",
    "myocardium_to_lv_ed": "ED myocardial mass / LV ED volume", "mass_to_edv_ratio": "Mean myocardial mass / LV ED volume",
    "mass_ratio_systole_diastole": "ES / ED myocardial mass",
}
for ph in ("ed", "es"):
    P = ph.upper()
    FEATURE_DEFS.update({
        f"{ph}_thick_mean": f"{P} wall thickness, mean over all samples (mm)",
        f"{ph}_thick_max": f"{P} wall thickness, maximum sample (mm)",
        f"{ph}_thick_slicemean_max": f"{P} wall thickness, largest per-slice mean (mm)",
        f"{ph}_thick_slicestd_mean": f"{P} wall thickness, mean of per-slice std (mm)",
        f"{ph}_thick_slicemean_std": f"{P} wall thickness, std of per-slice means (mm)",
    })
FEATURE_DEFS["wall_thickening"] = "Systolic thickening, ES / ED mean thickness"


def num(x, nd=3):
    """Fixed decimals with a proper minus sign; '--' for missing values."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}".replace("-", "$-$")


def num_pm(m, s, nd=3):
    return num(m, nd) if pd.isna(s) else f"{num(m, nd)} $\\pm$ {num(s, nd)}"


def tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&")


def tt(s: str) -> str:
    return f"\\texttt{{{tex_escape(s)}}}"


def save_fig(fig, name, jpg=False):
    path = FIGS / f"{name}.{'jpg' if jpg else 'png'}"
    fig.savefig(path, **({"dpi": 200, "pil_kwargs": {"quality": 88}} if jpg else {}))
    plt.close(fig)
    print("wrote", path)


def write_tex(name, text):
    path = TABS / f"{name}.tex"
    path.write_text(text, encoding="utf-8")
    print("wrote", path)


def table(rows, header, colspec, caption, label, star=False, size=r"\scriptsize", source=""):
    # Appendix floats stay in their subsection ([H], float package); wide tables shrink to the text width.
    body = "\n".join(r if isinstance(r, str) else " & ".join(map(str, r)) + r" \\" for r in rows)
    return (f"% Generated by regenerate_supplementary.py from {source}\n"
            f"\\begin{{table}}[H]\n\\centering\n{size}\n\\setlength{{\\tabcolsep}}{{4pt}}\n"
            f"\\begin{{adjustbox}}{{max width=\\linewidth}}\n"
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n"
            f"\\end{{tabular}}\n\\end{{adjustbox}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n")


def heatmap(ax, df, fmt="{:.2f}", vmin=None, vmax=None):
    data = df.to_numpy(float)
    im = ax.imshow(data, cmap=SEQ, aspect="auto", vmin=vmin, vmax=vmax)
    lo, hi = np.nanmin(data) if vmin is None else vmin, np.nanmax(data) if vmax is None else vmax
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                continue
            dark = (data[i, j] - lo) / (hi - lo + 1e-12) > 0.55
            ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if dark else "#0b0b0b")
    ax.set_xticks(range(df.shape[1]), df.columns)
    ax.set_yticks(range(df.shape[0]), df.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


from cardiac import config  # noqa: E402
from cardiac.data import load_seg  # noqa: E402

# ====================================================================== A. data
stats = pd.read_csv(RESULTS / "dataset/tables/class_stats.csv")
summary = json.loads((RESULTS / "dataset/summary.json").read_text())
write_tex("class_stats", table(
    [(tex_escape(r.cls), int(r.n), num(r.height, 1), num(r.weight, 1), num(r.slices, 1), num(r.spacing_xy, 2),
      num(r.spacing_z, 1)) for r in stats.itertuples()],
    r"Group & Patients & Height (cm) & Weight (kg) & Slices & In-plane (mm) & Slice step (mm)", "lrrrrrr",
    "Patients per class (means). The training set has 20 patients per class; the test set has the same number of "
    f"patients per class in ACDC, but its labels are not available. Slice counts range from {summary['slices_range'][0]} "
    f"to {summary['slices_range'][1]}; slice steps take the values "
    + ", ".join(f"{v:g}" for v in summary["spacing_z_values"]) + " mm.",
    "tab:supp-classes", size=r"\small", source="results/dataset/tables/class_stats.csv"))

# ====================================================================== B. features
feats = pd.read_csv(FINAL / "cache/features/train_fill_hull.csv", dtype={"id": str})
feats["cls"] = feats["category"].map(config.CLASS_NAMES)
from cardiac.features import COMPACT, THICKNESS, VOLUMETRIC  # noqa: E402

rows = []
for f in VOLUMETRIC + THICKNESS:
    group = "V" if f in VOLUMETRIC else "T"
    means = feats.groupby("cls")[f].mean().loc[CLASSES]
    rows.append((tt(f), FEATURE_DEFS[f], group + (", C" if f in COMPACT else ""),
                 *[num(v, 2 if abs(v) < 10 else 1) for v in means]))
write_tex("features_full", table(
    rows, r"Feature & Definition & Set & NOR & MINF & DCM & HCM & ARV", r"lp{0.4\linewidth}crrrrr",
    "The 32 features and their mean per class (training set, reconstructed LV). \\emph{Set}: V = volumetric, "
    "T = wall thickness, C = compact set. Volumes in mL, indices per m$^2$ of BSA.",
    "tab:supp-features", source="final/cache/features/train_fill_hull.csv"))

fig, axes = plt.subplots(3, 4, figsize=(FULL_W, 4.4))
for ax, col in zip(axes.ravel(), THICKNESS):
    data = [feats.loc[feats["cls"] == c, col] for c in CLASSES]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, medianprops={"color": "#0b0b0b", "lw": 1},
                    flierprops={"markersize": 2, "markeredgecolor": GREY}, whiskerprops={"color": GREY},
                    capprops={"color": GREY}, boxprops={"color": GREY})
    for patch in bp["boxes"]:
        patch.set_facecolor(ORANGE)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, 6), CLASSES, fontsize=6)
    ax.set_title(col.replace("_", " "), fontsize=7)
    ax.grid(axis="x", visible=False)
for ax in axes.ravel()[len(THICKNESS):]:
    ax.axis("off")
fig.tight_layout(h_pad=0.8, w_pad=0.6)
save_fig(fig, "thickness_by_class")

# ====================================================================== C. LV reconstruction
imp = pd.read_csv(RESULTS / "lv_reconstruction/tables/per_volume.csv", dtype={"id": str})
fill = imp[imp["method"] == "fill"].set_index(["id", "phase"])
hull = imp[imp["method"] == "fill_hull[min_hull_fraction=0.25]"].set_index(["id", "phase"])
bad = fill[fill["vol_err_ml"].abs() > 1e-9].sort_values("vol_err_ml")
ef = pd.read_csv(RESULTS / "lv_reconstruction/tables/ef_per_patient.csv", dtype={"id": str})
ef_fill = ef[ef["method"] == "fill"].set_index("id")["ef_err"]
ef_hull = ef[ef["method"] == "fill_hull[min_hull_fraction=0.25]"].set_index("id")["ef_err"]
write_tex("lv_imperfect", table(
    [(pid, ph, r.category, num(r.vol_gt_ml, 1), num(r.vol_err_ml, 1), num(r.dice, 3),
      num(hull.loc[(pid, ph), "vol_err_ml"], 1), num(hull.loc[(pid, ph), "dice"], 3), num(ef_fill[pid], 1),
      num(ef_hull[pid], 1)) for (pid, ph), r in bad.iterrows()],
    r"Patient & Phase & Class & True LV (mL) & \multicolumn{2}{c}{Filling: $\Delta V$, Dice} & "
    r"\multicolumn{2}{c}{+ convex hull: $\Delta V$, Dice} & \multicolumn{2}{c}{$\Delta$EF (points)}", "lllrrrrrrr",
    "The training volumes that the filling does not reconstruct exactly. Each one misses its most basal slice "
    "(slice 0, or 1 for patient 096), where the myocardium does not close around the cavity. The two $\\Delta$EF "
    "columns are for the filling and for the filling with the convex-hull fallback.",
    "tab:supp-lv", source="results/lv_reconstruction/tables/per_volume.csv"))

shift = pd.read_csv(RESULTS / "features/tables/lv_source_shift.csv")
top = shift[shift["patients_changed"] > 0].pivot_table(index="feature", columns="lv_source",
                                                       values="max_rel_change").fillna(0)
top = top.sort_values("fill", ascending=False).head(10)
write_tex("lv_shift", table(
    [(tt(f), num(100 * r["fill"], 1), num(100 * r.get("fill_hull", 0), 1)) for f, r in top.iterrows()],
    r"Feature & Filling & + convex hull", "lrr",
    "Largest relative change (\\%, over the 100 training patients) of a feature when the LV is reconstructed instead "
    "of taken from the ground truth. Only 7 patients change at all; the ejection fraction of one of them changes by "
    "98\\% with the filling.",
    "tab:supp-shift", size=r"\small", source="results/features/tables/lv_source_shift.csv"))

# ====================================================================== D. protocol study
fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 1.75), sharey=False)
for ax, m in zip(axes, ("rf", "xgboost", "svm", "knn")):
    g = pd.read_csv(RESULTS / f"protocol_study/tables/grid_single_split_{m}.csv")["accuracy"].round(2)
    vc = g.value_counts().sort_index()
    ax.bar(vc.index, vc.values, width=0.008, color=BLUE)
    ax.bar(vc.index[-1], vc.values[-1], width=0.008, color=ORANGE)
    ax.set(title=f"{MODEL_NAMES[m].replace(' (linear or RBF)', '')}\n({len(g)} configurations)", xlabel="Accuracy")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Configurations")
fig.tight_layout(w_pad=0.8)
save_fig(fig, "grid_scores")

mlp = pd.read_csv(RESULTS / "protocol_study/tables/mlp_training_vs_cv.csv")
sel = pd.read_csv(RESULTS / "protocol_study/tables/feature_selection_leak.csv")
write_tex("mlp_training", table(
    [(int(r.epochs), num(r.training_accuracy, 2), num(r.cv_accuracy, 2)) for r in mlp.itertuples()],
    r"Epochs & Accuracy on its training patients & 5-fold CV accuracy", "rrr",
    "Ensemble of 10 MLPs (same architecture as the final ensemble, learning rate $10^{-3}$, batch 30, 75\\% of the "
    "patients per MLP), scored on the patients it was trained on and by cross-validation.",
    "tab:supp-mlp", size=r"\small", source="results/protocol_study/tables/mlp_training_vs_cv.csv"))
cols = [c for c in sel.columns if c != "seed"]
labels = {"all_features": "All 21 features", "top16_selected_on_all": "Top 16, selected on all patients",
          "top16_selected_in_folds": "Top 16, selected inside each training fold"}
write_tex("feature_selection", table(
    [(labels[c], num_pm(sel[c].mean(), sel[c].std())) for c in cols],
    r"Random forest (200 trees) & Accuracy", "lr",
    "Feature selection by random-forest importance, 10 random 5-fold splits (mean $\\pm$ std). Selecting on all "
    "patients leaks the evaluation folds, but the effect is negligible here.",
    "tab:supp-selection", size=r"\small", source="results/protocol_study/tables/feature_selection_leak.csv"))
# ====================================================================== E. full benchmark
bench = pd.read_csv(RESULTS / "benchmark/tables/summary_mean.csv")
params = pd.read_csv(RESULTS / "benchmark/tables/summary.csv")
rows, prev = [], None
for r in bench.itertuples():
    if prev is not None and r.features != prev:
        rows.append("\\midrule")
    prev = r.features
    rows.append((FS_NAMES[r.features] if not rows or rows[-1] == "\\midrule" else "", MODEL_NAMES[r.model],
                 num_pm(r.accuracy, r.accuracy_std), num(r.macro_f1),
                 *[num(getattr(r, f"recall_{c}"), 2) for c in CLASSES],
                 num_pm(r.delta_vs_best, r.delta_vs_best_std)))
write_tex("benchmark_full", table(
    rows, r"Features & Model & Accuracy & Macro-F1 & \multicolumn{5}{c}{Recall: NOR, MINF, DCM, HCM, ARV} & $\Delta$ vs best",
    "llrrrrrrrr",
    "All configurations of the benchmark (nested CV, 10 repeats, reconstructed LV). $\\Delta$ vs best: per-repeat "
    "difference with the best configuration (MLP + linear SVM, volumetric + thickness), mean $\\pm$ std; since all "
    "configurations share the same splits, this paired difference is less noisy than the accuracies themselves.",
    "tab:supp-benchmark", source="results/benchmark/tables/summary_mean.csv"))
rows = []
for fs in FS_NAMES:
    for r in params[params["features"] == fs].itertuples():
        if isinstance(r.chosen_params, str):
            rows.append((FS_NAMES[fs], MODEL_NAMES[r.model], tex_escape(r.chosen_params)))
write_tex("hyperparams", table(
    rows, r"Features & Model & Most frequent choice of the inner CV (count over the 50 outer folds)", r"llp{0.58\linewidth}",
    "Hyper-parameters selected by the inner cross-validation. Grids: $C\\in\\{0.01,0.1,1,10,100\\}$ for logistic "
    "regression and the linear SVM; $C\\in\\{0.1,1,10,100\\}$, $\\gamma\\in\\{$scale$, 0.01, 0.1\\}$ for the RBF SVM; "
    "$k\\in\\{3,5,7,9\\}$, uniform or distance weights, Euclidean or Manhattan for $k$-NN; depth $\\in\\{$none$, 5\\}$, "
    "features per split $\\in\\{\\sqrt{d}, d/2\\}$ for the forests (300 trees); 100 or 300 trees, depth 2 or 3, "
    "learning rate 0.05 or 0.1, column subsampling 0.6 or 1 for XGBoost; $\\alpha\\in\\{0,0.1,\\dots,1\\}$ for the "
    "soft votes. A frequent \\texttt{None} depth means fully grown trees.",
    "tab:supp-hyper", source="results/benchmark/tables/summary.csv"))

rec = bench[bench["features"] == "volumetric+thickness"].set_index("model")[[f"recall_{c}" for c in CLASSES]]
rec.index = [MODEL_NAMES[m] for m in rec.index]
rec.columns = CLASSES
fig, ax = plt.subplots(figsize=(COL_W, 2.6))
heatmap(ax, rec, vmin=0.75, vmax=1.0)
ax.set_title("Recall per class, volumetric + thickness features")
save_fig(fig, "recall_heatmap")

# ====================================================================== F. final model
per_rep = pd.read_csv(RESULTS / "final_model/tables/nested_per_repeat.csv")
write_tex("final_repeats", table(
    [(int(r.repeat), num(r.accuracy, 2), *[num(getattr(r, f"recall_{c}"), 2) for c in CLASSES])
     for r in per_rep.itertuples()]
    + ["\\midrule", ("Mean", num_pm(per_rep["accuracy"].mean(), per_rep["accuracy"].std()),
                     *[num(per_rep[f"recall_{c}"].mean(), 3) for c in CLASSES])],
    r"Repeat & Accuracy & NOR & MINF & DCM & HCM & ARV", "rrrrrrr",
    "Final model, nested CV: accuracy and recall per class for each of the 10 repeats (100 patients each).",
    "tab:supp-repeats", size=r"\small", source="results/final_model/tables/nested_per_repeat.csv"))
mis = pd.read_csv(RESULTS / "final_model/tables/misclassified_patients.csv", dtype={"id": str})
idx = feats.set_index("id")
write_tex("misclassified", table(
    [(r.id, r.true, r.most_predicted, num(r.error_rate, 1), num(idx.loc[r.id, "lv_ef"], 1),
      num(idx.loc[r.id, "ed_lv_index"], 1), num(idx.loc[r.id, "es_thick_slicestd_mean"], 2),
      num(idx.loc[r.id, "rv_ef"], 1)) for r in mis.itertuples()],
    r"Patient & True & Most predicted & Error rate & LV EF & LV EDV / BSA & ES thick.\ std & RV EF", "lllrrrrr",
    "Patients misclassified in at least one of the 10 repeats of the final model, with four of their features "
    "(class means in Table~\\ref{tab:supp-features}).",
    "tab:supp-errors", size=r"\small", source="results/final_model/tables/misclassified_patients.csv"))
imp_all = pd.read_csv(RESULTS / "final_model/tables/permutation_importance.csv")
half = (len(imp_all) + 1) // 2
left, right = imp_all.iloc[:half], imp_all.iloc[half:]
rows = []
for i in range(half):
    l = left.iloc[i]
    cells = [tt(l.feature), num_pm(l.importance, l["std"])]
    if i < len(right):
        r = right.iloc[i]
        cells += [tt(r.feature), num_pm(r.importance, r["std"])]
    else:
        cells += ["", ""]
    rows.append(cells)
write_tex("importance_full", table(
    rows, r"Feature & Accuracy drop & Feature & Accuracy drop", "lrlr",
    "Permutation importance of every feature for the final model: drop of held-out accuracy when the feature is "
    "shuffled (20 permutations, 15 folds of 3 repeated 5-fold CVs; mean $\\pm$ std over folds). Correlated features "
    "share their importance, so low values do not mean the information is useless.",
    "tab:supp-importance", source="results/final_model/tables/permutation_importance.csv"))

probs = pd.read_csv(RESULTS / "final_model/tables/test_probabilities.csv", dtype={"id": str})
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 1.8))
counts = probs["predicted"].value_counts().reindex(CLASSES, fill_value=0)
axes[0].bar(CLASSES, counts.values, color=BLUE)
axes[0].axhline(10, color=GREY, ls="--", lw=0.8)
axes[0].set(ylabel="Test patients", title="(a) Predicted class (ACDC: 10 per class)")
axes[0].grid(axis="x", visible=False)
axes[1].hist(probs["confidence"], bins=np.linspace(0.3, 1, 15), color=BLUE, edgecolor="white", linewidth=0.6)
axes[1].set(xlabel="Probability of the predicted class", ylabel="Test patients", title="(b) Confidence")
fig.tight_layout(w_pad=1.5)
save_fig(fig, "test_predictions")

# ====================================================================== G. reproducibility
runs = pd.read_csv(RESULTS / "runs.csv")
runs = runs[runs["status"] == "ok"].drop_duplicates("name", keep="last")
desc = {"dataset": ("Patient index, file checks", 1), "lv_reconstruction": ("LV reconstruction on the training set", 2),
        "features": ("Features for every LV source", 3), "protocol_study": ("Protocol study (single split vs nested CV)", 4),
        "benchmark": ("Benchmark (30 configurations, nested CV)", 5), "lv_protocol": ("LV protocols (32 configurations)", 5),
        "final_model": ("Final model and test predictions", 6)}
write_tex("runs", table(
    [(desc[r.name][0], desc[r.name][1], num(r.seconds / 60, 1)) for r in runs.itertuples() if r.name in desc],
    r"Experiment & Script & Time (min)", "lcr",
    "Runs that produced the results of this report (Apple M3, 16 GB, 8 CPU cores; no GPU). Most of the time is spent "
    "training MLP ensembles inside the nested cross-validation.",
    "tab:supp-runs", size=r"\small", source="results/runs.csv"))

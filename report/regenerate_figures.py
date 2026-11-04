"""Regenerate the figures and tables of the main text from the experiment outputs in final/results.

    python3 report/regenerate_figures.py      # from the repository root (locally: overleaf/ from "IM05 Mosca Noé")

Reads the tables written by final/scripts/01..06, the cached features (final/cache/features)
and, for the example slices, the segmentations themselves through the cardiac package.
Writes figs/*.png and tabs/{protocol,benchmark,lv_reconstruction,lv_protocol}.tex.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

ROOT = Path(__file__).resolve().parent.parent
# The code is in "IM05 Mosca Noé/final" locally, and at the root of the GitHub repository (report/ next to it).
FINAL = ROOT / "final" if (ROOT / "final" / "cardiac").exists() else ROOT
RESULTS = FINAL / "results"
OUT = Path(__file__).resolve().parent / "figs"
TABS = Path(__file__).resolve().parent / "tabs"
OUT.mkdir(exist_ok=True)
TABS.mkdir(exist_ok=True)

sys.path.insert(0, str(FINAL))

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "lines.markersize": 4, "savefig.bbox": "tight", "savefig.dpi": 300,
})
# Categorical slots 1-3 (validated all-pairs), then a recessive grey for references.
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fc", "#2a78d6", "#0b2f66"])
COL_W = 3.33  # half-width figures
FULL_W = 5.5  # ICLR text width (in)

CLASSES = ["NOR", "MINF", "DCM", "HCM", "ARV"]
MODEL_NAMES = {"logreg": "Logistic regression", "svm_linear": "Linear SVM", "svm_rbf": "RBF SVM",
               "knn": "$k$-NN", "rf": "Random forest", "extra_trees": "Extra trees", "xgboost": "XGBoost",
               "mlp_ensemble": "MLP ensemble", "mlp+rf": "MLP + RF", "mlp+svm": "MLP + linear SVM",
               "svm": "SVM (linear or RBF)"}
FS_NAMES = {"volumetric": "Volumetric (21)", "volumetric+thickness": "Volumetric + thickness (32)", "compact": "Compact (12)"}
PROTO_NAMES = {"gt->gt": "true $\\to$ true", "gt->fill": "true $\\to$ fill", "fill->fill": "fill $\\to$ fill",
               "fill_hull->fill_hull": "fill+hull $\\to$ fill+hull"}
# Label overlay: RV aqua, myocardium blue, LV orange.
LABEL_CMAP = ListedColormap([(0, 0, 0, 0), (*plt.matplotlib.colors.to_rgb(AQUA), 0.55),
                             (*plt.matplotlib.colors.to_rgb(BLUE), 0.55), (*plt.matplotlib.colors.to_rgb(ORANGE), 0.55)])


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def write_tex(name, text):
    path = TABS / f"{name}.tex"
    path.write_text(text, encoding="utf-8")
    print("wrote", path)


def pm(m, s, nd=3, bold=False):
    t = f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"
    return f"\\textbf{{{m:.{nd}f}}} $\\pm$ {s:.{nd}f}" if bold else t


def clean_axis(ax):
    ax.set(xticks=[], yticks=[])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def crop_box(seg, margin=18):
    """Square box around the labeled region, so that all panels have the same aspect."""
    xs, ys = np.nonzero(seg > 0)
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) // 2 + margin
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    cx, cy = min(max(cx, half), seg.shape[0] - half), min(max(cy, half), seg.shape[1] - half)
    return slice(cx - half, cx + half), slice(cy - half, cy + half)


from cardiac import config  # noqa: E402
from cardiac.data import load_image, load_seg  # noqa: E402
from cardiac.lv import reconstruct_lv  # noqa: E402

index = pd.read_csv(RESULTS / "dataset/tables/index.csv", dtype={"id": str})

# ---------------------------------------------------------------------- example slices, one patient per class
train = index[index["split"] == "Train"]
fig, axes = plt.subplots(1, 5, figsize=(FULL_W, 1.45))
for ax, (c, name) in zip(axes, config.CLASS_NAMES.items()):
    pid = train.loc[train["category"] == c, "id"].iloc[0]
    seg, _ = load_seg("Train", pid, "ED")
    img = load_image("Train", pid, "ED")
    z = int(np.argmax((seg == config.LV).sum((0, 1))))
    bx, by = crop_box(seg[:, :, z], margin=22)
    ax.imshow(img[bx, by, z].T, cmap="gray")
    ax.imshow(seg[bx, by, z].T, cmap=LABEL_CMAP, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(f"{name} (patient {pid})", fontsize=7, pad=2)
    clean_axis(ax)
fig.tight_layout(w_pad=0.3)
save(fig, "data_overview")

# ---------------------------------------------------------------------- LV reconstruction
per_vol = pd.read_csv(RESULTS / "lv_reconstruction/tables/per_volume.csv", dtype={"id": str})
imperfect = pd.read_csv(RESULTS / "lv_reconstruction/tables/imperfect_volumes.csv", dtype={"id": str})
case = imperfect[imperfect["method"] == "fill"].sort_values("vol_err_ml").iloc[0]  # largest volume error
seg, _ = load_seg("Train", case["id"], case["phase"])
img = load_image("Train", case["id"], case["phase"])
gt = seg == config.LV
fill, hull = reconstruct_lv(seg, "fill"), reconstruct_lv(seg, "fill_hull")
z = int(np.argmax(gt.any((0, 1)) & ~fill.any((0, 1))))
bx, by = crop_box(seg[:, :, z], margin=14)
fig = plt.figure(figsize=(FULL_W, 1.75))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.75], wspace=0.08)
for i, (lv, title) in enumerate(((gt, "(a) ground truth"), (fill, "(b) filling"), (hull, "(c) + convex hull"))):
    ax = fig.add_subplot(gs[i])
    s = np.where(seg == config.LV, 0, seg)[:, :, z].copy()
    s[lv[:, :, z]] = config.LV
    ax.imshow(img[bx, by, z].T, cmap="gray")
    ax.imshow(s[bx, by].T, cmap=LABEL_CMAP, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(title, fontsize=7, pad=2)
    clean_axis(ax)
ax = fig.add_subplot(gs[3])
for method, label, color, marker in (("fill", "filling", ORANGE, "o"),
                                     ("fill_hull[min_hull_fraction=0.25]", "+ convex hull", BLUE, "s")):
    d = per_vol[per_vol["method"] == method]
    ax.scatter((d["vol_gt_ml"] + d["vol_rec_ml"]) / 2, d["vol_rec_ml"] - d["vol_gt_ml"], s=9, color=color,
               marker=marker, label=label, linewidths=0, alpha=0.85)
ax.axhline(0, color=GREY, lw=0.8)
ax.set(xlabel="Mean of true and reconstructed LV volume (mL)", ylabel="Error (mL)", title="(d) 200 training volumes")
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.legend(frameon=False, loc="lower right")
fig.subplots_adjust(left=0.01, right=0.9)
save(fig, "lv_reconstruction")
print(f"LV example: patient {case['id']} {case['phase']} slice {z}, error {case['vol_err_ml']:.1f} mL")

# ---------------------------------------------------------------------- features by class
feats = pd.read_csv(FINAL / "cache/features/train_fill_hull.csv", dtype={"id": str})
feats["cls"] = feats["category"].map(config.CLASS_NAMES)
panels = (("lv_ef", "LV ejection fraction (%)"), ("ed_lv_index", "LV ED volume / BSA (mL/m$^2$)"),
          ("rv_ef", "RV ejection fraction (%)"), ("lv_mass_index", "LV mass / BSA (g/m$^2$)"),
          ("ed_thick_slicemean_max", "ED wall, thickest slice (mm)"),
          ("es_thick_slicestd_mean", "ES wall, thickness std (mm)"))
fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 3.1))
for ax, (col, title) in zip(axes.ravel(), panels):
    data = [feats.loc[feats["cls"] == c, col] for c in CLASSES]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, medianprops={"color": "#0b0b0b", "lw": 1},
                    flierprops={"markersize": 2, "markeredgecolor": GREY}, whiskerprops={"color": GREY},
                    capprops={"color": GREY}, boxprops={"color": GREY})
    for patch, c in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(ORANGE if col.startswith(("ed_thick", "es_thick")) else BLUE)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, 6), CLASSES)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
fig.tight_layout(h_pad=1.0, w_pad=1.0)
save(fig, "features_by_class")

# ---------------------------------------------------------------------- protocol study
nested = pd.read_csv(RESULTS / "protocol_study/metrics.csv")
alpha = pd.read_csv(RESULTS / "protocol_study/tables/alpha_curves.csv")
# Single-split score of the soft votes: best alpha on the split, averaged over the MLP seeds.
best_alpha = alpha.groupby(["other", "mlp_seed"])["accuracy"].max().groupby("other").mean()
nested["single_split"] = nested["single_split"].fillna(nested["model"].map({"mlp+rf": best_alpha["rf"],
                                                                            "mlp+svm": best_alpha["svm"]}))
fig = plt.figure(figsize=(FULL_W, 2.1))
gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1], wspace=0.55)
ax = fig.add_subplot(gs[0])
models = nested["model"].tolist()
x = np.arange(len(models))
ax.bar(x - 0.2, nested["single_split"], 0.38, color=GREY, label="single split, best of grid")
ax.bar(x + 0.2, nested["accuracy"], 0.38, yerr=nested["accuracy_std"], color=BLUE, capsize=1.5,
       error_kw={"lw": 0.8}, label="nested CV (5 repeats)")
ax.set_xticks(x, [MODEL_NAMES[m].replace("MLP + linear SVM", "MLP + SVM").replace("SVM (linear or RBF)", "SVM")
                  .replace("Random forest", "RF").replace("MLP ensemble", "MLP") for m in models],
              rotation=35, ha="right")
ax.set(ylim=(0.85, 1.0), ylabel="Accuracy", title="(a) Single split vs nested CV")
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="upper left", fontsize=6, ncol=1, borderaxespad=0.2)
for i, (other, name) in enumerate((("rf", "RF"), ("svm", "linear SVM"))):
    ax = fig.add_subplot(gs[i + 1])
    for s, d in alpha[alpha["other"] == other].groupby("mlp_seed"):
        ax.plot(d["alpha"], d["accuracy"], marker="o", ms=2.5, lw=1, color=BLUE, alpha=0.35 + 0.13 * s)
        best = d.loc[d["accuracy"].idxmax()]
        ax.plot(best["alpha"], best["accuracy"], marker="*", ms=7, color=ORANGE, zorder=5, lw=0)
    ax.set(xlabel=r"Weight of the MLP $\alpha$", ylim=(0.9, 0.95), title=f"({'bc'[i]}) MLP + {name}")
    if i == 0:
        ax.set_ylabel("Accuracy (single split)")
fig.subplots_adjust(bottom=0.3)
save(fig, "protocol")

# ---------------------------------------------------------------------- benchmark
bench = pd.read_csv(RESULTS / "benchmark/tables/summary_mean.csv")
order = list(dict.fromkeys(bench["model"]))
fig, ax = plt.subplots(figsize=(FULL_W, 2.1))
x = np.arange(len(order))
for i, (fs, color) in enumerate((("volumetric", GREY), ("volumetric+thickness", BLUE), ("compact", AQUA))):
    d = bench[bench["features"] == fs].set_index("model").loc[order]
    ax.bar(x + (i - 1) * 0.27, d["accuracy"], 0.26, yerr=d["accuracy_std"], color=color, capsize=1.2,
           error_kw={"lw": 0.7}, label=FS_NAMES[fs])
ax.set_xticks(x, [MODEL_NAMES[m] for m in order], rotation=25, ha="right")
ax.set(ylim=(0.85, 1.0), ylabel="Nested-CV accuracy")
ax.grid(axis="x", visible=False)
ax.legend(ncol=3, frameon=False, loc="upper left")
fig.tight_layout()
save(fig, "benchmark")

# ---------------------------------------------------------------------- final model
cm = pd.read_csv(RESULTS / "final_model/tables/confusion.csv").set_index("true")
imp = pd.read_csv(RESULTS / "final_model/tables/permutation_importance.csv").head(12).iloc[::-1]
fig = plt.figure(figsize=(FULL_W, 2.25))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=1.0)
ax = fig.add_subplot(gs[0])
ax.imshow(cm.to_numpy(), cmap=SEQ, vmin=0, vmax=1)
for i in range(5):
    for j in range(5):
        v = cm.iat[i, j]
        if v > 0:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.55 else "#0b0b0b")
ax.set_xticks(range(5), cm.columns)
ax.set_yticks(range(5), cm.index)
ax.set(xlabel="Predicted", ylabel="True", title="(a) Confusion (nested CV)")
ax.grid(False)
ax = fig.add_subplot(gs[1])
colors = [ORANGE if "thick" in f else BLUE for f in imp["feature"]]
ax.barh(imp["feature"].str.replace("_", r"\_").map(lambda s: f"$\\mathtt{{{s}}}$"), imp["importance"],
        xerr=imp["std"], color=colors, error_kw={"lw": 0.7, "ecolor": GREY}, height=0.65)
ax.set(xlabel="Accuracy drop when permuted", title="(b) Permutation importance")
ax.tick_params(axis="y", labelsize=6.3)
ax.grid(axis="y", visible=False)
save(fig, "final_model")

# ====================================================================== tables of the main text
lvr = pd.read_csv(RESULTS / "lv_reconstruction/tables/summary.csv")
names = {"fill": "Filling (holes of the myocardium)", "fill_hull[min_hull_fraction=0.0]": "+ convex hull, no threshold",
         "fill_hull[min_hull_fraction=0.25]": "+ convex hull, area $\\geq$ 25\\% (ours)",
         "fill_hull[min_hull_fraction=0.5]": "+ convex hull, area $\\geq$ 50\\%"}
rows = "\n".join(
    f"{names[r.method]} & {r.dice_mean:.4f} & {r.dice_min:.3f} & {r.exact_volumes} & {r.vol_err_absmax:.1f} & "
    f"{r.slices_missed} & {r.slices_spurious} & {r.ef_abs_err_mean:.2f} & {r.ef_abs_err_max:.1f}\\\\" for r in lvr.itertuples())
write_tex("lv_reconstruction", f"""% Generated by regenerate_figures.py from final/results/lv_reconstruction/tables/summary.csv
\\begin{{table}}[!htb]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{adjustbox}}{{max width=\\linewidth}}
\\begin{{tabular}}{{lrrrrrrrr}}
\\toprule
& \\multicolumn{{2}}{{c}}{{Dice}} & Exact & Max $|\\Delta V|$ & \\multicolumn{{2}}{{c}}{{Slices}} & \\multicolumn{{2}}{{c}}{{$|\\Delta$EF$|$ (points)}}\\\\
\\cmidrule(lr){{2-3}}\\cmidrule(lr){{6-7}}\\cmidrule(lr){{8-9}}
Method & mean & min & volumes & (mL) & missed & spurious & mean & max\\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{adjustbox}}
\\caption{{Reconstruction of the LV cavity on the 200 training volumes (100 patients, ED and ES), compared with the
true LV. \\emph{{Exact}}: volumes reconstructed voxel for voxel. EF errors are per patient.}}
\\label{{tab:lv}}
\\end{{table}}
""")

rows = []
for r in nested.itertuples():
    grid = RESULTS / f"protocol_study/tables/grid_single_split_{r.model}.csv"
    if grid.exists():
        g = pd.read_csv(grid)["accuracy"]
        n_cfg, ties, med = str(len(g)), str(int((g == g.max()).sum())), f"{g.median():.3f}"
    elif r.model.startswith("mlp+"):
        n_cfg, ties, med = "11", "--", "--"  # values of alpha
    else:
        n_cfg = ties = med = "--"
    rows.append(f"{MODEL_NAMES[r.model]} & {n_cfg} & {ties} & {med} & {r.single_split:.3f} & "
                f"{pm(r.accuracy, r.accuracy_std)}\\\\")
write_tex("protocol", """% Generated by regenerate_figures.py from final/results/protocol_study
\\begin{table}[!htb]
\\centering
\\small
\\setlength{\\tabcolsep}{4pt}
\\begin{adjustbox}{max width=\\linewidth}
\\begin{tabular}{lrrrrr}
\\toprule
& \\multicolumn{4}{c}{Single split (scaler fitted on all patients)} & \\\\
\\cmidrule(lr){2-5}
Model & Configs & Tied best & Median & Best & Nested CV\\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{adjustbox}
\\caption{Model selection on a single split vs nested cross-validation (21 volumetric features, true LV).
\\emph{Single split}: every configuration of the grid scored on one stratified 5-fold split; \\emph{Best}: the best
of them, \\emph{Tied best}: number of configurations reaching it, \\emph{Median}: median over the grid. For the
MLP ensemble (no grid), mean over 5 seeds; for the soft votes, best $\\alpha$ on the split, mean over 5 seeds.
\\emph{Nested CV}: same grids, selected inside the training folds, mean $\\pm$ std over 5 repeats.}
\\label{tab:protocol-study}
\\end{table}
""")

rows = []
best = {fs: bench.loc[bench["features"] == fs, "accuracy"].max() for fs in FS_NAMES}
for m in order:
    cells = []
    for fs in FS_NAMES:
        r = bench[(bench["model"] == m) & (bench["features"] == fs)].iloc[0]
        cells.append(pm(r["accuracy"], r["accuracy_std"], bold=r["accuracy"] == best[fs]))
    rows.append(f"{MODEL_NAMES[m]} & " + " & ".join(cells) + "\\\\")
    if m in ("knn", "xgboost"):
        rows.append("\\midrule")
write_tex("benchmark", """% Generated by regenerate_figures.py from final/results/benchmark/tables/summary_mean.csv
\\begin{table}[!htb]
\\centering
\\small
\\setlength{\\tabcolsep}{5pt}
\\begin{tabular}{lccc}
\\toprule
Model & """ + " & ".join(FS_NAMES.values()) + """\\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\caption{Nested-CV accuracy (mean $\\pm$ std over 10 repeats of stratified 5-fold CV) with the LV reconstructed in
training and evaluated patients, as for the test set. Hyper-parameters and the ensemble weight $\\alpha$ are
selected by an inner 5-fold CV. Best value of each column in bold. Classes are balanced, so the balanced accuracy
equals the accuracy.}
\\label{tab:benchmark}
\\end{table}
""")

proto = pd.read_csv(RESULTS / "lv_protocol/tables/summary_mean.csv")
protos = list(dict.fromkeys(proto["protocol"]))
rows = []
for fs in ("volumetric", "volumetric+thickness"):
    rows.append(f"\\multicolumn{{{len(protos) + 1}}}{{l}}{{\\emph{{{FS_NAMES[fs]}}}}}\\\\")
    for m in dict.fromkeys(proto["model"]):
        d = proto[(proto["features"] == fs) & (proto["model"] == m)].set_index("protocol").loc[protos]
        rows.append(f"\\quad {MODEL_NAMES[m]} & " + " & ".join(pm(a, s) for a, s in zip(d["accuracy"], d["accuracy_std"])) + "\\\\")
write_tex("lv_protocol", """% Generated by regenerate_figures.py from final/results/lv_protocol/tables/summary_mean.csv
\\begin{table}[!htb]
\\centering
\\small
\\setlength{\\tabcolsep}{4pt}
\\begin{adjustbox}{max width=\\linewidth}
\\begin{tabular}{l""" + "c" * len(protos) + """}
\\toprule
LV of training $\\to$ evaluated patients & """ + " & ".join(PROTO_NAMES[p] for p in protos) + """\\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{adjustbox}
\\caption{Effect of the LV used to compute the features (nested CV, 10 repeats). \\emph{true}: ground-truth LV
(training set only); \\emph{fill}: filling; \\emph{fill+hull}: filling with the convex-hull fallback.
``true $\\to$ fill'' is the situation of a model trained with the true LV and applied to the test set.}
\\label{tab:protocol}
\\end{table}
""")

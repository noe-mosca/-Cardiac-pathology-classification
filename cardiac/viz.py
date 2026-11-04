"""Figures (matplotlib only, saved through Run.save_figure)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def confusion(cm: pd.DataFrame, title: str = ""):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cm.columns)), cm.columns)
    ax.set_yticks(range(len(cm.index)), cm.index)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return fig


def heatmap(pivot: pd.DataFrame, title: str = "", fmt: str = "{:.3f}"):
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * pivot.shape[1], 0.9 + 0.38 * pivot.shape[0]))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]), pivot.index, fontsize=8)
    lo, hi = np.nanmin(pivot.values), np.nanmax(pivot.values)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="black" if v > lo + 0.6 * (hi - lo) else "white")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def bars(df: pd.DataFrame, x: str, y: str, err: str | None = None, hue: str | None = None, title: str = "",
         ylim: tuple | None = None, ref: float | None = None):
    """Grouped bar chart of ``y`` (+- ``err``) by ``x`` and ``hue``."""
    hues = list(dict.fromkeys(df[hue])) if hue else [None]
    xs = list(dict.fromkeys(df[x]))
    w = 0.8 / len(hues)
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(xs) * max(1, len(hues) / 2)), 3.8))
    for i, h in enumerate(hues):
        sub = df[df[hue] == h] if hue else df
        sub = sub.set_index(x).reindex(xs)
        ax.bar(np.arange(len(xs)) + (i - (len(hues) - 1) / 2) * w, sub[y], w, label=h,
               yerr=sub[err] if err else None, capsize=2)
    ax.set_xticks(range(len(xs)), xs, rotation=30, ha="right")
    ax.set_ylabel(y)
    if ylim:
        ax.set_ylim(*ylim)
    if ref is not None:
        ax.axhline(ref, color="grey", lw=0.8, ls="--")
    if hue:
        ax.legend(fontsize=8, ncol=min(3, len(hues)))
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def bland_altman(ref: np.ndarray, est: np.ndarray, groups: pd.Series | None = None, title: str = "", unit: str = "mL"):
    mean, diff = (ref + est) / 2, est - ref
    fig, ax = plt.subplots(figsize=(5, 3.6))
    if groups is None:
        ax.scatter(mean, diff, s=12)
    else:
        for g in dict.fromkeys(groups):
            m = (groups == g).to_numpy()
            ax.scatter(mean[m], diff[m], s=12, label=str(g))
        ax.legend(fontsize=8)
    md, sd = diff.mean(), diff.std()
    for v, ls in ((md, "-"), (md - 1.96 * sd, "--"), (md + 1.96 * sd, "--")):
        ax.axhline(v, color="grey", lw=0.8, ls=ls)
    ax.set_xlabel(f"mean of reference and reconstruction ({unit})")
    ax.set_ylabel(f"reconstruction - reference ({unit})")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def line(df: pd.DataFrame, x: str, y: str, hue: str | None = None, err: str | None = None, title: str = "",
         ylabel: str | None = None):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for h, sub in (df.groupby(hue, sort=False) if hue else [(None, df)]):
        ax.errorbar(sub[x], sub[y], yerr=sub[err] if err else None, marker="o", ms=3, capsize=2, label=h)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    if hue:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def boxgrid(df: pd.DataFrame, cols: list[str], by: str, order: list[str], extra: pd.DataFrame | None = None,
            extra_name: str = "", ncols: int = 4):
    """One box plot per feature, grouped by class (+ an optional extra group, e.g. the test set)."""
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows))
    labels = order + ([extra_name] if extra is not None else [])
    for ax, c in zip(axes.ravel(), cols):
        data = [df.loc[df[by] == g, c].to_numpy() for g in order]
        if extra is not None:
            data.append(extra[c].to_numpy())
        ax.boxplot(data, widths=0.6, flierprops={"markersize": 2})
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=7)
        ax.set_title(c, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def slices(images: list[np.ndarray], segs: list[np.ndarray], titles: list[str]):
    """Mid-ventricular slices with the label overlay (RV, myocardium, LV)."""
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap([(0, 0, 0, 0), (0.1, 0.7, 0.2, 0.45), (0.15, 0.35, 0.9, 0.45), (0.9, 0.2, 0.2, 0.45)])
    fig, axes = plt.subplots(1, len(images), figsize=(2.6 * len(images), 2.8))
    for ax, img, seg, t in zip(np.atleast_1d(axes), images, segs, titles):
        ax.imshow(img.T, cmap="gray")
        ax.imshow(seg.T, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    return fig

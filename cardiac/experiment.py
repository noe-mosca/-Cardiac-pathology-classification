"""Helpers shared by the scripts in ``scripts/``."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import config
from .features import FEATURE_SETS, load_features


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = config.CONFIGS_DIR / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_file"] = str(path)
    return cfg


def feature_columns(feature_set: str | list[str]) -> list[str]:
    return list(feature_set) if isinstance(feature_set, list) else FEATURE_SETS[feature_set]


def load_xy(feature_set: str | list[str], fit_lv: str, eval_lv: str | None = None):
    """Train matrices for training with ``fit_lv`` features and evaluating with
    ``eval_lv`` features (same patients, same order). Returns X_fit, X_eval, y, ids."""
    cols = feature_columns(feature_set)
    fit = load_features("Train", fit_lv)
    ev = load_features("Train", eval_lv or fit_lv).loc[fit.index]
    return fit[cols].to_numpy(float), ev[cols].to_numpy(float), fit["category"].to_numpy(int), fit.index.to_numpy()


def load_test(feature_set: str | list[str], lv: str) -> tuple[np.ndarray, np.ndarray]:
    test = load_features("Test", lv)
    return test[feature_columns(feature_set)].to_numpy(float), test.index.to_numpy()


def mean_std_text(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """'0.934 ± 0.012' columns from <metric> / <metric>_std columns."""
    out = df.drop(columns=[c for c in df if c.endswith("_std")]).copy()
    for m in metrics:
        out[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(df[m], df[f"{m}_std"])]
    return out

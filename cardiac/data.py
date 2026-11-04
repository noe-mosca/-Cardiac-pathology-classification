"""Dataset access: metadata, segmentations, patient index.

Layout expected under ``config.DATA_ROOT``::

    Train/<id>/<id>_{ED,ES}.nii, <id>_{ED,ES}_seg.nii     labels 0 bg, 1 RV, 2 myocardium, 3 LV
    Test/<id>/...                                           same, but the LV cavity is not given

If a test segmentation contains an LV label (e.g. written by an earlier
reconstruction), :func:`load_seg` maps it back to background, so the pipeline
always starts from the data as distributed and reconstructs the LV itself.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from . import config


def load_metadata(split: str) -> pd.DataFrame:
    """Height (cm), weight (kg) and, for Train, the category; indexed by the 3-digit patient id."""
    name = {"Train": "metaDataTrain.csv", "Test": "metaDataTest.csv"}[split]
    df = pd.read_csv(config.META_DIR / name, dtype={"Id": str})
    df["Id"] = df["Id"].str.zfill(3)
    df = df.set_index("Id").rename(columns=str.lower)
    if "category" in df:
        df["category"] = df["category"].astype(int)
    return df


def patient_ids(split: str) -> list[str]:
    root = config.DATA_ROOT / split
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit())


def seg_path(split: str, pid: str, phase: str) -> Path:
    return config.DATA_ROOT / split / pid / f"{pid}_{phase}_seg.nii"


def image_path(split: str, pid: str, phase: str) -> Path:
    return config.DATA_ROOT / split / pid / f"{pid}_{phase}.nii"


def load_seg(split: str, pid: str, phase: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Label volume (uint8, x * y * slices) and voxel spacing in mm."""
    img = nib.load(seg_path(split, pid, phase))
    seg = np.rint(np.asanyarray(img.dataobj)).astype(np.uint8)
    if split == "Test":
        seg[seg == config.LV] = config.BACKGROUND
    return seg, tuple(float(s) for s in img.header.get_zooms()[:3])


def load_image(split: str, pid: str, phase: str) -> np.ndarray:
    return np.asanyarray(nib.load(image_path(split, pid, phase)).dataobj).astype(np.float32)


def build_index() -> pd.DataFrame:
    """One row per patient with metadata and per-phase volume information."""
    rows = []
    for split in ("Train", "Test"):
        meta = load_metadata(split)
        for pid in patient_ids(split):
            row = {"split": split, "id": pid, **meta.loc[pid].to_dict()}
            for phase in config.PHASES:
                img = nib.load(seg_path(split, pid, phase))
                raw = np.asanyarray(img.dataobj)
                sx, sy, sz = (float(s) for s in img.header.get_zooms()[:3])
                row.update({
                    f"{phase}_shape": "x".join(map(str, raw.shape)), f"{phase}_slices": raw.shape[2],
                    f"{phase}_spacing_xy": sx, f"{phase}_spacing_z": sz, f"{phase}_dtype": str(raw.dtype),
                    f"{phase}_has_lv_on_disk": bool((raw == config.LV).any()),
                    f"{phase}_integer_labels": bool(np.all(raw == np.rint(raw))),
                })
            rows.append(row)
    df = pd.DataFrame(rows)
    if "category" in df:
        df["category"] = df["category"].astype("Int64")
    return df


def load_index() -> pd.DataFrame:
    if not config.INDEX_PATH.exists():
        raise FileNotFoundError(f"{config.INDEX_PATH} missing: run scripts/01_build_index.py first")
    return pd.read_csv(config.INDEX_PATH, dtype={"id": str})

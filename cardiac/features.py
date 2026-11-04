"""Patient-level features computed from the ED / ES segmentations + height / weight.

``volumetric`` features are 21 classical biomarkers: LV / RV volumes at ED and
ES, ejection fractions, myocardial mass (1.05 g/mL x myocardium volume),
volumes and mass indexed by the body surface area (Mosteller), RV / LV ratios,
height and weight (``x / 0 -> 0``). Two of them are almost identical
(``myocardium_to_lv_ed`` uses the ED mass, ``mass_to_edv_ratio`` the ED/ES average).

``thickness`` features describe the myocardial wall, which separates
MINF (locally thin, akinetic wall) and HCM (thick wall) from DCM / NOR better
than global volumes: per slice, the wall thickness is sampled at every
myocardium pixel touching the cavity as its distance (mm) to the outside of the
myocardium + cavity region; it is then summarised over the samples and over slices.

The LV used for the features is either the ground truth (``gt``, Train only)
or a reconstruction (``fill`` / ``fill_hull``, see lv.py): the test set only
has reconstructions, so train/test consistency requires training on them too.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import config
from .data import load_metadata, load_seg, patient_ids
from .lv import with_reconstructed_lv

VOLUMETRIC = [
    "ed_lv_vol", "es_lv_vol", "lv_ef", "ed_rv_vol", "es_rv_vol", "rv_ef", "myocardium_mass", "lv_mass_index",
    "stroke_volume", "es_lv_index", "ed_rv_index", "ratio_rv_lv_es", "ratio_rv_lv_ed", "weight", "height", "bsa",
    "ed_lv_index", "es_rv_index", "myocardium_to_lv_ed", "mass_to_edv_ratio", "mass_ratio_systole_diastole",
]
THICKNESS = [f"{ph}_thick_{s}" for ph in ("ed", "es")
             for s in ("mean", "max", "slicemean_max", "slicestd_mean", "slicemean_std")] + ["wall_thickening"]
# BSA-indexed volumes, both ejection fractions, mass and wall descriptors; no raw size / weight.
COMPACT = ["ed_lv_index", "es_lv_index", "lv_ef", "ed_rv_index", "es_rv_index", "rv_ef", "lv_mass_index",
           "ratio_rv_lv_ed", "mass_to_edv_ratio", "ed_thick_slicemean_max", "ed_thick_slicestd_mean",
           "wall_thickening"]

FEATURE_SETS = {
    "volumetric": VOLUMETRIC,
    "volumetric+thickness": VOLUMETRIC + THICKNESS,
    "compact": COMPACT,
}
LV_SOURCES = ("gt", "fill", "fill_hull")


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def volume_ml(seg: np.ndarray, label: int, spacing) -> float:
    return float((seg == label).sum() * np.prod(spacing) / 1000)


def wall_thickness(seg: np.ndarray, spacing) -> list[np.ndarray]:
    """Per slice, thickness samples (mm) along the endocardial border."""
    out = []
    for z in range(seg.shape[2]):
        myo, lv = seg[:, :, z] == config.MYO, seg[:, :, z] == config.LV
        if not (myo.any() and lv.any()):
            continue
        solid = ndi.binary_fill_holes(myo | lv)
        depth = ndi.distance_transform_edt(solid, sampling=spacing[:2])
        endo = myo & ndi.binary_dilation(lv)
        if endo.any():
            out.append(depth[endo])
    return out


def thickness_features(seg: np.ndarray, spacing, prefix: str) -> dict:
    slices = wall_thickness(seg, spacing)
    if not slices:
        return {f"{prefix}_thick_{s}": 0.0 for s in ("mean", "max", "slicemean_max", "slicestd_mean", "slicemean_std")}
    allv = np.concatenate(slices)
    means = np.array([s.mean() for s in slices])
    return {
        f"{prefix}_thick_mean": float(allv.mean()),
        f"{prefix}_thick_max": float(allv.max()),
        f"{prefix}_thick_slicemean_max": float(means.max()),
        f"{prefix}_thick_slicestd_mean": float(np.mean([s.std() for s in slices])),
        f"{prefix}_thick_slicemean_std": float(means.std()),
    }


def patient_features(ed: np.ndarray, es: np.ndarray, spacing_ed, spacing_es, height: float, weight: float) -> dict:
    ed_lv, es_lv = volume_ml(ed, config.LV, spacing_ed), volume_ml(es, config.LV, spacing_es)
    ed_rv, es_rv = volume_ml(ed, config.RV, spacing_ed), volume_ml(es, config.RV, spacing_es)
    mass_ed = 1.05 * volume_ml(ed, config.MYO, spacing_ed)
    mass_es = 1.05 * volume_ml(es, config.MYO, spacing_es)
    mass = (mass_ed + mass_es) / 2
    bsa = float(np.sqrt(weight * height / 3600))
    f = {
        "ed_lv_vol": ed_lv, "es_lv_vol": es_lv, "lv_ef": 100 * _div(ed_lv - es_lv, ed_lv),
        "ed_rv_vol": ed_rv, "es_rv_vol": es_rv, "rv_ef": 100 * _div(ed_rv - es_rv, ed_rv),
        "myocardium_mass": mass, "lv_mass_index": _div(mass, bsa), "stroke_volume": ed_lv - es_lv,
        "es_lv_index": _div(es_lv, bsa), "ed_rv_index": _div(ed_rv, bsa),
        "ratio_rv_lv_es": _div(es_rv, es_lv), "ratio_rv_lv_ed": _div(ed_rv, ed_lv),
        "weight": weight, "height": height, "bsa": bsa,
        "ed_lv_index": _div(ed_lv, bsa), "es_rv_index": _div(es_rv, bsa),
        "myocardium_to_lv_ed": _div(mass_ed, ed_lv), "mass_to_edv_ratio": _div(mass, ed_lv),
        "mass_ratio_systole_diastole": _div(mass_es, mass_ed),
    }
    f.update(thickness_features(ed, spacing_ed, "ed"))
    f.update(thickness_features(es, spacing_es, "es"))
    f["wall_thickening"] = _div(f["es_thick_mean"], f["ed_thick_mean"])
    return f


def extract(split: str, lv_source: str, **lv_kw) -> pd.DataFrame:
    """Feature table of a split, one row per patient (``id``, ``category`` for Train, features)."""
    if lv_source not in LV_SOURCES:
        raise ValueError(f"unknown LV source {lv_source!r}")
    if split == "Test" and lv_source == "gt":
        raise ValueError("the test set has no ground-truth LV")
    meta = load_metadata(split)
    rows = []
    for pid in patient_ids(split):
        (ed, sp_ed), (es, sp_es) = (load_seg(split, pid, ph) for ph in config.PHASES)
        if lv_source != "gt":
            ed, es = with_reconstructed_lv(ed, lv_source, **lv_kw), with_reconstructed_lv(es, lv_source, **lv_kw)
        m = meta.loc[pid]
        row = {"id": pid}
        if "category" in meta:
            row["category"] = int(m["category"])
        row.update(patient_features(ed, es, sp_ed, sp_es, float(m["height"]), float(m["weight"])))
        rows.append(row)
    return pd.DataFrame(rows)


def features_path(split: str, lv_source: str):
    return config.FEATURES_DIR / f"{split.lower()}_{lv_source}.csv"


def load_features(split: str, lv_source: str) -> pd.DataFrame:
    path = features_path(split, lv_source)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/03_extract_features.py first")
    return pd.read_csv(path, dtype={"id": str}).set_index("id")

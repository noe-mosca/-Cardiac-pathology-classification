"""Reconstruction of the LV cavity, which is not given in the test segmentations.

``fill``       per slice, the holes of the myocardium mask.
``fill_hull``  same, but on slices where the myocardium ring is open (no hole,
               typically the most basal slice) the cavity is taken as the convex
               hull of the myocardium minus the myocardium. ``min_hull_fraction``
               discards hulls smaller than that fraction of the myocardium area
               (thin slivers on apical caps); 0.25 by default, chosen on the
               training set (0 gives spurious slices on apical caps, 0.25 / 0.5 none).

Both are evaluated on the training set, where the true LV is known
(scripts/02_lv_reconstruction.py).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image

from .config import BACKGROUND, LV, MYO

METHODS = ("fill", "fill_hull")


def reconstruct_lv(seg: np.ndarray, method: str = "fill_hull", min_hull_fraction: float = 0.25) -> np.ndarray:
    """Boolean LV mask from a segmentation whose LV label is ignored."""
    if method not in METHODS:
        raise ValueError(f"unknown LV reconstruction {method!r}")
    out = np.zeros(seg.shape, dtype=bool)
    for z in range(seg.shape[2]):
        sl = seg[:, :, z]
        myo = sl == MYO
        if not myo.any():
            continue
        free = (sl == BACKGROUND) | (sl == LV)
        cavity = ndi.binary_fill_holes(myo) & free
        if method == "fill_hull" and not cavity.any():
            hull = convex_hull_image(myo) & free
            if hull.sum() > min_hull_fraction * myo.sum():
                cavity = hull
        out[:, :, z] = cavity
    return out


def with_reconstructed_lv(seg: np.ndarray, method: str, **kw) -> np.ndarray:
    """Copy of ``seg`` where label 3 is replaced by the reconstructed cavity."""
    out = seg.copy()
    out[out == LV] = BACKGROUND
    out[reconstruct_lv(out, method, **kw)] = LV
    return out


def compare_masks(gt: np.ndarray, rec: np.ndarray, spacing) -> dict:
    """Overlap and volume agreement between the true and reconstructed LV."""
    voxel_ml = float(np.prod(spacing)) / 1000
    inter, s_gt, s_rec = (gt & rec).sum(), gt.sum(), rec.sum()
    has_gt, has_rec = gt.any(axis=(0, 1)), rec.any(axis=(0, 1))
    return {
        "dice": 2 * inter / (s_gt + s_rec) if s_gt + s_rec else 1.0,
        "vol_gt_ml": s_gt * voxel_ml,
        "vol_rec_ml": s_rec * voxel_ml,
        "vol_err_ml": (s_rec - s_gt) * voxel_ml,
        "slices_missed": int((has_gt & ~has_rec).sum()),
        "slices_spurious": int((~has_gt & has_rec).sum()),
    }

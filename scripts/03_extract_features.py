"""Compute the patient features for every LV source and check them.

    python scripts/03_extract_features.py

Writes cache/features/{train,test}_{gt,fill,fill_hull}.csv (test: no gt), then
* measures how much each feature changes when the training LV is
  reconstructed instead of the ground truth;
* describes the features per class (table + box plots).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from cardiac import config, viz
from cardiac.features import COMPACT, LV_SOURCES, THICKNESS, VOLUMETRIC, extract, features_path
from cardiac.tracking import Run


def main():
    with Run("features", {"lv_sources": list(LV_SOURCES)}) as run:
        config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        tables = {}
        for split in ("Train", "Test"):
            for src in LV_SOURCES:
                if split == "Test" and src == "gt":
                    continue
                df = extract(split, src)
                df.to_csv(features_path(split, src), index=False)
                tables[(split, src)] = df.set_index("id")
                run.log.info("%s / %s: %d patients x %d features", split, src, len(df), df.shape[1] - 1)

        # effect of reconstructing the training LV, per feature
        gt = tables[("Train", "gt")]
        cols = VOLUMETRIC + THICKNESS
        rows = []
        for src in ("fill", "fill_hull"):
            rel = (tables[("Train", src)][cols] - gt[cols]).abs() / gt[cols].abs().clip(lower=1e-9)
            for c in cols:
                rows.append({"lv_source": src, "feature": c, "patients_changed": int((rel[c] > 1e-9).sum()),
                             "max_rel_change": float(rel[c].max()), "mean_rel_change": float(rel[c].mean())})
        shift = pd.DataFrame(rows)
        run.save_table(shift, "lv_source_shift", latex=False)
        shift_summary = (shift.groupby("lv_source")
                         .agg(patients_changed=("patients_changed", "max"), max_rel_change=("max_rel_change", "max"))
                         .reset_index())
        run.save_table(shift_summary, "lv_source_shift_summary")
        run.log.info("train features vs ground-truth LV:\n%s", shift_summary.to_string(index=False))

        # per-class description (training LV = ground truth)
        desc = gt.assign(cls=gt["category"].map(config.CLASS_NAMES))
        order = list(config.CLASS_NAMES.values())
        means = desc.groupby("cls")[COMPACT].mean().loc[order].T.reset_index().rename(columns={"index": "feature"})
        run.save_table(means, "class_means_compact", float_format="%.2f")
        test = tables[("Test", "fill_hull")]
        run.save_figure(viz.boxgrid(desc, COMPACT, "cls", order, extra=test, extra_name="Test"), "boxplots_compact")
        run.save_figure(viz.boxgrid(desc, THICKNESS, "cls", order, extra=test, extra_name="Test"), "boxplots_thickness")
        run.finish({"lv_source_shift": shift_summary.to_dict("records")})


if __name__ == "__main__":
    main()

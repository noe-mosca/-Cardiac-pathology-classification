"""Index the dataset, check the files and describe the patients.

    python scripts/01_build_index.py

* cache/index.csv: one row per patient (metadata, shapes, spacings, label checks);
* dataset statistics per class (tables/class_stats);
* label checks (the test segmentations should not contain the LV; one found on disk is ignored on load);
* example slices of one patient per class.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from cardiac import config, viz
from cardiac.data import build_index, load_image, load_seg
from cardiac.tracking import Run


def main():
    with Run("dataset", {"data_root": str(config.DATA_ROOT)}) as run:
        index = build_index()
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        index.to_csv(config.INDEX_PATH, index=False)
        run.save_table(index, "index", latex=False)
        train, test = index[index.split == "Train"], index[index.split == "Test"]
        run.log.info("%d train / %d test patients", len(train), len(test))

        stats = (train.assign(cls=train["category"].map(config.CLASS_NAMES))
                 .groupby("cls", sort=False)
                 .agg(n=("id", "size"), height=("height", "mean"), weight=("weight", "mean"),
                      slices=("ED_slices", "mean"), spacing_xy=("ED_spacing_xy", "mean"),
                      spacing_z=("ED_spacing_z", "mean")).reset_index())
        overall = pd.DataFrame([{"cls": f"Test (n={len(test)})", "n": len(test), "height": test["height"].mean(),
                                 "weight": test["weight"].mean(), "slices": test["ED_slices"].mean(),
                                 "spacing_xy": test["ED_spacing_xy"].mean(), "spacing_z": test["ED_spacing_z"].mean()}])
        run.save_table(pd.concat([stats, overall]), "class_stats", float_format="%.1f")

        summary = {
            "train": len(train), "test": len(test),
            "train_class_counts": {config.CLASS_NAMES[int(k)]: int(v) for k, v in train["category"].value_counts().sort_index().items()},
            "test_files_with_lv_on_disk": int(test[["ED_has_lv_on_disk", "ES_has_lv_on_disk"]].to_numpy().sum()),
            "test_dtypes": sorted(set(test["ED_dtype"]) | set(test["ES_dtype"])),
            "train_files_without_lv": int((~train[["ED_has_lv_on_disk", "ES_has_lv_on_disk"]]).to_numpy().sum()),
            "non_integer_label_files": int((~index[["ED_integer_labels", "ES_integer_labels"]]).to_numpy().sum()),
            "spacing_z_values": sorted(index["ED_spacing_z"].unique().tolist()),
            "slices_range": [int(index["ED_slices"].min()), int(index["ED_slices"].max())],
        }
        run.log.info("test segmentation files containing an LV label (ignored on load): %d / %d",
                     summary["test_files_with_lv_on_disk"], 2 * len(test))

        examples = train.groupby("category").first()["id"]
        imgs, segs, titles = [], [], []
        for c, pid in examples.items():
            seg, _ = load_seg("Train", pid, "ED")
            z = int(np.argmax((seg == config.LV).sum((0, 1))))
            imgs.append(load_image("Train", pid, "ED")[:, :, z])
            segs.append(seg[:, :, z])
            titles.append(f"{config.CLASS_NAMES[c]} (patient {pid}, ED)")
        run.save_figure(viz.slices(imgs, segs, titles), "examples")
        run.finish(summary)


if __name__ == "__main__":
    main()

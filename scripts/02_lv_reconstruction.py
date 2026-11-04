"""Evaluate the reconstruction of the LV cavity on the training set (true LV known).

    python scripts/02_lv_reconstruction.py [--config lv_reconstruction.yaml]

For each training volume the LV label is removed, reconstructed with every
method of the config and compared with the truth (Dice, volume error, missed /
spurious slices); the effect on the LV ejection fraction is computed per patient.
Also counts, on the test set, how many slices each method adds to ``fill``
(i.e. how often ``fill_hull`` falls back to the convex hull).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from cardiac import config, viz
from cardiac.data import load_index, load_seg, patient_ids
from cardiac.experiment import load_config
from cardiac.lv import compare_masks, reconstruct_lv
from cardiac.tracking import Run


def method_name(m: dict) -> str:
    extra = ",".join(f"{k}={v}" for k, v in m.items() if k != "method")
    return f"{m['method']}[{extra}]" if extra else m["method"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="lv_reconstruction.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    methods = cfg["methods"]

    with Run("lv_reconstruction", cfg) as run:
        index = load_index().set_index("id")
        rows = []
        for pid in patient_ids("Train"):
            for phase in config.PHASES:
                seg, spacing = load_seg("Train", pid, phase)
                gt = seg == config.LV
                for m in methods:
                    kw = {k: v for k, v in m.items() if k != "method"}
                    rec = reconstruct_lv(seg, m["method"], **kw)
                    rows.append({"id": pid, "phase": phase, "category": config.CLASS_NAMES[int(index.loc[pid, "category"])],
                                 "method": method_name(m), **compare_masks(gt, rec, spacing)})
        df = pd.DataFrame(rows)
        run.save_table(df, "per_volume", latex=False)

        # LV ejection fraction from the reconstructed volumes, per patient
        wide = df.pivot_table(index=["id", "category", "method"], columns="phase",
                              values=["vol_gt_ml", "vol_rec_ml"]).reset_index()
        ef = lambda ed, es: 100 * (ed - es) / ed
        wide["ef_gt"] = ef(wide[("vol_gt_ml", "ED")], wide[("vol_gt_ml", "ES")])
        wide["ef_rec"] = ef(wide[("vol_rec_ml", "ED")], wide[("vol_rec_ml", "ES")])
        wide["ef_err"] = wide["ef_rec"] - wide["ef_gt"]
        ef_df = pd.DataFrame({"id": wide["id"], "category": wide["category"], "method": wide["method"],
                              "ef_gt": wide["ef_gt"], "ef_rec": wide["ef_rec"], "ef_err": wide["ef_err"]})
        run.save_table(ef_df, "ef_per_patient", latex=False)

        summary = (df.groupby("method", sort=False)
                   .agg(dice_mean=("dice", "mean"), dice_min=("dice", "min"),
                        exact_volumes=("vol_err_ml", lambda v: int((v.abs() < 1e-9).sum())),
                        vol_err_mean=("vol_err_ml", "mean"), vol_err_absmax=("vol_err_ml", lambda v: v.abs().max()),
                        slices_missed=("slices_missed", "sum"), slices_spurious=("slices_spurious", "sum"))
                   .join(ef_df.groupby("method", sort=False)
                         .agg(ef_abs_err_mean=("ef_err", lambda v: v.abs().mean()),
                              ef_abs_err_max=("ef_err", lambda v: v.abs().max())))
                   .reset_index())
        run.save_table(summary, "summary")
        for r in summary.itertuples():
            run.log.info("%-28s dice=%.4f (min %.3f)  exact %d/200  |dV|max=%.1f mL  missed=%d spurious=%d  |dEF| mean=%.2f max=%.2f",
                         r.method, r.dice_mean, r.dice_min, r.exact_volumes, r.vol_err_absmax, r.slices_missed,
                         r.slices_spurious, r.ef_abs_err_mean, r.ef_abs_err_max)
        worst = df[df["vol_err_ml"].abs() > 1e-9].sort_values(["method", "dice"])
        run.save_table(worst, "imperfect_volumes", latex=False)

        for m in dict.fromkeys(df["method"]):
            sub = df[df["method"] == m]
            run.save_figure(viz.bland_altman(sub["vol_gt_ml"].to_numpy(), sub["vol_rec_ml"].to_numpy(),
                                             sub["phase"].reset_index(drop=True),
                                             title=f"LV volume, {m} vs ground truth (train, 200 volumes)"),
                            f"bland_altman_{m}")

        # ---------------------------------------------------------------- test set
        test_rows = []
        for pid in patient_ids("Test"):
            for phase in config.PHASES:
                seg, _ = load_seg("Test", pid, phase)  # LV not given
                fill = reconstruct_lv(seg, "fill")
                row = {"id": pid, "phase": phase}
                for m in methods:
                    kw = {k: v for k, v in m.items() if k != "method"}
                    rec = reconstruct_lv(seg, m["method"], **kw)
                    row[f"extra_slices_{method_name(m)}"] = int((rec.any((0, 1)) & ~fill.any((0, 1))).sum())
                test_rows.append(row)
        tdf = pd.DataFrame(test_rows)
        run.save_table(tdf, "test_check", latex=False)
        out = {"summary": summary.to_dict("records"), "test_volumes": len(tdf),
               **{f"test_{c}": int(tdf[c].sum()) for c in tdf if c.startswith("extra_slices_")}}
        run.log.info("test: slices added to fill by each method: %s",
                     {k: v for k, v in out.items() if k.startswith("test_extra")})
        run.finish(out)


if __name__ == "__main__":
    main()

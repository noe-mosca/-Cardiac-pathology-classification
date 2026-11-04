"""Experiment tracking: everything an experiment produces goes through a :class:`Run`.

    runs/<YYYYmmdd-HHMMSS>_<name>/
        config.json      experiment config + command line + library versions + device
        run.log          full log
        metrics.csv      one row per evaluated configuration (appended live)
        summary.json     final summary (written by finish())
        tables/*.csv|tex tables, LaTeX versions ready for \\input in the report
        figures/*.png|pdf
        arrays/*.npz     labels, assignments... to redo figures without refitting

On success the run is mirrored to ``results/<name>/`` (stable paths for the
report) and appended to ``results/runs.csv`` (history of all runs).
"""
import json
import logging
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "sklearn", "torch", "xgboost", "nibabel", "scipy", "skimage", "pandas"):
        try:
            out[mod] = __import__(mod).__version__
        except ImportError:
            pass
    return out


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray, Path)):
        return o.tolist() if isinstance(o, np.ndarray) else str(o)
    raise TypeError(type(o))


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_jsonable)


class Run:
    def __init__(self, name: str, cfg: dict, root: Path = config.RUNS_DIR):
        self.name = name
        self.id = f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
        self.dir = Path(root) / self.id
        for sub in ("tables", "figures", "arrays"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self._t0 = time.time()
        save_json({"name": name, "run_id": self.id, "command": " ".join(sys.argv),
                   "started": datetime.now().isoformat(timespec="seconds"),
                   "versions": _versions(), "config": cfg}, self.dir / "config.json")
        self.log = self._setup_logging()
        self.log.info("run %s -> %s", self.id, self.dir)

    def _setup_logging(self) -> logging.Logger:
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        for h in list(root.handlers):
            if getattr(h, "_cardiac", False):
                root.removeHandler(h)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        for h in (logging.FileHandler(self.dir / "run.log", encoding="utf-8"), logging.StreamHandler()):
            h.setFormatter(fmt)
            h._cardiac = True
            root.addHandler(h)
        return logging.getLogger(self.name)

    # ------------------------------------------------------------------ outputs
    def log_metrics(self, row: dict, table: str = "metrics") -> None:
        path = self.dir / f"{table}.csv"
        rows = pd.read_csv(path).to_dict("records") if path.exists() else []
        pd.DataFrame(rows + [row]).to_csv(path, index=False)

    def save_table(self, df: pd.DataFrame, name: str, latex: bool = True, float_format: str = "%.3f",
                   **latex_kw) -> None:
        df.to_csv(self.dir / "tables" / f"{name}.csv", index=False)
        if latex:
            (self.dir / "tables" / f"{name}.tex").write_text(
                df.to_latex(index=False, float_format=float_format.__mod__, escape=True, **latex_kw), encoding="utf-8")

    def save_figure(self, fig, name: str, dpi: int = 200) -> None:
        import matplotlib.pyplot as plt

        for ext in ("png", "pdf"):
            fig.savefig(self.dir / "figures" / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def save_json(self, obj, name: str) -> None:
        save_json(obj, self.dir / f"{name}.json")

    def save_arrays(self, name: str, **arrays) -> None:
        np.savez_compressed(self.dir / "arrays" / f"{name}.npz", **arrays)

    # ------------------------------------------------------------------ lifecycle
    def finish(self, summary: dict | None = None, status: str = "ok") -> None:
        elapsed = round(time.time() - self._t0, 1)
        self.save_json({"status": status, "seconds": elapsed, **(summary or {})}, "summary")
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if status == "ok":
            mirror = config.RESULTS_DIR / self.name
            if mirror.exists():
                shutil.rmtree(mirror)
            shutil.copytree(self.dir, mirror, ignore=shutil.ignore_patterns("arrays"))
        history = config.RESULTS_DIR / "runs.csv"
        row = pd.DataFrame([{"run_id": self.id, "name": self.name, "status": status, "seconds": elapsed,
                             "command": " ".join(sys.argv)}])
        if history.exists():
            row = pd.concat([pd.read_csv(history), row], ignore_index=True)
        row.to_csv(history, index=False)
        self.log.info("run %s finished (%s) in %.1fs", self.id, status, elapsed)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.log.exception("run failed")
            self.finish(status=f"failed: {exc_type.__name__}")
        return False

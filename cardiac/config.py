"""Project-wide paths and runtime settings.

Every path can be overridden with an environment variable so the code runs
unchanged on another machine:

    CARDIAC_DATA     folder containing Train/ and Test/ (default: data/acdc)
    CARDIAC_META     folder containing metaDataTrain.csv / metaDataTest.csv
    CARDIAC_CACHE    where the index and extracted features are stored
    CARDIAC_RUNS     where experiment runs (logs, metrics, figures) are written
    CARDIAC_RESULTS  where the latest successful run of each experiment is mirrored

The data folders are only ever read: nothing is written next to the images.
"""
import os
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(os.environ.get("CARDIAC_DATA", PROJECT_ROOT / "data" / "acdc"))
META_DIR = Path(os.environ.get("CARDIAC_META", PROJECT_ROOT / "data"))
CACHE_DIR = Path(os.environ.get("CARDIAC_CACHE", PROJECT_ROOT / "cache"))
RUNS_DIR = Path(os.environ.get("CARDIAC_RUNS", PROJECT_ROOT / "runs"))
RESULTS_DIR = Path(os.environ.get("CARDIAC_RESULTS", PROJECT_ROOT / "results"))
CONFIGS_DIR = PROJECT_ROOT / "configs"

INDEX_PATH = CACHE_DIR / "index.csv"
FEATURES_DIR = CACHE_DIR / "features"

# Segmentation labels (ACDC convention). In the test set the LV cavity is not given.
BACKGROUND, RV, MYO, LV = 0, 1, 2, 3
CLASS_NAMES = {0: "NOR", 1: "MINF", 2: "DCM", 3: "HCM", 4: "ARV"}
PHASES = ("ED", "ES")

DEFAULT_SEED = 0


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass

#!/usr/bin/env bash
# Full pipeline, in order. Each step writes a run in runs/ and mirrors it to results/<name>/.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"   # interpreter, e.g. PYTHON=python inside a virtual environment

$PY scripts/01_build_index.py
$PY scripts/02_lv_reconstruction.py
$PY scripts/03_extract_features.py
$PY scripts/04_protocol_study.py                       # ~14 min on an M3 (8 cores)
$PY scripts/05_benchmark.py --config benchmark.yaml    # ~50 min (soft-vote ensembles)
$PY scripts/05_benchmark.py --config lv_protocol.yaml   # ~18 min
# -> update configs/final.yaml with the chosen feature set / model, then:
$PY scripts/06_final_model.py --config final.yaml      # ~7 min, writes results/final_model/submission.csv

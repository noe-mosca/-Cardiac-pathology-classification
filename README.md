# Cardiac Pathology Classification from MRI Segmentations

IMA205 challenge, Télécom Paris — Noé Mosca.

We classify patients of the ACDC dataset into five groups (NOR, MINF, DCM, HCM, ARV) from the
segmentations of their end-diastolic and end-systolic cardiac MRI, height and weight: 100 labeled
training patients (20 per group), 50 test patients whose left-ventricular (LV) cavity is not
segmented. Every accuracy is estimated by nested cross-validation, and every experiment is a
script that logs its configuration, metrics, tables (CSV + LaTeX) and figures (PNG + PDF).

## Main results

Nested cross-validation (10 repeats of stratified 5-fold CV, hyper-parameters selected inside the
training folds), LV reconstructed for training and evaluated patients as for the test set:

| | Volumetric features (21) | Volumetric + wall thickness (32) |
|---|---|---|
| Logistic regression | 0.896 ± 0.016 | 0.953 ± 0.014 |
| Linear SVM | 0.925 ± 0.016 | 0.956 ± 0.014 |
| Random forest | 0.923 ± 0.016 | 0.938 ± 0.015 |
| XGBoost | 0.920 ± 0.013 | 0.916 ± 0.021 |
| Ensemble of 50 MLPs | 0.920 ± 0.012 | 0.957 ± 0.008 |
| **Final model**: MLP ensemble + linear SVM, α = 0.5 | 0.927 ± 0.012 | **0.966 ± 0.011** |

* Selecting hyper-parameters on a single 5-fold split and reporting the best score gives 0.91 to
  0.95 on the volumetric features; by nested CV the same models and grids reach 0.90 to 0.94, and
  an ensemble weight α chosen on the evaluation folds changes from 0 to 0.9 with the random seed.
* Filling the holes of the myocardium recovers the missing LV cavity with a mean Dice of 0.997;
  a convex-hull fallback on open basal slices brings it to 0.9998. Training on the true or the
  reconstructed LV makes no measurable difference.
* Myocardial wall-thickness features add 3 to 6 points to linear models and to the MLP ensemble;
  the remaining errors are infarcted hearts that look dilated, and conversely.

## Layout

```
cardiac/                library
  config.py             paths (overridable by env vars), labels, class names, seeds
  data.py               metadata, segmentations (test LV label stripped on load), patient index
  lv.py                 reconstruction of the LV cavity (not given in the test set)
  features.py           21 volumetric biomarkers + myocardial wall thickness, feature sets
  models.py             scikit-learn pipelines, batched MLP ensemble, soft-vote ensemble
  evaluate.py           nested cross-validation, metrics, confusion matrices
  tracking.py           Run: logs, metrics, tables, figures -> runs/ and results/
  viz.py                figures
scripts/                01 ... 06, run in order (see run_all.sh)
configs/                YAML configs of the experiments
data/                   metaDataTrain/Test.csv (height, weight, label), SampleSubmission.csv
data/acdc/              the segmentations (not included, see below)
results/<experiment>/   latest successful run of each experiment (tables, metrics, figures)
results/runs.csv        history of all runs
report/                 the report (PDF and LaTeX sources)
cache/, runs/           extracted features and full run folders (git-ignored)
```

## Pipeline

| step | script | what it does |
|---|---|---|
| 1 | `01_build_index.py` | indexes the patients, checks the label files, class statistics, example slices |
| 2 | `02_lv_reconstruction.py --config lv_reconstruction.yaml` | LV-cavity reconstruction evaluated on the training set (Dice, volume and ejection-fraction errors) |
| 3 | `03_extract_features.py` | features for every LV source (true, filled, filled + hull), effect of the reconstruction on the features |
| 4 | `04_protocol_study.py --config protocol_study.yaml` | single-split model selection vs nested CV, MLPs scored on their training patients, α chosen on the evaluation folds, feature selection before the CV |
| 5 | `05_benchmark.py --config X.yaml` | nested CV of models × feature sets × LV protocols: `benchmark`, `lv_protocol` |
| 6 | `06_final_model.py --config final.yaml` | final model: nested-CV estimate, permutation importance, test predictions → `results/final_model/submission.csv` |

```bash
pip install -r requirements.txt
./run_all.sh                                                        # or run the scripts one by one
python scripts/05_benchmark.py --config benchmark.yaml --repeats 2   # quick test
```

The whole pipeline takes about 1 h 30 on an Apple M3 (8 CPU cores, no GPU), mostly to train MLP
ensembles inside the nested cross-validation (the 50 MLPs are trained as one batched network).

## Data

The ACDC segmentations are not distributed with the code. Put the challenge folders `Train/` and
`Test/` (one sub-folder per patient with `<id>_{ED,ES}.nii` and `<id>_{ED,ES}_seg.nii`) in
`data/acdc/`, or point `CARDIAC_DATA` to them. The data folders are only read; if the test
segmentations contain an LV label, it is removed on load. `CARDIAC_META`, `CARDIAC_CACHE`,
`CARDIAC_RUNS` and `CARDIAC_RESULTS` relocate the other folders.

## Evaluation

* **Nested cross-validation**: 10 repeats × stratified 5-fold outer CV; hyper-parameters (and the
  MLP / other mixing weight α) chosen by an inner 5-fold CV on the training folds; scaling fitted
  on the training folds only. Reported as mean ± std over repeats.
* All configurations share the same outer splits, so differences are paired (`delta_vs_best`).
* Accuracy, balanced accuracy, macro-F1, per-class recall, pooled confusion matrices, permutation
  importance on held-out folds.
* With 100 patients one error is one point of accuracy: differences below ~2 points are noise.

## Report

[`report/report.pdf`](report/report.pdf): the report in ICLR format, with an appendix that gives
every table. Its LaTeX sources are in `report/`, and its figures and tables are regenerated from
`results/` by `report/regenerate_figures.py` and `report/regenerate_supplementary.py`.

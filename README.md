# 🫀 Cardiac Pathology Classification

This project aims to classify five types of cardiac pathologies based on cardiac MRI data using an ensemble of machine learning models.

##  Objective

Develop a machine learning pipeline to predict cardiac pathologies using MRI data from two phases:
- End-Diastole (ED)
- End-Systole (ES)

Each patient is associated with four `.nii` images:
- ED image
- ED segmentation
- ES image
- ES segmentation

The task is to classify patients into one of five cardiac conditions:
1. Healthy controls (label 0)
2. Myocardial infarction (label 1)
3. Dilated cardiomyopathy (label 2)
4. Hypertrophic cardiomyopathy (label 3)
5. Abnormal right ventricle (label 4)

---

## 🗂 Dataset

- **Training set:** 100 patients with 4 `.nii` images each and labeled pathology
- **Test set:** 50 patients with identical image formats but **no labels**

Additional data:
- Weight and height per patient (used to compute body surface area and indexed volumes)

---

## ⚙ Feature Engineering

23 features were extracted from the segmented images, inspired by clinical biomarkers from peer-reviewed literature (PubMed, Springer). Examples include:
- Left/right ventricular volumes at ED and ES
- Myocardial mass
- Ejection fractions
- Ratios and indexed volumes
- Patient body surface area and anthropometric data

---

##  Model Pipeline

###  Data Augmentation (optional)
- Random rotations, flips, translations, and scaling were applied to artificially expand the dataset.
- However, augmentation led to **performance degradation** and was eventually discarded for the final model.

###  Base Models
Several base classifiers were tested and evaluated via cross-validation:
- **Multi-layer Perceptrons (MLP)** trained in ensemble (50 instances)
- **Random Forest**
- **Support Vector Machine (SVM)**
- **XGBoost**
- **K-Nearest Neighbors (KNN)**

###  Ensemble Learning
Two main combinations were retained:
- `MLP + Random Forest`
- `MLP + SVM`

The final prediction is a weighted average of the outputs:
Prediction = α * MLP + (1 - α) * OtherClassifier

Best results were obtained when both models contributed equally or slightly favoring MLP.

---

##  Results

- **Best validation accuracy:** ~96%
- **Best models:** MLP + SVM and MLP + Random Forest
- **Feature reduction** slightly decreased accuracy but improved generalization

---

##  Post-processing (Planned)

A **KNN-based refinement** was proposed to improve classification between similar classes (e.g., label 1 and 2), but was not implemented in the current pipeline.

---

##  Tools & Libraries

- `numpy`, `pandas`, `scikit-learn`
- `PyTorch` for deep learning (MLP)
- `joblib` for model serialization
- `StratifiedKFold` for robust cross-validation
- `matplotlib`, `seaborn` for visualization

---

##  Author

Project by **Noé Mosca** — developed as part of the IMA05 course final project.

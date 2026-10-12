# Predicting 30-Day Hospital Readmission in Diabetic Patients

Summative project for the Machine Learning module: a multi-class classification study comparing
**traditional machine learning (scikit-learn)** with **deep learning (TensorFlow/Keras with `tf.data`)**.

## Project links
- **Google Drive folder** (raw dataset zip, best model weights): `<ADD LINK>`
- **Video presentation**: `<ADD LINK>`
- **Report**: `<ADD LINK OR FILE NAME>`

## Problem
Predict whether a hospitalised diabetic patient will be readmitted within 30 days (`<30`), after 30 days (`>30`),
or not at all (`NO`), to support discharge planning and hospital resource allocation.

## Dataset
*Diabetes 130-US Hospitals for Years 1999-2008*, UCI Machine Learning Repository
(about 101,000 encounters, 50 features). Original study: B. Strack et al., "Impact of HbA1c Measurement on Hospital
Readmission Rates: Analysis of 70,000 Clinical Database Patient Records," *BioMed Research International*, 2014.

The raw data is **not stored in this repository**. Download it from the Google Drive folder above (or from UCI) and
place `diabetic_data.csv` and `IDs_mapping.csv` in a `data/` folder.

## Repository structure
```
.
├── diabetes_readmission.ipynb   # full pipeline: EDA, cleaning, features, ML, DL, evaluation
├── requirements.txt
├── figures/                     # saved plots used in the report
├── data/                        # (git-ignored) raw and processed data
└── README.md
```

## How to run
1. `pip install -r requirements.txt`
2. Put `diabetic_data.csv` and `IDs_mapping.csv` in `data/`.
3. Open `diabetes_readmission.ipynb` and run all cells from top to bottom (or open it in Google Colab).

Random seed is fixed (`SEED = 42`). Weights & Biases runs in offline mode, so no account is needed.

## Methodology highlights
- Encounters ending in death or hospice are removed (they cannot be readmitted).
- Data is split **by patient** (train/validation/test = 70/15/15) to prevent leakage from repeat patients.
- Missing values (coded as `?`) are handled explicitly; `weight` (about 97% missing) is dropped.
- Metrics: macro-F1, per-class precision/recall, confusion matrices and ROC-AUC, because the classes are imbalanced.

## Results
*To be completed after experiments (see the experiment table in the notebook and report).*

## Acknowledgements
Dataset: UCI Machine Learning Repository. List any pretrained models, shared code or collaborators here.
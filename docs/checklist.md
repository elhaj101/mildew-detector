# Project Checklist: Cherry Leaf Mildew Detector

Use this checklist to track progress in converting Assessment "No's" to "Yes's".

## LO2: Business Requirements
- [ ] Update `README.md` with "Rationale to map business requirements to Data Viz & ML tasks".
- [ ] Update User Stories in `README.md` to follow standard format ("As a... I want... so that...").

## LO1 & LO4: Hypothesis & Validation (Merit)
- [x] Define project hypothesis in `README.md`: "Powdery mildew leaves have distinct white patches...".
- [x] Validate in `Visualization.ipynb`:
    - [x] Average Image (Healthy vs Mildew).
    - [x] Difference between Averages.
    - [x] Image Variability.
- [x] Validate via Model Performance (Accuracy > 97%).
- [ ] Document conclusions in `README.md`.

## LO3 & LO6: Data Analysis & Visualization
- [x] Update `Visualization.ipynb`:
    - [x] Check for data imbalance.
    - [x] Create Image Montage function.
- [x] Implement Dashboard Pages:
    - [x] **Page 1: Summary**: Project terms, dataset details, requirements.
    - [x] **Page 2: Visualizer**: Checkboxes for Difference/Variability, Montage.
    - [x] **Page 3: Detector**: File uploader, prediction output, results table.
    - [x] **Page 4: Hypothesis**: Text explanation.
    - [x] **Page 5: ML Performance**: Loss/Accuracy plots, confusion matrix.

## LO3, LO4, LO5: ML Business Case & Evaluation
- [x] Update `ModelingandEvaluation.ipynb`:
    - [x] Data Splitting.
    - [x] Data Augmentation (rotation, zoom, etc.).
    - [x] Model Architecture (CNN).
    - [x] Hyperparameter Tuning.
    - [x] Evaluation on Test Set.
- [x] Document ML Business Case in `README.md` (Aim, Outcome, Metrics).

## LO5.6: Code Structure
- [x] Refactor `app.py` to use `app_pages/` directory for modularity.

## General
- [x] Ensure all Dashboard plots have text interpretation.
- [x] Ensure Notebooks have Objectives/Inputs/Outputs descriptions.
- [x] Verify `requirements.txt` and `Procfile`.


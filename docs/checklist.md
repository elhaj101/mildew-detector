# Project Checklist: Cherry Leaf Mildew Detector

Use this checklist to track progress in converting Assessment "No's" to "Yes's".

## LO2: Business Requirements
- [ ] Update `README.md` with "Rationale to map business requirements to Data Viz & ML tasks".
- [ ] Update User Stories in `README.md` to follow standard format ("As a... I want... so that...").

## LO1 & LO4: Hypothesis & Validation (Merit)
- [ ] Define project hypothesis in `README.md`: "Powdery mildew leaves have distinct white patches...".
- [ ] Validate in `Visualization.ipynb`:
    - [ ] Average Image (Healthy vs Mildew).
    - [ ] Difference between Averages.
    - [ ] Image Variability.
- [ ] Validate via Model Performance (Accuracy > 97%).
- [ ] Document conclusions in `README.md`.

## LO3 & LO6: Data Analysis & Visualization
- [ ] Update `Visualization.ipynb`:
    - [ ] Check for data imbalance.
    - [ ] Create Image Montage function.
- [ ] Implement Dashboard Pages:
    - [ ] **Page 1: Summary**: Project terms, dataset details, requirements.
    - [ ] **Page 2: Visualizer**: Checkboxes for Difference/Variability, Montage.
    - [ ] **Page 3: Detector**: File uploader, prediction output, results table.
    - [ ] **Page 4: Hypothesis**: Text explanation.
    - [ ] **Page 5: ML Performance**: Loss/Accuracy plots, confusion matrix.

## LO3, LO4, LO5: ML Business Case & Evaluation
- [ ] Update `ModelingandEvaluation.ipynb`:
    - [ ] Data Splitting.
    - [ ] Data Augmentation (rotation, zoom, etc.).
    - [ ] Model Architecture (CNN).
    - [ ] Hyperparameter Tuning.
    - [ ] Evaluation on Test Set.
- [ ] Document ML Business Case in `README.md` (Aim, Outcome, Metrics).

## LO5.6: Code Structure
- [ ] Refactor `app.py` to use `app_pages/` directory for modularity.

## General
- [ ] Ensure all Dashboard plots have text interpretation.
- [ ] Ensure Notebooks have Objectives/Inputs/Outputs descriptions.
- [ ] Verify `requirements.txt` and `Procfile`.

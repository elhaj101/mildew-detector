# Actual Workflow: Cherry Leaf Mildew Detector

This workflow is customized for the **Cherry Leaf Mildew Detector** project to ensure all assessment criteria are met and "No" grades are converted to "Yes".

## 1. Business Requirements & User Stories (LO2)
### Objective: Define clear requirements and map them to technical tasks.
- [ ] **Action**: Update `README.md` with the "Rationale to map the business requirements to the Data Visualisations and ML tasks".
    -   **Requirement 1**: Visually differentiate healthy vs. mildew leaves.
        -   *Task*: Data Visualization (Image Montage, Average Image, Difference Plot).
    -   **Requirement 2**: Predict if a leaf is healthy or contains mildew.
        -   *Task*: ML Classification (CNN).
- [ ] **Action**: Ensure User Stories in `README.md` follow the format: "As a [role], I want [action], so that [benefit]".
    -   *Refine*: "As a farm manager, I want to differentiate healthy from infected leaves visually..."
    -   *Refine*: "As a worker, I want to predict disease instantly..."

## 2. Hypothesis & Validation (LO1, LO2, LO4 - Merit)
### Objective: Scientific validation of the project.
- [ ] **Action**: Define Hypothesis in `README.md` and `dashboard`.
    -   *Hypothesis*: "Cherry leaves infected with powdery mildew have distinct white, powdery fungal patches that differentiate them from healthy leaves."
- [ ] **Action**: Validate Hypothesis (Data Analysis).
    -   *Task*: In `Visualization.ipynb`, create an "Average Image" and "Difference between Averages" to show the white pattern.
    -   *Task*: Compute and visualize "Image Variability" (Standard Deviation) to show where changes occur.
- [ ] **Action**: Validate Hypothesis (Model Performance).
    -   *Task*: Verify model accuracy > 97% (aim for high accuracy) to prove the pattern is learned.
- [ ] **Action**: Document conclusions in `README.md` (Hypothesis Validation section).

## 3. Data Analysis & Visualization (LO3, LO6)
### Objective: Explore data and visualize findings.
- [ ] **Action**: Update `Visualization.ipynb`:
    -   [ ] Check for data imbalance (bar chart of class distribution).
    -   [ ] Create Image Montage function.
    -   [ ] Plot Average Image (Healthy vs Mildew).
    -   [ ] Plot Difference between Averages.
- [ ] **Action**: Dashboard Implementation (`app.py` & `app_pages/`):
    -   [ ] **Page 1: Project Summary**: Project terms, dataset details, business requirements.
    -   [ ] **Page 2: Leaves Visualizer**:
        -   Checkbox: "Difference between average and variability image".
        -   Checkbox: "Differences between average healthy and average mildew leaf".
        -   Checkbox: "Image Montage".
    -   [ ] **Page 3: Mildew Detector**:
        -   File uploader.
        -   Prediction output (class & probability).
        -   Table with results.
    -   [ ] **Page 4: Project Hypothesis**: Text explaining the hypothesis and validation.
    -   [ ] **Page 5: ML Performance**:
        -   Train/Validation Loss & Accuracy plots.
        -   Confusion Matrix? (If applicable/available).
        -   Generalization info.

## 4. Machine Learning Business Case & Evaluation (LO3, LO4, LO5)
### Objective: Train and evaluate the model.
- [ ] **Action**: Update `ModelingandEvaluation.ipynb`:
    -   [ ] Split data (Train/Validation/Test).
    -   [ ] Data Augmentation (rotation, zoom, flip) - *explain why*.
    -   [ ] Model Architecture (CNN) - *document why layers were chosen*.
    -   [ ] Hyperparameter Tuning (document trials in notebook or `docs/`).
    -   [ ] Evaluate on Test Set.
    -   [ ] Save model as `cherry-mildew-model.h5`.
- [ ] **Action**: Document Business Case in `README.md`:
    -   **Aim**: Classify cherry leaves.
    -   **Ideal Outcome**: Instant detection in the field.
    -   **Success Metric**: Accuracy > 97% on test set.

## 5. Dashboard Code Structure (LO5.6)
### Objective: Clean, modular code.
- [ ] **Action**: Refactor `app.py` to use a multi-page app structure.
    -   [ ] Create `app_pages/` directory.
    -   [ ] Create `src/` directory for helper functions (data loading, plotting).
    -   [ ] `app_pages/page_summary.py`
    -   [ ] `app_pages/page_visualizer.py`
    -   [ ] `app_pages/page_detector.py`
    -   [ ] `app_pages/page_project_hypothesis.py`
    -   [ ] `app_pages/page_ml_performance.py`
    -   [ ] Main `app.py` should import and render these pages based on sidebar selection.

## 6. Final Polish (LO6)
- [ ] **Action**: Ensure all dashboard plots have textual interpretation.
- [ ] **Action**: Ensure all notebook cells have markdown descriptions of Inputs, Objectives, and Outputs.
- [ ] **Action**: Verify `requirements.txt` is minimal and correct.
- [ ] **Action**: Verify `Procfile` is correct for Heroku.

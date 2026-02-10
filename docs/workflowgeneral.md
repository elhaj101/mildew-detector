# Workflow to Address Assessment Failures

This workflow is designed to systematically turn all "No" grades in the Assessment Report (`docs/Assessment.md`) into "Yes" by addressing the specific criteria in `docs/learningobjectives.md`.

## 1. Business Requirements & User Stories (LO2)
### Address Criteria 2.1 & 2.2 (Pass)
- [ ] **Action**: Map business requirements to Data Visualization and Machine Learning tasks.
- [ ] **Action**: Update the README file to include a section "Rationale to map the business requirements to the Data Visualisations and ML tasks".
- [ ] **Detail**: Ensure at least one ML task is explicitly mentioned in this section.
- [ ] **Format**: parameters should be strictly User Story based format (As a [role], I want [action], so that [benefit]).

## 2. Hypothesis & Validation (LO1, LO2, LO4 - Merit)
### Address Criteria 1.2, 2.3, 4.3
- [ ] **Action**: Clearly state project hypotheses in the README.
- [ ] **Action**: Describe the validation process for each hypothesis.
- [ ] **Action**: Provide statistical evidence (e.g., correlation coefficients, p-values, accuracy scores) to support or refute these hypotheses.
- [ ] **Action**: Explicitly state conclusions in both the README and the Dashboard.

## 3. Data Analysis & Visualization (LO3, LO6)
### Address Criteria 3.1 (Pass) & 6.2, 6.4, 6.5 (Merit/Pass)
- [ ] **Action**: Conduct comprehensive data analysis in Jupyter Notebooks (Pandas Profiling, Correlation, Imbalance checks).
- [ ] **Action**: Implement at least 4 relevant plots in the Dashboard.
- [ ] **Action**: Ensure at least one plot is **interactive** (using Altair, Plotly, or Bokeh).
- [ ] **Action**: Provide textual interpretation for every plot on the dashboard.

## 4. Machine Learning Business Case & Evaluation (LO3, LO4, LO5)
### Address Criteria 3.2 (Pass) & 4.1, 4.2 (Pass)
- [ ] **Action**: Define and document the Business Case for the ML task.
    -   Aim of the predictive task.
    -   Learning method (Classification/Regression).
    -   Ideal outcome.
    -   Success/Failure metrics.
    -   Relevance/Output for the user.
- [ ] **Action**: Provide objective conclusive proof in the dashboard that the analysis supports/refutes the hypothesis.
- [ ] **Action**: State clearly if the model meets the performance requirements defined in the business case.

## 5. Model Development & Tuning (LO5 - Merit)
### Address Criteria 5.7
- [ ] **Action**: Document the iterations of hyperparameter tuning (e.g., GridSearch, RandomSearch results).
- [ ] **Action**: Document model evolution (baseline model vs. final model) and explain why the final model was chosen.
- [ ] **Location**: This should be in the Jupyter Notebooks and summarized in the README/Dashboard.

## 6. Dashboard Implementation (LO6)
- [ ] **Action**: Ensure all pages have necessary subtext and explanations (Criteria 6.1).
- [ ] **Action**: Ensure the main navigation menu covers all requirements (Criteria 6.3).

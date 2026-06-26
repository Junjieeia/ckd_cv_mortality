# CKD Cardiovascular Outcomes and In-Hospital Mortality Prediction

A multimodal machine-learning framework integrating 12-lead ECG signals with clinical and laboratory features to predict two outcomes in chronic kidney disease: long-term cardiovascular events (Task 1) and in-hospital mortality during the index admission (Task 2).

---

## Overview

The framework addresses three methodological gaps common in the literature:

1. **Single-modality limitation** — combines raw ECG signals with structured clinical and laboratory variables via a dual-branch neural architecture.
2. **Binary-outcome oversimplification** — models the long-term cardiovascular composite under competing risks (DeepHit), retaining non-cardiovascular death and progression to kidney replacement therapy as competing events.
3. **Internal-only validation** — reserves an entire third center as a fully held-out external test set for transportability assessment, including calibration drift measurement and recalibration.

An interpretable nomogram (penalized logistic regression for Task 2; penalized Cox for Task 1) is provided as a paper-based bedside approximation, with its performance gap relative to the primary model stated explicitly. A prespecified modality ablation quantifies the incremental value of the ECG rather than assuming it.

---

## Project Structure

```
ckd_cv_mortality/
├── config.py                          # All hyperparameters, paths, and constants
├── main.py                            # Entry point — orchestrates the full pipeline
├── requirements.txt
├── data/                              # Place development.csv and external.csv here
├── preprocessing/
│   ├── data_loader.py                 # Eligibility filtering, MICE imputation, scaling
│   ├── ecg_preprocessor.py            # ECG denoising, SQI scoring, augmentation
│   └── feature_selection.py           # LASSO + Boruta + VIF selection
├── models/
│   ├── classical_models.py            # Logistic, XGBoost, LightGBM, RF, Cox, RSF, benchmark equations
│   ├── deep_models.py                 # ECG branch (frozen backbone + attention), tabular MLP, multimodal fusion, DeepSurv, DeepHit
│   └── trainer.py                     # Training loops with early stopping and class-weight handling
├── evaluation/
│   ├── metrics.py                     # AUC, C-index, calibration, DeLong test, NRI, IDI, fairness
│   └── decision_curve.py              # Net benefit, DCA, simulated clinical impact
├── interpretability/
│   └── explainer.py                   # SHAP, 1D Grad-CAM, attention weight extraction, t-SNE
├── nomogram/
│   └── nomogram_builder.py            # Point-allocation scoring and risk-score mapping
├── figures/
│   └── figure_generator.py            # All publication-quality plots
└── utils/
    ├── cross_validation.py             # Stratified k-fold CV and Bayesian HPO
    └── sensitivity_analysis.py         # Complete-case, missing-indicator, SQI-stratified ablation
```

---

## Data Format

Place two CSV files in `data/`:

| Column | Description |
|---|---|
| `age` | Age in years |
| `sex_female` | 1 = female, 0 = male |
| `egfr_baseline` | eGFR at index date (mL/min/1.73 m²) |
| `egfr_slope_12m` | 12-month eGFR slope (mL/min/yr) |
| `proteinuria_uacr` | Urinary albumin-to-creatinine ratio (mg/g) |
| `hemoglobin` | Hemoglobin (g/dL) |
| `hemoglobin_slope` | Hemoglobin trend (g/dL/yr) |
| `serum_albumin` | Serum albumin (g/dL) |
| `serum_phosphate` | Serum phosphate (mg/dL) |
| `serum_creatinine` | Serum creatinine (mg/dL) |
| `platelet_count` | Platelet count (×10⁹/L) |
| `systolic_bp` | Systolic blood pressure (mmHg) |
| `admission_glucose` | Admission glucose (mg/dL) |
| `ldl_cholesterol` | LDL cholesterol (mg/dL) |
| `diabetes_mellitus` | 1 = yes, 0 = no |
| `hypertension` | 1 = yes, 0 = no |
| `dialysis_maintenance` | 1 = on dialysis, 0 = no |
| `cvd_history` | Prior cardiovascular disease: 1 = yes, 0 = no |
| `cv_event` | Task 1 event indicator: 1 = cardiovascular composite, 0 = censored/competing |
| `time_to_event` | Follow-up time in years |
| `competing_event` | Competing event indicator (non-CV death or KRT) |
| `in_hospital_mortality` | Task 2 label: 1 = died, 0 = survived |
| `ecg_sqc_pass` | ECG quality control passed: 1 = yes |
| `n_egfr_measurements` | Number of eGFR values in lookback window |
| `n_proteinuria_measurements` | Number of proteinuria values in lookback window |
| `n_hgb_measurements` | Number of hemoglobin values in lookback window |
| `core_values_valid` | Core values passed range checks: 1 = yes |
| `malignancy_short_survival` | Malignancy with expected survival < 6 months: 1 = yes |
| `kidney_transplant_prior` | Prior kidney transplant: 1 = yes |
| `pregnancy` | Pregnant at index date: 1 = yes |

ECG waveforms are expected as separate NumPy arrays (shape: `n_patients × 12 × 5000`) stored alongside the CSV, referenced by patient ID.

---

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11 is recommended. GPU use is automatic when available (CUDA); CPU fallback is supported.

The ECG branch uses the [ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder) pretrained backbone (Li et al., NEJM AI 2025). Download the checkpoint and place it at the path expected by your backbone loader before running the multimodal pipeline.

---

## Running the Pipeline

```bash
python main.py --dev data/development.csv --ext data/external.csv
```

This executes sequentially:

1. Eligibility filtering and exclusion logging
2. MICE imputation (20 imputations, parameters fitted on development only)
3. Tabular feature selection: LASSO + Boruta union, then VIF screening
4. Task 2 (in-hospital mortality): trains Logistic, XGBoost, LightGBM, RF; selects operating threshold on development data; evaluates externally; builds nomogram; runs SHAP, DCA, fairness analysis
5. Task 1 (cardiovascular outcome): trains Cox, RSF, DeepSurv, DeepHit; evaluates C-index externally; compares against CKD-PC and KFRE benchmarks; builds nomogram; runs SHAP
6. Sensitivity analyses: complete-case and missing-indicator
7. All figures exported to `outputs/figures/`
8. Result summaries exported to `outputs/results/`

---

## Configuration

All parameters are centralized in `config.py`. Key entries:

| Parameter | Value | Meaning |
|---|---|---|
| `RANDOM_SEED` | 42 | Global random seed |
| `N_FOLDS` | 5 | Cross-validation folds |
| `N_IMPUTATIONS` | 20 | MICE imputation chains |
| `MIN_SENSITIVITY` | 0.70 | Minimum sensitivity for threshold selection (Youden + constraint) |
| `ECG_SAMPLING_RATE` | 500 | Hz |
| `ECG_DURATION_SEC` | 10 | Seconds per recording |
| `FOLLOW_UP_YEARS` | 3 | Evaluation time for nomogram risk readout |
| `SQI_HIGH_THRESHOLD` | 0.80 | Signal quality index cutoff for high-quality stratum |
| `N_BOOTSTRAP` | 2000 | Bootstrap resamples for confidence intervals (BCa) |

---

## Outputs

| Path | Content |
|---|---|
| `outputs/figures/task2_roc_external.pdf` | ROC curves – in-hospital mortality – external |
| `outputs/figures/task2_calibration_external.pdf` | Calibration plot – in-hospital mortality – external |
| `outputs/figures/task2_cm_xgboost.pdf` | Confusion matrix – XGBoost at operating threshold |
| `outputs/figures/task2_dca_external.pdf` | Decision curve – in-hospital mortality |
| `outputs/figures/task2_shap.pdf` | SHAP summary – in-hospital mortality |
| `outputs/figures/task2_fairness_age.pdf` | False-negative rate by age group |
| `outputs/figures/task2_nomogram.pdf` | Nomogram – in-hospital mortality |
| `outputs/figures/task1_nomogram.pdf` | Nomogram – cardiovascular outcome |
| `outputs/figures/task1_shap.pdf` | SHAP summary – cardiovascular outcome |
| `outputs/figures/task1_loss_deephit.pdf` | Training and validation loss curves – DeepHit |
| `outputs/results/task2_results.json` | AUC and threshold summary – Task 2 |
| `outputs/results/task1_results.json` | C-index summary – Task 1 |

---

## Key Design Decisions

**Frozen ECG backbone.** All layers of the pretrained ECGFounder backbone are frozen. Only a single linear projection head (~49,000 trainable parameters) is trained on fixed embeddings. This keeps the trainable parameter count compatible with the available event count (264 development events for Task 2) and prevents overfitting. The multimodal results are therefore treated as exploratory.

**Competing-risks formulation.** Task 1 uses DeepHit to model the cumulative incidence function directly, with non-cardiovascular death and progression to kidney replacement therapy handled as competing events rather than ordinary censoring. The Cox model uses the Fine–Gray subdistribution hazard for the same purpose.

**Threshold fixed on development data.** The operating threshold is selected on development data (Youden index subject to sensitivity ≥ 0.70) and applied unchanged to the external test set. Threshold drift is reported and motivates site-specific re-derivation before any local deployment.

**Missing data.** MICE with 20 chains and 20 iterations; parameters fitted on development data only and applied to the external center. Predictions combined by Rubin's rules. Complete-case and missing-indicator sensitivity analyses are prespecified.

**Fairness.** False-negative and false-positive rates reported by sex, age group, and recruiting site. Group-aware threshold selection is applied where a statistically significant disparity in the false-negative rate is detected.

---

## Reproducibility

The random seed is fixed at 42 across all stochastic components. `torch.use_deterministic_algorithms(True, warn_only=True)` is set. Residual non-determinism from certain GPU operations means bit-for-bit reproducibility is not guaranteed; reported metrics should be read with this small residual variance in mind.

---

## Limitations

- The in-hospital mortality cohort is enriched for patients under established longitudinal care (trajectory features require repeated prior measurements); acutely presenting patients without a prior record are excluded, representing a central threat to external validity for the acute task.
- The multimodal results are exploratory given the unfavorable parameter-to-event ratio for the ECG branch.
- Both models showed modest external calibration overestimation (calibration slopes 0.87–0.93 before recalibration); site-specific recalibration is required before any local deployment.
- The incremental ECG contribution is conditional on the specific frozen backbone used (ECGFounder); a different backbone may yield a different estimate.
- These models are not ready for clinical use. Independent replication and prospective evaluation are required.

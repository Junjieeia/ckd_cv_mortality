import os
from dataclasses import dataclass, field
from typing import List, Optional

RANDOM_SEED = 42
N_FOLDS = 5
N_IMPUTATIONS = 20
N_IMPUTATION_ITER = 20
N_BOOTSTRAP = 2000
FOLLOW_UP_YEARS = 3
MIN_SENSITIVITY = 0.70
SQI_HIGH_THRESHOLD = 0.80
SQI_LOW_THRESHOLD = 0.50
ECG_SAMPLING_RATE = 500
ECG_DURATION_SEC = 10
ECG_N_LEADS = 12
ECG_N_SAMPLES = ECG_SAMPLING_RATE * ECG_DURATION_SEC
ECG_WINDOW_HOURS = 48
LOOKBACK_MONTHS_EGFR = 12
MIN_EGFR_MEASUREMENTS = 3
MIN_PROTEINURIA_MEASUREMENTS = 2
MIN_HGB_MEASUREMENTS = 2
MAX_NOMOGRAM_FEATURES = 8
TASK1_THRESHOLDS = [0.20, 0.25, 0.30]
TASK2_THRESHOLDS = [0.10, 0.15, 0.20]

TABULAR_FEATURES_TASK1 = [
    "egfr_slope_12m",
    "egfr_baseline",
    "proteinuria_uacr",
    "age",
    "serum_albumin",
    "hemoglobin",
    "diabetes_mellitus",
    "serum_phosphate",
    "systolic_bp",
    "ldl_cholesterol",
    "platelet_count",
    "serum_creatinine",
    "dialysis_maintenance",
    "cvd_history",
]

TABULAR_FEATURES_TASK2 = [
    "serum_albumin",
    "age",
    "egfr_baseline",
    "hemoglobin",
    "platelet_count",
    "serum_phosphate",
    "admission_glucose",
    "diabetes_mellitus",
    "systolic_bp",
    "hemoglobin_slope",
    "egfr_slope_12m",
    "serum_creatinine",
]

NOMOGRAM_FEATURES_TASK1 = [
    "egfr_slope_12m",
    "egfr_baseline",
    "proteinuria_uacr",
    "age",
    "serum_albumin",
    "hemoglobin",
    "diabetes_mellitus",
    "serum_phosphate",
]

NOMOGRAM_FEATURES_TASK2 = [
    "serum_albumin",
    "age",
    "egfr_baseline",
    "hemoglobin",
    "platelet_count",
    "serum_phosphate",
    "admission_glucose",
    "diabetes_mellitus",
]

NOMOGRAM_RANGES_TASK2 = {
    "serum_albumin": (1.5, 5.0),
    "age": (18, 90),
    "egfr_baseline": (5, 59),
    "hemoglobin": (5, 16),
    "platelet_count": (50, 400),
    "serum_phosphate": (2, 9),
    "admission_glucose": (70, 400),
    "diabetes_mellitus": (0, 1),
}

NOMOGRAM_RANGES_TASK1 = {
    "egfr_slope_12m": (-15, 2),
    "egfr_baseline": (5, 59),
    "proteinuria_uacr": (30, 5000),
    "age": (18, 90),
    "serum_albumin": (1.5, 5.0),
    "hemoglobin": (5, 16),
    "diabetes_mellitus": (0, 1),
    "serum_phosphate": (2, 9),
}

PROTECTED_ATTRIBUTES = ["sex", "age_group", "race_ethnicity"]
AGE_THRESHOLD = 65

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_lambda": 1.5,
    "scale_pos_weight": 8.1,
    "random_state": RANDOM_SEED,
    "eval_metric": "auc",
    "use_label_encoder": False,
}

LIGHTGBM_PARAMS = {
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "feature_fraction": 0.7,
    "random_state": RANDOM_SEED,
    "verbosity": -1,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "min_samples_leaf": 15,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

LOGISTIC_PARAMS = {
    "penalty": "l2",
    "C": 0.1,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
}

DEEPSURV_PARAMS = {
    "n_layers": 3,
    "units": [256, 128, 64],
    "dropout": 0.4,
    "lr": 1e-4,
    "batch_size": 128,
    "epochs": 100,
    "patience": 15,
}

DEEPHIT_PARAMS = {
    "n_layers": 3,
    "units": [256, 128, 64],
    "dropout": 0.3,
    "alpha": 0.2,
    "sigma": 0.1,
    "lr": 5e-5,
    "batch_size": 64,
    "epochs": 120,
    "patience": 15,
}

RSF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "min_samples_leaf": 15,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

FINEGRAY_PARAMS = {
    "penalizer": 0.01,
}

MLP_TABULAR_PARAMS = {
    "units": [128, 64],
    "dropout": 0.5,
    "lr": 3e-4,
    "batch_size": 128,
    "weight_decay": 1e-3,
}

ECG_PROJECTION_HEAD_PARAMS = {
    "lr": 2e-5,
    "patience": 7,
    "embedding_dim": 64,
}

TEMPORAL_ATTENTION_PARAMS = {
    "n_heads": 2,
    "key_dim": 32,
}

FUSION_PARAMS = {
    "units": 64,
    "dropout": 0.5,
}

FT_TRANSFORMER_PARAMS = {
    "d_token": 192,
    "n_heads": 8,
    "d_ffn_factor": 4.0,
    "n_layers": 3,
    "dropout": 0.2,
    "lr": 1e-4,
    "batch_size": 64,
}

TABNET_PARAMS = {
    "n_d": 32,
    "n_a": 32,
    "n_steps": 5,
    "gamma": 1.5,
    "lr": 2e-2,
    "batch_size": 1024,
}

OUTPUT_DIR = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

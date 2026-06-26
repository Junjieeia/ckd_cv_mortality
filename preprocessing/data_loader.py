import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import sys
sys.path.append('..')
from config import (
    RANDOM_SEED, N_FOLDS, N_IMPUTATIONS, N_IMPUTATION_ITER,
    TABULAR_FEATURES_TASK1, TABULAR_FEATURES_TASK2,
    MIN_EGFR_MEASUREMENTS, MIN_PROTEINURIA_MEASUREMENTS, MIN_HGB_MEASUREMENTS,
    LOOKBACK_MONTHS_EGFR, AGE_THRESHOLD
)


def load_tabular_data(dev_path: str, ext_path: str):
    dev_df = pd.read_csv(dev_path)
    ext_df = pd.read_csv(ext_path)
    return dev_df, ext_df


def apply_eligibility_criteria(df: pd.DataFrame) -> pd.DataFrame:
    n_init = len(df)
    exclusion_log = {}

    mask_ecg = df["ecg_sqc_pass"].astype(bool)
    exclusion_log["ecg_quality_failure"] = (~mask_ecg).sum()
    df = df[mask_ecg].copy()

    mask_traj = (
        (df["n_egfr_measurements"] >= MIN_EGFR_MEASUREMENTS) &
        (df["n_proteinuria_measurements"] >= MIN_PROTEINURIA_MEASUREMENTS) &
        (df["n_hgb_measurements"] >= MIN_HGB_MEASUREMENTS)
    )
    exclusion_log["insufficient_trajectory"] = (~mask_traj).sum()
    df = df[mask_traj].copy()

    mask_vals = df["core_values_valid"].astype(bool)
    exclusion_log["implausible_values"] = (~mask_vals).sum()
    df = df[mask_vals].copy()

    mask_malig = ~df["malignancy_short_survival"].astype(bool)
    exclusion_log["malignancy"] = (~mask_malig).sum()
    df = df[mask_malig].copy()

    mask_tx = ~df["kidney_transplant_prior"].astype(bool)
    exclusion_log["kidney_transplant"] = (~mask_tx).sum()
    df = df[mask_tx].copy()

    mask_age = df["age"] >= 18
    exclusion_log["age_under_18"] = (~mask_age).sum()
    df = df[mask_age].copy()

    mask_preg = ~df["pregnancy"].astype(bool)
    exclusion_log["pregnancy"] = (~mask_preg).sum()
    df = df[mask_preg].copy()

    n_final = len(df)
    exclusion_log["total_excluded"] = n_init - n_final
    exclusion_log["total_included"] = n_final

    return df, exclusion_log


def compute_egfr_slope(measurements: pd.DataFrame, lookback_months: int = 12) -> float:
    if len(measurements) < MIN_EGFR_MEASUREMENTS:
        return np.nan
    cutoff = measurements["date"].max() - pd.DateOffset(months=lookback_months)
    subset = measurements[measurements["date"] >= cutoff]
    if len(subset) < MIN_EGFR_MEASUREMENTS:
        return np.nan
    x = (subset["date"] - subset["date"].min()).dt.days.values.reshape(-1, 1)
    y = subset["egfr"].values
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(x, y)
    slope_per_day = reg.coef_[0]
    slope_per_year = slope_per_day * 365.25
    return slope_per_year


def compute_trend(measurements: pd.DataFrame, value_col: str, min_count: int = 2) -> float:
    if len(measurements) < min_count:
        return np.nan
    x = (measurements["date"] - measurements["date"].min()).dt.days.values.reshape(-1, 1)
    y = measurements[value_col].values
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(x, y)
    return reg.coef_[0] * 365.25


def add_age_group(df: pd.DataFrame, threshold: int = AGE_THRESHOLD) -> pd.DataFrame:
    df = df.copy()
    df["age_group"] = np.where(df["age"] >= threshold, f">={threshold}", f"<{threshold}")
    return df


def split_development_folds(df: pd.DataFrame, task: str) -> list:
    if task == "task1":
        y = df["cv_event"].values
    else:
        y = df["in_hospital_mortality"].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = []
    for train_idx, val_idx in skf.split(df, y):
        folds.append((train_idx, val_idx))
    return folds


def fit_imputer(X_train: np.ndarray, feature_names: list) -> IterativeImputer:
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=N_IMPUTATION_ITER,
        random_state=RANDOM_SEED,
        sample_posterior=True,
    )
    imputer.fit(X_train)
    return imputer


def apply_multiple_imputation(X: np.ndarray, imputer: IterativeImputer, n_imputations: int = N_IMPUTATIONS) -> np.ndarray:
    imputed_arrays = []
    for i in range(n_imputations):
        imp = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=N_IMPUTATION_ITER,
            random_state=RANDOM_SEED + i,
            sample_posterior=True,
        )
        imp.fit_transform(X)
        imputed_arrays.append(imp.transform(X))
    return np.stack(imputed_arrays, axis=0)


def combine_rubins_rules(predictions_list: list) -> np.ndarray:
    stacked = np.stack(predictions_list, axis=0)
    return stacked.mean(axis=0)


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def preprocess_tabular(
    dev_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    task: str,
) -> tuple:
    if task == "task1":
        features = TABULAR_FEATURES_TASK1
        y_dev = dev_df[["cv_event", "time_to_event", "competing_event"]].values
        y_ext = ext_df[["cv_event", "time_to_event", "competing_event"]].values
    else:
        features = TABULAR_FEATURES_TASK2
        y_dev = dev_df["in_hospital_mortality"].values
        y_ext = ext_df["in_hospital_mortality"].values

    X_dev = dev_df[features].values
    X_ext = ext_df[features].values

    scaler = fit_scaler(X_dev)
    X_dev_scaled = scaler.transform(X_dev)
    X_ext_scaled = scaler.transform(X_ext)

    imputer = fit_imputer(X_dev_scaled, features)
    X_dev_imp = imputer.transform(X_dev_scaled)
    X_ext_imp = imputer.transform(X_ext_scaled)

    return X_dev_imp, y_dev, X_ext_imp, y_ext, scaler, imputer, features

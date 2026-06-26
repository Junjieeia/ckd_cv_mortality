import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
from typing import Dict, List, Tuple
import sys
sys.path.append('..')
from config import RANDOM_SEED


def complete_case_analysis(
    df: pd.DataFrame,
    feature_names: List[str],
    outcome_col: str,
) -> Tuple[pd.DataFrame, int]:
    subset = df[feature_names + [outcome_col]].dropna()
    n_dropped = len(df) - len(subset)
    return subset, n_dropped


def missing_indicator_augment(
    X: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    indicators = (np.isnan(X)).astype(float)
    missing_cols = [f"{f}_missing" for f in feature_names]
    any_missing = indicators.any(axis=0)
    cols_with_missing = [i for i, m in enumerate(any_missing) if m]

    if len(cols_with_missing) == 0:
        return X, feature_names

    X_filled = X.copy()
    X_filled[np.isnan(X_filled)] = 0.0

    indicator_subset = indicators[:, cols_with_missing]
    missing_names_subset = [missing_cols[i] for i in cols_with_missing]

    X_aug = np.hstack([X_filled, indicator_subset])
    aug_feature_names = feature_names + missing_names_subset
    return X_aug, aug_feature_names


def run_sensitivity_complete_case(
    df_dev: pd.DataFrame,
    df_ext: pd.DataFrame,
    feature_names: List[str],
    outcome_col: str,
    fit_and_eval: callable,
) -> Dict:
    dev_complete, n_dropped_dev = complete_case_analysis(df_dev, feature_names, outcome_col)
    ext_complete, n_dropped_ext = complete_case_analysis(df_ext, feature_names, outcome_col)

    X_dev = dev_complete[feature_names].values
    y_dev = dev_complete[outcome_col].values
    X_ext = ext_complete[feature_names].values
    y_ext = ext_complete[outcome_col].values

    result = fit_and_eval(X_dev, y_dev, X_ext, y_ext)

    return {
        "n_dev_complete": len(dev_complete),
        "n_dev_dropped": n_dropped_dev,
        "n_ext_complete": len(ext_complete),
        "n_ext_dropped": n_dropped_ext,
        "result": result,
    }


def run_sensitivity_missing_indicator(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_ext: np.ndarray,
    y_ext: np.ndarray,
    feature_names: List[str],
    fit_and_eval: callable,
) -> Dict:
    X_dev_aug, aug_names = missing_indicator_augment(X_dev, feature_names)
    X_ext_aug, _ = missing_indicator_augment(X_ext, feature_names)

    X_dev_filled = np.nan_to_num(X_dev_aug, nan=0.0)
    X_ext_filled = np.nan_to_num(X_ext_aug, nan=0.0)

    result = fit_and_eval(X_dev_filled, y_dev, X_ext_filled, y_ext)

    return {
        "n_augmented_features": len(aug_names),
        "augmented_feature_names": aug_names,
        "result": result,
    }


def compare_sensitivity_results(
    primary_metric: float,
    cc_metric: float,
    mi_metric: float,
    metric_name: str = "AUC",
) -> pd.DataFrame:
    return pd.DataFrame([
        {"analysis": "Primary (MICE)", "metric": metric_name, "value": primary_metric, "difference": 0.0},
        {"analysis": "Complete-case", "metric": metric_name, "value": cc_metric, "difference": cc_metric - primary_metric},
        {"analysis": "Missing-indicator", "metric": metric_name, "value": mi_metric, "difference": mi_metric - primary_metric},
    ])


def signal_quality_stratified_ablation(
    y_true: np.ndarray,
    y_prob_tabular: np.ndarray,
    y_prob_multimodal: np.ndarray,
    sqi_values: np.ndarray,
    sqi_high_threshold: float = 0.80,
    sqi_low_threshold: float = 0.50,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    from evaluation.metrics import compute_auc_ci, delong_test

    mask_high = sqi_values >= sqi_high_threshold
    mask_low = (sqi_values >= sqi_low_threshold) & (sqi_values < sqi_high_threshold)
    mask_full = np.ones(len(y_true), dtype=bool)

    records = []
    for label, mask in [("High-SQI (≥0.80)", mask_high), (f"Low-SQI (0.50–0.80)", mask_low), ("Full external test set", mask_full)]:
        if mask.sum() < 20:
            continue
        yt = y_true[mask]
        tab_p = y_prob_tabular[mask]
        mm_p = y_prob_multimodal[mask]

        auc_tab, _, _ = compute_auc_ci(yt, tab_p, n_bootstrap=n_bootstrap)
        auc_mm, _, _ = compute_auc_ci(yt, mm_p, n_bootstrap=n_bootstrap)

        delta, pval = delong_test(yt, mm_p, tab_p)

        rng = np.random.RandomState(RANDOM_SEED)
        boot_deltas = []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, mask.sum(), mask.sum())
            try:
                d, _ = delong_test(yt[idx], mm_p[idx], tab_p[idx])
                boot_deltas.append(d)
            except Exception:
                continue
        ci_low = np.percentile(boot_deltas, 100 * alpha / 2)
        ci_high = np.percentile(boot_deltas, 100 * (1 - alpha / 2))

        records.append({
            "subset": label,
            "n": mask.sum(),
            "tabular_auc": round(auc_tab, 3),
            "multimodal_auc": round(auc_mm, 3),
            "delta_auc": round(delta, 3),
            "ci_low": round(ci_low, 3),
            "ci_high": round(ci_high, 3),
            "p_value": round(pval, 3),
        })

    return pd.DataFrame(records)

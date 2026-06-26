import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import sys
sys.path.append('..')
from config import N_BOOTSTRAP, RANDOM_SEED


def compute_net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    n = len(y_true)
    net_benefits = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        nb = (tp / n) - (fp / n) * (t / (1 - t + 1e-8))
        net_benefits.append(nb)
    return np.array(net_benefits)


def compute_treat_all_net_benefit(y_true: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n = len(y_true)
    prevalence = y_true.mean()
    net_benefits = []
    for t in thresholds:
        nb = prevalence - (1 - prevalence) * (t / (1 - t + 1e-8))
        net_benefits.append(nb)
    return np.array(net_benefits)


def decision_curve_analysis(
    y_true: np.ndarray,
    models: Dict[str, np.ndarray],
    thresholds: np.ndarray = None,
    n_bootstrap: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)

    records = []
    for model_name, y_prob in models.items():
        nb = compute_net_benefit(y_true, y_prob, thresholds)
        for i, t in enumerate(thresholds):
            records.append({
                "model": model_name,
                "threshold": t,
                "net_benefit": nb[i],
            })

    treat_all_nb = compute_treat_all_net_benefit(y_true, thresholds)
    for i, t in enumerate(thresholds):
        records.append({
            "model": "treat_all",
            "threshold": t,
            "net_benefit": treat_all_nb[i],
        })
        records.append({
            "model": "treat_none",
            "threshold": t,
            "net_benefit": 0.0,
        })

    return pd.DataFrame(records)


def compute_net_benefit_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    nb = compute_net_benefit(y_true, y_prob, np.array([threshold]))[0]

    rng = np.random.RandomState(RANDOM_SEED)
    boot_nbs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_true))
        boot_nb = compute_net_benefit(y_true[idx], y_prob[idx], np.array([threshold]))[0]
        boot_nbs.append(boot_nb)

    ci_low = np.percentile(boot_nbs, 100 * alpha / 2)
    ci_high = np.percentile(boot_nbs, 100 * (1 - alpha / 2))
    return nb, ci_low, ci_high


def simulated_clinical_impact(
    y_true: np.ndarray,
    y_prob_primary: np.ndarray,
    y_prob_reference: np.ndarray,
    threshold_primary: float,
    threshold_reference: float,
    n_per_thousand: int = 1000,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = 0.05,
) -> Dict:
    def impact_at_threshold(y_t, y_p, thresh):
        n = len(y_t)
        y_pred = (y_p >= thresh).astype(int)
        tp = ((y_pred == 1) & (y_t == 1)).sum()
        fp = ((y_pred == 1) & (y_t == 0)).sum()
        true_events = tp / n * n_per_thousand
        unnecessary_escalations = fp / n * n_per_thousand
        return true_events, unnecessary_escalations

    true_primary, unnecc_primary = impact_at_threshold(y_true, y_prob_primary, threshold_primary)
    true_ref, unnecc_ref = impact_at_threshold(y_true, y_prob_reference, threshold_reference)

    rng = np.random.RandomState(RANDOM_SEED)
    boot_true_primary, boot_true_ref = [], []
    boot_unnecc_primary, boot_unnecc_ref = [], []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_true))
        t_p, u_p = impact_at_threshold(y_true[idx], y_prob_primary[idx], threshold_primary)
        t_r, u_r = impact_at_threshold(y_true[idx], y_prob_reference[idx], threshold_reference)
        boot_true_primary.append(t_p)
        boot_true_ref.append(t_r)
        boot_unnecc_primary.append(u_p)
        boot_unnecc_ref.append(u_r)

    def bca_ci(values, point_est, alpha=0.05):
        lo = np.percentile(values, 100 * alpha / 2)
        hi = np.percentile(values, 100 * (1 - alpha / 2))
        return lo, hi

    return {
        "primary_true_events": true_primary,
        "primary_true_events_ci": bca_ci(boot_true_primary, true_primary),
        "primary_unnecessary": unnecc_primary,
        "primary_unnecessary_ci": bca_ci(boot_unnecc_primary, unnecc_primary),
        "reference_true_events": true_ref,
        "reference_true_events_ci": bca_ci(boot_true_ref, true_ref),
        "reference_unnecessary": unnecc_ref,
        "reference_unnecessary_ci": bca_ci(boot_unnecc_ref, unnecc_ref),
        "net_events_gained": true_primary - true_ref,
        "net_escalations_avoided": unnecc_ref - unnecc_primary,
    }

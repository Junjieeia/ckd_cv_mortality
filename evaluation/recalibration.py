import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from scipy.stats import linregress
from sklearn.metrics import brier_score_loss
from typing import Dict, List, Tuple
import sys
sys.path.append('..')
from config import N_BOOTSTRAP, RANDOM_SEED


def compute_calibration_intercept_slope(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    if len(mean_pred) < 2:
        return {"intercept": np.nan, "slope": np.nan, "brier": np.nan}
    slope, intercept, r, p, se = linregress(mean_pred, frac_pos)
    brier = brier_score_loss(y_true, y_prob)
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r_squared": float(r ** 2),
        "brier": float(brier),
        "mean_pred": mean_pred,
        "frac_pos": frac_pos,
    }


def recalibrate_in_the_large(
    y_prob: np.ndarray,
    y_true_cal: np.ndarray,
    y_prob_cal: np.ndarray,
) -> np.ndarray:
    observed_mean = y_true_cal.mean()
    predicted_mean = y_prob_cal.mean()
    log_odds = np.log(y_prob / (1 - y_prob + 1e-10) + 1e-10)
    offset = np.log(observed_mean / (1 - observed_mean + 1e-10)) - \
             np.log(predicted_mean / (1 - predicted_mean + 1e-10))
    recal_prob = 1 / (1 + np.exp(-(log_odds + offset)))
    return np.clip(recal_prob, 1e-6, 1 - 1e-6)


def logistic_recalibration(
    y_prob_cal: np.ndarray,
    y_true_cal: np.ndarray,
    y_prob_apply: np.ndarray,
) -> np.ndarray:
    log_odds_cal = np.log(y_prob_cal / (1 - y_prob_cal + 1e-10) + 1e-10).reshape(-1, 1)
    recal_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    recal_model.fit(log_odds_cal, y_true_cal)
    log_odds_apply = np.log(y_prob_apply / (1 - y_prob_apply + 1e-10) + 1e-10).reshape(-1, 1)
    return recal_model.predict_proba(log_odds_apply)[:, 1]


def two_step_recalibration(
    y_prob_dev: np.ndarray,
    y_true_dev: np.ndarray,
    y_prob_ext: np.ndarray,
    y_true_ext: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    step1 = recalibrate_in_the_large(y_prob_ext, y_true_dev, y_prob_dev)
    step2 = logistic_recalibration(y_prob_dev, y_true_dev, step1)

    cal_before = compute_calibration_intercept_slope(y_true_ext, y_prob_ext)
    cal_after = compute_calibration_intercept_slope(y_true_ext, step2)

    return step2, {
        "calibration_before": cal_before,
        "calibration_after": cal_after,
        "intercept_before": cal_before["intercept"],
        "slope_before": cal_before["slope"],
        "intercept_after": cal_after["intercept"],
        "slope_after": cal_after["slope"],
        "brier_before": cal_before["brier"],
        "brier_after": cal_after["brier"],
    }


def net_benefit_change_from_recalibration(
    y_true: np.ndarray,
    y_prob_before: np.ndarray,
    y_prob_after: np.ndarray,
    threshold: float,
) -> float:
    def nb_at_threshold(y_t, y_p, t):
        n = len(y_t)
        y_pred = (y_p >= t).astype(int)
        tp = ((y_pred == 1) & (y_t == 1)).sum()
        fp = ((y_pred == 1) & (y_t == 0)).sum()
        return (tp / n) - (fp / n) * (t / (1 - t + 1e-10))

    nb_before = nb_at_threshold(y_true, y_prob_before, threshold)
    nb_after = nb_at_threshold(y_true, y_prob_after, threshold)
    return float(nb_after - nb_before)


def full_recalibration_report(
    models: Dict[str, Dict],
    thresholds: List[float] = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [0.10, 0.15, 0.20]

    records = []
    for model_name, model_data in models.items():
        y_true = model_data["y_true"]
        y_prob_before = model_data["y_prob_before"]
        y_prob_after = model_data["y_prob_after"]

        cal_before = compute_calibration_intercept_slope(y_true, y_prob_before)
        cal_after = compute_calibration_intercept_slope(y_true, y_prob_after)

        delta_nb_list = []
        for t in thresholds:
            delta_nb = net_benefit_change_from_recalibration(y_true, y_prob_before, y_prob_after, t)
            delta_nb_list.append(delta_nb)

        records.append({
            "model": model_name,
            "intercept_before": round(cal_before["intercept"], 3),
            "slope_before": round(cal_before["slope"], 3),
            "brier_before": round(cal_before["brier"], 4),
            "intercept_after": round(cal_after["intercept"], 3),
            "slope_after": round(cal_after["slope"], 3),
            "brier_after": round(cal_after["brier"], 4),
            "delta_nb_mean": round(float(np.mean(delta_nb_list)), 3),
        })

    return pd.DataFrame(records)


def bootstrap_calibration_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = 0.05,
) -> Dict:
    rng = np.random.RandomState(RANDOM_SEED)
    boot_slopes, boot_intercepts, boot_briers = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0 or y_true[idx].sum() == len(idx):
            continue
        try:
            cal = compute_calibration_intercept_slope(y_true[idx], y_prob[idx])
            if not np.isnan(cal["slope"]):
                boot_slopes.append(cal["slope"])
                boot_intercepts.append(cal["intercept"])
                boot_briers.append(cal["brier"])
        except Exception:
            continue

    def bca(values, alpha):
        lo = np.percentile(values, 100 * alpha / 2)
        hi = np.percentile(values, 100 * (1 - alpha / 2))
        return lo, hi

    return {
        "slope_ci": bca(boot_slopes, alpha),
        "intercept_ci": bca(boot_intercepts, alpha),
        "brier_ci": bca(boot_briers, alpha),
    }


def cox_recalibration_survival(
    baseline_survival: np.ndarray,
    lp_dev: np.ndarray,
    lp_ext: np.ndarray,
    time_ext: np.ndarray,
    event_ext: np.ndarray,
    eval_time: float = 3.0,
) -> np.ndarray:
    mean_lp_dev = lp_dev.mean()
    mean_lp_ext = lp_ext.mean()
    lp_ext_centered = lp_ext - mean_lp_ext + mean_lp_dev

    recal_survival = baseline_survival ** np.exp(lp_ext_centered - mean_lp_dev)
    recal_risk = 1 - recal_survival
    return np.clip(recal_risk, 0, 1)

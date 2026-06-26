import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, brier_score_loss,
)
from scipy import stats
from lifelines.utils import concordance_index
from typing import Optional, Tuple, List, Dict
import sys
sys.path.append('..')
from config import N_BOOTSTRAP, RANDOM_SEED, MIN_SENSITIVITY


def compute_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    auc = roc_auc_score(y_true, y_prob)
    rng = np.random.RandomState(RANDOM_SEED)
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0 or y_true[idx].sum() == len(idx):
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    ci_low = np.percentile(boot_aucs, 100 * alpha / 2)
    ci_high = np.percentile(boot_aucs, 100 * (1 - alpha / 2))
    return auc, ci_low, ci_high


def compute_cindex_ci(
    time: np.ndarray,
    event: np.ndarray,
    risk_score: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    ci = concordance_index(time, -risk_score, event)
    rng = np.random.RandomState(RANDOM_SEED)
    boot_cis = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(time), len(time))
        if event[idx].sum() == 0:
            continue
        try:
            boot_cis.append(concordance_index(time[idx], -risk_score[idx], event[idx]))
        except Exception:
            continue
    ci_low = np.percentile(boot_cis, 100 * alpha / 2)
    ci_high = np.percentile(boot_cis, 100 * (1 - alpha / 2))
    return ci, ci_low, ci_high


def delong_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
) -> Tuple[float, float]:
    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=float)
        T2[J] = T + 1
        return T2

    def fastDeLong(predictions_sorted_transposed, label_1_count):
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=float)
        ty = np.empty([k, n], dtype=float)
        tz = np.empty([k, m + n], dtype=float)
        for r in range(k):
            tx[r, :] = compute_midrank(positive_examples[r, :])
            ty[r, :] = compute_midrank(negative_examples[r, :])
            tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
        aucs = (tz[:, :m].sum(axis=1) - tx.sum(axis=1)) / (m * n)
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov

    order = (-y_true).argsort()
    label_1_count = y_true.sum()
    predictions_sorted_transposed = np.vstack([y_prob_a, y_prob_b])[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    auc_diff = aucs[0] - aucs[1]
    se = np.sqrt(delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1])
    z = auc_diff / (se + 1e-10)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc_diff, p_value


def bootstrap_cindex_diff(
    time: np.ndarray,
    event: np.ndarray,
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    ci_a = concordance_index(time, -risk_a, event)
    ci_b = concordance_index(time, -risk_b, event)
    diff = ci_a - ci_b

    rng = np.random.RandomState(RANDOM_SEED)
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(time), len(time))
        if event[idx].sum() == 0:
            continue
        try:
            d = (
                concordance_index(time[idx], -risk_a[idx], event[idx])
                - concordance_index(time[idx], -risk_b[idx], event[idx])
            )
            boot_diffs.append(d)
        except Exception:
            continue

    ci_low = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_high = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    p_value = 2 * min(
        (np.array(boot_diffs) > 0).mean(),
        (np.array(boot_diffs) < 0).mean(),
    )
    return diff, ci_low, ci_high, p_value


def select_threshold_youden(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_sensitivity: float = MIN_SENSITIVITY,
) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid = tpr >= min_sensitivity
    if valid.sum() == 0:
        return thresholds[np.argmax(tpr - fpr)]
    youden = tpr - fpr
    youden[~valid] = -np.inf
    return float(thresholds[np.argmax(youden)])


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    ppv = tp / (tp + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)
    fnr = fn / (fn + tp + 1e-8)
    fpr = fp / (fp + tn + 1e-8)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fnr": fnr,
        "fpr_rate": fpr,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict:
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )
    from scipy.stats import linregress
    slope, intercept, _, _, _ = linregress(mean_predicted_value, fraction_of_positives)
    brier = brier_score_loss(y_true, y_prob)

    return {
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "brier_score": brier,
        "fraction_of_positives": fraction_of_positives,
        "mean_predicted_value": mean_predicted_value,
    }


def recalibrate_predictions(
    y_prob_dev: np.ndarray,
    y_true_dev: np.ndarray,
    y_prob_ext: np.ndarray,
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    log_odds = np.log(y_prob_dev / (1 - y_prob_dev + 1e-8) + 1e-8).reshape(-1, 1)
    recal_model = LogisticRegression(penalty=None, solver="lbfgs")
    recal_model.fit(log_odds, y_true_dev)
    log_odds_ext = np.log(y_prob_ext / (1 - y_prob_ext + 1e-8) + 1e-8).reshape(-1, 1)
    return recal_model.predict_proba(log_odds_ext)[:, 1]


def compute_nri(
    y_true: np.ndarray,
    y_prob_new: np.ndarray,
    y_prob_ref: np.ndarray,
) -> Tuple[float, float, float]:
    events = y_true == 1
    non_events = y_true == 0

    up_events = ((y_prob_new > y_prob_ref) & events).sum()
    down_events = ((y_prob_new < y_prob_ref) & events).sum()
    up_non_events = ((y_prob_new > y_prob_ref) & non_events).sum()
    down_non_events = ((y_prob_new < y_prob_ref) & non_events).sum()

    n_events = events.sum()
    n_non_events = non_events.sum()

    nri_events = (up_events - down_events) / (n_events + 1e-8)
    nri_non_events = (down_non_events - up_non_events) / (n_non_events + 1e-8)
    nri = nri_events + nri_non_events

    return nri, nri_events, nri_non_events


def compute_idi(
    y_true: np.ndarray,
    y_prob_new: np.ndarray,
    y_prob_ref: np.ndarray,
) -> float:
    events = y_true == 1
    non_events = y_true == 0

    idi = (
        (y_prob_new[events].mean() - y_prob_ref[events].mean())
        - (y_prob_new[non_events].mean() - y_prob_ref[non_events].mean())
    )
    return float(idi)


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    group: np.ndarray,
    group_labels: List[str] = None,
) -> pd.DataFrame:
    y_pred = (y_prob >= threshold).astype(int)
    unique_groups = np.unique(group)
    records = []
    for g in unique_groups:
        mask = group == g
        y_t = y_true[mask]
        y_p = y_pred[mask]
        y_pb = y_prob[mask]
        tp = ((y_p == 1) & (y_t == 1)).sum()
        fn = ((y_p == 0) & (y_t == 1)).sum()
        fp = ((y_p == 1) & (y_t == 0)).sum()
        tn = ((y_p == 0) & (y_t == 0)).sum()
        fnr = fn / (fn + tp + 1e-8)
        fpr_rate = fp / (fp + tn + 1e-8)
        auc, _, _ = compute_auc_ci(y_t, y_pb, n_bootstrap=200)
        records.append({
            "group": g,
            "n": mask.sum(),
            "n_events": y_t.sum(),
            "auc": auc,
            "fnr": fnr,
            "fpr_rate": fpr_rate,
        })
    return pd.DataFrame(records)


def mcnemar_fnr_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Tuple[float, float]:
    events = y_true == 1
    fn_a = (y_pred_a[events] == 0)
    fn_b = (y_pred_b[events] == 0)
    b = (fn_a & ~fn_b).sum()
    c = (~fn_a & fn_b).sum()
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_val = 1 - stats.chi2.cdf(chi2, df=1)
    return float(chi2), float(p_val)


def group_aware_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group: np.ndarray,
    base_threshold: float,
    target_group: object,
    max_fpr_increase: float = 0.02,
    min_sensitivity: float = MIN_SENSITIVITY,
) -> Dict:
    thresholds = np.linspace(0.05, 0.95, 100)
    mask_target = group == target_group
    mask_other = ~mask_target

    best = {"threshold_target": base_threshold, "threshold_other": base_threshold}

    for t_target in thresholds:
        for t_other in thresholds:
            thresh_arr = np.where(mask_target, t_target, t_other)
            y_pred = (y_prob >= thresh_arr).astype(int)

            fnr_target = compute_classification_metrics(
                y_true[mask_target], y_prob[mask_target], t_target
            )["fnr"]
            fpr_target = compute_classification_metrics(
                y_true[mask_target], y_prob[mask_target], t_target
            )["fpr_rate"]
            sensitivity_target = 1 - fnr_target

            if sensitivity_target >= min_sensitivity:
                best["threshold_target"] = t_target
                best["threshold_other"] = t_other
                break

    return best

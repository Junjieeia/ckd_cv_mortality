import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
from typing import Dict, List, Callable
import sys
sys.path.append('..')
from config import N_BOOTSTRAP, RANDOM_SEED


SUBGROUP_DEFINITIONS = {
    "sex": {
        "Female": lambda df: df["sex_female"] == 1,
        "Male": lambda df: df["sex_female"] == 0,
    },
    "age": {
        "Age < 65": lambda df: df["age"] < 65,
        "Age >= 65": lambda df: df["age"] >= 65,
    },
    "diabetes": {
        "Diabetes": lambda df: df["diabetes_mellitus"] == 1,
        "No diabetes": lambda df: df["diabetes_mellitus"] == 0,
    },
    "ckd_stage": {
        "CKD stage 3": lambda df: df["ckd_stage"] == 3,
        "CKD stage 4-5": lambda df: df["ckd_stage"].isin([4, 5]),
    },
    "dialysis": {
        "On dialysis": lambda df: df["dialysis_maintenance"] == 1,
        "Not on dialysis": lambda df: df["dialysis_maintenance"] == 0,
    },
    "site": {
        "Site A": lambda df: df["site"] == "A",
        "Site B": lambda df: df["site"] == "B",
    },
}


def auc_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = 0.05,
) -> tuple:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan, np.nan, np.nan
    auc = roc_auc_score(y_true, y_prob)
    rng = np.random.RandomState(RANDOM_SEED)
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0 or y_true[idx].sum() == len(idx):
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    ci_lo = np.percentile(boot_aucs, 100 * alpha / 2)
    ci_hi = np.percentile(boot_aucs, 100 * (1 - alpha / 2))
    return auc, ci_lo, ci_hi


def cindex_with_ci(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = 0.05,
) -> tuple:
    if event.sum() == 0:
        return np.nan, np.nan, np.nan
    ci = concordance_index(time, -risk, event)
    rng = np.random.RandomState(RANDOM_SEED)
    boot_cis = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(time), len(time))
        if event[idx].sum() == 0:
            continue
        try:
            boot_cis.append(concordance_index(time[idx], -risk[idx], event[idx]))
        except Exception:
            continue
    ci_lo = np.percentile(boot_cis, 100 * alpha / 2)
    ci_hi = np.percentile(boot_cis, 100 * (1 - alpha / 2))
    return ci, ci_lo, ci_hi


def run_subgroup_analysis_task2(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_bootstrap: int = 200,
) -> pd.DataFrame:
    records = []
    for group_name, subgroups in SUBGROUP_DEFINITIONS.items():
        for subgroup_label, mask_fn in subgroups.items():
            try:
                mask = mask_fn(df).values
            except Exception:
                continue
            if mask.sum() < 20:
                continue
            y_t = y_true[mask]
            y_p = y_prob[mask]
            auc, ci_lo, ci_hi = auc_with_ci(y_t, y_p, n_bootstrap)
            records.append({
                "group_variable": group_name,
                "subgroup": subgroup_label,
                "n": int(mask.sum()),
                "n_events": int(y_t.sum()),
                "auc": round(auc, 3) if not np.isnan(auc) else np.nan,
                "ci_low": round(ci_lo, 3) if not np.isnan(ci_lo) else np.nan,
                "ci_high": round(ci_hi, 3) if not np.isnan(ci_hi) else np.nan,
            })
    return pd.DataFrame(records)


def run_subgroup_analysis_task1(
    df: pd.DataFrame,
    risk_scores: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    n_bootstrap: int = 200,
) -> pd.DataFrame:
    records = []
    for group_name, subgroups in SUBGROUP_DEFINITIONS.items():
        for subgroup_label, mask_fn in subgroups.items():
            try:
                mask = mask_fn(df).values
            except Exception:
                continue
            if mask.sum() < 20:
                continue
            t_sub = time[mask]
            e_sub = event[mask]
            r_sub = risk_scores[mask]
            ci, ci_lo, ci_hi = cindex_with_ci(t_sub, e_sub, r_sub, n_bootstrap)
            records.append({
                "group_variable": group_name,
                "subgroup": subgroup_label,
                "n": int(mask.sum()),
                "n_events": int(e_sub.sum()),
                "c_index": round(ci, 3) if not np.isnan(ci) else np.nan,
                "ci_low": round(ci_lo, 3) if not np.isnan(ci_lo) else np.nan,
                "ci_high": round(ci_hi, 3) if not np.isnan(ci_hi) else np.nan,
            })
    return pd.DataFrame(records)


def run_both_tasks_subgroup(
    df: pd.DataFrame,
    y_prob_t2: np.ndarray,
    y_true_t2: np.ndarray,
    risk_t1: np.ndarray,
    time_t1: np.ndarray,
    event_t1: np.ndarray,
    n_bootstrap: int = 200,
) -> pd.DataFrame:
    t2_df = run_subgroup_analysis_task2(df, y_prob_t2, y_true_t2, n_bootstrap)
    t1_df = run_subgroup_analysis_task1(df, risk_t1, time_t1, event_t1, n_bootstrap)

    t2_df = t2_df.rename(columns={"auc": "task2_auc", "ci_low": "task2_ci_low", "ci_high": "task2_ci_high", "n_events": "task2_n_events"})
    t1_df = t1_df.rename(columns={"c_index": "task1_c_index", "ci_low": "task1_ci_low", "ci_high": "task1_ci_high", "n_events": "task1_n_events"})

    merged = pd.merge(
        t2_df[["group_variable", "subgroup", "n", "task2_n_events", "task2_auc", "task2_ci_low", "task2_ci_high"]],
        t1_df[["group_variable", "subgroup", "task1_n_events", "task1_c_index", "task1_ci_low", "task1_ci_high"]],
        on=["group_variable", "subgroup"],
        how="outer",
    )
    return merged


def print_subgroup_table(df: pd.DataFrame, task: str = "task2") -> None:
    print(f"\n{'=' * 70}")
    print(f"Subgroup Performance – {task.upper()}")
    print(f"{'=' * 70}")
    for group_var in df["group_variable"].unique():
        sub = df[df["group_variable"] == group_var]
        print(f"\n  {group_var.upper()}")
        for _, row in sub.iterrows():
            if task == "task2" and "task2_auc" in row:
                print(f"    {row['subgroup']:<25} n={row['n']:<6} AUC={row['task2_auc']:.3f} ({row['task2_ci_low']:.3f}–{row['task2_ci_high']:.3f})")
            elif task == "task1" and "task1_c_index" in row:
                print(f"    {row['subgroup']:<25} n={row['n']:<6} C={row['task1_c_index']:.3f} ({row['task1_ci_low']:.3f}–{row['task1_ci_high']:.3f})")

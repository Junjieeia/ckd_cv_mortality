import numpy as np
import pandas as pd
import torch
import argparse
import os
import sys
sys.path.append('..')

from config import (
    RANDOM_SEED, N_BOOTSTRAP, RESULTS_DIR, FIGURES_DIR,
    SQI_HIGH_THRESHOLD, SQI_LOW_THRESHOLD,
)
from evaluation.metrics import (
    compute_auc_ci, delong_test, compute_nri, compute_idi,
    compute_cindex_ci, bootstrap_cindex_diff,
)
from utils.sensitivity_analysis import signal_quality_stratified_ablation
from figures.figure_generator import plot_roc_curves

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def run_task2_ablation(
    y_true: np.ndarray,
    y_prob_tabular: np.ndarray,
    y_prob_ecg_only: np.ndarray,
    y_prob_multimodal: np.ndarray,
    sqi_values: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    records = []

    auc_mm, ci_mm_lo, ci_mm_hi = compute_auc_ci(y_true, y_prob_multimodal, n_bootstrap)
    auc_tab, ci_tab_lo, ci_tab_hi = compute_auc_ci(y_true, y_prob_tabular, n_bootstrap)
    auc_ecg, ci_ecg_lo, ci_ecg_hi = compute_auc_ci(y_true, y_prob_ecg_only, n_bootstrap)

    delta_mm_tab, p_mm_tab = delong_test(y_true, y_prob_multimodal, y_prob_tabular)
    delta_mm_ecg, p_mm_ecg = delong_test(y_true, y_prob_multimodal, y_prob_ecg_only)

    rng = np.random.RandomState(RANDOM_SEED)
    boot_delta_mm_tab = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0 or y_true[idx].sum() == len(idx):
            continue
        d, _ = delong_test(y_true[idx], y_prob_multimodal[idx], y_prob_tabular[idx])
        boot_delta_mm_tab.append(d)
    ci_delta_lo = np.percentile(boot_delta_mm_tab, 2.5)
    ci_delta_hi = np.percentile(boot_delta_mm_tab, 97.5)

    nri, nri_events, nri_non_events = compute_nri(y_true, y_prob_multimodal, y_prob_tabular)
    idi = compute_idi(y_true, y_prob_multimodal, y_prob_tabular)

    rng2 = np.random.RandomState(RANDOM_SEED + 1)
    boot_nri, boot_idi = [], []
    for _ in range(n_bootstrap):
        idx = rng2.randint(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0:
            continue
        n, _, _ = compute_nri(y_true[idx], y_prob_multimodal[idx], y_prob_tabular[idx])
        i = compute_idi(y_true[idx], y_prob_multimodal[idx], y_prob_tabular[idx])
        boot_nri.append(n)
        boot_idi.append(i)
    nri_ci_lo, nri_ci_hi = np.percentile(boot_nri, [2.5, 97.5])
    idi_ci_lo, idi_ci_hi = np.percentile(boot_idi, [2.5, 97.5])

    records.append({
        "comparison": "Multimodal vs Tabular-only (Task 2 AUC)",
        "model_a": "multimodal",
        "auc_a": round(auc_mm, 3),
        "ci_a": f"{ci_mm_lo:.3f}–{ci_mm_hi:.3f}",
        "model_b": "tabular_only",
        "auc_b": round(auc_tab, 3),
        "ci_b": f"{ci_tab_lo:.3f}–{ci_tab_hi:.3f}",
        "delta_auc": round(delta_mm_tab, 3),
        "delta_ci": f"{ci_delta_lo:.3f}–{ci_delta_hi:.3f}",
        "delong_p": round(p_mm_tab, 3),
        "nri": round(nri, 3),
        "nri_ci": f"{nri_ci_lo:.3f}–{nri_ci_hi:.3f}",
        "idi": round(idi, 3),
        "idi_ci": f"{idi_ci_lo:.3f}–{idi_ci_hi:.3f}",
    })

    records.append({
        "comparison": "Multimodal vs ECG-only (Task 2 AUC)",
        "model_a": "multimodal",
        "auc_a": round(auc_mm, 3),
        "ci_a": f"{ci_mm_lo:.3f}–{ci_mm_hi:.3f}",
        "model_b": "ecg_only",
        "auc_b": round(auc_ecg, 3),
        "ci_b": f"{ci_ecg_lo:.3f}–{ci_ecg_hi:.3f}",
        "delta_auc": round(delta_mm_ecg, 3),
        "delta_ci": "n/a",
        "delong_p": round(p_mm_ecg, 3),
        "nri": "n/a",
        "nri_ci": "n/a",
        "idi": "n/a",
        "idi_ci": "n/a",
    })

    print("\n[Ablation – Task 2] Results:")
    for r in records:
        print(f"  {r['comparison']}")
        print(f"    Model A AUC: {r['auc_a']} {r['ci_a']}")
        print(f"    Model B AUC: {r['auc_b']} {r['ci_b']}")
        print(f"    ΔAUC: {r['delta_auc']} ({r['delta_ci']}), p={r['delong_p']}")
        if r["nri"] != "n/a":
            print(f"    NRI: {r['nri']} ({r['nri_ci']}), IDI: {r['idi']} ({r['idi_ci']})")

    print("\n[Ablation – Task 2] Signal-quality-stratified ablation:")
    sqi_df = signal_quality_stratified_ablation(
        y_true, y_prob_tabular, y_prob_multimodal, sqi_values, n_bootstrap=n_bootstrap
    )
    print(sqi_df.to_string(index=False))

    ablation_df = pd.DataFrame(records)
    ablation_path = os.path.join(RESULTS_DIR, "task2_ablation.csv")
    ablation_df.to_csv(ablation_path, index=False)
    sqi_path = os.path.join(RESULTS_DIR, "task2_sqi_ablation.csv")
    sqi_df.to_csv(sqi_path, index=False)

    return ablation_df, sqi_df


def run_task1_ablation(
    time_ext: np.ndarray,
    event_ext: np.ndarray,
    risk_tabular: np.ndarray,
    risk_multimodal: np.ndarray,
    risk_ckdpc: np.ndarray,
    risk_kfre8: np.ndarray,
    risk_kfre4: np.ndarray,
    risk_score2_pce: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    records = []

    ci_mm, ci_mm_lo, ci_mm_hi = compute_cindex_ci(time_ext, event_ext, risk_multimodal, n_bootstrap)
    ci_tab, ci_tab_lo, ci_tab_hi = compute_cindex_ci(time_ext, event_ext, risk_tabular, n_bootstrap)
    ci_ckdpc, ci_ckdpc_lo, ci_ckdpc_hi = compute_cindex_ci(time_ext, event_ext, risk_ckdpc, n_bootstrap)
    ci_kfre8, ci_kfre8_lo, ci_kfre8_hi = compute_cindex_ci(time_ext, event_ext, risk_kfre8, n_bootstrap)
    ci_kfre4, ci_kfre4_lo, ci_kfre4_hi = compute_cindex_ci(time_ext, event_ext, risk_kfre4, n_bootstrap)
    ci_s2, ci_s2_lo, ci_s2_hi = compute_cindex_ci(time_ext, event_ext, risk_score2_pce, n_bootstrap)

    comparisons = [
        ("Primary (DeepHit) vs CKD-PC", risk_tabular, risk_ckdpc, ci_tab, ci_ckdpc, False),
        ("Primary (DeepHit) vs KFRE-8var", risk_tabular, risk_kfre8, ci_tab, ci_kfre8, True),
        ("Primary (DeepHit) vs KFRE-4var", risk_tabular, risk_kfre4, ci_tab, ci_kfre4, False),
        ("Primary (DeepHit) vs SCORE2/PCE", risk_tabular, risk_score2_pce, ci_tab, ci_s2, True),
        ("Multimodal vs CKD-PC", risk_multimodal, risk_ckdpc, ci_mm, ci_ckdpc, False),
        ("Multimodal vs Primary (DeepHit)", risk_multimodal, risk_tabular, ci_mm, ci_tab, False),
    ]

    for label, risk_a, risk_b, ci_a, ci_b, compute_reclassification in comparisons:
        diff, diff_lo, diff_hi, p_val = bootstrap_cindex_diff(
            time_ext, event_ext, risk_a, risk_b, n_bootstrap
        )

        nri_str, idi_str = "n/a", "n/a"
        if compute_reclassification:
            from evaluation.metrics import compute_nri, compute_idi
            event_binary = (event_ext == 1).astype(int)
            nri, _, _ = compute_nri(event_binary, risk_a, risk_b)
            idi = compute_idi(event_binary, risk_a, risk_b)

            rng = np.random.RandomState(RANDOM_SEED)
            boot_nri, boot_idi = [], []
            for _ in range(n_bootstrap):
                idx = rng.randint(0, len(event_binary), len(event_binary))
                if event_binary[idx].sum() == 0:
                    continue
                n_, _, _ = compute_nri(event_binary[idx], risk_a[idx], risk_b[idx])
                i_ = compute_idi(event_binary[idx], risk_a[idx], risk_b[idx])
                boot_nri.append(n_)
                boot_idi.append(i_)
            nri_ci_lo, nri_ci_hi = np.percentile(boot_nri, [2.5, 97.5])
            idi_ci_lo, idi_ci_hi = np.percentile(boot_idi, [2.5, 97.5])
            nri_str = f"{nri:.3f} ({nri_ci_lo:.3f}–{nri_ci_hi:.3f})"
            idi_str = f"{idi:.3f} ({idi_ci_lo:.3f}–{idi_ci_hi:.3f})"

        records.append({
            "comparison": label,
            "c_index_a": round(ci_a, 3),
            "c_index_b": round(ci_b, 3),
            "delta": round(diff, 3),
            "delta_ci": f"{diff_lo:.3f}–{diff_hi:.3f}",
            "p_value": round(p_val, 3),
            "nri": nri_str,
            "idi": idi_str,
        })

    print("\n[Ablation – Task 1] Results:")
    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    out_path = os.path.join(RESULTS_DIR, "task1_ablation.csv")
    df.to_csv(out_path, index=False)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["task1", "task2", "both"], default="both")
    parser.add_argument("--prob_tabular", default=None, help="Path to .npy file with tabular model probabilities")
    parser.add_argument("--prob_multimodal", default=None, help="Path to .npy file with multimodal probabilities")
    parser.add_argument("--prob_ecg", default=None, help="Path to .npy file with ECG-only probabilities (Task 2)")
    parser.add_argument("--y_true", default=None, help="Path to .npy file with true labels (Task 2)")
    parser.add_argument("--sqi", default=None, help="Path to .npy file with SQI values")
    parser.add_argument("--time_ext", default=None, help="Path to .npy with survival times (Task 1)")
    parser.add_argument("--event_ext", default=None, help="Path to .npy with event indicators (Task 1)")
    parser.add_argument("--risk_ckdpc", default=None, help="Path to .npy with CKD-PC risk scores")
    parser.add_argument("--risk_kfre8", default=None, help="Path to .npy with KFRE 8-var risk scores")
    parser.add_argument("--risk_kfre4", default=None, help="Path to .npy with KFRE 4-var risk scores")
    parser.add_argument("--risk_score2", default=None, help="Path to .npy with SCORE2/PCE risk scores")
    args = parser.parse_args()

    if args.task in ("task2", "both"):
        if all(v is not None for v in [args.y_true, args.prob_tabular, args.prob_ecg, args.prob_multimodal, args.sqi]):
            y_true = np.load(args.y_true)
            y_prob_tab = np.load(args.prob_tabular)
            y_prob_ecg = np.load(args.prob_ecg)
            y_prob_mm = np.load(args.prob_multimodal)
            sqi = np.load(args.sqi)
            run_task2_ablation(y_true, y_prob_tab, y_prob_ecg, y_prob_mm, sqi)
        else:
            print("[Task 2 ablation] Input arrays not provided — skipping.")

    if args.task in ("task1", "both"):
        if all(v is not None for v in [args.time_ext, args.event_ext, args.prob_tabular,
                                        args.prob_multimodal, args.risk_ckdpc,
                                        args.risk_kfre8, args.risk_kfre4, args.risk_score2]):
            time_ext = np.load(args.time_ext)
            event_ext = np.load(args.event_ext)
            risk_tab = np.load(args.prob_tabular)
            risk_mm = np.load(args.prob_multimodal)
            risk_ckdpc = np.load(args.risk_ckdpc)
            risk_kfre8 = np.load(args.risk_kfre8)
            risk_kfre4 = np.load(args.risk_kfre4)
            risk_s2 = np.load(args.risk_score2)
            run_task1_ablation(time_ext, event_ext, risk_tab, risk_mm, risk_ckdpc, risk_kfre8, risk_kfre4, risk_s2)
        else:
            print("[Task 1 ablation] Input arrays not provided — skipping.")


if __name__ == "__main__":
    main()

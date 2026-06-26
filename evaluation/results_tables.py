import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
import sys
sys.path.append('..')
from config import RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)


def format_ci(value: float, lo: float, hi: float, decimals: int = 3) -> str:
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(value)} ({fmt.format(lo)}–{fmt.format(hi)})"


def format_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def make_table1_baseline(df_dev: pd.DataFrame, df_ext: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import ttest_ind, chi2_contingency

    rows = []
    continuous_vars = [
        ("Age, years", "age"),
        ("eGFR, mL/min/1.73 m²", "egfr_baseline"),
        ("Hemoglobin, g/dL", "hemoglobin"),
        ("Serum albumin, g/dL", "serum_albumin"),
        ("Serum phosphate, mg/dL", "serum_phosphate"),
        ("Platelet count, ×10⁹/L", "platelet_count"),
        ("Serum creatinine, mg/dL", "serum_creatinine"),
        ("Systolic BP, mmHg", "systolic_bp"),
        ("Admission glucose, mg/dL", "admission_glucose"),
        ("LDL cholesterol, mg/dL", "ldl_cholesterol"),
        ("eGFR slope, mL/min/yr", "egfr_slope_12m"),
        ("Hemoglobin slope, g/dL/yr", "hemoglobin_slope"),
    ]
    categorical_vars = [
        ("Sex, female, n (%)", "sex_female"),
        ("Diabetes mellitus, n (%)", "diabetes_mellitus"),
        ("Hypertension, n (%)", "hypertension"),
        ("Dialysis (maintenance), n (%)", "dialysis_maintenance"),
        ("Cardiovascular disease history, n (%)", "cvd_history"),
    ]
    skewed_vars = {"admission_glucose", "proteinuria_uacr"}

    for label, col in continuous_vars:
        if col not in df_dev.columns:
            continue
        dev_vals = df_dev[col].dropna().values
        ext_vals = df_ext[col].dropna().values
        missing_pct = (df_dev[col].isna().mean() + df_ext[col].isna().mean()) / 2 * 100

        if col in skewed_vars:
            dev_str = f"{np.median(dev_vals):.0f} ({np.percentile(dev_vals,25):.0f}–{np.percentile(dev_vals,75):.0f})"
            ext_str = f"{np.median(ext_vals):.0f} ({np.percentile(ext_vals,25):.0f}–{np.percentile(ext_vals,75):.0f})"
        else:
            dev_str = f"{dev_vals.mean():.1f} ({dev_vals.std():.1f})"
            ext_str = f"{ext_vals.mean():.1f} ({ext_vals.std():.1f})"

        pooled_std = np.sqrt((dev_vals.std() ** 2 + ext_vals.std() ** 2) / 2 + 1e-10)
        smd = abs(dev_vals.mean() - ext_vals.mean()) / (pooled_std + 1e-10)

        rows.append({
            "Variable": label,
            f"Development (n={len(df_dev)})": dev_str,
            f"External (n={len(df_ext)})": ext_str,
            "Missing, %": f"{missing_pct:.1f}",
            "SMD": f"{smd:.2f}",
        })

    for label, col in categorical_vars:
        if col not in df_dev.columns:
            continue
        dev_n = df_dev[col].sum()
        ext_n = df_ext[col].sum()
        dev_pct = dev_n / len(df_dev) * 100
        ext_pct = ext_n / len(df_ext) * 100

        p_dev = dev_n / len(df_dev)
        p_ext = ext_n / len(df_ext)
        pooled_p = (dev_n + ext_n) / (len(df_dev) + len(df_ext))
        smd = abs(p_dev - p_ext) / np.sqrt(pooled_p * (1 - pooled_p) + 1e-10)

        rows.append({
            "Variable": label,
            f"Development (n={len(df_dev)})": f"{int(dev_n)} ({dev_pct:.1f}%)",
            f"External (n={len(df_ext)})": f"{int(ext_n)} ({ext_pct:.1f}%)",
            "Missing, %": "0.0",
            "SMD": f"{smd:.2f}",
        })

    return pd.DataFrame(rows)


def make_table2_class_distribution(
    n_dev: int, n_ext: int,
    n_cv_dev: int, n_cv_ext: int,
    n_mort_dev: int, n_mort_ext: int,
) -> pd.DataFrame:
    rows = [
        {
            "Task": "Task 1 – Cardiovascular outcome",
            "Partition": "Development",
            "Total, n": n_dev,
            "Event, n (%)": f"{n_cv_dev} ({n_cv_dev/n_dev*100:.1f}%)",
            "Non-event, n (%)": f"{n_dev-n_cv_dev} ({(n_dev-n_cv_dev)/n_dev*100:.1f}%)",
        },
        {
            "Task": "Task 1 – Cardiovascular outcome",
            "Partition": "External test",
            "Total, n": n_ext,
            "Event, n (%)": f"{n_cv_ext} ({n_cv_ext/n_ext*100:.1f}%)",
            "Non-event, n (%)": f"{n_ext-n_cv_ext} ({(n_ext-n_cv_ext)/n_ext*100:.1f}%)",
        },
        {
            "Task": "Task 2 – In-hospital mortality",
            "Partition": "Development",
            "Total, n": n_dev,
            "Event, n (%)": f"{n_mort_dev} ({n_mort_dev/n_dev*100:.1f}%)",
            "Non-event, n (%)": f"{n_dev-n_mort_dev} ({(n_dev-n_mort_dev)/n_dev*100:.1f}%)",
        },
        {
            "Task": "Task 2 – In-hospital mortality",
            "Partition": "External test",
            "Total, n": n_ext,
            "Event, n (%)": f"{n_mort_ext} ({n_mort_ext/n_ext*100:.1f}%)",
            "Non-event, n (%)": f"{n_ext-n_mort_ext} ({(n_ext-n_mort_ext)/n_ext*100:.1f}%)",
        },
    ]
    return pd.DataFrame(rows)


def make_table4_task1_performance(results_dict: Dict) -> pd.DataFrame:
    rows = []
    model_order = ["multimodal", "tabular_only", "ecg_only", "cox", "rsf", "deepsurv", "deephit"]
    partition_order = ["Development", "Validation", "External test"]

    for model in model_order:
        if model not in results_dict:
            continue
        for partition in partition_order:
            key = f"{model}_{partition.lower().replace(' ', '_')}"
            if key not in results_dict[model]:
                continue
            d = results_dict[model][partition.lower().replace(" ", "_")]
            rows.append({
                "Model": model.replace("_", " ").title(),
                "Partition": partition,
                "C-index (95% CI)": format_ci(d.get("c_index", 0), d.get("c_index_lo", 0), d.get("c_index_hi", 0)),
                "Time-dep. AUC (95% CI)": format_ci(d.get("td_auc", 0), d.get("td_auc_lo", 0), d.get("td_auc_hi", 0)),
                "Calibration slope": f"{d.get('cal_slope', 0):.2f}",
            })
    return pd.DataFrame(rows)


def make_table5_task2_performance(results_dict: Dict) -> pd.DataFrame:
    rows = []
    model_order = ["multimodal", "tabular_only", "ecg_only", "logistic", "xgboost",
                   "lightgbm", "random_forest", "ft_transformer", "tabnet"]
    partition_order = ["Development", "Validation", "External"]

    for model in model_order:
        if model not in results_dict:
            continue
        for partition in partition_order:
            key = partition.lower()
            if key not in results_dict.get(model, {}):
                continue
            d = results_dict[model][key]
            rows.append({
                "Model": model.replace("_", " ").title(),
                "Partition": partition,
                "AUC (95% CI)": format_ci(d.get("auc", 0), d.get("auc_lo", 0), d.get("auc_hi", 0)),
                "Accuracy": f"{d.get('accuracy', 0):.3f}",
                "Sensitivity": f"{d.get('sensitivity', 0):.3f}",
                "Specificity": f"{d.get('specificity', 0):.3f}",
                "F1": f"{d.get('f1', 0):.3f}",
                "Cal. slope": f"{d.get('cal_slope', 0):.2f}",
            })
    return pd.DataFrame(rows)


def make_table8_shap(shap_importance_t1: pd.DataFrame, shap_importance_t2: pd.DataFrame) -> pd.DataFrame:
    top_n = 10
    t1 = shap_importance_t1.head(top_n).reset_index(drop=True)
    t2 = shap_importance_t2.head(top_n).reset_index(drop=True)

    rows = []
    for i in range(top_n):
        rows.append({
            "Rank": i + 1,
            "Task 1 – Cardiovascular outcome": t1.iloc[i]["feature"].replace("_", " ").title() if i < len(t1) else "",
            "Mean |SHAP| (Task 1)": f"{t1.iloc[i]['mean_abs_shap']:.3f}" if i < len(t1) else "",
            "Task 2 – In-hospital mortality": t2.iloc[i]["feature"].replace("_", " ").title() if i < len(t2) else "",
            "Mean |SHAP| (Task 2)": f"{t2.iloc[i]['mean_abs_shap']:.3f}" if i < len(t2) else "",
        })
    return pd.DataFrame(rows)


def make_table12_recalibration(recal_df: pd.DataFrame) -> pd.DataFrame:
    return recal_df[[
        "model", "intercept_before", "slope_before",
        "intercept_after", "slope_after", "delta_nb_mean"
    ]].rename(columns={
        "model": "Model",
        "intercept_before": "Int. (before)",
        "slope_before": "Slope (before)",
        "intercept_after": "Int. (after)",
        "slope_after": "Slope (after)",
        "delta_nb_mean": "ΔNet benefit",
    })


def df_to_latex(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    n_cols = len(df.columns)
    col_fmt = "l" + "r" * (n_cols - 1)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
    ]
    header = " & ".join(str(c) for c in df.columns) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    for _, row in df.iterrows():
        row_str = " & ".join(str(v) for v in row.values) + " \\\\"
        lines.append(row_str)
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def save_all_tables(tables: Dict[str, pd.DataFrame], latex: bool = True) -> List[str]:
    saved_paths = []
    for table_name, df in tables.items():
        csv_path = os.path.join(RESULTS_DIR, f"{table_name}.csv")
        df.to_csv(csv_path, index=False)
        saved_paths.append(csv_path)

        if latex:
            tex_path = os.path.join(RESULTS_DIR, f"{table_name}.tex")
            caption = table_name.replace("_", " ").title()
            label = f"tab:{table_name}"
            latex_str = df_to_latex(df, caption=caption, label=label)
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(latex_str)
            saved_paths.append(tex_path)

    return saved_paths


def generate_results_summary(
    task2_auc_results: Dict,
    task1_cindex_results: Dict,
    operating_threshold: float,
    recal_df: pd.DataFrame,
    impact_dict: Dict,
) -> pd.DataFrame:
    rows = []

    for model_name, (auc, lo, hi) in task2_auc_results.items():
        rows.append({
            "task": "Task 2 – In-hospital mortality",
            "model": model_name.replace("_", " ").title(),
            "metric": "AUC",
            "value": round(auc, 3),
            "ci_low": round(lo, 3),
            "ci_high": round(hi, 3),
            "formatted": format_ci(auc, lo, hi),
        })

    for model_name, (ci, lo, hi) in task1_cindex_results.items():
        rows.append({
            "task": "Task 1 – Cardiovascular outcome",
            "model": model_name.replace("_", " ").title(),
            "metric": "C-index",
            "value": round(ci, 3),
            "ci_low": round(lo, 3),
            "ci_high": round(hi, 3),
            "formatted": format_ci(ci, lo, hi),
        })

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(RESULTS_DIR, "results_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n[Results Summary]")
    for task in summary_df["task"].unique():
        print(f"\n  {task}")
        sub = summary_df[summary_df["task"] == task]
        for _, row in sub.iterrows():
            print(f"    {row['model']:<30} {row['metric']}: {row['formatted']}")

    return summary_df

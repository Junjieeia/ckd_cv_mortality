import os
import json
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

from config import (
    RANDOM_SEED, OUTPUT_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR,
    TABULAR_FEATURES_TASK1, TABULAR_FEATURES_TASK2,
    NOMOGRAM_FEATURES_TASK1, NOMOGRAM_FEATURES_TASK2,
    TASK1_THRESHOLDS, TASK2_THRESHOLDS, N_BOOTSTRAP,
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, RANDOM_FOREST_PARAMS, LOGISTIC_PARAMS,
    DEEPHIT_PARAMS, DEEPSURV_PARAMS, RSF_PARAMS, FINEGRAY_PARAMS,
    AGE_THRESHOLD,
)
from preprocessing.data_loader import (
    load_tabular_data, apply_eligibility_criteria, preprocess_tabular, add_age_group
)
from preprocessing.feature_selection import combined_feature_selection, get_lasso_coefficient_path
from models.classical_models import (
    LogisticRegressionModel, XGBoostModel, LightGBMModel,
    RandomForestModel, CoxCompetingRisksModel, RandomSurvivalForestModel,
    BenchmarkEquations,
)
from models.deep_models import DeepSurvNet, DeepHitNet
from models.trainer import (
    train_binary_classifier, train_deepsurv, train_deephit, compute_class_weight
)
from evaluation.metrics import (
    compute_auc_ci, compute_cindex_ci, delong_test, bootstrap_cindex_diff,
    select_threshold_youden, compute_classification_metrics,
    compute_calibration, recalibrate_predictions,
    compute_nri, compute_idi, compute_fairness_metrics, mcnemar_fnr_test,
    group_aware_threshold,
)
from evaluation.decision_curve import (
    decision_curve_analysis, compute_net_benefit_at_threshold, simulated_clinical_impact
)
from interpretability.explainer import (
    compute_shap_values, compute_shap_summary, compute_tsne_representation
)
from nomogram.nomogram_builder import build_task2_nomogram, build_task1_nomogram
from figures.figure_generator import (
    plot_roc_curves, plot_calibration, plot_confusion_matrix,
    plot_decision_curve, plot_shap_summary, plot_loss_curves,
    plot_lasso_coefficient_path, plot_tsne, plot_fairness_fnr,
    plot_subgroup_performance, plot_nomogram, plot_time_dependent_auc,
)
from utils.cross_validation import cross_validate_classifier
from utils.sensitivity_analysis import (
    signal_quality_stratified_ablation, run_sensitivity_complete_case,
    run_sensitivity_missing_indicator, compare_sensitivity_results,
)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.use_deterministic_algorithms(True, warn_only=True)

for d in [OUTPUT_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


def run_task2_pipeline(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_ext: np.ndarray,
    y_ext: np.ndarray,
    feature_names: list,
    df_dev_meta: pd.DataFrame,
    df_ext_meta: pd.DataFrame,
) -> dict:
    print("[Task 2] Feature selection...")
    selected_features, fs_report = combined_feature_selection(
        X_dev, y_dev, feature_names, task="classification"
    )
    selected_idx = [feature_names.index(f) for f in selected_features]
    X_dev_sel = X_dev[:, selected_idx]
    X_ext_sel = X_ext[:, selected_idx]

    lasso_path_df = get_lasso_coefficient_path(X_dev, y_dev, feature_names)

    print("[Task 2] Training classical models...")
    lr_model = LogisticRegressionModel()
    lr_model.fit(X_dev_sel, y_dev)

    xgb_model = XGBoostModel()
    xgb_model.fit(X_dev_sel, y_dev, eval_set=[(X_dev_sel, y_dev)])

    lgbm_model = LightGBMModel()
    lgbm_model.fit(X_dev_sel, y_dev)

    rf_model = RandomForestModel()
    rf_model.fit(X_dev_sel, y_dev)

    print("[Task 2] Cross-validation...")
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    cv_lr = cross_validate_classifier(LogisticRegression, LOGISTIC_PARAMS, X_dev_sel, y_dev)
    cv_xgb = cross_validate_classifier(XGBClassifier, XGBOOST_PARAMS, X_dev_sel, y_dev)

    print("[Task 2] Selecting operating threshold on development data...")
    y_prob_dev_xgb = xgb_model.predict_proba(X_dev_sel)
    operating_threshold = select_threshold_youden(y_dev, y_prob_dev_xgb)
    print(f"  Operating threshold (XGBoost): {operating_threshold:.3f}")

    print("[Task 2] External evaluation...")
    y_prob_ext = {
        "logistic": lr_model.predict_proba(X_ext_sel),
        "xgboost": xgb_model.predict_proba(X_ext_sel),
        "lightgbm": lgbm_model.predict_proba(X_ext_sel),
        "random_forest": rf_model.predict_proba(X_ext_sel),
    }

    auc_results = {}
    for model_name, y_prob in y_prob_ext.items():
        auc, ci_low, ci_high = compute_auc_ci(y_ext, y_prob)
        auc_results[model_name] = (auc, ci_low, ci_high)
        print(f"  {model_name}: AUC={auc:.3f} ({ci_low:.3f}–{ci_high:.3f})")

    print("[Task 2] Calibration analysis...")
    calibration_results = {}
    for model_name, y_prob in y_prob_ext.items():
        cal = compute_calibration(y_ext, y_prob)
        calibration_results[model_name] = cal

    print("[Task 2] Recalibration...")
    y_prob_recal = {}
    y_prob_dev_lr = lr_model.predict_proba(X_dev_sel)
    y_prob_dev_xgb_full = xgb_model.predict_proba(X_dev_sel)
    for model_name, y_prob_d, y_prob_e in [
        ("logistic", y_prob_dev_lr, y_prob_ext["logistic"]),
        ("xgboost", y_prob_dev_xgb_full, y_prob_ext["xgboost"]),
    ]:
        y_prob_recal[model_name] = recalibrate_predictions(y_prob_d, y_dev, y_prob_e)

    print("[Task 2] Building nomogram...")
    nom_idx = [feature_names.index(f) for f in NOMOGRAM_FEATURES_TASK2 if f in feature_names]
    X_dev_nom = X_dev[:, nom_idx]
    X_ext_nom = X_ext[:, nom_idx]
    nom_feat = [feature_names[i] for i in nom_idx]

    nomogram = build_task2_nomogram(X_dev_nom, y_dev, nom_feat)
    y_prob_nom_ext = nomogram.predict_probability(X_ext_nom, nom_feat)
    auc_nom, ci_nom_low, ci_nom_high = compute_auc_ci(y_ext, y_prob_nom_ext)
    print(f"  Nomogram (logistic): AUC={auc_nom:.3f} ({ci_nom_low:.3f}–{ci_nom_high:.3f})")

    print("[Task 2] SHAP interpretability...")
    shap_values, shap_importance = compute_shap_values(
        xgb_model.model, X_dev_sel, X_ext_sel, selected_features, model_type="tree"
    )

    print("[Task 2] Decision curve analysis...")
    dca_models = {k: v for k, v in y_prob_ext.items()}
    dca_models["nomogram"] = y_prob_nom_ext
    dca_df = decision_curve_analysis(y_ext, dca_models, np.linspace(0.01, 0.5, 100))

    print("[Task 2] Fairness analysis...")
    sex_groups = df_ext_meta["sex"].values if "sex" in df_ext_meta.columns else np.zeros(len(y_ext))
    age_groups = (df_ext_meta["age"].values >= AGE_THRESHOLD).astype(int) if "age" in df_ext_meta.columns else np.zeros(len(y_ext))

    fairness_sex = compute_fairness_metrics(y_ext, y_prob_ext["xgboost"], operating_threshold, sex_groups)
    fairness_age = compute_fairness_metrics(y_ext, y_prob_ext["xgboost"], operating_threshold, age_groups)

    print("[Task 2] Simulated clinical impact...")
    impact = simulated_clinical_impact(
        y_ext,
        y_prob_ext["xgboost"],
        y_prob_nom_ext,
        operating_threshold,
        operating_threshold,
        n_bootstrap=N_BOOTSTRAP,
    )

    print("[Task 2] Generating figures...")
    plot_roc_curves(y_ext, y_prob_ext, auc_results, title="In-Hospital Mortality – ROC Curves (External)", filename="task2_roc_external.pdf")
    plot_calibration(y_ext, y_prob_ext, title="In-Hospital Mortality – Calibration (External)", filename="task2_calibration_external.pdf")
    y_pred_xgb = (y_prob_ext["xgboost"] >= operating_threshold).astype(int)
    plot_confusion_matrix(y_ext, y_pred_xgb, "XGBoost", "In-Hospital Mortality – Confusion Matrix", "task2_cm_xgboost.pdf")
    plot_decision_curve(dca_df, title="In-Hospital Mortality – Decision Curve (External)", filename="task2_dca_external.pdf")
    plot_shap_summary(shap_values, X_ext_sel, selected_features, title="In-Hospital Mortality – SHAP", filename="task2_shap.pdf")
    plot_fairness_fnr(fairness_age, title="False-Negative Rate by Age Group", filename="task2_fairness_age.pdf")
    nom_table = nomogram.generate_nomogram_table()
    plot_nomogram(nom_table, task="task2", title="In-Hospital Mortality Nomogram", filename="task2_nomogram.pdf")
    plot_lasso_coefficient_path(lasso_path_df, selected_features, 0.1, filename="task2_lasso_path.pdf")

    return {
        "auc_results": auc_results,
        "auc_nomogram": (auc_nom, ci_nom_low, ci_nom_high),
        "operating_threshold": operating_threshold,
        "calibration_results": calibration_results,
        "shap_importance": shap_importance,
        "fairness_sex": fairness_sex,
        "fairness_age": fairness_age,
        "impact": impact,
        "selected_features": selected_features,
        "nomogram": nomogram,
        "dca_df": dca_df,
        "y_prob_ext": y_prob_ext,
    }


def run_task1_pipeline(
    X_dev: np.ndarray,
    time_dev: np.ndarray,
    event_dev: np.ndarray,
    X_ext: np.ndarray,
    time_ext: np.ndarray,
    event_ext: np.ndarray,
    feature_names: list,
    df_dev_meta: pd.DataFrame,
    df_ext_meta: pd.DataFrame,
) -> dict:
    print("[Task 1] Feature selection...")
    selected_features, fs_report = combined_feature_selection(
        X_dev, event_dev, feature_names, task="regression"
    )
    selected_idx = [feature_names.index(f) for f in selected_features]
    X_dev_sel = X_dev[:, selected_idx]
    X_ext_sel = X_ext[:, selected_idx]

    print("[Task 1] Training survival models...")
    cox_model = CoxCompetingRisksModel()
    cox_model.fit(X_dev_sel, time_dev, event_dev, selected_features)

    rsf_model = RandomSurvivalForestModel()
    rsf_model.fit(X_dev_sel, time_dev, event_dev)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    deepsurv_net = DeepSurvNet(input_dim=len(selected_features), units=DEEPSURV_PARAMS["units"], dropout=DEEPSURV_PARAMS["dropout"])
    X_dev_val_split = int(0.8 * len(X_dev_sel))
    deepsurv_net, ds_train_loss, ds_val_loss = train_deepsurv(
        deepsurv_net,
        X_dev_sel[:X_dev_val_split], time_dev[:X_dev_val_split], event_dev[:X_dev_val_split],
        X_dev_sel[X_dev_val_split:], time_dev[X_dev_val_split:], event_dev[X_dev_val_split:],
    )

    n_time_bins = 100
    time_bins = np.quantile(time_dev[event_dev == 1], np.linspace(0, 1, n_time_bins))
    time_bin_idx_dev = np.digitize(time_dev, time_bins).clip(0, n_time_bins - 1)
    time_bin_idx_ext = np.digitize(time_ext, time_bins).clip(0, n_time_bins - 1)

    deephit_net = DeepHitNet(
        input_dim=len(selected_features), n_time_bins=n_time_bins, n_causes=2,
        units=DEEPHIT_PARAMS["units"], dropout=DEEPHIT_PARAMS["dropout"],
    )
    competing_dev = np.zeros_like(event_dev)
    competing_dev[event_dev == 0] = 1

    deephit_net, dh_train_loss, dh_val_loss = train_deephit(
        deephit_net,
        X_dev_sel[:X_dev_val_split], time_bin_idx_dev[:X_dev_val_split],
        event_dev[:X_dev_val_split], competing_dev[:X_dev_val_split],
        X_dev_sel[X_dev_val_split:], time_bin_idx_dev[X_dev_val_split:],
        event_dev[X_dev_val_split:], competing_dev[X_dev_val_split:],
    )

    print("[Task 1] External evaluation – C-index...")
    cox_risk_ext = cox_model.predict_partial_hazard(X_ext_sel)
    rsf_risk_ext = rsf_model.predict_risk(X_ext_sel) if hasattr(rsf_model, '_use_sksurv') and rsf_model._use_sksurv else cox_risk_ext

    deepsurv_net.eval()
    with torch.no_grad():
        ds_risk_ext = deepsurv_net(torch.FloatTensor(X_ext_sel)).numpy()

    deephit_net.eval()
    with torch.no_grad():
        dh_probs_ext = deephit_net(torch.FloatTensor(X_ext_sel)).numpy()
    dh_cif_ext = dh_probs_ext[:, 0, :].cumsum(axis=1)
    dh_risk_ext = dh_cif_ext[:, -1]

    cindex_results = {}
    for model_name, risk in [
        ("cox", cox_risk_ext),
        ("rsf", rsf_risk_ext),
        ("deepsurv", ds_risk_ext),
        ("deephit", dh_risk_ext),
    ]:
        ci, ci_low, ci_high = compute_cindex_ci(time_ext, event_ext, risk)
        cindex_results[model_name] = (ci, ci_low, ci_high)
        print(f"  {model_name}: C-index={ci:.3f} ({ci_low:.3f}–{ci_high:.3f})")

    print("[Task 1] Benchmark equations...")
    if all(c in df_ext_meta.columns for c in ["age","sex","egfr_baseline","proteinuria_uacr"]):
        kfre4_risk = BenchmarkEquations.kfre_4var(
            df_ext_meta["age"].values, df_ext_meta["sex_female"].values,
            df_ext_meta["egfr_baseline"].values, df_ext_meta["proteinuria_uacr"].values,
        )
        ci_kfre4, _, _ = compute_cindex_ci(time_ext, event_ext, kfre4_risk)
        print(f"  KFRE 4-var: C-index={ci_kfre4:.3f}")
        diff_dh_kfre4, ci_low_d, ci_high_d, p_d = bootstrap_cindex_diff(time_ext, event_ext, dh_risk_ext, kfre4_risk)
        print(f"  DeepHit vs KFRE4: Δ={diff_dh_kfre4:.3f} ({ci_low_d:.3f}–{ci_high_d:.3f}), p={p_d:.3f}")

    print("[Task 1] Building nomogram...")
    nom_idx = [feature_names.index(f) for f in NOMOGRAM_FEATURES_TASK1 if f in feature_names]
    X_dev_nom = X_dev[:, nom_idx]
    X_ext_nom = X_ext[:, nom_idx]
    nom_feat = [feature_names[i] for i in nom_idx]
    nomogram = build_task1_nomogram(X_dev_nom, time_dev, event_dev, nom_feat)
    y_prob_nom_ext = nomogram.predict_probability(X_ext_nom, nom_feat)
    ci_nom, ci_nom_low, ci_nom_high = compute_cindex_ci(time_ext, event_ext, y_prob_nom_ext)
    print(f"  Nomogram (Cox): C-index={ci_nom:.3f}")

    print("[Task 1] SHAP for DeepSurv...")
    ds_shap_values, ds_shap_importance = compute_shap_values(
        deepsurv_net, X_dev_sel, X_ext_sel, selected_features, model_type="deep"
    )

    print("[Task 1] Generating figures...")
    plot_loss_curves(ds_train_loss, ds_val_loss, "DeepSurv", filename="task1_loss_deepsurv.pdf")
    plot_loss_curves(dh_train_loss, dh_val_loss, "DeepHit", filename="task1_loss_deephit.pdf")
    nom_table = nomogram.generate_nomogram_table()
    plot_nomogram(nom_table, task="task1", title="Cardiovascular Outcome Nomogram", filename="task1_nomogram.pdf")
    plot_shap_summary(ds_shap_values, X_ext_sel, selected_features, title="CV Outcome – SHAP (DeepSurv)", filename="task1_shap.pdf")

    return {
        "cindex_results": cindex_results,
        "cindex_nomogram": (ci_nom, ci_nom_low, ci_nom_high),
        "deephit_risk_ext": dh_risk_ext,
        "deepsurv_risk_ext": ds_risk_ext,
        "cox_risk_ext": cox_risk_ext,
        "selected_features": selected_features,
        "nomogram": nomogram,
        "shap_importance": ds_shap_importance,
        "models": {
            "deephit": deephit_net,
            "deepsurv": deepsurv_net,
            "cox": cox_model,
            "rsf": rsf_model,
        },
    }


def save_results(results: dict, filename: str) -> None:
    path = os.path.join(RESULTS_DIR, filename)
    serializable = {}
    for k, v in results.items():
        if isinstance(v, (np.ndarray, pd.DataFrame, pd.Series)):
            serializable[k] = str(v.shape) if hasattr(v, "shape") else str(v)
        elif isinstance(v, dict):
            serializable[k] = {
                kk: (float(vv) if isinstance(vv, (np.floating, float)) else str(vv))
                for kk, vv in v.items()
            }
        elif isinstance(v, (float, int, str, bool)):
            serializable[k] = v
        elif isinstance(v, tuple):
            serializable[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
        else:
            serializable[k] = str(type(v))
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def main(
    dev_path: str,
    ext_path: str,
    ecg_dev_path: str,
    ecg_ext_path: str,
    ecg_checkpoint_path: str,
) -> None:
    print("=" * 60)
    print("CKD Cardiovascular Outcomes and In-Hospital Mortality")
    print("=" * 60)

    print("\n[Data] Loading tabular data...")
    dev_df, ext_df = load_tabular_data(dev_path, ext_path)

    print("[Data] Loading ECG arrays...")
    ecg_dev_full = np.load(ecg_dev_path)
    ecg_ext_full = np.load(ecg_ext_path)

    print("[Data] Applying eligibility criteria...")
    dev_df, dev_exclusion = apply_eligibility_criteria(dev_df)
    ext_df, ext_exclusion = apply_eligibility_criteria(ext_df)

    dev_keep_idx = dev_df.index.values
    ext_keep_idx = ext_df.index.values
    ecg_dev = ecg_dev_full[dev_keep_idx]
    ecg_ext = ecg_ext_full[ext_keep_idx]
    dev_df = dev_df.reset_index(drop=True)
    ext_df = ext_df.reset_index(drop=True)

    print(f"  Development: {len(dev_df)} records (excluded: {dev_exclusion.get('total_excluded', 0)})")
    print(f"  External: {len(ext_df)} records (excluded: {ext_exclusion.get('total_excluded', 0)})")
    print(f"  ECG arrays — dev: {ecg_dev.shape}, ext: {ecg_ext.shape}")

    dev_df = add_age_group(dev_df)
    ext_df = add_age_group(ext_df)

    print("\n[Task 2] In-Hospital Mortality Pipeline")
    print("-" * 40)
    X_dev_t2, y_dev_t2, X_ext_t2, y_ext_t2, scaler_t2, imputer_t2, feat_t2 = preprocess_tabular(dev_df, ext_df, "task2")
    results_t2 = run_task2_pipeline(X_dev_t2, y_dev_t2, X_ext_t2, y_ext_t2, feat_t2, dev_df, ext_df)
    save_results({"auc_results": {k: list(v) for k, v in results_t2["auc_results"].items()},
                  "operating_threshold": float(results_t2["operating_threshold"])}, "task2_results.json")

    print("\n[Task 1] Cardiovascular Outcome Pipeline")
    print("-" * 40)
    X_dev_t1, y_dev_t1_raw, X_ext_t1, y_ext_t1_raw, scaler_t1, imputer_t1, feat_t1 = preprocess_tabular(dev_df, ext_df, "task1")
    time_dev = y_dev_t1_raw[:, 1].astype(float)
    event_dev = y_dev_t1_raw[:, 0].astype(int)
    time_ext = y_ext_t1_raw[:, 1].astype(float)
    event_ext = y_ext_t1_raw[:, 0].astype(int)

    results_t1 = run_task1_pipeline(
        X_dev_t1, time_dev, event_dev,
        X_ext_t1, time_ext, event_ext,
        feat_t1, dev_df, ext_df,
    )
    save_results({"cindex_results": {k: list(v) for k, v in results_t1["cindex_results"].items()}}, "task1_results.json")

    print("\n[Multimodal] Running multimodal pipeline (Task 2)...")
    from models.multimodal_pipeline import run_multimodal_pipeline
    from utils.sensitivity_analysis import signal_quality_stratified_ablation

    mm_results_t2 = run_multimodal_pipeline(
        ecg_dev=ecg_dev,
        tab_dev=X_dev_t2,
        y_dev=y_dev_t2,
        ecg_ext=ecg_ext,
        tab_ext=X_ext_t2,
        y_ext=y_ext_t2,
        ecg_checkpoint_path=ecg_checkpoint_path,
        task="task2",
    )

    sqi_values = ext_df["sqi"].values if "sqi" in ext_df.columns else np.ones(len(ext_df))
    sqi_ablation_df = signal_quality_stratified_ablation(
        y_true=y_ext_t2,
        y_prob_tabular=results_t2["y_prob_ext"]["xgboost"],
        y_prob_multimodal=mm_results_t2["y_prob_ext"],
        sqi_values=sqi_values,
    )
    sqi_ablation_df.to_csv(os.path.join(RESULTS_DIR, "sqi_stratified_ablation.csv"), index=False)
    print(sqi_ablation_df.to_string(index=False))

    print("\n[Multimodal] Running multimodal pipeline (Task 1)...")
    mm_results_t1 = run_multimodal_pipeline(
        ecg_dev=ecg_dev,
        tab_dev=X_dev_t1,
        y_dev=event_dev,
        ecg_ext=ecg_ext,
        tab_ext=X_ext_t1,
        y_ext=event_ext,
        ecg_checkpoint_path=ecg_checkpoint_path,
        task="task1",
        time_dev=time_dev,
        event_dev=event_dev,
        time_ext=time_ext,
        event_ext=event_ext,
    )
    save_results({"multimodal_c_index": list(mm_results_t1["c_index_ci"])}, "task1_multimodal_results.json")

    print("\n[Done] All outputs saved to:", OUTPUT_DIR)
    print("  Figures:", FIGURES_DIR)
    print("  Results:", RESULTS_DIR)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", required=True, help="Path to development CSV")
    parser.add_argument("--ext", required=True, help="Path to external test CSV")
    parser.add_argument("--ecg_dev", required=True, help="Path to development ECG array (.npy, shape n x 12 x 5000)")
    parser.add_argument("--ecg_ext", required=True, help="Path to external ECG array (.npy, shape n x 12 x 5000)")
    parser.add_argument("--ecg_checkpoint", required=True, help="Path to ECGFounder pretrained checkpoint")
    args = parser.parse_args()
    main(args.dev, args.ext, args.ecg_dev, args.ecg_ext, args.ecg_checkpoint)

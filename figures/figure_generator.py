import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, confusion_matrix
from typing import Dict, List, Optional, Tuple
import os
import sys
sys.path.append('..')
from config import FIGURES_DIR, RANDOM_SEED

os.makedirs(FIGURES_DIR, exist_ok=True)

PALETTE = {
    "multimodal": "#1f4e79",
    "tabular_only": "#2e86ab",
    "ecg_only": "#a8dadc",
    "xgboost": "#e63946",
    "lightgbm": "#f4a261",
    "random_forest": "#2a9d8f",
    "logistic": "#8ecae6",
    "deephit": "#023047",
    "deepsurv": "#219ebc",
    "rsf": "#8338ec",
    "cox": "#fb8500",
    "treat_all": "#6c757d",
    "treat_none": "#adb5bd",
    "ckd_pc": "#dc2f02",
    "kfre": "#e85d04",
    "score2_pce": "#f48c06",
}

FIGURE_PARAMS = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}
plt.rcParams.update(FIGURE_PARAMS)


def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curves(
    y_true: np.ndarray,
    models: Dict[str, np.ndarray],
    model_aucs: Dict[str, Tuple[float, float, float]],
    title: str = "ROC Curves",
    filename: str = "roc_curves.pdf",
) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)

    for model_name, y_prob in models.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc, ci_low, ci_high = model_aucs.get(model_name, (0, 0, 0))
        color = PALETTE.get(model_name, "#333333")
        label = f"{model_name.replace('_', ' ').title()} (AUC={auc:.3f}, 95%CI {ci_low:.3f}–{ci_high:.3f})"
        ax.plot(fpr, tpr, color=color, lw=1.5, label=label)

    ax.set_xlabel("1 – Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    return _save(fig, filename)


def plot_calibration(
    y_true: np.ndarray,
    models: Dict[str, np.ndarray],
    n_bins: int = 10,
    title: str = "Calibration Plot",
    filename: str = "calibration.pdf",
) -> str:
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect calibration")

    for model_name, y_prob in models.items():
        fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
        color = PALETTE.get(model_name, "#333333")
        ax.plot(
            mean_pred, fraction_pos,
            marker="o", markersize=4, lw=1.5,
            color=color, label=model_name.replace("_", " ").title(),
        )

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    return _save(fig, filename)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Primary Model",
    title: str = "Confusion Matrix",
    filename: str = "confusion_matrix.pdf",
) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))

    cmap = LinearSegmentedColormap.from_list("custom", ["#ffffff", "#1f4e79"])
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["No Event", "Event"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=11)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"{title}\n({model_name})")

    return _save(fig, filename)


def plot_decision_curve(
    dca_df: pd.DataFrame,
    model_order: List[str] = None,
    thresholds_of_interest: List[float] = None,
    title: str = "Decision Curve Analysis",
    filename: str = "decision_curve.pdf",
) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))

    if model_order is None:
        model_order = dca_df["model"].unique().tolist()

    for model_name in model_order:
        sub = dca_df[dca_df["model"] == model_name]
        color = PALETTE.get(model_name, "#333333")
        linestyle = "--" if model_name in ("treat_all", "treat_none") else "-"
        lw = 1.0 if model_name in ("treat_all", "treat_none") else 1.5
        ax.plot(
            sub["threshold"], sub["net_benefit"],
            color=color, lw=lw, linestyle=linestyle,
            label=model_name.replace("_", " ").title(),
        )

    if thresholds_of_interest:
        for t in thresholds_of_interest:
            ax.axvline(x=t, color="gray", lw=0.6, linestyle=":")

    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none")

    return _save(fig, filename)


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
    title: str = "SHAP Feature Importance",
    filename: str = "shap_summary.pdf",
) -> str:
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    feat_labels = [feature_names[i].replace("_", " ").title() for i in top_idx]
    vals = mean_abs[top_idx]
    colors = ["#1f4e79"] * len(vals)
    bars = ax.barh(range(len(vals)), vals[::-1], color=colors[::-1], height=0.6)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(feat_labels[::-1])
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Global Feature Importance")

    ax = axes[1]
    top_shap = shap_values[:, top_idx]
    top_feat_vals = X[:, top_idx]

    for i, idx in enumerate(top_idx):
        sv = shap_values[:, idx]
        fv = top_feat_vals[:, list(top_idx).index(idx)]
        if fv.std() > 0:
            norm_fv = (fv - fv.min()) / (fv.max() - fv.min() + 1e-8)
        else:
            norm_fv = np.zeros_like(fv)
        scatter = ax.scatter(
            sv, [i] * len(sv),
            c=norm_fv, cmap="RdBu_r", alpha=0.5, s=8, vmin=0, vmax=1,
        )

    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(feat_labels)
    ax.axvline(x=0, color="black", lw=0.8)
    ax.set_xlabel("SHAP Value")
    ax.set_title("SHAP Beeswarm")
    plt.colorbar(scatter, ax=ax, label="Feature Value (normalized)")

    plt.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()

    return _save(fig, filename)


def plot_ecg_gradcam(
    ecg: np.ndarray,
    cam: np.ndarray,
    attention_weights: Optional[np.ndarray] = None,
    fs: int = 500,
    lead_names: List[str] = None,
    title: str = "ECG Grad-CAM",
    filename: str = "ecg_gradcam.pdf",
) -> str:
    if lead_names is None:
        lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

    n_leads = ecg.shape[0]
    n_display = min(6, n_leads)
    time_axis = np.arange(ecg.shape[-1]) / fs

    fig, axes = plt.subplots(n_display, 1, figsize=(14, n_display * 1.6), sharex=True)
    if n_display == 1:
        axes = [axes]

    cmap = LinearSegmentedColormap.from_list("cam", ["#ffffff", "#e63946"])
    cam_resized = np.interp(
        np.linspace(0, 1, ecg.shape[-1]),
        np.linspace(0, 1, len(cam)),
        cam,
    ) if len(cam) != ecg.shape[-1] else cam

    for lead_idx in range(n_display):
        ax = axes[lead_idx]
        signal = ecg[lead_idx]
        ax.plot(time_axis, signal, color="#1f4e79", lw=0.8, zorder=3)

        for j in range(len(time_axis) - 1):
            ax.axvspan(
                time_axis[j], time_axis[j+1],
                alpha=float(cam_resized[j]) * 0.4,
                color="#e63946",
                zorder=2,
                linewidth=0,
            )

        ax.set_ylabel(lead_names[lead_idx], rotation=0, labelpad=20, va="center")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()

    return _save(fig, filename)


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    model_name: str = "Model",
    early_stop_epoch: Optional[int] = None,
    title: str = "Training Curves",
    filename: str = "loss_curves.pdf",
) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))

    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color="#1f4e79", lw=1.5, label="Train loss")
    ax.plot(epochs, val_losses, color="#e63946", lw=1.5, label="Validation loss")

    if early_stop_epoch is not None:
        ax.axvline(x=early_stop_epoch, color="gray", lw=1.0, linestyle="--", label=f"Early stop (epoch {early_stop_epoch})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title} – {model_name}")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none")

    return _save(fig, filename)


def plot_lasso_coefficient_path(
    coef_df: pd.DataFrame,
    selected_features: List[str],
    cv_alpha: float,
    title: str = "LASSO Coefficient Path",
    filename: str = "lasso_path.pdf",
) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))

    feature_cols = [c for c in coef_df.columns if c != "alpha"]
    alphas = coef_df["alpha"].values

    cmap = plt.cm.get_cmap("tab20", len(feature_cols))
    for i, feat in enumerate(feature_cols):
        highlight = feat in selected_features
        lw = 1.8 if highlight else 0.6
        alpha_val = 1.0 if highlight else 0.3
        ax.plot(
            np.log10(alphas + 1e-10), coef_df[feat].values,
            color=cmap(i), lw=lw, alpha=alpha_val,
            label=feat.replace("_", " ").title() if highlight else None,
        )

    ax.axvline(x=np.log10(cv_alpha + 1e-10), color="black", lw=1.0, linestyle="--", label=f"CV α = {cv_alpha:.4f}")
    ax.axhline(y=0, color="gray", lw=0.5)

    ax.set_xlabel("log₁₀(α)")
    ax.set_ylabel("Coefficient")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=8)

    return _save(fig, filename)


def plot_tsne(
    tsne_df: pd.DataFrame,
    title: str = "t-SNE Projection of Fused Representation",
    filename: str = "tsne_projection.pdf",
) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))

    for label_val, color, label_str in [(0, "#8ecae6", "No Event"), (1, "#e63946", "Event")]:
        mask = tsne_df["label"] == label_val
        ax.scatter(
            tsne_df.loc[mask, "dim_1"],
            tsne_df.loc[mask, "dim_2"],
            c=color, alpha=0.5, s=8, label=label_str, linewidths=0,
        )

    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none")

    return _save(fig, filename)


def plot_time_dependent_auc(
    time_points: np.ndarray,
    models_td_auc: Dict[str, np.ndarray],
    models_ci: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Time-Dependent AUC",
    filename: str = "time_dep_auc.pdf",
) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name, td_auc in models_td_auc.items():
        color = PALETTE.get(model_name, "#333333")
        ax.plot(time_points, td_auc, color=color, lw=1.5, label=model_name.replace("_", " ").title())
        if model_name in models_ci:
            ci_low, ci_high = models_ci[model_name]
            ax.fill_between(time_points, ci_low, ci_high, color=color, alpha=0.15)

    ax.axhline(y=0.5, color="gray", lw=0.8, linestyle="--")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Time-Dependent AUC")
    ax.set_title(title)
    ax.set_ylim(0.4, 1.0)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none")

    return _save(fig, filename)


def plot_nomogram(
    nomogram_table: pd.DataFrame,
    task: str = "task2",
    title: str = "Clinical Nomogram",
    filename: str = "nomogram.pdf",
) -> str:
    n_predictors = len(nomogram_table)
    fig_height = 2.0 + n_predictors * 0.9
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")

    ax.set_xlim(0, 100)
    ax.set_ylim(-1, n_predictors + 1)

    ax.text(50, n_predictors + 0.5, title, ha="center", va="center", fontsize=13, fontweight="bold")

    points_axis_y = n_predictors - 0.1
    ax.text(-2, points_axis_y, "Points", ha="right", va="center", fontsize=9, fontweight="bold")
    for pt in range(0, 110, 10):
        ax.plot([pt, pt], [points_axis_y - 0.15, points_axis_y + 0.15], color="black", lw=0.8)
        ax.text(pt, points_axis_y + 0.25, str(pt), ha="center", va="bottom", fontsize=7)

    for row_idx, (_, row) in enumerate(nomogram_table.iterrows()):
        y = n_predictors - 1 - row_idx
        ax.text(-2, y, row["predictor"].replace("_", " ").title(),
                ha="right", va="center", fontsize=9)

        ax.plot([0, row["points_max"]], [y, y], color="#1f4e79", lw=2.0)
        ax.plot(0, y, "|", color="#1f4e79", markersize=8)
        ax.plot(row["points_max"], y, "|", color="#1f4e79", markersize=8)

        ax.text(0, y - 0.25, f"{row['clinical_range_min']}", ha="center", va="top", fontsize=7, color="#555555")
        ax.text(row["points_max"], y - 0.25, f"{row['clinical_range_max']}", ha="center", va="top", fontsize=7, color="#555555")

    return _save(fig, filename)


def plot_subgroup_performance(
    subgroup_df: pd.DataFrame,
    metric_col: str = "auc",
    metric_label: str = "AUC",
    title: str = "Subgroup Performance",
    filename: str = "subgroup_performance.pdf",
) -> str:
    fig, ax = plt.subplots(figsize=(8, max(4, len(subgroup_df) * 0.5)))

    y_pos = np.arange(len(subgroup_df))
    values = subgroup_df[metric_col].values
    ci_low = subgroup_df.get("ci_low", values).values
    ci_high = subgroup_df.get("ci_high", values).values

    ax.barh(y_pos, values, height=0.5, color="#1f4e79", alpha=0.8)
    ax.errorbar(
        values, y_pos,
        xerr=[values - ci_low, ci_high - values],
        fmt="none", color="black", capsize=3, lw=1.0,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [str(s).replace("_", " ").title() for s in subgroup_df["subgroup"].values]
    )
    ax.set_xlabel(metric_label)
    ax.set_title(title)
    ax.axvline(x=0.5, color="gray", lw=0.6, linestyle="--")
    ax.set_xlim(0.4, 1.0)

    return _save(fig, filename)


def plot_fairness_fnr(
    fairness_df: pd.DataFrame,
    title: str = "False-Negative Rate by Group",
    filename: str = "fairness_fnr.pdf",
) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))

    groups = fairness_df["group"].astype(str).values
    fnrs = fairness_df["fnr"].values
    colors = ["#e63946" if fnr == fnrs.max() else "#1f4e79" for fnr in fnrs]

    bars = ax.bar(groups, fnrs, color=colors, width=0.5, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, fnrs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xlabel("Group")
    ax.set_ylabel("False-Negative Rate")
    ax.set_title(title)
    ax.set_ylim(0, min(1.0, fnrs.max() * 1.3))

    return _save(fig, filename)

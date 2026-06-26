import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import sys
sys.path.append('..')

from config import (
    RANDOM_SEED, N_BOOTSTRAP, RESULTS_DIR, FIGURES_DIR,
    ECG_PROJECTION_HEAD_PARAMS, FUSION_PARAMS, MLP_TABULAR_PARAMS,
    DEEPHIT_PARAMS,
)
from models.ecg_backbone import load_ecgfounder_backbone, ECGEmbeddingCache
from models.deep_models import MultimodalModel, DeepHitNet
from models.trainer import (
    train_multimodal_classifier, train_deephit, compute_class_weight,
)
from evaluation.metrics import (
    compute_auc_ci, delong_test, compute_cindex_ci, bootstrap_cindex_diff,
    compute_calibration, recalibrate_predictions,
)
from figures.figure_generator import plot_roc_curves, plot_loss_curves

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def build_multimodal_task2(
    backbone: nn.Module,
    backbone_output_dim: int,
    tabular_input_dim: int,
    embedding_dim: int = 64,
) -> MultimodalModel:
    model = MultimodalModel(
        ecg_backbone=backbone,
        backbone_output_dim=backbone_output_dim,
        tabular_input_dim=tabular_input_dim,
        embedding_dim=embedding_dim,
        fusion_units=FUSION_PARAMS["units"],
        dropout=FUSION_PARAMS["dropout"],
        n_classes=1,
    )
    return model


def train_multimodal_task2(
    ecg_train: np.ndarray,
    tab_train: np.ndarray,
    y_train: np.ndarray,
    ecg_val: np.ndarray,
    tab_val: np.ndarray,
    y_val: np.ndarray,
    backbone: nn.Module,
    backbone_output_dim: int,
) -> tuple:
    model = build_multimodal_task2(backbone, backbone_output_dim, tab_train.shape[1])
    class_weight = compute_class_weight(y_train)

    model, train_losses, val_losses = train_multimodal_classifier(
        model=model,
        ecg_train=ecg_train,
        tab_train=tab_train,
        y_train=y_train,
        ecg_val=ecg_val,
        tab_val=tab_val,
        y_val=y_val,
        lr=ECG_PROJECTION_HEAD_PARAMS["lr"],
        patience=ECG_PROJECTION_HEAD_PARAMS["patience"],
        class_weights=class_weight,
    )
    return model, train_losses, val_losses


def predict_multimodal_task2(
    model: MultimodalModel,
    ecg: np.ndarray,
    tabular: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_probs = []
    n = len(ecg)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ecg_b = torch.FloatTensor(ecg[start:end]).to(device)
            tab_b = torch.FloatTensor(tabular[start:end]).to(device)
            logits = model(ecg_b, tab_b).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def extract_fused_representation(
    model: MultimodalModel,
    ecg: np.ndarray,
    tabular: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_reps = []
    n = len(ecg)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ecg_b = torch.FloatTensor(ecg[start:end]).to(device)
            tab_b = torch.FloatTensor(tabular[start:end]).to(device)
            rep = model.get_fused_representation(ecg_b, tab_b)
            all_reps.append(rep.cpu().numpy())
    return np.concatenate(all_reps, axis=0)


def build_multimodal_task1_deephit(
    backbone: nn.Module,
    backbone_output_dim: int,
    tabular_input_dim: int,
    n_time_bins: int,
    embedding_dim: int = 64,
) -> nn.Module:
    class MultimodalDeepHit(nn.Module):
        def __init__(self):
            super().__init__()
            from models.deep_models import ECGBranch, TabularMLP, DeepHitNet
            self.ecg_branch = ECGBranch(
                backbone=backbone,
                backbone_output_dim=backbone_output_dim,
                output_dim=embedding_dim,
            )
            self.tabular_branch = TabularMLP(
                input_dim=tabular_input_dim,
                output_dim=embedding_dim,
                dropout=FUSION_PARAMS["dropout"],
            )
            self.fusion = nn.Sequential(
                nn.Linear(embedding_dim * 2, FUSION_PARAMS["units"]),
                nn.ReLU(),
                nn.Dropout(FUSION_PARAMS["dropout"]),
            )
            self.cause_heads = nn.ModuleList([
                nn.Linear(FUSION_PARAMS["units"], n_time_bins) for _ in range(2)
            ])
            self.n_time_bins = n_time_bins
            self.n_causes = 2

        def forward(self, ecg: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
            import torch.nn.functional as F
            ecg_emb, _ = self.ecg_branch(ecg)
            tab_emb = self.tabular_branch(tabular)
            fused = self.fusion(torch.cat([ecg_emb, tab_emb], dim=-1))
            cause_outputs = [head(fused) for head in self.cause_heads]
            stacked = torch.stack(cause_outputs, dim=1)
            probs = F.softmax(stacked.view(ecg.size(0), -1), dim=-1)
            probs = probs.view(ecg.size(0), self.n_causes, self.n_time_bins)
            return probs

    return MultimodalDeepHit()


def run_multimodal_pipeline(
    ecg_dev: np.ndarray,
    tab_dev: np.ndarray,
    y_dev: np.ndarray,
    ecg_ext: np.ndarray,
    tab_ext: np.ndarray,
    y_ext: np.ndarray,
    ecg_checkpoint_path: str = None,
    task: str = "task2",
    time_dev: np.ndarray = None,
    event_dev: np.ndarray = None,
    time_ext: np.ndarray = None,
    event_ext: np.ndarray = None,
    n_time_bins: int = 100,
) -> dict:
    print(f"\n[Multimodal] Loading ECGFounder backbone...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, backbone_output_dim = load_ecgfounder_backbone(
        checkpoint_path=ecg_checkpoint_path,
        device=device,
    )
    print(f"  Backbone output dim: {backbone_output_dim}")

    cache = ECGEmbeddingCache()

    split_idx = int(0.8 * len(ecg_dev))
    ecg_tr = ecg_dev[:split_idx]
    ecg_val_split = ecg_dev[split_idx:]
    tab_tr = tab_dev[:split_idx]
    tab_val_split = tab_dev[split_idx:]

    if task == "task2":
        y_tr = y_dev[:split_idx]
        y_val_s = y_dev[split_idx:]

        print(f"[Multimodal] Training multimodal Task 2 model...")
        print(f"  Train: {len(ecg_tr)}, Val: {len(ecg_val_split)}, External: {len(ecg_ext)}")

        model, train_losses, val_losses = train_multimodal_task2(
            ecg_tr, tab_tr, y_tr,
            ecg_val_split, tab_val_split, y_val_s,
            backbone, backbone_output_dim,
        )

        early_stop_epoch = len(train_losses)
        plot_loss_curves(
            train_losses, val_losses, "Multimodal (Task 2)",
            early_stop_epoch=early_stop_epoch,
            filename="task2_loss_multimodal.pdf",
        )

        print("[Multimodal] Predicting on external test set...")
        y_prob_ext = predict_multimodal_task2(model, ecg_ext, tab_ext)
        auc, ci_lo, ci_hi = compute_auc_ci(y_ext, y_prob_ext, N_BOOTSTRAP)
        print(f"  External AUC: {auc:.3f} ({ci_lo:.3f}–{ci_hi:.3f})")

        print("[Multimodal] Extracting fused representations for t-SNE...")
        fused_ext = extract_fused_representation(model, ecg_ext, tab_ext)

        cal = compute_calibration(y_ext, y_prob_ext)
        print(f"  Calibration slope (before recalibration): {cal['calibration_slope']:.3f}")

        y_prob_dev_mm = predict_multimodal_task2(model, ecg_dev, tab_dev)
        y_prob_ext_recal = recalibrate_predictions(y_prob_dev_mm, y_dev, y_prob_ext)
        cal_after = compute_calibration(y_ext, y_prob_ext_recal)
        print(f"  Calibration slope (after recalibration): {cal_after['calibration_slope']:.3f}")

        return {
            "model": model,
            "y_prob_ext": y_prob_ext,
            "y_prob_ext_recalibrated": y_prob_ext_recal,
            "auc": auc,
            "auc_ci": (ci_lo, ci_hi),
            "calibration_before": cal,
            "calibration_after": cal_after,
            "fused_representations": fused_ext,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

    elif task == "task1":
        assert time_dev is not None and event_dev is not None
        assert time_ext is not None and event_ext is not None

        t_tr = time_dev[:split_idx]
        t_val = time_dev[split_idx:]
        e_tr = event_dev[:split_idx]
        e_val = event_dev[split_idx:]

        time_bins = np.quantile(t_tr[e_tr == 1], np.linspace(0, 1, n_time_bins))
        t_bin_tr = np.digitize(t_tr, time_bins).clip(0, n_time_bins - 1)
        t_bin_val = np.digitize(t_val, time_bins).clip(0, n_time_bins - 1)
        t_bin_ext = np.digitize(time_ext, time_bins).clip(0, n_time_bins - 1)

        competing_tr = np.zeros_like(e_tr)
        competing_tr[e_tr == 0] = 1
        competing_val = np.zeros_like(e_val)
        competing_val[e_val == 0] = 1

        print(f"[Multimodal] Building multimodal DeepHit (Task 1)...")
        mm_model = build_multimodal_task1_deephit(
            backbone, backbone_output_dim, tab_dev.shape[1], n_time_bins
        )

        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mm_model = mm_model.to(device_t)

        from models.deep_models import DeepHitLoss
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, mm_model.parameters()),
            lr=DEEPHIT_PARAMS["lr"],
            weight_decay=1e-3,
        )
        criterion = DeepHitLoss(alpha=DEEPHIT_PARAMS["alpha"], sigma=DEEPHIT_PARAMS["sigma"])

        from torch.utils.data import DataLoader, TensorDataset
        ecg_tr_t = torch.FloatTensor(ecg_tr).to(device_t)
        tab_tr_t = torch.FloatTensor(tab_tr).to(device_t)
        t_tr_t = torch.LongTensor(t_bin_tr).to(device_t)
        e_tr_t = torch.LongTensor(e_tr).to(device_t)
        c_tr_t = torch.LongTensor(competing_tr).to(device_t)

        dataset = TensorDataset(ecg_tr_t, tab_tr_t, t_tr_t, e_tr_t, c_tr_t)
        loader = DataLoader(dataset, batch_size=DEEPHIT_PARAMS["batch_size"], shuffle=True)

        ecg_val_t = torch.FloatTensor(ecg_val_split).to(device_t)
        tab_val_t = torch.FloatTensor(tab_val_split).to(device_t)
        t_val_t = torch.LongTensor(t_bin_val).to(device_t)
        e_val_t = torch.LongTensor(e_val).to(device_t)
        c_val_t = torch.LongTensor(competing_val).to(device_t)

        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(DEEPHIT_PARAMS["epochs"]):
            mm_model.train()
            epoch_loss = 0.0
            for ecg_b, tab_b, t_b, e_b, c_b in loader:
                optimizer.zero_grad()
                probs = mm_model(ecg_b, tab_b)
                loss = criterion(probs, t_b, e_b, c_b)
                loss.backward()
                nn.utils.clip_grad_norm_(mm_model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(e_b)
            epoch_loss /= len(e_tr)
            train_losses.append(epoch_loss)

            mm_model.eval()
            with torch.no_grad():
                val_probs = mm_model(ecg_val_t, tab_val_t)
                val_loss = criterion(val_probs, t_val_t, e_val_t, c_val_t).item()
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in mm_model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= DEEPHIT_PARAMS["patience"]:
                    break

        if best_state:
            mm_model.load_state_dict(best_state)

        plot_loss_curves(train_losses, val_losses, "Multimodal DeepHit (Task 1)", filename="task1_loss_multimodal.pdf")

        mm_model.eval()
        ecg_ext_t = torch.FloatTensor(ecg_ext).to(device_t)
        tab_ext_t = torch.FloatTensor(tab_ext).to(device_t)
        with torch.no_grad():
            probs_ext = mm_model(ecg_ext_t, tab_ext_t).cpu().numpy()
        cif_ext = probs_ext[:, 0, :].cumsum(axis=1)
        risk_ext = cif_ext[:, -1]

        ci, ci_lo, ci_hi = compute_cindex_ci(time_ext, event_ext, risk_ext, N_BOOTSTRAP)
        print(f"  External C-index (Multimodal DeepHit): {ci:.3f} ({ci_lo:.3f}–{ci_hi:.3f})")

        return {
            "model": mm_model,
            "risk_ext": risk_ext,
            "cif_ext": cif_ext,
            "c_index": ci,
            "c_index_ci": (ci_lo, ci_hi),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "time_bins": time_bins,
        }

    return {}

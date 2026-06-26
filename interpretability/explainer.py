import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
import sys
sys.path.append('..')
from config import RANDOM_SEED


def compute_shap_values(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    model_type: str = "tree",
    n_background: int = 100,
) -> Tuple[np.ndarray, pd.DataFrame]:
    import shap

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X_background)
        shap_values = explainer.shap_values(X_explain)
    else:
        rng = np.random.RandomState(RANDOM_SEED)
        background_idx = rng.choice(len(X_background), min(n_background, len(X_background)), replace=False)
        background = X_background[background_idx]
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.predict(x),
            background,
        )
        shap_values = explainer.shap_values(X_explain, nsamples=100)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    return shap_values, importance_df


def compute_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    records = []
    for i, feat in enumerate(feature_names):
        sv = shap_values[:, i]
        fv = X[:, i]
        records.append({
            "feature": feat,
            "mean_abs_shap": np.abs(sv).mean(),
            "mean_shap": sv.mean(),
            "std_shap": sv.std(),
            "correlation_with_feature": np.corrcoef(sv, fv)[0, 1] if fv.std() > 0 else 0.0,
        })
    return pd.DataFrame(records).sort_values("mean_abs_shap", ascending=False)


class GradCAM1D:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, ecg_input: torch.Tensor, tabular_input: Optional[torch.Tensor] = None) -> np.ndarray:
        self.model.eval()
        ecg_input = ecg_input.requires_grad_(True)

        if tabular_input is not None:
            output = self.model(ecg_input, tabular_input)
        else:
            output = self.model(ecg_input)

        self.model.zero_grad()
        output.sum().backward()

        if self.gradients is None or self.activations is None:
            return np.zeros(ecg_input.shape[-1])

        weights = self.gradients.mean(dim=-1, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze(0).cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def generate_cam_batch(
        self,
        ecg_inputs: np.ndarray,
        tabular_inputs: Optional[np.ndarray] = None,
        device: str = "cpu",
    ) -> np.ndarray:
        cams = []
        for i in range(len(ecg_inputs)):
            ecg_t = torch.FloatTensor(ecg_inputs[i:i+1]).to(device)
            tab_t = None
            if tabular_inputs is not None:
                tab_t = torch.FloatTensor(tabular_inputs[i:i+1]).to(device)
            cam = self.generate_cam(ecg_t, tab_t)
            cams.append(cam)
        return np.stack(cams, axis=0)


def compute_attention_weights(model, ecg_input: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'ecg_branch') and hasattr(model.ecg_branch, 'temporal_attention'):
            emb = model.ecg_branch.frozen_backbone(ecg_input)
            _, weights = model.ecg_branch.temporal_attention(emb)
            return weights.squeeze(0).cpu().numpy()
    return np.array([])


def ecg_waveform_segment_importance(
    attention_weights: np.ndarray,
    ecg: np.ndarray,
    fs: int = 500,
    segment_labels: List[str] = None,
    segment_boundaries: Dict = None,
) -> pd.DataFrame:
    if segment_labels is None:
        segment_labels = ["P_wave", "PR_interval", "QRS", "ST_segment", "T_wave", "QT_interval"]

    if segment_boundaries is None:
        segment_boundaries = {
            "P_wave": (0.0, 0.12),
            "PR_interval": (0.0, 0.20),
            "QRS": (0.20, 0.28),
            "ST_segment": (0.28, 0.40),
            "T_wave": (0.40, 0.60),
            "QT_interval": (0.20, 0.60),
        }

    n_samples = ecg.shape[-1]
    total_duration = n_samples / fs
    records = []
    attn_flat = attention_weights.mean(axis=0) if attention_weights.ndim > 1 else attention_weights

    for label, (start_frac, end_frac) in segment_boundaries.items():
        start_sample = int(start_frac / total_duration * n_samples)
        end_sample = int(end_frac / total_duration * n_samples)
        segment_attention = attn_flat[min(start_sample, len(attn_flat)-1):min(end_sample, len(attn_flat))]
        if len(segment_attention) > 0:
            cumulative_weight = segment_attention.sum() / (attn_flat.sum() + 1e-8)
        else:
            cumulative_weight = 0.0
        records.append({
            "segment": label,
            "cumulative_attention_weight": cumulative_weight,
            "start_sec": start_frac,
            "end_sec": end_frac,
        })

    df = pd.DataFrame(records).sort_values("cumulative_attention_weight", ascending=False)
    return df


def compute_tsne_representation(
    fused_embeddings: np.ndarray,
    labels: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    from sklearn.manifold import TSNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
    )
    embeddings_2d = tsne.fit_transform(fused_embeddings)
    df = pd.DataFrame(embeddings_2d, columns=["dim_1", "dim_2"])
    df["label"] = labels
    return df

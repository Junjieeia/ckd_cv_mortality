import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import sys
sys.path.append('..')
from config import (
    ECG_N_LEADS, ECG_N_SAMPLES, ECG_PROJECTION_HEAD_PARAMS,
    TEMPORAL_ATTENTION_PARAMS, MLP_TABULAR_PARAMS, FUSION_PARAMS,
    DEEPSURV_PARAMS, DEEPHIT_PARAMS, RANDOM_SEED
)

torch.manual_seed(RANDOM_SEED)


class FrozenECGBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, embedding_dim: int = 512):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.backbone(x)
        return embeddings


class TemporalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int = 2, key_dim: int = 32):
        super().__init__()
        self.n_heads = n_heads
        self.key_dim = key_dim
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        out, weights = self.attn(x, x, x)
        self.attention_weights = weights.detach()
        return out.squeeze(1), weights


class ECGProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projection(x))


class ECGBranch(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        backbone_output_dim: int,
        output_dim: int = 64,
        n_attention_heads: int = 2,
    ):
        super().__init__()
        self.frozen_backbone = FrozenECGBackbone(backbone, embedding_dim=backbone_output_dim)
        self.temporal_attention = TemporalSelfAttention(
            embed_dim=backbone_output_dim,
            n_heads=n_attention_heads,
            key_dim=TEMPORAL_ATTENTION_PARAMS["key_dim"],
        )
        self.projection = ECGProjectionHead(backbone_output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.frozen_backbone(x)
        attended, weights = self.temporal_attention(embeddings)
        projected = self.projection(attended)
        return projected, weights


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, units: list = None, dropout: float = 0.5, output_dim: int = 64):
        super().__init__()
        if units is None:
            units = MLP_TABULAR_PARAMS["units"]
        layers = []
        in_dim = input_dim
        for u in units:
            layers.extend([
                nn.Linear(in_dim, u),
                nn.BatchNorm1d(u),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = u
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalFusion(nn.Module):
    def __init__(
        self,
        ecg_dim: int = 64,
        tabular_dim: int = 64,
        fusion_units: int = 64,
        dropout: float = 0.5,
        n_classes: int = 1,
    ):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(ecg_dim + tabular_dim, fusion_units),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_units, n_classes)

    def forward(self, ecg_emb: torch.Tensor, tab_emb: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([ecg_emb, tab_emb], dim=-1)
        fused = self.fusion_layer(fused)
        return self.classifier(fused)


class MultimodalModel(nn.Module):
    def __init__(
        self,
        ecg_backbone: nn.Module,
        backbone_output_dim: int,
        tabular_input_dim: int,
        embedding_dim: int = 64,
        fusion_units: int = 64,
        dropout: float = 0.5,
        n_classes: int = 1,
    ):
        super().__init__()
        self.ecg_branch = ECGBranch(
            backbone=ecg_backbone,
            backbone_output_dim=backbone_output_dim,
            output_dim=embedding_dim,
        )
        self.tabular_branch = TabularMLP(
            input_dim=tabular_input_dim,
            output_dim=embedding_dim,
            dropout=dropout,
        )
        self.fusion = MultimodalFusion(
            ecg_dim=embedding_dim,
            tabular_dim=embedding_dim,
            fusion_units=fusion_units,
            dropout=dropout,
            n_classes=n_classes,
        )
        self.ecg_attention_weights = None

    def forward(self, ecg: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        ecg_emb, attn_weights = self.ecg_branch(ecg)
        self.ecg_attention_weights = attn_weights
        tab_emb = self.tabular_branch(tabular)
        logits = self.fusion(ecg_emb, tab_emb)
        return logits

    def get_fused_representation(self, ecg: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        ecg_emb, _ = self.ecg_branch(ecg)
        tab_emb = self.tabular_branch(tabular)
        fused = torch.cat([ecg_emb, tab_emb], dim=-1)
        return self.fusion.fusion_layer(fused)


class DeepSurvNet(nn.Module):
    def __init__(self, input_dim: int, units: list = None, dropout: float = 0.4):
        super().__init__()
        if units is None:
            units = DEEPSURV_PARAMS["units"]
        layers = []
        in_dim = input_dim
        for u in units:
            layers.extend([
                nn.Linear(in_dim, u),
                nn.BatchNorm1d(u),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = u
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DeepHitNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_time_bins: int,
        n_causes: int = 2,
        units: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if units is None:
            units = DEEPHIT_PARAMS["units"]
        layers = []
        in_dim = input_dim
        for u in units:
            layers.extend([
                nn.Linear(in_dim, u),
                nn.BatchNorm1d(u),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = u
        self.shared = nn.Sequential(*layers)
        self.cause_heads = nn.ModuleList([
            nn.Linear(in_dim, n_time_bins) for _ in range(n_causes)
        ])
        self.n_causes = n_causes
        self.n_time_bins = n_time_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_out = self.shared(x)
        cause_outputs = [head(shared_out) for head in self.cause_heads]
        stacked = torch.stack(cause_outputs, dim=1)
        probs = F.softmax(stacked.view(x.size(0), -1), dim=-1)
        probs = probs.view(x.size(0), self.n_causes, self.n_time_bins)
        return probs


class NegativeLogLikelihoodLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        n = risk_scores.shape[0]
        sorted_idx = torch.argsort(times, descending=True)
        sorted_risks = risk_scores[sorted_idx]
        sorted_events = events[sorted_idx]

        log_cumsum_exp = torch.logcumsumexp(sorted_risks, dim=0)
        diff = sorted_risks - log_cumsum_exp
        loss = -torch.mean(diff * sorted_events)
        return loss


class DeepHitLoss(nn.Module):
    def __init__(self, alpha: float = 0.2, sigma: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma

    def forward(
        self,
        probs: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
        competing_events: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = probs.shape[0]
        n_causes = probs.shape[1]
        n_bins = probs.shape[2]

        nll_loss = self._nll(probs, times, events)
        rank_loss = self._ranking_loss(probs, times, events)

        return self.alpha * nll_loss + (1 - self.alpha) * rank_loss

    def _nll(self, probs, times, events):
        eps = 1e-8
        batch_size = probs.shape[0]
        time_idx = times.long().clamp(0, probs.shape[2] - 1)
        event_idx = events.long().clamp(0, probs.shape[1] - 1)
        selected = probs[torch.arange(batch_size), event_idx, time_idx]
        nll = -torch.mean(torch.log(selected + eps) * (events > 0).float())
        return nll

    def _ranking_loss(self, probs, times, events):
        cif = probs[:, 0, :].cumsum(dim=-1)
        n = probs.shape[0]
        loss = torch.tensor(0.0, device=probs.device)
        count = 0
        for i in range(n):
            if events[i] == 0:
                continue
            t_i = times[i].long().clamp(0, cif.shape[1] - 1)
            for j in range(n):
                if times[j] > times[i]:
                    diff = cif[i, t_i] - cif[j, t_i]
                    loss += torch.exp(-diff / self.sigma)
                    count += 1
        if count > 0:
            loss = loss / count
        return loss

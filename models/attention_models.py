import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional
import sys
sys.path.append('..')
from config import FT_TRANSFORMER_PARAMS, TABNET_PARAMS, RANDOM_SEED

torch.manual_seed(RANDOM_SEED)


class FTTransformerEmbedding(nn.Module):
    def __init__(self, n_features: int, d_token: int = 192):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_features, d_token))
        self.bias = nn.Parameter(torch.Tensor(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        return torch.cat([cls, x_emb], dim=1)


class FTTransformerBlock(nn.Module):
    def __init__(self, d_token: int, n_heads: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_token),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_token: int = 192,
        n_heads: int = 8,
        d_ffn_factor: float = 4.0,
        n_layers: int = 3,
        dropout: float = 0.2,
        n_classes: int = 1,
    ):
        super().__init__()
        self.embedding = FTTransformerEmbedding(n_features, d_token)
        d_ffn = int(d_token * d_ffn_factor)
        self.blocks = nn.ModuleList([
            FTTransformerBlock(d_token, n_heads, d_ffn, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(x)
        for block in self.blocks:
            tokens = block(tokens)
        cls_output = self.norm(tokens[:, 0])
        return self.head(cls_output)


class FTTransformerClassifier:
    def __init__(
        self,
        n_features: int,
        d_token: int = None,
        n_heads: int = None,
        d_ffn_factor: float = None,
        n_layers: int = None,
        dropout: float = None,
        lr: float = None,
        batch_size: int = None,
        epochs: int = 100,
        patience: int = 15,
        weight_decay: float = 1e-4,
    ):
        self.n_features = n_features
        self.d_token = d_token or FT_TRANSFORMER_PARAMS["d_token"]
        self.n_heads = n_heads or FT_TRANSFORMER_PARAMS["n_heads"]
        self.d_ffn_factor = d_ffn_factor or FT_TRANSFORMER_PARAMS["d_ffn_factor"]
        self.n_layers = n_layers or FT_TRANSFORMER_PARAMS["n_layers"]
        self.dropout = dropout or FT_TRANSFORMER_PARAMS["dropout"]
        self.lr = lr or FT_TRANSFORMER_PARAMS["lr"]
        self.batch_size = batch_size or FT_TRANSFORMER_PARAMS["batch_size"]
        self.epochs = epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses_ = []
        self.val_losses_ = []
        self.model_ = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weight: Optional[float] = None,
    ) -> "FTTransformerClassifier":
        self.model_ = FTTransformer(
            n_features=self.n_features,
            d_token=self.d_token,
            n_heads=self.n_heads,
            d_ffn_factor=self.d_ffn_factor,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        pos_weight = torch.tensor([class_weight or 1.0], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_tr = torch.FloatTensor(X_train).to(self.device)
        y_tr = torch.FloatTensor(y_train).to(self.device)
        X_v = torch.FloatTensor(X_val).to(self.device)
        y_v = torch.FloatTensor(y_val).to(self.device)

        dataset = TensorDataset(X_tr, y_tr)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for X_b, y_b in loader:
                optimizer.zero_grad()
                logits = self.model_(X_b).squeeze(-1)
                loss = criterion(logits, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(y_b)
            epoch_loss /= len(y_train)
            self.train_losses_.append(epoch_loss)
            scheduler.step()

            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(X_v).squeeze(-1)
                val_loss = criterion(val_logits, y_v).item()
            self.val_losses_.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state:
            self.model_.load_state_dict(best_state)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits = self.model_(X_t).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class TabNetStep(nn.Module):
    def __init__(self, n_features: int, n_d: int, n_a: int, gamma: float):
        super().__init__()
        self.fc = nn.Linear(n_features, n_d + n_a)
        self.bn = nn.BatchNorm1d(n_d + n_a)
        self.attention_fc = nn.Linear(n_a, n_features)
        self.gamma = gamma

    def forward(
        self,
        x: torch.Tensor,
        prior_scales: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.bn(self.fc(x))
        a = torch.relu(h[:, :h.size(1) // 2])
        d = torch.relu(h[:, h.size(1) // 2:])

        attention_logits = self.attention_fc(a) * prior_scales
        attention = torch.softmax(attention_logits, dim=-1)
        updated_prior = prior_scales * (self.gamma - attention)

        masked_x = attention * x
        return masked_x, d, updated_prior


class TabNetClassifier:
    def __init__(
        self,
        n_features: int,
        n_d: int = None,
        n_a: int = None,
        n_steps: int = None,
        gamma: float = None,
        lr: float = None,
        batch_size: int = None,
        epochs: int = 100,
        patience: int = 15,
    ):
        self.n_features = n_features
        self.n_d = n_d or TABNET_PARAMS["n_d"]
        self.n_a = n_a or TABNET_PARAMS["n_a"]
        self.n_steps = n_steps or TABNET_PARAMS["n_steps"]
        self.gamma = gamma or TABNET_PARAMS["gamma"]
        self.lr = lr or TABNET_PARAMS["lr"]
        self.batch_size = batch_size or TABNET_PARAMS["batch_size"]
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses_ = []
        self.val_losses_ = []
        self.model_ = None

    def _build_model(self) -> nn.Module:
        class TabNetNet(nn.Module):
            def __init__(self, n_features, n_d, n_a, n_steps, gamma):
                super().__init__()
                self.initial_bn = nn.BatchNorm1d(n_features)
                self.steps = nn.ModuleList([
                    TabNetStep(n_features, n_d, n_a, gamma) for _ in range(n_steps)
                ])
                self.final_fc = nn.Linear(n_d, 1)
                self.n_steps = n_steps

            def forward(self, x):
                x = self.initial_bn(x)
                prior_scales = torch.ones_like(x)
                step_outputs = []
                for step in self.steps:
                    masked_x, d, prior_scales = step(x, prior_scales)
                    step_outputs.append(d)
                out = torch.stack(step_outputs, dim=0).mean(dim=0)
                return self.final_fc(out)

        return TabNetNet(self.n_features, self.n_d, self.n_a, self.n_steps, self.gamma)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weight: Optional[float] = None,
    ) -> "TabNetClassifier":
        self.model_ = self._build_model().to(self.device)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        pos_weight = torch.tensor([class_weight or 1.0], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_tr = torch.FloatTensor(X_train).to(self.device)
        y_tr = torch.FloatTensor(y_train).to(self.device)
        X_v = torch.FloatTensor(X_val).to(self.device)
        y_v = torch.FloatTensor(y_val).to(self.device)

        dataset = TensorDataset(X_tr, y_tr)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for X_b, y_b in loader:
                optimizer.zero_grad()
                logits = self.model_(X_b).squeeze(-1)
                loss = criterion(logits, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(y_b)
            epoch_loss /= len(y_train)
            self.train_losses_.append(epoch_loss)

            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(X_v).squeeze(-1)
                val_loss = criterion(val_logits, y_v).item()
            self.val_losses_.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state:
            self.model_.load_state_dict(best_state)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits = self.model_(X_t).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

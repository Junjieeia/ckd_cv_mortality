import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple, List
import sys
sys.path.append('..')
from config import RANDOM_SEED, DEEPSURV_PARAMS, DEEPHIT_PARAMS, MLP_TABULAR_PARAMS

torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_binary_classifier(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 3e-4,
    batch_size: int = 128,
    epochs: int = 100,
    patience: int = 15,
    weight_decay: float = 1e-3,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[nn.Module, List[float], List[float]]:
    device = get_device()
    model = model.to(device)

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.FloatTensor(y_val).to(device)

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    pos_weight = None
    if class_weights is not None:
        pos_weight = class_weights.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            logits = model(X_b).squeeze(-1)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(y_b)

        epoch_loss /= len(y_train)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_v).squeeze(-1)
            val_loss = criterion(val_logits, y_v).item()
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


def train_multimodal_classifier(
    model: nn.Module,
    ecg_train: np.ndarray,
    tab_train: np.ndarray,
    y_train: np.ndarray,
    ecg_val: np.ndarray,
    tab_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 2e-5,
    batch_size: int = 64,
    epochs: int = 100,
    patience: int = 7,
    weight_decay: float = 1e-3,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[nn.Module, List[float], List[float]]:
    device = get_device()
    model = model.to(device)

    ecg_tr = torch.FloatTensor(ecg_train).to(device)
    tab_tr = torch.FloatTensor(tab_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    ecg_v = torch.FloatTensor(ecg_val).to(device)
    tab_v = torch.FloatTensor(tab_val).to(device)
    y_v = torch.FloatTensor(y_val).to(device)

    dataset = TensorDataset(ecg_tr, tab_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    pos_weight = None
    if class_weights is not None:
        pos_weight = class_weights.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for ecg_b, tab_b, y_b in loader:
            optimizer.zero_grad()
            logits = model(ecg_b, tab_b).squeeze(-1)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(y_b)

        epoch_loss /= len(y_train)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_logits = model(ecg_v, tab_v).squeeze(-1)
            val_loss = criterion(val_logits, y_v).item()
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


def train_deepsurv(
    model: nn.Module,
    X_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    X_val: np.ndarray,
    time_val: np.ndarray,
    event_val: np.ndarray,
    lr: float = None,
    batch_size: int = None,
    epochs: int = None,
    patience: int = None,
    weight_decay: float = 1e-3,
) -> Tuple[nn.Module, List[float], List[float]]:
    from models.deep_models import NegativeLogLikelihoodLoss

    lr = lr or DEEPSURV_PARAMS["lr"]
    batch_size = batch_size or DEEPSURV_PARAMS["batch_size"]
    epochs = epochs or DEEPSURV_PARAMS["epochs"]
    patience = patience or DEEPSURV_PARAMS["patience"]

    device = get_device()
    model = model.to(device)

    X_tr = torch.FloatTensor(X_train).to(device)
    t_tr = torch.FloatTensor(time_train).to(device)
    e_tr = torch.FloatTensor(event_train).to(device)

    dataset = TensorDataset(X_tr, t_tr, e_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = NegativeLogLikelihoodLoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    X_v = torch.FloatTensor(X_val).to(device)
    t_v = torch.FloatTensor(time_val).to(device)
    e_v = torch.FloatTensor(event_val).to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_b, t_b, e_b in loader:
            optimizer.zero_grad()
            risk = model(X_b)
            loss = criterion(risk, t_b, e_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(e_b)
        epoch_loss /= len(event_train)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_risk = model(X_v)
            val_loss = criterion(val_risk, t_v, e_v).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


def train_deephit(
    model: nn.Module,
    X_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    competing_train: np.ndarray,
    X_val: np.ndarray,
    time_val: np.ndarray,
    event_val: np.ndarray,
    competing_val: np.ndarray,
    lr: float = None,
    batch_size: int = None,
    epochs: int = None,
    patience: int = None,
    alpha: float = None,
    sigma: float = None,
    weight_decay: float = 1e-3,
) -> Tuple[nn.Module, List[float], List[float]]:
    from models.deep_models import DeepHitLoss

    lr = lr or DEEPHIT_PARAMS["lr"]
    batch_size = batch_size or DEEPHIT_PARAMS["batch_size"]
    epochs = epochs or DEEPHIT_PARAMS["epochs"]
    patience = patience or DEEPHIT_PARAMS["patience"]
    alpha = alpha or DEEPHIT_PARAMS["alpha"]
    sigma = sigma or DEEPHIT_PARAMS["sigma"]

    device = get_device()
    model = model.to(device)

    X_tr = torch.FloatTensor(X_train).to(device)
    t_tr = torch.LongTensor(time_train).to(device)
    e_tr = torch.LongTensor(event_train).to(device)
    c_tr = torch.LongTensor(competing_train).to(device)

    dataset = TensorDataset(X_tr, t_tr, e_tr, c_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = DeepHitLoss(alpha=alpha, sigma=sigma)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    X_v = torch.FloatTensor(X_val).to(device)
    t_v = torch.LongTensor(time_val).to(device)
    e_v = torch.LongTensor(event_val).to(device)
    c_v = torch.LongTensor(competing_val).to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_b, t_b, e_b, c_b in loader:
            optimizer.zero_grad()
            probs = model(X_b)
            loss = criterion(probs, t_b, e_b, c_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(e_b)
        epoch_loss /= len(event_train)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_probs = model(X_v)
            val_loss = criterion(val_probs, t_v, e_v, c_v).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


def compute_class_weight(y: np.ndarray) -> torch.Tensor:
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    pos_weight = n_neg / (n_pos + 1e-8)
    return torch.tensor([pos_weight], dtype=torch.float32)

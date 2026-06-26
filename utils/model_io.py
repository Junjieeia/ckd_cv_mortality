import os
import pickle
import torch
import numpy as np
import sys
sys.path.append('..')
from config import MODELS_DIR

os.makedirs(MODELS_DIR, exist_ok=True)


def save_sklearn_model(model, name: str) -> str:
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_sklearn_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_torch_model(model: torch.nn.Module, name: str) -> str:
    path = os.path.join(MODELS_DIR, f"{name}.pt")
    torch.save(model.state_dict(), path)
    return path


def load_torch_model(model: torch.nn.Module, name: str) -> torch.nn.Module:
    path = os.path.join(MODELS_DIR, f"{name}.pt")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model


def save_numpy_array(array: np.ndarray, name: str) -> str:
    path = os.path.join(MODELS_DIR, f"{name}.npy")
    np.save(path, array)
    return path


def load_numpy_array(name: str) -> np.ndarray:
    path = os.path.join(MODELS_DIR, f"{name}.npy")
    return np.load(path)


def save_scaler_imputer(scaler, imputer, task: str) -> tuple:
    scaler_path = save_sklearn_model(scaler, f"{task}_scaler")
    imputer_path = save_sklearn_model(imputer, f"{task}_imputer")
    return scaler_path, imputer_path


def load_scaler_imputer(task: str) -> tuple:
    scaler = load_sklearn_model(f"{task}_scaler")
    imputer = load_sklearn_model(f"{task}_imputer")
    return scaler, imputer


def checkpoint_exists(name: str, model_type: str = "sklearn") -> bool:
    ext = ".pkl" if model_type == "sklearn" else ".pt"
    path = os.path.join(MODELS_DIR, f"{name}{ext}")
    return os.path.exists(path)


def list_saved_models() -> list:
    if not os.path.exists(MODELS_DIR):
        return []
    return [
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".pkl") or f.endswith(".pt") or f.endswith(".npy")
    ]

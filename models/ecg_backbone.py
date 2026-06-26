import torch
import torch.nn as nn
import numpy as np
import os
from typing import Optional, Tuple
import sys
sys.path.append('..')
from config import ECG_N_LEADS, RANDOM_SEED

torch.manual_seed(RANDOM_SEED)


def load_ecgfounder_backbone(
    checkpoint_path: str,
    output_dim: int = 512,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, int]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"ECGFounder checkpoint not found at: {checkpoint_path}\n"
            "Download the pretrained weights from:\n"
            "  https://github.com/PKUDigitalHealth/ECGFounder\n"
            "  or https://huggingface.co/PKUDigitalHealth/ECGFounder\n"
            "Then pass the path via --ecg_checkpoint."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, nn.Module):
        backbone = checkpoint
    elif isinstance(checkpoint, dict):
        if "model" in checkpoint:
            backbone = checkpoint["model"]
        else:
            raise KeyError(
                "Checkpoint dict does not contain a 'model' key. "
                f"Available keys: {list(checkpoint.keys())}"
            )
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)}")

    for param in backbone.parameters():
        param.requires_grad = False

    backbone = backbone.to(device)
    backbone.eval()

    probe = torch.zeros(1, ECG_N_LEADS, 5000, device=device)
    with torch.no_grad():
        probe_out = backbone(probe)
    actual_output_dim = probe_out.shape[-1]

    return backbone, actual_output_dim


def verify_no_overlap_with_pretraining(
    institution_ids: list,
    date_range: tuple,
    pretraining_institutions: list = None,
    pretraining_date_range: tuple = None,
) -> dict:
    pretraining_institutions = pretraining_institutions or ["Harvard", "Emory"]
    pretraining_date_range = pretraining_date_range or (None, None)

    institution_overlap = [
        inst for inst in institution_ids
        if any(pt.lower() in inst.lower() for pt in pretraining_institutions)
    ]

    report = {
        "study_institutions": institution_ids,
        "pretraining_institutions": pretraining_institutions,
        "institution_overlap": institution_overlap,
        "overlap_detected": len(institution_overlap) > 0,
        "date_range_study": date_range,
        "date_range_pretraining": pretraining_date_range,
        "conclusion": (
            "OVERLAP DETECTED — review before proceeding"
            if len(institution_overlap) > 0
            else "No institutional overlap detected with Harvard–Emory pretraining corpus"
        ),
    }
    return report


class ECGEmbeddingCache:
    def __init__(self, cache_dir: str = "outputs/ecg_embeddings"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, split: str) -> str:
        return os.path.join(self.cache_dir, f"{split}_embeddings.npy")

    def exists(self, split: str) -> bool:
        return os.path.exists(self._path(split))

    def save(self, embeddings: np.ndarray, split: str) -> str:
        path = self._path(split)
        np.save(path, embeddings)
        return path

    def load(self, split: str) -> np.ndarray:
        return np.load(self._path(split))

    def extract_and_cache(
        self,
        backbone: nn.Module,
        ecg_array: np.ndarray,
        split: str,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        if self.exists(split):
            return self.load(split)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        backbone = backbone.to(device)
        backbone.eval()

        all_embeddings = []
        n = len(ecg_array)

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = torch.FloatTensor(ecg_array[start:end]).to(device)
                emb = backbone(batch).cpu().numpy()
                all_embeddings.append(emb)

        embeddings = np.concatenate(all_embeddings, axis=0)
        self.save(embeddings, split)
        return embeddings

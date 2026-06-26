import numpy as np
import torch
from scipy import signal as scipy_signal
from typing import Tuple, Optional
import sys
sys.path.append('..')
from config import (
    ECG_SAMPLING_RATE, ECG_DURATION_SEC, ECG_N_LEADS, ECG_N_SAMPLES,
    SQI_HIGH_THRESHOLD, SQI_LOW_THRESHOLD
)


def remove_baseline_wander(ecg: np.ndarray, fs: int = ECG_SAMPLING_RATE) -> np.ndarray:
    fc = 0.5
    b, a = scipy_signal.butter(3, fc / (fs / 2), btype='high')
    filtered = scipy_signal.filtfilt(b, a, ecg, axis=-1)
    return filtered


def bandpass_filter(ecg: np.ndarray, fs: int = ECG_SAMPLING_RATE, lowcut: float = 0.5, highcut: float = 150.0) -> np.ndarray:
    b, a = scipy_signal.butter(4, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
    filtered = scipy_signal.filtfilt(b, a, ecg, axis=-1)
    return filtered


def notch_filter(ecg: np.ndarray, fs: int = ECG_SAMPLING_RATE, freq: float = 50.0) -> np.ndarray:
    b, a = scipy_signal.iirnotch(freq / (fs / 2), Q=30.0)
    filtered = scipy_signal.filtfilt(b, a, ecg, axis=-1)
    return filtered


def normalize_amplitude(ecg: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(ecg)
    for i in range(ecg.shape[0]):
        lead = ecg[i]
        std = lead.std()
        if std > 1e-8:
            normalized[i] = (lead - lead.mean()) / std
        else:
            normalized[i] = lead
    return normalized


def resample_and_window(ecg: np.ndarray, original_fs: int, target_fs: int = ECG_SAMPLING_RATE, duration: int = ECG_DURATION_SEC) -> np.ndarray:
    target_samples = target_fs * duration
    if original_fs != target_fs:
        num_samples = int(ecg.shape[-1] * target_fs / original_fs)
        ecg = scipy_signal.resample(ecg, num_samples, axis=-1)
    current_samples = ecg.shape[-1]
    if current_samples >= target_samples:
        start = (current_samples - target_samples) // 2
        ecg = ecg[:, start:start + target_samples]
    else:
        pad_total = target_samples - current_samples
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        ecg = np.pad(ecg, ((0, 0), (pad_left, pad_right)), mode='constant')
    return ecg


def compute_sqi(ecg: np.ndarray) -> float:
    flatline_scores = []
    for i in range(ecg.shape[0]):
        lead = ecg[i]
        flatline_ratio = (np.abs(np.diff(lead)) < 1e-4).mean()
        flatline_scores.append(1.0 - flatline_ratio)
    flatline_score = np.mean(flatline_scores)

    noise_scores = []
    for i in range(ecg.shape[0]):
        lead = ecg[i]
        noise_level = np.percentile(np.abs(lead - np.median(lead)), 95)
        signal_level = lead.std()
        if signal_level > 1e-8:
            snr = signal_level / (noise_level + 1e-8)
            noise_scores.append(min(snr / 10.0, 1.0))
        else:
            noise_scores.append(0.0)
    noise_score = np.mean(noise_scores)

    lead_corr = np.corrcoef(ecg)
    off_diag = lead_corr[np.triu_indices(ECG_N_LEADS, k=1)]
    consistency_score = np.clip(np.mean(np.abs(off_diag)), 0, 1)

    sqi = 0.4 * flatline_score + 0.4 * noise_score + 0.2 * consistency_score
    return float(np.clip(sqi, 0, 1))


def classify_sqi(sqi: float) -> str:
    if sqi >= SQI_HIGH_THRESHOLD:
        return "high"
    elif sqi >= SQI_LOW_THRESHOLD:
        return "low"
    else:
        return "fail"


def preprocess_ecg(
    ecg: np.ndarray,
    original_fs: int = ECG_SAMPLING_RATE,
) -> Tuple[Optional[np.ndarray], float, str]:
    ecg = remove_baseline_wander(ecg, fs=original_fs)
    ecg = bandpass_filter(ecg, fs=original_fs)
    ecg = notch_filter(ecg, fs=original_fs)
    ecg = resample_and_window(ecg, original_fs=original_fs)
    ecg = normalize_amplitude(ecg)
    sqi = compute_sqi(ecg)
    quality_class = classify_sqi(sqi)
    if quality_class == "fail":
        return None, sqi, quality_class
    return ecg, sqi, quality_class


def ecg_to_tensor(ecg: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(ecg).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


class ECGAugmentor:
    def __init__(self, fs: int = ECG_SAMPLING_RATE):
        self.fs = fs

    def time_shift(self, ecg: np.ndarray, max_shift_sec: float = 0.1) -> np.ndarray:
        max_shift = int(max_shift_sec * self.fs)
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(ecg, shift, axis=-1)

    def amplitude_scale(self, ecg: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        scale = np.random.uniform(*scale_range)
        return ecg * scale

    def add_baseline_noise(self, ecg: np.ndarray, noise_std: float = 0.02) -> np.ndarray:
        noise = np.random.randn(*ecg.shape) * noise_std
        return ecg + noise

    def add_gaussian_noise(self, ecg: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
        noise = np.random.randn(*ecg.shape) * noise_std
        return ecg + noise

    def augment(self, ecg: np.ndarray) -> np.ndarray:
        ecg = self.time_shift(ecg)
        ecg = self.amplitude_scale(ecg)
        if np.random.rand() > 0.5:
            ecg = self.add_baseline_noise(ecg)
        if np.random.rand() > 0.5:
            ecg = self.add_gaussian_noise(ecg)
        return ecg


class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, ecg_array: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.ecg_array = ecg_array
        self.labels = labels
        self.augment = augment
        self.augmentor = ECGAugmentor()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ecg = self.ecg_array[idx].copy()
        if self.augment:
            ecg = self.augmentor.augment(ecg)
        ecg_tensor = torch.from_numpy(ecg).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return ecg_tensor, label

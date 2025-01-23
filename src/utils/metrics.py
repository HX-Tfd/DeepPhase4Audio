import torch
import torch.nn as nn

def MAE(x_true, x_pred):
    """computes the Mean Absolute Error"""
    return nn.L1Loss()(x_true, x_pred)


def log_spectral_distance(x, y, n_fft=2048, hop_length=512, win_length=2048):
    """
    Compute the log spectral distance between two audio signals.

    Args:
    - x (Tensor): Audio signal tensor of shape [B, L]
    - y (Tensor): Audio signal tensor of shape [B, L]
    - n_fft (int): FFT size (default 2048)
    - hop_length (int): Hop length (default 512)
    - win_length (int): Window length (default 2048)

    Returns:
    - log_spectral_distance (Tensor): Log spectral distance for each sample in the batch, shape [B,]
    """
    # Apply Short-Time Fourier Transform (STFT)
    x_stft = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)
    y_stft = torch.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)

    # Compute magnitudes of the STFT
    x_mag = torch.abs(x_stft)
    y_mag = torch.abs(y_stft)

    # Logarithmic transformation of magnitudes
    x_log_mag = torch.log(x_mag + 1e-8)  # Add small constant to avoid log(0)
    y_log_mag = torch.log(y_mag + 1e-8)

    # Compute the squared difference of the log-magnitudes
    # Sum over the frequency bins (axis=-2), time bins (axis=-1), and the batch (axis=0)
    log_spectral_distance = torch.mean((x_log_mag - y_log_mag) ** 2, dim=[-2, -1])

    return log_spectral_distance

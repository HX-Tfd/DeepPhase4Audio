"""
Spectrogram-based Losses

A pytorch implementation of parts in 
https://github.com/magenta/ddsp/blob/main/ddsp/losses.py#L102 
"""
import torch
import torch.nn as nn
import torch.functional as F

import functools 


def safe_log(x: torch.Tensor, eps=1e-5):
    return torch.log(x.clamp_min_(eps))


def compute_magnitude(audio, size=2048, overlap=0.75, pad_end=True):
    """
    Compute the magnitude of the STFT of the input audio.

    Args:
    - audio (Tensor): Input audio tensor of shape [B, L] (batch, length)
    - size (int): Size of the FFT window (default: 2048)
    - overlap (float): Overlap ratio (default: 0.75)
    - pad_end (bool): Whether to pad the end of the audio (default: True)

    Returns:
    - mag (Tensor): Magnitude of the STFT with shape [B, F, T]
      - B: Batch size
      - F: Number of frequency bins
      - T: Number of time frames
    """
    hop_length = int(size * (1 - overlap))
    stft_result = torch.stft(audio, n_fft=size, hop_length=hop_length, win_length=size, return_complex=True, pad_mode='constant' if pad_end else 'reflect')
    mag = torch.abs(stft_result)
    
    return mag.float()

    

def mean_difference(target, value, loss_type='L1', weights=None):
    """
    Compute the average loss based on the specified loss type.

    Args:
    - target (Tensor): Target tensor.
    - value (Tensor): Value tensor.
    - loss_type (str): One of 'L1', 'L2', or 'COSINE'. Specifies the loss type.
    - weights (Tensor, optional): A weighting mask for the per-element differences.

    Returns:
    - Tensor: The average loss.

    Raises:
    - ValueError: If loss_type is not one of 'L1', 'L2', or 'COSINE'.
    """
    difference = target - value
    weights = torch.ones_like(target) if weights is None else weights
    loss_type = loss_type.upper()

    if loss_type == 'L1':
        return torch.mean(torch.abs(difference * weights))
    elif loss_type == 'L2':
        return torch.mean((difference ** 2) * weights)
    elif loss_type == 'COSINE':
        # Compute cosine similarity, output is 1 - cosine similarity
        cosine_sim = F.cosine_similarity(target, value, dim=-1)
        return torch.mean(1 - cosine_sim)
    else:
        raise ValueError(f'Loss type ({loss_type}) must be "L1", "L2", or "COSINE".')
    
    

def diff(x, axis=-1):
    """
    Take the finite difference of a tensor along an axis.

    Args:
    - x (Tensor): Input tensor of any dimension.
    - axis (int): Axis on which to take the finite difference.

    Returns:
    - Tensor: Tensor with size reduced by 1 along the difference dimension.
    
    Raises:
    - ValueError: If the axis is out of range for the tensor.
    """
    ndim = x.dim()  # Get the number of dimensions of the tensor
    if axis >= ndim or axis < -ndim:
        raise ValueError(f"Invalid axis index: {axis} for tensor with {ndim} dimensions.")
    
    # Create slices for the front and back parts of the tensor
    front_slice = [slice(None)] * ndim
    back_slice = [slice(None)] * ndim
    
    if axis < 0:
        axis = ndim + axis  # Handle negative axes (from the end)
    
    front_slice[axis] = slice(1, None)  # Slice for the front (starting from index 1 along the axis)
    back_slice[axis] = slice(None, -1)  # Slice for the back (up to the second-to-last index along the axis)
    
    # Apply slicing
    slice_front = x[tuple(front_slice)]
    slice_back = x[tuple(back_slice)]
    
    # Compute the finite difference
    d = slice_front - slice_back
    
    return d


    

class SpectralLoss(nn.Module):
  """Multi-scale spectrogram loss.

  This loss is the bread-and-butter of comparing two audio signals. It offers
  a range of options to compare spectrograms, many of which are redunant, but
  emphasize different aspects of the signal. By far, the most common comparisons
  are magnitudes (mag_weight) and log magnitudes (logmag_weight).
  """

  def __init__(self,
               fft_sizes=(2048, 1024, 512, 256, 128, 64),
               loss_type='L1',
               mag_weight=1.0,
               delta_time_weight=0.0,
               delta_freq_weight=0.0,
               cumsum_freq_weight=0.0,
               logmag_weight=0.0,
            #    loudness_weight=0.0,
                ):
    """Constructor, set loss weights of various components.

    Args:
      fft_sizes: Compare spectrograms at each of this list of fft sizes. Each
        spectrogram has a time-frequency resolution trade-off based on fft size,
        so comparing multiple scales allows multiple resolutions.
      loss_type: One of 'L1', 'L2', or 'COSINE'.
      mag_weight: Weight to compare linear magnitudes of spectrograms. Core
        audio similarity loss. More sensitive to peak magnitudes than log
        magnitudes.
      delta_time_weight: Weight to compare the first finite difference of
        spectrograms in time. Emphasizes changes of magnitude in time, such as
        at transients.
      delta_freq_weight: Weight to compare the first finite difference of
        spectrograms in frequency. Emphasizes changes of magnitude in frequency,
        such as at the boundaries of a stack of harmonics.
      cumsum_freq_weight: Weight to compare the cumulative sum of spectrograms
        across frequency for each slice in time. Similar to a 1-D Wasserstein
        loss, this hopefully provides a non-vanishing gradient to push two
        non-overlapping sinusoids towards eachother.
      logmag_weight: Weight to compare log magnitudes of spectrograms. Core
        audio similarity loss. More sensitive to quiet magnitudes than linear
        magnitudes.
      loudness_weight: Weight to compare the overall perceptual loudness of two
        signals. Very high-level loss signal that is a subset of mag and
        logmag losses.
      name: Name of the module.
    """
    super().__init__()
    self.fft_sizes = fft_sizes
    self.loss_type = loss_type
    self.mag_weight = mag_weight
    self.delta_time_weight = delta_time_weight
    self.delta_freq_weight = delta_freq_weight
    self.cumsum_freq_weight = cumsum_freq_weight
    self.logmag_weight = logmag_weight
    # self.loudness_weight = loudness_weight

    self.spectrogram_ops = []
    for size in self.fft_sizes:
        spectrogram_op = functools.partial(compute_magnitude, size=size)
        self.spectrogram_ops.append(spectrogram_op)

  def forward(self, x, y, weights=None):
    """x: target audio, y: prediction"""
    loss = 0.0

    # Compute loss for each fft size.
    for loss_op in self.spectrogram_ops:
        target_mag = loss_op(x)
        value_mag = loss_op(y)

        # Add magnitude loss.
        if self.mag_weight > 0:
            loss += self.mag_weight * mean_difference(
                target_mag, value_mag, self.loss_type, weights=weights)

        if self.delta_time_weight > 0:
            target = diff(target_mag, axis=1)
            value = diff(value_mag, axis=1)
            loss += self.delta_time_weight * mean_difference(
                target, value, self.loss_type, weights=weights)

        if self.delta_freq_weight > 0:
            target = diff(target_mag, axis=2)
            value = diff(value_mag, axis=2)
            loss += self.delta_freq_weight * mean_difference(
                target, value, self.loss_type, weights=weights)

        if self.cumsum_freq_weight > 0:
            target = torch.cumsum(target_mag, axis=2)
            value = torch.cumsum(value_mag, axis=2)
            loss += self.cumsum_freq_weight * mean_difference(
                target, value, self.loss_type, weights=weights)

        # Add logmagnitude loss, reusing spectrogram.
        if self.logmag_weight > 0:
            target = safe_log(target_mag)
            value = safe_log(value_mag)
            loss += self.logmag_weight * mean_difference(
                target, value, self.loss_type, weights=weights)

    return loss


if __name__ == "__main__":
    # Example usage
    B = 4  # Batch size
    L = 16000  # Length of the signal
    x = torch.randn(B, L)  
    y = torch.randn(B, L) 
    
    loss = SpectralLoss(loss_type='L1')
    print(loss(x, y))
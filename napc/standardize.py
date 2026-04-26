import h5py
import numpy as np
from pathlib import Path
from typing import Tuple

import numpy.typing as npt


def calcMeanStdPerPixel(
    inpath: str, scale_factor: float = 1000.0
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Calculate per-pixel mean and standard deviation across all videos in HDF5 file.

    Computes pixel-wise statistics across all frames and videos, with optional scaling
    to prevent numerical overflow. Handles both grayscale (H, W) and multi-channel (H, W, C)
    video formats.

    Args:
        inpath: Path to HDF5 file containing videos of shape (frames, H, W[, C]).
        scale_factor: Scaling factor to divide video values before processing (default: 1000.0).
            Used to prevent overflow during accumulation of large values.

    Returns:
        Tuple of (means, stds) where each has shape (H, W[, C]):
            - means: Per-pixel mean values, rescaled to original units.
            - stds: Per-pixel standard deviation values, rescaled to original units.

    Raises:
        ValueError: If video shape is not (frames, H, W) or (frames, H, W, C).
    """
    with h5py.File(inpath, "r") as infile:
        video_keys = list(infile.keys())
        image_shape = infile[video_keys[0]].shape  # (frames, height, width[, channels])
        print("image shape:", image_shape)

        # Determine shape based on number of dimensions
        if len(image_shape) == 4:
            _, height, width, channels = image_shape
            mean_shape = (height, width, channels)
        elif len(image_shape) == 3:
            _, height, width = image_shape
            mean_shape = (height, width)
        else:
            raise ValueError("Unexpected video shape: expected (frames, H, W[, C])")

        # Accumulators for computing mean and variance
        sum_pixels = np.zeros(mean_shape, dtype=np.float64)
        sum_sq_pixels = np.zeros(mean_shape, dtype=np.float64)
        total_frames = 0

        # Process each video and accumulate statistics
        for key in video_keys:
            # Scale video to prevent overflow during accumulation
            video = infile[key][()].astype(np.float64) / scale_factor
            # Accumulate first moment (sum across time)
            sum_pixels += np.sum(video, axis=0)
            # Accumulate second moment (sum of squares across time)
            sum_sq_pixels += np.sum(np.square(video), axis=0)
            total_frames += video.shape[0]

        # Compute scaled statistics
        means_scaled = sum_pixels / total_frames
        # Variance formula: E[X²] - E[X]²; clamp negative values to 0 for numerical stability
        stds_scaled = np.sqrt(np.maximum((sum_sq_pixels / total_frames) - (means_scaled**2), 0.0))

        # Rescale statistics back to original units
        means = means_scaled * scale_factor
        stds = stds_scaled * scale_factor

        return means, stds


def calcGlobalMeanStd(inpath: str, scale_factor: float = 1000.0) -> Tuple[float, float]:
    """Calculate global mean and standard deviation across all pixels and frames.

    Computes statistics across all frames and pixels in all videos, with optional scaling
    to prevent numerical overflow.

    Args:
        inpath: Path to HDF5 file containing videos of shape (frames, H, W[, C]).
        scale_factor: Scaling factor to divide video values before processing (default: 1000.0).
            Used to prevent overflow during accumulation of large values.

    Returns:
        Tuple of (mean, std):
            - mean: Global scalar mean value, rescaled to original units.
            - std: Global scalar standard deviation value, rescaled to original units.
    """
    with h5py.File(inpath, "r") as infile:
        video_keys = list(infile.keys())
        total_sum = 0.0
        total_sum_sq = 0.0
        total_count = 0

        # Process each video and accumulate global statistics
        for key in video_keys:
            # Scale video to prevent overflow during accumulation
            video = infile[key][()].astype(np.float64) / scale_factor
            # Accumulate first moment
            total_sum += np.sum(video)
            # Accumulate second moment
            total_sum_sq += np.sum(np.square(video))
            # Count total number of elements
            total_count += np.prod(video.shape)

        # Compute scaled statistics
        mean_scaled = total_sum / total_count
        # Variance formula: E[X²] - E[X]²; clamp negative values to 0 for numerical stability
        std_scaled = np.sqrt(max((total_sum_sq / total_count) - (mean_scaled**2), 0.0))

        # Rescale statistics back to original units
        mean = mean_scaled * scale_factor
        std = std_scaled * scale_factor

        return mean, std

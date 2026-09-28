"""Preprocessing module for illumination equalization and contrast normalization.

Implements Contrast Limited Adaptive Histogram Equalization (CLAHE) on the
luminance (L) channel in LAB color space, preserving chromatic fidelity
in a and b channels while compensating for non-uniform lighting conditions.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

def compute_luma_stats(img_rgb: np.ndarray) -> tuple[float, float]:
    """Computes mean and standard deviation of the LAB L channel.

    Args:
        img_rgb: Input RGB image (H, W, 3) in uint8 format.

    Returns:
        (mean, std) of the L channel as float tuple.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0]
    return float(l.mean()), float(l.std())

def match_luminance(
    img_rgb: np.ndarray,
    ref_mean: float,
    ref_std: float,
) -> np.ndarray:
    """Matches the global luminance of an image to a reference statistic pair.

    Reinhard-style linear mapping of the L channel (mean/std normalization):
    removes global exposure differences between captures while preserving
    local structure, color, and any spatially localized illumination artifacts.

    Args:
        img_rgb: Input RGB image (H, W, 3) in uint8 format.
        ref_mean: Target mean of the L channel.
        ref_std: Target standard deviation of the L channel.

    Returns:
        Luminance-matched RGB image (uint8, same dimensions).
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0]
    src_mean, src_std = float(l.mean()), float(l.std()) + 1e-6
    l_matched = np.clip((l - src_mean) * (ref_std / src_std) + ref_mean, 0, 255)
    lab_eq = cv2.merge([l_matched.astype(np.uint8),
                        lab[:, :, 1].astype(np.uint8), lab[:, :, 2].astype(np.uint8)])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

def apply_clahe(
    img_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """Applies CLAHE to the L channel of an RGB image.

    Args:
        img_rgb: Input RGB image (H, W, 3) in uint8 format.
        clip_limit: Contrast threshold for local equalization.
        tile_grid_size: Number of grid subdivisions per axis (rows, cols).

    Returns:
        Equalized RGB image with same dimensions as input.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

def apply_clahe_masked(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """Applies CLAHE confined strictly within the internal region defined by mask.

    Args:
        img_rgb: Input RGB image (H, W, 3) in uint8 format.
        mask: Binary mask (H, W) where values > 0 indicate valid card pixels.
        clip_limit: Contrast threshold for local equalization.
        tile_grid_size: Number of grid subdivisions per axis.

    Returns:
        Equalized RGB image; pixels outside mask remain unchanged.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    l_eq = clahe.apply(l)

    m = mask > 0
    l_out = l.copy()
    l_out[m] = l_eq[m]
    lab_eq = cv2.merge([l_out, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

def clahe_comparison_figure(
    img_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
    title: str = "CLAHE Comparison",
):
    """Builds a 2x2 diagnostic figure displaying CLAHE effects and L-channel histograms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eq = apply_clahe(img_rgb, clip_limit, tile_grid_size)

    l_orig = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
    l_eq = cv2.cvtColor(eq, cv2.COLOR_RGB2LAB)[:, :, 0]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].imshow(img_rgb)
    axs[0, 0].set_title("Original")
    axs[0, 1].imshow(eq)
    axs[0, 1].set_title(f"CLAHE (clip={clip_limit}, tile={tile_grid_size[0]}x{tile_grid_size[1]})")
    axs[1, 0].hist(l_orig.ravel(), bins=256, range=(0, 256), color="#3B9EFF", alpha=0.85)
    axs[1, 0].set_title("L Histogram - Original")
    axs[1, 1].hist(l_eq.ravel(), bins=256, range=(0, 256), color="#00E396", alpha=0.85)
    axs[1, 1].set_title("L Histogram - CLAHE")
    for ax in axs.flat:
        if ax in (axs[0, 0], axs[0, 1]):
            ax.set_xticks([])
        else:
            ax.set_xlim(0, 256)
            ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, eq

def l_channel_rms_contrast(img_rgb: np.ndarray) -> float:
    """Computes RMS contrast on the L channel to quantify illumination homogeneity."""
    l = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
    return float(np.sqrt(np.mean((l - l.mean()) ** 2)))

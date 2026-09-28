"""Module for card perimeter masking and boundary isolation.

Constructs rounded-corner geometric masks to accurately delineate the physical
card surface and suppress background remnants along the warped perimeter.
The mask is utilized to:
1. Confine CLAHE equalization strictly to internal card pixels.
2. Filter patch descriptors in both reference memory bank and query cards.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

def rounded_rect_mask(
    h: int,
    w: int,
    corner_radius_frac: float = 0.035,
    erode_frac: float = 0.02,
    feather_px: int = 5,
) -> np.ndarray:
    """Generates a rounded-rectangle mask for a warped patch of shape (h, w).

    Args:
        h: Patch height in pixels.
        w: Patch width in pixels.
        corner_radius_frac: Corner radius as a fraction of patch width.
        erode_frac: Perimeter erosion factor to eliminate outer boundary artifacts.
        feather_px: Gaussian blur width applied to mask boundary.

    Returns:
        float32 NumPy array with values in [0.0, 1.0], where 1.0 denotes valid card interior.
    """
    mask = np.zeros((h, w), np.uint8)
    radius = int(corner_radius_frac * w)
    erode = int(erode_frac * min(h, w))
    color = 255

    cv2.rectangle(mask, (erode, erode + radius), (w - erode, h - erode - radius), color, -1)
    cv2.rectangle(mask, (erode + radius, erode), (w - erode - radius, h - erode), color, -1)
    for cx, cy in [
        (erode + radius, erode + radius),
        (w - erode - radius, erode + radius),
        (erode + radius, h - erode - radius),
        (w - erode - radius, h - erode - radius),
    ]:
        cv2.circle(mask, (cx, cy), radius, color, -1)

    if feather_px > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather_px)
    return mask.astype(np.float32) / 255.0

def mask_to_grid(mask: np.ndarray, grid_size: tuple) -> np.ndarray:
    """Downsamples pixel-level mask onto the DINOv2 feature patch grid.

    A patch is deemed valid if at least 50% of its area lies within the valid card region.

    Args:
        mask: 2D float mask of shape (H, W).
        grid_size: Tuple (rows, cols) of the feature patch grid.

    Returns:
        2D boolean array of shape specified by grid_size.
    """
    gh, gw = grid_size
    coverage = cv2.resize(mask, (gw, gh), interpolation=cv2.INTER_AREA)
    return coverage >= 0.5

def grid_mask_to_pixel(mask_grid: np.ndarray, out_shape: tuple) -> np.ndarray:
    """Interpolates boolean patch grid mask back to target pixel dimensions."""
    m = mask_grid.astype(np.uint8) * 255
    return cv2.resize(m, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_NEAREST)

def mask_overlay_figure(img_rgb: np.ndarray, mask: np.ndarray, title: str = "Card mask"):
    """Constructs diagnostic figure visualizing perimeter mask applied to card patch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overlay = img_rgb.copy()
    excluded = mask < 0.5
    overlay[excluded] = (overlay[excluded] * 0.35 + np.array([255, 60, 60]) * 0.65).astype(np.uint8)
    border = cv2.Canny((mask > 0.5).astype(np.uint8) * 255, 50, 150)
    overlay[border > 0] = [0, 255, 120]

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.5))
    axs[0].imshow(img_rgb)
    axs[0].set_title("Warped card patch")
    axs[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axs[1].set_title("Rounded-corner mask")
    axs[2].imshow(overlay)
    axs[2].set_title("Excluded region (red)")
    for ax in axs:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig

def sam_refine_mask(
    img_rgb: np.ndarray,
    box: tuple,
    model_type: str = "sam2.1-hiera-small",
) -> Optional[np.ndarray]:
    """Refines geometric card mask using Segment Anything (SAM) model."""
    try:
        from ultralytics import SAM
    except ImportError:
        logger.warning("SAM module not installed; falling back to geometric default mask.")
        return None
    try:
        model = SAM(model_type)
        h, w = img_rgb.shape[:2]
        results = model(img_rgb, bboxes=[list(box)], verbose=False)
        if results and results[0].masks is not None:
            m = results[0].masks.data[0].cpu().numpy()
            return cv2.resize(m.astype(np.float32), (w, h))
    except Exception as e:
        logger.warning("SAM refinement failed: %s", e)
    return None

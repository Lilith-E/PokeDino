"""Module for spatial distance calculation and defect morphological scoring.

Provides two cooperating components for the PokéDINO Baseline pipeline:

1. ``SpatialKNNSearch``: k-NN with relative spatial neighborhood constraint.
   Cosine similarity between a query test patch and reference patches is softly
   penalized when the geometric distance between coordinates (y, x) exceeds
   a threshold radius ``radius_patches``, eliminating cross-region false matches
   (e.g., recurring energy symbols or duplicate glyphs across text boxes).

2. ``MorphologyAnalyzer``: Connected defect segmentation via
   ``cv2.connectedComponentsWithStats``. Extracts morphological features for
   each cluster (area, eccentricity, aspect ratio) and provides an additional
   penalty when linear scratches (eccentricity > threshold) or large clusters
   (area > threshold) are detected.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

@dataclass
class SpatialSearchConfig:
    """Configuration for spatial neighborhood constraints."""

    enabled: bool = True
    radius_patches: float = 3.0
    penalty_weight: float = 0.45

def build_patch_coordinates(grid_size: Tuple[int, int]) -> np.ndarray:
    """Builds (y, x) coordinate matrix for each patch in the DINOv2 feature grid.

    Args:
        grid_size: Tuple (rows, cols) corresponding to the patch grid.

    Returns:
        float32 array of shape (N, 2) where N = rows * cols in row-major order.
    """
    rows, cols = grid_size
    yy, xx = np.meshgrid(np.arange(rows, dtype=np.float32), np.arange(cols, dtype=np.float32), indexing="ij")
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1)
    return coords

def spatial_distance_matrix(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Calculates spatial geometric distance matrix (in patch units) between two sets.

    Args:
        coords_a: Coordinates (N, 2) of first set (e.g. query).
        coords_b: Coordinates (M, 2) of second set (e.g. reference).
        radius: Tolerance radius beyond which penalty applies.

    Returns:
        float32 array (N, M) of Euclidean distances in patch units.
    """
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32)

def spatial_penalty(
    geo_dist: np.ndarray,
    radius: float,
    weight: float,
) -> np.ndarray:
    """Returns soft spatial penalty in [0, weight] based on distance.

    Penalty is 0 for distances <= radius and grows linearly up to ``weight``
    from ``2 * radius``. This ensures:
    - Micro-misalignments (within ``radius``) incur zero penalty;
    - Matches in ``[radius, 2*radius]`` are penalized smoothly;
    - Matches beyond ``2 * radius`` incur maximum penalty.

    Args:
        geo_dist: Matrix (N, M) of geometric distances in patch units.
        radius: Tolerance radius.
        weight: Maximum penalty weight.

    Returns:
        Matrix (N, M) of penalties in [0.0, weight].
    """
    if weight <= 0.0 or radius <= 0.0:
        return np.zeros_like(geo_dist)
    excess = np.maximum(0.0, geo_dist - radius)
    cap = float(radius)
    return weight * np.clip(excess / cap, 0.0, 1.0)

def compute_spatial_knn_distances(
    test_features: torch.Tensor,
    ref_features: torch.Tensor,
    coords_test: np.ndarray,
    coords_ref: np.ndarray,
    config: SpatialSearchConfig,
) -> np.ndarray:
    """Computes anomaly distances under spatial constraints.

    Cosine similarity minus spatial distance penalty. The maximum effective
    similarity (similarity - penalty) across valid reference patches is selected
    for each test patch.

    Args:
        test_features: Tensor (N, D) of L2-normalized query features.
        ref_features: Tensor (M, D) of L2-normalized reference features.
        coords_test: Coordinates (N, 2) of query patches.
        coords_ref: Coordinates (M, 2) of reference patches.
        config: Spatial search configuration.

    Returns:
        float32 numpy array (N,) of (1 - max_effective_similarity) clamped to [0, 1].
    """
    sim = torch.mm(test_features, ref_features.T).cpu().numpy().astype(np.float32)

    if not config.enabled or config.penalty_weight <= 0.0:
        max_sim = sim.max(axis=1)
        return (1.0 - max_sim).clip(min=0.0).astype(np.float32)

    geo = spatial_distance_matrix(coords_test, coords_ref, config.radius_patches)
    pen = spatial_penalty(geo, config.radius_patches, config.penalty_weight)
    eff_sim = sim - pen
    max_eff_sim = eff_sim.max(axis=1)
    return (1.0 - max_eff_sim).clip(min=0.0).astype(np.float32)

@dataclass
class MorphologyConfig:
    """Configuration for morphological defect analysis."""

    enabled: bool = True
    large_cluster_threshold: int = 15
    cluster_penalty: float = 0.025
    linear_eccentricity: float = 3.0
    linear_penalty: float = 0.04

@dataclass
class DefectCluster:
    """Statistics for a single connected defect cluster."""

    label: int
    area: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    eccentricity: float
    aspect_ratio: float

@dataclass
class MorphologyResult:
    """Overall outcome of morphological defect analysis."""

    cluster_count: int = 0
    max_cluster_area: int = 0
    mean_cluster_area: float = 0.0
    linear_cluster_count: int = 0
    total_defect_patches: int = 0
    morphology_bonus: float = 0.0
    clusters: list = field(default_factory=list)

def _eccentricity_from_stats(area: int, w: int, h: int) -> Tuple[float, float]:
    """Approximates eccentricity and aspect ratio given blob area and bounding box.

    For small clusters (<= 4 patches), aspect ratio is not statistically reliable
    and defaults to 1.0 (square). Otherwise max(w,h)/min(w,h) is returned as a
    linearity proxy. Values >= 3.0 indicate linear scratch defects.
    """
    if w <= 0 or h <= 0 or area <= 1:
        return 1.0, 1.0
    aspect = float(max(w, h)) / float(max(1, min(w, h)))
    if area < 5:
        return 1.0, aspect
    return aspect, aspect

def analyze_defect_morphology(
    d_map: np.ndarray,
    grid_mask: np.ndarray,
    defect_threshold: float,
    config: MorphologyConfig,
) -> MorphologyResult:
    """Analyzes morphological structure of anomalous patches.

    Args:
        d_map: 2D anomaly distance map (float32).
        grid_mask: 2D boolean mask of valid interior card patches.
        defect_threshold: Distance threshold beyond which a patch is anomalous.
        config: Configuration of morphological weights and thresholds.

    Returns:
        MorphologyResult containing cluster_count, max_cluster_area, linear_cluster_count,
        and morphology_bonus to be added to the final score.
    """
    empty = MorphologyResult()
    if not config.enabled:
        return empty

    gh, gw = d_map.shape
    bin_mask = (d_map > defect_threshold) & grid_mask
    bin_mask_u8 = (bin_mask.astype(np.uint8)) * 255

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask_u8, connectivity=8)
    clusters: list[DefectCluster] = []
    total_defect = 0

    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        total_defect += area
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        cx = x + w / 2.0
        cy = y + h / 2.0
        ecc, ar = _eccentricity_from_stats(area, w, h)
        clusters.append(
            DefectCluster(
                label=lbl,
                area=area,
                centroid=(cx, cy),
                bbox=(x, y, w, h),
                eccentricity=ecc,
                aspect_ratio=ar,
            )
        )

    if not clusters:
        return empty

    max_area = max(c.area for c in clusters)
    linear_count = sum(1 for c in clusters if c.eccentricity >= config.linear_eccentricity)
    mean_area = float(np.mean([c.area for c in clusters]))

    bonus = 0.0
    if max_area > config.large_cluster_threshold:
        bonus += (max_area - config.large_cluster_threshold) * config.cluster_penalty
    if linear_count > 0:
        bonus += linear_count * config.linear_penalty

    return MorphologyResult(
        cluster_count=len(clusters),
        max_cluster_area=max_area,
        mean_cluster_area=mean_area,
        linear_cluster_count=linear_count,
        total_defect_patches=total_defect,
        morphology_bonus=float(bonus),
        clusters=clusters,
    )
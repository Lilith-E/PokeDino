"""Perspective rectification and dynamic mask warping from segmentation masks.

Converts a binary or soft pixel-level card segmentation mask into a canonical
upright RGB card patch and a matching warped validity mask by fitting four
straight edge lines with outlier rejection and intersecting them to recover
the unrounded perspective quadrilateral.
"""

from dataclasses import dataclass
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TARGET_H = 630
TARGET_W = 448
TARGET_AR = 88.0 / 63.0


@dataclass
class RectifiedCard:
    """Output of perspective rectification from a segmentation mask."""

    corners: np.ndarray
    patch_rgb: np.ndarray
    warped_mask: np.ndarray
    sharpness: float


def order_quad(quad: np.ndarray) -> np.ndarray:
    """Order quadrilateral vertices cyclically as [TL, TR, BR, BL] in portrait view."""
    q = np.asarray(quad, dtype=np.float64)
    ctr = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - ctr[1], q[:, 0] - ctr[0])
    q = q[np.argsort(ang)]

    candidates = []
    for i in range(4):
        r = np.roll(q, -i, axis=0)
        if np.linalg.norm(r[1] - r[0]) < np.linalg.norm(r[3] - r[0]):
            mid_y = (r[0, 1] + r[1, 1]) / 2.0
            candidates.append((mid_y, r))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    return q


def _order_cyclic_ccw(pts: np.ndarray) -> np.ndarray:
    """Order 4 points cyclically around their centroid."""
    q = np.asarray(pts, dtype=np.float64)
    ctr = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - ctr[1], q[:, 0] - ctr[0])
    return q[np.argsort(ang)]


def _fit_line_huber(pts: np.ndarray) -> Optional[tuple[float, float, float]]:
    """Fit normalized 2D line (A, B, C) with A*x + B*y + C = 0 via Huber M-estimator."""
    if len(pts) < 8:
        return None
    line = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
    vx, vy, x0, y0 = [float(v) for v in line.flatten()]
    a = -vy
    b = vx
    c = vy * x0 - vx * y0
    norm = float(np.hypot(a, b))
    if norm < 1e-8:
        return None
    return (a / norm, b / norm, c / norm)


def _fit_line_ransac(
    pts: np.ndarray,
    thresh_px: float = 2.5,
    n_iters: int = 120,
) -> Optional[tuple[float, float, float]]:
    """Fit normalized 2D line (A, B, C) via deterministic RANSAC + Huber polish."""
    n = len(pts)
    if n < 8:
        return None

    pts_f = np.asarray(pts, dtype=np.float64)
    rng = np.random.default_rng(0)
    best_inliers: Optional[np.ndarray] = None
    best_count = -1
    best_err = float("inf")

    for _ in range(n_iters):
        idx = rng.choice(n, size=2, replace=False)
        p1, p2 = pts_f[idx[0]], pts_f[idx[1]]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = float(np.hypot(dx, dy))
        if length < 1e-3:
            continue
        a = -dy / length
        b = dx / length
        c = -(a * p1[0] + b * p1[1])

        dists = np.abs(pts_f[:, 0] * a + pts_f[:, 1] * b + c)
        inliers = dists <= thresh_px
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_err = float(dists[inliers].mean()) if count > 0 else float("inf")
        elif count == best_count and count > 0:
            err = float(dists[inliers].mean())
            if err < best_err:
                best_inliers = inliers
                best_err = err

    if best_inliers is None or int(best_inliers.sum()) < max(6, n // 4):
        return _fit_line_huber(pts_f)

    return _fit_line_huber(pts_f[best_inliers])


def _intersect_lines(
    l1: Optional[tuple[float, float, float]],
    l2: Optional[tuple[float, float, float]],
) -> Optional[np.ndarray]:
    """Compute 2D intersection of two normalized lines."""
    if l1 is None or l2 is None:
        return None
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 0.15:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return np.array([x, y], dtype=np.float64)


def _partition_contour_sides(
    contour_pts: np.ndarray,
    box_pts: np.ndarray,
    corner_excl_frac: float,
    max_perp_frac: float = 0.15,
) -> list[np.ndarray]:
    """Assign contour points to the 4 sides of an initial bounding quad.

    Points projecting within ``corner_excl_frac`` of either endpoint of an edge
    are excluded so rounded corner arcs do not bias line estimation. Points
    farther inward than ``max_perp_frac`` of the card short side (e.g., deep
    occlusion notches) are also discarded before line fitting.
    """
    ordered = _order_cyclic_ccw(box_pts)
    edge_lengths = [float(np.linalg.norm(ordered[(i + 1) % 4] - ordered[i])) for i in range(4)]
    short_side = max(min(edge_lengths), 1.0)
    max_perp_dist = max_perp_frac * short_side

    n_pts = len(contour_pts)
    dists = np.full((n_pts, 4), np.inf, dtype=np.float64)
    ts = np.zeros((n_pts, 4), dtype=np.float64)

    for i in range(4):
        a = ordered[i]
        edge = ordered[(i + 1) % 4] - a
        denom = float(np.dot(edge, edge))
        if denom <= 1e-6:
            continue
        rel = contour_pts - a
        t = np.sum(rel * edge[None, :], axis=1) / denom
        t_clamped = np.clip(t, 0.0, 1.0)
        proj = a + t_clamped[:, None] * edge
        dists[:, i] = np.linalg.norm(contour_pts - proj, axis=1)
        ts[:, i] = t

    nearest_side = np.argmin(dists, axis=1)
    min_dist = dists[np.arange(n_pts), nearest_side]

    sides: list[np.ndarray] = []
    lo = float(np.clip(corner_excl_frac, 0.02, 0.35))
    hi = 1.0 - lo
    for i in range(4):
        t_side = ts[:, i]
        sel = (
            (nearest_side == i)
            & (t_side >= lo)
            & (t_side <= hi)
            & (min_dist <= max_perp_dist)
        )
        sides.append(contour_pts[sel])
    return sides


def fit_quad_from_mask(
    mask: np.ndarray,
    edge_fit_method: str = "ransac",
    corner_excl_frac: float = 0.14,
    min_side_inliers: int = 12,
    ransac_thresh_px: float = 2.5,
    ransac_iters: int = 120,
) -> Optional[np.ndarray]:
    """Recover the 4 perspective card corners [TL, TR, BR, BL] from a binary/soft mask.

    Falls back to the minimum-area bounding rectangle if an edge lacks sufficient
    inliers (e.g., severe occlusion along an entire border).
    """
    if mask is None or mask.size == 0:
        return None

    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k_close)

    cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return None

    rect = cv2.minAreaRect(largest)
    rw, rh = rect[1]
    if min(rw, rh) < 20:
        return None
    init_box = cv2.boxPoints(rect).astype(np.float64)

    short_side = max(min(rw, rh), 1.0)
    eff_thresh = max(float(ransac_thresh_px), 0.005 * short_side)

    contour_pts = largest.reshape(-1, 2).astype(np.float64)
    sides = _partition_contour_sides(contour_pts, init_box, corner_excl_frac=corner_excl_frac)

    lines: list[Optional[tuple[float, float, float]]] = []
    for side_pts in sides:
        if len(side_pts) < min_side_inliers:
            lines.append(None)
            continue
        if edge_fit_method == "huber":
            lines.append(_fit_line_huber(side_pts))
        else:
            lines.append(_fit_line_ransac(side_pts, thresh_px=eff_thresh, n_iters=ransac_iters))

    if any(line is None for line in lines):
        return order_quad(init_box)

    corners = []
    for i in range(4):
        pt = _intersect_lines(lines[i], lines[(i + 1) % 4])
        if pt is None or not np.isfinite(pt).all():
            return order_quad(init_box)
        corners.append(pt)
    quad = np.stack(corners, axis=0)

    rect_fit = cv2.minAreaRect(quad.astype(np.float32))
    fw, fh = rect_fit[1]
    if min(fw, fh) < 20:
        return order_quad(init_box)

    ar = max(fw, fh) / max(min(fw, fh), 1e-3)
    if not (1.15 <= ar <= 1.65):
        return order_quad(init_box)

    h_img, w_img = mask.shape[:2]
    max_dim = float(max(h_img, w_img))
    if np.any(quad < -0.25 * max_dim) or np.any(quad[:, 0] > w_img + 0.25 * max_dim) or np.any(quad[:, 1] > h_img + 0.25 * max_dim):
        return order_quad(init_box)

    return order_quad(quad)


def _refine_quad_from_image_edges(
    img_bgr: np.ndarray,
    quad: np.ndarray,
    mask: np.ndarray,
    target_size: tuple[int, int] = (TARGET_H, TARGET_W),
    pad: int = 36,
    band_out: int = 16,
    band_in: int = 14,
    n_samples: int = 72,
) -> np.ndarray:
    """Snap segmentation-derived card edges to full-resolution physical image gradients.

    Warps both the BGR scene and segmentation mask into a padded canonical portrait
    frame, then searches within a tight normal band around each card edge using fused
    directional Sobel + CIE-Lab chromatic gradients gated by mask continuity. Falls back
    safely to the input quad if refined edges fail geometric sanity checks.
    """
    q = order_quad(quad)
    target_h, target_w = target_size
    w_pad = target_w + 2 * pad
    h_pad = target_h + 2 * pad
    dst_pad = np.array(
        [
            [pad, pad],
            [pad + target_w - 1, pad],
            [pad + target_w - 1, pad + target_h - 1],
            [pad, pad + target_h - 1],
        ],
        dtype=np.float32,
    )
    try:
        M_pad = cv2.getPerspectiveTransform(q.astype(np.float32), dst_pad)
        M_inv = cv2.getPerspectiveTransform(dst_pad, q.astype(np.float32))
    except cv2.error:
        return q

    patch_bgr = cv2.warpPerspective(
        img_bgr, M_pad, (w_pad, h_pad), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    mask_f = (mask > 0.5).astype(np.float32)
    mask_pad = cv2.warpPerspective(
        mask_f, M_pad, (w_pad, h_pad), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )

    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_blur = cv2.GaussianBlur(lab[:, :, 0], (3, 3), 0)
    a_blur = cv2.GaussianBlur(lab[:, :, 1], (3, 3), 0)
    b_blur = cv2.GaussianBlur(lab[:, :, 2], (3, 3), 0)

    # Fused directional gradient magnitude across L* and chromatic a*,b* channels
    gy = (
        np.abs(cv2.Sobel(L_blur, cv2.CV_32F, 0, 1, ksize=3))
        + 0.7 * np.abs(cv2.Sobel(a_blur, cv2.CV_32F, 0, 1, ksize=3))
        + 0.7 * np.abs(cv2.Sobel(b_blur, cv2.CV_32F, 0, 1, ksize=3))
    )
    gx = (
        np.abs(cv2.Sobel(L_blur, cv2.CV_32F, 1, 0, ksize=3))
        + 0.7 * np.abs(cv2.Sobel(a_blur, cv2.CV_32F, 1, 0, ksize=3))
        + 0.7 * np.abs(cv2.Sobel(b_blur, cv2.CV_32F, 1, 0, ksize=3))
    )

    def _sample_side(mag: np.ndarray, expected: int, axis: str, s_out: int, s_in: int) -> np.ndarray:
        pts = []
        for t in np.linspace(0.14, 0.86, n_samples):
            if axis == "h":
                x = int(pad + t * (target_w - 1))
                y_lo = max(2, expected - s_out)
                y_hi = min(h_pad - 3, expected + s_in)
                # Require solid interior mask near this sample so occluded cutouts are skipped
                inner_y = min(h_pad - 1, max(0, expected + (18 if expected < h_pad // 2 else -18)))
                if mask_pad[inner_y, x] < 0.5:
                    continue
                col = mag[y_lo:y_hi, x]
                if len(col) > 0 and float(col.max()) > 18.0:
                    pts.append([float(x), float(y_lo + int(np.argmax(col)))])
            else:
                y = int(pad + t * (target_h - 1))
                x_lo = max(2, expected - s_out)
                x_hi = min(w_pad - 3, expected + s_in)
                inner_x = min(w_pad - 1, max(0, expected + (18 if expected < w_pad // 2 else -18)))
                if mask_pad[y, inner_x] < 0.5:
                    continue
                row = mag[y, x_lo:x_hi]
                if len(row) > 0 and float(row.max()) > 18.0:
                    pts.append([float(x_lo + int(np.argmax(row))), float(y)])
        return np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 2), dtype=np.float64)

    def _mad_clean(pts: np.ndarray, axis_idx: int) -> np.ndarray:
        if len(pts) < 10:
            return pts
        vals = pts[:, axis_idx]
        med = float(np.median(vals))
        mad = max(float(np.median(np.abs(vals - med))), 1.0)
        return pts[np.abs(vals - med) <= 3.0 * mad]

    top_pts = _mad_clean(_sample_side(gy, pad, "h", band_out, band_in), 1)
    bot_pts = _mad_clean(_sample_side(gy, pad + target_h - 1, "h", band_in, band_out), 1)
    left_pts = _mad_clean(_sample_side(gx, pad, "v", band_out, band_in), 0)
    right_pts = _mad_clean(_sample_side(gx, pad + target_w - 1, "v", band_in, band_out), 0)

    def _canonical_line(axis: str, pos: int) -> tuple[float, float, float]:
        # A*x + B*y + C = 0
        return (0.0, 1.0, -float(pos)) if axis == "h" else (1.0, 0.0, -float(pos))

    def _fit_or_default(pts: np.ndarray, axis: str, default_pos: int, max_tilt_rad: float = 0.14) -> tuple[float, float, float]:
        line = _fit_line_huber(pts) if len(pts) >= 14 else None
        if line is None:
            return _canonical_line(axis, default_pos)
        a, b, _ = line
        # Guard slope orientation against expected canonical border
        if axis == "h" and abs(a) > max_tilt_rad:
            return _canonical_line(axis, default_pos)
        if axis == "v" and abs(b) > max_tilt_rad:
            return _canonical_line(axis, default_pos)
        return line

    l_top = _fit_or_default(top_pts, "h", pad)
    l_right = _fit_or_default(right_pts, "v", pad + target_w - 1)
    l_bot = _fit_or_default(bot_pts, "h", pad + target_h - 1)
    l_left = _fit_or_default(left_pts, "v", pad)

    corners_pad = []
    for l1, l2 in ((l_top, l_left), (l_top, l_right), (l_bot, l_right), (l_bot, l_left)):
        pt = _intersect_lines(l1, l2)
        if pt is None or not np.isfinite(pt).all():
            return q
        corners_pad.append(pt)

    refined_pad = np.asarray(corners_pad, dtype=np.float32).reshape(1, 4, 2)
    refined_scene = cv2.perspectiveTransform(refined_pad, M_inv).reshape(4, 2).astype(np.float64)

    # Sanity check: reject unstable shifts or distorted aspect ratio
    rect_ref = cv2.minAreaRect(refined_scene.astype(np.float32))
    rw, rh = rect_ref[1]
    if min(rw, rh) < 20:
        return q
    ar = max(rw, rh) / max(min(rw, rh), 1e-3)
    if not (1.20 <= ar <= 1.60):
        return q

    shift_mean = float(np.linalg.norm(refined_scene - q, axis=1).mean())
    if shift_mean > max(16.0, 0.04 * max(rw, rh)):
        return q

    return order_quad(refined_scene)


def rectify_from_mask(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    target_size: tuple[int, int] = (TARGET_H, TARGET_W),
    edge_fit_method: str = "ransac",
    corner_excl_frac: float = 0.14,
    min_side_inliers: int = 12,
    ransac_thresh_px: float = 2.5,
    ransac_iters: int = 120,
    erode_mask_px: int = 2,
    feather_px: int = 3,
    refine_edges: bool = True,
) -> Optional[RectifiedCard]:
    """Rectify an image and its segmentation mask into canonical upright coordinates."""
    quad = fit_quad_from_mask(
        mask,
        edge_fit_method=edge_fit_method,
        corner_excl_frac=corner_excl_frac,
        min_side_inliers=min_side_inliers,
        ransac_thresh_px=ransac_thresh_px,
        ransac_iters=ransac_iters,
    )
    if quad is None:
        return None
    if refine_edges:
        quad = _refine_quad_from_image_edges(img_bgr, quad, mask, target_size=target_size)

    rect = cv2.minAreaRect(quad.astype(np.float32))
    rw, rh = rect[1]
    ar = max(rw, rh) / max(min(rw, rh), 1e-3)
    if not (1.15 <= ar <= 1.65):
        return None

    target_h, target_w = target_size
    dst = np.array(
        [
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped_bgr = cv2.warpPerspective(img_bgr, M, (target_w, target_h), flags=cv2.INTER_LANCZOS4)
    if warped_bgr is None or warped_bgr.mean() < 1.0:
        return None

    mask_f32 = mask.astype(np.float32)
    if mask_f32.max() > 1.0:
        mask_f32 = mask_f32 / 255.0
    warped_mask = cv2.warpPerspective(mask_f32, M, (target_w, target_h), flags=cv2.INTER_LINEAR)
    warped_mask = (warped_mask >= 0.5).astype(np.float32)

    if erode_mask_px > 0:
        k = erode_mask_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        warped_mask = cv2.erode(warped_mask, kernel)

    if feather_px > 0:
        warped_mask = cv2.GaussianBlur(warped_mask, (0, 0), float(feather_px))
    warped_mask = np.clip(warped_mask, 0.0, 1.0).astype(np.float32)

    patch_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return RectifiedCard(
        corners=quad,
        patch_rgb=patch_rgb,
        warped_mask=warped_mask,
        sharpness=sharpness,
    )

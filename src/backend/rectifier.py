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

CANON_W = 544
CANON_H = 768
CANON_PAD_X = 48
CANON_PAD_Y = 69


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


def _line_from_segment(p1: np.ndarray, p2: np.ndarray) -> Optional[tuple[float, float, float]]:
    """Construct normalized line (A, B, C) passing through segment endpoints p1 -> p2."""
    dx, dy = float(p2[0] - p1[0]), float(p2[1] - p1[1])
    length = float(np.hypot(dx, dy))
    if length < 1e-6:
        return None
    a = -dy / length
    b = dx / length
    c = -(a * float(p1[0]) + b * float(p1[1]))
    return (a, b, c)


def _angle_diff_sin(l1: tuple[float, float, float], l2: tuple[float, float, float]) -> float:
    """Return |sin(theta)| between two unit-normalized lines (orientation sign invariant)."""
    return abs(l1[0] * l2[1] - l1[1] * l2[0])


def _side_inlier_span_frac(
    pts: np.ndarray,
    line: tuple[float, float, float],
    seg_p1: np.ndarray,
    seg_p2: np.ndarray,
    thresh_px: float,
) -> float:
    """Compute normalized span [0, 1] along segment seg_p1->seg_p2 covered by line inliers."""
    if len(pts) == 0:
        return 0.0
    if not (np.all(np.isfinite(seg_p1)) and np.all(np.isfinite(seg_p2))):
        return 0.0
    edge = np.asarray(seg_p2, dtype=np.float64) - np.asarray(seg_p1, dtype=np.float64)
    seg_len = float(np.linalg.norm(edge))
    if not np.isfinite(seg_len) or seg_len < 1e-6 or seg_len > 1e7:
        return 0.0
    a, b, c = line
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
        return 0.0
    pts_f = np.asarray(pts, dtype=np.float64)
    valid = np.all(np.isfinite(pts_f), axis=1)
    if not np.any(valid):
        return 0.0
    pts_f = pts_f[valid]
    dists = np.abs(pts_f[:, 0] * a + pts_f[:, 1] * b + c)
    inliers = pts_f[dists <= thresh_px]
    if len(inliers) < 2:
        return 0.0
    u = edge / seg_len
    proj = np.sum((inliers - np.asarray(seg_p1, dtype=np.float64)) * u[None, :], axis=1)
    t = np.clip(proj / seg_len, 0.0, 1.0)
    return float(t.max() - t.min())


def _subpixel_peak_1d(vals: np.ndarray, idx: int) -> float:
    """Quadratic 3-point sub-pixel refinement around discrete maximum index ``idx``."""
    n = len(vals)
    if idx <= 0 or idx >= n - 1:
        return float(idx)
    yl, yc, yr = float(vals[idx - 1]), float(vals[idx]), float(vals[idx + 1])
    denom = yl - 2.0 * yc + yr
    if denom >= -1e-6:
        return float(idx)
    delta = float(np.clip(0.5 * (yl - yr) / denom, -0.5, 0.5))
    return float(idx) + delta


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


def _is_convex_quad(quad: np.ndarray) -> bool:
    """Return True if ordered vertices form a simple convex quadrilateral."""
    q = np.asarray(quad, dtype=np.float64)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        return False
    signs = []
    for i in range(4):
        e1 = q[(i + 1) % 4] - q[i]
        e2 = q[(i + 2) % 4] - q[(i + 1) % 4]
        cross = float(e1[0] * e2[1] - e1[1] * e2[0])
        if abs(cross) < 1e-4:
            return False
        signs.append(cross > 0.0)
    return all(signs) or not any(signs)


def _opposite_taper_ok(quad: np.ndarray, max_ratio: float = 1.32) -> bool:
    """Check that opposite side lengths stay within realistic planar perspective taper."""
    q = order_quad(quad)
    w_top = float(np.linalg.norm(q[1] - q[0]))
    w_bot = float(np.linalg.norm(q[2] - q[3]))
    h_left = float(np.linalg.norm(q[3] - q[0]))
    h_right = float(np.linalg.norm(q[2] - q[1]))
    if min(w_top, w_bot, h_left, h_right) < 10.0:
        return False
    rw = max(w_top, w_bot) / min(w_top, w_bot)
    rh = max(h_left, h_right) / min(h_left, h_right)
    return rw <= max_ratio and rh <= max_ratio


def _partition_contour_sides(
    contour_pts: np.ndarray,
    box_pts: np.ndarray,
    corner_excl_frac: float,
    max_perp_frac: float = 0.11,
) -> list[np.ndarray]:
    """Assign contour points to the 4 sides of an initial bounding quad.

    Points projecting within ``corner_excl_frac`` of either endpoint of an edge
    are excluded so rounded corner arcs do not bias line estimation. Points
    farther inward than ``max_perp_frac`` of the card short side (e.g., deep
    occlusion notches) are also discarded before line fitting.
    """
    ordered = order_quad(box_pts)
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


def _sample_soft_mask_quad_sides(
    mask_f32: np.ndarray,
    ordered_box: np.ndarray,
    corner_excl_frac: float = 0.14,
    n_rays: int = 56,
    scan_out_px: Optional[int] = None,
    scan_in_px: Optional[int] = None,
) -> list[np.ndarray]:
    """Extract sub-pixel 0.5-isosurface crossings along outward-to-inward rays of a quad."""
    h_img, w_img = mask_f32.shape[:2]
    edge_lengths = [float(np.linalg.norm(ordered_box[(i + 1) % 4] - ordered_box[i])) for i in range(4)]
    short_side = max(min(edge_lengths), 1.0)
    s_out = int(scan_out_px if scan_out_px is not None else max(10, round(0.05 * short_side)))
    s_in = int(scan_in_px if scan_in_px is not None else max(16, round(0.09 * short_side)))
    ctr = ordered_box.mean(axis=0)
    lo = float(np.clip(corner_excl_frac, 0.04, 0.35))
    ts = np.linspace(lo, 1.0 - lo, n_rays, dtype=np.float64)
    offsets = np.arange(-s_out, s_in + 1, dtype=np.float64)
    n_steps = len(offsets)

    sides: list[np.ndarray] = []
    for i in range(4):
        p1 = ordered_box[i]
        p2 = ordered_box[(i + 1) % 4]
        edge = p2 - p1
        elen = float(np.linalg.norm(edge))
        if elen < 10.0 or n_steps < 12:
            sides.append(np.empty((0, 2), dtype=np.float64))
            continue
        tx, ty = edge[0] / elen, edge[1] / elen
        # Choose unit normal pointing strictly toward quad centroid
        n_cand = np.array([-ty, tx], dtype=np.float64)
        mid = 0.5 * (p1 + p2)
        if float(np.dot(ctr - mid, n_cand)) < 0.0:
            n_cand = -n_cand

        bases = p1[None, :] + ts[:, None] * edge[None, :]
        grid_x = (bases[:, 0:1] + offsets[None, :] * n_cand[0]).astype(np.float32)
        grid_y = (bases[:, 1:2] + offsets[None, :] * n_cand[1]).astype(np.float32)
        sampled = cv2.remap(
            mask_f32,
            grid_x,
            grid_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        pts_side: list[list[float]] = []
        for r_idx in range(n_rays):
            vals = sampled[r_idx]
            for j in range(1, n_steps - 6):
                if vals[j - 1] < 0.5 <= vals[j]:
                    if float(np.mean(vals[j : min(n_steps, j + 7)])) >= 0.56:
                        denom = max(float(vals[j] - vals[j - 1]), 1e-5)
                        frac = (0.5 - float(vals[j - 1])) / denom
                        off = float(offsets[j - 1] + frac)
                        px = float(bases[r_idx, 0] + off * n_cand[0])
                        py = float(bases[r_idx, 1] + off * n_cand[1])
                        if -8.0 <= px <= w_img + 8.0 and -8.0 <= py <= h_img + 8.0:
                            pts_side.append([px, py])
                        break
        sides.append(np.asarray(pts_side, dtype=np.float64) if pts_side else np.empty((0, 2), dtype=np.float64))
    return sides


def _refine_contour_points_subpixel(
    mask_f32: np.ndarray,
    pts: np.ndarray,
    p_start: np.ndarray,
    p_end: np.ndarray,
    centroid: np.ndarray,
) -> np.ndarray:
    """Shift quantized >=0.5 contour coordinates onto continuous 0.5-isosurface along edge normal."""
    if len(pts) == 0:
        return pts
    edge = p_end - p_start
    elen = float(np.linalg.norm(edge))
    if elen < 10.0:
        return pts
    n_in = np.array([-edge[1] / elen, edge[0] / elen], dtype=np.float64)
    if float(np.dot(centroid - 0.5 * (p_start + p_end), n_in)) < 0.0:
        n_in = -n_in

    offsets = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    gx = (pts[:, 0:1] + offsets[None, :] * n_in[0]).astype(np.float32)
    gy = (pts[:, 1:2] + offsets[None, :] * n_in[1]).astype(np.float32)
    vals = cv2.remap(
        mask_f32,
        gx,
        gy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    out = pts.astype(np.float64, copy=True)
    for k in range(len(pts)):
        v0, v1, v2 = float(vals[k, 0]), float(vals[k, 1]), float(vals[k, 2])
        if v0 < 0.5 <= v1:
            frac = (0.5 - v0) / max(v1 - v0, 1e-5)
            shift = -1.0 + frac
        elif v1 < 0.5 <= v2:
            frac = (0.5 - v1) / max(v2 - v1, 1e-5)
            shift = frac
        else:
            continue
        out[k, 0] += shift * n_in[0]
        out[k, 1] += shift * n_in[1]
    return out


def fit_quad_from_mask(
    mask: np.ndarray,
    edge_fit_method: str = "ransac",
    corner_excl_frac: float = 0.14,
    min_side_inliers: int = 12,
    ransac_thresh_px: float = 2.5,
    ransac_iters: int = 120,
    min_span_frac: float = 0.30,
    max_side_tilt_rad: float = 0.20,
) -> Optional[np.ndarray]:
    """Recover the 4 perspective card corners [TL, TR, BR, BL] from a binary/soft mask.

    Refines dense contour pixels onto continuous 0.5-isosurface crossings along local
    edge normals, enforces per-side tilt and support-span bounds, and recovers partial
    or occluded sides by locking the opposing edge's angle and solving for 1D offset.
    """
    if mask is None or mask.size == 0:
        return None

    mask_f32 = mask.astype(np.float32)
    if mask_f32.max() > 1.0:
        mask_f32 = mask_f32 / 255.0
    mask_f32 = np.clip(mask_f32, 0.0, 1.0)

    mask_u8 = (mask_f32 >= 0.5).astype(np.uint8) * 255
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
    ordered_init = order_quad(init_box)
    ctr = ordered_init.mean(axis=0)

    short_side = max(min(rw, rh), 1.0)
    eff_thresh = max(float(ransac_thresh_px), 0.005 * short_side)

    contour_pts = largest.reshape(-1, 2).astype(np.float64)
    contour_sides = _partition_contour_sides(contour_pts, ordered_init, corner_excl_frac=corner_excl_frac)

    has_soft = bool(np.any((mask_f32 > 0.04) & (mask_f32 < 0.96)))
    side_pts_list: list[np.ndarray] = []
    for i in range(4):
        pts_i = contour_sides[i]
        if has_soft and len(pts_i) > 0:
            pts_i = _refine_contour_points_subpixel(
                mask_f32, pts_i, ordered_init[i], ordered_init[(i + 1) % 4], ctr
            )
        side_pts_list.append(pts_i)

    sin_max_tilt = float(np.sin(max_side_tilt_rad))
    init_lines = [_line_from_segment(ordered_init[i], ordered_init[(i + 1) % 4]) for i in range(4)]

    fitted_lines: list[Optional[tuple[float, float, float]]] = [None] * 4
    spans: list[float] = [0.0] * 4

    for i in range(4):
        pts = side_pts_list[i]
        if len(pts) < min_side_inliers:
            continue
        if edge_fit_method == "huber":
            line = _fit_line_huber(pts)
        else:
            line = _fit_line_ransac(pts, thresh_px=eff_thresh, n_iters=ransac_iters)
        if line is None:
            continue

        span = _side_inlier_span_frac(
            pts,
            line,
            ordered_init[i],
            ordered_init[(i + 1) % 4],
            thresh_px=eff_thresh * 1.25,
        )
        base_line = init_lines[i]
        if base_line is not None and _angle_diff_sin(line, base_line) > sin_max_tilt:
            continue
        if span < min_span_frac:
            continue

        fitted_lines[i] = line
        spans[i] = span

    def _recover_side(idx: int) -> Optional[tuple[float, float, float]]:
        opp = fitted_lines[(idx + 2) % 4]
        if opp is not None:
            a, b, _ = opp
            pts_i = side_pts_list[idx]
            if len(pts_i) >= 8:
                # 1D robust offset recovery anchored near the initial box edge
                base_l = init_lines[idx]
                if base_l is not None:
                    d_init = np.abs(pts_i[:, 0] * base_l[0] + pts_i[:, 1] * base_l[1] + base_l[2])
                    pts_near = pts_i[d_init <= max(eff_thresh * 4.0, 0.04 * short_side)]
                    if len(pts_near) >= 6:
                        c_opt = -float(np.median(a * pts_near[:, 0] + b * pts_near[:, 1]))
                        return (a, b, c_opt)
            mid = 0.5 * (ordered_init[idx] + ordered_init[(idx + 1) % 4])
            return (a, b, -(a * float(mid[0]) + b * float(mid[1])))
        return init_lines[idx]

    resolved_lines: list[Optional[tuple[float, float, float]]] = [
        fitted_lines[i] if fitted_lines[i] is not None else _recover_side(i)
        for i in range(4)
    ]
    if any(line is None for line in resolved_lines):
        return ordered_init

    def _intersect_quad(lines_4: list[Optional[tuple[float, float, float]]]) -> Optional[np.ndarray]:
        pts_c = []
        for idx_l in range(4):
            pt = _intersect_lines(lines_4[(idx_l - 1) % 4], lines_4[idx_l])
            if pt is None or not np.isfinite(pt).all():
                return None
            pts_c.append(pt)
        return np.stack(pts_c, axis=0)

    quad = _intersect_quad(resolved_lines)
    if quad is None:
        return ordered_init

    # If opposite-side taper became extreme due to one weak side, recover that weakest side.
    if not _opposite_taper_ok(quad, max_ratio=1.30):
        valid_indices = [i for i in range(4) if fitted_lines[i] is not None]
        if valid_indices:
            weakest = min(valid_indices, key=lambda idx: spans[idx])
            fitted_lines[weakest] = None
            resolved_lines[weakest] = _recover_side(weakest)
            quad_repaired = _intersect_quad(resolved_lines)
            if quad_repaired is not None and _opposite_taper_ok(quad_repaired, max_ratio=1.30):
                quad = quad_repaired
            else:
                return ordered_init
        else:
            return ordered_init

    if not _is_convex_quad(quad):
        return ordered_init

    rect_fit = cv2.minAreaRect(quad.astype(np.float32))
    fw, fh = rect_fit[1]
    if min(fw, fh) < 20:
        return ordered_init

    ar = max(fw, fh) / max(min(fw, fh), 1e-3)
    if not (1.15 <= ar <= 1.65):
        return ordered_init

    h_img, w_img = mask.shape[:2]
    max_dim = float(max(h_img, w_img))
    if np.any(quad < -0.25 * max_dim) or np.any(quad[:, 0] > w_img + 0.25 * max_dim) or np.any(quad[:, 1] > h_img + 0.25 * max_dim):
        return ordered_init

    return order_quad(quad)


def warp_perspective_antialiased(
    src: np.ndarray,
    M: np.ndarray,
    dsize: tuple[int, int],
    quad_corners: Optional[np.ndarray] = None,
    flags: int = cv2.INTER_LANCZOS4,
    border_mode: int = cv2.BORDER_REPLICATE,
    antialiased: bool = True,
) -> np.ndarray:
    """Warp a quad from ``src`` to ``dsize`` with ROI mipmap anti-aliasing on downsampling.

    When downsampling high-resolution scenes (e.g., 12 MP phone captures -> 448x630),
    unfiltered ``warpPerspective`` introduces severe spatial aliasing and Lanczos ringing.
    This crops a tight padded ROI around ``quad_corners``, applies exact power-of-two
    ``pyrDown`` + residual Gaussian pre-filtering matching the local scale, and composes
    the offset and scale directly into the homography before warping.
    """
    if not antialiased or quad_corners is None or src.ndim < 2:
        return cv2.warpPerspective(src, M, dsize, flags=flags, borderMode=border_mode)

    target_w, target_h = int(dsize[0]), int(dsize[1])
    h_src, w_src = src.shape[:2]
    q_arr = np.asarray(quad_corners, dtype=np.float32).reshape(-1, 2)
    area_src = abs(float(cv2.contourArea(q_arr)))
    area_dst = float(max(target_w * target_h, 1))
    scale = float(np.sqrt(area_dst / max(area_src, 1.0)))

    if scale >= 0.72:
        return cv2.warpPerspective(src, M, dsize, flags=flags, borderMode=border_mode)

    x_min = float(q_arr[:, 0].min())
    y_min = float(q_arr[:, 1].min())
    x_max = float(q_arr[:, 0].max())
    y_max = float(q_arr[:, 1].max())
    pad = max(24, int(0.06 * max(x_max - x_min, y_max - y_min)))
    rx0 = max(0, int(np.floor(x_min)) - pad)
    ry0 = max(0, int(np.floor(y_min)) - pad)
    rx1 = min(w_src, int(np.ceil(x_max)) + pad)
    ry1 = min(h_src, int(np.ceil(y_max)) + pad)
    if rx1 - rx0 < 32 or ry1 - ry0 < 32:
        return cv2.warpPerspective(src, M, dsize, flags=flags, borderMode=border_mode)

    roi = src[ry0:ry1, rx0:rx1]
    pyr_steps = 0
    rem_scale = scale
    while rem_scale < 0.54 and min(roi.shape[:2]) >= 64 and pyr_steps < 3:
        roi = cv2.pyrDown(roi)
        pyr_steps += 1
        rem_scale *= 2.0

    if rem_scale < 0.88:
        sigma = float(np.clip(0.45 * np.sqrt(max((1.0 / max(rem_scale, 0.25)) ** 2 - 1.0, 0.0)), 0.0, 2.0))
        if sigma >= 0.35:
            roi = cv2.GaussianBlur(roi, (0, 0), sigma)

    T_roi = np.array([[1.0, 0.0, float(rx0)], [0.0, 1.0, float(ry0)], [0.0, 0.0, 1.0]], dtype=np.float64)
    factor = float(1 << pyr_steps)
    S_pyr = np.array([[factor, 0.0, 0.0], [0.0, factor, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    M_eff = M.astype(np.float64) @ T_roi @ S_pyr
    return cv2.warpPerspective(roi, M_eff, dsize, flags=flags, borderMode=border_mode)


def _refine_quad_from_image_edges(
    img_bgr: np.ndarray,
    quad: np.ndarray,
    mask: np.ndarray,
    target_size: tuple[int, int] = (TARGET_H, TARGET_W),
    pad: int = 36,
    band_out: int = 10,
    band_in: int = 12,
    n_samples: int = 72,
    n_iters: int = 2,
) -> np.ndarray:
    """Snap segmentation-derived card edges to full-resolution physical image gradients.

    Runs multi-pass canonical alignment (coarse then tight fine polish) using proximity-prior
    weighted directional Sobel + CIE-Lab chromatic gradients with parabolic 3-point sub-pixel
    peak interpolation, RANSAC line fitting, and span/keystone consistency gates.
    """
    q_cur = order_quad(quad)
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
    mask_f = (mask > 0.5).astype(np.float32)

    n_passes = max(1, int(n_iters))
    for p_idx in range(n_passes):
        b_out = band_out if p_idx == 0 else max(4, min(6, band_out))
        b_in = band_in if p_idx == 0 else max(4, min(6, band_in))
        sigma_prior = 5.5 if p_idx == 0 else 3.2

        try:
            M_pad = cv2.getPerspectiveTransform(q_cur.astype(np.float32), dst_pad)
            M_inv = cv2.getPerspectiveTransform(dst_pad, q_cur.astype(np.float32))
        except cv2.error:
            return q_cur

        patch_bgr = cv2.warpPerspective(
            img_bgr, M_pad, (w_pad, h_pad), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        mask_pad = cv2.warpPerspective(
            mask_f, M_pad, (w_pad, h_pad), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )

        lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_blur = cv2.GaussianBlur(lab[:, :, 0], (3, 3), 0)
        a_blur = cv2.GaussianBlur(lab[:, :, 1], (3, 3), 0)
        b_blur = cv2.GaussianBlur(lab[:, :, 2], (3, 3), 0)

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
            lo_idx = max(2, expected - s_out)
            hi_idx = min((h_pad - 3) if axis == "h" else (w_pad - 3), expected + s_in)
            if hi_idx <= lo_idx + 2:
                return np.empty((0, 2), dtype=np.float64)
            dists = np.arange(lo_idx, hi_idx, dtype=np.float32) - float(expected)
            prior = np.exp(-0.5 * (dists / sigma_prior) ** 2)

            for t in np.linspace(0.14, 0.86, n_samples):
                if axis == "h":
                    x = int(pad + t * (target_w - 1))
                    inner_y = min(h_pad - 1, max(0, expected + (18 if expected < h_pad // 2 else -18)))
                    if mask_pad[inner_y, x] < 0.5:
                        continue
                    col = mag[lo_idx:hi_idx, x]
                    if len(col) == 0:
                        continue
                    weighted = col * prior
                    idx_max = int(np.argmax(weighted))
                    if float(col[idx_max]) > 18.0:
                        sub_y = float(lo_idx) + _subpixel_peak_1d(col, idx_max)
                        pts.append([float(x), sub_y])
                else:
                    y = int(pad + t * (target_h - 1))
                    inner_x = min(w_pad - 1, max(0, expected + (18 if expected < w_pad // 2 else -18)))
                    if mask_pad[y, inner_x] < 0.5:
                        continue
                    row = mag[y, lo_idx:hi_idx]
                    if len(row) == 0:
                        continue
                    weighted = row * prior
                    idx_max = int(np.argmax(weighted))
                    if float(row[idx_max]) > 18.0:
                        sub_x = float(lo_idx) + _subpixel_peak_1d(row, idx_max)
                        pts.append([sub_x, float(y)])
            return np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 2), dtype=np.float64)

        def _mad_clean(pts: np.ndarray, axis_idx: int) -> np.ndarray:
            if len(pts) < 10:
                return pts
            vals = pts[:, axis_idx]
            med = float(np.median(vals))
            mad = max(float(np.median(np.abs(vals - med))), 0.8)
            return pts[np.abs(vals - med) <= 2.8 * mad]

        top_pts = _mad_clean(_sample_side(gy, pad, "h", b_out, b_in), 1)
        bot_pts = _mad_clean(_sample_side(gy, pad + target_h - 1, "h", b_in, b_out), 1)
        left_pts = _mad_clean(_sample_side(gx, pad, "v", b_out, b_in), 0)
        right_pts = _mad_clean(_sample_side(gx, pad + target_w - 1, "v", b_in, b_out), 0)

        def _canonical_line(axis: str, pos: int) -> tuple[float, float, float]:
            return (0.0, 1.0, -float(pos)) if axis == "h" else (1.0, 0.0, -float(pos))

        def _fit_or_default(
            pts: np.ndarray,
            axis: str,
            default_pos: int,
            seg_len: float,
            max_tilt_rad: float = 0.11,
        ) -> tuple[float, float, float]:
            if len(pts) < 14:
                return _canonical_line(axis, default_pos)
            line = _fit_line_ransac(pts, thresh_px=1.6, n_iters=90)
            if line is None:
                return _canonical_line(axis, default_pos)
            a, b, _ = line
            if axis == "h" and abs(a) > max_tilt_rad:
                return _canonical_line(axis, default_pos)
            if axis == "v" and abs(b) > max_tilt_rad:
                return _canonical_line(axis, default_pos)
            # Require broad edge coverage so localized inner borders cannot tilt the side
            span_axis = pts[:, 0] if axis == "h" else pts[:, 1]
            dists = np.abs(pts[:, 0] * a + pts[:, 1] * b + line[2])
            inl = span_axis[dists <= 2.0]
            if len(inl) < 10 or float(inl.max() - inl.min()) < 0.30 * seg_len:
                return _canonical_line(axis, default_pos)
            return line

        l_top = _fit_or_default(top_pts, "h", pad, float(target_w))
        l_right = _fit_or_default(right_pts, "v", pad + target_w - 1, float(target_h))
        l_bot = _fit_or_default(bot_pts, "h", pad + target_h - 1, float(target_w))
        l_left = _fit_or_default(left_pts, "v", pad, float(target_h))

        corners_pad = []
        ok_inter = True
        for l1, l2 in ((l_top, l_left), (l_top, l_right), (l_bot, l_right), (l_bot, l_left)):
            pt = _intersect_lines(l1, l2)
            if pt is None or not np.isfinite(pt).all():
                ok_inter = False
                break
            corners_pad.append(pt)
        if not ok_inter:
            break

        cand_pad = np.asarray(corners_pad, dtype=np.float64)
        if not _is_convex_quad(cand_pad) or not _opposite_taper_ok(cand_pad, max_ratio=1.24):
            break

        refined_scene = cv2.perspectiveTransform(
            cand_pad.astype(np.float32).reshape(1, 4, 2), M_inv
        ).reshape(4, 2).astype(np.float64)

        rect_ref = cv2.minAreaRect(refined_scene.astype(np.float32))
        rw, rh = rect_ref[1]
        if min(rw, rh) < 20:
            break
        ar = max(rw, rh) / max(min(rw, rh), 1e-3)
        if not (1.18 <= ar <= 1.62):
            break

        shift_mean = float(np.linalg.norm(refined_scene - q_cur, axis=1).mean())
        if shift_mean > max(16.0, 0.04 * max(rw, rh)):
            break

        q_cur = order_quad(refined_scene)

    return q_cur


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
    refine_iters: int = 2,
    antialiased_warp: bool = True,
    min_span_frac: float = 0.30,
    max_side_tilt_rad: float = 0.20,
) -> Optional[RectifiedCard]:
    """Rectify an image and its segmentation mask into canonical upright coordinates."""
    quad = fit_quad_from_mask(
        mask,
        edge_fit_method=edge_fit_method,
        corner_excl_frac=corner_excl_frac,
        min_side_inliers=min_side_inliers,
        ransac_thresh_px=ransac_thresh_px,
        ransac_iters=ransac_iters,
        min_span_frac=min_span_frac,
        max_side_tilt_rad=max_side_tilt_rad,
    )
    if quad is None:
        return None
    if refine_edges:
        quad = _refine_quad_from_image_edges(
            img_bgr, quad, mask, target_size=target_size, n_iters=refine_iters
        )

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
    warped_bgr = warp_perspective_antialiased(
        img_bgr,
        M,
        (target_w, target_h),
        quad_corners=quad,
        flags=cv2.INTER_LANCZOS4,
        border_mode=cv2.BORDER_REPLICATE,
        antialiased=antialiased_warp,
    )
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


def canonical_obb_box(
    canon_size: tuple[int, int] = (CANON_H, CANON_W),
    target_size: tuple[int, int] = (TARGET_H, TARGET_W),
) -> np.ndarray:
    """Return [TL, TR, BR, BL] coordinates of the centered card inside the canonical crop."""
    ch, cw = canon_size
    th, tw = target_size
    x0 = float((cw - tw) // 2)
    y0 = float((ch - th) // 2)
    return np.array(
        [
            [x0, y0],
            [x0 + tw - 1.0, y0],
            [x0 + tw - 1.0, y0 + th - 1.0],
            [x0, y0 + th - 1.0],
        ],
        dtype=np.float32,
    )


def warp_to_canonical_obb(
    img_bgr: np.ndarray,
    obb_corners: np.ndarray,
    canon_size: tuple[int, int] = (CANON_H, CANON_W),
    target_size: tuple[int, int] = (TARGET_H, TARGET_W),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp a BGR scene crop around an Stage-1 OBB proposal into padded canonical view."""
    q = order_quad(obb_corners).astype(np.float32)
    dst_box = canonical_obb_box(canon_size=canon_size, target_size=target_size)
    ch, cw = canon_size
    M_canon = cv2.getPerspectiveTransform(q, dst_box)
    M_canon_inv = np.linalg.inv(M_canon).astype(np.float64)
    canon_bgr = cv2.warpPerspective(
        img_bgr,
        M_canon,
        (cw, ch),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return canon_bgr, M_canon.astype(np.float64), M_canon_inv


def _sample_corridor_side(
    prob: np.ndarray,
    side: str,
    nom_box: np.ndarray,
    corridor_px: int = 28,
    corner_excl_frac: float = 0.14,
) -> np.ndarray:
    """Locate outer-to-inner 0.5-isosurface card border points strictly inside an OBB corridor."""
    ch, cw = prob.shape[:2]
    x0, y0 = float(nom_box[0, 0]), float(nom_box[0, 1])
    x1, y1 = float(nom_box[2, 0]), float(nom_box[2, 1])
    tw = x1 - x0
    th = y1 - y0
    pts: list[list[float]] = []

    def _find_crossing(line_vals: np.ndarray, coords: np.ndarray) -> Optional[float]:
        n_v = len(line_vals)
        for idx in range(1, n_v - 6):
            if line_vals[idx - 1] < 0.5 <= line_vals[idx]:
                if float(np.mean(line_vals[idx : idx + 7])) >= 0.58:
                    denom = max(float(line_vals[idx] - line_vals[idx - 1]), 1e-5)
                    frac = (0.5 - float(line_vals[idx - 1])) / denom
                    return float(coords[idx - 1] + frac * (coords[idx] - coords[idx - 1]))
        return None

    if side in ("top", "bottom"):
        xs = np.arange(
            int(round(x0 + corner_excl_frac * tw)),
            int(round(x1 - corner_excl_frac * tw)) + 1,
            2,
        )
        nom_y = int(round(y0 if side == "top" else y1))
        for x in xs:
            if not (0 <= x < cw):
                continue
            if side == "top":
                ya = max(0, nom_y - corridor_px)
                yb = min(ch - 1, nom_y + corridor_px)
                ys = np.arange(ya, yb + 1, dtype=np.float64)
            else:
                ya = min(ch - 1, nom_y + corridor_px)
                yb = max(0, nom_y - corridor_px)
                ys = np.arange(ya, yb - 1, -1, dtype=np.float64)
            if len(ys) < 10:
                continue
            vals = prob[ys.astype(np.int32), x]
            c = _find_crossing(vals, ys)
            if c is not None:
                pts.append([float(x), c])
    else:
        ys = np.arange(
            int(round(y0 + corner_excl_frac * th)),
            int(round(y1 - corner_excl_frac * th)) + 1,
            2,
        )
        nom_x = int(round(x0 if side == "left" else x1))
        for y in ys:
            if not (0 <= y < ch):
                continue
            if side == "left":
                xa = max(0, nom_x - corridor_px)
                xb = min(cw - 1, nom_x + corridor_px)
                xs_ray = np.arange(xa, xb + 1, dtype=np.float64)
            else:
                xa = min(cw - 1, nom_x + corridor_px)
                xb = max(0, nom_x - corridor_px)
                xs_ray = np.arange(xa, xb - 1, -1, dtype=np.float64)
            if len(xs_ray) < 10:
                continue
            vals = prob[y, xs_ray.astype(np.int32)]
            c = _find_crossing(vals, xs_ray)
            if c is not None:
                pts.append([c, float(y)])

    return np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 2), dtype=np.float64)


def fit_quad_from_canonical_prob(
    prob_canon: np.ndarray,
    nom_box: Optional[np.ndarray] = None,
    corridor_px: int = 28,
    corner_excl_frac: float = 0.14,
    min_side_inliers: int = 16,
    ransac_thresh_px: float = 1.6,
    ransac_iters: int = 120,
    max_tilt_rad: float = 0.16,
    min_span_frac: float = 0.30,
) -> np.ndarray:
    """Fit exact canonical quadrilateral corners via two-pass corridor-constrained RANSAC.

    Pass 1 locates outer-to-inner 0.5-isosurface crossings around ``nom_box`` with
    support-span and tilt gates. Pass 2 casts bilinearly interpolated rays normal to
    the Pass-1 edges inside a tighter half-width corridor to eliminate staircase bias.
    """
    ch, cw = prob_canon.shape[:2]
    box = canonical_obb_box(canon_size=(ch, cw)) if nom_box is None else order_quad(nom_box.astype(np.float64))
    x0, y0 = float(box[0, 0]), float(box[0, 1])
    x1, y1 = float(box[2, 0]), float(box[2, 1])
    nom_area = max((x1 - x0) * (y1 - y0), 1.0)
    side_names = ("top", "right", "bottom", "left")
    axes = ("h", "v", "h", "v")
    defaults = (y0, x1, y1, x0)

    def _fit_checked(
        pts: np.ndarray,
        idx: int,
        ref_quad: np.ndarray,
        tilt_limit: float,
    ) -> Optional[tuple[float, float, float]]:
        if len(pts) < min_side_inliers:
            return None
        line = _fit_line_ransac(pts, thresh_px=ransac_thresh_px, n_iters=ransac_iters)
        if line is None:
            return None
        a, b, _ = line
        if axes[idx] == "h" and abs(a) > tilt_limit:
            return None
        if axes[idx] == "v" and abs(b) > tilt_limit:
            return None
        span = _side_inlier_span_frac(
            pts,
            line,
            ref_quad[idx],
            ref_quad[(idx + 1) % 4],
            thresh_px=ransac_thresh_px * 1.25,
        )
        if span < min_span_frac:
            return None
        return line

    def _resolve_quad(
        lines_4: list[Optional[tuple[float, float, float]]],
        pts_4: list[np.ndarray],
        ref_quad: np.ndarray,
    ) -> Optional[np.ndarray]:
        resolved: list[tuple[float, float, float]] = []
        for idx in range(4):
            if lines_4[idx] is not None:
                resolved.append(lines_4[idx])  # type: ignore[arg-type]
                continue
            opp = lines_4[(idx + 2) % 4]
            pts_i = pts_4[idx]
            if opp is not None:
                oa, ob, _ = opp
                if len(pts_i) >= 8:
                    # Estimate constrained 1D parallel offset from partial boundary crossings
                    c_opt = -float(np.median(oa * pts_i[:, 0] + ob * pts_i[:, 1]))
                    resolved.append((oa, ob, c_opt))
                else:
                    mid = 0.5 * (ref_quad[idx] + ref_quad[(idx + 1) % 4])
                    resolved.append((oa, ob, -(oa * float(mid[0]) + ob * float(mid[1]))))
            elif len(pts_i) >= 8:
                na, nb = (0.0, 1.0) if axes[idx] == "h" else (1.0, 0.0)
                c_opt = -float(np.median(na * pts_i[:, 0] + nb * pts_i[:, 1]))
                resolved.append((na, nb, c_opt))
            else:
                d_pos = defaults[idx]
                resolved.append((0.0, 1.0, -d_pos) if axes[idx] == "h" else (1.0, 0.0, -d_pos))

        pts_out = []
        for l1, l2 in (
            (resolved[0], resolved[3]),
            (resolved[0], resolved[1]),
            (resolved[2], resolved[1]),
            (resolved[2], resolved[3]),
        ):
            pt = _intersect_lines(l1, l2)
            if pt is None or not np.isfinite(pt).all():
                return None
            pts_out.append(pt)

        q_out = np.asarray(pts_out, dtype=np.float64)
        if not _is_convex_quad(q_out) or not _opposite_taper_ok(q_out, max_ratio=1.28):
            return None
        rect = cv2.minAreaRect(q_out.astype(np.float32))
        rw, rh = rect[1]
        if min(rw, rh) < 40.0:
            return None
        ar = max(rw, rh) / max(min(rw, rh), 1e-3)
        if not (1.15 <= ar <= 1.65):
            return None
        ref_area = float(cv2.contourArea(q_out.astype(np.float32)))
        if not (0.62 * nom_area <= ref_area <= 1.38 * nom_area):
            return None
        return order_quad(q_out)

    # Pass 1: Axis-aligned corridor around initial OBB proposal
    pts_p1: list[np.ndarray] = []
    lines_p1: list[Optional[tuple[float, float, float]]] = []
    for idx in range(4):
        pts = _sample_corridor_side(
            prob_canon,
            side=side_names[idx],
            nom_box=box,
            corridor_px=corridor_px,
            corner_excl_frac=corner_excl_frac,
        )
        pts_p1.append(pts)
        lines_p1.append(_fit_checked(pts, idx, box, tilt_limit=max_tilt_rad))

    quad_1 = _resolve_quad(lines_p1, pts_p1, box)
    if quad_1 is None:
        return box.astype(np.float64)

    # Pass 2: Normal bilinear refinement around Pass-1 tilted edges inside tighter corridor
    prob_f32 = np.clip(prob_canon.astype(np.float32), 0.0, 1.0)
    corridor_2 = max(10, corridor_px // 2)
    soft_sides_p2 = _sample_soft_mask_quad_sides(
        prob_f32,
        quad_1,
        corner_excl_frac=corner_excl_frac,
        n_rays=110,
        scan_out_px=corridor_2,
        scan_in_px=corridor_2,
    )
    lines_p2 = [
        _fit_checked(soft_sides_p2[idx], idx, quad_1, tilt_limit=max_tilt_rad)
        for idx in range(4)
    ]
    quad_2 = _resolve_quad(lines_p2, soft_sides_p2, quad_1)
    return quad_2 if quad_2 is not None else quad_1


def rectify_from_canonical_prob(
    img_bgr: np.ndarray,
    obb_corners: np.ndarray,
    prob_canon: np.ndarray,
    M_canon_inv: np.ndarray,
    target_size: tuple[int, int] = (TARGET_H, TARGET_W),
    corridor_px: int = 28,
    corner_excl_frac: float = 0.14,
    erode_mask_px: int = 2,
    feather_px: int = 3,
    antialiased_warp: bool = True,
) -> Optional[RectifiedCard]:
    """Refine OBB corners via canonical boundary probabilities and extract clean patch + mask."""
    ch, cw = prob_canon.shape[:2]
    nom_box = canonical_obb_box(canon_size=(ch, cw), target_size=target_size)
    quad_canon = fit_quad_from_canonical_prob(
        prob_canon,
        nom_box=nom_box,
        corridor_px=corridor_px,
        corner_excl_frac=corner_excl_frac,
    )

    quad_scene = cv2.perspectiveTransform(
        quad_canon.astype(np.float32).reshape(1, 4, 2),
        M_canon_inv.astype(np.float32),
    ).reshape(4, 2).astype(np.float64)
    quad_scene = order_quad(quad_scene)

    rect = cv2.minAreaRect(quad_scene.astype(np.float32))
    rw, rh = rect[1]
    ar = max(rw, rh) / max(min(rw, rh), 1e-3)
    if not (1.15 <= ar <= 1.65):
        return None

    target_h, target_w = target_size
    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )
    M_scene = cv2.getPerspectiveTransform(quad_scene.astype(np.float32), dst)
    warped_bgr = warp_perspective_antialiased(
        img_bgr,
        M_scene,
        (target_w, target_h),
        quad_corners=quad_scene,
        flags=cv2.INTER_LANCZOS4,
        border_mode=cv2.BORDER_REPLICATE,
        antialiased=antialiased_warp,
    )
    if warped_bgr is None or warped_bgr.mean() < 1.0:
        return None

    M_mask = cv2.getPerspectiveTransform(quad_canon.astype(np.float32), dst)
    warped_mask = cv2.warpPerspective(
        np.clip(prob_canon, 0.0, 1.0).astype(np.float32),
        M_mask,
        (target_w, target_h),
        flags=cv2.INTER_LINEAR,
    )
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
        corners=quad_scene,
        patch_rgb=patch_rgb,
        warped_mask=warped_mask,
        sharpness=sharpness,
    )


def build_alignment_feature_map(
    img_rgb: np.ndarray,
    align_size_wh: tuple[int, int] = (224, 315),
) -> np.ndarray:
    """Build an illumination-invariant structural feature map for reference-guided ECC.

    Blends CLAHE luminance with normalized Sobel gradient magnitude so artwork frame,
    cardboard border, typography, and energy icons align reliably under glare or white
    balance shifts.
    """
    aw, ah = int(align_size_wh[0]), int(align_size_wh[1])
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if gray.shape[:2] != (ah, aw):
        gray = cv2.resize(gray, (aw, ah), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    eq = cv2.GaussianBlur(clahe.apply(gray), (3, 3), 0).astype(np.float32)
    gx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    g_p99 = max(float(np.percentile(grad, 99)), 1.0)
    grad_norm = np.clip(grad / g_p99, 0.0, 1.0)
    eq_norm = eq / 255.0
    fused = 0.65 * grad_norm + 0.35 * eq_norm
    mean_v = float(fused.mean())
    std_v = max(float(fused.std()), 1e-4)
    return ((fused - mean_v) / std_v).astype(np.float32)


def align_patch_to_reference_homography(
    patch_rgb: np.ndarray,
    ref_align_map: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_corner_shift_px: float = 18.0,
) -> tuple[np.ndarray, Optional[np.ndarray], bool]:
    """Refine residual scale, shear, and projective skew against a canonical reference template.

    Runs coarse affine ECC followed by bounded homography ECC on gradient-normalized
    structural maps, co-warping both ``patch_rgb`` and optional ``mask`` only when corner
    displacements satisfy strict convexity and displacement bounds.
    """
    if ref_align_map is None or patch_rgb is None or patch_rgb.size == 0:
        return patch_rgb, mask, False

    th, tw = patch_rgb.shape[:2]
    ah, aw = ref_align_map.shape[:2]
    q_map = build_alignment_feature_map(patch_rgb, align_size_wh=(aw, ah))

    ecc_mask = None
    if mask is not None and mask.shape[:2] == (th, tw):
        m_small = cv2.resize(mask.astype(np.float32), (aw, ah), interpolation=cv2.INTER_AREA)
        if float((m_small > 0.5).mean()) >= 0.45:
            ecc_mask = (m_small > 0.5).astype(np.uint8) * 255

    warp_aff = np.eye(2, 3, dtype=np.float32)
    crit_aff = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 1e-3)
    try:
        _, warp_aff = cv2.findTransformECC(
            ref_align_map, q_map, warp_aff, cv2.MOTION_AFFINE, crit_aff, ecc_mask, 3
        )
    except cv2.error:
        warp_aff = np.eye(2, 3, dtype=np.float32)

    warp_h = np.eye(3, 3, dtype=np.float32)
    warp_h[:2, :] = warp_aff
    crit_h = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3)
    try:
        _, warp_h = cv2.findTransformECC(
            ref_align_map, q_map, warp_h, cv2.MOTION_HOMOGRAPHY, crit_h, ecc_mask, 3
        )
    except cv2.error:
        warp_h = np.eye(3, 3, dtype=np.float32)
        warp_h[:2, :] = warp_aff

    sx = float(tw) / float(aw)
    sy = float(th) / float(ah)
    S = np.diag([sx, sy, 1.0]).astype(np.float64)
    S_inv = np.diag([1.0 / sx, 1.0 / sy, 1.0]).astype(np.float64)
    H_full = S @ warp_h.astype(np.float64) @ S_inv
    try:
        H_inv = np.linalg.inv(H_full)
    except np.linalg.LinAlgError:
        return patch_rgb, mask, False

    corners = np.array(
        [[0.0, 0.0], [tw - 1.0, 0.0], [tw - 1.0, th - 1.0], [0.0, th - 1.0]],
        dtype=np.float32,
    ).reshape(1, 4, 2)
    warped_corners = cv2.perspectiveTransform(corners, H_inv.astype(np.float32)).reshape(4, 2)
    if not _is_convex_quad(warped_corners):
        return patch_rgb, mask, False

    shifts = np.linalg.norm(warped_corners - corners.reshape(4, 2), axis=1)
    if float(shifts.mean()) > max_corner_shift_px or float(shifts.max()) > 1.65 * max_corner_shift_px:
        return patch_rgb, mask, False

    if float(shifts.mean()) < 0.15:
        return patch_rgb, mask, False

    flags = cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP
    aligned_rgb = cv2.warpPerspective(
        patch_rgb,
        H_full.astype(np.float32),
        (tw, th),
        flags=flags,
        borderMode=cv2.BORDER_REPLICATE,
    )
    aligned_mask = mask
    if mask is not None and mask.shape[:2] == (th, tw):
        aligned_mask = cv2.warpPerspective(
            mask.astype(np.float32),
            H_full.astype(np.float32),
            (tw, th),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        aligned_mask = np.clip(aligned_mask, 0.0, 1.0).astype(np.float32)

    return aligned_rgb, aligned_mask, True

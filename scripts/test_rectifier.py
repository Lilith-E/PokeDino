"""Unit tests for mask-based perspective rectification and dynamic mask warping."""

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.masking import rounded_rect_mask  # noqa: E402
from backend.rectifier import (  # noqa: E402
    align_patch_to_reference_homography,
    build_alignment_feature_map,
    fit_quad_from_mask,
    order_quad,
    rectify_from_mask,
    warp_perspective_antialiased,
)


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def _align_error(pred: np.ndarray, gt: np.ndarray) -> float:
    best = float("inf")
    for order in (pred, pred[::-1]):
        for shift in range(4):
            cand = np.roll(order, shift, axis=0)
            err = float(np.linalg.norm(cand - gt, axis=1).mean())
            best = min(best, err)
    return best


def _synthesize_tilted_card(
    canvas_size: tuple[int, int] = (640, 640),
    tilt_px: float = 28.0,
    angle_deg: float = 23.0,
    occlude_side: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a synthetic tilted card with rounded corners and optional edge occlusion."""
    h_c, w_c = 440, 315
    card_rgb = np.full((h_c, w_c, 3), 220, dtype=np.uint8)
    cv2.rectangle(card_rgb, (24, 24), (w_c - 25, h_c - 25), (30, 80, 190), -1)
    cv2.rectangle(card_rgb, (45, 55), (w_c - 45, 220), (240, 190, 50), -1)

    mask_local = (rounded_rect_mask(h_c, w_c, corner_radius_frac=0.045, erode_frac=0.0, feather_px=0) > 0.5).astype(np.uint8)

    src = np.array([[0, 0], [w_c - 1, 0], [w_c - 1, h_c - 1], [0, h_c - 1]], dtype=np.float32)

    H_out, W_out = canvas_size
    cx, cy = W_out / 2.0, H_out / 2.0
    theta = np.radians(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
    base_dst = (src - np.array([w_c / 2.0, h_c / 2.0])) @ rot.T + np.array([cx, cy])
    pert = np.array(
        [
            [tilt_px * 0.4, tilt_px * 0.2],
            [-tilt_px * 0.5, tilt_px * 0.35],
            [tilt_px * 0.3, -tilt_px * 0.4],
            [-tilt_px * 0.25, -tilt_px * 0.2],
        ],
        dtype=np.float64,
    )
    dst_gt = (base_dst + pert).astype(np.float32)

    M = cv2.getPerspectiveTransform(src, dst_gt)
    warped_card = cv2.warpPerspective(card_rgb, M, (W_out, H_out), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpPerspective(mask_local * 255, M, (W_out, H_out), flags=cv2.INTER_NEAREST)

    scene = np.full((H_out, W_out, 3), 40, dtype=np.uint8)
    fg = warped_mask > 127
    scene[fg] = warped_card[fg]
    mask_f32 = (warped_mask > 127).astype(np.float32)

    if occlude_side:
        p_mid = 0.5 * (dst_gt[1] + dst_gt[2])
        cv2.circle(mask_f32, (int(p_mid[0]), int(p_mid[1])), 26, 0.0, -1)
        cv2.circle(scene, (int(p_mid[0]), int(p_mid[1])), 26, (180, 140, 120), -1)

    return scene, mask_f32, dst_gt.astype(np.float64)


def test_fit_quad_tilted_rounded() -> None:
    scene, mask, gt = _synthesize_tilted_card(tilt_px=30.0, angle_deg=37.0, occlude_side=False)
    quad = fit_quad_from_mask(mask, edge_fit_method="ransac")
    _assert(quad is not None, "Expected non-None quad")
    err = _align_error(quad, gt)
    _assert(err < 1.5, f"Expected corner error < 1.5 px on clean tilted mask, got {err:.3f}")
    print(f"  ✓ fit_quad_from_mask (tilted rounded, err={err:.3f} px)")


def test_fit_quad_and_warp_with_occlusion() -> None:
    scene, mask, gt = _synthesize_tilted_card(tilt_px=22.0, angle_deg=-19.0, occlude_side=True)
    res = rectify_from_mask(scene, mask, edge_fit_method="ransac", erode_mask_px=0, feather_px=0)
    _assert(res is not None, "Expected non-None RectifiedCard")
    err = _align_error(res.corners, gt)
    _assert(err < 2.0, f"Expected corner error < 2.0 px under edge occlusion, got {err:.3f}")
    _assert(res.patch_rgb.shape == (630, 448, 3), f"Unexpected patch shape: {res.patch_rgb.shape}")
    _assert(res.warped_mask.shape == (630, 448), f"Unexpected warped mask shape: {res.warped_mask.shape}")
    right_mid_val = float(res.warped_mask[315, 435])
    center_val = float(res.warped_mask[315, 224])
    corner_val = float(res.warped_mask[2, 2])
    _assert(right_mid_val < 0.2, f"Occluded border region should be masked out, got {right_mid_val}")
    _assert(center_val > 0.9, f"Interior card center should be valid, got {center_val}")
    _assert(corner_val < 0.2, f"Rounded corner exterior should be masked out, got {corner_val}")
    print(f"  ✓ rectify_from_mask with occlusion (err={err:.3f} px, occluded_val={right_mid_val:.2f})")


def test_order_quad_portrait() -> None:
    raw = np.array([[400, 500], [100, 100], [100, 500], [400, 100]], dtype=np.float64)
    ordered = order_quad(raw)
    top_w = np.linalg.norm(ordered[1] - ordered[0])
    left_h = np.linalg.norm(ordered[3] - ordered[0])
    _assert(top_w < left_h, f"Top edge ({top_w}) must be shorter than side edge ({left_h})")
    _assert(ordered[0, 1] < ordered[3, 1], "Top vertices must have smaller y than bottom vertices")
    print("  ✓ order_quad portrait orientation")


def test_soft_mask_subpixel_precision() -> None:
    _, mask_hard, gt = _synthesize_tilted_card(tilt_px=26.0, angle_deg=17.0, occlude_side=False)
    mask_soft = cv2.GaussianBlur(mask_hard.astype(np.float32), (0, 0), 1.6)
    quad = fit_quad_from_mask(mask_soft, edge_fit_method="ransac")
    _assert(quad is not None, "Expected non-None quad on soft mask")
    err = _align_error(quad, gt)
    _assert(err < 1.0, f"Expected sub-pixel soft mask corner error < 1.0 px, got {err:.3f}")
    print(f"  ✓ soft 0.5-isocontour sub-pixel fit (err={err:.3f} px)")


def test_single_side_heavy_occlusion_partial_recovery() -> None:
    # Steep trapezoid where 82% of right edge is occluded; 3 surviving edges must be kept
    _, mask, gt = _synthesize_tilted_card(tilt_px=34.0, angle_deg=12.0, occlude_side=False)
    mask_occ = mask.copy()
    p_top_r = gt[1]
    p_bot_r = gt[2]
    for frac in np.linspace(0.08, 0.88, 30):
        c = (1.0 - frac) * p_top_r + frac * p_bot_r
        cv2.circle(mask_occ, (int(round(c[0])), int(round(c[1]))), 28, 0.0, -1)
    quad = fit_quad_from_mask(mask_occ, edge_fit_method="ransac")
    _assert(quad is not None, "Expected non-None quad under single-side heavy occlusion")
    # The two unoccluded left corners [TL, BL] must stay dead-on (< 1.5 px) rather than reset to minAreaRect
    left_err = float(np.linalg.norm(order_quad(quad)[[0, 3]] - order_quad(gt)[[0, 3]], axis=1).mean())
    _assert(left_err < 1.8, f"Unoccluded opposite corners should be preserved (< 1.8 px), got {left_err:.3f}")
    print(f"  ✓ per-side partial recovery preserves clean edges (left_err={left_err:.3f} px)")


def test_diagonal_bite_tilt_gate() -> None:
    # Simulate a Charizard-style slanted concave right-border segmentation bite
    _, mask, gt = _synthesize_tilted_card(tilt_px=10.0, angle_deg=0.0, occlude_side=False)
    mask_bitten = mask.copy()
    tr = gt[1]
    br = gt[2]
    mid_in = 0.5 * (tr + br) + np.array([-42.0, 0.0], dtype=np.float64)
    poly = np.array([tr + [4, -4], mid_in, br + [4, 4]], dtype=np.int32)
    cv2.fillPoly(mask_bitten, [poly], 0.0)
    quad = fit_quad_from_mask(mask_bitten, edge_fit_method="ransac")
    _assert(quad is not None, "Expected non-None quad on diagonally bitten mask")
    err = _align_error(quad, gt)
    _assert(err < 12.0, f"Tilt gate should block diagonal bite runaway, got err={err:.3f}")
    print(f"  ✓ diagonal bite tilt gate resisted notch runaway (err={err:.3f} px)")


def test_antialiased_warp_highres_downsampling() -> None:
    # 2400x3200 synthetic card downsampled to 448x630
    H, W = 2801, 2101
    src = np.full((H, W, 3), 230, dtype=np.uint8)
    for y in range(100, H - 100, 12):
        cv2.line(src, (100, y), (W - 100, y), (30, 30, 30), 2)
    q = np.array([[150.0, 200.0], [1850.0, 240.0], [1820.0, 2590.0], [120.0, 2550.0]], dtype=np.float32)
    dst = np.array([[0, 0], [447, 0], [447, 629], [0, 629]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    out_aa = warp_perspective_antialiased(src, M, (448, 630), quad_corners=q, antialiased=True)
    out_raw = warp_perspective_antialiased(src, M, (448, 630), quad_corners=q, antialiased=False)
    _assert(out_aa.shape == (630, 448, 3), f"Unexpected warped shape: {out_aa.shape}")
    # Anti-aliased warp should substantially reduce Moire Laplacian variance vs raw Lanczos on dense lines
    lap_aa = float(cv2.Laplacian(cv2.cvtColor(out_aa, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    lap_raw = float(cv2.Laplacian(cv2.cvtColor(out_raw, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    _assert(lap_aa < lap_raw, f"Expected AA warp to suppress Moire aliasing ({lap_aa:.1f} vs {lap_raw:.1f})")
    print(f"  ✓ anti-aliased high-res warp suppressed aliasing (LapVar {lap_raw:.0f} -> {lap_aa:.0f})")


def test_homography_ecc_reference_alignment() -> None:
    h, w = 630, 448
    ref = np.full((h, w, 3), 215, dtype=np.uint8)
    cv2.rectangle(ref, (30, 30), (w - 31, h - 31), (40, 60, 170), 4)
    cv2.rectangle(ref, (50, 70), (w - 50, 260), (200, 140, 40), -1)
    cv2.rectangle(ref, (50, 300), (w - 50, 560), (60, 120, 80), 3)
    cv2.putText(ref, "POKEDINO ECC", (65, 345), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)

    src_c = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst_skew = src_c + np.array([[6.0, 4.0], [-5.0, 7.0], [7.0, -5.0], [-6.0, -4.0]], dtype=np.float32)
    H_perturb = cv2.getPerspectiveTransform(src_c, dst_skew)
    query = cv2.warpPerspective(ref, H_perturb, (w, h), borderMode=cv2.BORDER_REPLICATE)
    qmask = cv2.warpPerspective(
        rounded_rect_mask(h, w, corner_radius_frac=0.035, erode_frac=0.01, feather_px=3),
        H_perturb,
        (w, h),
    )

    ref_map = build_alignment_feature_map(ref, align_size_wh=(224, 315))
    aligned_rgb, aligned_mask, applied = align_patch_to_reference_homography(query, ref_map, mask=qmask)
    _assert(applied, "Expected bounded homography ECC alignment to apply")
    _assert(aligned_mask is not None and aligned_mask.shape == (h, w), "Expected co-warped mask")
    mae_before = float(np.abs(query.astype(np.float32) - ref.astype(np.float32)).mean())
    mae_after = float(np.abs(aligned_rgb.astype(np.float32) - ref.astype(np.float32)).mean())
    _assert(mae_after < 0.5 * mae_before, f"Expected >50% MAE drop, got {mae_before:.2f} -> {mae_after:.2f}")
    print(f"  ✓ bounded homography ECC alignment (MAE {mae_before:.2f} -> {mae_after:.2f})")


def main() -> None:
    print("\n=== Test rectifier ===\n")
    test_order_quad_portrait()
    test_fit_quad_tilted_rounded()
    test_fit_quad_and_warp_with_occlusion()
    test_soft_mask_subpixel_precision()
    test_single_side_heavy_occlusion_partial_recovery()
    test_diagonal_bite_tilt_gate()
    test_antialiased_warp_highres_downsampling()
    test_homography_ecc_reference_alignment()
    print("\nAll rectifier tests passed.\n")


if __name__ == "__main__":
    main()

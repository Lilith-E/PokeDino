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
from backend.rectifier import fit_quad_from_mask, order_quad, rectify_from_mask  # noqa: E402


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


def main() -> None:
    print("\n=== Test rectifier ===\n")
    test_order_quad_portrait()
    test_fit_quad_tilted_rounded()
    test_fit_quad_and_warp_with_occlusion()
    print("\nAll rectifier tests passed.\n")


if __name__ == "__main__":
    main()

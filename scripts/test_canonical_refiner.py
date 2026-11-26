"""Unit and integration verification for Stage-2 Canonical Boundary Refiner (obb_unet).

Covers:
1. Canonical OBB warp geometry and exact corner round-trip mapping.
2. Corridor-constrained 4-line RANSAC under trapezoidal perspective tilt.
3. Immunity to Charizard-style internal holo/dark margin holes (outside-in scan).
4. Finger occlusion dynamic validity mask preservation and corner stability.
5. Adaptive TTA OBB proposals + OBBCanonicalUNetDetector end-to-end inference.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.rectifier import (  # noqa: E402
    CANON_H,
    CANON_W,
    TARGET_H,
    TARGET_W,
    canonical_obb_box,
    fit_quad_from_canonical_prob,
    rectify_from_canonical_prob,
    warp_to_canonical_obb,
)
from backend.seg_detector import MobileNetV3UNet, OBBCanonicalUNetDetector  # noqa: E402
from scripts.benchmark_segmentation_vs_obb import align_corner_error  # noqa: E402


def _render_quad_prob(quad: np.ndarray, blur_sigma: float = 0.8) -> np.ndarray:
    mask = np.zeros((CANON_H, CANON_W), dtype=np.float32)
    cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 1.0)
    if blur_sigma > 0:
        mask = cv2.GaussianBlur(mask, (5, 5), blur_sigma)
    return np.clip(mask, 0.0, 1.0)


def test_canonical_warp_roundtrip() -> None:
    scene = np.full((1000, 800, 3), 120, dtype=np.uint8)
    obb = np.array(
        [[180.0, 150.0], [560.0, 175.0], [530.0, 710.0], [150.0, 685.0]],
        dtype=np.float32,
    )
    canon_bgr, M_canon, M_inv = warp_to_canonical_obb(scene, obb)
    assert canon_bgr.shape == (CANON_H, CANON_W, 3)

    mapped = cv2.perspectiveTransform(obb.reshape(1, 4, 2), M_canon.astype(np.float32)).reshape(4, 2)
    nom = canonical_obb_box()
    err = float(np.linalg.norm(mapped - nom, axis=1).max())
    assert err < 1e-2, f"Canonical OBB mapping error too large: {err}"

    back = cv2.perspectiveTransform(nom.reshape(1, 4, 2), M_inv.astype(np.float32)).reshape(4, 2)
    back_err = float(np.linalg.norm(back - obb, axis=1).max())
    assert back_err < 1e-2, f"Inverse canonical mapping error too large: {back_err}"
    print(f"PASS test_canonical_warp_roundtrip (max_err={err:.4f} px)")


def test_corridor_fit_trapezoidal_tilt() -> None:
    nom = canonical_obb_box()
    # Apply perspective trapezoid tilt + translation still inside ±28 px corridor.
    gt = nom + np.array(
        [[14.0, -10.0], [-12.0, -6.0], [16.0, 14.0], [-15.0, 11.0]],
        dtype=np.float32,
    )
    prob = _render_quad_prob(gt)
    pred = fit_quad_from_canonical_prob(prob, nom_box=nom, corridor_px=28)
    err = align_corner_error(pred, gt)
    assert err < 1.2, f"Trapezoid corner error too high: {err:.3f} px"
    print(f"PASS test_corridor_fit_trapezoidal_tilt (corner_err={err:.3f} px)")


def test_corridor_fit_internal_holo_cutout() -> None:
    nom = canonical_obb_box()
    gt = nom + np.array([[6.0, 4.0], [8.0, 5.0], [7.0, 6.0], [5.0, 5.0]], dtype=np.float32)
    prob = _render_quad_prob(gt)
    # Carve a Charizard-style internal dark holo/foil seam right inside the right edge
    # leaving the outer 12 px border intact so outer-to-inner scanning locks cleanly.
    x_right = int(round(gt[1, 0]))
    y_top = int(round(gt[1, 1] + 150))
    y_bot = int(round(gt[2, 1] - 150))
    prob[y_top:y_bot, x_right - 48 : x_right - 12] = 0.0

    pred = fit_quad_from_canonical_prob(prob, nom_box=nom, corridor_px=28)
    err = align_corner_error(pred, gt)
    assert err < 1.2, f"Internal holo hole pulled fitted right edge (err={err:.3f} px)"
    print(f"PASS test_corridor_fit_internal_holo_cutout (corner_err={err:.3f} px)")


def test_finger_occlusion_dynamic_mask() -> None:
    scene = np.full((1000, 800, 3), 50, dtype=np.uint8)
    obb_scene = np.array(
        [[170.0, 160.0], [618.0, 160.0], [618.0, 790.0], [170.0, 790.0]],
        dtype=np.float32,
    )
    _, M_canon, M_inv = warp_to_canonical_obb(scene, obb_scene)
    nom = canonical_obb_box()
    gt_canon = nom + np.array([[4.0, -5.0], [6.0, -4.0], [5.0, 6.0], [3.0, 5.0]], dtype=np.float32)
    gt_scene = cv2.perspectiveTransform(gt_canon.reshape(1, 4, 2), M_inv.astype(np.float32)).reshape(4, 2)

    prob = _render_quad_prob(gt_canon, blur_sigma=0.6)
    # Simulate a finger occluding ~25% of the right edge center.
    fx = int(round(gt_canon[1, 0] - 8))
    fy = int(round(0.5 * (gt_canon[1, 1] + gt_canon[2, 1])))
    cv2.circle(prob, (fx, fy), 55, 0.0, -1)

    rect = rectify_from_canonical_prob(
        scene,
        obb_scene,
        prob_canon=prob,
        M_canon_inv=M_inv,
        corridor_px=28,
    )
    assert rect is not None
    assert rect.patch_rgb.shape == (TARGET_H, TARGET_W, 3)
    assert rect.warped_mask.shape == (TARGET_H, TARGET_W)

    err_scene = align_corner_error(rect.corners, gt_scene)
    assert err_scene < 2.0, f"Finger occlusion degraded scene corner fit: {err_scene:.3f} px"

    # Center must remain valid (>0.9), while right-middle border occluded by finger must drop (<0.1).
    assert rect.warped_mask[TARGET_H // 2, TARGET_W // 2] > 0.9
    assert rect.warped_mask[TARGET_H // 2, TARGET_W - 12] < 0.1
    print(f"PASS test_finger_occlusion_dynamic_mask (scene_err={err_scene:.3f} px)")


def test_detector_e2e_weights_only_and_adaptive_tta() -> None:
    obb_w = PROJECT_ROOT / "data" / "weights" / "yolo11s_obb_finetuned_best.pt"
    if not obb_w.exists():
        print("SKIP test_detector_e2e_weights_only_and_adaptive_tta (no OBB weights)")
        return

    probe = PROJECT_ROOT / "experiments" / "synthetic-to-real" / "charizard-damaged.jpeg"
    if not probe.exists():
        test_imgs = sorted((PROJECT_ROOT / "data" / "datasets" / "pokemon_seg" / "images" / "test").glob("*.jpg"))
        if not test_imgs:
            print("SKIP test_detector_e2e_weights_only_and_adaptive_tta (no probe image)")
            return
        probe = test_imgs[0]

    scene = cv2.imread(str(probe))
    assert scene is not None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_ckpt = Path(tmp) / "dummy_canon_unet.pt"
        model = MobileNetV3UNet(pretrained=False)
        torch.save({"model": model.state_dict(), "epoch": 1, "val_iou": 0.9}, tmp_ckpt)

        # Verify weights_only=True load succeeds.
        loaded = torch.load(tmp_ckpt, map_location="cpu", weights_only=True)
        assert "model" in loaded

        det = OBBCanonicalUNetDetector(
            obb_weights=str(obb_w),
            unet_weights=str(tmp_ckpt),
            adaptive_tta=True,
            adaptive_tta_conf=0.75,
            corridor_px=28,
            device="cpu",
        )
        proposals = det.obb_detector.propose_quads(scene)
        assert len(proposals) >= 1, "Adaptive TTA failed to propose card quad"
        dets = det.detect(scene)
        assert len(dets) >= 1, "OBBCanonicalUNetDetector returned 0 detections"
        d0 = dets[0]
        assert d0.patch.shape == (TARGET_H, TARGET_W, 3)
        assert d0.mask.shape == (TARGET_H, TARGET_W)
        print(f"PASS test_detector_e2e_weights_only_and_adaptive_tta (conf={d0.confidence:.3f})")


def main() -> int:
    test_canonical_warp_roundtrip()
    test_corridor_fit_trapezoidal_tilt()
    test_corridor_fit_internal_holo_cutout()
    test_finger_occlusion_dynamic_mask()
    test_detector_e2e_weights_only_and_adaptive_tta()
    print("ALL CANONICAL REFINER TESTS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

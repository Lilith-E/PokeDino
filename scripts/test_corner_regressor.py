"""Unit and integration tests for SOTA 4-Corner Heatmap Regressor."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.corner_regressor import (  # noqa: E402
    PATCH_H,
    PATCH_W,
    CornerHeatmapRegressor,
    CornerHeatmapUNet,
    extract_oriented_padded_patch,
    soft_argmax_2d,
)
from backend.rectifier import TARGET_H, TARGET_W, order_quad  # noqa: E402


def test_similarity_patch_roundtrip() -> None:
    img = np.zeros((900, 1200, 3), dtype=np.uint8)
    gt = order_quad(np.array([[300.0, 200.0], [620.0, 260.0], [540.0, 700.0], [210.0, 640.0]]))
    _, M_sim, M_inv = extract_oriented_padded_patch(img, gt, patch_hw=(PATCH_H, PATCH_W), pad_frac=0.24)
    ones = np.ones((4, 1), dtype=np.float64)
    in_patch = (np.hstack([gt, ones]) @ M_sim.T)
    back_scene = (np.hstack([in_patch, ones]) @ M_inv.T)
    err = float(np.max(np.abs(back_scene - gt)))
    assert err < 1e-6, f"Roundtrip error {err}"
    assert np.all(in_patch[:, 0] > 10) and np.all(in_patch[:, 0] < PATCH_W - 10)
    assert np.all(in_patch[:, 1] > 10) and np.all(in_patch[:, 1] < PATCH_H - 10)
    print(f"  ✓ similarity_patch_roundtrip (max_err={err:.2e} px)")


def test_soft_argmax_2d_exact_peak() -> None:
    h_hm, w_hm = PATCH_H // 2, PATCH_W // 2
    logits = torch.full((1, 4, h_hm, w_hm), -30.0)
    targets_hm = [(30, 40), (160, 45), (155, 210), (35, 215)]
    for k, (x, y) in enumerate(targets_hm):
        logits[0, k, y, x] = 20.0
    coords, probs = soft_argmax_2d(logits, out_hw=(PATCH_H, PATCH_W))
    assert probs.shape == (1, 4, h_hm, w_hm)
    for k, (x, y) in enumerate(targets_hm):
        exp_x = float(x) * float(PATCH_W - 1) / float(w_hm - 1)
        exp_y = float(y) * float(PATCH_H - 1) / float(h_hm - 1)
        err = float(torch.hypot(coords[0, k, 0] - exp_x, coords[0, k, 1] - exp_y))
        assert err < 1e-3, f"Soft-argmax error {err}"
    print("  ✓ soft_argmax_2d_exact_peak (err < 1e-3 px)")


def test_model_forward_and_regressor() -> None:
    net = CornerHeatmapUNet(pretrained=False)
    x = torch.randn(2, 3, PATCH_H, PATCH_W)
    c_logits, m_logits = net(x)
    assert c_logits.shape == (2, 4, PATCH_H // 2, PATCH_W // 2)
    assert m_logits.shape == (2, 1, PATCH_H // 2, PATCH_W // 2)

    weights_p = PROJECT_ROOT / "data" / "weights" / "corner_heatmap_best.pt"
    if weights_p.exists():
        reg = CornerHeatmapRegressor(weights_p)
        scene = np.full((800, 800, 3), 180, dtype=np.uint8)
        gt = order_quad(np.array([[220.0, 140.0], [540.0, 170.0], [560.0, 620.0], [200.0, 590.0]]))
        cv2.fillPoly(scene, [gt.astype(np.int32)], (40, 200, 240))
        pred = reg.predict_from_proposal(scene, gt)
        assert pred.rectified_rgb.shape == (TARGET_H, TARGET_W, 3)
        assert pred.scene_corners.shape == (4, 2)
        print("  ✓ corner_heatmap_regressor_e2e")


def main() -> None:
    print("\n=== Test corner_regressor ===\n")
    test_similarity_patch_roundtrip()
    test_soft_argmax_2d_exact_peak()
    test_model_forward_and_regressor()
    print("\nAll corner_regressor tests passed.\n")


if __name__ == "__main__":
    main()

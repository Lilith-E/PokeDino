"""Direct patch-level visual & quantitative benchmark of SOTA 4-Corner Heatmap Regressor.

Starts from the EXACT SAME frozen detector ROI patch (oriented + 24% padded) and compares:
  1. Initial Proposal (Raw OBB)
  2. Classical DP-IRLS (Peak 8-DOF)
  3. SOTA Deep Learning: 4-Corner Heatmap Regressor (MobileNetV3-UNet + Soft-Argmax)

Evaluates on:
  - pokemon_seg/test (exact corner error, <=1/2/3px hit rate, Mask IoU, Border Clipping Rate)
  - All real captures in experiments/ (trainer-4, extreme-dragonair-case, rayquaza-delta,
    mega-charizard, greeninja, trainer, ditto, charizard-damaged, realgengar-1, twocards, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.corner_regressor import (  # noqa: E402
    PAD_FRAC,
    PATCH_H,
    PATCH_W,
    CornerHeatmapRegressor,
    extract_oriented_padded_patch,
)
from backend.experiment import Experiment, load_config, set_seed  # noqa: E402
from backend.rectifier import (  # noqa: E402
    TARGET_H,
    TARGET_W,
    order_quad,
    refine_card_quad,
    warp_perspective_antialiased,
)
from backend.seg_detector import OBBCanonicalUNetDetector  # noqa: E402
from scripts.benchmark_quality_gate_perspective import _gather_real_images  # noqa: E402
from scripts.benchmark_same_bbox_perspective import (  # noqa: E402
    CORNER_COLORS_BGR,
    CORNER_NAMES,
    _summarize_errs,
)
from scripts.benchmark_segmentation_vs_obb import (  # noqa: E402
    align_corner_error,
    project_patch_mask_to_scene,
)

logger = logging.getLogger(__name__)


def _draw_corners_on_patch(patch_bgr: np.ndarray, patch_corners: np.ndarray, heatmaps: np.ndarray | None = None) -> np.ndarray:
    vis = patch_bgr.copy()
    if heatmaps is not None:
        hm_max = np.max(heatmaps, axis=0)
        hm_max = hm_max / max(float(hm_max.max()), 1e-6)
        hm_up = cv2.resize(hm_max, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_LINEAR)
        hm_col = cv2.applyColorMap(np.clip(hm_up * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        vis = cv2.addWeighted(vis, 0.78, hm_col, 0.22, 0)

    pts = patch_corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts], True, (255, 110, 20), 2, cv2.LINE_AA)
    for i, (cx, cy) in enumerate(patch_corners):
        c = CORNER_COLORS_BGR[i]
        cv2.circle(vis, (int(round(cx)), int(round(cy))), 5, c, -1, cv2.LINE_AA)
        cv2.putText(
            vis,
            CORNER_NAMES[i],
            (int(round(cx)) + 6, int(round(cy)) + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            c,
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def _patch_corner_zoom(patch_bgr: np.ndarray, patch_corners: np.ndarray, half: int = 44) -> np.ndarray:
    vis_rgb = _draw_corners_on_patch(patch_bgr, patch_corners, heatmaps=None)
    h, w = vis_rgb.shape[:2]
    tiles = []
    for i, (cx, cy) in enumerate(patch_corners):
        cx_i, cy_i = int(round(cx)), int(round(cy))
        x0, x1 = max(0, cx_i - half), min(w, cx_i + half)
        y0, y1 = max(0, cy_i - half), min(h, cy_i + half)
        crop = vis_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            crop = np.zeros((2 * half, 2 * half, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (140, 140), interpolation=cv2.INTER_LINEAR)
        cv2.putText(crop, CORNER_NAMES[i], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(crop)
    return np.vstack([np.hstack([tiles[0], tiles[1]]), np.hstack([tiles[3], tiles[2]])])


def _warp_scene_quad(img_bgr: np.ndarray, scene_quad: np.ndarray) -> np.ndarray:
    q = order_quad(scene_quad).astype(np.float32)
    dst = np.array([[0, 0], [TARGET_W - 1, 0], [TARGET_W - 1, TARGET_H - 1], [0, TARGET_H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    warped = warp_perspective_antialiased(img_bgr, M, (TARGET_W, TARGET_H), quad_corners=q, antialiased=True)
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)


def _render_patch_board(
    title: str,
    patch_bgr: np.ndarray,
    variants: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    out_path: Path,
) -> None:
    cols = list(variants.keys())
    n_cols = len(cols)
    fig, axes = plt.subplots(3, n_cols, figsize=(4.0 * n_cols, 11.8), dpi=150)
    fig.suptitle(title, fontsize=13.5, fontweight="bold", y=0.992)
    for c_idx, name in enumerate(cols):
        p_corners, rect_rgb, hm = variants[name]
        axes[0, c_idx].imshow(_draw_corners_on_patch(patch_bgr, p_corners, heatmaps=hm))
        axes[0, c_idx].set_title(name, fontsize=10, fontweight="bold")
        axes[0, c_idx].axis("off")

        axes[1, c_idx].imshow(_patch_corner_zoom(patch_bgr, p_corners))
        axes[1, c_idx].set_title("Patch Corner Zoom (TL / TR / BL / BR)", fontsize=8.5)
        axes[1, c_idx].axis("off")

        axes[2, c_idx].imshow(rect_rgb)
        axes[2, c_idx].set_title("Rectified Card (448x630)", fontsize=9)
        axes[2, c_idx].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct patch-level comparison: Raw OBB vs DP-IRLS vs Canonical UNet vs 4-Corner Heatmap"
    )
    parser.add_argument("--obb-weights", type=str, default=None)
    parser.add_argument("--unet-weights", type=str, default=None)
    parser.add_argument("--weights", type=str, default="data/weights/corner_heatmap_best.pt")
    parser.add_argument("--max-synth-scenes", type=int, default=200)
    parser.add_argument("--only-folder", type=str, default="")
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    set_seed(int(cfg["project"]["seed"]))

    exp = Experiment("corner_regressor_patch_benchmark", cfg)
    out_dir = PROJECT_ROOT / "report" / "figures" / "05_segmentation_pipeline" / "patch_corner_regressor"
    out_dir.mkdir(parents=True, exist_ok=True)

    ou_cfg = cfg.get("detector", {}).get("obb_unet", {})
    obb_weights = args.obb_weights or ou_cfg.get("obb_weights", "data/weights/yolo11l_obb_heavy_best.pt")
    unet_weights = args.unet_weights or ou_cfg.get("unet_weights", "data/weights/seg_canonical_unet_best.pt")

    unet_det = OBBCanonicalUNetDetector(
        obb_weights=obb_weights,
        unet_weights=unet_weights,
        conf=float(cfg["obb"]["inference"]["conf"]),
        iou=float(cfg["obb"]["inference"]["iou"]),
        max_det=int(cfg["obb"]["inference"]["max_det"]),
    )
    det_obb = unet_det.obb_detector
    regressor = CornerHeatmapRegressor(PROJECT_ROOT / args.weights)

    # 1. Quantitative benchmark on pokemon_seg/test single-card scenes (true perspective corners_gt)
    ds_root = PROJECT_ROOT / "data" / "datasets" / "pokemon_seg"
    img_dir = ds_root / "images" / "test"
    meta_dir = ds_root / "meta" / "test"
    mask_dir = ds_root / "masks" / "test"
    all_imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    single_imgs = []
    for p in all_imgs:
        mp = meta_dir / f"{p.stem}.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
            if len(meta.get("cards", [])) == 1:
                single_imgs.append(p)
    chosen = single_imgs[: args.max_synth_scenes]

    methods = [
        "1. Raw OBB Heavy",
        "2. Classical DP-IRLS (Peak 8-DOF)",
        "3. Canonical UNet (obb_unet)",
        "4. SOTA 4-Corner Heatmap (obb_corner)",
    ]
    errs = {m: [] for m in methods}
    ious = {m: [] for m in methods}
    clips = {m: [] for m in methods}

    for p in chosen:
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        h, w = img_bgr.shape[:2]
        meta = json.loads((meta_dir / f"{p.stem}.json").read_text())
        gt_mask_img = cv2.imread(str(mask_dir / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if gt_mask_img is None:
            continue
        gt_q = order_quad(np.asarray(meta["cards"][0]["corners_gt"], dtype=np.float64))
        gt_m = (gt_mask_img > 127).astype(np.float32)
        gt_u8 = (gt_m > 0.5).astype(np.uint8)
        gt_ring = (gt_u8 - cv2.erode(gt_u8, np.ones((9, 9), np.uint8))) > 0
        ring_area = max(float(gt_ring.sum()), 1.0)

        props = det_obb.propose_quads(img_bgr)
        if not props:
            continue
        raw_q = order_quad(props[0]["corners"])
        q_dp = refine_card_quad(img_bgr, raw_q, mask=None, outer_ramp_anchor=False, rigid_6dof_pose=False)
        res_unet = unet_det.refine_proposal(img_bgr, raw_q)
        q_unet = order_quad(res_unet.corners) if res_unet is not None else raw_q
        pred_hm = regressor.predict_from_proposal(img_bgr, raw_q)

        for m_name, q_pred in (
            ("1. Raw OBB Heavy", raw_q),
            ("2. Classical DP-IRLS (Peak 8-DOF)", q_dp),
            ("3. Canonical UNet (obb_unet)", q_unet),
            ("4. SOTA 4-Corner Heatmap (obb_corner)", pred_hm.scene_corners),
        ):
            errs[m_name].append(align_corner_error(q_pred, gt_q))
            proj_m = project_patch_mask_to_scene(q_pred, np.ones((TARGET_H, TARGET_W), dtype=np.float32), (h, w))
            inter = float(np.logical_and(proj_m > 0.5, gt_m > 0.5).sum())
            union = float(np.logical_or(proj_m > 0.5, gt_m > 0.5).sum())
            ious[m_name].append(inter / union if union > 0 else 0.0)
            clipped = float(np.logical_and(gt_ring, ~(proj_m > 0.5)).sum())
            clips[m_name].append(clipped / ring_area * 100.0)

    summary = {m: (_summarize_errs(errs[m], ious[m], clips[m]) if errs[m] else {"n": 0}) for m in methods}

    # 2. Direct patch-level visual comparison on real captures
    real_items = [] if args.skip_real else _gather_real_images()
    if args.only_folder:
        real_items = [(tag, p) for tag, p in real_items if tag == args.only_folder]
    for tag, img_path in real_items:
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        props = det_obb.propose_quads(img_bgr)
        if not props:
            continue
        raw_q = order_quad(props[0]["corners"])
        patch_bgr, M_sim, _ = extract_oriented_padded_patch(img_bgr, raw_q, patch_hw=(PATCH_H, PATCH_W), pad_frac=PAD_FRAC)

        def _to_patch(q_scene: np.ndarray) -> np.ndarray:
            ones = np.ones((4, 1), dtype=np.float64)
            return (np.hstack([order_quad(q_scene), ones]) @ M_sim.T).astype(np.float64)

        q_dp = refine_card_quad(img_bgr, raw_q, mask=None, outer_ramp_anchor=False, rigid_6dof_pose=False)
        res_unet = unet_det.refine_proposal(img_bgr, raw_q)
        q_unet = order_quad(res_unet.corners) if res_unet is not None else raw_q
        rect_unet_rgb = res_unet.patch_rgb if res_unet is not None else _warp_scene_quad(img_bgr, q_unet)
        pred_hm = regressor.predict_from_proposal(img_bgr, raw_q)

        variants = {
            "1. Frozen Raw OBB Heavy": (_to_patch(raw_q), _warp_scene_quad(img_bgr, raw_q), None),
            "2. Classical DP-IRLS": (_to_patch(q_dp), _warp_scene_quad(img_bgr, q_dp), None),
            "3. Canonical UNet (obb_unet)": (_to_patch(q_unet), rect_unet_rgb, None),
            "4. 4-Corner Heatmap (obb_corner)": (pred_hm.patch_corners, pred_hm.rectified_rgb, pred_hm.heatmaps),
        }
        _render_patch_board(
            f"Direct Patch Comparison (Same YOLO11l-OBB Heavy ROI) — {tag}/{img_path.name}",
            patch_bgr,
            variants,
            out_dir / f"patch_{img_path.stem}.png",
        )

    (exp.artifacts_dir / "corner_regressor_summary.json").write_text(json.dumps(summary, indent=2))
    exp.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

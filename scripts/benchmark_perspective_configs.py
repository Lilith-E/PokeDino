"""Side-by-side ablation of 4 perspective alignment configurations from the SAME frozen proposal.

Compares C0..C3 across:
  - Track A (Frozen Raw YOLO11-OBB quad)
  - Track B (Frozen Raw YOLO11-Seg @ 640 probability mask)
on:
  1. 120 diverse scenes from pokemon_seg/test (exact corner error, IoU, Border Clipping Rate)
  2. All 21 real captures across quality-gate-tests/, synthetic-to-real/, and two-cards/

Configurations:
  - C0: Raw Proposal (no refinement)
  - C1: Peak 8-DOF        (outer_ramp_anchor=False, rigid_6dof_pose=False)
  - C2: Outer-Ramp 8-DOF  (outer_ramp_anchor=True,  rigid_6dof_pose=False)
  - C3: Outer-Ramp 6-DOF  (outer_ramp_anchor=True,  rigid_6dof_pose=True)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.card_detector import CardDetector  # noqa: E402
from backend.experiment import Experiment, load_config, set_seed  # noqa: E402
from backend.rectifier import (  # noqa: E402
    TARGET_H,
    TARGET_W,
    fit_quad_from_mask,
    order_quad,
    refine_card_quad,
)
from backend.seg_detector import SegmentationCardDetector  # noqa: E402
from scripts.benchmark_quality_gate_perspective import (  # noqa: E402
    _boundary_alignment_score,
    _gather_real_images,
)
from scripts.benchmark_same_bbox_perspective import (  # noqa: E402
    _corner_zoom_mosaic,
    _draw_proposal_and_fitted,
    _extract_raw_yolo_seg_prob,
    _load_gt_corners_yolo_obb,
    _load_gt_mask_yolo_seg,
    _summarize_errs,
    _warp_antialiased,
    _warp_unfiltered,
)
from scripts.benchmark_segmentation_vs_obb import (  # noqa: E402
    align_corner_error,
    project_patch_mask_to_scene,
)

logger = logging.getLogger(__name__)

CONFIGS = [
    ("C1: Peak 8-DOF", False, False),
    ("C2: Outer-Ramp 8-DOF", True, False),
    ("C3: Outer-Ramp 6-DOF", True, True),
]


def _render_c0_c3_board(
    title: str,
    img_bgr: np.ndarray,
    init_quad: np.ndarray,
    mask: Optional[np.ndarray],
    variants: dict[str, tuple[np.ndarray, np.ndarray, float]],
    out_path: Path,
) -> None:
    cols = list(variants.keys())
    n_cols = len(cols)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.8 * n_cols, 11.2), dpi=150)
    fig.suptitle(title, fontsize=13.5, fontweight="bold", y=0.992)

    for c_idx, name in enumerate(cols):
        quad, patch, edge_e = variants[name]
        shift = float(np.mean(np.linalg.norm(quad - init_quad, axis=1)))

        axes[0, c_idx].imshow(_draw_proposal_and_fitted(img_bgr, init_quad, quad, mask))
        axes[0, c_idx].set_title(f"{name}\nshift={shift:.1f}px | EdgeE={edge_e:.3f}", fontsize=9.5, fontweight="bold")
        axes[0, c_idx].axis("off")

        axes[1, c_idx].imshow(_corner_zoom_mosaic(img_bgr, init_quad, quad, mask))
        axes[1, c_idx].set_title("Corner Zoom (White=Init, Blue=Fit)", fontsize=8.5)
        axes[1, c_idx].axis("off")

        axes[2, c_idx].imshow(patch)
        axes[2, c_idx].set_title("Rectified Card (448x630)", fontsize=9)
        axes[2, c_idx].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-config perspective ablation C0..C3")
    parser.add_argument("--max-synth-scenes", type=int, default=120)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    set_seed(int(cfg["project"]["seed"]))

    exp = Experiment("perspective_configs_ablation", cfg)
    out_dir = PROJECT_ROOT / "report" / "figures" / "05_segmentation_pipeline" / "config_ablation_cards"
    out_dir.mkdir(parents=True, exist_ok=True)

    det_obb = CardDetector(
        conf=float(cfg["obb"]["inference"]["conf"]),
        iou=float(cfg["obb"]["inference"]["iou"]),
        max_det=int(cfg["obb"]["inference"]["max_det"]),
        use_pbl=False,
    )
    det_seg = SegmentationCardDetector(
        backend="seg_yolo",
        weights_path=str(PROJECT_ROOT / cfg["segmentation"]["yolo_seg"]["weights"]),
        imgsz=640,
    )

    # 1. Synthetic benchmark (120 diverse scenes)
    ds_root = PROJECT_ROOT / "data" / "datasets" / "pokemon_seg"
    img_dir = ds_root / "images" / "test"
    lbl_obb_dir = ds_root / "labels_obb" / "test"
    lbl_seg_dir = ds_root / "labels" / "test"
    all_imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    pos_imgs = [p for p in all_imgs if (lbl_obb_dir / f"{p.stem}.txt").exists() and (lbl_obb_dir / f"{p.stem}.txt").read_text().strip()]
    chosen = pos_imgs[: args.max_synth_scenes]

    names_a = ["C0: Raw OBB"] + [c[0] for c in CONFIGS]
    names_b = ["C0: Mask Base Fit"] + [c[0] for c in CONFIGS]
    errs_a = {k: [] for k in names_a}
    ious_a = {k: [] for k in names_a}
    clips_a = {k: [] for k in names_a}
    errs_b = {k: [] for k in names_b}
    ious_b = {k: [] for k in names_b}
    clips_b = {k: [] for k in names_b}

    logger.info("Running C0..C3 ablation on %d synthetic scenes...", len(chosen))
    for p in chosen:
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        h, w = img_bgr.shape[:2]
        gt_quads = _load_gt_corners_yolo_obb(lbl_obb_dir / f"{p.stem}.txt", w, h)
        gt_masks = _load_gt_mask_yolo_seg(lbl_seg_dir / f"{p.stem}.txt", w, h)
        if not gt_quads or not gt_masks:
            continue
        gt_q = order_quad(gt_quads[0])
        gt_m = gt_masks[0]
        gt_u8 = (gt_m > 0.5).astype(np.uint8)
        gt_ring = (gt_u8 - cv2.erode(gt_u8, np.ones((9, 9), np.uint8))) > 0
        ring_area = max(float(gt_ring.sum()), 1.0)

        def _record(store_e, store_i, store_c, key, q_pred):
            store_e[key].append(align_corner_error(q_pred, gt_q))
            proj_m = project_patch_mask_to_scene(q_pred, np.ones((TARGET_H, TARGET_W), dtype=np.float32), (h, w))
            inter = float(np.logical_and(proj_m > 0.5, gt_m > 0.5).sum())
            union = float(np.logical_or(proj_m > 0.5, gt_m > 0.5).sum())
            store_i[key].append(inter / union if union > 0 else 0.0)
            clipped = float(np.logical_and(gt_ring, ~(proj_m > 0.5)).sum())
            store_c[key].append(clipped / ring_area * 100.0)

        raw_dets = det_obb.detect(img_bgr)
        if raw_dets:
            raw_q = order_quad(raw_dets[0].corners)
            _record(errs_a, ious_a, clips_a, "C0: Raw OBB", raw_q)
            for cname, ramp, rigid in CONFIGS:
                q_c = refine_card_quad(img_bgr, raw_q, mask=None, outer_ramp_anchor=ramp, rigid_6dof_pose=rigid)
                _record(errs_a, ious_a, clips_a, cname, q_c)

        prob_mask = _extract_raw_yolo_seg_prob(det_seg, img_bgr)
        if prob_mask is not None:
            q_base = fit_quad_from_mask(prob_mask, edge_fit_method="huber", min_span_frac=0.30, max_side_tilt_rad=0.20)
            if q_base is not None:
                q_base = order_quad(q_base)
                _record(errs_b, ious_b, clips_b, "C0: Mask Base Fit", q_base)
                for cname, ramp, rigid in CONFIGS:
                    q_c = refine_card_quad(img_bgr, q_base, mask=prob_mask, outer_ramp_anchor=ramp, rigid_6dof_pose=rigid)
                    _record(errs_b, ious_b, clips_b, cname, q_c)

    # 2. All 21 real captures in experiments/
    real_items = _gather_real_images()
    real_energy_a = {k: [] for k in names_a}
    real_energy_b = {k: [] for k in names_b}
    logger.info("Rendering C0..C3 visual boards across %d real captures...", len(real_items))

    for tag, img_path in real_items:
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        raw_dets = det_obb.detect(img_bgr)
        if raw_dets:
            raw_q = order_quad(raw_dets[0].corners)
            var_a = {"C0: Raw OBB": (raw_q, _warp_unfiltered(img_bgr, raw_q), _boundary_alignment_score(img_bgr, raw_q))}
            real_energy_a["C0: Raw OBB"].append(var_a["C0: Raw OBB"][2])
            for cname, ramp, rigid in CONFIGS:
                q_c = refine_card_quad(img_bgr, raw_q, mask=None, outer_ramp_anchor=ramp, rigid_6dof_pose=rigid)
                e_c = _boundary_alignment_score(img_bgr, q_c)
                var_a[cname] = (q_c, _warp_antialiased(img_bgr, q_c), e_c)
                real_energy_a[cname].append(e_c)
            _render_c0_c3_board(
                f"Track A (Frozen Raw OBB) — {tag}/{img_path.name}",
                img_bgr,
                raw_q,
                None,
                var_a,
                out_dir / f"obb_{img_path.stem}.png",
            )

        prob_mask = _extract_raw_yolo_seg_prob(det_seg, img_bgr)
        if prob_mask is not None:
            q_base = fit_quad_from_mask(prob_mask, edge_fit_method="huber", min_span_frac=0.30, max_side_tilt_rad=0.20)
            if q_base is not None:
                q_base = order_quad(q_base)
                var_b = {"C0: Mask Base Fit": (q_base, _warp_unfiltered(img_bgr, q_base), _boundary_alignment_score(img_bgr, q_base))}
                real_energy_b["C0: Mask Base Fit"].append(var_b["C0: Mask Base Fit"][2])
                for cname, ramp, rigid in CONFIGS:
                    q_c = refine_card_quad(img_bgr, q_base, mask=prob_mask, outer_ramp_anchor=ramp, rigid_6dof_pose=rigid)
                    e_c = _boundary_alignment_score(img_bgr, q_c)
                    var_b[cname] = (q_c, _warp_antialiased(img_bgr, q_c), e_c)
                    real_energy_b[cname].append(e_c)
                _render_c0_c3_board(
                    f"Track B (Frozen Raw YOLO11-Seg Mask) — {tag}/{img_path.name}",
                    img_bgr,
                    q_base,
                    prob_mask,
                    var_b,
                    out_dir / f"seg_{img_path.stem}.png",
                )

    summary = {
        "track_a_synth": {k: _summarize_errs(errs_a[k], ious_a[k], clips_a[k]) for k in names_a},
        "track_b_synth": {k: _summarize_errs(errs_b[k], ious_b[k], clips_b[k]) for k in names_b},
        "track_a_real_edge_energy": {k: round(float(np.mean(real_energy_a[k])), 4) for k in names_a},
        "track_b_real_edge_energy": {k: round(float(np.mean(real_energy_b[k])), 4) for k in names_b},
    }

    # Plot master summary chart
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=160)
    x = np.arange(len(names_a))
    w = 0.36
    med_a = [summary["track_a_synth"][k]["median_px"] for k in names_a]
    med_b = [summary["track_b_synth"][k]["median_px"] for k in names_b]
    clip_a = [summary["track_a_synth"][k]["border_clip_pct_mean"] for k in names_a]
    clip_b = [summary["track_b_synth"][k]["border_clip_pct_mean"] for k in names_b]
    rea_a = [summary["track_a_real_edge_energy"][k] for k in names_a]
    rea_b = [summary["track_b_real_edge_energy"][k] for k in names_b]

    for ax, va, vb, title, ylabel in (
        (axes[0], med_a, med_b, "Median Corner Error (px, Lower=Better)", "Pixels"),
        (axes[1], clip_a, clip_b, "Border Clipping Rate (%, Lower=Better)", "Clipped Border (%)"),
        (axes[2], rea_a, rea_b, "20 Real Captures Boundary Energy (Higher=Better)", "Mean Edge Energy"),
    ):
        b1 = ax.bar(x - w / 2, va, width=w, color="#F59E0B", label="Track A (Same Raw OBB)")
        b2 = ax.bar(x + w / 2, vb, width=w, color="#0284C7", label="Track B (Same Seg Mask)")
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(va + vb) * 0.02, f"{b.get_height():.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["C0 Base", "C1 Peak 8-DOF", "C2 Ramp 8-DOF", "C3 Ramp 6-DOF"], rotation=16, ha="right", fontsize=8.5)
        ax.set_ylim(0, max(va + vb) * 1.18)
        ax.set_title(title, fontweight="bold", fontsize=10.5)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(exp.plots_dir / "ablation_summary_c0_c3.png", dpi=160)
    plt.savefig(out_dir / "ablation_summary_c0_c3.png", dpi=160)
    plt.close(fig)

    (exp.artifacts_dir / "ablation_c0_c3.json").write_text(json.dumps(summary, indent=2))
    exp.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

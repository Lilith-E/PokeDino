"""Frozen-proposal perspective alignment benchmark across all real captures in experiments/.

Evaluates multiple perspective alignment methods starting from the EXACT SAME frozen
card proposal (Track A: frozen raw YOLO11-OBB box; Track B: frozen raw YOLO11-Seg @ 640
probability mask) across:
  - experiments/quality-gate-tests/ (12 challenging real photos)
  - experiments/synthetic-to-real/ (7 smartphone captures)
  - experiments/two-cards/ (2 multi-card captures)

Generates high-resolution corner-zoom + rectified-card visual boards and computes
boundary-alignment coherence and aspect/keystone consistency across all real samples.
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
    _compute_normalized_boundary_energy,
    fit_quad_from_mask,
    order_quad,
    rectify_from_mask,
    refine_card_quad,
    warp_perspective_antialiased,
)
from backend.seg_detector import SegmentationCardDetector  # noqa: E402
from scripts.benchmark_same_bbox_perspective import (  # noqa: E402
    _corner_zoom_mosaic,
    _draw_proposal_and_fitted,
    _extract_raw_yolo_seg_prob,
    _warp_antialiased,
    _warp_unfiltered,
)

logger = logging.getLogger(__name__)


def _gather_real_images() -> list[tuple[str, Path]]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    folders = [
        ("quality-gate-tests", PROJECT_ROOT / "experiments" / "quality-gate-tests"),
        ("synthetic-to-real", PROJECT_ROOT / "experiments" / "synthetic-to-real"),
        ("two-cards", PROJECT_ROOT / "experiments" / "two-cards"),
    ]
    items: list[tuple[str, Path]] = []
    for tag, d in folders:
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in exts and "reference" not in p.stem.lower():
                items.append((tag, p))
    return items


def _boundary_alignment_score(img_bgr: np.ndarray, quad: np.ndarray) -> float:
    """Measure contrast-normalized boundary alignment energy along the fitted quad perimeter."""
    pad = 24
    w_pad = TARGET_W + 2 * pad
    h_pad = TARGET_H + 2 * pad
    dst_pad = np.array(
        [[pad, pad], [pad + TARGET_W - 1, pad], [pad + TARGET_W - 1, pad + TARGET_H - 1], [pad, pad + TARGET_H - 1]],
        dtype=np.float32,
    )
    try:
        M_pad = cv2.getPerspectiveTransform(order_quad(quad).astype(np.float32), dst_pad)
    except cv2.error:
        return 0.0
    patch = cv2.warpPerspective(img_bgr, M_pad, (w_pad, h_pad), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ey, ex = _compute_normalized_boundary_energy(patch, pad, TARGET_W, TARGET_H, band_px=12)
    ts = np.linspace(0.14, 0.86, 64)
    xs = (pad + ts * (TARGET_W - 1)).astype(np.int32)
    ys = (pad + ts * (TARGET_H - 1)).astype(np.int32)
    s_top = float(np.mean(ey[pad, xs]))
    s_bot = float(np.mean(ey[pad + TARGET_H - 1, xs]))
    s_left = float(np.mean(ex[ys, pad]))
    s_right = float(np.mean(ex[ys, pad + TARGET_W - 1]))
    return round(0.25 * (s_top + s_bot + s_left + s_right), 4)


def _render_multi_method_board(
    title: str,
    img_bgr: np.ndarray,
    init_quad: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    variants: dict[str, tuple[np.ndarray, np.ndarray, float]],
    out_path: Path,
) -> None:
    cols = list(variants.keys())
    n_cols = len(cols)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.9 * n_cols, 11.4), dpi=150)
    if n_cols == 1:
        axes = axes[:, None]
    fig.suptitle(title, fontsize=13.5, fontweight="bold", y=0.992)

    for c_idx, name in enumerate(cols):
        quad, patch, bscore = variants[name]
        shift = float(np.mean(np.linalg.norm(quad - init_quad, axis=1))) if init_quad is not None else 0.0

        axes[0, c_idx].imshow(_draw_proposal_and_fitted(img_bgr, init_quad, quad, mask))
        axes[0, c_idx].set_title(f"{name}\nshift={shift:.1f}px | EdgeE={bscore:.3f}", fontsize=9.5, fontweight="bold")
        axes[0, c_idx].axis("off")

        axes[1, c_idx].imshow(_corner_zoom_mosaic(img_bgr, init_quad, quad, mask))
        axes[1, c_idx].set_title("Corner Zoom (White=Frozen Init, Blue=Fit)", fontsize=8.5)
        axes[1, c_idx].axis("off")

        axes[2, c_idx].imshow(patch)
        axes[2, c_idx].set_title("Rectified Card (448x630)", fontsize=9)
        axes[2, c_idx].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-proposal perspective benchmark across all real captures")
    parser.add_argument("--max-boards", type=int, default=12)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    set_seed(int(cfg["project"]["seed"]))

    exp = Experiment("quality_gate_perspective_benchmark", cfg)
    plots_dir = exp.plots_dir
    report_dir = PROJECT_ROOT / "report" / "figures" / "05_segmentation_pipeline" / "quality_gate_perspective"
    report_dir.mkdir(parents=True, exist_ok=True)

    det_obb = CardDetector(
        conf=float(cfg["obb"]["inference"]["conf"]),
        iou=float(cfg["obb"]["inference"]["iou"]),
        max_det=int(cfg["obb"]["inference"]["max_det"]),
        use_pbl=True,
    )
    det_seg = SegmentationCardDetector(
        backend="seg_yolo",
        weights_path=str(PROJECT_ROOT / cfg["segmentation"]["yolo_seg"]["weights"]),
        imgsz=640,
    )

    items = _gather_real_images()
    logger.info("Evaluating frozen-proposal perspective methods across %d real captures...", len(items))

    rows: list[dict] = []
    board_count = 0
    for tag, img_path in items:
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        # Track A: Frozen Raw OBB Box
        det_obb.use_pbl = False
        raw_dets = det_obb.detect(img_bgr)
        if raw_dets:
            raw_q = order_quad(raw_dets[0].corners)
            det_obb.use_pbl = True

            # A1: Raw OBB
            p_a1 = _warp_unfiltered(img_bgr, raw_q)
            e_a1 = _boundary_alignment_score(img_bgr, raw_q)

            # A2: Legacy PBL
            locked_q = det_obb._lock_physical_card(img_bgr, raw_q.copy())
            p_a2, q_a2 = det_obb._refine_quad_lines(img_bgr, locked_q)
            if p_a2 is None or det_obb._quad_distance(q_a2, locked_q) > 6.5:
                q_a2 = locked_q
                p_a2 = _warp_antialiased(img_bgr, q_a2)
            q_a2 = order_quad(q_a2)
            e_a2 = _boundary_alignment_score(img_bgr, q_a2)

            # A3: Unified DP-IRLS + AA (starting from exact same raw_q!)
            q_a3 = refine_card_quad(img_bgr, raw_q, mask=None)
            p_a3 = _warp_antialiased(img_bgr, q_a3)
            e_a3 = _boundary_alignment_score(img_bgr, q_a3)

            for m_name, q_m, e_m in (
                ("Raw OBB", raw_q, e_a1),
                ("OBB + Legacy PBL", q_a2, e_a2),
                ("OBB + Unified DP-IRLS + AA", q_a3, e_a3),
            ):
                rows.append({"folder": tag, "image": img_path.name, "track": "A_OBB", "method": m_name, "edge_energy": e_m})

            if tag == "quality-gate-tests" and board_count < args.max_boards:
                var_a = {
                    "Raw OBB": (raw_q, p_a1, e_a1),
                    "OBB + Legacy PBL": (q_a2, p_a2, e_a2),
                    "OBB + Unified DP-IRLS + AA": (q_a3, p_a3, e_a3),
                }
                _render_multi_method_board(
                    f"Track A (Frozen Raw OBB) — {tag}/{img_path.name}",
                    img_bgr,
                    raw_q,
                    None,
                    var_a,
                    report_dir / f"obb_{img_path.stem}.png",
                )

        # Track B: Frozen Raw YOLO11-Seg @ 640 Mask
        prob_mask = _extract_raw_yolo_seg_prob(det_seg, img_bgr)
        if prob_mask is not None:
            bin_mask = (prob_mask >= 0.5).astype(np.float32)
            cnts, _ = cv2.findContours((bin_mask > 0.5).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                largest = max(cnts, key=cv2.contourArea)
                q_b1 = order_quad(cv2.boxPoints(cv2.minAreaRect(largest)).astype(np.float64))
                p_b1 = _warp_unfiltered(img_bgr, q_b1)
                e_b1 = _boundary_alignment_score(img_bgr, q_b1)

                q_b2_cand = fit_quad_from_mask(bin_mask, edge_fit_method="ransac", min_span_frac=0.0, max_side_tilt_rad=1.57)
                q_b2 = order_quad(q_b2_cand if q_b2_cand is not None else q_b1)
                p_b2 = _warp_unfiltered(img_bgr, q_b2)
                e_b2 = _boundary_alignment_score(img_bgr, q_b2)

                q_b3_cand = fit_quad_from_mask(prob_mask, edge_fit_method="huber", min_span_frac=0.30, max_side_tilt_rad=0.20)
                q_b3 = order_quad(q_b3_cand if q_b3_cand is not None else q_b1)
                p_b3 = _warp_antialiased(img_bgr, q_b3)
                e_b3 = _boundary_alignment_score(img_bgr, q_b3)

                res_b4 = rectify_from_mask(img_bgr, prob_mask, edge_fit_method="huber", refine_edges=True, antialiased_warp=True)
                if res_b4 is not None:
                    q_b4, p_b4 = order_quad(res_b4.corners), res_b4.patch_rgb
                else:
                    q_b4, p_b4 = q_b3, p_b3
                e_b4 = _boundary_alignment_score(img_bgr, q_b4)

                for m_name, e_m in (
                    ("Mask -> minAreaRect", e_b1),
                    ("Mask -> Legacy Binary Fit", e_b2),
                    ("Mask -> Upgraded 0.5-Iso", e_b3),
                    ("Mask -> Unified DP-IRLS + AA", e_b4),
                ):
                    rows.append({"folder": tag, "image": img_path.name, "track": "B_SEG", "method": m_name, "edge_energy": e_m})

                if tag == "quality-gate-tests" and board_count < args.max_boards:
                    var_b = {
                        "Mask -> minAreaRect": (q_b1, p_b1, e_b1),
                        "Mask -> Legacy Binary Fit": (q_b2, p_b2, e_b2),
                        "Mask -> Upgraded 0.5-Iso": (q_b3, p_b3, e_b3),
                        "Mask -> Unified DP-IRLS + AA": (q_b4, p_b4, e_b4),
                    }
                    _render_multi_method_board(
                        f"Track B (Frozen Raw YOLO11-Seg Mask) — {tag}/{img_path.name}",
                        img_bgr,
                        q_b1,
                        prob_mask,
                        var_b,
                        report_dir / f"seg_{img_path.stem}.png",
                    )
                    board_count += 1

    # Aggregate summary across all 21 real captures
    summary: dict[str, dict[str, float]] = {"track_a_obb": {}, "track_b_seg": {}}
    for track_key, label in (("A_OBB", "track_a_obb"), ("B_SEG", "track_b_seg")):
        methods = list(dict.fromkeys(r["method"] for r in rows if r["track"] == track_key))
        for m in methods:
            vals = [r["edge_energy"] for r in rows if r["track"] == track_key and r["method"] == m]
            summary[label][m] = {"n": len(vals), "mean_edge_energy": round(float(np.mean(vals)), 4)}

    # Plot summary comparison
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=160)
    for ax, (t_key, t_title) in zip(axes, (("track_a_obb", "Track A: Frozen Raw OBB (All Real Captures)"), ("track_b_seg", "Track B: Frozen Raw Seg Mask (All Real Captures)"))):
        names = list(summary[t_key].keys())
        vals = [summary[t_key][n]["mean_edge_energy"] for n in names]
        bars = ax.bar(range(len(names)), vals, color=["#94A3B8", "#F59E0B", "#0284C7", "#10B981"][: len(names)], width=0.55)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=16, ha="right", fontsize=8.5)
        ax.set_ylim(0, max(vals + [0.5]) * 1.18)
        ax.set_ylabel("Mean Perimeter Boundary Energy (0-1)")
        ax.set_title(t_title, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots_dir / "quality_gate_perspective_summary.png", dpi=160)
    plt.savefig(report_dir / "quality_gate_perspective_summary.png", dpi=160)
    plt.close(fig)

    payload = {"summary": summary, "rows": rows}
    (exp.artifacts_dir / "quality_gate_perspective.json").write_text(json.dumps(payload, indent=2))
    exp.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

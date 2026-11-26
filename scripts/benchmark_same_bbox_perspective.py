"""Paired visual & quantitative benchmark of perspective alignment starting from the SAME bounding box / mask.

Isolates rectification quality from upstream detector variance by feeding the EXACT SAME
initial proposal (Track A: raw YOLO11-OBB quad; Track B: raw YOLO11-Seg @ 640 probability map)
through progressive alignment stages on both:
  - ``pokemon_seg/test`` (n=200 paired synthetic scenes with ground-truth corners & masks)
  - Real smartphone captures with reference scans (charizard-damaged, giratina-kanda-damaged,
    realgengar-1/3/5)
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
from scipy.stats import wilcoxon  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.card_detector import CardDetector  # noqa: E402
from backend.experiment import Experiment, load_config, set_seed  # noqa: E402
from backend.rectifier import (  # noqa: E402
    TARGET_H,
    TARGET_W,
    _refine_quad_from_image_edges,
    align_patch_to_reference_homography,
    build_alignment_feature_map,
    fit_quad_from_mask,
    order_quad,
    rectify_from_mask,
    warp_perspective_antialiased,
)
from backend.seg_detector import SegmentationCardDetector  # noqa: E402
from scripts.benchmark_segmentation_vs_obb import (  # noqa: E402
    align_corner_error,
    project_patch_mask_to_scene,
)


def _load_gt_corners_yolo_obb(lbl_path: Path, w: int, h: int) -> list[np.ndarray]:
    quads = []
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 9:
            coords = np.array([float(v) for v in parts[1:]], dtype=np.float64).reshape(4, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            quads.append(coords)
    return quads


def _load_gt_mask_yolo_seg(lbl_path: Path, w: int, h: int) -> list[np.ndarray]:
    masks = []
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 7:
            pts = np.array([float(v) for v in parts[1:]], dtype=np.float64).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [pts.astype(np.int32)], 1)
            masks.append(m.astype(np.float32))
    return masks

logger = logging.getLogger(__name__)

CORNER_COLORS_BGR = [
    (0, 255, 0),     # TL: green
    (255, 180, 0),   # TR: cyan-blue
    (0, 0, 255),     # BR: red
    (255, 0, 255),   # BL: magenta
]
CORNER_NAMES = ["TL", "TR", "BR", "BL"]

REAL_CASES = [
    ("charizard-reference.webp", "charizard-damaged.jpeg"),
    ("giratina-kanda-reference.jpeg", "giratina-kanda-damaged.jpg"),
    ("reference-gengarprime.webp", "realgengar-1.jpg"),
    ("reference-gengarprime.webp", "realgengar-3.jpg"),
    ("reference-gengarprime.webp", "realgengar-5.jpg"),
]


def _warp_unfiltered(img_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    q = order_quad(quad).astype(np.float32)
    dst = np.array([[0, 0], [TARGET_W - 1, 0], [TARGET_W - 1, TARGET_H - 1], [0, TARGET_H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    warped = cv2.warpPerspective(img_bgr, M, (TARGET_W, TARGET_H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)


def _warp_antialiased(img_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    q = order_quad(quad).astype(np.float32)
    dst = np.array([[0, 0], [TARGET_W - 1, 0], [TARGET_W - 1, TARGET_H - 1], [0, TARGET_H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    warped = warp_perspective_antialiased(
        img_bgr, M, (TARGET_W, TARGET_H), quad_corners=q, flags=cv2.INTER_LANCZOS4, border_mode=cv2.BORDER_REPLICATE
    )
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)


def _soft_mask_from_quad(shape_hw: tuple[int, int], quad: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(m, [order_quad(quad).astype(np.int32)], 1.0)
    return m


def _extract_raw_yolo_seg_prob(det_seg: SegmentationCardDetector, img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Return raw continuous [0, 1] probability mask for the top YOLO11-Seg instance."""
    h_img, w_img = img_bgr.shape[:2]
    res = det_seg.model.predict(
        img_bgr,
        conf=det_seg.conf,
        iou=det_seg.iou,
        imgsz=det_seg.imgsz,
        max_det=det_seg.max_det,
        verbose=False,
    )[0]
    if res.masks is None or len(res.masks) == 0:
        return None
    confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.ones(len(res.masks))
    best_i = int(np.argmax(confs))
    raw = res.masks.data.cpu().numpy()[best_i].astype(np.float32)
    if raw.shape[:2] != (h_img, w_img):
        raw = cv2.resize(raw, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
    raw = np.clip(raw, 0.0, 1.0)
    if det_seg.crop_refine:
        raw = det_seg._refine_mask_on_roi(
            img_bgr,
            raw,
            {"conf": det_seg.conf, "iou": det_seg.iou, "imgsz": det_seg.imgsz},
        )
    return raw


def _run_obb_variants_from_same_quad(
    det_obb: CardDetector,
    img_bgr: np.ndarray,
    raw_quad: np.ndarray,
    ref_map: Optional[np.ndarray] = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run A1..A4 starting from the exact same raw OBB quadrilateral."""
    q_raw = order_quad(raw_quad)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # A1: Raw OBB
    out["Raw OBB"] = (q_raw, _warp_unfiltered(img_bgr, q_raw))

    # A2: Legacy PBL (Color-Ring + legacy _refine_quad_lines with 6.5 px gate)
    locked_q = det_obb._lock_physical_card(img_bgr, q_raw.copy())
    pbl_patch, pbl_q = det_obb._refine_quad_lines(img_bgr, locked_q)
    if pbl_patch is None or det_obb._quad_distance(pbl_q, locked_q) > 6.5:
        pbl_q = locked_q
        pbl_patch = _warp_antialiased(img_bgr, pbl_q)
    out["OBB + Legacy PBL"] = (order_quad(pbl_q), pbl_patch)

    # A3: 2-Pass Proximity-Weighted Edge Snap + AA (same initial OBB quad)
    init_mask = _soft_mask_from_quad(img_bgr.shape[:2], q_raw)
    snapped_q = _refine_quad_from_image_edges(img_bgr, q_raw, init_mask, n_iters=2)
    out["OBB + 2-Pass Edge Snap + AA"] = (order_quad(snapped_q), _warp_antialiased(img_bgr, snapped_q))

    # A4: + Stage-6 Reference-Guided Homography ECC
    if ref_map is not None:
        aligned_rgb, _, _ = align_patch_to_reference_homography(
            out["OBB + 2-Pass Edge Snap + AA"][1], ref_map
        )
        out["OBB + Edge Snap + ECC"] = (order_quad(snapped_q), aligned_rgb)

    return out


def _run_seg_variants_from_same_mask(
    img_bgr: np.ndarray,
    prob_mask: np.ndarray,
    ref_map: Optional[np.ndarray] = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run B1..B5 starting from the exact same segmentation probability map."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    bin_mask = (prob_mask >= 0.5).astype(np.float32)

    # B1: minAreaRect on binary mask
    cnts, _ = cv2.findContours((bin_mask > 0.5).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        q_rect = order_quad(cv2.boxPoints(cv2.minAreaRect(largest)).astype(np.float64))
    else:
        return out
    out["Mask -> minAreaRect"] = (q_rect, _warp_unfiltered(img_bgr, q_rect))

    # B2: Legacy Binary Fit (thresholded, no span/tilt gate, 1-pass unweighted, no snap)
    q_leg = fit_quad_from_mask(
        bin_mask,
        edge_fit_method="ransac",
        corner_excl_frac=0.14,
        min_side_inliers=12,
        ransac_thresh_px=2.5,
        ransac_iters=120,
        min_span_frac=0.0,
        max_side_tilt_rad=1.57,
    )
    if q_leg is None:
        q_leg = q_rect
    out["Mask -> Legacy Binary Fit"] = (order_quad(q_leg), _warp_unfiltered(img_bgr, q_leg))

    # B3: Upgraded Fit (0.5-isocontour + span/tilt gate + 1D parallel recovery, no edge snap)
    q_upg = fit_quad_from_mask(
        prob_mask,
        edge_fit_method="huber",
        corner_excl_frac=0.14,
        min_side_inliers=12,
        ransac_thresh_px=2.5,
        ransac_iters=120,
        min_span_frac=0.30,
        max_side_tilt_rad=0.20,
    )
    if q_upg is None:
        q_upg = q_rect
    out["Mask -> Upgraded Fit (No Snap)"] = (order_quad(q_upg), _warp_antialiased(img_bgr, q_upg))

    # B4: Full Upgraded (2-Pass Edge Snap + AA)
    res = rectify_from_mask(
        img_bgr,
        prob_mask,
        edge_fit_method="huber",
        refine_edges=True,
        refine_iters=2,
        antialiased_warp=True,
        min_span_frac=0.30,
        max_side_tilt_rad=0.20,
    )
    if res is not None:
        q_full, patch_full = order_quad(res.corners), res.patch_rgb
    else:
        q_full, patch_full = order_quad(q_upg), out["Mask -> Upgraded Fit (No Snap)"][1]
    out["Mask -> Upgraded + 2-Pass Snap"] = (q_full, patch_full)

    # B5: + Stage-6 Reference-Guided Homography ECC
    if ref_map is not None:
        aligned_rgb, _, _ = align_patch_to_reference_homography(patch_full, ref_map)
        out["Mask -> Full + Stage-6 ECC"] = (q_full, aligned_rgb)

    return out


def _structural_ncc(patch_rgb: np.ndarray, ref_map: np.ndarray) -> float:
    fmap = build_alignment_feature_map(patch_rgb, align_size_wh=(224, 315))
    return float(np.mean(fmap * ref_map))


def _residual_heatmap_rgb(patch_rgb: np.ndarray, ref_rgb: np.ndarray) -> np.ndarray:
    """Visualize gradient/luminance mismatch against canonical reference."""
    p_gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    r_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Local mean-normalized diff to highlight geometric misalignment over global exposure
    p_norm = (p_gray - cv2.GaussianBlur(p_gray, (0, 0), 9.0))
    r_norm = (r_gray - cv2.GaussianBlur(r_gray, (0, 0), 9.0))
    diff = np.clip(np.abs(p_norm - r_norm) * 3.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _draw_proposal_and_fitted(
    img_bgr: np.ndarray,
    init_quad: Optional[np.ndarray],
    fitted_quad: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    vis = img_bgr.copy()
    if mask is not None:
        m = mask >= 0.5
        cnts, _ = cv2.findContours(m.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 230, 255), max(2, vis.shape[1] // 450))
    if init_quad is not None:
        pts_i = init_quad.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts_i], True, (240, 240, 240), max(2, vis.shape[1] // 450), cv2.LINE_AA)
    pts = fitted_quad.astype(np.int32).reshape(-1, 1, 2)
    thick = max(3, vis.shape[1] // 340)
    cv2.polylines(vis, [pts], True, (255, 100, 20), thick, cv2.LINE_AA)
    r = max(8, vis.shape[1] // 140)
    for i, (cx, cy) in enumerate(fitted_quad):
        c = CORNER_COLORS_BGR[i]
        cv2.circle(vis, (int(round(cx)), int(round(cy))), r, c, -1, cv2.LINE_AA)
        cv2.putText(
            vis,
            CORNER_NAMES[i],
            (int(round(cx)) + r + 4, int(round(cy)) + r),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.65, vis.shape[1] / 1450.0),
            c,
            max(2, thick // 2),
            cv2.LINE_AA,
        )
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def _corner_zoom_mosaic(
    img_bgr: np.ndarray,
    init_quad: Optional[np.ndarray],
    fitted_quad: np.ndarray,
    mask: Optional[np.ndarray] = None,
    patch_half: int = 100,
) -> np.ndarray:
    vis_rgb = _draw_proposal_and_fitted(img_bgr, init_quad, fitted_quad, mask)
    h, w = vis_rgb.shape[:2]
    tiles = []
    for i, (cx, cy) in enumerate(fitted_quad):
        cx_i, cy_i = int(round(cx)), int(round(cy))
        x0, x1 = max(0, cx_i - patch_half), min(w, cx_i + patch_half)
        y0, y1 = max(0, cy_i - patch_half), min(h, cy_i + patch_half)
        crop = vis_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            crop = np.zeros((2 * patch_half, 2 * patch_half, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (200, 200), interpolation=cv2.INTER_LINEAR)
        cv2.putText(crop, CORNER_NAMES[i], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(crop)
    return np.vstack([np.hstack([tiles[0], tiles[1]]), np.hstack([tiles[3], tiles[2]])])


def _render_visual_board(
    title: str,
    img_bgr: np.ndarray,
    ref_rgb: np.ndarray,
    ref_map: np.ndarray,
    init_quad: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    variants: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    cols = list(variants.keys())
    n_cols = len(cols)
    fig, axes = plt.subplots(4, n_cols, figsize=(3.9 * n_cols, 15.2), dpi=150)
    if n_cols == 1:
        axes = axes[:, None]
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.992)

    for c_idx, name in enumerate(cols):
        quad, patch = variants[name]
        ncc = _structural_ncc(patch, ref_map)
        shift = float(np.mean(np.linalg.norm(quad - init_quad, axis=1))) if init_quad is not None else 0.0

        axes[0, c_idx].imshow(_draw_proposal_and_fitted(img_bgr, init_quad, quad, mask))
        axes[0, c_idx].set_title(f"{name}\nshift={shift:.2f}px", fontsize=10, fontweight="bold")
        axes[0, c_idx].axis("off")

        axes[1, c_idx].imshow(_corner_zoom_mosaic(img_bgr, init_quad, quad, mask))
        axes[1, c_idx].set_title("Corner Zoom (White=Init, Blue=Fit)", fontsize=8.5)
        axes[1, c_idx].axis("off")

        axes[2, c_idx].imshow(patch)
        axes[2, c_idx].set_title("Rectified Card (448x630)", fontsize=9)
        axes[2, c_idx].axis("off")

        axes[3, c_idx].imshow(_residual_heatmap_rgb(patch, ref_rgb))
        axes[3, c_idx].set_title(f"Align Residual | NCC={ncc:.3f}", fontsize=9.5, fontweight="bold")
        axes[3, c_idx].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.975])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _summarize_errs(errs: list[float], ious: list[float], clips: Optional[list[float]] = None) -> dict[str, float]:
    ea = np.asarray(errs, dtype=np.float64)
    ia = np.asarray(ious, dtype=np.float64)
    res = {
        "n": int(len(ea)),
        "mean_px": round(float(np.mean(ea)), 4),
        "median_px": round(float(np.median(ea)), 4),
        "p90_px": round(float(np.percentile(ea, 90)), 4),
        "pct_lt_1px": round(float((ea <= 1.0).mean() * 100.0), 2),
        "pct_lt_2px": round(float((ea <= 2.0).mean() * 100.0), 2),
        "pct_lt_3px": round(float((ea <= 3.0).mean() * 100.0), 2),
        "mask_iou_mean": round(float(np.mean(ia)), 5),
    }
    if clips:
        res["border_clip_pct_mean"] = round(float(np.mean(clips)), 3)
        res["border_clip_pct_p90"] = round(float(np.percentile(clips, 90)), 3)
    return res


def _plot_summary(
    synth_obb_errs: dict[str, list[float]],
    synth_seg_errs: dict[str, list[float]],
    synth_summary: dict,
    real_ncc_rows: list[dict],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), dpi=160)

    # Panel 1: Track A CDF
    ax = axes[0, 0]
    grid = np.linspace(0, 15, 300)
    for k, errs in synth_obb_errs.items():
        ea = np.asarray(errs)
        cdf = [(ea <= g).mean() * 100.0 for g in grid]
        med = float(np.median(ea))
        ax.plot(grid, cdf, linewidth=2.2, label=f"{k} (med={med:.2f}px)")
    ax.set_title("Track A: Same Raw OBB Proposal (n=200 Scenes)", fontweight="bold")
    ax.set_xlabel("Corner Error (px)")
    ax.set_ylabel("Scenes <= Threshold (%)")
    ax.grid(alpha=0.28)
    ax.legend(fontsize=8.5, loc="lower right")

    # Panel 2: Track B CDF
    ax = axes[0, 1]
    for k, errs in synth_seg_errs.items():
        ea = np.asarray(errs)
        cdf = [(ea <= g).mean() * 100.0 for g in grid]
        med = float(np.median(ea))
        ax.plot(grid, cdf, linewidth=2.2, label=f"{k} (med={med:.2f}px)")
    ax.set_title("Track B: Same YOLO11-Seg Mask Proposal (n=200 Scenes)", fontweight="bold")
    ax.set_xlabel("Corner Error (px)")
    ax.set_ylabel("Scenes <= Threshold (%)")
    ax.grid(alpha=0.28)
    ax.legend(fontsize=8.5, loc="lower right")

    # Panel 3: Sub-pixel hit rates (<=1px & <=2px)
    ax = axes[1, 0]
    labels = list(synth_summary["track_a_same_obb"].keys()) + list(synth_summary["track_b_same_seg_mask"].keys())
    all_stats = {**synth_summary["track_a_same_obb"], **synth_summary["track_b_same_seg_mask"]}
    lt1 = [all_stats[l]["pct_lt_1px"] for l in labels]
    lt2 = [all_stats[l]["pct_lt_2px"] for l in labels]
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, lt1, width=w, color="#10B981", label="<= 1.0 px")
    b2 = ax.bar(x + w / 2, lt2, width=w, color="#0284C7", label="<= 2.0 px")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}%", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=8)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Scenes (%)")
    ax.set_title("Sub-Pixel Precision from Same Proposal", fontweight="bold")
    ax.grid(axis="y", alpha=0.28)
    ax.legend()

    # Panel 4: Real Smartphone Structural NCC Progression
    ax = axes[1, 1]
    seg_methods = ["Mask -> minAreaRect", "Mask -> Legacy Binary Fit", "Mask -> Upgraded Fit (No Snap)", "Mask -> Upgraded + 2-Pass Snap", "Mask -> Full + Stage-6 ECC"]
    obb_methods = ["Raw OBB", "OBB + Legacy PBL", "OBB + 2-Pass Edge Snap + AA", "OBB + Edge Snap + ECC"]
    seg_means = [float(np.mean([r["ncc"] for r in real_ncc_rows if r["track"] == "B" and r["method"] == m])) for m in seg_methods]
    obb_means = [float(np.mean([r["ncc"] for r in real_ncc_rows if r["track"] == "A" and r["method"] == m])) for m in obb_methods]
    ax.plot(range(len(seg_methods)), seg_means, "o-", linewidth=2.4, color="#2563EB", label="Track B (Same Seg Mask, 5 imgs)")
    ax.plot(range(len(obb_methods)), obb_means, "s--", linewidth=2.4, color="#F59E0B", label="Track A (Same Raw OBB, 4 imgs)")
    for idx, val in enumerate(seg_means):
        ax.annotate(f"{val:.3f}", (idx, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#1D4ED8")
    for idx, val in enumerate(obb_means):
        ax.annotate(f"{val:.3f}", (idx, val), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#B45309")
    ax.set_xticks(range(len(seg_methods)))
    ax.set_xticklabels(["Step 1\n(Raw/Rect)", "Step 2\n(Legacy)", "Step 3\n(Upgraded)", "Step 4\n(+2-Pass Snap)", "Step 5\n(+Stage-6 ECC)"], fontsize=8)
    ax.set_ylim(0.15, 0.92)
    ax.set_ylabel("Mean Structural NCC vs Reference")
    ax.set_title("Real Smartphone Alignment Progression", fontweight="bold")
    ax.grid(alpha=0.28)
    ax.legend(fontsize=8.5, loc="lower right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Same-bbox/mask perspective alignment benchmark")
    parser.add_argument("--max-synth-scenes", type=int, default=200)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    set_seed(int(cfg["project"]["seed"]))

    exp = Experiment("same_bbox_perspective_benchmark", cfg)
    plots_dir = exp.plots_dir
    report_dir = PROJECT_ROOT / "report" / "figures" / "05_segmentation_pipeline" / "same_bbox_comparison"
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

    # ------------------------------------------------------------------
    # 1. Paired synthetic benchmark on pokemon_seg/test (same proposal!)
    # ------------------------------------------------------------------
    ds_root = PROJECT_ROOT / "data" / "datasets" / "pokemon_seg"
    img_dir = ds_root / "images" / "test"
    lbl_obb_dir = ds_root / "labels_obb" / "test"
    lbl_seg_dir = ds_root / "labels" / "test"
    all_imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    pos_imgs = [p for p in all_imgs if (lbl_obb_dir / f"{p.stem}.txt").exists() and (lbl_obb_dir / f"{p.stem}.txt").read_text().strip()]
    chosen = pos_imgs[: args.max_synth_scenes]

    synth_obb_errs: dict[str, list[float]] = {"Raw OBB": [], "OBB + Legacy PBL": [], "OBB + 2-Pass Edge Snap + AA": []}
    synth_obb_ious: dict[str, list[float]] = {k: [] for k in synth_obb_errs}
    synth_obb_clips: dict[str, list[float]] = {k: [] for k in synth_obb_errs}
    synth_seg_errs: dict[str, list[float]] = {
        "Mask -> minAreaRect": [],
        "Mask -> Legacy Binary Fit": [],
        "Mask -> Upgraded Fit (No Snap)": [],
        "Mask -> Upgraded + 2-Pass Snap": [],
    }
    synth_seg_ious: dict[str, list[float]] = {k: [] for k in synth_seg_errs}
    synth_seg_clips: dict[str, list[float]] = {k: [] for k in synth_seg_errs}

    logger.info("Running same-proposal synthetic benchmark on %d scenes...", len(chosen))
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

        # Track A: single raw OBB proposal shared across A1..A3
        det_obb.use_pbl = False
        raw_dets = det_obb.detect(img_bgr)
        if raw_dets:
            raw_q = raw_dets[0].corners
            det_obb.use_pbl = True
            variants_a = _run_obb_variants_from_same_quad(det_obb, img_bgr, raw_q, ref_map=None)
            for k in synth_obb_errs:
                q_pred, _ = variants_a[k]
                synth_obb_errs[k].append(align_corner_error(q_pred, gt_q))
                proj_m = project_patch_mask_to_scene(q_pred, np.ones((TARGET_H, TARGET_W), dtype=np.float32), (h, w))
                inter = float(np.logical_and(proj_m > 0.5, gt_m > 0.5).sum())
                union = float(np.logical_or(proj_m > 0.5, gt_m > 0.5).sum())
                synth_obb_ious[k].append(inter / union if union > 0 else 0.0)
                clipped = float(np.logical_and(gt_ring, ~(proj_m > 0.5)).sum())
                synth_obb_clips[k].append(clipped / ring_area * 100.0)

        # Track B: single raw YOLO11-Seg probability map shared across B1..B4
        prob_mask = _extract_raw_yolo_seg_prob(det_seg, img_bgr)
        if prob_mask is not None:
            variants_b = _run_seg_variants_from_same_mask(img_bgr, prob_mask, ref_map=None)
            if set(synth_seg_errs).issubset(variants_b.keys()):
                for k in synth_seg_errs:
                    q_pred, _ = variants_b[k]
                    synth_seg_errs[k].append(align_corner_error(q_pred, gt_q))
                    proj_m = project_patch_mask_to_scene(q_pred, np.ones((TARGET_H, TARGET_W), dtype=np.float32), (h, w))
                    inter = float(np.logical_and(proj_m > 0.5, gt_m > 0.5).sum())
                    union = float(np.logical_or(proj_m > 0.5, gt_m > 0.5).sum())
                    synth_seg_ious[k].append(inter / union if union > 0 else 0.0)
                    clipped = float(np.logical_and(gt_ring, ~(proj_m > 0.5)).sum())
                    synth_seg_clips[k].append(clipped / ring_area * 100.0)

    # ------------------------------------------------------------------
    # 2. Real smartphone captures (same proposal + visual boards + NCC)
    # ------------------------------------------------------------------
    real_dir = PROJECT_ROOT / "experiments" / "synthetic-to-real"
    real_ncc_rows: list[dict] = []
    for ref_name, q_name in REAL_CASES:
        ref_p, qp = real_dir / ref_name, real_dir / q_name
        if not ref_p.exists() or not qp.exists():
            continue
        img_bgr = cv2.imread(str(qp), cv2.IMREAD_COLOR)
        ref_rgb = cv2.cvtColor(
            cv2.resize(cv2.imread(str(ref_p), cv2.IMREAD_COLOR), (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        ref_map = build_alignment_feature_map(ref_rgb, align_size_wh=(224, 315))

        # Track A on real capture
        det_obb.use_pbl = False
        raw_dets = det_obb.detect(img_bgr)
        if raw_dets:
            raw_q = order_quad(raw_dets[0].corners)
            det_obb.use_pbl = True
            var_a = _run_obb_variants_from_same_quad(det_obb, img_bgr, raw_q, ref_map=ref_map)
            for m_name, (_, patch) in var_a.items():
                real_ncc_rows.append({"image": q_name, "track": "A", "method": m_name, "ncc": round(_structural_ncc(patch, ref_map), 4)})
            if q_name in ("charizard-damaged.jpeg", "realgengar-1.jpg", "realgengar-3.jpg"):
                _render_visual_board(
                    f"Track A (Same Raw YOLO11-OBB Proposal) — {q_name}",
                    img_bgr,
                    ref_rgb,
                    ref_map,
                    raw_q,
                    None,
                    var_a,
                    plots_dir / f"same_obb_{Path(q_name).stem}.png",
                )
                _render_visual_board(
                    f"Track A (Same Raw YOLO11-OBB Proposal) — {q_name}",
                    img_bgr,
                    ref_rgb,
                    ref_map,
                    raw_q,
                    None,
                    var_a,
                    report_dir / f"same_obb_{Path(q_name).stem}.png",
                )

        # Track B on real capture
        prob_mask = _extract_raw_yolo_seg_prob(det_seg, img_bgr)
        if prob_mask is not None:
            var_b = _run_seg_variants_from_same_mask(img_bgr, prob_mask, ref_map=ref_map)
            init_rect = var_b["Mask -> minAreaRect"][0]
            for m_name, (_, patch) in var_b.items():
                real_ncc_rows.append({"image": q_name, "track": "B", "method": m_name, "ncc": round(_structural_ncc(patch, ref_map), 4)})
            if q_name in ("charizard-damaged.jpeg", "giratina-kanda-damaged.jpg", "realgengar-1.jpg"):
                _render_visual_board(
                    f"Track B (Same YOLO11-Seg @ 640 Probability Mask) — {q_name}",
                    img_bgr,
                    ref_rgb,
                    ref_map,
                    init_rect,
                    prob_mask,
                    var_b,
                    plots_dir / f"same_mask_{Path(q_name).stem}.png",
                )
                _render_visual_board(
                    f"Track B (Same YOLO11-Seg @ 640 Probability Mask) — {q_name}",
                    img_bgr,
                    ref_rgb,
                    ref_map,
                    init_rect,
                    prob_mask,
                    var_b,
                    report_dir / f"same_mask_{Path(q_name).stem}.png",
                )

    synth_summary = {
        "track_a_same_obb": {k: _summarize_errs(synth_obb_errs[k], synth_obb_ious[k], synth_obb_clips[k]) for k in synth_obb_errs},
        "track_b_same_seg_mask": {k: _summarize_errs(synth_seg_errs[k], synth_seg_ious[k], synth_seg_clips[k]) for k in synth_seg_errs},
    }
    wilcoxon_tests = {
        "track_a_edge_snap_vs_raw_obb_p": float(
            wilcoxon(synth_obb_errs["Raw OBB"], synth_obb_errs["OBB + 2-Pass Edge Snap + AA"]).pvalue
        ),
        "track_a_edge_snap_vs_legacy_pbl_p": float(
            wilcoxon(synth_obb_errs["OBB + Legacy PBL"], synth_obb_errs["OBB + 2-Pass Edge Snap + AA"]).pvalue
        ),
        "track_b_upgraded_vs_legacy_fit_p": float(
            wilcoxon(synth_seg_errs["Mask -> Legacy Binary Fit"], synth_seg_errs["Mask -> Upgraded + 2-Pass Snap"]).pvalue
        ),
    }

    _plot_summary(
        synth_obb_errs,
        synth_seg_errs,
        synth_summary,
        real_ncc_rows,
        plots_dir / "same_bbox_summary_benchmark.png",
    )
    _plot_summary(
        synth_obb_errs,
        synth_seg_errs,
        synth_summary,
        real_ncc_rows,
        report_dir / "same_bbox_summary_benchmark.png",
    )

    payload = {
        "synthetic_summary": synth_summary,
        "wilcoxon_tests": wilcoxon_tests,
        "real_ncc_progression": real_ncc_rows,
    }
    (exp.artifacts_dir / "same_bbox_benchmark.json").write_text(json.dumps(payload, indent=2))
    exp.finish()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

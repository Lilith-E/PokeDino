"""
Benchmark script to compute Raw YOLO Mean Corner Error vs Refined (Option A) Mean Corner Error.
Measures:
1. Permutation-invariant cyclic Euclidean corner error (in pixels)
2. Polygon IoU against ground-truth
3. Error percentiles (P25, P50, P75, P90, P95, Mean, Std)
4. Ratio of improved vs degraded detections
5. Error distribution visualization (Histogram + CDF)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cv2
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from backend.card_detector import CardDetector

def align_and_compute_corner_error(pred_pts: np.ndarray, gt_pts: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Computes the permutation-invariant cyclic Euclidean corner error between two 4-point polygons.
    
    E(P, G) = min_{sigma in {+1, -1}, k in {0,1,2,3}} (1/4) * sum_{i=0}^3 || P_{perm(sigma, k, i)} - G_i ||_2
    
    Returns:
        mean_error: mean euclidean distance in pixels across the 4 aligned corners
        aligned_pred: the predicted corners arranged in matching order with GT
    """
    min_mean_dist = 1e9
    best_aligned = pred_pts
    
    for flip in [pred_pts, pred_pts[::-1]]:
        for shift in range(4):
            candidate = np.roll(flip, shift, axis=0)
            dists = np.linalg.norm(candidate - gt_pts, axis=1)
            mean_d = float(np.mean(dists))
            if mean_d < min_mean_dist:
                min_mean_dist = mean_d
                best_aligned = candidate
                
    return min_mean_dist, best_aligned

def compute_polygon_iou(p1: np.ndarray, p2: np.ndarray) -> float:
    """Computes exact intersection-over-union between two convex 4-vertex polygons using OpenCV."""
    p1_f = p1.astype(np.float32)
    p2_f = p2.astype(np.float32)
    ret, inter_pts = cv2.intersectConvexConvex(p1_f, p2_f)
    if ret <= 0 or inter_pts is None or len(inter_pts) < 3:
        return 0.0
    area_inter = ret
    area1 = cv2.contourArea(p1_f)
    area2 = cv2.contourArea(p2_f)
    denom = area1 + area2 - area_inter
    return float(area_inter / denom) if denom > 0 else 0.0

def run_benchmark(max_samples: int = 500):
    val_img_dir = PROJECT_ROOT / "data/datasets/pokemon_obb/images/val"
    val_lbl_dir = PROJECT_ROOT / "data/datasets/pokemon_obb/labels/val"
    
    detector = CardDetector()
    
    all_img_paths = sorted(val_img_dir.glob("*.jpg"))
    if max_samples and max_samples < len(all_img_paths):
        step = max(1, len(all_img_paths) // max_samples)
        img_paths = all_img_paths[::step][:max_samples]
    else:
        img_paths = all_img_paths
        
    print(f"Running Corner Error Benchmark on {len(img_paths)} validation scenes...")
    
    raw_corner_errors = []
    ref_corner_errors = []
    raw_ious = []
    ref_ious = []
    
    per_card_diffs = []
    
    for img_p in tqdm(img_paths, desc="Evaluating"):
        lbl_p = val_lbl_dir / (img_p.stem + ".txt")
        if not lbl_p.exists():
            continue
            
        lines = lbl_p.read_text().strip().splitlines()
        if not lines or len(lines[0].split()) < 9:
            continue
            
        if len(lines) > 1:
            continue
            
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        
        gt_parts = [float(x) for x in lines[0].split()[1:]]
        gt_corners = np.array(gt_parts).reshape(-1, 2) * [w, h]
        
        preds = detector.model.predict(
            img_bgr, conf=detector.conf, iou=detector.iou,
            imgsz=detector.imgsz, verbose=False
        )
        if preds[0].obb is None or len(preds[0].obb) == 0:
            continue
        raw_box = preds[0].obb.xyxyxyxy.cpu().numpy()[0]
        
        dets = detector.detect(img_bgr)
        if not dets:
            continue
        ref_box = dets[0].corners
        
        err_raw, _ = align_and_compute_corner_error(raw_box, gt_corners)
        err_ref, _ = align_and_compute_corner_error(ref_box, gt_corners)
        
        iou_raw = compute_polygon_iou(raw_box, gt_corners)
        iou_ref = compute_polygon_iou(ref_box, gt_corners)
        
        raw_corner_errors.append(err_raw)
        ref_corner_errors.append(err_ref)
        raw_ious.append(iou_raw)
        ref_ious.append(iou_ref)
        per_card_diffs.append(err_raw - err_ref)
        
    N = len(raw_corner_errors)
    print(f"\n================ BENCHMARK RESULTS (N = {N} cards) ================")
    
    raw_arr = np.array(raw_corner_errors)
    ref_arr = np.array(ref_corner_errors)
    diff_arr = np.array(per_card_diffs)
    
    improved_count = int(np.sum(diff_arr > 0.05))
    neutral_count = int(np.sum(np.abs(diff_arr) <= 0.05))
    degraded_count = int(np.sum(diff_arr < -0.05))
    
    metrics = {
        "sample_size": N,
        "raw_mean_corner_error_px": float(np.mean(raw_arr)),
        "raw_median_corner_error_px": float(np.median(raw_arr)),
        "raw_std_corner_error_px": float(np.std(raw_arr)),
        "raw_p25_px": float(np.percentile(raw_arr, 25)),
        "raw_p75_px": float(np.percentile(raw_arr, 75)),
        "raw_p90_px": float(np.percentile(raw_arr, 90)),
        "raw_p95_px": float(np.percentile(raw_arr, 95)),
        "raw_mean_iou": float(np.mean(raw_ious)),
        
        "ref_mean_corner_error_px": float(np.mean(ref_arr)),
        "ref_median_corner_error_px": float(np.median(ref_arr)),
        "ref_std_corner_error_px": float(np.std(ref_arr)),
        "ref_p25_px": float(np.percentile(ref_arr, 25)),
        "ref_p75_px": float(np.percentile(ref_arr, 75)),
        "ref_p90_px": float(np.percentile(ref_arr, 90)),
        "ref_p95_px": float(np.percentile(ref_arr, 95)),
        "ref_mean_iou": float(np.mean(ref_ious)),
        
        "mean_improvement_px": float(np.mean(raw_arr) - np.mean(ref_arr)),
        "percent_error_reduction": float((np.mean(raw_arr) - np.mean(ref_arr)) / np.mean(raw_arr) * 100.0),
        "median_improvement_px": float(np.median(raw_arr) - np.median(ref_arr)),
        "percent_median_reduction": float((np.median(raw_arr) - np.median(ref_arr)) / np.median(raw_arr) * 100.0),
        
        "improved_cards_pct": float(improved_count / N * 100.0),
        "neutral_cards_pct": float(neutral_count / N * 100.0),
        "degraded_cards_pct": float(degraded_count / N * 100.0),
    }
    
    print(f"Raw YOLO Mean Corner Error:     {metrics['raw_mean_corner_error_px']:.3f} px (std: {metrics['raw_std_corner_error_px']:.2f})")
    print(f"Refined (Opt A) Mean Corner Error: {metrics['ref_mean_corner_error_px']:.3f} px (std: {metrics['ref_std_corner_error_px']:.2f})")
    print(f"--> Net Mean Error Reduction:   {metrics['mean_improvement_px']:.3f} px ({metrics['percent_error_reduction']:.1f}%)")
    print(f"Raw YOLO Median Corner Error:   {metrics['raw_median_corner_error_px']:.3f} px")
    print(f"Refined (Opt A) Median Error:   {metrics['ref_median_corner_error_px']:.3f} px")
    print(f"--> Net Median Error Reduction: {metrics['median_improvement_px']:.3f} px ({metrics['percent_median_reduction']:.1f}%)")
    print(f"Mean IoU: Raw = {metrics['raw_mean_iou']:.4f} --> Refined = {metrics['ref_mean_iou']:.4f}")
    print(f"Improved Cards: {improved_count}/{N} ({metrics['improved_cards_pct']:.1f}%)")
    print(f"Neutral Cards:  {neutral_count}/{N} ({metrics['neutral_cards_pct']:.1f}%)")
    print(f"Degraded Cards: {degraded_count}/{N} ({metrics['degraded_cards_pct']:.1f}%)")
    
    out_json = PROJECT_ROOT / "report/figures/corner_error_benchmark.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.grid(True, linestyle="--", alpha=0.3, color="#64748b")
        ax.tick_params(colors="#cbd5e1", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#475569")
            
    bins = np.linspace(0, 20, 41)
    axes[0].hist(raw_arr, bins=bins, alpha=0.55, color="#06b6d4", label=f"Raw YOLO11s (Mean: {metrics['raw_mean_corner_error_px']:.2f}px)", density=True)
    axes[0].hist(ref_arr, bins=bins, alpha=0.65, color="#f59e0b", label=f"Refined Opt A (Mean: {metrics['ref_mean_corner_error_px']:.2f}px)", density=True)
    axes[0].axvline(metrics['raw_mean_corner_error_px'], color="#06b6d4", linestyle="--", linewidth=2.0)
    axes[0].axvline(metrics['ref_mean_corner_error_px'], color="#f59e0b", linestyle="--", linewidth=2.0)
    axes[0].set_title("Corner Error Distribution (Density)", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    axes[0].set_xlabel("Mean Corner Error Across 4 Vertices (Pixels)", color="#94a3b8", fontsize=11)
    axes[0].set_ylabel("Probability Density", color="#94a3b8", fontsize=11)
    axes[0].legend(facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=10)
    
    eval_pts = np.linspace(0, 18, 100)
    cdf_raw = [np.mean(raw_arr <= x) * 100.0 for x in eval_pts]
    cdf_ref = [np.mean(ref_arr <= x) * 100.0 for x in eval_pts]
    axes[1].plot(eval_pts, cdf_raw, color="#06b6d4", linewidth=2.5, label="Raw YOLO11s-OBB")
    axes[1].plot(eval_pts, cdf_ref, color="#f59e0b", linewidth=2.5, label="Refined Option A (Locked AR 88:63)")
    axes[1].axhline(90.0, color="#64748b", linestyle=":", alpha=0.7)
    axes[1].text(0.5, 91.5, "90% of samples", color="#94a3b8", fontsize=9)
    axes[1].set_title("Cumulative Error Distribution Function (CDF)", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_xlabel("Maximum Corner Error Threshold (Pixels)", color="#94a3b8", fontsize=11)
    axes[1].set_ylabel("% Cards with Error <= Threshold", color="#94a3b8", fontsize=11)
    axes[1].legend(facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=10, loc="lower right")
    
    fig.suptitle(
        f"Quantitative Corner Error Benchmark: Raw YOLO11s-OBB vs Refined Option A (N = {N} Cards)",
        color="#ffffff", fontsize=14, fontweight="bold", y=0.98
    )
    fig.tight_layout()
    fig_out = PROJECT_ROOT / "report/figures/corner_error_benchmark.png"
    fig.savefig(fig_out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved benchmark figure to {fig_out}")

if __name__ == "__main__":
    run_benchmark(max_samples=500)

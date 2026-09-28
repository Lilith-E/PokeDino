"""
3-Way Corner Error Benchmark:
1. Raw YOLO (Pre-finetuning - Epoch 10 Baseline)
2. YOLO Fine-tuned (Raw neural predictions from Phase 1c checkpoint)
3. YOLO Fine-tuned + OpenCV Boundary Refinement (Hybrid overkill model)

Evaluates on the validation set, measuring:
- Permutation-invariant cyclic corner error (mean, median, P25, P75, P90, P95)
- Exact polygon IoU
- Histograms + CDF curves
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
from ultralytics import YOLO

from backend.card_detector import CardDetector

def align_and_compute_corner_error(pred_pts: np.ndarray, gt_pts: np.ndarray) -> tuple[float, np.ndarray]:
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

def run_3way_benchmark(raw_weights: str, ft_weights: str, max_samples: int = 500):
    val_img_dir = PROJECT_ROOT / "data/datasets/pokemon_obb/images/val"
    val_lbl_dir = PROJECT_ROOT / "data/datasets/pokemon_obb/labels/val"

    print(f"Loading Base Raw YOLO from: {raw_weights}")
    raw_model = YOLO(raw_weights)
    
    print(f"Loading Fine-tuned YOLO from: {ft_weights}")
    ft_model = YOLO(ft_weights)
    
    hybrid_detector = CardDetector(weights_path=ft_weights)

    all_img_paths = sorted(val_img_dir.glob("*.jpg"))
    if max_samples and max_samples < len(all_img_paths):
        step = max(1, len(all_img_paths) // max_samples)
        img_paths = all_img_paths[::step][:max_samples]
    else:
        img_paths = all_img_paths

    print(f"Running 3-Way Benchmark on {len(img_paths)} validation scenes...")

    errs_raw_base = []
    errs_ft_raw = []
    errs_ft_hybrid = []
    ious_raw_base = []
    ious_ft_raw = []
    ious_ft_hybrid = []

    for img_p in tqdm(img_paths, desc="Evaluating 3-way"):
        lbl_p = val_lbl_dir / (img_p.stem + ".txt")
        if not lbl_p.exists():
            continue
        lines = lbl_p.read_text().strip().splitlines()
        if len(lines) != 1 or len(lines[0].split()) < 9:
            continue
            
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        
        gt_parts = [float(x) for x in lines[0].split()[1:]]
        gt_corners = np.array(gt_parts).reshape(-1, 2) * [w, h]

        res_raw = raw_model.predict(img_bgr, conf=0.18, iou=0.45, imgsz=640, verbose=False)
        if not res_raw or res_raw[0].obb is None or len(res_raw[0].obb) == 0:
            continue
        b_raw = res_raw[0].obb.xyxyxyxy.cpu().numpy()[0]

        res_ft = ft_model.predict(img_bgr, conf=0.18, iou=0.45, imgsz=640, verbose=False)
        if not res_ft or res_ft[0].obb is None or len(res_ft[0].obb) == 0:
            continue
        b_ft = res_ft[0].obb.xyxyxyxy.cpu().numpy()[0]

        dets = hybrid_detector.detect(img_bgr)
        if not dets:
            continue
        b_hybrid = dets[0].corners

        e1, _ = align_and_compute_corner_error(b_raw, gt_corners)
        e2, _ = align_and_compute_corner_error(b_ft, gt_corners)
        e3, _ = align_and_compute_corner_error(b_hybrid, gt_corners)

        iou1 = compute_polygon_iou(b_raw, gt_corners)
        iou2 = compute_polygon_iou(b_ft, gt_corners)
        iou3 = compute_polygon_iou(b_hybrid, gt_corners)

        errs_raw_base.append(e1)
        errs_ft_raw.append(e2)
        errs_ft_hybrid.append(e3)

        ious_raw_base.append(iou1)
        ious_ft_raw.append(iou2)
        ious_ft_hybrid.append(iou3)

    N = len(errs_raw_base)
    a_raw, a_ft, a_hy = np.array(errs_raw_base), np.array(errs_ft_raw), np.array(errs_ft_hybrid)

    stats = {
        "sample_size": N,
        "raw_base": {
            "mean": float(np.mean(a_raw)),
            "median": float(np.median(a_raw)),
            "std": float(np.std(a_raw)),
            "p25": float(np.percentile(a_raw, 25)),
            "p75": float(np.percentile(a_raw, 75)),
            "p90": float(np.percentile(a_raw, 90)),
            "mean_iou": float(np.mean(ious_raw_base)),
        },
        "finetuned_raw": {
            "mean": float(np.mean(a_ft)),
            "median": float(np.median(a_ft)),
            "std": float(np.std(a_ft)),
            "p25": float(np.percentile(a_ft, 25)),
            "p75": float(np.percentile(a_ft, 75)),
            "p90": float(np.percentile(a_ft, 90)),
            "mean_iou": float(np.mean(ious_ft_raw)),
            "improvement_vs_base_mean": float(np.mean(a_raw) - np.mean(a_ft)),
            "pct_improvement_mean": float((np.mean(a_raw) - np.mean(a_ft)) / np.mean(a_raw) * 100),
            "pct_improvement_median": float((np.median(a_raw) - np.median(a_ft)) / np.median(a_raw) * 100),
        },
        "finetuned_hybrid": {
            "mean": float(np.mean(a_hy)),
            "median": float(np.median(a_hy)),
            "std": float(np.std(a_hy)),
            "p25": float(np.percentile(a_hy, 25)),
            "p75": float(np.percentile(a_hy, 75)),
            "p90": float(np.percentile(a_hy, 90)),
            "mean_iou": float(np.mean(ious_ft_hybrid)),
            "improvement_vs_base_mean": float(np.mean(a_raw) - np.mean(a_hy)),
            "pct_improvement_mean": float((np.mean(a_raw) - np.mean(a_hy)) / np.mean(a_raw) * 100),
            "pct_improvement_median": float((np.median(a_raw) - np.median(a_hy)) / np.median(a_raw) * 100),
        }
    }

    out_json = PROJECT_ROOT / "report/figures/corner_error_3way_benchmark.json"
    out_json.write_text(json.dumps(stats, indent=2))
    dir_03 = PROJECT_ROOT / "report/figures/03_comparative_benchmark"
    dir_03.mkdir(parents=True, exist_ok=True)
    (dir_03 / "corner_error_3way_benchmark.json").write_text(json.dumps(stats, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.grid(True, linestyle="--", alpha=0.3, color="#64748b")
        ax.tick_params(colors="#cbd5e1", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#475569")

    bins = np.linspace(0, 18, 37)
    axes[0].hist(a_raw, bins=bins, alpha=0.45, color="#06b6d4", label=f"1. Raw YOLO Base (Mean: {stats['raw_base']['mean']:.2f}px)", density=True)
    axes[0].hist(a_ft, bins=bins, alpha=0.55, color="#8b5cf6", label=f"2. YOLO Fine-Tuned (Mean: {stats['finetuned_raw']['mean']:.2f}px)", density=True)
    axes[0].hist(a_hy, bins=bins, alpha=0.65, color="#10b981", label=f"3. FT + PBL (Mean: {stats['finetuned_hybrid']['mean']:.2f}px)", density=True)
    axes[0].axvline(stats['raw_base']['mean'], color="#06b6d4", linestyle="--", linewidth=1.8)
    axes[0].axvline(stats['finetuned_raw']['mean'], color="#8b5cf6", linestyle="--", linewidth=1.8)
    axes[0].axvline(stats['finetuned_hybrid']['mean'], color="#10b981", linestyle="--", linewidth=1.8)
    axes[0].set_title("Corner Error Distribution (3 Approaches)", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    axes[0].set_xlabel("Mean Corner Error Across 4 Vertices (Pixels)", color="#94a3b8", fontsize=11)
    axes[0].set_ylabel("Probability Density", color="#94a3b8", fontsize=11)
    axes[0].legend(facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=10)

    eval_pts = np.linspace(0, 16, 100)
    cdf_raw = [np.mean(a_raw <= x) * 100.0 for x in eval_pts]
    cdf_ft = [np.mean(a_ft <= x) * 100.0 for x in eval_pts]
    cdf_hy = [np.mean(a_hy <= x) * 100.0 for x in eval_pts]

    axes[1].plot(eval_pts, cdf_raw, color="#06b6d4", linewidth=2.2, label="1. Raw YOLO (Epoch 10 Base)")
    axes[1].plot(eval_pts, cdf_ft, color="#8b5cf6", linewidth=2.5, label="2. YOLO Fine-Tuned (Edge-Loss)")
    axes[1].plot(eval_pts, cdf_hy, color="#10b981", linewidth=2.8, label="3. FT + PBL")
    axes[1].axhline(90.0, color="#64748b", linestyle=":", alpha=0.7)
    axes[1].text(0.5, 91.5, "90% of samples", color="#94a3b8", fontsize=9)
    axes[1].set_title("Cumulative Error Distribution Function (CDF)", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_xlabel("Maximum Corner Error Threshold (Pixels)", color="#94a3b8", fontsize=11)
    axes[1].set_ylabel("% Cards with Error <= Threshold", color="#94a3b8", fontsize=11)
    axes[1].legend(facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=10, loc="lower right")

    fig.suptitle(f"3-Way Corner Alignment Benchmark: Raw vs Fine-Tuned vs PBL (N = {N} Cards)",
                 color="#ffffff", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()

    fig_out = PROJECT_ROOT / "report/figures/corner_error_3way_benchmark.png"
    fig.savefig(fig_out, dpi=180, bbox_inches="tight")
    dir_03 = PROJECT_ROOT / "report/figures/03_comparative_benchmark"
    dir_03.mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_03 / "corner_error_3way_benchmark.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"3-Way benchmark saved to {fig_out} and {dir_03 / 'corner_error_3way_benchmark.png'}")
    return stats

if __name__ == "__main__":
    from benchmark_pbl_pipeline import main as run_current_pipeline_benchmark

    run_current_pipeline_benchmark()

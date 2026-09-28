"""Paired validation benchmark for the rotation-robust detector and PBL."""

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from tqdm import tqdm  # noqa: E402
from ultralytics import YOLO  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.card_detector import CardDetector, TARGET_AR  # noqa: E402

VAL_IMG = PROJECT_ROOT / "data/datasets/pokemon_obb/images/val"
VAL_LBL = PROJECT_ROOT / "data/datasets/pokemon_obb/labels/val"
FT_WEIGHTS = PROJECT_ROOT / "data/weights/yolo11s_obb_finetuned_best.pt"
RAW_WEIGHTS = PROJECT_ROOT / "experiments/20260915_142536_phase1b_train_obb/artifacts/yolo11s_obb_best.pt"
FIG_DIR = PROJECT_ROOT / "report/figures/03_comparative_benchmark"
THEME = {
    "raw": "#06b6d4",
    "ft": "#8b5cf6",
    "tta": "#f59e0b",
    "pbl": "#10b981",
    "text": "#0f172a",
    "grid": "#cbd5e1",
}

def align_quad(points: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
    best, best_error = points, float("inf")
    for order in (points, points[::-1]):
        for shift in range(4):
            candidate = np.roll(order, shift, axis=0)
            error = float(np.linalg.norm(candidate - reference, axis=1).mean())
            if error < best_error:
                best, best_error = candidate, error
    return best, best_error

def unrotate_quad(points: np.ndarray, k: int, width: int, height: int) -> np.ndarray:
    if k == 0:
        return points.copy()
    if k == 1:
        return np.stack([width - 1 - points[:, 1], points[:, 0]], axis=1)
    if k == 2:
        return np.stack([width - 1 - points[:, 0], height - 1 - points[:, 1]], axis=1)
    return np.stack([points[:, 1], height - 1 - points[:, 0]], axis=1)

def corner_error(pred: np.ndarray, gt: np.ndarray) -> float:
    return align_quad(pred, gt)[1]

def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    a, b = first.astype(np.float32), second.astype(np.float32)
    intersection, polygon = cv2.intersectConvexConvex(a, b)
    if polygon is None or len(polygon) < 3:
        return 0.0
    union = cv2.contourArea(a) + cv2.contourArea(b) - intersection
    return float(intersection / union) if union > 0 else 0.0

def _predict_candidates(model, image: np.ndarray, rotation: int, width: int, height: int,
                        conf: float, iou: float, max_det: int, imgsz: int) -> list[dict]:
    view = np.ascontiguousarray(np.rot90(image, rotation)) if rotation else image
    result = model.predict(view, conf=conf, iou=iou, max_det=max_det, imgsz=imgsz, verbose=False)[0]
    if result.obb is None or len(result.obb) == 0:
        return []
    output = []
    boxes = result.obb.xyxyxyxy.cpu().numpy()
    confs = result.obb.conf.cpu().numpy()
    for box, confidence in zip(boxes, confs):
        corners = unrotate_quad(box.astype(np.float64), rotation, width, height)
        rect = cv2.minAreaRect(corners.astype(np.float32))
        rw, rh = rect[1]
        short, long = min(rw, rh), max(rw, rh)
        ar = long / max(short, 1e-3)
        area_fraction = short * long / max(width * height, 1)
        if 1.15 <= ar <= 1.65 and area_fraction >= 0.06:
            score = float(confidence) / (1.0 + 3.0 * abs(ar - TARGET_AR))
            output.append({"corners": corners, "confidence": float(confidence), "score": score})
    return output

def fuse_single_card_views(candidates: list[dict]) -> np.ndarray | None:
    if not candidates:
        return None
    candidates.sort(key=lambda item: item["score"], reverse=True)
    reference = candidates[0]["corners"]
    diag = float(np.linalg.norm(reference[0] - reference[2]))
    aligned = []
    for item in candidates:
        quad, distance = align_quad(item["corners"], reference)
        if distance <= max(12.0, 0.05 * diag):
            aligned.append((item, quad))
    weights = np.array([max(item["score"], 1e-4) ** 2 for item, _ in aligned])
    weights /= weights.sum()
    return np.sum(np.stack([quad for _, quad in aligned]) * weights[:, None, None], axis=0)

def describe(errors: np.ndarray, ious: np.ndarray) -> dict:
    return {
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "std": float(errors.std()),
        "p25": float(np.percentile(errors, 25)),
        "p75": float(np.percentile(errors, 75)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(errors.max()),
        "mean_iou": float(ious.mean()),
    }

def _setup_axis(ax):
    ax.set_facecolor("#f8fafc")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, color=THEME["grid"])
    ax.tick_params(colors="#475569", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(THEME["grid"])

def write_figures(arrays: dict[str, tuple[np.ndarray, np.ndarray]], stats: dict, sample_count: int):
    names = ["raw_base", "ft_single", "ft_tta", "pbl_final"]
    labels = ["YOLO base", "YOLO fine-tuned", "4-view TTA", "TTA + PBL"]
    colors = [THEME["raw"], THEME["ft"], THEME["tta"], THEME["pbl"]]
    errors = [arrays[name][0] for name in names]
    ious = [arrays[name][1] for name in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax in axes:
        _setup_axis(ax)
    bins = np.linspace(0, max(12, float(np.percentile(np.concatenate(errors), 99))), 31)
    for e, color, label in zip(errors, colors, labels):
        axlabel = f"{label} (mean {e.mean():.2f}px)"
        axes[0].hist(e, bins=bins, density=True, alpha=.42, color=color, label=axlabel)
    axes[0].set_title("Corner-error distribution", color=THEME["text"], fontweight="bold")
    axes[0].set_xlabel("Mean aligned corner error (px)")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8, frameon=False)

    xs = np.linspace(0, max(12, float(np.percentile(np.concatenate(errors), 99))), 200)
    for e, color, label in zip(errors, colors, labels):
        cdf = np.array([np.mean(e <= x) * 100 for x in xs])
        axes[1].plot(xs, cdf, color=color, lw=2, label=f"{label} (P90 {np.percentile(e,90):.2f}px)")
    axes[1].set_title("Empirical CDF", color=THEME["text"], fontweight="bold")
    axes[1].set_xlabel("Corner-error threshold (px)")
    axes[1].set_ylabel("Scenes at or below threshold (%)")
    axes[1].set_ylim(0, 102)
    axes[1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(f"Card localization on held-out validation scenes (N={sample_count})",
                 color=THEME["text"], fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "corner_error_3way_benchmark.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax in axes:
        _setup_axis(ax)
    bp = axes[0].boxplot(ious, patch_artist=True, showmeans=True, tick_labels=labels,
                         medianprops={"color": THEME["text"], "linewidth": 1.8})
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(.65)
    vp = axes[1].violinplot(ious, showmeans=True, showmedians=True)
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(.6)
    axes[0].set_xticklabels(labels, rotation=12, ha="right")
    axes[1].set_xticks(np.arange(1, len(labels) + 1), labels, rotation=12, ha="right")
    axes[0].set_title("Polygon IoU (boxplot)", color=THEME["text"], fontweight="bold")
    axes[1].set_title("Polygon IoU (violin)", color=THEME["text"], fontweight="bold")
    for ax in axes:
        ax.set_ylabel("Exact polygon IoU")
        ax.set_ylim(.85, 1.005)
    fig.suptitle("Geometric overlap against exact OBB labels", color=THEME["text"], fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "iou_distribution_3way.png", dpi=180)
    plt.close(fig)

    keysets = [stats[name] for name in names]
    percentiles = [("Mean", "mean"), ("Median", "median"), ("P90", "p90"), ("P95", "p95")]
    fig, ax = plt.subplots(figsize=(11.5, 5.3))
    _setup_axis(ax)
    x = np.arange(len(percentiles)); width = .19
    for i, (label, color, result) in enumerate(zip(labels, colors, keysets)):
        heights = [result[key] for _, key in percentiles]
        bars = ax.bar(x + (i - 1.5) * width, heights, width, label=label, color=color, alpha=.88)
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(x, [label for label, _ in percentiles])
    ax.set_ylabel("Mean aligned corner error (px; lower is better)")
    ax.set_title("Corner-error percentiles by detector stage", color=THEME["text"], fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_percentiles_bar_chart.png", dpi=180)
    plt.close(fig)

    before, after = arrays["ft_tta"][0], arrays["pbl_final"][0]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax in axes:
        _setup_axis(ax)
    violin = axes[0].violinplot([before, after], showmeans=True, showmedians=True)
    for body, color in zip(violin["bodies"], [THEME["tta"], THEME["pbl"]]):
        body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.6)
    axes[0].set_xticks([1, 2], ["TTA only", "TTA + PBL"])
    axes[0].set_ylabel("Mean aligned corner error (px)")
    axes[0].set_title("Paired PBL contribution", color=THEME["text"], fontweight="bold")
    delta = before - after
    axes[1].hist(delta, bins=30, color=THEME["pbl"], alpha=.75)
    axes[1].axvline(0, color=THEME["text"], lw=1.4)
    axes[1].axvline(float(np.mean(delta)), color="#dc2626", ls="--", lw=1.7,
                    label=f"mean reduction {np.mean(delta):.2f}px")
    axes[1].set_xlabel("TTA error − TTA+PBL error (px; positive = improved)")
    axes[1].set_ylabel("Scenes")
    axes[1].set_title("Per-scene paired effect", color=THEME["text"], fontweight="bold")
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pbl_paired_violin.png", dpi=180)
    plt.close(fig)

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw_model = YOLO(str(RAW_WEIGHTS))
    detector = CardDetector(weights_path=str(FT_WEIGHTS), refine=False)
    all_images = sorted(VAL_IMG.glob("*.jpg"))
    step = max(1, len(all_images) // 500)
    paths = all_images[::step][:500]
    names = ("raw_base", "ft_single", "ft_tta", "pbl_final")
    err_lists = {name: [] for name in names}
    iou_lists = {name: [] for name in names}
    per_scene = []
    scene_count = 0

    for path in tqdm(paths, desc="Final TTA+PBL benchmark"):
        label_path = VAL_LBL / f"{path.stem}.txt"
        if not label_path.exists():
            continue
        lines = label_path.read_text().strip().splitlines()
        if len(lines) != 1 or len(lines[0].split()) < 9:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        gt = np.array([float(x) for x in lines[0].split()[1:]], dtype=np.float64).reshape(4, 2) * [w, h]

        raw_result = raw_model.predict(image, conf=detector.conf, iou=detector.iou,
                                       max_det=detector.max_det, imgsz=detector.imgsz,
                                       verbose=False)[0]
        if raw_result.obb is None or len(raw_result.obb) == 0:
            continue
        q_raw = raw_result.obb.xyxyxyxy.cpu().numpy()[0].astype(np.float64)

        ft_result = detector.model.predict(image, conf=detector.conf, iou=detector.iou,
                                            max_det=detector.max_det, imgsz=detector.imgsz,
                                            verbose=False)[0]
        if ft_result.obb is None or len(ft_result.obb) == 0:
            continue
        q_single = ft_result.obb.xyxyxyxy.cpu().numpy()[0].astype(np.float64)

        lock = detector._lock_physical_card
        detector._lock_physical_card = lambda _image, corners: corners
        try:
            no_lock = detector.detect(image)
        finally:
            detector._lock_physical_card = lock
        with_lock = detector.detect(image)
        if not no_lock or not with_lock:
            continue
        q_tta = no_lock[0].corners
        q_final = with_lock[0].corners

        qs = {"raw_base": q_raw, "ft_single": q_single, "ft_tta": q_tta, "pbl_final": q_final}
        for name, q in qs.items():
            err_lists[name].append(corner_error(q, gt))
            iou_lists[name].append(polygon_iou(q, gt))
        per_scene.append({
            "scene": path.stem,
            "raw_base_error": corner_error(q_raw, gt),
            "ft_single_error": corner_error(q_single, gt),
            "ft_tta_error": corner_error(q_tta, gt),
            "pbl_final_error": corner_error(q_final, gt),
            "pbl_shift_from_tta_px": detector._quad_distance(q_final, q_tta),
            "ft_tta_iou": polygon_iou(q_tta, gt),
            "pbl_final_iou": polygon_iou(q_final, gt),
        })
        scene_count += 1

    arrays = {name: (np.array(err_lists[name]), np.array(iou_lists[name])) for name in names}
    stats = {name: describe(*arrays[name]) for name in names}
    before, after = arrays["ft_tta"][0], arrays["pbl_final"][0]
    delta = before - after
    pbl_test = wilcoxon(before, after)
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(10000)])
    improved = int((delta > .05).sum())
    degraded = int((delta < -.05).sum())
    raw_compat = dict(stats["raw_base"])
    ft_compat = dict(stats["ft_single"])
    pbl_compat = dict(stats["pbl_final"])
    ft_compat["improvement_vs_base_mean"] = raw_compat["mean"] - ft_compat["mean"]
    ft_compat["pct_improvement_mean"] = 100.0 * (raw_compat["mean"] - ft_compat["mean"]) / raw_compat["mean"]
    ft_compat["pct_improvement_median"] = 100.0 * (raw_compat["median"] - ft_compat["median"]) / raw_compat["median"]
    pbl_compat["improvement_vs_base_mean"] = raw_compat["mean"] - pbl_compat["mean"]
    pbl_compat["pct_improvement_mean"] = 100.0 * (raw_compat["mean"] - pbl_compat["mean"]) / raw_compat["mean"]
    pbl_compat["pct_improvement_median"] = 100.0 * (raw_compat["median"] - pbl_compat["median"]) / raw_compat["median"]
    summary = {
        "n": int(scene_count),
        "sample_size": int(scene_count),
        "stages": stats,
        "raw_base": raw_compat,
        "finetuned_raw": ft_compat,
        "finetuned_hybrid": pbl_compat,
        "paired_tta_vs_pbl": {
            "improved": improved, "degraded": degraded,
            "neutral": int(len(delta) - improved - degraded),
            "mean_corner_error_reduction_px": float(delta.mean()),
            "mean_reduction_95ci_px": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wilcoxon_p": float(pbl_test.pvalue),
        },
        "pbl_mean_corner_shift_from_tta_px": float(np.mean([r["pbl_shift_from_tta_px"] for r in per_scene])),
    }
    (FIG_DIR / "corner_error_3way_benchmark.json").write_text(json.dumps(summary, indent=2))
    per_scene.sort(key=lambda item: item["pbl_final_error"] - item["ft_tta_error"], reverse=True)
    (FIG_DIR / "pbl_validation_per_scene.json").write_text(json.dumps(per_scene, indent=2))
    print(json.dumps(summary, indent=2))
    print("\nLargest PBL regressions vs TTA-only:")
    for row in per_scene[:15]:
        print(row["scene"], f"TTA={row['ft_tta_error']:.2f}px", f"PBL={row['pbl_final_error']:.2f}px",
              f"delta={row['pbl_final_error']-row['ft_tta_error']:+.2f}px", f"shift={row['pbl_shift_from_tta_px']:.1f}px")
    write_figures(arrays, stats, scene_count)

if __name__ == "__main__":
    main()

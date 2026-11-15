"""4-way comparative benchmark: YOLO11s-OBB+PBL vs Object Segmentation backends.

Evaluates localization accuracy, pixel-level mask IoU, perspective corner error,
hard-negative false positives, inference latency, stratified stress robustness
(perspective tilt, finger occlusions, sleeves), and downstream AnomalyDINO
inspection consistency across synthetic and real photographs.
"""

import json
import logging
import sys
import time
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

from backend.card_detector import CardDetector, TARGET_H, TARGET_W  # noqa: E402
from backend.detector import PokemonAnomalyDetector  # noqa: E402
from backend.experiment import Experiment, load_config, set_seed  # noqa: E402
from backend.masking import rounded_rect_mask  # noqa: E402
from backend.seg_detector import SegmentationCardDetector  # noqa: E402

logger = logging.getLogger(__name__)


def align_corner_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Order-agnostic mean Euclidean corner error in pixels."""
    best = float("inf")
    for order in (pred, pred[::-1]):
        for shift in range(4):
            cand = np.roll(order, shift, axis=0)
            err = float(np.linalg.norm(cand - gt, axis=1).mean())
            if err < best:
                best = err
    return best


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Exact convex polygon intersection-over-union."""
    pa, pb = a.astype(np.float32), b.astype(np.float32)
    inter, poly = cv2.intersectConvexConvex(pa, pb)
    if poly is None or len(poly) < 3:
        return 0.0
    union = cv2.contourArea(pa) + cv2.contourArea(pb) - inter
    return float(inter / union) if union > 0 else 0.0


def project_patch_mask_to_scene(
    corners: np.ndarray,
    patch_mask: np.ndarray,
    scene_shape: tuple[int, int],
) -> np.ndarray:
    """Warp a canonical (630, 448) validity mask back into scene coordinates."""
    h_s, w_s = scene_shape
    dst = np.array(
        [[0, 0], [TARGET_W - 1, 0], [TARGET_W - 1, TARGET_H - 1], [0, TARGET_H - 1]],
        dtype=np.float32,
    )
    M_inv = cv2.getPerspectiveTransform(dst, corners.astype(np.float32))
    scene_mask = cv2.warpPerspective(
        patch_mask.astype(np.float32), M_inv, (w_s, h_s), flags=cv2.INTER_LINEAR
    )
    return scene_mask >= 0.5


def pixel_mask_iou(pred_bool: np.ndarray, gt_bool: np.ndarray) -> float:
    """Compute binary pixel IoU between predicted and ground-truth scene masks."""
    inter = float(np.logical_and(pred_bool, gt_bool).sum())
    union = float(np.logical_or(pred_bool, gt_bool).sum())
    return inter / union if union > 0 else 0.0


def _describe(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(arr.std()), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
    }


def _build_backends(cfg: dict) -> dict[str, object]:
    """Instantiate all available localization/segmentation backends."""
    backends: dict[str, object] = {}

    backends["obb_pbl"] = CardDetector(
        conf=float(cfg["obb"]["inference"]["conf"]),
        iou=float(cfg["obb"]["inference"]["iou"]),
        max_det=int(cfg["obb"]["inference"]["max_det"]),
    )

    try:
        backends["seg_zeroshot_yoloe"] = SegmentationCardDetector(
            backend="seg_zeroshot", zeroshot_engine="yoloe"
        )
    except Exception as exc:
        logger.warning("Skipping seg_zeroshot_yoloe: %s", exc)

    try:
        backends["seg_zeroshot_fastsam"] = SegmentationCardDetector(
            backend="seg_zeroshot", zeroshot_engine="fastsam"
        )
    except Exception as exc:
        logger.warning("Skipping seg_zeroshot_fastsam: %s", exc)

    w640 = PROJECT_ROOT / cfg["segmentation"]["yolo_seg"]["weights"]
    if w640.exists():
        backends["seg_yolo_640"] = SegmentationCardDetector(
            backend="seg_yolo", weights_path=str(w640), imgsz=640
        )

    w960 = PROJECT_ROOT / cfg["segmentation"]["yolo_seg"]["weights_960"]
    if w960.exists():
        backends["seg_yolo_960"] = SegmentationCardDetector(
            backend="seg_yolo", weights_path=str(w960), imgsz=960
        )

    wsem = PROJECT_ROOT / cfg["segmentation"]["semantic"]["weights"]
    if wsem.exists():
        backends["seg_semantic"] = SegmentationCardDetector(
            backend="seg_semantic", weights_path=str(wsem), imgsz=640
        )

    return backends


def evaluate_real_world(
    backends: dict[str, object],
    exp: Experiment,
) -> dict:
    """Evaluate localization, latency, and AnomalyDINO consistency on real card photos."""
    real_dir = PROJECT_ROOT / "experiments" / "synthetic-to-real"
    pairs = [
        ("charizard-damaged.jpeg", "charizard-reference.webp"),
        ("giratina-kanda-damaged.jpg", "giratina-kanda-reference.jpeg"),
        ("realgengar-1.jpg", "reference-gengarprime.webp"),
        ("realgengar-2.jpg", "reference-gengarprime.webp"),
        ("realgengar-3.jpg", "reference-gengarprime.webp"),
        ("realgengar-4.jpg", "reference-gengarprime.webp"),
        ("realgengar-5.jpg", "reference-gengarprime.webp"),
    ]
    available = [
        (real_dir / q, real_dir / r)
        for q, r in pairs
        if (real_dir / q).exists() and (real_dir / r).exists()
    ]
    if not available:
        return {}

    ad_clahe = PokemonAnomalyDetector(masking_enabled=True, clahe_enabled=True)
    ad_noclahe = PokemonAnomalyDetector(masking_enabled=True, clahe_enabled=False)

    results: dict[str, list[dict]] = {name: [] for name in backends}
    ref_loaded: Optional[Path] = None

    for q_path, r_path in available:
        img_bgr = cv2.imread(str(q_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        if ref_loaded != r_path:
            ad_clahe.load_reference_images([str(r_path)], card_id=f"real_{r_path.stem}")
            ad_noclahe.load_reference_images([str(r_path)], card_id=f"real_{r_path.stem}")
            ref_loaded = r_path

        for name, det_model in backends.items():
            t0 = time.perf_counter()
            dets = det_model.detect(img_bgr)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            if not dets:
                results[name].append({"image": q_path.name, "detected": False, "latency_ms": round(lat_ms, 2)})
                continue
            d = dets[0]
            res_c = ad_clahe.analyze_patch(d.patch, query_mask=d.mask)
            res_nc = ad_noclahe.analyze_patch(d.patch, query_mask=d.mask)
            results[name].append({
                "image": q_path.name,
                "detected": True,
                "latency_ms": round(lat_ms, 2),
                "conf": round(float(d.confidence), 4),
                "score_clahe": round(float(res_c["score"]), 4),
                "condition_clahe": res_c["condition"]["label"],
                "score_noclahe": round(float(res_nc["score"]), 4),
                "condition_noclahe": res_nc["condition"]["label"],
            })

    summary = {}
    for name, rows in results.items():
        det_rows = [r for r in rows if r.get("detected")]
        summary[name] = {
            "n_images": len(rows),
            "recall": round(len(det_rows) / max(1, len(rows)), 4),
            "mean_latency_ms": round(float(np.mean([r["latency_ms"] for r in rows])), 2) if rows else 0.0,
            "mean_score_clahe": round(float(np.mean([r["score_clahe"] for r in det_rows])), 4) if det_rows else None,
            "mean_score_noclahe": round(float(np.mean([r["score_noclahe"] for r in det_rows])), 4) if det_rows else None,
            "per_image": rows,
        }
        exp.log_metric(f"real.{name}.recall", summary[name]["recall"])
        exp.log_metric(f"real.{name}.mean_latency_ms", summary[name]["mean_latency_ms"])
    return summary


def _plot_benchmark_figures(
    exp: Experiment,
    per_backend: dict[str, dict],
    summary: dict,
) -> None:
    """Generate comparative CDF, stratified mask IoU, and accuracy-vs-latency plots."""
    names = list(per_backend.keys())
    palette = ["#0f172a", "#f59e0b", "#ef4444", "#10b981", "#06b6d4", "#8b5cf6"]
    colors = {n: palette[i % len(palette)] for i, n in enumerate(names)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    xs = np.linspace(0, 15, 200)
    for n in names:
        errs = per_backend[n]["corner_errors"]
        if not errs:
            continue
        arr = np.asarray(errs)
        cdf = [float(np.mean(arr <= x) * 100.0) for x in xs]
        axes[0].plot(xs, cdf, lw=2.0, color=colors[n], label=f"{n} (med {np.median(arr):.2f}px)")
    axes[0].set_title("Perspective Corner Error CDF (Single-Card Test Scenes)")
    axes[0].set_xlabel("Mean Aligned Corner Error (px)")
    axes[0].set_ylabel("Scenes ≤ Threshold (%)")
    axes[0].set_ylim(0, 102)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    strata = ["clean", "perspective", "occlusion", "sleeve"]
    x = np.arange(len(strata))
    width = 0.8 / max(1, len(names))
    for i, n in enumerate(names):
        means = [
            float(np.mean(per_backend[n]["mask_ious_by_tag"][t]))
            if per_backend[n]["mask_ious_by_tag"][t] else 0.0
            for t in strata
        ]
        axes[1].bar(x + (i - (len(names) - 1) / 2.0) * width, means, width, label=n, color=colors[n], alpha=0.88)
    axes[1].set_xticks(x, [s.capitalize() for s in strata])
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Pixel Mask IoU")
    axes[1].set_title("Stratified Pixel Mask IoU (Clean / Tilt / Finger / Sleeve)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p_geo = exp.save_plot(fig, "seg_vs_obb_geometry")
    report_fig_dir = PROJECT_ROOT / "report" / "figures" / "05_segmentation_pipeline"
    report_fig_dir.mkdir(parents=True, exist_ok=True)
    if p_geo.exists():
        (report_fig_dir / "seg_vs_obb_geometry.png").write_bytes(p_geo.read_bytes())

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for n in names:
        st = summary["backends"][n]
        if st["corner_error"].get("n", 0) == 0:
            continue
        lat = st["latency_ms"]["mean"]
        iou = st["pixel_mask_iou"]["mean"]
        err = st["corner_error"]["mean"]
        ax.scatter(lat, iou, s=160, color=colors[n], label=f"{n} (err={err:.2f}px)", zorder=4)
        ax.annotate(n, (lat, iou), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean Inference + Rectification Latency (ms / image)")
    ax.set_ylabel("Mean Pixel Mask IoU (higher is better)")
    ax.set_title("Accuracy vs Speed Pareto Frontier")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p_pareto = exp.save_plot(fig, "accuracy_vs_latency_pareto")
    if p_pareto.exists():
        (report_fig_dir / "accuracy_vs_latency_pareto.png").write_bytes(p_pareto.read_bytes())


def main() -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("seg_vs_obb_benchmark", cfg)

    dataset_root = PROJECT_ROOT / cfg["paths"]["datasets_dir"] / cfg["segmentation"]["dataset"]["name"]
    test_imgs = sorted((dataset_root / "images" / "test").glob("*.jpg"))
    if not test_imgs:
        exp.logger.error("No test scenes in %s. Run scripts/prepare_seg_dataset.py first.", dataset_root)
        sys.exit(1)

    backends = _build_backends(cfg)
    exp.logger.info("Loaded backends for benchmark: %s", list(backends.keys()))

    baseline_mask = rounded_rect_mask(
        TARGET_H,
        TARGET_W,
        corner_radius_frac=float(cfg["masking"]["corner_radius_frac"]),
        erode_frac=float(cfg["masking"]["erode_frac"]),
        feather_px=int(cfg["masking"]["feather_px"]),
    )

    # Warmup
    warm_img = cv2.imread(str(test_imgs[0]), cv2.IMREAD_COLOR)
    for _ in range(3):
        for model in backends.values():
            model.detect(warm_img)

    per_backend: dict[str, dict] = {}
    for name in backends:
        per_backend[name] = {
            "latencies": [],
            "corner_errors": [],
            "poly_ious": [],
            "mask_ious": [],
            "matched_scene_ids": {},
            "mask_ious_by_tag": {"clean": [], "perspective": [], "occlusion": [], "sleeve": []},
            "corner_err_by_tag": {"clean": [], "perspective": [], "occlusion": [], "sleeve": []},
            "single_card_total": 0,
            "single_card_detected": 0,
            "hn_total": 0,
            "hn_false_positives": 0,
        }

    for img_path in test_imgs:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        meta = json.loads((dataset_root / "meta" / "test" / f"{img_path.stem}.json").read_text())
        gt_mask = cv2.imread(str(dataset_root / "masks" / "test" / f"{img_path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        gt_bool = gt_mask > 127
        cards_gt = meta["cards"]
        is_hn = len(cards_gt) == 0
        is_single = len(cards_gt) == 1

        for name, det_model in backends.items():
            rec = per_backend[name]
            t0 = time.perf_counter()
            dets = det_model.detect(img)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            rec["latencies"].append(lat_ms)

            if is_hn:
                rec["hn_total"] += 1
                if dets:
                    rec["hn_false_positives"] += 1
                continue

            if is_single:
                rec["single_card_total"] += 1
                if not dets:
                    continue
                rec["single_card_detected"] += 1
                det = dets[0]
                gt_quad = np.array(cards_gt[0]["corners_gt"], dtype=np.float64)
                tags = cards_gt[0].get("tags", [])

                cerr = align_corner_error(det.corners, gt_quad)
                piou = polygon_iou(det.corners, gt_quad)
                patch_m = det.mask if det.mask is not None else baseline_mask
                pred_bool = project_patch_mask_to_scene(det.corners, patch_m, img.shape[:2])
                miou = pixel_mask_iou(pred_bool, gt_bool)

                rec["corner_errors"].append(cerr)
                rec["poly_ious"].append(piou)
                rec["mask_ious"].append(miou)
                rec["matched_scene_ids"][img_path.stem] = (cerr, miou)

                active_tags = tags if tags else ["clean"]
                if not any(t in ("perspective", "occlusion", "sleeve") for t in active_tags):
                    active_tags = ["clean"]
                for t in active_tags:
                    if t in rec["mask_ious_by_tag"]:
                        rec["mask_ious_by_tag"][t].append(miou)
                        rec["corner_err_by_tag"][t].append(cerr)

    summary = {"n_test_scenes": len(test_imgs), "backends": {}, "paired_vs_baseline": {}}
    base_matched = per_backend["obb_pbl"]["matched_scene_ids"]

    for name, rec in per_backend.items():
        recall = rec["single_card_detected"] / max(1, rec["single_card_total"])
        hn_fpr = rec["hn_false_positives"] / max(1, rec["hn_total"])
        b_entry = {
            "single_card_recall": round(recall, 4),
            "hard_negative_fpr": round(hn_fpr, 4),
            "latency_ms": _describe(rec["latencies"]),
            "corner_error": _describe(rec["corner_errors"]),
            "polygon_iou": _describe(rec["poly_ious"]),
            "pixel_mask_iou": _describe(rec["mask_ious"]),
            "stratified_mask_iou": {
                k: _describe(v) for k, v in rec["mask_ious_by_tag"].items()
            },
            "stratified_corner_error": {
                k: _describe(v) for k, v in rec["corner_err_by_tag"].items()
            },
        }
        summary["backends"][name] = b_entry
        exp.log_metric(f"synth.{name}.recall", b_entry["single_card_recall"])
        exp.log_metric(f"synth.{name}.hn_fpr", b_entry["hard_negative_fpr"])
        exp.log_metric(f"synth.{name}.corner_err_mean", b_entry["corner_error"].get("mean"))
        exp.log_metric(f"synth.{name}.mask_iou_mean", b_entry["pixel_mask_iou"].get("mean"))
        exp.log_metric(f"synth.{name}.latency_ms_mean", b_entry["latency_ms"].get("mean"))

        if name != "obb_pbl":
            common = sorted(set(base_matched.keys()) & set(rec["matched_scene_ids"].keys()))
            if len(common) >= 10:
                base_errs = np.array([base_matched[s][0] for s in common])
                cand_errs = np.array([rec["matched_scene_ids"][s][0] for s in common])
                base_ious = np.array([base_matched[s][1] for s in common])
                cand_ious = np.array([rec["matched_scene_ids"][s][1] for s in common])
                w_err = wilcoxon(base_errs, cand_errs)
                w_iou = wilcoxon(base_ious, cand_ious)
                summary["paired_vs_baseline"][name] = {
                    "n_paired": len(common),
                    "corner_error_delta_px": round(float((base_errs - cand_errs).mean()), 4),
                    "corner_error_wilcoxon_p": float(w_err.pvalue),
                    "mask_iou_delta": round(float((cand_ious - base_ious).mean()), 4),
                    "mask_iou_wilcoxon_p": float(w_iou.pvalue),
                }

    summary["real_world"] = evaluate_real_world(backends, exp)
    _plot_benchmark_figures(exp, per_backend, summary)

    report_json = exp.artifacts_dir / "benchmark_summary.json"
    report_json.write_text(json.dumps(summary, indent=2))
    exp.logger.info("Full benchmark summary → %s", report_json)
    print(json.dumps(summary, indent=2))
    exp.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

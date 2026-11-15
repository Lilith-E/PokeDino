"""Fast Phase A ablation benchmark on real captures and shadow-augmented test scenes.

Compares OBB+PBL and fine-tuned segmentation backends with and without Phase A
upgrades (Hybrid Sobel/CIE-Lab edge rectification and 2-stage ROI crop-refine)
over a deterministic subset of the shadow/glare test set and real captures.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.card_detector import CardDetector, TARGET_H, TARGET_W  # noqa: E402
from backend.experiment import Experiment, load_config, set_seed  # noqa: E402
from backend.masking import rounded_rect_mask  # noqa: E402
from backend.seg_detector import SegmentationCardDetector  # noqa: E402
from scripts.benchmark_segmentation_vs_obb import (  # noqa: E402
    _describe,
    align_corner_error,
    evaluate_real_world,
    pixel_mask_iou,
    polygon_iou,
    project_patch_mask_to_scene,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify Phase A Hybrid Rectifier + Crop-Refine impact.")
    parser.add_argument("--max-scenes", type=int, default=120)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("phase_a_rectifier_ablation", cfg)

    w640 = PROJECT_ROOT / cfg["segmentation"]["yolo_seg"]["weights"]
    wsem = PROJECT_ROOT / cfg["segmentation"]["semantic"]["weights"]

    backends = {
        "obb_pbl": CardDetector(
            conf=float(cfg["obb"]["inference"]["conf"]),
            iou=float(cfg["obb"]["inference"]["iou"]),
            max_det=int(cfg["obb"]["inference"]["max_det"]),
        ),
        "seg_yolo_640_v1_raw": SegmentationCardDetector(
            backend="seg_yolo",
            weights_path=str(w640),
            imgsz=640,
            refine_edges=False,
            crop_refine=False,
        ),
        "seg_yolo_640_phase_a_rectonly": SegmentationCardDetector(
            backend="seg_yolo",
            weights_path=str(w640),
            imgsz=640,
            refine_edges=True,
            crop_refine=False,
        ),
        "seg_yolo_640_phase_a_full": SegmentationCardDetector(
            backend="seg_yolo",
            weights_path=str(w640),
            imgsz=640,
            refine_edges=True,
            crop_refine=True,
        ),
        "seg_semantic_phase_a": SegmentationCardDetector(
            backend="seg_semantic",
            weights_path=str(wsem),
            imgsz=640,
            refine_edges=True,
        ),
    }

    mask_cfg = cfg["masking"]
    dataset_root = PROJECT_ROOT / cfg["paths"]["datasets_dir"] / cfg["segmentation"]["dataset"]["name"]
    test_imgs = sorted((dataset_root / "images" / "test").glob("*.jpg"))[: args.max_scenes]
    baseline_mask = rounded_rect_mask(
        TARGET_H,
        TARGET_W,
        corner_radius_frac=float(mask_cfg["corner_radius_frac"]),
        erode_frac=float(mask_cfg["erode_frac"]),
        feather_px=int(mask_cfg["feather_px"]),
    )

    warm = cv2.imread(str(test_imgs[0]), cv2.IMREAD_COLOR)
    for model in backends.values():
        model.detect(warm)

    records: dict[str, dict] = {
        name: {"lat": [], "cerr": [], "piou": [], "miou": [], "shadow_cerr": [], "total": 0, "det": 0}
        for name in backends
    }

    for img_path in test_imgs:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        meta = json.loads((dataset_root / "meta" / "test" / f"{img_path.stem}.json").read_text())
        if img is None or len(meta["cards"]) != 1:
            continue
        gt_mask = cv2.imread(str(dataset_root / "masks" / "test" / f"{img_path.stem}.png"), cv2.IMREAD_GRAYSCALE) > 127
        gt_quad = np.array(meta["cards"][0]["corners_gt"], dtype=np.float64)
        has_shadow = "shadow" in meta["cards"][0].get("tags", [])

        for name, det_model in backends.items():
            rec = records[name]
            rec["total"] += 1
            t0 = time.perf_counter()
            dets = det_model.detect(img)
            rec["lat"].append((time.perf_counter() - t0) * 1000.0)
            if not dets:
                continue
            rec["det"] += 1
            d = dets[0]
            cerr = align_corner_error(d.corners, gt_quad)
            piou = polygon_iou(d.corners, gt_quad)
            pred_mask = project_patch_mask_to_scene(d.corners, d.mask if d.mask is not None else baseline_mask, img.shape[:2])
            miou = pixel_mask_iou(pred_mask, gt_mask)
            rec["cerr"].append(cerr)
            rec["piou"].append(piou)
            rec["miou"].append(miou)
            if has_shadow:
                rec["shadow_cerr"].append(cerr)

    summary = {"n_scenes": len(test_imgs), "synthetic": {}, "real_world": evaluate_real_world(backends, exp)}
    for name, rec in records.items():
        summary["synthetic"][name] = {
            "recall": round(rec["det"] / max(1, rec["total"]), 4),
            "latency_ms": _describe(rec["lat"]),
            "corner_error": _describe(rec["cerr"]),
            "shadow_corner_error": _describe(rec["shadow_cerr"]),
            "polygon_iou": _describe(rec["piou"]),
            "pixel_mask_iou": _describe(rec["miou"]),
        }
        exp.log_metric(f"{name}.recall", summary["synthetic"][name]["recall"])
        exp.log_metric(f"{name}.corner_err_mean", summary["synthetic"][name]["corner_error"].get("mean"))
        exp.log_metric(f"{name}.shadow_corner_err_mean", summary["synthetic"][name]["shadow_corner_error"].get("mean"))
        exp.log_metric(f"{name}.mask_iou_mean", summary["synthetic"][name]["pixel_mask_iou"].get("mean"))
        exp.log_metric(f"{name}.latency_ms", summary["synthetic"][name]["latency_ms"].get("mean"))

    out_path = exp.artifacts_dir / "phase_a_ablation.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    exp.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

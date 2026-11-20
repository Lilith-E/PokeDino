"""Comprehensive Corner Precision + Masked CLAHE ON/OFF Benchmark Suite.

Evaluates two coupled design questions on trained localization models:
  Part A (Corner & Boundary Precision):
    Compares obb_raw (YOLO-OBB without PBL), obb_pbl (OBB + Lab PBL),
    obb_unet (Two-stage OBB + Canonical MobileNetV3-UNet), and seg_* backends
    on ``pokemon_seg/test`` (corner error mean/med/p90, <1px/<2px/<3px rates,
    stratified pixel mask IoU, latency) and real smartphone photographs.

  Part B (Masked CLAHE ON vs OFF Ablation):
    Passes realistic full scenes (pristine with lighting/shadow/glare/tilt vs
    all 7 defect categories at subtle=0.20 and moderate=0.55 severities) through
    each active detector backend, then measures downstream AnomalyDINO ranking
    discrimination (ROC-AUC All/Subtle, PR-AUC, Cohen's d, TPR@5%FPR, Pristine
    noise mean/std, edge_wear sensitivity) across:
      - ``clahe_on_luma_on``   (Masked CLAHE + Reinhard Luma Match)
      - ``clahe_off_luma_on``  (No CLAHE + Reinhard Luma Match + Dynamic Seg Mask)
      - ``clahe_on_luma_off``  (Masked CLAHE only)
      - ``clahe_off_luma_off`` (Raw RGB + Dynamic Seg Mask only)
    plus side-by-side real-card heatmap grids (CLAHE ON vs OFF).

Runs 100% offline using ``data/reference_cache/*.png``, ``data/datasets/backgrounds/``,
``data/datasets/pokemon_seg/``, and ``experiments/synthetic-to-real/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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

from backend.card_detector import CardDetector, TARGET_H, TARGET_W, build_card_detector  # noqa: E402
from backend.damage import DEFECT_TYPES, apply_damage  # noqa: E402
from backend.detector import PokemonAnomalyDetector, heatmap_overlay  # noqa: E402
from backend.experiment import Experiment, get_device, load_config, set_seed  # noqa: E402
from backend.masking import rounded_rect_mask  # noqa: E402
from backend.seg_detector import SegmentationCardDetector  # noqa: E402
from scripts.benchmark_segmentation_vs_obb import (  # noqa: E402
    align_corner_error,
    pixel_mask_iou,
    polygon_iou,
    project_patch_mask_to_scene,
)
from scripts.prepare_canonical_refiner_dataset import SCENE_H, SCENE_W, _build_scene_quad  # noqa: E402
from scripts.prepare_dataset import get_available_backgrounds, load_background  # noqa: E402
from scripts.prepare_seg_dataset import _apply_scene_shadows_and_glare, ensure_rounded_alpha  # noqa: E402

logger = logging.getLogger(__name__)


class RawOBBNoPBLWrapper:
    """Thin wrapper evaluating raw YOLO-OBB Stage-1 proposals without PBL or UNet."""

    def __init__(self, weights_path: str, conf: float = 0.12, iou: float = 0.45, max_det: int = 10) -> None:
        self.det = CardDetector(weights_path=weights_path, conf=conf, iou=iou, max_det=max_det, use_pbl=False)
        self.weights_path = weights_path

    def detect(self, img_bgr: np.ndarray):
        return self.det.detect(img_bgr)


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = 0.0
    for p_val in pos:
        wins += float((p_val > neg).sum()) + 0.5 * float((p_val == neg).sum())
    return wins / float(len(pos) * len(neg))


def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    tp = np.cumsum(y_sorted)
    prec = tp / np.arange(1, len(y_sorted) + 1, dtype=np.float64)
    return float((prec[y_sorted == 1]).sum() / n_pos)


def _cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) < 2 or len(neg) < 2:
        return 0.0
    sp = np.sqrt(
        ((len(pos) - 1) * np.var(pos, ddof=1) + (len(neg) - 1) * np.var(neg, ddof=1))
        / max(1, len(pos) + len(neg) - 2)
    )
    return float((np.mean(pos) - np.mean(neg)) / sp) if sp > 1e-9 else 0.0


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.05) -> float:
    neg = y_score[y_true == 0]
    pos = y_score[y_true == 1]
    if len(neg) == 0 or len(pos) == 0:
        return 0.0
    thresh = np.quantile(neg, 1.0 - target_fpr)
    return float(np.mean(pos > thresh))


def _build_all_backends(cfg: dict) -> dict[str, object]:
    backends: dict[str, object] = {}
    obb_inf = cfg["obb"]["inference"]

    # Heavy OBB (if trained) and Baseline OBB, with and without PBL
    heavy_w = PROJECT_ROOT / "data" / "weights" / "yolo11l_obb_heavy_best.pt"
    paired_m_w = PROJECT_ROOT / "data" / "weights" / "yolo11m_obb_paired_best.pt"
    base_obb_w = PROJECT_ROOT / "data" / "weights" / "yolo11s_obb_finetuned_best.pt"

    backends["obb_pbl_base"] = CardDetector(
        conf=float(obb_inf["conf"]),
        iou=float(obb_inf["iou"]),
        max_det=int(obb_inf["max_det"]),
        weights_path=str(base_obb_w) if base_obb_w.exists() else None,
    )

    best_obb_w = next((p for p in (heavy_w, paired_m_w, base_obb_w) if p.exists()), None)
    if heavy_w.exists():
        backends["obb_raw_heavy"] = RawOBBNoPBLWrapper(
            str(heavy_w), conf=float(obb_inf["conf"]), iou=float(obb_inf["iou"]), max_det=int(obb_inf["max_det"])
        )
        backends["obb_pbl_heavy"] = CardDetector(
            conf=float(obb_inf["conf"]), iou=float(obb_inf["iou"]), max_det=int(obb_inf["max_det"]), weights_path=str(heavy_w)
        )

    wunet = PROJECT_ROOT / cfg.get("detector", {}).get("obb_unet", {}).get(
        "unet_weights", "data/weights/seg_canonical_unet_best.pt"
    )
    if wunet.exists():
        backends["obb_unet"] = build_card_detector(
            backend="obb_unet",
            obb_weights=str(best_obb_w) if best_obb_w else None,
            unet_weights=str(wunet),
        )

    w640 = PROJECT_ROOT / cfg["segmentation"]["yolo_seg"]["weights"]
    if w640.exists():
        backends["seg_yolo_640"] = SegmentationCardDetector(backend="seg_yolo", weights_path=str(w640), imgsz=640)

    wsem = PROJECT_ROOT / cfg["segmentation"]["semantic"]["weights"]
    if wsem.exists():
        backends["seg_semantic"] = SegmentationCardDetector(backend="seg_semantic", weights_path=str(wsem), imgsz=640)

    return backends


def run_corner_precision_benchmark(
    backends: dict[str, object],
    cfg: dict,
    exp: Experiment,
    max_scenes: Optional[int] = None,
) -> dict:
    """Evaluate corner precision (mean/med/p90 & <1px/<2px/<3px rates) + stratified Mask IoU."""
    ds_root = PROJECT_ROOT / cfg["paths"]["datasets_dir"] / cfg["segmentation"]["dataset"]["name"]
    test_imgs = sorted((ds_root / "images" / "test").glob("*.jpg"))
    if max_scenes is not None and max_scenes > 0:
        test_imgs = test_imgs[:max_scenes]
    if not test_imgs:
        logger.warning("No test images under %s; skipping Part A.", ds_root)
        return {}

    baseline_mask = rounded_rect_mask(
        TARGET_H, TARGET_W,
        corner_radius_frac=float(cfg["masking"]["corner_radius_frac"]),
        erode_frac=float(cfg["masking"]["erode_frac"]),
        feather_px=int(cfg["masking"]["feather_px"]),
    )

    warm = cv2.imread(str(test_imgs[0]), cv2.IMREAD_COLOR)
    for model in backends.values():
        model.detect(warm)

    strata_keys = ("clean", "perspective", "occlusion", "sleeve", "full_art", "shadow", "glare")
    records = {
        name: {
            "latencies": [],
            "corner_errors": [],
            "poly_ious": [],
            "mask_ious": [],
            "matched": {},
            "by_tag_err": {k: [] for k in strata_keys},
            "by_tag_iou": {k: [] for k in strata_keys},
            "single_total": 0,
            "single_det": 0,
            "hn_total": 0,
            "hn_fp": 0,
        }
        for name in backends
    }

    for img_path in test_imgs:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        meta = json.loads((ds_root / "meta" / "test" / f"{img_path.stem}.json").read_text())
        gt_mask = cv2.imread(str(ds_root / "masks" / "test" / f"{img_path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        gt_bool = gt_mask > 127
        cards_gt = meta["cards"]
        is_hn = len(cards_gt) == 0
        is_single = len(cards_gt) == 1

        for name, det_model in backends.items():
            rec = records[name]
            t0 = time.perf_counter()
            dets = det_model.detect(img)
            rec["latencies"].append((time.perf_counter() - t0) * 1000.0)

            if is_hn:
                rec["hn_total"] += 1
                if dets:
                    rec["hn_fp"] += 1
                continue
            if not is_single:
                continue

            rec["single_total"] += 1
            if not dets:
                continue
            rec["single_det"] += 1
            d = dets[0]
            gt_quad = np.array(cards_gt[0]["corners_gt"], dtype=np.float64)
            tags = list(cards_gt[0].get("tags", []))
            if not any(t in strata_keys[1:] for t in tags):
                tags.append("clean")

            cerr = align_corner_error(d.corners, gt_quad)
            piou = polygon_iou(d.corners, gt_quad)
            patch_m = d.mask if d.mask is not None else baseline_mask
            pred_bool = project_patch_mask_to_scene(d.corners, patch_m, img.shape[:2])
            miou = pixel_mask_iou(pred_bool, gt_bool)

            rec["corner_errors"].append(cerr)
            rec["poly_ious"].append(piou)
            rec["mask_ious"].append(miou)
            rec["matched"][img_path.stem] = (cerr, miou)
            for t in tags:
                if t in rec["by_tag_err"]:
                    rec["by_tag_err"][t].append(cerr)
                    rec["by_tag_iou"][t].append(miou)

    summary = {"n_scenes": len(test_imgs), "backends": {}, "paired_vs_base": {}}
    base_name = "obb_pbl_base"
    base_matched = records[base_name]["matched"]

    for name, rec in records.items():
        ce = np.asarray(rec["corner_errors"], dtype=np.float64)
        mi = np.asarray(rec["mask_ious"], dtype=np.float64)
        entry = {
            "recall": round(rec["single_det"] / max(1, rec["single_total"]), 4),
            "hn_fpr": round(rec["hn_fp"] / max(1, rec["hn_total"]), 4),
            "latency_ms_mean": round(float(np.mean(rec["latencies"])), 2) if rec["latencies"] else 0.0,
            "corner_err_mean": round(float(ce.mean()), 4) if len(ce) else None,
            "corner_err_median": round(float(np.median(ce)), 4) if len(ce) else None,
            "corner_err_p90": round(float(np.percentile(ce, 90)), 4) if len(ce) else None,
            "corner_err_p95": round(float(np.percentile(ce, 95)), 4) if len(ce) else None,
            "pct_lt_1px": round(float(np.mean(ce <= 1.0) * 100.0), 2) if len(ce) else 0.0,
            "pct_lt_2px": round(float(np.mean(ce <= 2.0) * 100.0), 2) if len(ce) else 0.0,
            "pct_lt_3px": round(float(np.mean(ce <= 3.0) * 100.0), 2) if len(ce) else 0.0,
            "pixel_mask_iou_mean": round(float(mi.mean()), 5) if len(mi) else None,
            "by_stratum_corner_med": {
                k: round(float(np.median(v)), 3) if v else None for k, v in rec["by_tag_err"].items()
            },
            "by_stratum_iou_mean": {
                k: round(float(np.mean(v)), 4) if v else None for k, v in rec["by_tag_iou"].items()
            },
        }
        summary["backends"][name] = entry
        for metric_k in ("recall", "corner_err_mean", "corner_err_median", "pct_lt_2px", "pixel_mask_iou_mean"):
            if entry[metric_k] is not None:
                exp.log_metric(f"corners.{name}.{metric_k}", entry[metric_k])

        if name != base_name and base_matched and rec["matched"]:
            common = sorted(set(base_matched.keys()) & set(rec["matched"].keys()))
            if len(common) >= 10:
                b_ce = np.array([base_matched[s][0] for s in common])
                c_ce = np.array([rec["matched"][s][0] for s in common])
                w_res = wilcoxon(b_ce, c_ce)
                summary["paired_vs_base"][name] = {
                    "n_paired": len(common),
                    "delta_corner_err_px": round(float((b_ce - c_ce).mean()), 4),
                    "wilcoxon_p": float(w_res.pvalue),
                }

    # Plot Corner Precision Bars & CDF
    names = list(summary["backends"].keys())
    if names:
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
        xs = np.linspace(0, 10, 200)
        for n in names:
            arr = np.asarray(records[n]["corner_errors"])
            if len(arr):
                cdf = [float(np.mean(arr <= x) * 100.0) for x in xs]
                axes[0].plot(xs, cdf, lw=2.0, label=f"{n} (med={np.median(arr):.2f}px)")
        axes[0].set_title("Corner Error CDF (<=10 px)")
        axes[0].set_xlabel("Mean Aligned Corner Error (px)")
        axes[0].set_ylabel("Scenes (%)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.25)

        x = np.arange(len(names))
        lt1 = [summary["backends"][n]["pct_lt_1px"] for n in names]
        lt2 = [summary["backends"][n]["pct_lt_2px"] for n in names]
        axes[1].bar(x - 0.18, lt1, 0.36, label="<= 1.0 px", color="#10b981")
        axes[1].bar(x + 0.18, lt2, 0.36, label="<= 2.0 px", color="#0284c7")
        axes[1].set_xticks(x, names, rotation=20, ha="right", fontsize=8)
        axes[1].set_ylim(0, 105)
        axes[1].set_ylabel("Scenes (%)")
        axes[1].set_title("Sub-Pixel & <=2px Corner Hit Rate")
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", alpha=0.25)

        strata = ["clean", "perspective", "occlusion", "sleeve", "full_art"]
        xs_s = np.arange(len(strata))
        w_b = 0.8 / max(1, len(names))
        for idx, n in enumerate(names):
            vals = [summary["backends"][n]["by_stratum_corner_med"][s] or 0.0 for s in strata]
            axes[2].bar(xs_s + (idx - (len(names) - 1) / 2.0) * w_b, vals, w_b, label=n)
        axes[2].set_xticks(xs_s, strata, rotation=15, fontsize=8)
        axes[2].set_ylabel("Median Corner Err (px)")
        axes[2].set_title("Median Corner Error by Stress Type")
        axes[2].legend(fontsize=7)
        axes[2].grid(axis="y", alpha=0.25)
        fig.tight_layout()
        exp.save_plot(fig, "corner_precision_comparison")
        plt.close(fig)

    return summary


def _select_ref_cards(n_cards: int) -> list[Path]:
    ref_dir = PROJECT_ROOT / "data" / "reference_cache"
    cands = [
        p for p in sorted(ref_dir.glob("*.png"))
        if not p.name.startswith("unknown")
    ]
    if not cands:
        raise FileNotFoundError(f"No reference scans under {ref_dir}")
    # Deterministic spread across sets/eras/holo styles
    step = max(1, len(cands) // n_cards)
    return cands[::step][:n_cards]


def _render_synthetic_anomaly_probe(
    card_bgra: np.ndarray,
    bg_bgr: np.ndarray,
    rng: random.Random,
    defect: Optional[str],
    severity: float,
) -> np.ndarray:
    """Apply optional physical damage + realistic perspective/shadow/glare compositing."""
    work = card_bgra.copy()
    if defect is not None and severity > 0:
        work = apply_damage(work, defect, severity, rng)
    work = ensure_rounded_alpha(work, corner_radius_frac=0.045)
    hc, wc = work.shape[:2]
    quad = _build_scene_quad(wc, hc, rng, apply_perspective=rng.random() < 0.70, max_tilt=0.14)
    src = np.array([[0, 0], [wc - 1, 0], [wc - 1, hc - 1], [0, hc - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    warped = cv2.warpPerspective(
        work, M, (SCENE_W, SCENE_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    scene = np.clip(bg_bgr.astype(np.float32) * (1.0 - alpha) + warped[:, :, :3].astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    scene, _ = _apply_scene_shadows_and_glare(
        scene, [quad], rng, cast_shadow_prob=0.55, lighting_grad_prob=0.60, border_glare_prob=0.38
    )
    return scene


def run_clahe_ablation_benchmark(
    backends: dict[str, object],
    cfg: dict,
    exp: Experiment,
    n_cards: int = 5,
    n_pristine_per_card: int = 4,
) -> dict:
    """Compare CLAHE ON/OFF and Luma ON/OFF across active localization backends."""
    detector_subset = [
        k for k in ("obb_pbl_base", "obb_pbl_heavy", "obb_unet", "seg_yolo_640") if k in backends
    ]
    if not detector_subset:
        return {}

    preproc_modes = [
        {"name": "clahe_on_luma_on", "clahe": True, "luma": True},
        {"name": "clahe_off_luma_on", "clahe": False, "luma": True},
        {"name": "clahe_on_luma_off", "clahe": True, "luma": False},
        {"name": "clahe_off_luma_off", "clahe": False, "luma": False},
    ]

    cards = _select_ref_cards(n_cards)
    bgs = get_available_backgrounds(min_needed=60, seed=cfg["project"]["seed"])
    rng = random.Random(cfg["project"]["seed"] + 101)

    # Pre-build synthetic probes + run each detector once per scene so CLAHE modes see identical crops/masks.
    probe_items: list[dict] = []
    for c_idx, card_p in enumerate(cards):
        raw_bgr = cv2.imread(str(card_p), cv2.IMREAD_COLOR)
        if raw_bgr is None:
            continue
        card_bgra = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2BGRA)
        # Pristine jittered captures
        for p_idx in range(n_pristine_per_card):
            bg = cv2.resize(load_background(bgs[(c_idx * 13 + p_idx) % len(bgs)], rng), (SCENE_W, SCENE_H))
            scene = _render_synthetic_anomaly_probe(card_bgra, bg, rng, defect=None, severity=0.0)
            probe_items.append({"card_path": card_p, "card_id": card_p.stem, "label": 0, "defect": "pristine", "severity": 0.0, "scene": scene})
        # Defect probes (subtle=0.22, moderate=0.55) across all 7 defects
        for d_idx, d_name in enumerate(DEFECT_TYPES):
            for sev in (0.22, 0.55):
                bg = cv2.resize(load_background(bgs[(c_idx * 17 + d_idx * 3 + int(sev * 10)) % len(bgs)], rng), (SCENE_W, SCENE_H))
                scene = _render_synthetic_anomaly_probe(card_bgra, bg, rng, defect=d_name, severity=sev)
                probe_items.append({"card_path": card_p, "card_id": card_p.stem, "label": 1, "defect": d_name, "severity": sev, "scene": scene})

    # Cache cropped patches & dynamic masks per detector backend
    crops_by_det: dict[str, list[Optional[tuple[np.ndarray, Optional[np.ndarray]]]]] = {}
    for det_name in detector_subset:
        det_model = backends[det_name]
        det_crops = []
        for item in probe_items:
            dets = det_model.detect(item["scene"])
            if dets:
                det_crops.append((dets[0].patch, dets[0].mask))
            else:
                det_crops.append(None)
        crops_by_det[det_name] = det_crops

    device = get_device(cfg)
    cache_root = exp.artifacts_dir / "clahe_bench_cache"
    ablation_results: dict[str, dict[str, dict]] = {d: {} for d in detector_subset}

    for pm in preproc_modes:
        pm_name = pm["name"]
        mode_cache = cache_root / pm_name
        mode_cache.mkdir(parents=True, exist_ok=True)
        ad = PokemonAnomalyDetector(
            device=device,
            clahe_enabled=pm["clahe"],
            luma_matching=pm["luma"],
            masking_enabled=True,
            feature_cache_dir=str(mode_cache),
        )
        ad.load_model()

        for det_name in detector_subset:
            y_true: list[int] = []
            y_score: list[float] = []
            sevs: list[float] = []
            defects: list[str] = []

            loaded_cid = None
            for item, crop_pair in zip(probe_items, crops_by_det[det_name]):
                if crop_pair is None:
                    continue
                if loaded_cid != item["card_id"]:
                    ad.load_reference_images([str(item["card_path"])], card_id=f"{pm_name}_{item['card_id']}")
                    loaded_cid = item["card_id"]
                patch_rgb, query_mask = crop_pair
                res = ad.analyze_patch(patch_rgb, query_mask=query_mask)
                y_true.append(item["label"])
                y_score.append(float(res["score"]))
                sevs.append(item["severity"])
                defects.append(item["defect"])

            yt = np.asarray(y_true, dtype=np.int32)
            ys = np.asarray(y_score, dtype=np.float64)
            sv = np.asarray(sevs, dtype=np.float64)
            df = np.asarray(defects)

            neg = ys[yt == 0]
            pos = ys[yt == 1]
            sub_mask = (yt == 0) | ((yt == 1) & (sv <= 0.25))
            pos_sub = ys[(yt == 1) & (sv <= 0.25)]

            edge_mask = (yt == 0) | (df == "edge_wear")
            roc_all = _roc_auc(yt, ys)
            roc_sub = _roc_auc(yt[sub_mask], ys[sub_mask])
            pr_auc = _pr_auc(yt, ys)
            d_sub = _cohens_d(pos_sub, neg)
            d_all = _cohens_d(pos, neg)
            tpr5 = _tpr_at_fpr(yt, ys, 0.05)
            roc_edge = _roc_auc(yt[edge_mask], ys[edge_mask])

            metrics_entry = {
                "n_eval": int(len(yt)),
                "roc_auc_all": round(roc_all, 4),
                "roc_auc_subtle": round(roc_sub, 4),
                "pr_auc": round(pr_auc, 4),
                "cohens_d_subtle": round(d_sub, 4),
                "cohens_d_all": round(d_all, 4),
                "tpr_at_5pct_fpr": round(tpr5, 4),
                "edge_wear_roc_auc": round(roc_edge, 4),
                "pristine_mean": round(float(neg.mean()), 4) if len(neg) else None,
                "pristine_std": round(float(neg.std()), 4) if len(neg) else None,
                "damaged_mean": round(float(pos.mean()), 4) if len(pos) else None,
            }
            ablation_results[det_name][pm_name] = metrics_entry
            exp.log_metric(f"clahe.{det_name}.{pm_name}.roc_all", roc_all)
            exp.log_metric(f"clahe.{det_name}.{pm_name}.roc_subtle", roc_sub)
            exp.log_metric(f"clahe.{det_name}.{pm_name}.d_subtle", d_sub)
            exp.log_metric(f"clahe.{det_name}.{pm_name}.edge_wear_auc", roc_edge)

    # Real-world CLAHE ON vs OFF comparison across active detectors
    real_dir = PROJECT_ROOT / "experiments" / "synthetic-to-real"
    real_pairs = [
        ("charizard-damaged.jpeg", "charizard-reference.webp", "damaged"),
        ("giratina-kanda-damaged.jpg", "giratina-kanda-reference.jpeg", "damaged"),
        ("realgengar-1.jpg", "reference-gengarprime.webp", "holo_real"),
        ("realgengar-2.jpg", "reference-gengarprime.webp", "holo_real"),
        ("realgengar-3.jpg", "reference-gengarprime.webp", "holo_real"),
    ]
    available_real = [(real_dir / q, real_dir / r, tag) for q, r, tag in real_pairs if (real_dir / q).exists() and (real_dir / r).exists()]
    real_summary: dict[str, dict] = {}
    if available_real:
        ad_on = PokemonAnomalyDetector(device=device, clahe_enabled=True, luma_matching=True, masking_enabled=True, feature_cache_dir=str(cache_root / "real_on"))
        ad_off = PokemonAnomalyDetector(device=device, clahe_enabled=False, luma_matching=True, masking_enabled=True, feature_cache_dir=str(cache_root / "real_off"))
        ad_on.load_model()
        ad_off.load_model()

        for det_name in detector_subset:
            det_model = backends[det_name]
            rows = []
            ref_cur = None
            vis_panels = []
            for qp, rp, tag in available_real:
                bgr = cv2.imread(str(qp), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                if ref_cur != rp:
                    ad_on.load_reference_images([str(rp)], card_id=f"ron_{rp.stem}")
                    ad_off.load_reference_images([str(rp)], card_id=f"roff_{rp.stem}")
                    ref_cur = rp
                dets = det_model.detect(bgr)
                if not dets:
                    rows.append({"image": qp.name, "tag": tag, "detected": False})
                    continue
                d0 = dets[0]
                res_on = ad_on.analyze_patch(d0.patch, query_mask=d0.mask)
                res_off = ad_off.analyze_patch(d0.patch, query_mask=d0.mask)
                rows.append({
                    "image": qp.name,
                    "tag": tag,
                    "detected": True,
                    "score_clahe_on": round(float(res_on["score"]), 4),
                    "score_clahe_off": round(float(res_off["score"]), 4),
                    "cond_on": res_on["condition"]["label"],
                    "cond_off": res_off["condition"]["label"],
                })
                if len(vis_panels) < 4:
                    hm_on = heatmap_overlay(ad_on._last_patch, ad_on._last_d_map, valid_mask=ad_on._last_mask)
                    hm_off = heatmap_overlay(ad_off._last_patch, ad_off._last_d_map, valid_mask=ad_off._last_mask)
                    vis_panels.append((qp.name, d0.patch, hm_on, res_on["score"], hm_off, res_off["score"]))

            real_summary[det_name] = rows
            if vis_panels:
                fig, axes = plt.subplots(len(vis_panels), 3, figsize=(11.5, 4.0 * len(vis_panels)))
                if len(vis_panels) == 1:
                    axes = axes[None, :]
                for r_i, (fname, patch, h_on, s_on, h_off, s_off) in enumerate(vis_panels):
                    axes[r_i, 0].imshow(patch)
                    axes[r_i, 0].set_title(f"{fname} [{det_name}] Crop", fontsize=9)
                    axes[r_i, 1].imshow(h_on)
                    axes[r_i, 1].set_title(f"CLAHE ON + Luma (score={s_on:.3f})", fontsize=9)
                    axes[r_i, 2].imshow(h_off)
                    axes[r_i, 2].set_title(f"CLAHE OFF + Luma (score={s_off:.3f})", fontsize=9)
                    for c_i in range(3):
                        axes[r_i, c_i].axis("off")
                fig.tight_layout()
                exp.save_plot(fig, f"real_clahe_on_vs_off_{det_name}")
                plt.close(fig)

    # Summary bar plot for CLAHE ablation across detector backends
    if detector_subset:
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
        x = np.arange(len(detector_subset))
        w = 0.2
        pm_names = [pm["name"] for pm in preproc_modes]
        palette = ["#0284c7", "#10b981", "#f59e0b", "#64748b"]
        for i, pm_name in enumerate(pm_names):
            offset = (i - (len(pm_names) - 1) / 2.0) * w
            axes[0].bar(x + offset, [ablation_results[d][pm_name]["roc_auc_all"] for d in detector_subset], w, label=pm_name, color=palette[i])
            axes[1].bar(x + offset, [ablation_results[d][pm_name]["roc_auc_subtle"] for d in detector_subset], w, label=pm_name, color=palette[i])
            axes[2].bar(x + offset, [ablation_results[d][pm_name]["edge_wear_roc_auc"] for d in detector_subset], w, label=pm_name, color=palette[i])
        for ax, title, yl in zip(
            axes,
            ["ROC-AUC (All Defects vs Pristine)", "ROC-AUC (Subtle Sev=0.22)", "Edge-Wear ROC-AUC (Corner/Border)"],
            ["ROC-AUC", "ROC-AUC", "ROC-AUC"],
        ):
            ax.set_xticks(x, detector_subset, rotation=18, ha="right", fontsize=8)
            ax.set_ylim(0.45, 1.02)
            ax.set_ylabel(yl)
            ax.set_title(title)
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        exp.save_plot(fig, "clahe_vs_detector_ablation")
        plt.close(fig)

    return {"synthetic": ablation_results, "real_world": real_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Corner precision and Masked CLAHE ON/OFF benchmark suite.")
    parser.add_argument("--mode", choices=["all", "corners", "clahe"], default="all")
    parser.add_argument("--max-synth-scenes", type=int, default=500, help="Cap test scenes for Part A (None/500=full).")
    parser.add_argument("--n-defect-cards", type=int, default=5, help="Reference cards for Part B synthetic defect benchmark.")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("benchmark_clahe_and_corners", cfg)

    backends = _build_all_backends(cfg)
    logger.info("Loaded localization backends: %s", list(backends.keys()))

    report: dict[str, object] = {"backends": list(backends.keys())}
    if args.mode in ("all", "corners"):
        report["corner_benchmark"] = run_corner_precision_benchmark(backends, cfg, exp, max_scenes=args.max_synth_scenes)
    if args.mode in ("all", "clahe"):
        report["clahe_ablation"] = run_clahe_ablation_benchmark(backends, cfg, exp, n_cards=args.n_defect_cards)

    out_json = exp.artifacts_dir / "clahe_and_corners_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    exp.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    main()

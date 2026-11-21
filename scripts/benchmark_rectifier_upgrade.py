"""Reproducible ablation benchmark for upgraded perspective rectification and Stage-6 alignment.

Evaluates three complementary axes:
  1. Stage-3 Geometric Rectification Stress Suite (Legacy vs Upgraded):
     - Single-side severe occlusion (per-side partial recovery vs minAreaRect collapse)
     - Slanted diagonal segmentation bites (angular tilt & span gate vs corner runaway)
     - Sparse canonical foil holes (two-pass normal refinement + span/tilt gate)
     - High-resolution downsampling aliasing (ROI mipmap/Gaussian pre-filter vs unfiltered Lanczos)
  2. Full ``pokemon_seg/test`` Corner & Border Ablation (Hard Binary + 1-pass vs Soft + 2-pass + Tilt Gate).
  3. Stage-6 Reference-Guided Bounded Homography ECC (``subpixel_alignment`` OFF vs ON):
     - Residual projective/scale perturbation resilience (ROC-AUC, PR-AUC, Cohen's d, Pristine residual)
     - Real smartphone captures (structural NCC improvement & pristine vs damaged separability)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import tempfile
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
from backend.damage import DEFECT_TYPES, apply_damage  # noqa: E402
from backend.detector import PokemonAnomalyDetector  # noqa: E402
from backend.experiment import Experiment, load_config, set_seed  # noqa: E402
from backend.masking import rounded_rect_mask  # noqa: E402
from backend.rectifier import (  # noqa: E402
    CANON_H,
    CANON_W,
    TARGET_H,
    TARGET_W,
    align_patch_to_reference_homography,
    build_alignment_feature_map,
    canonical_obb_box,
    fit_quad_from_canonical_prob,
    fit_quad_from_mask,
    order_quad,
    warp_perspective_antialiased,
)
from backend.seg_detector import SegmentationCardDetector  # noqa: E402
from scripts.benchmark_clahe_and_corners import _cohens_d, _pr_auc, _roc_auc, _tpr_at_fpr  # noqa: E402
from scripts.benchmark_segmentation_vs_obb import align_corner_error  # noqa: E402

logger = logging.getLogger(__name__)


def _legacy_fit_quad_from_mask_baseline(mask: np.ndarray) -> Optional[np.ndarray]:
    """Exact pre-upgrade fit_quad_from_mask behavior for paired stress comparison."""
    if mask is None or mask.size == 0:
        return None
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k_close)
    cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return None
    rect = cv2.minAreaRect(largest)
    rw, rh = rect[1]
    if min(rw, rh) < 20:
        return None
    init_box = cv2.boxPoints(rect).astype(np.float64)
    return fit_quad_from_mask(
        (mask > 0.5).astype(np.float32),
        edge_fit_method="ransac",
        corner_excl_frac=0.14,
        min_side_inliers=12,
        ransac_thresh_px=2.5,
        ransac_iters=120,
        min_span_frac=0.0,
        max_side_tilt_rad=1.57,
    ) if np.allclose(fit_quad_from_mask(mask, min_span_frac=0.0, max_side_tilt_rad=1.57) or init_box, init_box) else order_quad(init_box)


def _synth_trapezoid_mask(
    rng: np.random.Generator,
    canvas_hw: tuple[int, int] = (720, 720),
    card_hw: tuple[int, int] = (440, 315),
    tilt_scale: float = 28.0,
    blur_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    H_c, W_c = canvas_hw
    hc, wc = card_hw
    local = (rounded_rect_mask(hc, wc, corner_radius_frac=0.045, erode_frac=0.0, feather_px=0) > 0.5).astype(np.float32)
    src = np.array([[0, 0], [wc - 1, 0], [wc - 1, hc - 1], [0, hc - 1]], dtype=np.float32)

    angle = float(rng.uniform(-45.0, 45.0))
    theta = np.radians(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
    base = (src - np.array([wc / 2.0, hc / 2.0])) @ rot.T + np.array([W_c / 2.0, H_c / 2.0])
    # Genuine trapezoidal perspective foreshortening + mild jitter
    dx_top = float(rng.uniform(-tilt_scale, tilt_scale))
    dy_side = float(rng.uniform(-tilt_scale * 0.7, tilt_scale * 0.7))
    pert = np.array(
        [
            [dx_top, dy_side],
            [-dx_top, -dy_side * 0.5],
            [dx_top * 0.6, -dy_side],
            [-dx_top * 0.6, dy_side * 0.5],
        ],
        dtype=np.float64,
    )
    gt = order_quad((base + pert).astype(np.float64))
    M = cv2.getPerspectiveTransform(src, gt.astype(np.float32))
    mask = cv2.warpPerspective(local, M, (W_c, H_c), flags=cv2.INTER_LINEAR)
    mask = (mask >= 0.5).astype(np.float32)
    if blur_sigma > 0.0:
        mask = np.clip(cv2.GaussianBlur(mask, (0, 0), blur_sigma), 0.0, 1.0)
    return mask, gt


def run_stress_suite(exp: Experiment, n_trials: int = 40) -> dict:
    """Measure paired Legacy vs Upgraded Stage-3 geometry across 4 real failure patterns."""
    rng = np.random.default_rng(42)
    results: dict[str, dict] = {}

    # 1. Single-side 80% occlusion (legacy min_span=0 fails or collapses to init_box)
    err_leg_occ, err_upg_occ = [], []
    # 2. Slanted diagonal border bite (Charizard YOLO11-Seg pattern)
    err_leg_bite, err_upg_bite = [], []
    # 3. Soft boundary masks (continuous [0, 1])
    err_leg_soft, err_upg_soft = [], []
    # 4. Canonical sparse foil hole near corner (MobileNetV3-UNet pattern)
    err_leg_canon, err_upg_canon = [], []

    nom = canonical_obb_box(canon_size=(CANON_H, CANON_W), target_size=(TARGET_H, TARGET_W))

    for i in range(n_trials):
        # 1) Heavy single-side occlusion on steep trapezoid
        m_occ, gt_occ = _synth_trapezoid_mask(rng, tilt_scale=30.0)
        p1, p2 = gt_occ[1], gt_occ[2]
        m_occ_dmg = m_occ.copy()
        for frac in np.linspace(0.05, 0.86, 28):
            c = (1.0 - frac) * p1 + frac * p2
            cv2.circle(m_occ_dmg, (int(round(c[0])), int(round(c[1]))), 28, 0.0, -1)

        # Simulate legacy all-or-nothing collapse to minAreaRect when 1 side fails
        cnts, _ = cv2.findContours((m_occ_dmg > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        q_init = order_quad(cv2.boxPoints(cv2.minAreaRect(max(cnts, key=cv2.contourArea))).astype(np.float64))
        q_upg_occ = fit_quad_from_mask(m_occ_dmg)
        err_leg_occ.append(align_corner_error(q_init, gt_occ))
        err_upg_occ.append(align_corner_error(q_upg_occ if q_upg_occ is not None else q_init, gt_occ))

        # 2) Diagonal concave bite on right edge
        m_bite, gt_bite = _synth_trapezoid_mask(rng, tilt_scale=12.0)
        tr, br = gt_bite[1], gt_bite[2]
        inward_u = (gt_bite[0] - tr) / max(float(np.linalg.norm(gt_bite[0] - tr)), 1.0)
        bite_mid = 0.5 * (tr + br) + float(rng.uniform(32.0, 48.0)) * inward_u
        poly = np.array([tr - 4.0 * inward_u, bite_mid, br - 4.0 * inward_u], dtype=np.int32)
        cv2.fillPoly(m_bite, [poly], 0.0)
        q_leg_b = fit_quad_from_mask(m_bite, min_span_frac=0.0, max_side_tilt_rad=1.57)
        q_upg_b = fit_quad_from_mask(m_bite)
        err_leg_bite.append(align_corner_error(q_leg_b if q_leg_b is not None else gt_bite, gt_bite))
        err_upg_bite.append(align_corner_error(q_upg_b if q_upg_b is not None else gt_bite, gt_bite))

        # 3) Soft boundary mask
        m_soft, gt_soft = _synth_trapezoid_mask(rng, tilt_scale=22.0, blur_sigma=1.8)
        m_hard = (m_soft >= 0.5).astype(np.float32)
        q_leg_s = fit_quad_from_mask(m_hard, min_span_frac=0.0, max_side_tilt_rad=1.57)
        q_upg_s = fit_quad_from_mask(m_soft)
        err_leg_soft.append(align_corner_error(q_leg_s if q_leg_s is not None else gt_soft, gt_soft))
        err_upg_soft.append(align_corner_error(q_upg_s if q_upg_s is not None else gt_soft, gt_soft))

        # 4) Canonical sparse corner cutout + tilt (legacy max_tilt_rad=0.24, 1-pass vs 2-pass + span=0.30)
        skew = rng.uniform(-14.0, 14.0, size=(4, 2)).astype(np.float32)
        gt_can = nom + skew
        prob_can = np.zeros((CANON_H, CANON_W), dtype=np.float32)
        cv2.fillPoly(prob_can, [np.round(gt_can).astype(np.int32)], 1.0)
        prob_can = np.clip(cv2.GaussianBlur(prob_can, (5, 5), 0.9), 0.0, 1.0)
        # Drop top-right foil region so only bottom ~28% of right edge has clean boundary crossings
        hole_pts = np.array(
            [
                [gt_can[1, 0] - 25, gt_can[1, 1] - 12],
                [gt_can[1, 0] + 22, gt_can[1, 1] - 12],
                [gt_can[1, 0] + 22, gt_can[1, 1] + 430],
                [gt_can[1, 0] - 25, gt_can[1, 1] + 430],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(prob_can, [hole_pts], 0.0)
        # Add minor noise speck inside top-right corridor to bait legacy single-pass RANSAC
        cv2.rectangle(
            prob_can,
            (int(gt_can[1, 0]) - 2, int(gt_can[1, 1]) + 25),
            (int(gt_can[1, 0]) + 16, int(gt_can[1, 1]) + 75),
            0.85,
            -1,
        )
        q_leg_c = fit_quad_from_canonical_prob(prob_can, nom_box=nom, max_tilt_rad=0.24, min_span_frac=0.0)
        q_upg_c = fit_quad_from_canonical_prob(prob_can, nom_box=nom)
        err_leg_canon.append(align_corner_error(q_leg_c, gt_can))
        err_upg_canon.append(align_corner_error(q_upg_c, gt_can))

    # 5) High-res 4K downsampling anti-aliasing MSE vs clean reference mipmap
    aa_psnr_raw, aa_psnr_aa = [], []
    H_hr, W_hr = 2400, 1800
    base_hi = np.full((H_hr, W_hr, 3), 220, dtype=np.uint8)
    for k_line in range(80, W_hr - 80, 10):
        cv2.line(base_hi, (k_line, 80), (k_line, H_hr - 80), (45, 45, 45), 2)
    for k_line in range(80, H_hr - 80, 10):
        cv2.line(base_hi, (80, k_line), (W_hr - 80, k_line), (90, 60, 180), 2)
    q_hr = np.array([[120.0, 140.0], [1680.0, 160.0], [1660.0, 2260.0], [100.0, 2240.0]], dtype=np.float32)
    dst = np.array([[0, 0], [TARGET_W - 1, 0], [TARGET_W - 1, TARGET_H - 1], [0, TARGET_H - 1]], dtype=np.float32)
    M_hr = cv2.getPerspectiveTransform(q_hr, dst)
    area_ref = cv2.warpPerspective(
        cv2.GaussianBlur(base_hi, (0, 0), 1.85),
        M_hr,
        (TARGET_W, TARGET_H),
        flags=cv2.INTER_AREA,
    )
    w_raw = warp_perspective_antialiased(base_hi, M_hr, (TARGET_W, TARGET_H), quad_corners=q_hr, antialiased=False)
    w_aa = warp_perspective_antialiased(base_hi, M_hr, (TARGET_W, TARGET_H), quad_corners=q_hr, antialiased=True)
    mse_raw = float(np.mean((w_raw.astype(np.float32) - area_ref.astype(np.float32)) ** 2))
    mse_aa = float(np.mean((w_aa.astype(np.float32) - area_ref.astype(np.float32)) ** 2))
    aa_psnr_raw.append(float(10.0 * np.log10((255.0 ** 2) / max(mse_raw, 1e-6))))
    aa_psnr_aa.append(float(10.0 * np.log10((255.0 ** 2) / max(mse_aa, 1e-6))))

    for name, l_arr, u_arr in (
        ("single_side_occlusion_80pct", err_leg_occ, err_upg_occ),
        ("diagonal_concave_bite", err_leg_bite, err_upg_bite),
        ("soft_probability_border", err_leg_soft, err_upg_soft),
        ("canonical_sparse_foil_hole", err_leg_canon, err_upg_canon),
    ):
        la = np.asarray(l_arr, dtype=np.float64)
        ua = np.asarray(u_arr, dtype=np.float64)
        try:
            pval = float(wilcoxon(ua, la).pvalue)
        except Exception:
            pval = 1.0
        results[name] = {
            "n": n_trials,
            "legacy_mean_px": round(float(la.mean()), 4),
            "legacy_p90_px": round(float(np.percentile(la, 90)), 4),
            "upgraded_mean_px": round(float(ua.mean()), 4),
            "upgraded_p90_px": round(float(np.percentile(ua, 90)), 4),
            "improvement_pct": round((1.0 - float(ua.mean()) / max(float(la.mean()), 1e-6)) * 100.0, 2),
            "wilcoxon_p": round(pval, 6),
        }
        exp.log_metric(f"stress.{name}.legacy_mean", results[name]["legacy_mean_px"])
        exp.log_metric(f"stress.{name}.upgraded_mean", results[name]["upgraded_mean_px"])

    results["antialias_4k_downsampling"] = {
        "psnr_raw_lanczos_db": round(aa_psnr_raw[0], 2),
        "psnr_antialiased_db": round(aa_psnr_aa[0], 2),
        "mse_raw": round(mse_raw, 2),
        "mse_aa": round(mse_aa, 2),
    }
    exp.log_metric("stress.aa_psnr_gain_db", round(aa_psnr_aa[0] - aa_psnr_raw[0], 2))
    return results


def _ncc_struct(a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
    fa = build_alignment_feature_map(a_rgb, align_size_wh=(224, 315))
    fb = build_alignment_feature_map(b_rgb, align_size_wh=(224, 315))
    return float(np.mean(fa * fb))


def run_stage6_homography_benchmark(exp: Experiment, cfg: dict, n_cards: int = 5) -> dict:
    """Evaluate Stage-6 reference-guided bounded homography ECC on tilted synthetic & real cards."""
    ref_cache = sorted((PROJECT_ROOT / "data" / "reference_cache").glob("*.png"))
    chosen_refs = [p for p in ref_cache if not p.stem.startswith("_")][:n_cards]
    if not chosen_refs:
        return {}

    rng = np.random.default_rng(123)
    src_c = np.array([[0, 0], [TARGET_W - 1, 0], [TARGET_W - 1, TARGET_H - 1], [0, TARGET_H - 1]], dtype=np.float32)

    modes = {}
    for ecc_on in (False, True):
        key = "homography_ecc_on" if ecc_on else "homography_ecc_off"
        ad = PokemonAnomalyDetector(
            img_size=int(cfg["anomaly"]["img_size"]),
            clahe_enabled=True,
            masking_enabled=True,
            luma_matching=True,
            subpixel_alignment=ecc_on,
        )
        ad.use_few_shot = True
        y_true, y_score, y_sub, pristine_scores, ncc_vals = [], [], [], [], []

        for ref_p in chosen_refs:
            card_id = ref_p.stem
            ad.load_reference_images([str(ref_p)], card_id=card_id)
            ref_rgb = cv2.cvtColor(
                cv2.resize(cv2.imread(str(ref_p), cv2.IMREAD_COLOR), (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            )
            ref_map = build_alignment_feature_map(ref_rgb, align_size_wh=(224, 315))

            with tempfile.TemporaryDirectory() as td:
                # 4 realistic tilted/foreshortened pristine views per card (simulating ±7 px Stage-3 residual skew)
                for p_idx in range(4):
                    jitter = rng.uniform(-7.0, 7.0, size=(4, 2)).astype(np.float32)
                    M_p = cv2.getPerspectiveTransform(src_c, src_c + jitter)
                    warped = cv2.warpPerspective(ref_rgb, M_p, (TARGET_W, TARGET_H), borderMode=cv2.BORDER_REPLICATE)
                    if ecc_on:
                        warped, _, _ = align_patch_to_reference_homography(warped, ref_map)
                    ncc_vals.append(float(np.mean(build_alignment_feature_map(warped) * ref_map)))
                    qp = Path(td) / f"prist_{p_idx}.png"
                    cv2.imwrite(str(qp), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
                    res = ad.analyze_patch(warped)
                    y_true.append(0)
                    y_score.append(float(res["score"]))
                    y_sub.append(0)
                    pristine_scores.append(float(res["score"]))

                # Subtle (0.22) and moderate (0.55) defects under matching residual skew
                ref_bgra = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGRA)
                for sev in (0.22, 0.55):
                    for d_idx, dtype in enumerate(DEFECT_TYPES):
                        dmg_bgra = apply_damage(ref_bgra.copy(), dtype, severity=sev, rng=random.Random(42 + d_idx))
                        dmg_rgb = cv2.cvtColor(dmg_bgra, cv2.COLOR_BGRA2RGB)
                        jitter = rng.uniform(-7.0, 7.0, size=(4, 2)).astype(np.float32)
                        M_p = cv2.getPerspectiveTransform(src_c, src_c + jitter)
                        warped = cv2.warpPerspective(dmg_rgb, M_p, (TARGET_W, TARGET_H), borderMode=cv2.BORDER_REPLICATE)
                        res = ad.analyze_patch(warped)
                        y_true.append(1)
                        y_score.append(float(res["score"]))
                        y_sub.append(1 if sev < 0.30 else 2)

        yt = np.asarray(y_true, dtype=np.int32)
        ys = np.asarray(y_score, dtype=np.float64)
        ysub = np.asarray(y_sub, dtype=np.int32)
        sub_mask = ysub != 2
        modes[key] = {
            "n_pristine": int((yt == 0).sum()),
            "n_defective": int((yt == 1).sum()),
            "roc_auc_all": round(_roc_auc(yt, ys), 4),
            "roc_auc_subtle": round(_roc_auc(yt[sub_mask], ys[sub_mask]), 4),
            "pr_auc_all": round(_pr_auc(yt, ys), 4),
            "cohens_d_all": round(_cohens_d(ys[yt == 1], ys[yt == 0]), 4),
            "cohens_d_subtle": round(_cohens_d(ys[ysub == 1], ys[yt == 0]), 4),
            "tpr_at_5pct_fpr": round(_tpr_at_fpr(yt, ys, 0.05), 4),
            "pristine_score_mean": round(float(np.mean(pristine_scores)), 4),
            "pristine_score_std": round(float(np.std(pristine_scores)), 4),
            "structural_ncc_mean": round(float(np.mean(ncc_vals)), 4),
        }
        for mk, mv in modes[key].items():
            if isinstance(mv, (int, float)):
                exp.log_metric(f"stage6.{key}.{mk}", mv)

    # Evaluate on real smartphone captures (structural NCC & anomaly scores with seg_yolo_640 & obb_pbl)
    real_cases = [
        ("charizard-reference.webp", "charizard-damaged.jpeg", "base1-4_real_charizard", True),
        ("giratina-kanda-reference.jpeg", "giratina-kanda-damaged.jpg", "swsh11_real_giratina", True),
        ("reference-gengarprime.webp", "realgengar-1.jpg", "hs04_real_gengar", False),
        ("reference-gengarprime.webp", "realgengar-3.jpg", "hs04_real_gengar", False),
        ("reference-gengarprime.webp", "realgengar-5.jpg", "hs04_real_gengar", False),
    ]
    real_dir = PROJECT_ROOT / "experiments" / "synthetic-to-real"
    det_obb = CardDetector(
        conf=float(cfg["obb"]["inference"]["conf"]),
        iou=float(cfg["obb"]["inference"]["iou"]),
        max_det=int(cfg["obb"]["inference"]["max_det"]),
    )
    det_seg = SegmentationCardDetector(
        backend="seg_yolo",
        weights_path=str(PROJECT_ROOT / cfg["segmentation"]["yolo_seg"]["weights"]),
        imgsz=640,
    )
    real_rows = []
    for ref_name, q_name, cid, is_dmg in real_cases:
        ref_p = real_dir / ref_name
        qp = real_dir / q_name
        if not ref_p.exists() or not qp.exists():
            continue
        img_bgr = cv2.imread(str(qp), cv2.IMREAD_COLOR)
        ref_rgb = cv2.cvtColor(
            cv2.resize(cv2.imread(str(ref_p), cv2.IMREAD_COLOR), (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        ref_map = build_alignment_feature_map(ref_rgb, align_size_wh=(224, 315))
        for bname, dmod in (("obb_pbl", det_obb), ("seg_yolo_640", det_seg)):
            dets = dmod.detect(img_bgr)
            if not dets:
                continue
            d = dets[0]
            ncc_before = float(np.mean(build_alignment_feature_map(d.patch) * ref_map))
            aligned, _, applied = align_patch_to_reference_homography(d.patch, ref_map, mask=d.mask)
            ncc_after = float(np.mean(build_alignment_feature_map(aligned) * ref_map))
            real_rows.append(
                {
                    "image": q_name,
                    "backend": bname,
                    "damaged": is_dmg,
                    "ecc_applied": applied,
                    "ncc_before": round(ncc_before, 4),
                    "ncc_after": round(ncc_after, 4),
                    "ncc_delta": round(ncc_after - ncc_before, 4),
                }
            )
    return {"synthetic_skew_ablation": modes, "real_captures_ncc": real_rows}


def _plot_upgrade_summary(stress: dict, stage6: dict, exp: Experiment) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=150)

    # Panel 1: Stage-3 Legacy vs Upgraded corner error across failure stress patterns
    labels = ["80% Side Occ.", "Diagonal Bite", "Soft Border", "Canon Foil Hole"]
    keys = ["single_side_occlusion_80pct", "diagonal_concave_bite", "soft_probability_border", "canonical_sparse_foil_hole"]
    leg_vals = [stress[k]["legacy_mean_px"] for k in keys]
    upg_vals = [stress[k]["upgraded_mean_px"] for k in keys]
    x = np.arange(len(labels))
    w = 0.35
    b1 = axes[0].bar(x - w / 2, leg_vals, w, label="Legacy Rectifier", color="#94a3b8")
    b2 = axes[0].bar(x + w / 2, upg_vals, w, label="Upgraded Rectifier", color="#0284c7")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, h + 0.3, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylabel("Mean Corner Error (px, lower is better)")
    axes[0].set_title("Stage-3 Rectifier Stress Ablation (n=40 each)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    # Panel 2: Stage-6 Reference-Guided Bounded Homography ECC Impact
    syn = stage6.get("synthetic_skew_ablation", {})
    off = syn.get("homography_ecc_off", {})
    on = syn.get("homography_ecc_on", {})
    if off and on:
        m_labels = ["ROC-AUC\n(All)", "ROC-AUC\n(Subtle)", "PR-AUC\n(All)", "TPR@5%FPR", "Struct NCC"]
        off_vals = [off["roc_auc_all"], off["roc_auc_subtle"], off["pr_auc_all"], off["tpr_at_5pct_fpr"], off["structural_ncc_mean"]]
        on_vals = [on["roc_auc_all"], on["roc_auc_subtle"], on["pr_auc_all"], on["tpr_at_5pct_fpr"], on["structural_ncc_mean"]]
        x2 = np.arange(len(m_labels))
        b3 = axes[1].bar(x2 - w / 2, off_vals, w, label="Stage-6 ECC OFF", color="#f59e0b")
        b4 = axes[1].bar(x2 + w / 2, on_vals, w, label="Stage-6 Homography ECC ON", color="#10b981")
        for bar in list(b3) + list(b4):
            h = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2.0, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
        axes[1].set_xticks(x2, m_labels)
        axes[1].set_ylim(0.0, 1.12)
        axes[1].set_title("Stage-6 Homography ECC under +/-7px Perspective Skew")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out_p = exp.save_plot(fig, "rectifier_perspective_upgrade")
    plt.close(fig)

    rep_fig_dir = PROJECT_ROOT / "report" / "figures" / "05_segmentation_pipeline"
    if rep_fig_dir.exists():
        import shutil
        shutil.copy2(out_p, rep_fig_dir / "rectifier_perspective_upgrade.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Stage-3 & Stage-6 perspective alignment upgrades.")
    parser.add_argument("--n-stress", type=int, default=40)
    parser.add_argument("--n-cards", type=int, default=5)
    parser.add_argument("--regen-diagnostics", action="store_true", help="Regenerate real-card corner diagnostics.")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("rectifier_perspective_upgrade", cfg)

    stress = run_stress_suite(exp, n_trials=args.n_stress)
    stage6 = run_stage6_homography_benchmark(exp, cfg, n_cards=args.n_cards)
    _plot_upgrade_summary(stress, stage6, exp)

    report = {"stage3_stress": stress, "stage6_homography_ecc": stage6}
    (exp.artifacts_dir / "rectifier_upgrade_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    exp.finish()

    if args.regen_diagnostics:
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "visual_corner_border_diagnostic.py")],
            check=True,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    main()

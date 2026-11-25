"""Incremental ablation benchmark on feature/anomaly-detection.

Evaluates step-by-step improvements on top of dinov2_vitb14_reg @ 672 (L5,8,11)
against the baseline dinov2_vitb14 @ 448, measuring threshold-free feature quality
(ROC-AUC, PR-AUC, Cohen's d, Spearman rho, per-defect subtle AUC/SNR) suited for
downstream classifier heads, plus real smartphone capture heatmaps.
"""

from __future__ import annotations

import base64
import gc
import json
import logging
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.card_detector import CardDetector  # noqa: E402
from backend.damage import DEFECT_TYPES, apply_damage  # noqa: E402
from backend.detector import PokemonAnomalyDetector  # noqa: E402
from backend.experiment import Experiment, get_device, load_config, set_seed  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UpgradeConfig:
    """Configuration for incremental AnomalyDINO ablation."""

    key: str
    label: str
    short_label: str
    model_name: str
    img_size: int
    feature_layers: Optional[tuple[int, ...]] = None
    lab_color_matching: bool = False
    subpixel_alignment: bool = False
    erode_frac: Optional[float] = None
    feather_px: Optional[int] = None
    patch_variance_norm: bool = False


UPGRADE_CONFIGS = [
    UpgradeConfig(
        key="0_prima_vitb14_448",
        label="0. PRIMA: vitb14 @ 448",
        short_label="0. PRIMA\nvitb14@448",
        model_name="dinov2_vitb14",
        img_size=448,
    ),
    UpgradeConfig(
        key="1_reg_672_ml",
        label="1. Reg+ML: vitb14_reg @ 672 (L5,8,11)",
        short_label="1. Reg+ML\n672",
        model_name="dinov2_vitb14_reg",
        img_size=672,
        feature_layers=(5, 8, 11),
    ),
    UpgradeConfig(
        key="2_plus_lab_color",
        label="2. + LAB Color Match (L+a+b)",
        short_label="2. +LAB\nColor",
        model_name="dinov2_vitb14_reg",
        img_size=672,
        feature_layers=(5, 8, 11),
        lab_color_matching=True,
    ),
    UpgradeConfig(
        key="3_plus_ecc_align",
        label="3. + ECC Sub-Pixel Align",
        short_label="3. +ECC\nAlign",
        model_name="dinov2_vitb14_reg",
        img_size=672,
        feature_layers=(5, 8, 11),
        lab_color_matching=True,
        subpixel_alignment=True,
    ),
    UpgradeConfig(
        key="4_plus_edge_mask",
        label="4. + Edge-Preserving Mask",
        short_label="4. +Edge\nMask",
        model_name="dinov2_vitb14_reg",
        img_size=672,
        feature_layers=(5, 8, 11),
        lab_color_matching=True,
        subpixel_alignment=True,
        erode_frac=0.002,
        feather_px=1,
    ),
    UpgradeConfig(
        key="5_full_stack_672",
        label="5. + Patch Var Norm (Full 672)",
        short_label="5. +VarNorm\n(Full 672)",
        model_name="dinov2_vitb14_reg",
        img_size=672,
        feature_layers=(5, 8, 11),
        lab_color_matching=True,
        subpixel_alignment=True,
        erode_frac=0.002,
        feather_px=1,
        patch_variance_norm=True,
    ),
    UpgradeConfig(
        key="6_full_stack_518",
        label="6. Full Stack @ 518 (Native Max)",
        short_label="6. Full\nStack 518",
        model_name="dinov2_vitb14_reg",
        img_size=518,
        feature_layers=(5, 8, 11),
        lab_color_matching=True,
        subpixel_alignment=True,
        erode_frac=0.002,
        feather_px=1,
        patch_variance_norm=True,
    ),
]


def _apply_realistic_capture_jitter(img_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulate realistic smartphone capture variations (tilt, white-balance cast, noise, JPEG)."""
    h, w = img_bgr.shape[:2]
    rs = np.random.RandomState(rng.randint(0, 2**31 - 1))

    angle = rng.uniform(-0.9, 0.9)
    dx = rng.uniform(-2.0, 2.0)
    dy = rng.uniform(-2.0, 2.0)
    mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, rng.uniform(0.990, 1.008))
    mat[0, 2] += dx
    mat[1, 2] += dy
    warped = cv2.warpAffine(img_bgr, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Global white-balance / color temperature shift + exposure delta
    b_gain = rng.uniform(0.90, 1.10)
    g_gain = rng.uniform(0.94, 1.06)
    r_gain = rng.uniform(0.90, 1.10)
    gamma = rng.uniform(0.92, 1.08)
    f = warped.astype(np.float32) / 255.0
    f[:, :, 0] *= b_gain
    f[:, :, 1] *= g_gain
    f[:, :, 2] *= r_gain
    f = np.clip(f, 0.0, 1.0) ** gamma

    noise = rs.normal(0.0, 2.2 / 255.0, size=f.shape).astype(np.float32)
    noisy = np.clip((f + noise) * 255.0, 0, 255).astype(np.uint8)

    ok, enc = cv2.imencode(".jpg", noisy, [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(86, 95)])
    if ok:
        decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if decoded is not None:
            return decoded
    return noisy


def _prepare_synthetic_suite(out_dir: Path, n_cards: int = 6, seed: int = 42) -> list[dict]:
    """Generate and persist deterministic synthetic test images on disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pool_dir = PROJECT_ROOT / "data" / "datasets" / "cards" / "pokemon_official_tcg" / "test"
    candidates = sorted(pool_dir.glob("*.webp"))
    step = max(1, len(candidates) // max(1, n_cards))
    chosen = [candidates[i * step] for i in range(n_cards)]

    severities = [("subtle", 0.18), ("moderate", 0.45), ("heavy", 0.85)]

    suite: list[dict] = []
    for idx, src_path in enumerate(chosen):
        card_id = f"upg_{src_path.stem}"
        ref_bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if ref_bgr is None:
            continue
        bgra = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255

        ref_png = out_dir / f"{card_id}_ref.png"
        cv2.imwrite(str(ref_png), ref_bgr)

        queries = []
        for p_idx in range(4):
            pristine_bgr = _apply_realistic_capture_jitter(ref_bgr, rng)
            q_path = out_dir / f"{card_id}_pristine_{p_idx}.png"
            cv2.imwrite(str(q_path), pristine_bgr)
            queries.append({
                "path": str(q_path),
                "is_damaged": False,
                "defect_type": "pristine",
                "severity": 0.0,
                "tier": "pristine",
            })

        if idx < 4:
            for defect in DEFECT_TYPES:
                for tier_name, sev in severities:
                    dmg_bgra = apply_damage(bgra, defect, sev, rng)
                    dmg_bgr = _apply_realistic_capture_jitter(dmg_bgra[:, :, :3], rng)
                    q_path = out_dir / f"{card_id}_{defect}_{tier_name}.png"
                    cv2.imwrite(str(q_path), dmg_bgr)
                    queries.append({
                        "path": str(q_path),
                        "is_damaged": True,
                        "defect_type": defect,
                        "severity": sev,
                        "tier": tier_name,
                    })

        suite.append({
            "card_id": card_id,
            "ref_path": str(ref_png),
            "queries": queries,
        })

    return suite


def _prepare_real_captures(out_dir: Path) -> list[dict]:
    """Warp raw real smartphone test captures ONCE to (1260, 896) crops."""
    out_dir.mkdir(parents=True, exist_ok=True)
    real_dir = PROJECT_ROOT / "experiments" / "synthetic-to-real"
    raw_cases = [
        ("charizard_real_dmg", real_dir / "charizard-reference.webp", real_dir / "charizard-damaged.jpeg", "Charizard (Real Damaged)"),
        ("charizard_png_dmg", real_dir / "charizard-reference.webp", PROJECT_ROOT / "data" / "test_charizard_damaged.png", "Charizard (Data Test PNG)"),
        ("giratina_real_dmg", real_dir / "giratina-kanda-reference.jpeg", real_dir / "giratina-kanda-damaged.jpg", "Giratina Full-Art (Real Damaged)"),
        ("gengar_real_1", real_dir / "reference-gengarprime.webp", real_dir / "realgengar-1.jpg", "Gengar Prime (Photo 1)"),
        ("gengar_real_2", real_dir / "reference-gengarprime.webp", real_dir / "realgengar-2.jpg", "Gengar Prime (Photo 2)"),
    ]

    det = CardDetector(target_size=(1260, 896))
    cases = []
    for case_id, ref_p, query_p, title in raw_cases:
        if not ref_p.exists() or not query_p.exists():
            continue
        raw_bgr = cv2.imread(str(query_p), cv2.IMREAD_COLOR)
        if raw_bgr is None:
            continue
        warped_path = out_dir / f"{case_id}_warped.png"
        detected = det.detect(str(query_p))
        if detected:
            best = max(detected, key=lambda d: d.confidence)
            cv2.imwrite(str(warped_path), cv2.cvtColor(best.patch, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(warped_path), raw_bgr)
        cases.append({
            "case_id": case_id,
            "title": title,
            "ref_path": str(ref_p),
            "query_path": str(warped_path),
        })

    del det
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    return cases


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size between damaged (b) and pristine (a)."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled = math.sqrt(max(1e-12, ((len(a) - 1) * va + (len(b) - 1) * vb) / max(1, len(a) + len(b) - 2)))
    return float((np.mean(b) - np.mean(a)) / pooled)


def _evaluate_config(
    ucfg: UpgradeConfig,
    suite: list[dict],
    real_cases: list[dict],
    cache_root: Path,
    device: str,
) -> dict:
    """Evaluate a single UpgradeConfig on synthetic + real captures."""
    logger.info("=== Evaluating %s ===", ucfg.label)
    detector = PokemonAnomalyDetector(
        device=device,
        model_name=ucfg.model_name,
        img_size=ucfg.img_size,
        feature_cache_dir=str(cache_root),
        scale_with_resolution=True,
        feature_layers=ucfg.feature_layers,
        lab_color_matching=ucfg.lab_color_matching,
        subpixel_alignment=ucfg.subpixel_alignment,
        erode_frac=ucfg.erode_frac,
        feather_px=ucfg.feather_px,
        patch_variance_norm=ucfg.patch_variance_norm,
    )
    detector.load_model()

    synthetic_records = []
    query_times = []
    for card in suite:
        # Card id shared across configs cleanly because _cache_model_tag differentiates model/layers/mask/lab
        detector.load_reference_images([card["ref_path"]], card_id=card["card_id"])
        for q in card["queries"]:
            tq0 = time.perf_counter()
            res = detector.analyze_card(q["path"], use_few_shot=True)
            dt_ms = (time.perf_counter() - tq0) * 1000.0
            query_times.append(dt_ms)
            synthetic_records.append({
                "card_id": card["card_id"],
                "is_damaged": q["is_damaged"],
                "defect_type": q["defect_type"],
                "severity": q["severity"],
                "tier": q["tier"],
                "score": float(res["score"]),
                "peak_score": float(res["peak_score"]),
            })

    real_results = {}
    last_ref = None
    for rc in real_cases:
        if rc["ref_path"] != last_ref:
            detector.load_reference_images([rc["ref_path"]], card_id=f"real_{Path(rc['ref_path']).stem}")
            last_ref = rc["ref_path"]
        tq0 = time.perf_counter()
        res = detector.analyze_card(rc["query_path"], use_few_shot=True)
        dt_ms = (time.perf_counter() - tq0) * 1000.0
        hm_bytes = base64.b64decode(res["heatmap_b64"])
        hm_bgr = cv2.imdecode(np.frombuffer(hm_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        real_results[rc["case_id"]] = {
            "title": rc["title"],
            "score": float(res["score"]),
            "peak_score": float(res["peak_score"]),
            "latency_ms": round(dt_ms, 1),
            "_heatmap_rgb": cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB) if hm_bgr is not None else None,
        }

    prist = np.array([r["score"] for r in synthetic_records if not r["is_damaged"]], dtype=np.float64)
    dmg_all = np.array([r["score"] for r in synthetic_records if r["is_damaged"]], dtype=np.float64)
    dmg_sub = np.array([r["score"] for r in synthetic_records if r["tier"] == "subtle"], dtype=np.float64)

    y_all = np.array([1 if r["is_damaged"] else 0 for r in synthetic_records])
    s_all = np.array([r["score"] for r in synthetic_records])

    sub_records = [r for r in synthetic_records if (not r["is_damaged"]) or r["tier"] == "subtle"]
    y_sub = np.array([1 if r["is_damaged"] else 0 for r in sub_records])
    s_sub = np.array([r["score"] for r in sub_records])

    spearman_rho, _ = spearmanr(
        np.array([r["severity"] for r in synthetic_records], dtype=np.float64),
        s_all,
    )
    thresh_p95 = float(np.percentile(prist, 95))

    per_defect = {}
    for d in DEFECT_TYPES:
        d_sub = np.array([r["score"] for r in synthetic_records if r["defect_type"] == d and r["tier"] == "subtle"])
        y_d = np.concatenate([np.zeros(len(prist)), np.ones(len(d_sub))])
        s_d = np.concatenate([prist, d_sub])
        per_defect[d] = {
            "subtle_auc": round(float(roc_auc_score(y_d, s_d)), 4),
            "subtle_snr": round(float((np.mean(d_sub) - np.mean(prist)) / max(1e-6, np.std(prist, ddof=1))), 3),
        }

    summary = {
        "key": ucfg.key,
        "label": ucfg.label,
        "short_label": ucfg.short_label,
        "img_size": ucfg.img_size,
        "roc_auc_all": round(float(roc_auc_score(y_all, s_all)), 4),
        "pr_auc_all": round(float(average_precision_score(y_all, s_all)), 4),
        "roc_auc_subtle": round(float(roc_auc_score(y_sub, s_sub)), 4),
        "cohens_d_all": round(_cohens_d(prist, dmg_all), 3),
        "cohens_d_subtle": round(_cohens_d(prist, dmg_sub), 3),
        "spearman_rho": round(float(spearman_rho), 4),
        "tpr_subtle_at_p95": round(float(np.mean(dmg_sub > thresh_p95)), 4),
        "pristine_mean": round(float(np.mean(prist)), 4),
        "pristine_std": round(float(np.std(prist, ddof=1)), 4),
        "query_latency_mean_ms": round(float(np.mean(query_times)), 1),
        "per_defect": per_defect,
        "real_results": real_results,
    }

    del detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    return summary


def _plot_incremental_ablation(summaries: list[dict], out_paths: list[Path]) -> None:
    """Plot step-by-step feature quality progression across upgrades."""
    x = np.arange(len(summaries))
    short_labels = [s["short_label"] for s in summaries]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), dpi=180)

    w = 0.25
    b1 = axes[0].bar(x - w, [s["roc_auc_all"] for s in summaries], w, label="ROC-AUC (All)", color="#1D4ED8")
    b2 = axes[0].bar(x, [s["roc_auc_subtle"] for s in summaries], w, label="ROC-AUC (Subtle 0.18)", color="#0D9488")
    b3 = axes[0].bar(x + w, [s["tpr_subtle_at_p95"] for s in summaries], w, label="TPR@5%FPR (Subtle)", color="#D97706")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_labels, fontsize=7.5)
    axes[0].set_ylim(0.45, 1.08)
    axes[0].set_title("Threshold-Free Discrimination Progression")
    axes[0].legend(fontsize=7.5, loc="lower right")
    axes[0].grid(axis="y", alpha=0.35)
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.2f}", ha="center", va="bottom", fontsize=6.5)

    b_d = axes[1].bar(x - 0.17, [s["cohens_d_subtle"] for s in summaries], 0.34, label="Cohen's d (Subtle)", color="#7C3AED")
    ax1r = axes[1].twinx()
    b_r = ax1r.bar(x + 0.17, [s["spearman_rho"] for s in summaries], 0.34, label="Spearman rho", color="#059669", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_labels, fontsize=7.5)
    axes[1].set_ylabel("Cohen's d (Subtle)", color="#7C3AED")
    ax1r.set_ylabel("Spearman rho", color="#059669")
    ax1r.set_ylim(0.45, 1.02)
    axes[1].set_title("Effect Size & Severity Monotonicity")
    axes[1].grid(axis="y", alpha=0.35)
    for bar in b_d:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.04, f"{h:.2f}", ha="center", va="bottom", fontsize=6.8)
    for bar in b_r:
        h = bar.get_height()
        ax1r.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.2f}", ha="center", va="bottom", fontsize=6.8)

    lat = [s["query_latency_mean_ms"] for s in summaries]
    bars_l = axes[2].bar(x, lat, 0.52, color="#334155")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_labels, fontsize=7.5)
    axes[2].set_ylabel("Latency / Query (ms)")
    axes[2].set_title("Runtime Cost per Upgrade")
    axes[2].grid(axis="y", alpha=0.35)
    for bar in bars_l:
        h = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2, h + max(lat) * 0.02, f"{h:.0f} ms", ha="center", va="bottom", fontsize=7.2)

    fig.suptitle("PokéDINO Incremental Upgrades Ablation (Before vs Reg@672+ML + Pipeline Improvements)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p)
    plt.close(fig)


def _plot_per_defect_stack(summaries: list[dict], out_paths: list[Path]) -> None:
    """Heatmap of subtle ROC-AUC & SNR per defect type across upgrades."""
    mat_auc = np.array([[s["per_defect"][d]["subtle_auc"] for d in DEFECT_TYPES] for s in summaries], dtype=np.float32)
    mat_snr = np.array([[s["per_defect"][d]["subtle_snr"] for d in DEFECT_TYPES] for s in summaries], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), dpi=180)
    for ax, mat, title, fmt, vmin, vmax in (
        (axes[0], mat_auc, "Subtle Defect (sev=0.18) ROC-AUC by Defect Type", ".3f", 0.45, 1.0),
        (axes[1], mat_snr, "Subtle Defect SNR = (mean - pristine_mean) / pristine_std", ".2f", -0.5, max(3.5, float(mat_snr.max()))),
    ):
        im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(DEFECT_TYPES)))
        ax.set_xticklabels(DEFECT_TYPES, rotation=25, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(summaries)))
        ax.set_yticklabels([s["label"] for s in summaries], fontsize=8)
        ax.set_title(title, fontsize=9.5)
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                val = mat[r, c]
                norm_v = (val - vmin) / max(1e-6, vmax - vmin)
                ax.text(c, r, format(val, fmt), ha="center", va="center",
                        color="white" if norm_v > 0.62 else "black", fontsize=8, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.82)

    fig.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p)
    plt.close(fig)


def _plot_real_stack_heatmaps(summaries: list[dict], real_cases: list[dict], out_paths: list[Path]) -> None:
    """Visual heatmap progression on real captures: PRIMA -> Reg@672+ML -> Full Stack 672 -> Full Stack 518."""
    keys = ["0_prima_vitb14_448", "1_reg_672_ml", "5_full_stack_672", "6_full_stack_518"]
    cols_s = [s for s in summaries if s["key"] in keys]
    target_cases = [rc for rc in real_cases if rc["case_id"] in {"charizard_real_dmg", "giratina_real_dmg", "gengar_real_1"}]
    if not cols_s or not target_cases:
        return

    rows, cols = len(target_cases), len(cols_s)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.9 * rows), dpi=170)
    axes = np.atleast_2d(axes)
    for r, rc in enumerate(target_cases):
        for c, s in enumerate(cols_s):
            ax = axes[r, c]
            rr = s["real_results"].get(rc["case_id"], {})
            hm = rr.get("_heatmap_rgb")
            if hm is not None:
                ax.imshow(hm)
            ax.set_title(
                f"{s['short_label'].replace(chr(10), ' ')}\n{rc['title'].split('(')[0].strip()}: peak={rr.get('peak_score', 0.0):.3f}",
                fontsize=7.8,
            )
            ax.axis("off")

    fig.suptitle("Real Smartphone Captures: Before vs Reg@672+ML vs Full Upgraded Stack", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("anomaly_upgrades_ablation", cfg)
    device = get_device(cfg)

    bench_img_dir = exp.artifacts_dir / "bench_images"
    cache_root = exp.artifacts_dir / "bench_cache"
    fig_report_dir = PROJECT_ROOT / "report" / "figures" / "06_anomaly_detection"

    suite = _prepare_synthetic_suite(bench_img_dir / "synthetic", n_cards=6, seed=int(cfg["project"]["seed"]))
    real_cases = _prepare_real_captures(bench_img_dir / "real")

    summaries = []
    for ucfg in UPGRADE_CONFIGS:
        summary = _evaluate_config(ucfg, suite, real_cases, cache_root, device)
        summaries.append(summary)
        exp.log_metric(f"{ucfg.key}.roc_auc_all", summary["roc_auc_all"])
        exp.log_metric(f"{ucfg.key}.roc_auc_subtle", summary["roc_auc_subtle"])
        exp.log_metric(f"{ucfg.key}.cohens_d_subtle", summary["cohens_d_subtle"])
        exp.log_metric(f"{ucfg.key}.spearman_rho", summary["spearman_rho"])

    _plot_incremental_ablation(summaries, [
        exp.plots_dir / "05_incremental_improvements_ablation.png",
        fig_report_dir / "05_incremental_improvements_ablation.png",
    ])
    _plot_per_defect_stack(summaries, [
        exp.plots_dir / "06_per_defect_before_vs_after_stack.png",
        fig_report_dir / "06_per_defect_before_vs_after_stack.png",
    ])
    _plot_real_stack_heatmaps(summaries, real_cases, [
        exp.plots_dir / "07_real_captures_stack_comparison.png",
        fig_report_dir / "07_real_captures_stack_comparison.png",
    ])

    json_summaries = []
    for s in summaries:
        s_clean = {k: v for k, v in s.items() if k != "real_results"}
        s_clean["real_results"] = {
            cid: {rk: rv for rk, rv in rdata.items() if rk != "_heatmap_rgb"}
            for cid, rdata in s["real_results"].items()
        }
        json_summaries.append(s_clean)

    (exp.artifacts_dir / "upgrades_ablation_summary.json").write_text(json.dumps(json_summaries, indent=2))
    exp.finish()
    shutil.rmtree(exp.artifacts_dir / "bench_cache", ignore_errors=True)

    print("\n=== INCREMENTAL ANOMALYDINO UPGRADES SUMMARY ===")
    hdr = f"{'Configuration':<38} | {'ROC-All':>7} | {'ROC-Sub':>7} | {'d-Sub':>6} | {'TPR@5%':>6} | {'Rho':>6} | {'Lat(ms)':>7}"
    print(hdr)
    print("-" * len(hdr))
    for s in json_summaries:
        print(
            f"{s['label']:<38} | {s['roc_auc_all']:7.4f} | {s['roc_auc_subtle']:7.4f} | "
            f"{s['cohens_d_subtle']:6.3f} | {s['tpr_subtle_at_p95']:6.3f} | {s['spearman_rho']:6.4f} | {s['query_latency_mean_ms']:7.1f}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    main()

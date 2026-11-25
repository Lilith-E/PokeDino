"""Before-vs-After ablation benchmark for DINOv2 register tokens and input resolution.

Evaluates standard ``dinov2_vitb14 @ 448`` (baseline) against ``dinov2_vitb14_reg``
across ``448``, ``672``, ``672 (multi-layer)``, and ``896`` px on:
1. A fixed, reproducible multi-card synthetic defect suite with realistic camera/optical
   jitter across all 7 physical defect types and 3 severity levels.
2. Real smartphone captures from ``experiments/synthetic-to-real/`` and
   ``data/test_charizard_damaged.png``.

Outputs quantitative tables (JSON/CSV), calibrated PR/ROC/Cohen's d separation
metrics, latency measurements, and before-vs-after visual heatmap comparisons.
"""

from __future__ import annotations

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
class EvalConfig:
    """Single anomaly-detector configuration for comparative benchmarking."""

    key: str
    label: str
    model_name: str
    img_size: int
    feature_layers: Optional[tuple[int, ...]] = None
    scale_with_resolution: bool = True


EVAL_CONFIGS = [
    EvalConfig(
        key="prima_vitb14_448",
        label="PRIMA: vitb14 @ 448",
        model_name="dinov2_vitb14",
        img_size=448,
    ),
    EvalConfig(
        key="reg_vitb14_448",
        label="Reg: vitb14_reg @ 448",
        model_name="dinov2_vitb14_reg",
        img_size=448,
    ),
    EvalConfig(
        key="reg_vitb14_448_ml",
        label="Reg+ML: vitb14_reg @ 448 (L5,8,11)",
        model_name="dinov2_vitb14_reg",
        img_size=448,
        feature_layers=(5, 8, 11),
    ),
    EvalConfig(
        key="reg_vitb14_672",
        label="Reg: vitb14_reg @ 672",
        model_name="dinov2_vitb14_reg",
        img_size=672,
    ),
    EvalConfig(
        key="reg_vitb14_672_ml",
        label="Reg+ML: vitb14_reg @ 672 (L5,8,11)",
        model_name="dinov2_vitb14_reg",
        img_size=672,
        feature_layers=(5, 8, 11),
    ),
    EvalConfig(
        key="reg_vitb14_896",
        label="Reg: vitb14_reg @ 896",
        model_name="dinov2_vitb14_reg",
        img_size=896,
    ),
]


def _apply_realistic_capture_jitter(img_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulate mild smartphone acquisition jitter (exposure, tilt, sensor noise, JPEG)."""
    h, w = img_bgr.shape[:2]
    rs = np.random.RandomState(rng.randint(0, 2**31 - 1))

    # Mild affine jitter (+/- 0.7 deg, +/- 1.5 px) matching slight hand-held imperfections
    angle = rng.uniform(-0.7, 0.7)
    dx = rng.uniform(-1.5, 1.5)
    dy = rng.uniform(-1.5, 1.5)
    mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, rng.uniform(0.992, 1.005))
    mat[0, 2] += dx
    mat[1, 2] += dy
    warped = cv2.warpAffine(img_bgr, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Mild global brightness/gamma variation
    gamma = rng.uniform(0.93, 1.07)
    gain = rng.uniform(0.94, 1.06)
    f = np.clip((warped.astype(np.float32) / 255.0) ** gamma * gain, 0.0, 1.0)

    # Mild sensor noise
    noise = rs.normal(0.0, 2.2 / 255.0, size=f.shape).astype(np.float32)
    noisy = np.clip((f + noise) * 255.0, 0, 255).astype(np.uint8)

    # Mild JPEG re-encoding
    ok, enc = cv2.imencode(".jpg", noisy, [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(88, 96)])
    if ok:
        decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if decoded is not None:
            return decoded
    return noisy


def _prepare_synthetic_suite(out_dir: Path, n_cards: int = 6, seed: int = 42) -> list[dict]:
    """Generate and persist deterministic pristine + damaged test images on disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pool_dir = PROJECT_ROOT / "data" / "datasets" / "cards" / "pokemon_official_tcg" / "test"
    candidates = sorted(pool_dir.glob("*.webp"))
    if len(candidates) < n_cards:
        pool_dir = PROJECT_ROOT / "data" / "datasets" / "cards" / "pokemon_pocket_tcgp"
        candidates = sorted(pool_dir.glob("*.webp"))

    # Sample evenly across sorted sets for visual diversity
    step = max(1, len(candidates) // max(1, n_cards))
    chosen = [candidates[i * step] for i in range(n_cards)]

    severities = [
        ("subtle", 0.18),
        ("moderate", 0.45),
        ("heavy", 0.85),
    ]

    suite: list[dict] = []
    for idx, src_path in enumerate(chosen):
        card_id = f"bench_{src_path.stem}"
        ref_bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if ref_bgr is None:
            continue
        h, w = ref_bgr.shape[:2]
        bgra = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255

        ref_png = out_dir / f"{card_id}_ref.png"
        cv2.imwrite(str(ref_png), ref_bgr)

        queries = []
        # 4 realistic pristine smartphone variations per card
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

        # Evaluate all 7 defect types across the suite cleanly: every card gets all 7 at one severity cycle,
        # plus we cover every (defect, severity) pair cleanly across cards (3 cards per defect-severity pair = 21 per card?
        # Keep fast & balanced: every card evaluates subtle(0.18) on all 7 defects or 4 defects x 3 severities = 12?
        # Even better: 4 cards * 7 defects * 3 severities / 2 -> assign all 7 defects on cards idx%2, giving 3 full repeats
        # per (defect, severity)! Across 6 cards: 3 full cards per defect x 3 severities = 63 damaged + 24 pristine = 87!)
        assigned_defects = DEFECT_TYPES if idx % 2 == 0 else DEFECT_TYPES  # Keep all 7 on first 4 cards (<=100 queries)
        if idx < 4:
            for defect in assigned_defects:
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
            "src_shape": (h, w),
        })

    return suite


def _prepare_real_captures(out_dir: Path, cfg: dict) -> list[dict]:
    """Warp raw smartphone test photos ONCE to high-resolution canonical crops (1260x896)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    real_dir = PROJECT_ROOT / "experiments" / "synthetic-to-real"

    raw_cases = [
        ("charizard_real_dmg", real_dir / "charizard-reference.webp", real_dir / "charizard-damaged.jpeg", "Charizard (Real Damaged)"),
        ("charizard_png_dmg", real_dir / "charizard-reference.webp", PROJECT_ROOT / "data" / "test_charizard_damaged.png", "Charizard (Data Test Damaged)"),
        ("giratina_real_dmg", real_dir / "giratina-kanda-reference.jpeg", real_dir / "giratina-kanda-damaged.jpg", "Giratina Full-Art (Real Damaged)"),
        ("gengar_real_1", real_dir / "reference-gengarprime.webp", real_dir / "realgengar-1.jpg", "Gengar Prime (Real Photo 1)"),
        ("gengar_real_2", real_dir / "reference-gengarprime.webp", real_dir / "realgengar-2.jpg", "Gengar Prime (Real Photo 2)"),
        ("gengar_real_3", real_dir / "reference-gengarprime.webp", real_dir / "realgengar-3.jpg", "Gengar Prime (Real Photo 3)"),
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
            # Warp highest-confidence detection directly to (1260, 896) RGB -> save BGR PNG
            best = max(detected, key=lambda d: d.confidence)
            patch_bgr = cv2.cvtColor(best.patch, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(warped_path), patch_bgr)
            used_yolo = True
        else:
            cv2.imwrite(str(warped_path), raw_bgr)
            used_yolo = False

        cases.append({
            "case_id": case_id,
            "title": title,
            "ref_path": str(ref_p),
            "query_path": str(warped_path),
            "used_yolo_warp": used_yolo,
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
    ecfg: EvalConfig,
    suite: list[dict],
    real_cases: list[dict],
    cache_root: Path,
    device: str,
) -> dict:
    """Run both synthetic and real benchmarks for a single EvalConfig."""
    logger.info("=== Evaluating %s (%s @ %d) ===", ecfg.key, ecfg.model_name, ecfg.img_size)
    cfg_cache = cache_root / ecfg.key
    cfg_cache.mkdir(parents=True, exist_ok=True)

    detector = PokemonAnomalyDetector(
        device=device,
        model_name=ecfg.model_name,
        img_size=ecfg.img_size,
        feature_cache_dir=str(cfg_cache),
        scale_with_resolution=ecfg.scale_with_resolution,
        feature_layers=ecfg.feature_layers,
    )
    t_load0 = time.perf_counter()
    detector.load_model()
    model_load_s = time.perf_counter() - t_load0

    synthetic_records = []
    ref_times = []
    query_times = []

    for card in suite:
        t0 = time.perf_counter()
        detector.load_reference_images([card["ref_path"]], card_id=f"{ecfg.key}_{card['card_id']}")
        ref_times.append(time.perf_counter() - t0)

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
                "defect_area_pct": float(res["defect_area_pct"]),
                "morphology_bonus": float(res["morphology"]["morphology_bonus"]),
                "cluster_count": int(res["morphology"]["cluster_count"]),
                "linear_cluster_count": int(res["morphology"]["linear_cluster_count"]),
                "latency_ms": round(dt_ms, 2),
            })

    # Real smartphone captures + save RGB heatmap images for before-vs-after inspection
    real_results = {}
    last_ref = None
    for rc in real_cases:
        if rc["ref_path"] != last_ref:
            detector.load_reference_images([rc["ref_path"]], card_id=f"{ecfg.key}_real_{Path(rc['ref_path']).stem}")
            last_ref = rc["ref_path"]
        tq0 = time.perf_counter()
        res = detector.analyze_card(rc["query_path"], use_few_shot=True)
        dt_ms = (time.perf_counter() - tq0) * 1000.0

        # Decode base64 PNG heatmap overlay for visual report
        import base64
        hm_bytes = base64.b64decode(res["heatmap_b64"])
        hm_arr = np.frombuffer(hm_bytes, dtype=np.uint8)
        hm_bgr = cv2.imdecode(hm_arr, cv2.IMREAD_COLOR)

        real_results[rc["case_id"]] = {
            "title": rc["title"],
            "score": float(res["score"]),
            "peak_score": float(res["peak_score"]),
            "defect_area_pct": float(res["defect_area_pct"]),
            "morphology_bonus": float(res["morphology"]["morphology_bonus"]),
            "cluster_count": int(res["morphology"]["cluster_count"]),
            "linear_cluster_count": int(res["morphology"]["linear_cluster_count"]),
            "condition": res["condition"]["label"],
            "verdict": res["verdict"],
            "latency_ms": round(dt_ms, 2),
            "grid_size": res["grid_size"],
            "_heatmap_rgb": cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB) if hm_bgr is not None else None,
        }

    # Aggregate synthetic metrics
    prist = np.array([r["score"] for r in synthetic_records if not r["is_damaged"]], dtype=np.float64)
    dmg_all = np.array([r["score"] for r in synthetic_records if r["is_damaged"]], dtype=np.float64)
    dmg_sub = np.array([r["score"] for r in synthetic_records if r["tier"] == "subtle"], dtype=np.float64)
    dmg_mod = np.array([r["score"] for r in synthetic_records if r["tier"] == "moderate"], dtype=np.float64)
    dmg_hvy = np.array([r["score"] for r in synthetic_records if r["tier"] == "heavy"], dtype=np.float64)

    y_all = np.array([1 if r["is_damaged"] else 0 for r in synthetic_records])
    s_all = np.array([r["score"] for r in synthetic_records])

    sub_records = [r for r in synthetic_records if (not r["is_damaged"]) or r["tier"] == "subtle"]
    y_sub = np.array([1 if r["is_damaged"] else 0 for r in sub_records])
    s_sub = np.array([r["score"] for r in sub_records])

    all_sevs = np.array([r["severity"] for r in synthetic_records], dtype=np.float64)
    spearman_rho, _ = spearmanr(all_sevs, s_all)

    thresh_p95 = float(np.percentile(prist, 95))
    tpr_all_p95 = float(np.mean(dmg_all > thresh_p95))
    tpr_sub_p95 = float(np.mean(dmg_sub > thresh_p95))

    per_defect = {}
    for d in DEFECT_TYPES:
        d_sub = np.array([r["score"] for r in synthetic_records if r["defect_type"] == d and r["tier"] == "subtle"])
        d_all = np.array([r["score"] for r in synthetic_records if r["defect_type"] == d])
        y_d = np.concatenate([np.zeros(len(prist)), np.ones(len(d_sub))])
        s_d = np.concatenate([prist, d_sub])
        per_defect[d] = {
            "subtle_mean": round(float(np.mean(d_sub)), 4),
            "all_mean": round(float(np.mean(d_all)), 4),
            "subtle_auc": round(float(roc_auc_score(y_d, s_d)), 4),
            "subtle_snr": round(float((np.mean(d_sub) - np.mean(prist)) / max(1e-6, np.std(prist, ddof=1))), 3),
        }

    summary = {
        "key": ecfg.key,
        "label": ecfg.label,
        "model_name": ecfg.model_name,
        "img_size": ecfg.img_size,
        "processing_size": list(detector._processing_size),
        "feature_layers": list(ecfg.feature_layers) if ecfg.feature_layers else None,
        "effective_radius_patches": detector.spatial_config.radius_patches,
        "effective_cluster_thresh": detector.morphology_config.large_cluster_threshold,
        "model_load_s": round(model_load_s, 2),
        "ref_load_mean_s": round(float(np.mean(ref_times)), 2),
        "query_latency_mean_ms": round(float(np.mean(query_times)), 1),
        "n_pristine": int(len(prist)),
        "n_damaged": int(len(dmg_all)),
        "pristine_mean": round(float(np.mean(prist)), 4),
        "pristine_std": round(float(np.std(prist, ddof=1)), 4),
        "pristine_p95": round(thresh_p95, 4),
        "subtle_mean": round(float(np.mean(dmg_sub)), 4),
        "moderate_mean": round(float(np.mean(dmg_mod)), 4),
        "heavy_mean": round(float(np.mean(dmg_hvy)), 4),
        "damaged_mean": round(float(np.mean(dmg_all)), 4),
        "roc_auc_all": round(float(roc_auc_score(y_all, s_all)), 4),
        "pr_auc_all": round(float(average_precision_score(y_all, s_all)), 4),
        "roc_auc_subtle": round(float(roc_auc_score(y_sub, s_sub)), 4),
        "cohens_d_all": round(_cohens_d(prist, dmg_all), 3),
        "cohens_d_subtle": round(_cohens_d(prist, dmg_sub), 3),
        "spearman_rho": round(float(spearman_rho), 4),
        "tpr_all_at_p95": round(tpr_all_p95, 4),
        "tpr_subtle_at_p95": round(tpr_sub_p95, 4),
        "per_defect": per_defect,
        "records": synthetic_records,
        "real_results": real_results,
    }

    del detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    return summary


def _plot_ablation_overview(summaries: list[dict], out_paths: list[Path]) -> None:
    """Create grouped summary chart comparing discrimination, SNR, and latency."""
    labels = [s["label"] for s in summaries]
    x = np.arange(len(labels))

    roc_all = [s["roc_auc_all"] for s in summaries]
    roc_sub = [s["roc_auc_subtle"] for s in summaries]
    tpr_sub = [s["tpr_subtle_at_p95"] for s in summaries]
    cohen_sub = [s["cohens_d_subtle"] for s in summaries]
    lat_ms = [s["query_latency_mean_ms"] for s in summaries]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=180)

    # Panel 1: ROC-AUC & Subtle Recall @ 5% FPR
    w = 0.25
    b1 = axes[0].bar(x - w, roc_all, w, label="ROC-AUC (All)", color="#1D4ED8")
    b2 = axes[0].bar(x, roc_sub, w, label="ROC-AUC (Subtle 0.18)", color="#0D9488")
    b3 = axes[0].bar(x + w, tpr_sub, w, label="TPR@5%FPR (Subtle)", color="#D97706")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s["key"].replace("_", "\n") for s in summaries], fontsize=7.5)
    axes[0].set_ylim(0.4, 1.08)
    axes[0].set_title("Discrimination Quality (Higher = Better)")
    axes[0].legend(fontsize=7.5, loc="lower right")
    axes[0].grid(axis="y", alpha=0.35)
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.2f}", ha="center", va="bottom", fontsize=6.5)

    # Panel 2: Cohen's d on Subtle Defects + Spearman Monotonicity
    bars_d = axes[1].bar(x - 0.16, cohen_sub, 0.32, label="Cohen's d (Subtle)", color="#7C3AED")
    ax1_r = axes[1].twinx()
    bars_r = ax1_r.bar(x + 0.16, [s["spearman_rho"] for s in summaries], 0.32, label="Spearman rho", color="#059669", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s["key"].replace("_", "\n") for s in summaries], fontsize=7.5)
    axes[1].set_ylabel("Cohen's d (Subtle vs Pristine)", color="#7C3AED")
    ax1_r.set_ylabel("Spearman rho (Severity)", color="#059669")
    ax1_r.set_ylim(0.5, 1.05)
    axes[1].set_title("Signal Separation & Severity Monotonicity")
    axes[1].grid(axis="y", alpha=0.35)
    for bar in bars_d:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.04, f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    for bar in bars_r:
        h = bar.get_height()
        ax1_r.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    # Panel 3: Query Latency (ms)
    bars_t = axes[2].bar(x, lat_ms, 0.5, color=["#64748B", "#2563EB", "#0D9488", "#7C3AED", "#DC2626"][:len(summaries)])
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([s["key"].replace("_", "\n") for s in summaries], fontsize=7.5)
    axes[2].set_ylabel("Latency / Query (ms)")
    axes[2].set_title("Inference Speed vs Resolution")
    axes[2].grid(axis="y", alpha=0.35)
    for bar in bars_t:
        h = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2, h + max(lat_ms) * 0.02, f"{h:.0f} ms", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle("PokéDINO AnomalyDINO Ablation: Register Tokens (reg) + Input Resolution", fontsize=12, fontweight="bold")
    fig.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p)
    plt.close(fig)


def _plot_score_distributions(summaries: list[dict], out_paths: list[Path]) -> None:
    """Compare score distributions across Pristine, Subtle, Moderate, and Heavy per config."""
    n = len(summaries)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 4.3), dpi=180, sharey=False)
    if n == 1:
        axes = [axes]

    tier_order = ["pristine", "subtle", "moderate", "heavy"]
    tier_colors = {"pristine": "#22C55E", "subtle": "#3B82F6", "moderate": "#F59E0B", "heavy": "#EF4444"}

    for ax, s in zip(axes, summaries):
        data = [[r["score"] for r in s["records"] if r["tier"] == t] for t in tier_order]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True)
        for patch, t in zip(bp["boxes"], tier_order):
            patch.set_facecolor(tier_colors[t])
            patch.set_alpha(0.65)
        ax.axhline(s["pristine_p95"], color="#111827", linestyle="--", linewidth=1.2, label=f"p95={s['pristine_p95']:.3f}")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["Prist", "Subtle\n0.18", "Mod\n0.45", "Heavy\n0.85"], fontsize=7.5)
        ax.set_title(f"{s['label']}\nAUC_sub={s['roc_auc_subtle']:.3f}", fontsize=8.5)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("Anomaly Score")
    fig.suptitle("Score Calibration Distributions Across Severity Tiers (Dashed = Calibrated 95th % Pristine)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p)
    plt.close(fig)


def _plot_per_defect_heatmap(summaries: list[dict], out_paths: list[Path]) -> None:
    """Visualize Subtle Defect ROC-AUC and SNR by physical defect category."""
    mat_auc = np.array([[s["per_defect"][d]["subtle_auc"] for d in DEFECT_TYPES] for s in summaries], dtype=np.float32)
    mat_snr = np.array([[s["per_defect"][d]["subtle_snr"] for d in DEFECT_TYPES] for s in summaries], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), dpi=180)
    for ax, mat, title, fmt, vmin, vmax in (
        (axes[0], mat_auc, "Subtle Defect (sev=0.18) ROC-AUC by Defect Type", ".3f", 0.5, 1.0),
        (axes[1], mat_snr, "Subtle Defect SNR = (mean - pristine_mean) / pristine_std", ".2f", 0.0, max(3.5, float(mat_snr.max()))),
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


def _plot_real_heatmaps_comparison(summaries: list[dict], real_cases: list[dict], out_paths: list[Path]) -> None:
    """Compare heatmaps on real damaged smartphone captures: PRIMA vs Reg + High-Res."""
    chosen_keys = ["prima_vitb14_448", "reg_vitb14_448", "reg_vitb14_672", "reg_vitb14_896"]
    cols_s = [s for s in summaries if s["key"] in chosen_keys]
    target_cases = [rc for rc in real_cases if rc["case_id"] in {"charizard_real_dmg", "charizard_png_dmg", "giratina_real_dmg", "gengar_real_1"}]
    if not cols_s or not target_cases:
        return

    rows, cols = len(target_cases), len(cols_s)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.8 * rows), dpi=170)
    axes = np.atleast_2d(axes)

    for r, rc in enumerate(target_cases):
        for c, s in enumerate(cols_s):
            ax = axes[r, c]
            rr = s["real_results"].get(rc["case_id"], {})
            hm = rr.get("_heatmap_rgb")
            if hm is not None:
                ax.imshow(hm)
            title = (
                f"{s['label']}\n"
                f"{rc['title'].split('(')[0].strip()}: {rr.get('score', 0.0):.3f} | area={rr.get('defect_area_pct', 0.0):.1f}% | cl={rr.get('cluster_count', 0)}"
            )
            ax.set_title(title, fontsize=7.5)
            ax.axis("off")

    fig.suptitle("Real Card Heatmaps: Before (dinov2_vitb14 @ 448) vs After (dinov2_vitb14_reg @ 448/672/896)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("anomaly_reg_res_ablation", cfg)
    device = get_device(cfg)

    bench_img_dir = exp.artifacts_dir / "bench_images"
    cache_root = exp.artifacts_dir / "bench_cache"
    fig_report_dir = PROJECT_ROOT / "report" / "figures" / "06_anomaly_detection"

    logger.info("Generating fixed deterministic benchmark suite...")
    suite = _prepare_synthetic_suite(bench_img_dir / "synthetic", n_cards=6, seed=int(cfg["project"]["seed"]))
    real_cases = _prepare_real_captures(bench_img_dir / "real", cfg)

    summaries = []
    for ecfg in EVAL_CONFIGS:
        summary = _evaluate_config(ecfg, suite, real_cases, cache_root, device)
        summaries.append(summary)
        exp.log_metric(f"{ecfg.key}.roc_auc_all", summary["roc_auc_all"])
        exp.log_metric(f"{ecfg.key}.roc_auc_subtle", summary["roc_auc_subtle"])
        exp.log_metric(f"{ecfg.key}.cohens_d_subtle", summary["cohens_d_subtle"])
        exp.log_metric(f"{ecfg.key}.tpr_subtle_at_p95", summary["tpr_subtle_at_p95"])
        exp.log_metric(f"{ecfg.key}.query_latency_mean_ms", summary["query_latency_mean_ms"])

    _plot_ablation_overview(summaries, [
        exp.plots_dir / "01_ablation_metrics_comparison.png",
        fig_report_dir / "01_ablation_metrics_comparison.png",
    ])
    _plot_score_distributions(summaries, [
        exp.plots_dir / "02_score_distributions_before_after.png",
        fig_report_dir / "02_score_distributions_before_after.png",
    ])
    _plot_per_defect_heatmap(summaries, [
        exp.plots_dir / "03_per_defect_sensitivity.png",
        fig_report_dir / "03_per_defect_sensitivity.png",
    ])
    _plot_real_heatmaps_comparison(summaries, real_cases, [
        exp.plots_dir / "04_real_captures_heatmaps_before_after.png",
        fig_report_dir / "04_real_captures_heatmaps_before_after.png",
    ])

    # Strip non-serializable numpy overlays before saving JSON
    json_summaries = []
    for s in summaries:
        s_clean = {k: v for k, v in s.items() if k not in ("records", "real_results")}
        s_clean["real_results"] = {
            cid: {rk: rv for rk, rv in rdata.items() if rk != "_heatmap_rgb"}
            for cid, rdata in s["real_results"].items()
        }
        json_summaries.append(s_clean)

    summary_json = exp.artifacts_dir / "ablation_summary.json"
    summary_json.write_text(json.dumps(json_summaries, indent=2))
    exp.finish()

    print("\n=== ANOMALYDINO BEFORE vs AFTER SUMMARY ===")
    header = f"{'Config':<34} | {'ROC-All':>7} | {'ROC-Sub':>7} | {'d-Sub':>6} | {'TPR@5%':>6} | {'Rho':>6} | {'Lat(ms)':>7}"
    print(header)
    print("-" * len(header))
    for s in json_summaries:
        print(
            f"{s['label']:<34} | {s['roc_auc_all']:7.4f} | {s['roc_auc_subtle']:7.4f} | "
            f"{s['cohens_d_subtle']:6.3f} | {s['tpr_subtle_at_p95']:6.3f} | {s['spearman_rho']:6.4f} | {s['query_latency_mean_ms']:7.1f}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    main()

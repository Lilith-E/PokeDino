"""Validation of anomaly pipeline via procedural defect synthesis.

Genera campioni sintetici applicando difetti calibrati (usura bordi, graffi, pieghe,
ammaccature, macchie, fori, umidità) a diversi livelli di severità (LP, MP, HP, DMG).
Verifica la monotonicità dello score di anomalia al variare del danno, misura la
capacità di discriminazione rispetto alla soglia e calcola la matrice di confusione.
"""

import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.experiment import Experiment, load_config, set_seed  # noqa: E402
from backend.detector import PokemonAnomalyDetector  # noqa: E402
from backend.card_detector import CardDetector  # noqa: E402
from backend.condition import classify_condition, COLORS  # noqa: E402
from backend.damage import DEFECT_TYPES, LEVELS, synthesize_damaged  # noqa: E402

logger = logging.getLogger(__name__)

TRUE_BY_TAG = {"NM": "NM", "LP": "GE", "MP": "GE", "HP": "PP", "DMG": "PP"}
PRED_TO_TAG = {"Near Mint/Mint": "NM", "Good/Excellent": "GE", "Poor/Played": "PP"}

def main(n_cards: int = 10):
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("phase2d_damage_validation", cfg)

    uploads_dir = PROJECT_ROOT / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    cards_dir = PROJECT_ROOT / "data" / "datasets" / "cards" / "pokemon_pocket_tcgp"
    bg_dir = PROJECT_ROOT / "data" / "datasets" / "backgrounds"
    card_files = sorted(cards_dir.glob("*.webp"))
    bg_files = sorted(bg_dir.rglob("*.jpg"))
    if not card_files or not bg_files:
        exp.logger.error("Missing cards/backgrounds. Run prepare_dataset.py first.")
        sys.exit(1)

    det = PokemonAnomalyDetector(masking_enabled=True, clahe_enabled=True)
    weights = CardDetector._find_latest_weights()
    if weights is None:
        exp.logger.error("No trained OBB weights found. Run scripts/train_obb.py first.")
        sys.exit(1)
    cd = CardDetector(weights_path=weights)
    threshold = det.anomaly_threshold
    bands = det.condition_bands
    exp.logger.info(f"threshold={threshold:.4f}  bands={ {k: (round(v,3) if v else None) for k, v in bands.items()} }")

    from scripts.prepare_dataset import composite_card, load_card_rgba
    from backend.quality_gate import ImageQualityGate

    rng = random.Random(cfg["project"]["seed"] + 7)
    fallbacks = [0]
    quality_gate = ImageQualityGate()

    def photograph(card_bgra, i, tag):
        """Composite on a background + OBB detect/warp → canonical RGB patch.

        Uses the production detection path (edge-refined warp + quality gate)
        so scores are comparable with the Phase 2b threshold and with
        end-to-end inference. On detector miss / low quality (3 tries) falls
        back to the GT-corner warp.
        """
        scene = corners = None
        for t in range(3):
            bg = cv2.imread(str(bg_files[(i + t * 7) % len(bg_files)]), cv2.IMREAD_COLOR)
            scene, corners = composite_card(bg.copy(), card_bgra, rng, angle_range=(-30.0, 30.0))
            dets = cd.detect(scene)
            if dets:
                d = dets[0]
                h_frame, w_frame = scene.shape[:2]
                box_area = cv2.contourArea(d.corners.astype(np.int32))
                box_area_ratio = box_area / (h_frame * w_frame)
                
                sides = []
                for j in range(4):
                    p1 = d.corners[j]
                    p2 = d.corners[(j + 1) % 4]
                    sides.append(np.linalg.norm(p2 - p1))
                sides.sort()
                box_aspect_ratio = sides[0] / sides[3] if sides[3] > 0 else 1.0
                
                quality_result = quality_gate.evaluate(
                    image=d.patch,
                    detection_confidence=d.confidence,
                    box_aspect_ratio=box_aspect_ratio,
                    box_area_ratio=box_area_ratio,
                )
                
                if quality_result.passed:
                    return d.patch
        fallbacks[0] += 1
        src = corners.astype(np.float32)
        dst = np.array([[0, 0], [447, 0], [447, 629], [0, 629]], np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(scene, M, (448, 630))
        return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    curve_card = card_files[0]
    curve_card_path = str(curve_card)
    curves = {d: [] for d in DEFECT_TYPES}
    for defect in DEFECT_TYPES:
        for s in severities:
            card = load_card_rgba(curve_card)
            if s > 0:
                card = _apply_at(card, defect, s, rng)
            patch = photograph(card, 0, f"curve_{defect}_{s}")
            det.load_reference_images([curve_card_path], card_id="curve")
            curves[defect].append(det.analyze_patch(patch)["score"])
        exp.logger.info(f"curve {defect}: "
                        + " ".join(f"{s:.2f}:{v:.3f}" for s, v in zip(severities, curves[defect])))
        exp.log_metric(f"severity_curve.{defect}", curves[defect])

    labels_true, labels_pred, scores_all, is_damaged = [], [], [], []
    sample_grid = []
    for i in range(n_cards):
        card_path = card_files[(i * 37 + 11) % len(card_files)]
        card_path_str = str(card_path)
        pristine = load_card_rgba(card_path)

        patch = photograph(pristine, i, "NM")
        det.load_reference_images([card_path_str], card_id=f"dmged_{i}")
        res = det.analyze_patch(patch)
        labels_true.append("NM")
        labels_pred.append(PRED_TO_TAG[res["condition"]["label"]])
        scores_all.append(res["score"])
        is_damaged.append(0)

        n_defects = {"LP": 1, "MP": 1, "HP": 2, "DMG": 3}
        for level in ["LP", "MP", "HP", "DMG"]:
            damaged = pristine.copy()
            defect = None
            for d in rng.sample(DEFECT_TYPES, n_defects[level]):
                damaged, defect = synthesize_damaged(damaged, level, rng, defect=d)
            patch = photograph(damaged, i, level)
            det.load_reference_images([card_path_str], card_id=f"dmged_{i}")
            res = det.analyze_patch(patch)
            labels_true.append(level)
            labels_pred.append(PRED_TO_TAG[res["condition"]["label"]])
            scores_all.append(res["score"])
            is_damaged.append(1)
            if len(sample_grid) < 12:
                hm = cv2.imdecode(np.frombuffer(
                    __import__("base64").b64decode(res["heatmap_b64"]), np.uint8),
                    cv2.IMREAD_COLOR)
                sample_grid.append((np.rot90(patch, 2) if res["flipped"] else patch,
                                    hm, level,
                                    res["condition"]["label"], res["score"], defect))
        exp.logger.info(f"card {i+1}/{n_cards} ({card_path.stem}) done")

    scores_arr = np.array(scores_all)
    is_damaged_arr = np.array(is_damaged)

    dmg_scores = scores_arr[is_damaged_arr == 1]
    pristine_scores = scores_arr[is_damaged_arr == 0]
    detection_rate = float((dmg_scores > threshold).mean())
    exp.log_metric("n_cards", n_cards)
    exp.log_metric("n_samples", len(scores_all))
    exp.log_metric("detection_rate", detection_rate)
    exp.log_metric("pristine_mean", float(pristine_scores.mean()))
    exp.log_metric("damaged_mean", float(dmg_scores.mean()))
    exp.log_metric("damaged_min", float(dmg_scores.min()))
    exp.log_metric("gt_warp_fallbacks", fallbacks[0])

    level_tags = ["NM", "LP", "MP", "HP", "DMG"]

    level_stats = {}
    for tag in level_tags:
        arr = np.array([s for t, s in zip(labels_true, scores_all) if t == tag])
        level_stats[tag] = {"mean": float(arr.mean()), "min": float(arr.min()),
                            "max": float(arr.max()), "n": int(len(arr))}
    exp.log_metric("level_stats", level_stats)

    groups_true = [TRUE_BY_TAG[t] for t in labels_true]
    groups_pred = list(labels_pred)
    group_tags = ["NM", "GE", "PP"]

    exact = float(np.mean([t == p for t, p in zip(groups_true, groups_pred)]))
    adjacent = float(np.mean([abs(group_tags.index(t) - group_tags.index(p)) <= 1
                              for t, p in zip(groups_true, groups_pred)]))
    exp.log_metric("condition_exact_accuracy", exact)
    exp.log_metric("condition_adjacent_accuracy", adjacent)

    cm = np.zeros((3, 3), int)
    for t, p in zip(groups_true, groups_pred):
        cm[group_tags.index(t), group_tags.index(p)] += 1
    exp.log_metric("confusion_matrix", {t: dict(zip(group_tags, cm[i].tolist()))
                                        for i, t in enumerate(group_tags)})

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for defect in DEFECT_TYPES:
        ax.plot(severities, curves[defect], marker="o", lw=1.8, label=defect)
    ax.axhline(threshold, color="#FF4D6A", ls="--", lw=2,
               label=f"threshold = {threshold:.3f}")
    ax.set_xlabel("defect severity (0 = pristine)")
    ax.set_ylabel("anomaly score")
    ax.set_title("Phase 2d - Anomaly score vs defect severity (card A1_001)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    exp.save_plot(fig, "score_vs_severity")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(pristine_scores, bins=10, color="#3B9EFF", alpha=0.8,
            edgecolor="white", label=f"pristine (n={len(pristine_scores)})")
    ax.hist(dmg_scores, bins=14, color="#FF4D6A", alpha=0.75,
            edgecolor="white", label=f"damaged (n={len(dmg_scores)})")
    ax.axvline(threshold, color="black", ls="--", lw=2,
               label=f"threshold = {threshold:.3f}")
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("count")
    ax.set_title(f"Phase 2d - Pristine vs damaged scores "
                 f"(detection rate = {detection_rate:.0%})")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    exp.save_plot(fig, "score_distributions")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), group_tags, fontsize=9)
    ax.set_yticks(range(3), group_tags, fontsize=9)
    for r in range(3):
        for c in range(3):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                    color="white" if cm[r, c] > cm.max() / 2 else "black",
                    fontsize=12, fontweight="bold")
    ax.set_xlabel("predicted condition")
    ax.set_ylabel("synthesized condition (ground truth)")
    ax.set_title(f"Phase 2d - Condition confusion matrix\n"
                 f"exact={exact:.0%}, ±1 adjacent={adjacent:.0%}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    exp.save_plot(fig, "confusion_matrix")

    if sample_grid:
        n = len(sample_grid)
        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 4.0))
        axes = np.atleast_2d(axes)
        for j, (photo, hm, lt, lp_, sc, df) in enumerate(sample_grid):
            ax = axes[j // cols, j % cols]
            ax.imshow(photo)
            if hm is not None:
                hm_rgb = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
                hm_r = cv2.resize(hm_rgb, (photo.shape[1], photo.shape[0]))
                ax.imshow(hm_r, alpha=0.45)
            ax.set_title(f"{df} → {lt} (pred {lp_})\nscore {sc:.3f}", fontsize=8)
            ax.axis("off")
        for j in range(n, rows * cols):
            axes[j // cols, j % cols].axis("off")
        fig.suptitle("Phase 2d - Synthesized damaged cards + anomaly heatmaps")
        fig.tight_layout()
        exp.save_plot(fig, "damage_samples")

    exp.finish()
    exp.logger.info(f"DONE. detection_rate={detection_rate:.0%}, "
                    f"exact={exact:.0%}, adjacent±1={adjacent:.0%}")

def _apply_at(card_bgra, defect, severity, rng):
    """Apply a defect at an arbitrary continuous severity."""
    from backend.damage import apply_damage
    return apply_damage(card_bgra, defect, severity, rng)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(n)

"""Anomaly threshold calibration procedure for the Baseline pipeline.

Evaluates a collection of pristine card samples processed through the full
pipeline (OBB detection, warping, CLAHE equalization, DINOv2) to determine the
statistical score distribution and calculate the optimal decision threshold.
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
from backend.condition import COLORS, TAGS  # noqa: E402

logger = logging.getLogger(__name__)

def main():
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("phase2b_calibration", cfg)

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

    from scripts.prepare_dataset import composite_card
    from backend.quality_gate import ImageQualityGate

    rng = random.Random(cfg["project"]["seed"] + 2)
    n = cfg["anomaly"]["calibration"]["n_pristine_cards"]
    quality_gate = ImageQualityGate()
    scores = []
    attempts = 0
    while len(scores) < n and attempts < n * 3:
        i = attempts
        attempts += 1
        card_path = card_files[(i * 29 + 3) % len(card_files)]
        card = cv2.imread(str(card_path), cv2.IMREAD_UNCHANGED)
        if card.shape[2] == 3:
            card = cv2.cvtColor(card, cv2.COLOR_BGR2BGRA)
        bg = cv2.imread(str(bg_files[i % len(bg_files)]), cv2.IMREAD_COLOR)
        scene, _ = composite_card(bg, card, rng, angle_range=(-30.0, 30.0))
        dets = cd.detect(scene)
        if not dets:
            exp.logger.info(f"  attempt {attempts}: no detection, retrying")
            continue
        
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
        
        if not quality_result.passed:
            exp.logger.info(f"  attempt {attempts}: low quality "
                            f"({quality_result.score:.2f}), retrying")
            continue

        det.load_reference_images([str(card_path)], card_id=f"calib_{len(scores)}")
        s = det.analyze_patch(dets[0].patch)["score"]
        scores.append(s)
        exp.log_metric(f"pristine_scores.card_{len(scores) - 1}", s)
        exp.logger.info(f"  card {len(scores)}/{n} ({card_path.stem}): score={s:.4f}")
    exp.log_metric("attempts", attempts)
    if len(scores) < n:
        exp.logger.warning(f"Only {len(scores)}/{n} pristine cards scored.")

    scores_arr = np.array(scores)
    mu, sigma = float(scores_arr.mean()), float(scores_arr.std())
    strategy = cfg["anomaly"]["calibration"]["strategy"]
    if strategy == "mean_plus_3sigma":
        threshold = mu + 3 * sigma
    elif strategy == "percentile_95":
        threshold = float(np.percentile(scores_arr, 95))
    elif strategy == "percentile_99":
        threshold = float(np.percentile(scores_arr, 99))
    else:
        threshold = float(scores_arr.max())
    threshold = float(np.clip(threshold, 0.01, 1.50))

    exp.log_metric("mean", mu)
    exp.log_metric("sigma", sigma)
    exp.log_metric("strategy", strategy)
    exp.log_metric("calibrated_threshold", threshold)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(scores_arr, bins=12, color="#3B9EFF", alpha=0.8, edgecolor="white")
    ax.axvline(threshold, color="#FF4D6A", ls="--", lw=2,
               label=f"calibrated threshold = {threshold:.3f} ({strategy})")
    ax.axvline(mu, color="black", ls=":", lw=1.5, label=f"μ = {mu:.3f}")
    ax.set_xlabel("anomaly score (pristine cards)")
    ax.set_ylabel("count")
    ax.set_title(f"Phase 2b - Threshold calibration on {n} pristine cards\n"
                 f"(mask + CLAHE pipeline)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    exp.save_plot(fig, "threshold_calibration")

    bands = dict(cfg["condition"]["bands"])
    exp.log_metric("calibrated_bands", bands)

    labels = list(cfg["condition"]["labels"])
    edges = [bands[l] for l in labels[:-1]]
    lower_bounds = [0.0] + edges
    upper_bounds = edges + [max(1.5, edges[-1] * 1.8)]
    for i, lab in enumerate(labels):
        ax.barh(0, upper_bounds[i] - lower_bounds[i], left=lower_bounds[i], height=0.55,
                color=COLORS[lab], alpha=0.45, edgecolor="white")
        ax.text((lower_bounds[i] + upper_bounds[i]) / 2, 0, f"{TAGS[lab]}\n{lab}",
                ha="center", va="center", fontsize=9, fontweight="bold", color="#111827")
    ax.axvline(threshold, color="#111827", linewidth=2.0, linestyle="--",
               label=f"calibrated threshold = {threshold:.3f}")
    ax.set_yticks([])
    ax.set_xlabel("anomaly score")
    ax.set_xlim(0, max(1.5, upper_bounds[-1]))
    ax.set_title("Phase 2c - Calibrated consolidated condition bands")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28))
    exp.save_plot(fig, "condition_bands")

    exp.finish()
    exp.logger.info(f"DONE. Calibrated threshold = {threshold:.4f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

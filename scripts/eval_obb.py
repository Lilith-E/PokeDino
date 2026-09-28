"""Quantitative evaluation of the YOLO OBB detection model on the test split.

Esegue la validazione sui dati di test indipendenti calcolando:
- Metriche standard di object detection: mAP50, mAP50-95, precision e recall.
- Grafici di predizione con sovrapposizione dei poligoni orientati.
- Robustezza della detection al variare dell'angolo di rotazione (0-360 gradi).
"""

import logging
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
from backend.card_detector import CardDetector  # noqa: E402

logger = logging.getLogger(__name__)

def main():
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("phase1c_eval_obb", cfg)

    data_yaml = PROJECT_ROOT / "data" / "datasets" / "pokemon_obb" / "pokemon_obb.yaml"

    weights = CardDetector._find_latest_weights()
    if weights is None:
        exp.logger.error("No trained OBB weights found. Run scripts/train_obb.py first.")
        sys.exit(1)
    exp.logger.info(f"Evaluating weights: {weights}")
    exp.log_metric("weights", weights)

    from ultralytics import YOLO
    model = YOLO(weights)
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=cfg["obb"]["model"]["imgsz"],
        device="cpu",
        verbose=False,
    )
    results = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    for k, v in results.items():
        exp.log_metric(k, v)
    exp.logger.info(f"Test metrics: {results}")

    cd = CardDetector(weights_path=weights)
    test_dir = data_yaml.parent / "images" / "test"
    label_dir = data_yaml.parent / "labels" / "test"
    scenes = sorted(test_dir.glob("*.jpg"))[:6]
    if scenes:
        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        for ax, scene_path in zip(axs.ravel(), scenes):
            img = cv2.imread(str(scene_path))
            dets = cd.detect(img)
            vis = cd.draw_detections(img, dets)
            lbl = label_dir / (scene_path.stem + ".txt")
            if lbl.exists():
                h, w = img.shape[:2]
                with open(lbl) as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) < 9:
                            continue
                        vals = np.array(list(map(float, parts[1:13])))
                        pts = vals.reshape(-1, 2) * [w, h]
                        cv2.polylines(vis, [pts.astype(np.int32)], True, (80, 220, 80), 2)
            ax.imshow(vis)
            ax.set_title(f"{scene_path.stem} - {len(dets)} det", fontsize=9)
            ax.axis("off")
        fig.suptitle("Phase 1c - OBB detections (yellow=pred, green=GT) on test scenes")
        fig.tight_layout()
        exp.save_plot(fig, "detection_samples")

    cards_dir = PROJECT_ROOT / "data" / "datasets" / "cards" / "pokemon_pocket_tcgp"
    bg_files = sorted((PROJECT_ROOT / "data" / "datasets" / "backgrounds").glob("*.jpg"))
    card_files = sorted(cards_dir.glob("*.webp"))
    if card_files and bg_files:
        from scripts.prepare_dataset import composite_card
        import random
        rng = random.Random(cfg["project"]["seed"])
        card = cv2.imread(str(card_files[0]), cv2.IMREAD_UNCHANGED)
        if card.shape[2] == 3:
            card = cv2.cvtColor(card, cv2.COLOR_BGR2BGRA)
        bg = cv2.imread(str(bg_files[0]), cv2.IMREAD_COLOR)

        angles = list(range(0, 360, 30))
        confs = []
        for ang in angles:
            h, w = card.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
            M[0, 2] += nw / 2 - w / 2
            M[1, 2] += nh / 2 - h / 2
            rot = cv2.warpAffine(card, M, (nw, nh))
            scene, _ = composite_card(bg.copy(), rot, rng)
            dets = cd.detect(scene)
            confs.append(dets[0].confidence if dets else 0.0)

        exp.log_metric("angle_robustness.angles", angles)
        exp.log_metric("angle_robustness.confidences", confs)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(angles, confs, "o-", color="#3B9EFF", lw=2)
        ax.set_xlabel("card rotation (degrees)")
        ax.set_ylabel("detection confidence")
        ax.set_title("Phase 1c - OBB robustness to arbitrary card rotation")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        exp.save_plot(fig, "angle_robustness")

    exp.finish()
    exp.logger.info("DONE. Phase 1c OBB evaluation complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

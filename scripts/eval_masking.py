"""Evaluation of perimeter geometric masking effects.

Esegue uno studio di ablazione confrontando i punteggi di anomalia calcolati
con e senza mascheramento ad angoli arrotondati, quantificando l'eliminazione
dei falsi positivi generati dalle porzioni triangolari di sfondo lungo i bordi.
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
from backend.masking import rounded_rect_mask, mask_overlay_figure  # noqa: E402
from backend.detector import PokemonAnomalyDetector  # noqa: E402

logger = logging.getLogger(__name__)

IMG_W, IMG_H = 640, 640

def composite_and_warp_back(card_bgra, bg, rng):
    """
    Composite a card onto a background at a random rotation, then warp it back
    to the upright 630x448 patch using the ground-truth quad - mimicking the
    OBB detect → warp step, but with known corners.
    """
    from scripts.prepare_dataset import composite_card, load_background

    scene = bg.copy()
    scene, corners = composite_card(scene, card_bgra, rng)
    src = corners.astype(np.float32)
    dst = np.array([[0, 0], [448 - 1, 0], [448 - 1, 630 - 1], [0, 630 - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(scene, M, (448, 630))
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

def main():
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("phase3_masking", cfg)

    cards_dir = PROJECT_ROOT / "data" / "datasets" / "cards" / "pokemon_pocket_tcgp"
    card_files = sorted(cards_dir.glob("*.webp"))
    bg_dir = PROJECT_ROOT / "data" / "datasets" / "backgrounds"
    bg_files = sorted(bg_dir.rglob("*.jpg"))
    if not card_files or not bg_files:
        exp.logger.error("Missing cards or backgrounds. Run prepare_dataset.py first.")
        sys.exit(1)

    rng = random.Random(cfg["project"]["seed"])
    nprng = np.random.default_rng(cfg["project"]["seed"])
    for i in range(3):
        card = cv2.imread(str(card_files[i * 53]), cv2.IMREAD_UNCHANGED)
        if card.shape[2] == 3:
            card = cv2.cvtColor(card, cv2.COLOR_BGR2BGRA)
        bg = cv2.imread(str(bg_files[i]), cv2.IMREAD_COLOR)
        warped = composite_and_warp_back(card, bg, rng)
        mask = rounded_rect_mask(630, 448,
                                 corner_radius_frac=cfg["masking"]["corner_radius_frac"],
                                 erode_frac=cfg["masking"]["erode_frac"],
                                 feather_px=cfg["masking"]["feather_px"])
        fig = mask_overlay_figure(warped, mask,
                                  title=f"Phase 3 - Rounded-corner mask (sample {i+1})")
        exp.save_plot(fig, f"mask_overlay_{i+1}")

    exp.logger.info("Loading AnomalyDINO model for ablation…")
    det_masked = PokemonAnomalyDetector(masking_enabled=True, clahe_enabled=True)
    det_plain = PokemonAnomalyDetector(masking_enabled=False, clahe_enabled=True)

    n_test = 8
    scores_masked, scores_plain = [], []
    test_patches = []
    for i in range(n_test):
        card_path = card_files[(i * 91 + 7) % len(card_files)]
        card = cv2.imread(str(card_path), cv2.IMREAD_UNCHANGED)
        if card.shape[2] == 3:
            card = cv2.cvtColor(card, cv2.COLOR_BGR2BGRA)
        bg = cv2.imread(str(bg_files[i % len(bg_files)]), cv2.IMREAD_COLOR)
        warped = composite_and_warp_back(card, bg, rng)
        tmp = PROJECT_ROOT / "data" / "uploads" / f"_ablation_{i}.png"
        cv2.imwrite(str(tmp), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
        test_patches.append(warped)
        det_masked.load_reference_images([str(card_path)], card_id=f"ablation_{i}")
        det_plain.load_reference_images([str(card_path)], card_id=f"ablation_{i}")
        s_m = det_masked.analyze_card(str(tmp))["score"]
        s_p = det_plain.analyze_card(str(tmp))["score"]
        scores_masked.append(s_m)
        scores_plain.append(s_p)
        exp.log_metric(f"ablation.card_{i}.score_masked", s_m)
        exp.log_metric(f"ablation.card_{i}.score_plain", s_p)
        exp.log_metric(f"ablation.card_{i}.card_id", card_path.stem)
        tmp.unlink()

    mean_m, mean_p = float(np.mean(scores_masked)), float(np.mean(scores_plain))
    exp.log_metric("ablation.mean_score_masked", mean_m)
    exp.log_metric("ablation.mean_score_plain", mean_p)
    exp.log_metric("ablation.delta", mean_p - mean_m)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(n_test)
    ax.bar(x - 0.18, scores_plain, 0.36, label="Without mask (background corners scored)",
           color="#FF4D6A")
    ax.bar(x + 0.18, scores_masked, 0.36, label="With rounded-corner mask", color="#00E396")
    ax.axhline(cfg["anomaly"]["threshold"], ls="--", color="black", alpha=0.6,
               label=f"threshold = {cfg['anomaly']['threshold']}")
    ax.set_xticks(x)
    ax.set_xlabel("test card")
    ax.set_ylabel("anomaly score")
    ax.set_title("Phase 3 - Ablation: anomaly score with vs without corner masking\n"
                 "(pristine cards; lower is better; masking removes edge false-positives)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    exp.save_plot(fig, "ablation_masking_scores")

    worst = int(np.argmax(np.array(scores_plain) - np.array(scores_masked)))
    card_path = card_files[(worst * 91 + 7) % len(card_files)]
    warped = test_patches[worst]
    tmp = PROJECT_ROOT / "data" / "uploads" / "_ablation_worst.png"
    cv2.imwrite(str(tmp), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
    det_masked.load_reference_images([str(card_path)], card_id=f"ablation_{worst}")
    det_plain.load_reference_images([str(card_path)], card_id=f"ablation_{worst}")
    r_plain = det_plain.analyze_card(str(tmp))
    r_masked = det_masked.analyze_card(str(tmp))
    tmp.unlink()

    import base64, io
    from PIL import Image

    def b64_to_img(b64):
        return np.array(Image.open(io.BytesIO(base64.b64decode(b64))))

    fig, axs = plt.subplots(1, 3, figsize=(13, 5))
    axs[0].imshow(warped)
    axs[0].set_title("Input patch (bg corners visible)")
    axs[1].imshow(b64_to_img(r_plain["heatmap_b64"]))
    axs[1].set_title(f"Without mask - score {r_plain['score']:.3f}")
    axs[2].imshow(b64_to_img(r_masked["heatmap_b64"]))
    axs[2].set_title(f"With mask - score {r_masked['score']:.3f}")
    for ax in axs:
        ax.axis("off")
    fig.suptitle("Phase 3 - Masking removes background corner false-positives")
    exp.save_plot(fig, "ablation_heatmaps")

    exp.finish()
    exp.logger.info("DONE. Phase 3 masking evaluation complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

"""Evaluation of CLAHE equalization on non-uniform illumination samples.

Analizza l'impatto dell'equalizzazione adattiva dell'istogramma (canale L)
su scansioni con gradienti di luce, ombre e riflessi simulati, quantificando
la variazione del contrasto RMS ed esportando i grafici comparativi.
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.experiment import Experiment, load_config  # noqa: E402
from backend.preprocessing import apply_clahe, clahe_comparison_figure, l_channel_rms_contrast  # noqa: E402

logger = logging.getLogger(__name__)

def add_hotspot(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate a bright specular spot (bright elliptical blob)."""
    out = img.astype(np.float32)
    h, w = out.shape[:2]
    cx, cy = rng.integers(int(0.2 * w), int(0.8 * w)), rng.integers(int(0.15 * h), int(0.4 * h))
    axes = (int(0.18 * w), int(0.10 * h))
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy)), axes, rng.integers(0, 180), 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 25).astype(np.float32) / 255.0
    out += mask[..., None] * 140
    return np.clip(out, 0, 255).astype(np.uint8)

def add_shadow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate a dark shadow gradient from one side."""
    out = img.astype(np.float32)
    h, w = out.shape[:2]
    gx = np.linspace(1, 0, w, dtype=np.float32)[None, :]
    gy = np.linspace(0, 0.4, h, dtype=np.float32)[:, None]
    grad = np.clip(gx * rng.uniform(0.5, 1.0) + gy * 0.3, 0, 1)
    out *= (0.45 + 0.55 * grad)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)

def add_low_contrast(img: np.ndarray) -> np.ndarray:
    """Compress contrast around mid gray."""
    out = img.astype(np.float32) * 0.45 + 110
    return np.clip(out, 0, 255).astype(np.uint8)

def main():
    cfg = load_config()
    exp = Experiment("phase2a_clahe", cfg)
    ccfg = cfg["clahe"]

    cards_dir = PROJECT_ROOT / "data" / "datasets" / "cards" / "pokemon_pocket_tcgp"
    card_files = sorted(cards_dir.glob("*.webp"))
    if not card_files:
        exp.logger.error("No card scans found.")
        sys.exit(1)

    rng = np.random.default_rng(cfg["project"]["seed"])
    cases = ["hotspot", "shadow", "low_contrast", "clean"]
    metrics = {}

    for i, case in enumerate(cases):
        img = cv2.cvtColor(cv2.imread(str(card_files[i * 37]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (448, 630))
        if case == "hotspot":
            degraded = add_hotspot(img, rng)
        elif case == "shadow":
            degraded = add_shadow(img, rng)
        elif case == "low_contrast":
            degraded = add_low_contrast(img)
        else:
            degraded = img.copy()

        fig, eq = clahe_comparison_figure(
            degraded, ccfg["clip_limit"], tuple(ccfg["tile_grid_size"]),
            title=f"Phase 2a - CLAHE: {case} case")
        exp.save_plot(fig, f"clahe_comparison_{case}")

        rms_before = l_channel_rms_contrast(degraded)
        rms_after = l_channel_rms_contrast(eq)
        metrics[case] = {"rms_before": round(rms_before, 2), "rms_after": round(rms_after, 2)}
        exp.log_metric(f"rms_contrast.{case}.before", rms_before)
        exp.log_metric(f"rms_contrast.{case}.after", rms_after)

    exp.logger.info(f"CLAHE RMS contrast summary: {metrics}")
    exp.finish()
    exp.logger.info("DONE. Phase 2a CLAHE evaluation complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

#!/usr/bin/env python3
"""Test generating the masking ablation heatmap with clearly evident corner defects."""

import sys
import random
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.experiment import load_config, set_seed
from backend.masking import rounded_rect_mask
from scripts.eval_masking import composite_and_warp_back

def render_ablation_overlay(warped_rgb, d_map, mask=None, corner_boost=True):
    h, w = warped_rgb.shape[:2]
    d_full = cv2.resize(d_map.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    
    if mask is not None:
        d_full = d_full * mask
    
    d_smooth = cv2.GaussianBlur(d_full, (0, 0), 2.5)
    
    if mask is not None:
        d_smooth = d_smooth * mask

    dz = 0.08
    vmax = 0.38
    norm = np.clip((d_smooth - dz) / (vmax - dz), 0.0, 1.0)
    norm = np.power(norm, 1.2)
    
    if mask is not None:
        norm = norm * mask
        
    heat_u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    base_alpha = 0.50
    alpha = 0.85
    alpha_map = np.clip(base_alpha + (alpha - base_alpha) * np.power(norm, 0.8), 0.0, 1.0)[:, :, np.newaxis]
    
    blended = (colored * alpha_map + warped_rgb * (1.0 - alpha_map)).astype(np.uint8)
    return blended

def main():
    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    cards_dir = PROJECT_ROOT / "data" / "datasets" / "cards" / "pokemon_pocket_tcgp"
    card_files = sorted(cards_dir.glob("*.webp"))
    bg_dir = PROJECT_ROOT / "data/datasets/backgrounds"
    bg_files = sorted(bg_dir.rglob("*.jpg"))

    rng = random.Random(cfg["project"]["seed"])
    card_path = card_files[(1 * 91 + 7) % len(card_files)]
    card = cv2.imread(str(card_path), cv2.IMREAD_UNCHANGED)
    if card.shape[2] == 3: card = cv2.cvtColor(card, cv2.COLOR_BGR2BGRA)
    bg = cv2.imread(str(bg_files[1 % len(bg_files)]), cv2.IMREAD_COLOR)
    warped = composite_and_warp_back(card, bg, rng)

    tmp = PROJECT_ROOT / "data/uploads/_test_abl_render.png"
    cv2.imwrite(str(tmp), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))

    det_masked = PokemonAnomalyDetector(masking_enabled=True, clahe_enabled=True)
    det_plain = PokemonAnomalyDetector(masking_enabled=False, clahe_enabled=True)
    det_masked.load_reference_images([str(card_path)], card_id="abl_1")
    det_plain.load_reference_images([str(card_path)], card_id="abl_1")

    r_plain = det_plain.analyze_card(str(tmp))
    r_masked = det_masked.analyze_card(str(tmp))
    tmp.unlink()

    d_plain = np.array(r_plain["grid_anomaly_map"], dtype=np.float32)
    d_masked = np.array(r_masked["grid_anomaly_map"], dtype=np.float32)

    mask_2d = rounded_rect_mask(630, 448, corner_radius_frac=0.035, erode_frac=0.01)

    overlay_plain = render_ablation_overlay(warped, d_plain, mask=None)
    overlay_masked = render_ablation_overlay(warped, d_masked, mask=mask_2d)

    DPI = 300
    THEME = {
        "fig_bg": "#ffffff",
        "text": "#0f172a",
        "rose": "#e11d48",
        "emerald": "#059669",
    }
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.8), dpi=DPI)
    fig.patch.set_facecolor(THEME["fig_bg"])

    axs[0].imshow(warped)
    axs[0].set_title("Input Warped Card\n(Background Corners Visible)",
                     fontsize=9.5, fontweight="bold", color=THEME["text"], pad=6)
    axs[0].axis("off")

    axs[1].imshow(overlay_plain)
    axs[1].set_title(f"Without Mask (Artifact Flawed: Score {r_plain['score']:.3f})",
                     fontsize=9.5, fontweight="bold", color=THEME["rose"], pad=6)
    axs[1].axis("off")

    axs[2].imshow(overlay_masked)
    axs[2].set_title(f"With Rounded-Corner Mask (Pristine: Score {r_masked['score']:.3f})",
                     fontsize=9.5, fontweight="bold", color=THEME["emerald"], pad=6)
    axs[2].axis("off")

    fig.suptitle(
        "Phase 3 — Ablation: False-Positive Boundary Mitigation via Analytical Rounded-Corner Masking",
        fontsize=12, fontweight="bold", color=THEME["text"], y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = PROJECT_ROOT / "scratch/test_ablation_fig.png"
    fig.savefig(out, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated test figure: {out}")

if __name__ == "__main__":
    main()

"""Synthetic dataset generator for Stage-2 Canonical Boundary MobileNetV3-UNet.

Simulates Stage-1 YOLO-OBB cropped/de-rotated card regions at 1:1 canonical
scale (768x544, centering a 630x448 card with 48/69 px padding) under realistic
OBB shift/scale/angle noise, 3D perspective trapezoidal residuals, crossing drop
shadows, border glare, sleeves, full-art cards, damage, and finger occlusions.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.experiment import Experiment, load_config, set_seed  # noqa: E402
from backend.rectifier import (  # noqa: E402
    CANON_H,
    CANON_W,
    TARGET_H,
    TARGET_W,
    order_quad,
    warp_to_canonical_obb,
)
from scripts.prepare_dataset import (  # noqa: E402
    apply_damage_augmentations,
    get_available_backgrounds,
    load_background,
    load_card_rgba,
)
from scripts.prepare_seg_dataset import (  # noqa: E402
    _apply_finger_occlusion,
    _apply_scene_shadows_and_glare,
    _draw_synthetic_sleeve,
    _make_full_art_borderless,
    ensure_rounded_alpha,
)

logger = logging.getLogger(__name__)

SCENE_W, SCENE_H = 960, 960


def _sample_canonical_card_pools(cfg: dict) -> tuple[dict[str, list[Path]], list[Path]]:
    """Collect disjoint card pools (train/val/test) and background images."""
    hf_dir = PROJECT_ROOT / "data" / "datasets" / "cards" / "pokemon_official_tcg"
    legacy_dir = PROJECT_ROOT / "data" / "datasets" / "cards" / "pokemon_pocket_tcgp"
    legacy_cards = sorted(legacy_dir.glob("*.webp")) if legacy_dir.exists() else []

    card_pools = {
        "train": sorted((hf_dir / "train").glob("*.webp")) if (hf_dir / "train").exists() else [],
        "val": sorted((hf_dir / "val").glob("*.webp")) if (hf_dir / "val").exists() else [],
        "test": sorted((hf_dir / "test").glob("*.webp")) if (hf_dir / "test").exists() else [],
    }
    if legacy_cards:
        rng_leg = random.Random(cfg["project"]["seed"])
        leg = legacy_cards.copy()
        rng_leg.shuffle(leg)
        s_ratios = cfg.get("segmentation", {}).get("dataset", {}).get("split", [0.70, 0.15, 0.15])
        n_tr = int(len(leg) * s_ratios[0])
        n_va = int(len(leg) * s_ratios[1])
        card_pools["train"].extend(leg[:n_tr])
        card_pools["val"].extend(leg[n_tr : n_tr + n_va])
        card_pools["test"].extend(leg[n_tr + n_va :])

    for pool in card_pools.values():
        random.Random(cfg["project"]["seed"]).shuffle(pool)

    bgs = get_available_backgrounds(
        min_needed=120,
        seed=cfg["project"]["seed"],
    )
    return card_pools, bgs


def _build_scene_quad(
    w_card: int,
    h_card: int,
    rng: random.Random,
    apply_perspective: bool,
    max_tilt: float = 0.16,
) -> np.ndarray:
    """Place one card near scene center with optional 3D perspective trapezoid tilt."""
    target_long = rng.randint(570, 680)
    scale = target_long / max(h_card, w_card)
    nw, nh = w_card * scale, h_card * scale
    local = np.array(
        [[-nw / 2, -nh / 2], [nw / 2, -nh / 2], [nw / 2, nh / 2], [-nw / 2, nh / 2]],
        dtype=np.float64,
    )
    if apply_perspective:
        tilt_x = rng.uniform(0.03, max_tilt) * nw
        tilt_y = rng.uniform(0.03, max_tilt) * nh
        side = rng.choice(["top", "bottom", "left", "right"])
        if side == "top":
            local[0, 0] += tilt_x * 0.5
            local[1, 0] -= tilt_x * 0.5
        elif side == "bottom":
            local[3, 0] += tilt_x * 0.5
            local[2, 0] -= tilt_x * 0.5
        elif side == "left":
            local[0, 1] += tilt_y * 0.5
            local[3, 1] -= tilt_y * 0.5
        else:
            local[1, 1] += tilt_y * 0.5
            local[2, 1] -= tilt_y * 0.5
        for k in range(4):
            local[k, 0] += rng.uniform(-0.03, 0.03) * nw
            local[k, 1] += rng.uniform(-0.03, 0.03) * nh

    theta = np.radians(rng.uniform(0.0, 360.0))
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    center = np.array([SCENE_W * 0.5 + rng.uniform(-25.0, 25.0), SCENE_H * 0.5 + rng.uniform(-25.0, 25.0)])
    return order_quad(local @ rot.T + center)


def _perturb_stage1_obb(corners_gt: np.ndarray, rng: random.Random) -> np.ndarray:
    """Approximate true quad by its min-area rectangle and add realistic Stage-1 OBB jitter."""
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(corners_gt.astype(np.float32))
    cx_n = float(cx) + rng.uniform(-16.0, 16.0)
    cy_n = float(cy) + rng.uniform(-16.0, 16.0)
    rw_n = float(rw) * rng.uniform(0.94, 1.07)
    rh_n = float(rh) * rng.uniform(0.94, 1.07)
    ang_n = float(ang) + rng.uniform(-3.5, 3.5)
    box = cv2.boxPoints(((cx_n, cy_n), (rw_n, rh_n), ang_n)).astype(np.float64)
    return order_quad(box)


def generate_split(
    split_name: str,
    n_samples: int,
    card_pool: list[Path],
    backgrounds: list[Path],
    out_dir: Path,
    rng: random.Random,
    corner_radius_frac: float = 0.045,
) -> dict[str, int]:
    """Generate canonical Stage-2 training/validation crops for one split."""
    stats = {
        "full_art": 0,
        "perspective": 0,
        "sleeve": 0,
        "occlusion": 0,
        "damage": 0,
        "shadow": 0,
        "glare": 0,
    }
    for sub in ("images", "masks", "meta"):
        d = out_dir / sub / split_name
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*"):
            if old.is_file():
                old.unlink()

    sample_vis: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for idx in range(n_samples):
        bg_640 = load_background(rng.choice(backgrounds), rng)
        bg = cv2.resize(bg_640, (SCENE_W, SCENE_H), interpolation=cv2.INTER_LINEAR)
        card_rgba = ensure_rounded_alpha(
            load_card_rgba(rng.choice(card_pool)),
            corner_radius_frac=corner_radius_frac,
        )
        tags: list[str] = []

        if rng.random() < 0.28:
            card_rgba = _make_full_art_borderless(card_rgba, corner_radius_frac, rng)
            tags.append("full_art")
            stats["full_art"] += 1
        if rng.random() < 0.45:
            card_rgba = apply_damage_augmentations(card_rgba, rng)
            tags.append("damage")
            stats["damage"] += 1

        use_persp = rng.random() < 0.72
        if use_persp:
            tags.append("perspective")
            stats["perspective"] += 1

        hc, wc = card_rgba.shape[:2]
        quad_gt = _build_scene_quad(wc, hc, rng, apply_perspective=use_persp)
        src_pts = np.array([[0, 0], [wc - 1, 0], [wc - 1, hc - 1], [0, hc - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, quad_gt.astype(np.float32))
        warped = cv2.warpPerspective(
            card_rgba, M, (SCENE_W, SCENE_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )
        alpha = (warped[:, :, 3:4].astype(np.float32) / 255.0)
        card_rgb = warped[:, :, :3].astype(np.float32)
        scene = np.clip(bg.astype(np.float32) * (1.0 - alpha) + card_rgb * alpha, 0, 255).astype(np.uint8)
        gt_mask = (warped[:, :, 3] > 127).astype(np.uint8) * 255

        if rng.random() < 0.35:
            scene = _draw_synthetic_sleeve(scene, quad_gt, rng)
            tags.append("sleeve")
            stats["sleeve"] += 1

        if rng.random() < 0.35:
            scene, gt_mask = _apply_finger_occlusion(scene, gt_mask, quad_gt, rng)
            tags.append("occlusion")
            stats["occlusion"] += 1

        scene, photo_tags = _apply_scene_shadows_and_glare(
            scene,
            [quad_gt],
            rng,
            cast_shadow_prob=0.62,
            lighting_grad_prob=0.55,
            border_glare_prob=0.42,
        )
        for pt in photo_tags:
            tags.append(pt)
            if pt in stats:
                stats[pt] += 1

        noisy_obb = _perturb_stage1_obb(quad_gt, rng)
        canon_bgr, M_canon, _ = warp_to_canonical_obb(
            scene,
            noisy_obb,
            canon_size=(CANON_H, CANON_W),
            target_size=(TARGET_H, TARGET_W),
        )
        canon_mask = cv2.warpPerspective(
            gt_mask,
            M_canon.astype(np.float32),
            (CANON_W, CANON_H),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        corners_canon = cv2.perspectiveTransform(
            quad_gt.astype(np.float32).reshape(1, 4, 2),
            M_canon.astype(np.float32),
        ).reshape(4, 2)

        stem = f"canon_{idx:05d}"
        cv2.imwrite(str(out_dir / "images" / split_name / f"{stem}.jpg"), canon_bgr, [cv2.IMWRITE_JPEG_QUALITY, 94])
        cv2.imwrite(str(out_dir / "masks" / split_name / f"{stem}.png"), canon_mask)
        (out_dir / "meta" / split_name / f"{stem}.json").write_text(
            json.dumps({"corners_canon": corners_canon.tolist(), "tags": tags})
        )

        if split_name == "train" and len(sample_vis) < 6:
            sample_vis.append((stem, canon_bgr.copy(), canon_mask.copy(), corners_canon.copy()))

    if sample_vis:
        fig, axes = plt.subplots(2, 3, figsize=(12, 10))
        for ax, (stem, img_b, m_u8, c_pts) in zip(axes.flat, sample_vis):
            vis = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB).copy()
            cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 230, 120), 2)
            cv2.polylines(vis, [c_pts.astype(np.int32).reshape(-1, 1, 2)], True, (255, 220, 0), 2)
            ax.imshow(vis)
            ax.set_title(stem, fontsize=9)
            ax.axis("off")
        fig.suptitle("Stage-2 Canonical Refiner Samples (green=GT mask, yellow=true quad)", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / "canon_samples.png", dpi=140)
        plt.close(fig)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage-2 Canonical Refiner dataset (768x544).")
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("canon_refiner_dataset_prep", cfg)
    rng = random.Random(cfg["project"]["seed"] + 1337)

    ou_ds = cfg["detector"]["obb_unet"]["dataset"]
    n_train = args.train_samples or int(ou_ds.get("n_train", 3500))
    n_val = args.val_samples or int(ou_ds.get("n_val", 500))
    n_test = args.test_samples or int(ou_ds.get("n_test", 500))

    out_dir = (PROJECT_ROOT / cfg["paths"]["datasets_dir"] / ou_ds["name"]).resolve()
    card_pools, backgrounds = _sample_canonical_card_pools(cfg)

    all_stats = {}
    for split_name, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        logger.info("Generating %d canonical crops for %s split...", count, split_name)
        all_stats[split_name] = generate_split(
            split_name=split_name,
            n_samples=count,
            card_pool=card_pools[split_name],
            backgrounds=backgrounds,
            out_dir=out_dir,
            rng=rng,
            corner_radius_frac=float(cfg["segmentation"]["dataset"].get("corner_radius_frac", 0.045)),
        )
        exp.log_metric(f"samples.{split_name}", count)
        exp.log_metric(f"stress.{split_name}", all_stats[split_name])

    vis_path = out_dir / "canon_samples.png"
    if vis_path.exists():
        shutil_dst = exp.plots_dir / "canon_samples.png"
        shutil_dst.write_bytes(vis_path.read_bytes())

    logger.info("Canonical dataset ready → %s", out_dir)
    exp.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    main()

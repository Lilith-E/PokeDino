"""Dual-label synthetic dataset generator for card segmentation and perspective benchmarking.

Generates photorealistic 640x640 scenes containing 0-2 Pokémon TCG cards with:
- Rounded-corner card alpha masks (applied consistently to 3-channel and 4-channel scans)
- Controlled 3D perspective homography tilts and 0-360 deg in-plane rotations
- Synthetic protective sleeves/toploaders (outer translucent border + specular sheen)
- Synthetic finger/edge occlusions subtracted from the ground-truth card mask
- Paired outputs: YOLO-Seg polygon labels, semantic pixel masks, and exact unrounded
  perspective corner ground-truth metadata for corner-error benchmarking.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.experiment import Experiment, load_config, set_seed  # noqa: E402
from backend.masking import rounded_rect_mask  # noqa: E402
from scripts.prepare_dataset import (  # noqa: E402
    apply_damage_augmentations,
    corners_to_yolo_obb,
    get_available_backgrounds,
    is_card_fully_inside,
    load_background,
    load_card_rgba,
)

logger = logging.getLogger(__name__)

IMG_W, IMG_H = 640, 640
SAFE_MARGIN = 16
CLASS_ID = 0


def ensure_rounded_alpha(card_bgra: np.ndarray, corner_radius_frac: float = 0.045) -> np.ndarray:
    """Apply a crisp rounded-corner alpha mask if the card scan has no transparency."""
    out = card_bgra.copy()
    h, w = out.shape[:2]
    alpha = out[:, :, 3]
    corners_opaque = (
        alpha[0, 0] > 200
        and alpha[0, w - 1] > 200
        and alpha[h - 1, 0] > 200
        and alpha[h - 1, w - 1] > 200
    )
    if corners_opaque:
        rmask = rounded_rect_mask(
            h, w, corner_radius_frac=corner_radius_frac, erode_frac=0.0, feather_px=0
        )
        out[:, :, 3] = np.clip(alpha.astype(np.float32) * rmask, 0, 255).astype(np.uint8)
    return out


def _make_full_art_borderless(
    card_rgba: np.ndarray,
    corner_radius_frac: float,
    rng: random.Random,
) -> np.ndarray:
    """Create a borderless full-art variant where artwork bleeds to rounded card edges."""
    h, w = card_rgba.shape[:2]
    crop_x = int(round(w * rng.uniform(0.055, 0.085)))
    crop_y = int(round(h * rng.uniform(0.045, 0.070)))
    crop_x = min(crop_x, max(1, w // 6))
    crop_y = min(crop_y, max(1, h // 6))

    inner_rgb = card_rgba[crop_y : h - crop_y, crop_x : w - crop_x, :3]
    bled_rgb = cv2.resize(inner_rgb, (w, h), interpolation=cv2.INTER_CUBIC)

    rmask = rounded_rect_mask(
        h, w, corner_radius_frac=corner_radius_frac, erode_frac=0.0, feather_px=0
    )
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = bled_rgb
    out[:, :, 3] = np.clip(np.round(rmask * 255.0), 0, 255).astype(np.uint8)
    return out


def _sample_perspective_quad(
    w_card: int,
    h_card: int,
    scene_w: int,
    scene_h: int,
    rng: random.Random,
    existing_corners: list[np.ndarray],
    apply_perspective: bool,
    max_tilt: float,
    tight_framing: bool = False,
    max_retries: int = 25,
) -> Optional[np.ndarray]:
    """Sample a valid 4x2 perspective destination quad inside the scene."""
    margin = 4 if tight_framing else SAFE_MARGIN
    if tight_framing:
        min_long = int(0.70 * min(scene_h, scene_w))
        max_long = int(0.93 * min(scene_h, scene_w))
    else:
        min_long = int(0.42 * min(scene_h, scene_w))
        max_long = int(0.68 * min(scene_h, scene_w))

    for _ in range(max_retries):
        target_long = rng.randint(min_long, max_long)
        scale = target_long / max(h_card, w_card)
        nw, nh = w_card * scale, h_card * scale

        angle = rng.uniform(0.0, 360.0)
        theta = np.radians(angle)
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=np.float64,
        )
        local = np.array(
            [[-nw / 2, -nh / 2], [nw / 2, -nh / 2], [nw / 2, nh / 2], [-nw / 2, nh / 2]],
            dtype=np.float64,
        )
        rotated = local @ rot.T

        if apply_perspective:
            tilt_x = rng.uniform(-max_tilt, max_tilt) * nw
            tilt_y = rng.uniform(-max_tilt, max_tilt) * nh
            persp_local = local.copy()
            # Symmetric trapezoidal foreshortening + mild independent corner jitter
            side = rng.choice(["top", "bottom", "left", "right"])
            if side == "top":
                persp_local[0, 0] += abs(tilt_x) * 0.5
                persp_local[1, 0] -= abs(tilt_x) * 0.5
            elif side == "bottom":
                persp_local[3, 0] += abs(tilt_x) * 0.5
                persp_local[2, 0] -= abs(tilt_x) * 0.5
            elif side == "left":
                persp_local[0, 1] += abs(tilt_y) * 0.5
                persp_local[3, 1] -= abs(tilt_y) * 0.5
            else:
                persp_local[1, 1] += abs(tilt_y) * 0.5
                persp_local[2, 1] -= abs(tilt_y) * 0.5

            for k in range(4):
                persp_local[k, 0] += rng.uniform(-0.04, 0.04) * nw
                persp_local[k, 1] += rng.uniform(-0.04, 0.04) * nh
            rotated = persp_local @ rot.T

        min_x, min_y = rotated[:, 0].min(), rotated[:, 1].min()
        max_x, max_y = rotated[:, 0].max(), rotated[:, 1].max()
        span_x, span_y = max_x - min_x, max_y - min_y
        if span_x >= scene_w - 2 * margin or span_y >= scene_h - 2 * margin:
            continue

        cx = rng.uniform(margin - min_x, scene_w - margin - max_x)
        cy = rng.uniform(margin - min_y, scene_h - margin - max_y)
        quad = rotated + np.array([cx, cy], dtype=np.float64)

        if not is_card_fully_inside(quad, scene_w, scene_h, margin=margin):
            continue
        if not cv2.isContourConvex(quad.astype(np.float32)):
            continue

        rect = cv2.minAreaRect(quad.astype(np.float32))
        rw, rh = rect[1]
        if min(rw, rh) < 40:
            continue
        ar = max(rw, rh) / max(min(rw, rh), 1e-3)
        if not (1.15 <= ar <= 1.65):
            continue

        if existing_corners:
            poly_new = quad.astype(np.float32)
            overlaps = False
            for exist in existing_corners:
                ret, _ = cv2.rotatedRectangleIntersection(
                    cv2.minAreaRect(poly_new), cv2.minAreaRect(exist.astype(np.float32))
                )
                if ret != cv2.INTERSECT_NONE:
                    overlaps = True
                    break
            if overlaps:
                continue

        return quad
    return None


def _draw_synthetic_sleeve(scene: np.ndarray, quad: np.ndarray, rng: random.Random) -> np.ndarray:
    """Render a translucent protective sleeve frame and mild specular streak around the card."""
    out = scene.copy()
    ctr = quad.mean(axis=0)
    sleeve_quad = (ctr + (quad - ctr) * rng.uniform(1.04, 1.09)).astype(np.int32)

    overlay = out.copy()
    cv2.fillPoly(overlay, [sleeve_quad], (235, 240, 245))
    alpha_fill = rng.uniform(0.06, 0.14)
    out = cv2.addWeighted(overlay, alpha_fill, out, 1.0 - alpha_fill, 0.0)

    border_overlay = out.copy()
    cv2.polylines(border_overlay, [sleeve_quad], True, (245, 248, 252), thickness=rng.choice([2, 3]))
    out = cv2.addWeighted(border_overlay, rng.uniform(0.35, 0.65), out, 1.0 - 0.5, 0.0)

    # Diagonal plastic sheen across card + sleeve
    sheen = np.zeros(out.shape[:2], dtype=np.float32)
    p0 = sleeve_quad[rng.randint(0, 3)]
    p1 = (p0 + np.array([rng.randint(-180, 180), rng.randint(-180, 180)])).astype(np.int32)
    cv2.line(sheen, tuple(p0), tuple(p1), 1.0, thickness=rng.randint(18, 42))
    sheen = cv2.GaussianBlur(sheen, (0, 0), 14.0)
    sheen_3 = sheen[:, :, None] * rng.uniform(0.18, 0.35)
    out = np.clip(out.astype(np.float32) * (1.0 - sheen_3) + 255.0 * sheen_3, 0, 255).astype(np.uint8)
    return out


def _apply_finger_occlusion(
    scene: np.ndarray,
    card_mask: np.ndarray,
    quad: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite a realistic finger tip overlapping one card edge and subtract from mask."""
    out = scene.copy()
    mask_out = card_mask.copy()
    h_img, w_img = out.shape[:2]

    side_idx = rng.randint(0, 3)
    p_a = quad[side_idx]
    p_b = quad[(side_idx + 1) % 4]
    edge = p_b - p_a
    edge_len = float(np.linalg.norm(edge))
    if edge_len < 40:
        return out, mask_out

    ctr = quad.mean(axis=0)
    mid = p_a + rng.uniform(0.28, 0.72) * edge
    inward = ctr - mid
    inward_norm = float(np.linalg.norm(inward))
    if inward_norm < 1e-3:
        return out, mask_out
    inward /= inward_norm

    finger_w = int(rng.uniform(0.09, 0.15) * edge_len)
    finger_l = int(rng.uniform(0.22, 0.38) * edge_len)
    tip_center = mid + inward * (finger_l * 0.25)
    outside_anchor = mid - inward * (finger_l * 0.85)
    angle_deg = float(np.degrees(np.arctan2(inward[1], inward[0])))

    finger_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.ellipse(
        finger_mask,
        (int(tip_center[0]), int(tip_center[1])),
        (max(8, finger_l // 2), max(6, finger_w // 2)),
        angle_deg,
        0,
        360,
        255,
        -1,
    )
    cv2.line(
        finger_mask,
        (int(tip_center[0]), int(tip_center[1])),
        (int(outside_anchor[0]), int(outside_anchor[1])),
        255,
        thickness=max(8, finger_w),
    )

    # Soft shadow cast onto card
    shadow = cv2.GaussianBlur((finger_mask > 0).astype(np.float32), (0, 0), 6.0)
    shadow = np.clip(shadow - (finger_mask > 0).astype(np.float32), 0.0, 1.0)[:, :, None]
    out = np.clip(out.astype(np.float32) * (1.0 - 0.35 * shadow), 0, 255).astype(np.uint8)

    # Skin-tone gradient fill with mild shading
    skin_tones_bgr = [
        (145, 178, 222),
        (118, 152, 198),
        (88, 118, 165),
        (162, 192, 232),
    ]
    base_skin = np.array(rng.choice(skin_tones_bgr), dtype=np.float32)
    dist_in = cv2.distanceTransform(finger_mask, cv2.DIST_L2, 5)
    max_d = max(float(dist_in.max()), 1.0)
    shading = 0.72 + 0.34 * np.clip(dist_in / max_d, 0.0, 1.0)
    finger_rgb = np.clip(base_skin[None, None, :] * shading[:, :, None], 0, 255)

    alpha_f = cv2.GaussianBlur((finger_mask > 0).astype(np.float32), (3, 3), 0)[:, :, None]
    out = np.clip(finger_rgb * alpha_f + out.astype(np.float32) * (1.0 - alpha_f), 0, 255).astype(np.uint8)
    mask_out[finger_mask > 127] = 0
    return out, mask_out


def _apply_scene_shadows_and_glare(
    scene_bgr: np.ndarray,
    card_quads: list[np.ndarray],
    rng: random.Random,
    cast_shadow_prob: float = 0.58,
    lighting_grad_prob: float = 0.50,
    border_glare_prob: float = 0.38,
) -> tuple[np.ndarray, list[str]]:
    """Apply realistic smartphone/hand cast shadows, lighting gradients, and rim glare.

    Executed at scene level after compositing so shadows and highlights cross both
    the card borders and the background table naturally, simulating low-contrast rims.
    """
    h, w = scene_bgr.shape[:2]
    out = scene_bgr.astype(np.float32)
    tags: list[str] = []

    if rng.random() < lighting_grad_prob:
        ang = rng.uniform(0.0, 2.0 * np.pi)
        dx, dy = float(np.cos(ang)), float(np.sin(ang))
        yg, xg = np.mgrid[:h, :w]
        proj = ((xg / w) - 0.5) * dx + ((yg / h) - 0.5) * dy
        proj = (proj - proj.min()) / max(float(proj.max() - proj.min()), 1e-5)
        lo = rng.uniform(0.58, 0.80)
        hi = rng.uniform(1.00, 1.18)
        gain = lo + (hi - lo) * proj
        out *= gain[:, :, None]

    if rng.random() < cast_shadow_prob:
        tags.append("shadow")
        shadow_map = np.zeros((h, w), dtype=np.float32)
        n_shadows = 2 if rng.random() < 0.35 else 1
        for _ in range(n_shadows):
            s_mask = np.zeros((h, w), dtype=np.uint8)
            if card_quads and rng.random() < 0.75:
                q = rng.choice(card_quads)
                cx = float(q[:, 0].mean()) + rng.uniform(-0.45, 0.45) * w * 0.35
                cy = float(q[:, 1].mean()) + rng.uniform(-0.45, 0.45) * h * 0.35
            else:
                cx = rng.uniform(0.1, 0.9) * w
                cy = rng.uniform(0.1, 0.9) * h

            # Phone/hand silhouette or half-plane shadow edge crossing card borders
            if rng.random() < 0.65:
                rect_w = rng.uniform(0.28, 0.72) * w
                rect_h = rng.uniform(0.32, 0.85) * h
                rect_ang = rng.uniform(0.0, 180.0)
                box = cv2.boxPoints(((cx, cy), (rect_w, rect_h), rect_ang)).astype(np.int32)
                cv2.fillPoly(s_mask, [box], 255)
            else:
                ang = rng.uniform(0.0, 2.0 * np.pi)
                nx, ny = np.cos(ang), np.sin(ang)
                yg, xg = np.mgrid[:h, :w]
                half_plane = ((xg - cx) * nx + (yg - cy) * ny) > 0
                s_mask[half_plane] = 255

            sigma = rng.uniform(6.0, 26.0)
            s_soft = cv2.GaussianBlur(s_mask.astype(np.float32) / 255.0, (0, 0), sigma)
            strength = rng.uniform(0.32, 0.65)
            shadow_map = np.maximum(shadow_map, s_soft * strength)

        # Slightly cooler RGB cast inside deep shadows
        tint = np.array(
            [rng.uniform(0.98, 1.05), 1.0, rng.uniform(0.92, 0.99)], dtype=np.float32
        )
        attn = 1.0 - shadow_map[:, :, None] * tint[None, None, :]
        out *= np.clip(attn, 0.25, 1.05)

    if rng.random() < border_glare_prob:
        tags.append("glare")
        g_mask = np.zeros((h, w), dtype=np.float32)
        if card_quads:
            q = rng.choice(card_quads)
            edge_pt = q[rng.randint(0, 3)]
            gx = float(edge_pt[0]) + rng.uniform(-25.0, 25.0)
            gy = float(edge_pt[1]) + rng.uniform(-25.0, 25.0)
        else:
            gx, gy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        rad_x = int(rng.uniform(0.08, 0.28) * w)
        rad_y = int(rng.uniform(0.04, 0.14) * h)
        ang_deg = rng.uniform(0.0, 180.0)
        tmp = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(tmp, (int(gx), int(gy)), (rad_x, rad_y), ang_deg, 0, 360, 255, -1)
        g_mask = cv2.GaussianBlur(tmp.astype(np.float32) / 255.0, (0, 0), rng.uniform(10.0, 24.0))
        glare_alpha = rng.uniform(0.14, 0.36) * g_mask[:, :, None]
        glare_color = np.array([255.0, 250.0, 244.0], dtype=np.float32)[None, None, :]
        out = out * (1.0 - glare_alpha) + glare_color * glare_alpha

    # Subtle ambient color balance shift
    if rng.random() < 0.45:
        b_gain = rng.uniform(0.90, 1.10)
        r_gain = rng.uniform(0.90, 1.10)
        out[:, :, 0] *= b_gain
        out[:, :, 2] *= r_gain

    return np.clip(out, 0, 255).astype(np.uint8), tags


def mask_to_yolo_seg_line(
    instance_mask: np.ndarray,
    img_w: int,
    img_h: int,
    class_id: int = CLASS_ID,
) -> Optional[str]:
    """Convert a single-instance binary mask into a normalized YOLO-Seg polygon line."""
    cnts, _ = cv2.findContours((instance_mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < 300:
        return None
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon=max(0.8, 0.0018 * peri), closed=True)
    pts = approx.reshape(-1, 2).astype(np.float64)
    if len(pts) < 6:
        pts = largest.reshape(-1, 2).astype(np.float64)
        if len(pts) > 48:
            idx = np.linspace(0, len(pts) - 1, 48, dtype=int)
            pts = pts[idx]
    if len(pts) < 4:
        return None
    pts[:, 0] = np.clip(pts[:, 0] / img_w, 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / img_h, 0.0, 1.0)
    coords = " ".join(f"{v:.6f}" for v in pts.flatten())
    return f"{class_id} {coords}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dual-label card segmentation dataset.")
    parser.add_argument("--train-scenes", type=int, default=3000)
    parser.add_argument("--val-scenes", type=int, default=500)
    parser.add_argument("--test-scenes", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("seg_dataset_prep", cfg)

    seg_cfg = cfg["segmentation"]["dataset"]
    out_dir = (PROJECT_ROOT / cfg["paths"]["datasets_dir"] / seg_cfg["name"]).resolve()
    splits = [("train", args.train_scenes), ("val", args.val_scenes), ("test", args.test_scenes)]

    for split_name, _ in splits:
        for sub in ("images", "labels", "labels_obb", "masks", "meta"):
            d = out_dir / sub / split_name
            d.mkdir(parents=True, exist_ok=True)
            for f in d.glob("*"):
                if f.is_file():
                    f.unlink()

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
        s_ratios = seg_cfg.get("split", [0.70, 0.15, 0.15])
        n_tr = int(len(leg) * s_ratios[0])
        n_va = int(len(leg) * s_ratios[1])
        card_pools["train"].extend(leg[:n_tr])
        card_pools["val"].extend(leg[n_tr : n_tr + n_va])
        card_pools["test"].extend(leg[n_tr + n_va :])

    bg_paths = get_available_backgrounds(min_needed=120, seed=cfg["project"]["seed"])
    hard_neg_ratio = float(seg_cfg.get("hard_negative_ratio", 0.06))
    persp_prob = float(seg_cfg.get("perspective_prob", 0.70))
    max_tilt = float(seg_cfg.get("perspective_max_tilt", 0.18))
    occ_prob = float(seg_cfg.get("occlusion_prob", 0.35))
    sleeve_prob = float(seg_cfg.get("sleeve_prob", 0.25))
    full_art_prob = float(seg_cfg.get("full_art_prob", 0.25))
    tight_framing_prob = float(seg_cfg.get("tight_framing_prob", 0.25))
    corner_rad = float(seg_cfg.get("corner_radius_frac", 0.045))

    stats = {"scenes": {}, "stress_counts": {}}
    global_id = 0

    for split_name, n_scenes in splits:
        pool = card_pools[split_name]
        if not pool:
            exp.logger.error(f"No card scans available for split '{split_name}'")
            sys.exit(1)

        rng = random.Random(cfg["project"]["seed"] + sum(ord(c) for c in split_name))
        split_stress = {
            "perspective": 0,
            "occlusion": 0,
            "sleeve": 0,
            "shadow": 0,
            "glare": 0,
            "multi_card": 0,
            "hard_negative": 0,
        }

        for i in range(n_scenes):
            scene = load_background(rng.choice(bg_paths), rng)
            semantic_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
            seg_lines: list[str] = []
            obb_lines: list[str] = []
            card_metas: list[dict] = []
            existing_quads: list[np.ndarray] = []

            if rng.random() < hard_neg_ratio:
                split_stress["hard_negative"] += 1
            else:
                n_cards = 2 if rng.random() < 0.12 else 1
                if n_cards > 1:
                    split_stress["multi_card"] += 1

                use_tight = n_cards == 1 and (rng.random() < tight_framing_prob)
                for _ in range(n_cards):
                    c_path = rng.choice(pool)
                    raw_card = ensure_rounded_alpha(load_card_rgba(c_path), corner_radius_frac=corner_rad)
                    if rng.random() < full_art_prob:
                        raw_card = _make_full_art_borderless(raw_card, corner_rad, rng)
                    damaged = apply_damage_augmentations(raw_card, rng)
                    ch, cw = damaged.shape[:2]

                    use_persp = rng.random() < persp_prob
                    quad = _sample_perspective_quad(
                        cw, ch, IMG_W, IMG_H, rng, existing_quads,
                        apply_perspective=use_persp, max_tilt=max_tilt,
                        tight_framing=use_tight,
                    )
                    if quad is None:
                        continue

                    src_pts = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], dtype=np.float32)
                    M = cv2.getPerspectiveTransform(src_pts, quad.astype(np.float32))
                    warped_bgra = cv2.warpPerspective(
                        damaged, M, (IMG_W, IMG_H),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
                    )
                    alpha = warped_bgra[:, :, 3:4].astype(np.float32) / 255.0
                    scene = np.clip(
                        warped_bgra[:, :, :3].astype(np.float32) * alpha + scene.astype(np.float32) * (1.0 - alpha),
                        0, 255,
                    ).astype(np.uint8)

                    inst_mask = (warped_bgra[:, :, 3] > 127).astype(np.uint8) * 255
                    tags = []
                    if use_persp:
                        tags.append("perspective")
                        split_stress["perspective"] += 1
                    if rng.random() < sleeve_prob:
                        scene = _draw_synthetic_sleeve(scene, quad, rng)
                        tags.append("sleeve")
                        split_stress["sleeve"] += 1
                    if rng.random() < occ_prob:
                        scene, inst_mask = _apply_finger_occlusion(scene, inst_mask, quad, rng)
                        tags.append("occlusion")
                        split_stress["occlusion"] += 1

                    line = mask_to_yolo_seg_line(inst_mask, IMG_W, IMG_H, CLASS_ID)
                    if line is None:
                        continue
                    seg_lines.append(line)
                    obb_lines.append(corners_to_yolo_obb(quad, IMG_W, IMG_H))
                    semantic_mask = np.maximum(semantic_mask, inst_mask)
                    existing_quads.append(quad)
                    card_metas.append({
                        "card_file": c_path.name,
                        "corners_gt": quad.tolist(),
                        "tags": tags,
                    })

            scene, illum_tags = _apply_scene_shadows_and_glare(scene, existing_quads, rng)
            for itag in illum_tags:
                split_stress[itag] += 1

            name = f"scene_{global_id:05d}"
            cv2.imwrite(str(out_dir / "images" / split_name / f"{name}.jpg"), scene, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(out_dir / "masks" / split_name / f"{name}.png"), semantic_mask)
            (out_dir / "labels" / split_name / f"{name}.txt").write_text(
                "\n".join(seg_lines) + "\n" if seg_lines else ""
            )
            (out_dir / "labels_obb" / split_name / f"{name}.txt").write_text(
                "\n".join(obb_lines) + "\n" if obb_lines else ""
            )
            (out_dir / "meta" / split_name / f"{name}.json").write_text(
                json.dumps({"scene": name, "cards": card_metas, "scene_tags": illum_tags}, indent=2)
            )
            global_id += 1
            if (i + 1) % 500 == 0 or (i + 1) == n_scenes:
                exp.logger.info(f"[{split_name}] {i + 1}/{n_scenes} scenes generated")

        stats["scenes"][split_name] = n_scenes
        stats["stress_counts"][split_name] = split_stress
        exp.log_metric(f"scenes.{split_name}", n_scenes)
        exp.log_metrics(split_stress, prefix=f"stress.{split_name}.")

    yaml_path = out_dir / "pokemon_seg.yaml"
    yaml_path.write_text(
        f"# Auto-generated by scripts/prepare_seg_dataset.py\n"
        f"path: {out_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n  0: card\n"
    )

    sample_files = sorted((out_dir / "images" / "train").glob("*.jpg"))[:6]
    fig, axs = plt.subplots(2, 3, figsize=(13, 9))
    for ax, img_path in zip(axs.flat, sample_files):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(out_dir / "masks" / "train" / f"{img_path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        meta = json.loads((out_dir / "meta" / "train" / f"{img_path.stem}.json").read_text())
        if mask is not None:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0, 230, 150), 2)
        for c in meta["cards"]:
            pts = np.array(c["corners_gt"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (255, 215, 0), 1)
        ax.imshow(img)
        ax.set_title(f"{img_path.name} ({len(meta['cards'])} cards)", fontsize=8)
        ax.axis("off")
    fig.suptitle("Dual-Label Segmentation Dataset Samples (green=GT mask, yellow=GT quad)")
    exp.save_plot(fig, "seg_dataset_samples")
    exp.finish()
    exp.logger.info(f"DONE. Segmentation dataset at {out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

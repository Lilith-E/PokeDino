"""Synthetic dataset generation for YOLO OBB training.

Costruisce scene fotorealistiche componendo le carte su sfondi eterogenei:
1. Composizione di 1-3 carte con rotazione continua (0-360°), scalatura e traslazione.
2. Calculation of oriented bounding box vertices in normalized YOLO OBB format.
3. Suddivisione deterministica in partizioni di train, val e test con relativo file YAML.
4. Generation of dataset statistics and visual audit samples.
"""

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

logger = logging.getLogger(__name__)

IMG_W, IMG_H = 640, 640
BG_CACHE = PROJECT_ROOT / "data" / "datasets" / "backgrounds"
CLASS_ID = 0

def get_available_backgrounds(min_needed: int = 120, seed: int = 42) -> list[Path]:
    """Discover all cached backgrounds from DTD, CC0 PBR, and curated subfolders.

    If none or fewer than min_needed are found, automatically downloads fallback
    backgrounds via download_backgrounds.
    """
    BG_CACHE.mkdir(parents=True, exist_ok=True)
    existing = sorted(list(BG_CACHE.rglob("*.jpg")) + list(BG_CACHE.rglob("*.png")))
    if len(existing) >= min_needed:
        return existing

    logger.info(f"Only {len(existing)} backgrounds found in cache; downloading up to {min_needed}...")
    return download_backgrounds(min_needed, seed)

def download_backgrounds(n: int, seed: int) -> list[Path]:
    """Download n background photos from picsum.photos (cached on disk)."""
    BG_CACHE.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths: list[Path] = []
    for i in range(n):
        out = BG_CACHE / f"bg_{i:04d}.jpg"
        if not out.exists():
            img_id = rng.randint(1, 1000)
            url = f"https://picsum.photos/id/{img_id}/{IMG_W}/{IMG_H}"
            try:
                import requests

                r = requests.get(url, timeout=20)
                r.raise_for_status()
                out.write_bytes(r.content)
            except Exception as e:
                logger.warning(f"Background {i} download failed ({e}); using procedural fallback")
                _make_procedural_background(out, rng)
        paths.append(out)
    return paths

def _make_procedural_background(out: Path, rng: random.Random):
    """Fallback: generate a cluttered procedural background (wood/fabric noise)."""
    base = np.full((IMG_H, IMG_W, 3), 0, np.uint8)
    c1 = np.array([rng.randint(40, 200) for _ in range(3)], np.float32)
    c2 = np.array([rng.randint(40, 200) for _ in range(3)], np.float32)
    gx, gy = np.meshgrid(np.linspace(0, 1, IMG_W), np.linspace(0, 1, IMG_H))
    t = (gx * rng.uniform(0.5, 1.5) + gy * rng.uniform(0.5, 1.5))[..., None]
    base = (c1 * (1 - t) + c2 * t).astype(np.uint8)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 18, base.shape).astype(np.uint8)
    base = cv2.add(base, noise)
    cv2.imwrite(str(out), base)

def load_background(path: Path, rng: random.Random) -> np.ndarray:
    """Load a background, resize to scene size, apply mild photometric jitter."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read background {path}")
    img = cv2.resize(img, (IMG_W, IMG_H))
    alpha = rng.uniform(0.75, 1.25)
    beta = rng.uniform(-25, 25)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return img

def load_card_rgba(path: Path) -> np.ndarray:
    """Load a card scan as BGRA."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read card {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

def apply_damage_augmentations(card_bgra: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Applies realistic physical alterations and defects to card prior to compositing:
    - Pieghe marcate e crepe superficiali (bordo scuro + fibra bianca del cartoncino spezzato)
    - Graffi lineari sottili (hairline scratches su olografia e box testo)
    - Perimeter wear (edge whitening / silvering and corner abrasion)
    - Riflessi / hotspot di luce da flash smartphone e lampade da tavolo
    - Gradienti di ombra asimmetrici
    """
    card = card_bgra.copy()
    h, w = card.shape[:2]

    if rng.random() < 0.50:
        n_creases = rng.randint(1, 3)
        for _ in range(n_creases):
            crease_type = rng.choice(["horizontal", "diagonal", "random"])
            if crease_type == "horizontal":
                y = rng.randint(int(h * 0.25), int(h * 0.75))
                pt1 = (0, y + rng.randint(-8, 8))
                pt2 = (w - 1, y + rng.randint(-8, 8))
            elif crease_type == "diagonal":
                pt1 = (0, rng.randint(0, h // 2))
                pt2 = (w - 1, rng.randint(h // 2, h - 1))
            else:
                pt1 = (rng.randint(0, w // 3), rng.randint(0, h - 1))
                pt2 = (rng.randint(2 * w // 3, w - 1), rng.randint(0, h - 1))

            thickness = rng.randint(2, 4)
            cv2.line(card, pt1, pt2, (20, 20, 20, 255), thickness)

            offset = rng.choice([-1, 1])
            pt1_w = (pt1[0], max(0, min(h - 1, pt1[1] + offset)))
            pt2_w = (pt2[0], max(0, min(h - 1, pt2[1] + offset)))
            cv2.line(card, pt1_w, pt2_w, (230, 230, 230, 255), max(1, thickness - 1))

    if rng.random() < 0.45:
        n_scratches = rng.randint(3, 8)
        for _ in range(n_scratches):
            sx1 = rng.randint(int(w * 0.1), int(w * 0.9))
            sy1 = rng.randint(int(h * 0.1), int(h * 0.9))
            length = rng.randint(20, 90)
            ang = rng.uniform(0, 2 * np.pi)
            sx2 = int(sx1 + length * np.cos(ang))
            sy2 = int(sy1 + length * np.sin(ang))
            sx2 = max(0, min(w - 1, sx2))
            sy2 = max(0, min(h - 1, sy2))
            color_val = rng.randint(210, 255)
            cv2.line(card, (sx1, sy1), (sx2, sy2), (color_val, color_val, color_val, 255), 1)

    if rng.random() < 0.55:
        n_whitening = rng.randint(15, 45)
        for _ in range(n_whitening):
            side = rng.choice(["top", "bottom", "left", "right", "corner"])
            if side == "top":
                px = rng.randint(0, w - 1)
                py = rng.randint(0, 8)
            elif side == "bottom":
                px = rng.randint(0, w - 1)
                py = rng.randint(h - 9, h - 1)
            elif side == "left":
                px = rng.randint(0, 8)
                py = rng.randint(0, h - 1)
            elif side == "right":
                px = rng.randint(w - 9, w - 1)
                py = rng.randint(0, h - 1)
            else:
                px = rng.choice([rng.randint(0, 15), rng.randint(w - 16, w - 1)])
                py = rng.choice([rng.randint(0, 15), rng.randint(h - 16, h - 1)])

            rad = rng.randint(1, 4)
            cv2.circle(card, (px, py), rad, (235, 235, 235, 255), -1)

    if rng.random() < 0.40:
        gx = rng.randint(int(w * 0.2), int(w * 0.8))
        gy = rng.randint(int(h * 0.2), int(h * 0.8))
        radius = rng.randint(int(min(w, h) * 0.15), int(min(w, h) * 0.45))
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_sq = (x_grid - gx) ** 2 + (y_grid - gy) ** 2
        hotspot_mask = np.exp(-dist_sq / (2.0 * (radius ** 2))).astype(np.float32)
        boost = rng.uniform(40, 110)
        card_bgr = card[:, :, :3].astype(np.float32)
        card_bgr += hotspot_mask[..., None] * boost
        card[:, :, :3] = np.clip(card_bgr, 0, 255).astype(np.uint8)

    if rng.random() < 0.45:
        dir_x = rng.uniform(-1, 1)
        dir_y = rng.uniform(-1, 1)
        norm = np.hypot(dir_x, dir_y) + 1e-6
        dir_x /= norm
        dir_y /= norm
        y_grid, x_grid = np.mgrid[:h, :w]
        proj = (x_grid / w - 0.5) * dir_x + (y_grid / h - 0.5) * dir_y
        proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
        factor = 0.55 + 0.45 * proj
        card_bgr = card[:, :, :3].astype(np.float32) * factor[..., None]
        card[:, :, :3] = np.clip(card_bgr, 0, 255).astype(np.uint8)

    if rng.random() < 0.30:
        k = rng.choice([3, 5])
        card[:, :, :3] = cv2.GaussianBlur(card[:, :, :3], (k, k), 0)

    return card

SAFE_MARGIN = 16

def is_card_fully_inside(corners: np.ndarray, W: int, H: int, margin: int = SAFE_MARGIN) -> bool:
    """Verify that all 4 corners of the card are strictly within [margin, dim - margin]."""
    if np.any(corners[:, 0] < margin) or np.any(corners[:, 0] > (W - margin)):
        return False
    if np.any(corners[:, 1] < margin) or np.any(corners[:, 1] > (H - margin)):
        return False
    area = cv2.contourArea(corners.astype(np.float32))
    if area < (W * H * 0.04):
        return False
    return True

def composite_card(
    scene: np.ndarray,
    card_bgra: np.ndarray,
    rng: random.Random,
    existing_corners: list[np.ndarray] | None = None,
    max_retries: int = 25,
    margin: int = SAFE_MARGIN,
    angle_range: tuple[float, float] = (0.0, 360.0),
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Composite one card onto the scene with GUARANTEED 100% full visibility,
    zero occlusion, and safety margin from image edges.

    Returns:
        (updated_scene, corners) where corners is a (4,2) float array of the card's
        rotated corner coordinates in scene pixel space, or (scene, None) if placement fails.
    """
    H, W = scene.shape[:2]
    ch, cw = card_bgra.shape[:2]

    min_allowed_long = int(0.42 * min(H, W))
    max_allowed_long = int(0.66 * min(H, W))

    for _ in range(max_retries):
        target_long = rng.randint(min_allowed_long, max_allowed_long)
        scale = target_long / max(ch, cw)
        nw, nh = max(2, int(cw * scale)), max(2, int(ch * scale))
        card = cv2.resize(card_bgra, (nw, nh), interpolation=cv2.INTER_AREA)

        angle = rng.uniform(angle_range[0], angle_range[1])
        cx, cy = nw / 2.0, nh / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        rw = int(nh * sin + nw * cos)
        rh = int(nh * cos + nw * sin)

        if rw >= (W - 2 * margin) or rh >= (H - 2 * margin):
            continue

        M[0, 2] += rw / 2.0 - cx
        M[1, 2] += rh / 2.0 - cy

        px = rng.randint(margin, W - rw - margin)
        py = rng.randint(margin, H - rh - margin)

        corners_local = np.array([[0, 0], [nw, 0], [nw, nh], [0, nh]], np.float64)
        ones = np.ones((4, 1))
        corners_rot = (M @ np.hstack([corners_local, ones]).T).T
        corners_scene = corners_rot[:, :2] + np.array([px, py], np.float64)

        if not is_card_fully_inside(corners_scene, W, H, margin=margin):
            continue

        if existing_corners:
            poly_new = corners_scene.astype(np.float32)
            has_overlap = False
            for exist in existing_corners:
                poly_exist = exist.astype(np.float32)
                c1 = np.mean(poly_new, axis=0)
                c2 = np.mean(poly_exist, axis=0)
                dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
                if dist < (target_long * 0.75):
                    has_overlap = True
                    break
                ret, _ = cv2.rotatedRectangleIntersection(
                    cv2.minAreaRect(poly_new), cv2.minAreaRect(poly_exist)
                )
                if ret != cv2.INTERSECT_NONE:
                    has_overlap = True
                    break
            if has_overlap:
                continue

        rotated = cv2.warpAffine(
            card, M, (rw, rh), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )

        alpha = rotated[:, :, 3:4].astype(np.float32) / 255.0
        roi = scene[py:py + rh, px:px + rw].astype(np.float32)
        blended = rotated[:, :, :3].astype(np.float32) * alpha + roi * (1 - alpha)
        scene_out = scene.copy()
        scene_out[py:py + rh, px:px + rw] = blended.astype(np.uint8)

        return scene_out, corners_scene

    return scene, None

def corners_to_yolo_obb(corners: np.ndarray, W: int, H: int) -> str:
    """Convert (4,2) pixel corners to a YOLO OBB label line (normalized)."""
    pts = corners.copy()
    pts[:, 0] = np.clip(pts[:, 0] / W, 0, 1)
    pts[:, 1] = np.clip(pts[:, 1] / H, 0, 1)
    coords = " ".join(f"{p:.6f}" for p in pts.flatten())
    return f"{CLASS_ID} {coords}"

def main():
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("phase1a_dataset", cfg)

    full_coverage = True
    n_scenes = 0
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg.isdigit():
            full_coverage = False
            n_scenes = int(arg)
        elif arg.lower() in ("--full", "--full-coverage", "all", "full"):
            full_coverage = True

    out_dir = (PROJECT_ROOT / cfg["paths"]["datasets_dir"] / cfg["obb"]["dataset"]["name"]).resolve()
    split_ratios = cfg["obb"]["dataset"]["split"]
    splits = ["train", "val", "test"]
    for s in splits:
        img_s = out_dir / "images" / s
        lbl_s = out_dir / "labels" / s
        img_s.mkdir(parents=True, exist_ok=True)
        lbl_s.mkdir(parents=True, exist_ok=True)
        for f in img_s.glob("*.jpg"):
            f.unlink()
        for f in lbl_s.glob("*.txt"):
            f.unlink()

    hf_dir = PROJECT_ROOT / "data" / "datasets" / "cards" / "pokemon_official_tcg"
    if not hf_dir.exists():
        hf_dir = PROJECT_ROOT / "data" / "datasets" / "raw" / "pokemon_cards_hf"

    legacy_dir = PROJECT_ROOT / "data" / "datasets" / "cards" / "pokemon_pocket_tcgp"
    if not legacy_dir.exists():
        legacy_dir = PROJECT_ROOT / "data" / "datasets" / "raw" / "Pokemon-TCGP-Card-Scanner-main" / \
            "pokemon-tcgp-card-scanner" / "public" / "images"

    legacy_cards = sorted(legacy_dir.glob("*.webp")) if legacy_dir.exists() else []

    card_pools = {
        "train": sorted((hf_dir / "train").glob("*.webp")) if (hf_dir / "train").exists() else [],
        "val": sorted((hf_dir / "val").glob("*.webp")) if (hf_dir / "val").exists() else [],
        "test": sorted((hf_dir / "test").glob("*.webp")) if (hf_dir / "test").exists() else [],
    }

    if legacy_cards:
        rng_leg = random.Random(cfg["project"]["seed"])
        leg_shuffled = legacy_cards.copy()
        rng_leg.shuffle(leg_shuffled)
        n_leg = len(leg_shuffled)
        n_l_train = int(n_leg * split_ratios[0])
        n_l_val = int(n_leg * split_ratios[1])
        card_pools["train"].extend(leg_shuffled[:n_l_train])
        card_pools["val"].extend(leg_shuffled[n_l_train:n_l_train + n_l_val])
        card_pools["test"].extend(leg_shuffled[n_l_train + n_l_val:])

    total_cards = sum(len(pool) for pool in card_pools.values())
    exp.log_metric("dataset.source_cards_total", total_cards)
    exp.log_metric("dataset.source_cards_train", len(card_pools["train"]))
    exp.log_metric("dataset.source_cards_val", len(card_pools["val"]))
    exp.log_metric("dataset.source_cards_test", len(card_pools["test"]))
    exp.logger.info(
        f"Available source cards: Total={total_cards} (Train={len(card_pools['train'])}, "
        f"Val={len(card_pools['val'])}, Test={len(card_pools['test'])})"
    )

    if total_cards == 0:
        exp.logger.error("No card scans found - run the raw dataset download first.")
        sys.exit(1)

    exp.logger.info("Discovering cached backgrounds (DTD, CC0 PBR, Desks, Photos)…")
    bg_paths = get_available_backgrounds(min_needed=120, seed=cfg["project"]["seed"])
    exp.logger.info(f"Loaded {len(bg_paths)} diverse background images from subfolders.")
    exp.log_metric("dataset.backgrounds_total", len(bg_paths))

    hard_negative_ratio = float(cfg.get("obb", {}).get("dataset", {}).get("hard_negative_ratio", 0.06))
    hard_negatives = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0}
    boxes_per_scene = []
    cards_placed_per_split = {"train": 0, "val": 0, "test": 0}

    if full_coverage:
        exp.logger.info(
            f"=== OPTION A GENERATION: 100% DETERMINISTIC COVERAGE OF ALL {total_cards} CARDS ==="
        )
        global_scene_id = 0
        p_hn = hard_negative_ratio / (1.0 - hard_negative_ratio)

        for split in splits:
            pool = card_pools[split]
            if not pool:
                continue
            rng_split = random.Random(cfg["project"]["seed"] + hash(split) % 10000)
            card_queue = pool.copy()
            rng_split.shuffle(card_queue)
            initial_count = len(card_queue)
            exp.logger.info(f"Generating split '{split}' with {initial_count} unique source cards…")

            split_scenes = 0
            while card_queue:
                if rng_split.random() < p_hn:
                    bg_path = rng_split.choice(bg_paths)
                    scene = load_background(bg_path, rng_split)
                    boxes_per_scene.append(0)
                    hard_negatives[split] += 1
                    name = f"scene_{global_scene_id:05d}"
                    cv2.imwrite(
                        str(out_dir / "images" / split / f"{name}.jpg"),
                        scene, [cv2.IMWRITE_JPEG_QUALITY, 92]
                    )
                    with open(out_dir / "labels" / split / f"{name}.txt", "w") as f:
                        f.write("")
                    counts[split] += 1
                    split_scenes += 1
                    global_scene_id += 1
                    continue

                n_target_cards = 2 if (len(card_queue) >= 2 and rng_split.random() < 0.12) else 1
                batch_cards = [card_queue.pop(0) for _ in range(n_target_cards)]

                bg_path = rng_split.choice(bg_paths)
                scene = load_background(bg_path, rng_split)
                scene_corners = []
                labels = []
                unplaced = []

                for c_path in batch_cards:
                    raw_card = load_card_rgba(c_path)
                    damaged_card = apply_damage_augmentations(raw_card, rng_split)

                    placed = False
                    for _ in range(15):
                        new_scene, c_pts = composite_card(
                            scene, damaged_card, rng_split,
                            existing_corners=scene_corners,
                            margin=SAFE_MARGIN
                        )
                        if c_pts is not None and is_card_fully_inside(c_pts, IMG_W, IMG_H, margin=SAFE_MARGIN):
                            scene = new_scene
                            scene_corners.append(c_pts)
                            labels.append(corners_to_yolo_obb(c_pts, IMG_W, IMG_H))
                            cards_placed_per_split[split] += 1
                            placed = True
                            break
                    if not placed:
                        unplaced.append(c_path)

                if unplaced:
                    card_queue.extend(unplaced)

                if len(labels) == 0:
                    hard_negatives[split] += 1
                    boxes_per_scene.append(0)
                else:
                    boxes_per_scene.append(len(labels))

                name = f"scene_{global_scene_id:05d}"
                cv2.imwrite(
                    str(out_dir / "images" / split / f"{name}.jpg"),
                    scene, [cv2.IMWRITE_JPEG_QUALITY, 92]
                )
                with open(out_dir / "labels" / split / f"{name}.txt", "w") as f:
                    f.write("\n".join(labels) + "\n" if labels else "")

                counts[split] += 1
                split_scenes += 1
                global_scene_id += 1

                if split_scenes % 500 == 0 or len(card_queue) == 0:
                    exp.logger.info(
                        f"  [{split}] {split_scenes} scenes generated | Remaining cards: {len(card_queue)}/{initial_count}"
                    )

        total_scenes = global_scene_id
    else:
        rng = random.Random(cfg["project"]["seed"])
        for i in range(n_scenes):
            split = "train" if rng.random() < split_ratios[0] else (
                "val" if rng.random() < split_ratios[1] / (split_ratios[1] + split_ratios[2]) else "test")
            scene = load_background(rng.choice(bg_paths), rng)

            labels = []
            is_hard_negative = (rng.random() < hard_negative_ratio)
            if is_hard_negative:
                boxes_per_scene.append(0)
                hard_negatives[split] += 1
            else:
                card_pool = card_pools[split] if card_pools[split] else card_pools["train"]
                n_target_cards = 2 if (rng.random() < 0.12) else 1
                scene_corners = []

                for _ in range(n_target_cards):
                    raw_card = load_card_rgba(rng.choice(card_pool))
                    damaged_card = apply_damage_augmentations(raw_card, rng)
                    new_scene, c_pts = composite_card(
                        scene, damaged_card, rng,
                        existing_corners=scene_corners,
                        margin=SAFE_MARGIN
                    )
                    if c_pts is not None and is_card_fully_inside(c_pts, IMG_W, IMG_H, margin=SAFE_MARGIN):
                        scene = new_scene
                        scene_corners.append(c_pts)
                        labels.append(corners_to_yolo_obb(c_pts, IMG_W, IMG_H))
                        cards_placed_per_split[split] += 1

                boxes_per_scene.append(len(labels))
                if len(labels) == 0:
                    hard_negatives[split] += 1

            name = f"scene_{i:05d}"
            cv2.imwrite(str(out_dir / "images" / split / f"{name}.jpg"),
                        scene, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(out_dir / "labels" / split / f"{name}.txt", "w") as f:
                f.write("\n".join(labels) + "\n" if labels else "")
            counts[split] += 1
            if (i + 1) % 100 == 0:
                exp.logger.info(f"  {i + 1}/{n_scenes} scenes generated")
        total_scenes = n_scenes

    exp.log_metrics({f"scenes_{k}": v for k, v in counts.items()})
    exp.log_metric("scenes_total", total_scenes)
    exp.log_metric("boxes_total", int(sum(boxes_per_scene)))
    exp.log_metric("boxes_per_scene_mean", float(np.mean(boxes_per_scene)))
    exp.log_metric("cards_placed_total", int(sum(cards_placed_per_split.values())))
    exp.log_metrics({f"cards_placed_{k}": v for k, v in cards_placed_per_split.items()})
    exp.log_metric("dataset.hard_negatives_total", int(sum(hard_negatives.values())))
    exp.log_metric("dataset.hard_negatives_ratio", float(sum(hard_negatives.values()) / max(1, total_scenes)))
    exp.log_metrics({f"hard_negatives_{k}": v for k, v in hard_negatives.items()})

    yaml_path = out_dir / "pokemon_obb.yaml"
    yaml_path.write_text(
        f"# Auto-generated by scripts/prepare_dataset.py\n"
        f"path: {out_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n  0: card\n"
    )
    exp.logger.info(f"Dataset YAML → {yaml_path}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(counts.keys(), counts.values(), color=["#3B9EFF", "#FFD700", "#00E396"])
    ax[0].set_title("Scenes per split")
    ax[0].set_ylabel("count")
    for k, v in counts.items():
        ax[0].text(k, v, str(v), ha="center", va="bottom")
    ax[1].hist(boxes_per_scene, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8, color="#A78BFA")
    ax[1].set_title("Cards per scene (0 = Hard Negative)")
    ax[1].set_xlabel("number of cards")
    ax[1].set_xticks([0, 1, 2])
    fig.suptitle("Phase 1a - Synthetic OBB dataset statistics")
    exp.save_plot(fig, "dataset_stats")

    sample_files = sorted((out_dir / "images" / "train").glob("*.jpg"))[:6]
    fig, axs = plt.subplots(2, 3, figsize=(13, 9))
    for ax_, img_path in zip(axs.flat, sample_files):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        lbl_text = (out_dir / "labels" / "train" / (img_path.stem + ".txt")).read_text().strip()
        lbl_lines = lbl_text.splitlines() if lbl_text else []
        for line in lbl_lines:
            vals = np.array(line.split()[1:], np.float64).reshape(4, 2) * [IMG_W, IMG_H]
            pts = vals.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (255, 215, 0), 2)
        ax_.imshow(img)
        title = f"{img_path.name} ({len(lbl_lines)} cards)" if lbl_lines else f"{img_path.name} (Hard Negative)"
        ax_.set_title(title, fontsize=8)
        ax_.axis("off")
    fig.suptitle("Phase 1a - Sample scenes with OBB ground truth (yellow)")
    exp.save_plot(fig, "dataset_samples")

    exp.finish()
    exp.logger.info(f"DONE. Dataset at {out_dir}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

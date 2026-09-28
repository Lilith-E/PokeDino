"""Module for deterministic generation of photometric and acquisition variants (Few-Shot).

Generates 20 realistic, high-fidelity variants simulating real smartphone captures:
1. Warm Tungsten (warm indoor incandescent lighting)
2. Cool Fluorescent (cool neon / office lighting)
3. ISO Sensor Noise (sensor noise at high ISO sensitivities)
4. Camera Defocus Blur (optical lens defocus blur)
5. Motion Blur (hand tremor / micro-shake)
6. Lens Vignetting (circular light falloff towards corners)
7. Flash LED Hotspot (direct specular flashlight point source)
8. JPEG Compression Artifacts (compression blockiness & ringing)
9. ISP Bilateral Smoothing (denoise filtering softening micro-textures)
10. Harsh Phone Shadow (vertical shadow gradient from smartphone body)
11. Lateral Window Light (natural directional window illumination)
12. Underexposed (low ambient lighting)
13. Overexposed (intense direct specular light)
14. High Contrast HDR (deepened darks and blown highlights)
15. Low Contrast Washed (diffuse haze / washed-out look)
16. Vivid Smartphone ISP (boosted chromatic saturation)
17. Chromatic Aberration (radial color fringing on edges)
18. Holo Rainbow Sheen (diagonal holographic specular reflection across artwork)

Also provides persistent caching of DINOv2 feature descriptors to avoid
re-computing neural inference on previously extracted reference cards.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

def _variant_cache_key(card_id: str, variant_label: str, grid_size: tuple, mask_hash: str, model_name: str = "") -> str:
    """Constructs deterministic cache key for DINOv2 feature descriptors."""
    h = hashlib.sha1(f"{VARIANTS_VERSION}|{card_id}|{variant_label}|{grid_size[0]}x{grid_size[1]}|{mask_hash}|{model_name}".encode("utf-8"))
    return h.hexdigest()[:16]

def get_feature_cache_path(cache_base_dir: str, card_id: str, label: str, grid_size: tuple, model_name: str = "") -> Path:
    """Returns filesystem path for cached feature descriptors of a variant."""
    base = Path(cache_base_dir)
    out_dir = base / card_id
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace(os.sep, "_") if label else "base"
    model_suffix = f"_{model_name}" if model_name else ""
    fname = f"{safe_label}_{grid_size[0]}x{grid_size[1]}{model_suffix}.npz"
    return out_dir / fname

def load_cached_features(
    cache_base_dir: Optional[str],
    card_id: str,
    label: str,
    grid_size: tuple,
    model_name: str = "",
) -> Optional[np.ndarray]:
    """Loads precomputed DINOv2 descriptors for a variant if available on disk."""
    if cache_base_dir is None:
        return None
    p = get_feature_cache_path(cache_base_dir, card_id, label, grid_size, model_name)
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=False) as data:
            if "features" not in data.files:
                return None
            feats = data["features"]
            if feats.dtype != np.float32:
                feats = feats.astype(np.float32, copy=False)
            return feats
    except Exception as e:
        logger.warning("Feature cache read failed for %s: %s", p, e)
        return None

def save_cached_features(
    cache_base_dir: Optional[str],
    card_id: str,
    label: str,
    grid_size: tuple,
    features: np.ndarray,
    model_name: str = "",
) -> bool:
    """Saves DINOv2 descriptors to disk for future reuse. Returns True if successful."""
    if cache_base_dir is None:
        return False
    p = get_feature_cache_path(cache_base_dir, card_id, label, grid_size, model_name)
    try:
        if features.dtype != np.float32:
            features = features.astype(np.float32, copy=False)
        np.savez_compressed(p, features=features)
        return True
    except Exception as e:
        logger.warning("Feature cache write failed for %s: %s", p, e)
        return False

def hash_mask(mask: Optional[np.ndarray]) -> str:
    """Returns a short, deterministic hash of the validity mask."""
    if mask is None:
        return "nomask"
    arr = (mask > 0.5).astype(np.uint8)
    h = hashlib.sha1(arr.tobytes()).hexdigest()[:8]
    return f"m{h}"

def generate_photometric_variants(img_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Generates 18 photometric variants while preserving card geometry strictly.

    Args:
        img_bgr: Canonical uint8 BGR card image.

    Returns:
        List of tuples (variant_name, bgr_image_uint8).
    """
    h, w = img_bgr.shape[:2]
    img_float = img_bgr.astype(np.float32) / 255.0

    variants = []

    warm = img_float.copy()
    warm[:, :, 2] = np.clip(warm[:, :, 2] * 1.18, 0.0, 1.0)
    warm[:, :, 1] = np.clip(warm[:, :, 1] * 1.05, 0.0, 1.0)
    warm[:, :, 0] = np.clip(warm[:, :, 0] * 0.86, 0.0, 1.0)
    variants.append(("01_warm_tungsten", (warm * 255).astype(np.uint8)))

    cool = img_float.copy()
    cool[:, :, 0] = np.clip(cool[:, :, 0] * 1.18, 0.0, 1.0)
    cool[:, :, 2] = np.clip(cool[:, :, 2] * 0.88, 0.0, 1.0)
    variants.append(("02_cool_fluorescent", (cool * 255).astype(np.uint8)))

    rng = np.random.RandomState(42)
    noise = rng.normal(0.0, 14.0 / 255.0, img_float.shape).astype(np.float32)
    noisy = np.clip(img_float + noise, 0.0, 1.0)
    variants.append(("03_iso_sensor_noise", (noisy * 255).astype(np.uint8)))

    defocus = cv2.GaussianBlur(img_bgr, (5, 5), sigmaX=1.3, sigmaY=1.3)
    variants.append(("04_camera_defocus_blur", defocus))

    k_motion = np.zeros((5, 5), dtype=np.float32)
    np.fill_diagonal(k_motion, 1.0 / 5.0)
    motion = cv2.filter2D(img_bgr, -1, k_motion)
    variants.append(("05_motion_blur", motion))

    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    radius_sq = (X ** 2 + Y ** 2) / 2.0
    vignette_mask = np.clip(1.0 - 0.40 * radius_sq, 0.35, 1.0)[:, :, np.newaxis].astype(np.float32)
    vignette = np.clip(img_float * vignette_mask, 0.0, 1.0)
    variants.append(("06_lens_vignette", (vignette * 255).astype(np.uint8)))

    fx, fy = w * 0.5, h * 0.28
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    dist_flash_sq = ((grid_x - fx) ** 2 + (grid_y - fy) ** 2).astype(np.float32)
    flash_sigma = (w * 0.38) ** 2
    flash_boost = 1.0 + 0.32 * np.exp(-dist_flash_sq / (2.0 * flash_sigma))
    flash_img = np.clip(img_float * flash_boost[:, :, np.newaxis], 0.0, 1.0)
    variants.append(("07_flash_hotspot", (flash_img * 255).astype(np.uint8)))

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
    _, enc = cv2.imencode(".jpg", img_bgr, encode_param)
    jpeg_dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    variants.append(("08_jpeg_compression", jpeg_dec))

    bilateral = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=45, sigmaSpace=45)
    variants.append(("09_isp_bilateral_smooth", bilateral))

    y_grad = np.linspace(1.22, 0.74, h, dtype=np.float32)[:, np.newaxis, np.newaxis]
    v_grad = np.clip(img_float * y_grad, 0.0, 1.0)
    variants.append(("10_harsh_shadow_grad", (v_grad * 255).astype(np.uint8)))

    x_grad = np.linspace(1.25, 0.70, w, dtype=np.float32)[np.newaxis, :, np.newaxis]
    h_grad = np.clip(img_float * x_grad, 0.0, 1.0)
    variants.append(("11_lateral_window_light", (h_grad * 255).astype(np.uint8)))

    exp_und = np.clip(img_float * 0.78, 0.0, 1.0)
    variants.append(("12_underexposed", (exp_und * 255).astype(np.uint8)))

    exp_ovr = np.clip(img_float * 1.22, 0.0, 1.0)
    variants.append(("13_overexposed", (exp_ovr * 255).astype(np.uint8)))

    exp_ovr2 = np.clip(img_float * 1.45, 0.0, 1.0)
    variants.append(("19_overexposed_strong", (exp_ovr2 * 255).astype(np.uint8)))

    exp_ovr3 = np.clip(img_float * 1.70, 0.0, 1.0)
    variants.append(("20_overexposed_blowout", (exp_ovr3 * 255).astype(np.uint8)))

    s_curve = np.clip(np.power(img_float, 1.28), 0.0, 1.0)
    variants.append(("14_high_contrast_hdr", (s_curve * 255).astype(np.uint8)))

    washed = np.clip(np.power(img_float, 0.82) * 0.90 + 0.08, 0.0, 1.0)
    variants.append(("15_low_contrast_washed", (washed * 255).astype(np.uint8)))

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)
    vivid = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    variants.append(("16_vivid_smartphone_isp", vivid))

    b, g, r = cv2.split(img_bgr)
    m_r = np.float32([[1, 0, 1], [0, 1, 0]])
    m_b = np.float32([[1, 0, -1], [0, 1, 0]])
    r_shift = cv2.warpAffine(r, m_r, (w, h), borderMode=cv2.BORDER_REFLECT)
    b_shift = cv2.warpAffine(b, m_b, (w, h), borderMode=cv2.BORDER_REFLECT)
    chroma = cv2.merge([b_shift, g, r_shift])
    variants.append(("17_chromatic_aberration", chroma))

    diag_sheen = np.sin((X * 2.5 + Y * 2.0) * np.pi) * 0.5 + 0.5
    art_mask = np.clip(1.0 - (grid_y / (h * 0.55)), 0.0, 1.0)[:, :, np.newaxis]
    sheen_boost = 1.0 + 0.22 * (diag_sheen[:, :, np.newaxis] * art_mask)
    holo_img = np.clip(img_float * sheen_boost, 0.0, 1.0)
    variants.append(("18_holo_rainbow_sheen", (holo_img * 255).astype(np.uint8)))

    return variants

EXPECTED_VARIANT_NAMES = [
    "01_warm_tungsten.png",
    "02_cool_fluorescent.png",
    "03_iso_sensor_noise.png",
    "04_camera_defocus_blur.png",
    "05_motion_blur.png",
    "06_lens_vignette.png",
    "07_flash_hotspot.png",
    "08_jpeg_compression.png",
    "09_isp_bilateral_smooth.png",
    "10_harsh_shadow_grad.png",
    "11_lateral_window_light.png",
    "12_underexposed.png",
    "13_overexposed.png",
    "14_high_contrast_hdr.png",
    "15_low_contrast_washed.png",
    "16_vivid_smartphone_isp.png",
    "17_chromatic_aberration.png",
    "18_holo_rainbow_sheen.png",
    "19_overexposed_strong.png",
    "20_overexposed_blowout.png",
]

VARIANTS_VERSION = "v2"

def get_or_create_few_shot_variants(
    card_id: str,
    base_image_path: str,
    cache_base_dir: Optional[str] = None,
) -> list[str]:
    """Retrieves or generates on-disk Few-Shot reference paths for the specified card.

    Always returns [base_image_path] + [variant_path_1, ..., variant_path_20] (21 images total).
    """
    base_p = Path(base_image_path)
    if not base_p.exists():
        raise FileNotFoundError(f"Base reference image not found: {base_image_path}")

    if cache_base_dir is None:
        target_dir = base_p.parent / "few_shot" / card_id
    else:
        target_dir = Path(cache_base_dir) / "few_shot" / card_id

    target_dir.mkdir(parents=True, exist_ok=True)

    all_exist = all((target_dir / name).exists() for name in EXPECTED_VARIANT_NAMES)

    if not all_exist:
        logger.info("Generating 20 heavy Few-Shot photometric variants for %s...", card_id)
        img_bgr = cv2.imread(str(base_p))
        if img_bgr is None:
            return [str(base_p)]

        variants = generate_photometric_variants(img_bgr)
        for name, var_bgr in variants:
            out_path = target_dir / f"{name}.png"
            cv2.imwrite(str(out_path), var_bgr)
        logger.info("20 Few-Shot variants generated and cached in %s", target_dir)

    variant_paths = [str(target_dir / name) for name in EXPECTED_VARIANT_NAMES if (target_dir / name).exists()]
    return [str(base_p)] + variant_paths

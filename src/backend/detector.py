"""Module for estimating visual anomalies on collectible cards using AnomalyDINO.

The inspection pipeline comprises:
1. Uniform resizing of the card image to canonical dimensions.
2. Generation of a rounded-corner perimeter mask to isolate the region of interest.
3. Contrast Limited Adaptive Histogram Equalization (CLAHE) confined to the card interior.
4. Patch descriptor extraction via the DINOv2 neural backbone.
5. Anomaly distance computation relative to the reference memory bank.
6. Dead Zone and Gamma Correction application for defect heat map generation.
7. Qualitative condition classification into standardized grading tiers.
"""

import base64
import certifi
import cv2
import io
import logging
import os
from pathlib import Path
import ssl
import sys
from typing import Optional

import numpy as np
from PIL import Image
import torch

_cert_path = certifi.where()
os.environ.setdefault("SSL_CERT_FILE", _cert_path)
os.environ.setdefault("REQUESTS_CA_BUNDLE", _cert_path)
try:
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=_cert_path)
except Exception:
    pass

ANOMALYDINO_PATH = Path(__file__).resolve().parent.parent / "anomalydino_src"
if not ANOMALYDINO_PATH.exists():
    ANOMALYDINO_PATH = Path(__file__).resolve().parents[2] / "src" / "anomalydino_src"
if str(ANOMALYDINO_PATH) not in sys.path:
    sys.path.insert(0, str(ANOMALYDINO_PATH))

try:
    from src.backbones import get_model
    from src.post_eval import mean_top1p
except (ImportError, AttributeError):
    import importlib.util
    def _load_anomalydino_mod(mod_name, file_rel_path):
        spec = importlib.util.spec_from_file_location(mod_name, ANOMALYDINO_PATH / "src" / file_rel_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    _bb_mod = _load_anomalydino_mod("anomalydino_src.backbones", "backbones.py")
    _pe_mod = _load_anomalydino_mod("anomalydino_src.post_eval", "post_eval.py")
    get_model = _bb_mod.get_model
    mean_top1p = _pe_mod.mean_top1p

from backend.condition import classify_condition
from backend.experiment import load_config as _load_config
from backend.few_shot import (
    get_or_create_few_shot_variants,
    hash_mask,
    load_cached_features,
    save_cached_features,
)
from backend.masking import mask_to_grid, rounded_rect_mask
from backend.preprocessing import apply_clahe, apply_clahe_masked, compute_luma_stats, match_luminance
from backend.spatial_morphology import (
    MorphologyConfig,
    MorphologyResult,
    SpatialSearchConfig,
    analyze_defect_morphology,
    build_patch_coordinates,
    compute_spatial_knn_distances,
)

logger = logging.getLogger(__name__)

_CFG = _load_config()
DEFAULT_MODEL = str(_CFG.get("anomaly", {}).get("model", "dinov2_vitb14"))
DEFAULT_IMG_SIZE = int(_CFG.get("anomaly", {}).get("img_size", 448))
TARGET_SIZE = (
    int(_CFG.get("obb", {}).get("warp", {}).get("target_h", 630)),
    int(_CFG.get("obb", {}).get("warp", {}).get("target_w", 448)),
)
ANOMALY_THRESHOLD = float(_CFG.get("anomaly", {}).get("threshold", 0.400))
DEFECT_AREA_WEIGHT = float(_CFG.get("anomaly", {}).get("defect_area_weight", 0.35))
DEFECT_PIXEL_THRESHOLD = float(_CFG.get("anomaly", {}).get("defect_pixel_threshold", 0.28))
DEFAULT_DEAD_ZONE = float(_CFG.get("anomaly", {}).get("dead_zone", 0.30))
DEFAULT_GAMMA = float(_CFG.get("anomaly", {}).get("gamma", 2.0))

_FEATURE_CACHE_DIR = str(_CFG.get("catalog", {}).get("feature_cache_dir", "data/reference_cache/dinov2_features"))

_SPATIAL_CFG_DICT = _CFG.get("anomaly", {}).get("spatial", {}) or {}
DEFAULT_SPATIAL_CONFIG = SpatialSearchConfig(
    enabled=bool(_SPATIAL_CFG_DICT.get("enabled", True)),
    radius_patches=float(_SPATIAL_CFG_DICT.get("radius_patches", 3.0)),
    penalty_weight=float(_SPATIAL_CFG_DICT.get("penalty_weight", 0.45)),
)

_MORPH_DICT = _CFG.get("anomaly", {}).get("morphology", {}) or {}
DEFAULT_MORPHOLOGY_CONFIG = MorphologyConfig(
    enabled=bool(_MORPH_DICT.get("enabled", True)),
    large_cluster_threshold=int(_MORPH_DICT.get("large_cluster_threshold", 15)),
    cluster_penalty=float(_MORPH_DICT.get("cluster_penalty", 0.025)),
    linear_eccentricity=float(_MORPH_DICT.get("linear_eccentricity", 3.0)),
    linear_penalty=float(_MORPH_DICT.get("linear_penalty", 0.04)),
)

CLAHE_CLIP_LIMIT = float(_CFG.get("clahe", {}).get("clip_limit", 2.0))
CLAHE_TILE_GRID = tuple(_CFG.get("clahe", {}).get("tile_grid_size", [8, 8]))
CORNER_RADIUS_FRAC = float(_CFG.get("masking", {}).get("corner_radius_frac", 0.035))
ERODE_FRAC = float(_CFG.get("masking", {}).get("erode_frac", 0.02))
FEATHER_PX = int(_CFG.get("masking", {}).get("feather_px", 5))

def get_device() -> str:
    """Detects and returns optimal available computing device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def preprocess_card_image(
    img_path: str,
    target_size: tuple = TARGET_SIZE,
    clahe: bool = True,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: tuple = CLAHE_TILE_GRID,
) -> np.ndarray:
    """Preprocesses card image applying resize and optional CLAHE equalization.

    Args:
        img_path: Path to image file.
        target_size: Target dimensions (height, width).
        clahe: If True, applies CLAHE on L channel in LAB color space.
        clip_limit: Contrast threshold for CLAHE.
        tile_grid: Local grid dimensions for CLAHE.

    Returns:
        uint8 RGB NumPy array of dimensions target_size.
    """
    target_h, target_w = target_size
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    if clahe:
        img = apply_clahe(img, clip_limit=clip_limit, tile_grid_size=tile_grid)
    return img

def ndarray_to_base64(img: np.ndarray, fmt: str = "PNG") -> str:
    """Serializza un array NumPy RGB in una stringa codificata Base64."""
    pil_img = Image.fromarray(img.astype(np.uint8))
    buffer = io.BytesIO()
    pil_img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def heatmap_overlay(
    original_img: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.88,
    valid_mask: Optional[np.ndarray] = None,
    dead_zone: float = DEFAULT_DEAD_ZONE,
    gamma: float = DEFAULT_GAMMA,
    colormap: int = cv2.COLORMAP_JET,
    adaptive_alpha: bool = True,
    bilateral_smoothing: bool = True,
    style: str = "jet",
    vmax_anchor: float = ANOMALY_THRESHOLD,
    base_alpha: float = 0.60,
) -> np.ndarray:
    """Generates defect heat map overlaid onto the original image.

    Full color-JET map (MVTec/AnomalyDINO style): the whole card is tinted
    BLUE (healthy zones), defects scale continuously cyan → yellow → RED.
    The color scale is anchored to fixed raw-distance values, so the same
    defect always gets the same color across images (red ≡ anomaly
    threshold), unlike the previous per-image min-max normalization.

    Args:
        original_img: Original uint8 RGB image.
        anomaly_map: Raw 2D anomaly distance map (raw cosine distances).
        alpha: Maximum overlay opacity on the hottest defects.
        valid_mask: Validity mask for interior card region.
        dead_zone: Raw noise gate in [0.0, 1.0]: distances below this value
            are rendered as full blue (physiological noise suppressed).
        gamma: Gamma correction exponent applied after normalization.
        colormap: OpenCV colormap identifier (used when style='jet').
        adaptive_alpha: If True, opacity rises with defect severity
            (from base_alpha up to alpha). If False, opacity is fixed at alpha.
        bilateral_smoothing: If True, applies bilateral filter on the colored
            map for perceptual smoothing that preserves defect edges.
        style: Heatmap style - 'jet' (color JET, default), 'medical'
            (custom blue-to-red gradient), 'grayscale'.
        vmax_anchor: Raw distance rendered as full red (default: anomaly threshold).
        base_alpha: Minimum overlay opacity for healthy (blue) zones.

    Returns:
        Blended RGB image with heat map overlay.
    """
    h, w = original_img.shape[:2]
    anomaly_map_resized = cv2.resize(
        anomaly_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LANCZOS4
    )

    anomaly_map_resized = cv2.GaussianBlur(anomaly_map_resized, (0, 0), 9)
    if valid_mask is not None:
        anomaly_map_resized = anomaly_map_resized * valid_mask

    dz = float(np.clip(dead_zone, 0.0, 0.95))
    hi = max(float(vmax_anchor), dz + 0.05)
    norm = np.clip((anomaly_map_resized - dz) / (hi - dz), 0.0, 1.0)

    if gamma > 0 and gamma != 1.0:
        norm = np.power(norm, gamma)

    if adaptive_alpha:
        mean_anomaly = float(norm.mean())
        base = max(base_alpha, 0.35 if mean_anomaly < 0.30 else 0.45)
    else:
        base = base_alpha
    alpha_mask = np.clip(base + (alpha - base) * np.power(norm, 0.85), 0.0, 1.0)[:, :, np.newaxis]

    heat_uint8 = (norm * 255).astype(np.uint8)

    if style == "medical":
        anchors = np.array([0.00, 0.25, 0.50, 0.70, 0.85, 1.00])
        ch_r = np.array([30, 0, 60, 250, 255, 220])
        ch_g = np.array([60, 170, 215, 225, 130, 25])
        ch_b = np.array([220, 235, 90, 45, 20, 25])
        heat_rgb = np.stack(
            [
                np.interp(norm, anchors, ch_r),
                np.interp(norm, anchors, ch_g),
                np.interp(norm, anchors, ch_b),
            ],
            axis=-1,
        )
        if bilateral_smoothing:
            heat_u8 = heat_rgb.clip(0, 255).astype(np.uint8)
            heat_rgb = cv2.bilateralFilter(heat_u8, d=9, sigmaColor=75, sigmaSpace=75).astype(np.float32)
        else:
            heat_rgb = heat_rgb.clip(0, 255).astype(np.float32)
    elif style == "grayscale":
        heat_colored = cv2.cvtColor(heat_uint8, cv2.COLOR_GRAY2BGR)
        if bilateral_smoothing:
            heat_colored = cv2.bilateralFilter(heat_colored, d=9, sigmaColor=75, sigmaSpace=75)
        heat_rgb = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB).astype(np.float32)
    else:
        heat_colored = cv2.applyColorMap(heat_uint8, colormap)
        if bilateral_smoothing:
            heat_colored = cv2.bilateralFilter(heat_colored, d=9, sigmaColor=75, sigmaSpace=75)
        heat_rgb = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB).astype(np.float32)

    blended = (
        original_img.astype(np.float32) * (1.0 - alpha_mask)
        + heat_rgb * alpha_mask
    )
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    if valid_mask is not None:
        m = valid_mask < 0.5
        blended[m] = (blended[m] * 0.3).astype(np.uint8)

    return blended

class PokemonAnomalyDetector:
    """Rilevatore di anomalie superficiali per carte basato su DINOv2."""

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        img_size: int = DEFAULT_IMG_SIZE,
        anomaly_threshold: float = ANOMALY_THRESHOLD,
        knn_neighbors: int = 1,
        clahe_enabled: bool = True,
        masking_enabled: bool = True,
        luma_matching: Optional[bool] = None,
        dead_zone: Optional[float] = None,
        gamma: Optional[float] = None,
        defect_area_weight: Optional[float] = None,
        defect_pixel_threshold: Optional[float] = None,
        spatial_config: Optional[SpatialSearchConfig] = None,
        morphology_config: Optional[MorphologyConfig] = None,
        feature_cache_dir: Optional[str] = None,
    ):
        """Inizializza i parametri operativi del detector."""
        self.device = device or get_device()
        self.model_name = model_name
        self.img_size = img_size
        self.anomaly_threshold = anomaly_threshold
        self.knn_neighbors = knn_neighbors
        self.clahe_enabled = clahe_enabled
        self.masking_enabled = masking_enabled
        self.luma_matching_enabled = (
            luma_matching if luma_matching is not None
            else bool(_CFG["anomaly"].get("luma_matching", False))
        )
        self._ref_luma_stats: Optional[tuple[float, float]] = None
        self._loading_reference: bool = False
        self.orientation_canonicalization = bool(
            _CFG["anomaly"].get("orientation_canonicalization", False)
        )
        self.dead_zone = dead_zone if dead_zone is not None else DEFAULT_DEAD_ZONE
        self.gamma = gamma if gamma is not None else DEFAULT_GAMMA
        self.defect_area_weight = defect_area_weight if defect_area_weight is not None else DEFECT_AREA_WEIGHT
        self.defect_pixel_threshold = defect_pixel_threshold if defect_pixel_threshold is not None else DEFECT_PIXEL_THRESHOLD
        self.spatial_config = spatial_config or DEFAULT_SPATIAL_CONFIG
        self.morphology_config = morphology_config or DEFAULT_MORPHOLOGY_CONFIG
        self.feature_cache_dir = feature_cache_dir or _FEATURE_CACHE_DIR

        self.model = None
        self.ref_features: Optional[torch.Tensor] = None
        self.ref_features_oneshot: Optional[torch.Tensor] = None
        self.ref_features_fewshot: Optional[torch.Tensor] = None
        self.ref_coords: Optional[np.ndarray] = None
        self.ref_coords_oneshot: Optional[np.ndarray] = None
        self.ref_coords_fewshot: Optional[np.ndarray] = None
        self.use_few_shot: bool = True
        self.few_shot_variants: list[str] = []
        self.reference_count: int = 0
        self.memory_bank_ready: bool = False
        self.current_card_id: Optional[str] = None

        self._last_patch: Optional[np.ndarray] = None
        self._last_raw_bgr: Optional[np.ndarray] = None
        self._last_d_map: Optional[np.ndarray] = None
        self._last_mask: Optional[np.ndarray] = None
        self._last_test_tensor: Optional[torch.Tensor] = None
        self._last_grid_mask_flat: Optional[np.ndarray] = None
        self._last_grid_mask_2d: Optional[np.ndarray] = None
        self._last_test_coords: Optional[np.ndarray] = None
        self._last_grid_size: Optional[tuple] = None
        self._last_full_shape: int = 0
        self._last_used_few_shot: bool = True
        self._last_score: float = 0.0
        self._last_peak_score: float = 0.0
        self._last_defect_area_pct: float = 0.0
        self._last_morphology_bonus: float = 0.0
        self._last_morphology: Optional[MorphologyResult] = None

    def load_model(self) -> None:
        """Loads DINOv2 neural backbone onto selected device."""
        if self.model is None:
            logger.info("Loading model %s on device %s...", self.model_name, self.device)
            self.model = get_model(self.model_name, self.device, smaller_edge_size=self.img_size)
            logger.info("Model loaded successfully.")

    def _build_mask(self, shape: tuple) -> Optional[np.ndarray]:
        """Builds rounded-corner mask for canonical patch."""
        if not self.masking_enabled:
            return None
        h, w = shape[:2]
        return rounded_rect_mask(
            h,
            w,
            corner_radius_frac=CORNER_RADIUS_FRAC,
            erode_frac=ERODE_FRAC,
            feather_px=FEATHER_PX,
        )

    def _preprocess_with_mask(self, img_path: str):
        """Loads, resizes, and masks image with guided CLAHE."""
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unable to load image: {img_path}")
        self._last_raw_bgr = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target_h, target_w = TARGET_SIZE
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        if (self.luma_matching_enabled and self._ref_luma_stats is not None
                and not self._loading_reference):
            img = match_luminance(img, *self._ref_luma_stats)

        mask = self._build_mask(img.shape)
        if self.clahe_enabled:
            if mask is not None:
                img = apply_clahe_masked(
                    img,
                    (mask > 0.5).astype(np.uint8) * 255,
                    clip_limit=CLAHE_CLIP_LIMIT,
                    tile_grid_size=CLAHE_TILE_GRID,
                )
            else:
                img = apply_clahe(img, clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=CLAHE_TILE_GRID)
        return img, mask

    def load_reference_images(self, image_paths: list[str], card_id: Optional[str] = None) -> None:
        """Extracts descriptors from reference images and populates the memory bank.

        Supports Dual-Bank mode:
        - One-Shot: only the first canonical scan (pokemontcg.io)
        - Few-Shot: 20 realistic photometric variants + 1 base scan (21 references total)

        When disk cache is available (``feature_cache_dir``), DINOv2 descriptors
        from deterministic few-shot variants are reused without re-running neural inference.

        Args:
            image_paths: List of paths to official reference images.
            card_id: Identifier of the associated card.
        """
        self.load_model()

        if card_id and card_id != self.current_card_id:
            self.ref_features = None
            self.ref_features_oneshot = None
            self.ref_features_fewshot = None
            self.ref_coords = None
            self.ref_coords_oneshot = None
            self.ref_coords_fewshot = None
            self.memory_bank_ready = False
            self.current_card_id = card_id
            self.few_shot_variants = []
            self._ref_luma_stats = None
            logger.info("Updated card_id: %s. Memory bank reset.", card_id)

        few_shot_paths = []
        if card_id and image_paths:
            try:
                few_shot_paths = get_or_create_few_shot_variants(card_id, image_paths[0])
                self.few_shot_variants = few_shot_paths[1:]
            except Exception as e:
                logger.warning("Few-shot variant generation failed: %s. Using one-shot only.", e)
                few_shot_paths = image_paths

        effective_paths = few_shot_paths if few_shot_paths else image_paths

        if self.luma_matching_enabled and image_paths:
            try:
                base_img = cv2.cvtColor(cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                th, tw = TARGET_SIZE
                base_img = cv2.resize(base_img, (tw, th), interpolation=cv2.INTER_LANCZOS4)
                self._ref_luma_stats = compute_luma_stats(base_img)
                logger.info("Luma matching target (mean=%.1f, std=%.1f)", *self._ref_luma_stats)
            except Exception as e:
                logger.warning("Luma stats computation failed: %s. Luma matching disabled.", e)
                self._ref_luma_stats = None

        features_list = []
        coords_list = []
        grid_size_ref: Optional[tuple] = None
        mask_hash_ref: Optional[str] = None

        self._loading_reference = True
        for img_path in effective_paths:
            label = Path(img_path).stem if img_path else "base"
            try:
                img, mask = self._preprocess_with_mask(img_path)
            except Exception as e:
                logger.warning("Impossibile caricare reference %s: %s", img_path, e)
                continue

            img_tensor, grid_size = self.model.prepare_image(img)
            if grid_size_ref is None:
                grid_size_ref = grid_size
            elif grid_size_ref != grid_size:
                logger.warning("Grid size incoerente tra varianti: %s vs %s", grid_size_ref, grid_size)
                grid_size_ref = grid_size

            if mask is not None:
                grid_mask_2d = mask_to_grid(mask, grid_size)
            else:
                grid_mask_2d = np.ones(grid_size, dtype=bool)

            grid_mask_flat = grid_mask_2d.reshape(-1)
            mask_hash = hash_mask(mask)
            mask_hash_ref = mask_hash

            coords_all = build_patch_coordinates(grid_size)
            if mask is not None and grid_mask_flat.any():
                coords_valid = coords_all[grid_mask_flat]
            else:
                coords_valid = coords_all

            cached = None
            if card_id is not None and self.feature_cache_dir:
                cached = load_cached_features(
                    self.feature_cache_dir, card_id, label, grid_size, self.model_name
                )
                if cached is not None and cached.shape[0] != coords_valid.shape[0]:
                    logger.info(
                        "Cache %s incompatibile (%d vs %d feature): ricalcolo.",
                        label, cached.shape[0], coords_valid.shape[0]
                    )
                    cached = None

            if cached is not None:
                feats = cached.astype(np.float32, copy=False)
                logger.info("Cache HIT: %d feature per %s", feats.shape[0], label)
            else:
                try:
                    features = self.model.extract_features(img_tensor)
                except Exception as e:
                    logger.warning("Estrazione DINOv2 fallita per %s: %s", img_path, e)
                    continue

                if mask is not None and grid_mask_flat.any():
                    feats = features[grid_mask_flat].astype(np.float32, copy=False)
                else:
                    feats = features.astype(np.float32, copy=False)

                if card_id is not None and self.feature_cache_dir:
                    save_cached_features(
                        self.feature_cache_dir, card_id, label, grid_size, feats, self.model_name
                    )

            features_list.append(feats)
            coords_list.append(coords_valid)
            logger.info("Reference %s: %d feature.", label, feats.shape[0])
        self._loading_reference = False

        if not features_list:
            raise ValueError("No valid reference images available for computation.")

        feats_oneshot = features_list[0].astype(np.float32)
        coords_oneshot = coords_list[0].astype(np.float32, copy=False)
        t_oneshot = torch.from_numpy(feats_oneshot).to(self.device)
        self.ref_features_oneshot = torch.nn.functional.normalize(t_oneshot, p=2, dim=1)
        self.ref_coords_oneshot = coords_oneshot

        feats_fewshot = np.concatenate(features_list, axis=0).astype(np.float32)
        coords_fewshot = np.concatenate(coords_list, axis=0).astype(np.float32, copy=False)
        t_fewshot = torch.from_numpy(feats_fewshot).to(self.device)
        self.ref_features_fewshot = torch.nn.functional.normalize(t_fewshot, p=2, dim=1)
        self.ref_coords_fewshot = coords_fewshot

        self.ref_features = self.ref_features_fewshot if self.use_few_shot else self.ref_features_oneshot
        self.ref_coords = self.ref_coords_fewshot if self.use_few_shot else self.ref_coords_oneshot
        self.reference_count = len(features_list) if self.use_few_shot else 1
        self.memory_bank_ready = True

        logger.info(
            "Memory bank configurato: One-Shot=%d feature, Few-Shot=%d feature (%d reference totali).",
            self.ref_features_oneshot.shape[0],
            self.ref_features_fewshot.shape[0],
            len(features_list),
        )

    def _score_test_tensor(
        self,
        test_tensor: torch.Tensor,
        test_coords: np.ndarray,
        grid_mask_flat: Optional[np.ndarray],
        grid_size: tuple,
        full_shape: int,
        active_ref: torch.Tensor,
        active_ref_coords: Optional[np.ndarray],
    ) -> dict:
        """Calcola distanze spaziali, scoring composto e morfologia dei difetti.

        Restituisce un dizionario con tutti i campi derivati dal confronto fra
        ``test_tensor`` e ``active_ref``, applicando spatial-aware k-NN,
        dead zone adattiva per regione e clustering morfologico dei difetti.
        """
        if active_ref_coords is None:
            coords_ref = build_patch_coordinates(grid_size)
            if grid_mask_flat is not None and grid_mask_flat.any():
                coords_ref = coords_ref[grid_mask_flat]
        else:
            coords_ref = active_ref_coords

        if self.spatial_config.enabled and self.spatial_config.penalty_weight > 0.0:
            distances = compute_spatial_knn_distances(
                test_tensor,
                active_ref,
                test_coords.astype(np.float32, copy=False),
                coords_ref.astype(np.float32, copy=False),
                self.spatial_config,
            )
        else:
            with torch.inference_mode():
                sim = torch.mm(test_tensor, active_ref.T)
                if self.knn_neighbors == 1:
                    max_sim, _ = sim.max(dim=1)
                else:
                    top_sim, _ = sim.topk(self.knn_neighbors, dim=1)
                    max_sim = top_sim.mean(dim=1)
                distances = (1.0 - max_sim).clamp(min=0.0).cpu().numpy().astype(np.float32)

        full_dists = np.zeros(full_shape, dtype=np.float32)
        if grid_mask_flat is not None:
            full_dists[grid_mask_flat] = distances
        else:
            full_dists = distances
        d_map = full_dists.reshape(grid_size)

        grid_mask_2d = (
            grid_mask_flat.reshape(grid_size)
            if grid_mask_flat is not None
            else np.ones(grid_size, dtype=bool)
        )

        effective_threshold = float(self.defect_pixel_threshold)
        defect_frac = float(np.mean(distances > effective_threshold))
        peak_score = float(mean_top1p(distances.flatten()))
        morph = analyze_defect_morphology(
            d_map,
            grid_mask_2d,
            defect_threshold=effective_threshold,
            config=self.morphology_config,
        )
        score = float(
            peak_score
            + self.defect_area_weight * defect_frac
            + morph.morphology_bonus
        )

        return {
            "d_map": d_map,
            "grid_mask_2d": grid_mask_2d,
            "peak_score": peak_score,
            "defect_frac": defect_frac,
            "score": score,
            "morphology": morph,
        }

    def analyze_card(
        self,
        test_image_path: str,
        dead_zone: Optional[float] = None,
        gamma: Optional[float] = None,
        use_few_shot: Optional[bool] = None,
    ) -> dict:
        """Executes complete inspection on query image comparing against memory bank.

        Args:
            test_image_path: Path to query image file.
            dead_zone: Instance-specific Dead Zone parameter.
            gamma: Instance-specific Gamma exponent.
            use_few_shot: If True uses 10-variant Few-Shot bank, else One-Shot.

        Returns:
            Dictionary containing anomaly metrics, condition grading, and Base64 heatmap.
        """
        if not self.memory_bank_ready or self.ref_features is None:
            raise RuntimeError("Memory bank uninitialized. Load reference card first.")

        if use_few_shot is not None:
            self.use_few_shot = use_few_shot

        active_ref = (
            self.ref_features_fewshot
            if self.use_few_shot and self.ref_features_fewshot is not None
            else (self.ref_features_oneshot if self.ref_features_oneshot is not None else self.ref_features)
        )
        active_ref_coords = (
            self.ref_coords_fewshot
            if self.use_few_shot and self.ref_coords_fewshot is not None
            else (self.ref_coords_oneshot if self.ref_coords_oneshot is not None else self.ref_coords)
        )

        self.load_model()
        img, mask = self._preprocess_with_mask(test_image_path)

        with torch.inference_mode():
            img_tensor, grid_size = self.model.prepare_image(img)
            features_test = self.model.extract_features(img_tensor)

            grid_mask_flat = None
            grid_mask_2d = None
            if mask is not None:
                grid_mask_2d = mask_to_grid(mask, grid_size)
                grid_mask_flat = grid_mask_2d.reshape(-1)
                if grid_mask_flat.sum() == 0:
                    grid_mask_flat = None
                    grid_mask_2d = None

            if grid_mask_flat is not None:
                feats_to_score = features_test[grid_mask_flat]
            else:
                feats_to_score = features_test

            test_tensor = torch.from_numpy(feats_to_score.astype(np.float32)).to(self.device)
            test_tensor = torch.nn.functional.normalize(test_tensor, p=2, dim=1)

            coords_all = build_patch_coordinates(grid_size)
            test_coords = (
                coords_all[grid_mask_flat]
                if grid_mask_flat is not None
                else coords_all
            )

            scoring = self._score_test_tensor(
                test_tensor=test_tensor,
                test_coords=test_coords,
                grid_mask_flat=grid_mask_flat,
                grid_size=grid_size,
                full_shape=features_test.shape[0],
                active_ref=active_ref,
                active_ref_coords=active_ref_coords,
            )
            d_map = scoring["d_map"]
            score = scoring["score"]
            peak_score = scoring["peak_score"]
            defect_frac = scoring["defect_frac"]
            morph = scoring["morphology"]

            h, w = img.shape[:2]
            anomaly_map_full = cv2.resize(
                d_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LANCZOS4
            )
            if mask is not None:
                anomaly_map_full = anomaly_map_full * mask

            score = float(
                peak_score
                + self.defect_area_weight * defect_frac
                + morph.morphology_bonus
            )

            dz = dead_zone if dead_zone is not None else self.dead_zone
            g = gamma if gamma is not None else self.gamma

            overlay = heatmap_overlay(
                img,
                anomaly_map_full,
                valid_mask=mask,
                dead_zone=dz,
                gamma=g,
                adaptive_alpha=True,
                bilateral_smoothing=True,
            )

            self._last_patch = img.copy()
            self._last_d_map = d_map.copy()
            self._last_mask = mask.copy() if mask is not None else None
            self._last_test_tensor = test_tensor.clone()
            self._last_grid_mask_flat = grid_mask_flat
            self._last_grid_mask_2d = grid_mask_2d
            self._last_test_coords = test_coords.astype(np.float32, copy=False)
            self._last_grid_size = grid_size
            self._last_full_shape = features_test.shape[0]
            self._last_used_few_shot = self.use_few_shot
            self._last_score = score
            self._last_peak_score = peak_score
            self._last_defect_area_pct = round(defect_frac * 100.0, 1)
            self._last_morphology_bonus = float(morph.morphology_bonus)
            self._last_morphology = morph

        is_anomaly = score > self.anomaly_threshold
        cond = classify_condition(score, self.condition_bands)

        return {
            "score": score,
            "peak_score": peak_score,
            "defect_area_pct": round(defect_frac * 100.0, 1),
            "threshold": self.anomaly_threshold,
            "is_anomaly": is_anomaly,
            "verdict": "DEFECTIVE" if is_anomaly else "OK",
            "condition": {
                "label": cond.label,
                "tag": cond.tag,
                "description": cond.description,
                "color": cond.color,
                "band_low": cond.band_low,
                "band_high": cond.band_high,
            },
            "morphology": {
                "cluster_count": morph.cluster_count,
                "max_cluster_area": morph.max_cluster_area,
                "mean_cluster_area": round(morph.mean_cluster_area, 2),
                "linear_cluster_count": morph.linear_cluster_count,
                "total_defect_patches": morph.total_defect_patches,
                "morphology_bonus": round(morph.morphology_bonus, 4),
            },
            "spatial_search": {
                "enabled": self.spatial_config.enabled,
                "radius_patches": self.spatial_config.radius_patches,
                "penalty_weight": self.spatial_config.penalty_weight,
            },
            "clahe_enabled": self.clahe_enabled,
            "masking_enabled": self.masking_enabled,
            "dead_zone": dz,
            "gamma": g,
            "few_shot": self.use_few_shot,
            "reference_count": (1 + len(self.few_shot_variants)) if (self.use_few_shot and self.few_shot_variants) else 1,
            "total_features": int(active_ref.shape[0]),
            "heatmap_b64": ndarray_to_base64(overlay),
            "original_b64": ndarray_to_base64(img),
            "grid_anomaly_map": d_map.tolist(),
            "grid_size": list(grid_size),
        }

    def analyze_patch(
        self,
        patch_rgb: np.ndarray,
        dead_zone: Optional[float] = None,
        gamma: Optional[float] = None,
        use_few_shot: Optional[bool] = None,
    ) -> dict:
        """Analyzes a canonical RGB patch produced by OBB warp.

        The OBB corner ordering is rotation-ambiguous, so when orientation
        canonicalization is enabled all four in-plane orientations (0/90/180/270)
        are scored and the orientation with the lowest anomaly score is kept.
        This makes the inspection invariant to how the card was photographed.
        """
        import tempfile

        best: Optional[dict] = None
        best_rot = 0
        last_rot = -1
        rotations = range(4) if self.orientation_canonicalization else range(1)
        for rot in rotations:
            p = np.rot90(patch_rgb, rot) if rot else patch_rgb
            if p.shape[0] < p.shape[1]:
                continue
            last_rot = rot
            with tempfile.TemporaryDirectory() as td:
                pa = Path(td) / "patch.png"
                cv2.imwrite(str(pa), cv2.cvtColor(np.ascontiguousarray(p), cv2.COLOR_RGB2BGR))
                result = self.analyze_card(str(pa), dead_zone=dead_zone, gamma=gamma, use_few_shot=use_few_shot)
            result["flipped"] = rot != 0
            result["rotation_index"] = rot
            if best is None or result["score"] < best["score"]:
                best = result
                best_rot = rot

        if best is not None and last_rot != best_rot:
            p = np.rot90(patch_rgb, best_rot) if best_rot else patch_rgb
            with tempfile.TemporaryDirectory() as td:
                pa = Path(td) / "patch.png"
                cv2.imwrite(str(pa), cv2.cvtColor(np.ascontiguousarray(p), cv2.COLOR_RGB2BGR))
                best = self.analyze_card(str(pa), dead_zone=dead_zone, gamma=gamma, use_few_shot=use_few_shot)
            best["flipped"] = best_rot != 0
            best["rotation_index"] = best_rot

        return best

    def re_render_heatmap(
        self,
        dead_zone: Optional[float] = None,
        gamma: Optional[float] = None,
        few_shot: Optional[bool] = None,
    ) -> Optional[dict]:
        """Ricalcola la mappa termica sull'ultimo ritaglio memorizzato aggiornando Dead Zone, Gamma e Few-Shot."""
        if self._last_patch is None or self._last_d_map is None:
            return None

        dz = dead_zone if dead_zone is not None else self.dead_zone
        g = gamma if gamma is not None else self.gamma
        self.dead_zone = dz
        self.gamma = g

        if few_shot is not None and few_shot != self._last_used_few_shot:
            self.use_few_shot = few_shot
            if self._last_raw_bgr is not None:
                import tempfile

                with tempfile.TemporaryDirectory() as td:
                    pa = Path(td) / "patch.png"
                    cv2.imwrite(str(pa), self._last_raw_bgr)
                    self.analyze_card(str(pa), dead_zone=dz, gamma=g, use_few_shot=few_shot)
                self._last_used_few_shot = few_shot

        h, w = self._last_patch.shape[:2]
        anomaly_map_full = cv2.resize(
            self._last_d_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LANCZOS4
        )
        if self._last_mask is not None:
            anomaly_map_full = anomaly_map_full * self._last_mask

        overlay = heatmap_overlay(
            self._last_patch,
            anomaly_map_full,
            valid_mask=self._last_mask,
            dead_zone=dz,
            gamma=g,
            adaptive_alpha=True,
            bilateral_smoothing=True,
        )

        is_anomaly = self._last_score > self.anomaly_threshold
        cond = classify_condition(self._last_score, self.condition_bands)
        morph = self._last_morphology

        return {
            "heatmap_b64": ndarray_to_base64(overlay),
            "dead_zone": dz,
            "gamma": g,
            "few_shot": self.use_few_shot,
            "score": self._last_score,
            "peak_score": self._last_peak_score,
            "defect_area_pct": self._last_defect_area_pct,
            "threshold": self.anomaly_threshold,
            "is_anomaly": is_anomaly,
            "verdict": "DEFECTIVE" if is_anomaly else "OK",
            "condition": {
                "label": cond.label,
                "tag": cond.tag,
                "description": cond.description,
                "color": cond.color,
                "band_low": cond.band_low,
                "band_high": cond.band_high,
            },
            "morphology": {
                "cluster_count": morph.cluster_count if morph else 0,
                "max_cluster_area": morph.max_cluster_area if morph else 0,
                "mean_cluster_area": round(morph.mean_cluster_area, 2) if morph else 0.0,
                "linear_cluster_count": morph.linear_cluster_count if morph else 0,
                "total_defect_patches": morph.total_defect_patches if morph else 0,
                "morphology_bonus": round(morph.morphology_bonus, 4) if morph else 0.0,
            },
        }

    def set_threshold(self, threshold: float) -> None:
        """Updates decision threshold for anomaly classification."""
        self.anomaly_threshold = threshold
        logger.info("Anomaly threshold set to: %f", threshold)

    @property
    def condition_bands(self) -> dict:
        """Returns calibration bands for qualitative condition tiers."""
        bands = dict(_CFG["condition"]["bands"])
        return bands

    @property
    def status(self) -> dict:
        """Returns current operational status of the detector."""
        return {
            "model": self.model_name,
            "device": self.device,
            "memory_bank_ready": self.memory_bank_ready,
            "reference_count": self.reference_count,
            "few_shot": self.use_few_shot,
            "few_shot_variants_count": len(self.few_shot_variants),
            "current_card_id": self.current_card_id,
            "anomaly_threshold": self.anomaly_threshold,
            "defect_area_weight": self.defect_area_weight,
            "defect_pixel_threshold": self.defect_pixel_threshold,
            "model_loaded": self.model is not None,
            "clahe_enabled": self.clahe_enabled,
            "masking_enabled": self.masking_enabled,
            "dead_zone": self.dead_zone,
            "gamma": self.gamma,
            "spatial_search": {
                "enabled": self.spatial_config.enabled,
                "radius_patches": self.spatial_config.radius_patches,
                "penalty_weight": self.spatial_config.penalty_weight,
            },
            "morphology": {
                "enabled": self.morphology_config.enabled,
                "large_cluster_threshold": self.morphology_config.large_cluster_threshold,
                "cluster_penalty": self.morphology_config.cluster_penalty,
                "linear_eccentricity": self.morphology_config.linear_eccentricity,
                "linear_penalty": self.morphology_config.linear_penalty,
            },
            "feature_cache_dir": self.feature_cache_dir,
        }

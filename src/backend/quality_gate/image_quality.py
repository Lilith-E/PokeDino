"""Stage 2: Detailed image quality gate on YOLO OBB crop."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.quality_gate.helper import Helper, config, logger

@dataclass
class QualityMetrics:
    """Per-metric quality scores and hard-reject flags.
    
    Attributes:
        sharpness: Normalized sharpness score [0, 1].
        exposure: Normalized exposure score [0, 1].
        glare: Normalized glare score [0, 1] (1 = no glare, 0 = full glare).
        noise: Normalized noise score [0, 1] (1 = no noise, 0 = high noise).
        resolution: Normalized resolution score [0, 1].
        detection_confidence: YOLO detection confidence [0, 1].
        aspect_ratio_score: Aspect ratio conformity score [0, 1].
        relative_area_score: Relative area score [0, 1].
        hard_rejects: List of metric names that triggered hard rejection.
    """
    
    sharpness: float = 0.0
    exposure: float = 0.0
    glare: float = 0.0
    noise: float = 0.0
    resolution: float = 0.0
    detection_confidence: float = 0.0
    aspect_ratio_score: float = 0.0
    relative_area_score: float = 0.0
    hard_rejects: List[str] = field(default_factory=list)

@dataclass
class QualityResult:
    """Complete quality gate result.
    
    Attributes:
        score: Aggregate quality score [0, 1].
        passed: Whether the image passed all checks.
        metrics: Per-metric breakdown.
        reject_reasons: List of rejection reasons (empty if passed).
    """
    
    score: float = 0.0
    passed: bool = False
    metrics: QualityMetrics = field(default_factory=QualityMetrics)
    reject_reasons: List[str] = field(default_factory=list)

class ImageQualityGate:
    """Detailed quality assessment on YOLO OBB crop.
    
    Produces a continuous quality score in [0, 1] plus per-metric breakdown
    and hard-reject flags. Evaluated on the rectified/cropped card image only.
    
    Attributes:
        weights: Metric weights for aggregation.
        hard_reject_thresholds: Per-metric hard-reject thresholds.
        min_resolution: Minimum crop dimensions (height, width).
        expected_aspect_ratio: Expected card aspect ratio (height/width).
        aspect_ratio_tolerance: Allowed deviation from expected aspect ratio.
    """
    
    EXPECTED_ASPECT_RATIO = 0.714
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        hard_reject_thresholds: Optional[Dict[str, float]] = None,
        min_resolution: Optional[Tuple[int, int]] = None,
        expected_aspect_ratio: Optional[float] = None,
        aspect_ratio_tolerance: Optional[float] = None,
    ):
        """Initialize with configurable parameters.
        
        Args:
            weights: Metric weights. Defaults to config values.
            hard_reject_thresholds: Hard-reject thresholds. Defaults to config.
            min_resolution: Minimum (height, width). Defaults to config.
            expected_aspect_ratio: Expected aspect ratio. Defaults to config.
            aspect_ratio_tolerance: Aspect ratio tolerance. Defaults to config.
        """
        qg_config = config.get("quality_gate", {}).get("image_quality", {})
        
        self.weights = weights if weights is not None else qg_config.get("weights", {
            "sharpness": 0.35,
            "exposure": 0.20,
            "glare": 0.10,
            "noise": 0.15,
            "resolution": 0.10,
            "detection_confidence": 0.05,
            "aspect_ratio": 0.03,
            "relative_area": 0.02,
        })
        
        self.hard_reject_thresholds = hard_reject_thresholds if hard_reject_thresholds is not None else qg_config.get("hard_reject", {
            "sharpness": 0.15,
            "exposure": 0.10,
            "glare_coverage": 0.50,
            "noise": 0.10,
            "resolution": 0.0,
            "detection_confidence": 0.30,
            "aspect_ratio": 0.30,
            "relative_area": 0.45,
        })
        
        self.min_resolution = min_resolution if min_resolution is not None else tuple(qg_config.get("min_resolution", [448, 630]))
        self.expected_aspect_ratio = expected_aspect_ratio if expected_aspect_ratio is not None else qg_config.get("expected_aspect_ratio", self.EXPECTED_ASPECT_RATIO)
        self.aspect_ratio_tolerance = aspect_ratio_tolerance if aspect_ratio_tolerance is not None else qg_config.get("aspect_ratio_tolerance", 0.30)
    
    def _compute_sharpness(self, image: np.ndarray) -> float:
        """Compute sharpness via Laplacian variance AND blur ratio.
        
        Blur ratio compares gradients of original vs downsampled-rescaled image:
        if downsampling loses little detail, the original was already blurred.
        
        Args:
            image: Input image (BGR, uint8).
            
        Returns:
            Normalized sharpness score [0, 1].
        """
        gray_u8 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray_u8.astype(np.float32)
        
        laplacian_var = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())
        
        h, w = gray.shape
        small = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        small = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        
        grad_orig = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())
        grad_down = float(cv2.Laplacian(small.astype(np.uint8), cv2.CV_64F).var())
        
        blur_ratio = grad_orig / (grad_down + 1e-8)
        
        blur_score = float(np.clip((blur_ratio - 1.0) / 2.0, 0.0, 1.0))
        
        lap_score = float(np.clip(laplacian_var / 500.0, 0.0, 1.0))
        
        return 0.5 * lap_score + 0.5 * blur_score
    
    def _compute_exposure(self, image: np.ndarray) -> float:
        """Compute exposure quality based on clipped pixels and mean luminance.
        
        More aggressive penalty for clipping: clipped pixels are unrecoverable
        detail loss, unlike a simple mean shift.
        
        Args:
            image: Input image (BGR, uint8).
            
        Returns:
            Normalized exposure score [0, 1].
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        clipped_low = np.sum(gray < 10)
        clipped_high = np.sum(gray > 245)
        total_pixels = gray.size
        clipped_ratio = (clipped_low + clipped_high) / total_pixels
        
        mean_lum = float(np.mean(gray))
        mean_deviation = abs(mean_lum - 128) / 128
        
        clipped_score = 1.0 - np.clip(clipped_ratio * 10.0, 0.0, 1.0)
        
        mean_score = 1.0 - np.clip(mean_deviation / 0.5, 0.0, 1.0)
        
        return 0.6 * clipped_score + 0.4 * mean_score
    
    def _compute_glare(self, image: np.ndarray) -> float:
        """Compute glare coverage using HSV-based detection.
        
        Only blown-out specular highlights are counted (very high Value, low
        Saturation). The high Value threshold avoids false positives on white
        card borders, pale artwork and light backgrounds, which are bright but
        not clipped by the sensor.
        
        Args:
            image: Input image (BGR, uint8).
            
        Returns:
            Normalized glare score [0, 1] (1 = no glare, 0 = full glare).
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        glare_mask = cv2.inRange(hsv, (0, 0, 240), (180, 40, 255))
        glare_coverage = float(np.sum(glare_mask > 0)) / glare_mask.size
        
        return 1.0 - Helper.clamp(glare_coverage * 2, 0, 1)
    
    def _compute_noise(self, image: np.ndarray) -> float:
        """Compute noise level as std of residuals after Gaussian blur.
        
        Args:
            image: Input image (BGR, uint8).
            
        Returns:
            Normalized noise score [0, 1] (1 = no noise, 0 = high noise).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        residuals = gray - blurred
        noise_std = float(np.std(residuals))
        
        noise_score = 1.0 - Helper.normalize(noise_std, 0, 30)
        return noise_score
    
    def _compute_resolution(self, image: np.ndarray) -> float:
        """Compute resolution score based on crop dimensions.
        
        Args:
            image: Input image (BGR, uint8).
            
        Returns:
            Normalized resolution score [0, 1].
        """
        h, w = image.shape[:2]
        min_h, min_w = self.min_resolution
        
        h_score = Helper.clamp(h / min_h, 0, 1)
        w_score = Helper.clamp(w / min_w, 0, 1)
        
        return (h_score + w_score) / 2.0
    
    def _compute_framing(
        self,
        detection_confidence: float,
        box_aspect_ratio: float,
        box_area_ratio: float,
    ) -> Tuple[float, float, float]:
        """Compute framing metrics from YOLO OBB output.
        
        Args:
            detection_confidence: YOLO detection confidence [0, 1].
            box_aspect_ratio: Aspect ratio of oriented box (height/width).
            box_area_ratio: Box area / frame area [0, 1].
            
        Returns:
            Tuple of (confidence_score, aspect_ratio_score, relative_area_score).
        """
        confidence_score = detection_confidence
        
        aspect_deviation = abs(box_aspect_ratio - self.expected_aspect_ratio) / self.expected_aspect_ratio
        aspect_score = 1.0 - Helper.clamp(aspect_deviation / self.aspect_ratio_tolerance, 0, 1)
        
        area_score = Helper.clamp(box_area_ratio * 3, 0, 1)
        
        return confidence_score, aspect_score, area_score
    
    def _check_hard_rejects(self, metrics: QualityMetrics) -> List[str]:
        """Check for hard-reject conditions.
        
        Args:
            metrics: Per-metric scores.
            
        Returns:
            List of metric names that triggered hard rejection.
        """
        hard_rejects = []
        
        if metrics.sharpness < self.hard_reject_thresholds.get("sharpness", 0.10):
            hard_rejects.append("sharpness")
        
        if metrics.exposure < self.hard_reject_thresholds.get("exposure", 0.10):
            hard_rejects.append("exposure")
        
        glare_coverage = 1.0 - metrics.glare
        if glare_coverage > self.hard_reject_thresholds.get("glare_coverage", 0.50):
            hard_rejects.append("glare")
        
        if metrics.noise < self.hard_reject_thresholds.get("noise", 0.10):
            hard_rejects.append("noise")
        
        if metrics.resolution < self.hard_reject_thresholds.get("resolution", 0.0):
            hard_rejects.append("resolution")
        
        if metrics.detection_confidence < self.hard_reject_thresholds.get("detection_confidence", 0.30):
            hard_rejects.append("detection_confidence")
        
        if metrics.aspect_ratio_score < self.hard_reject_thresholds.get("aspect_ratio", 0.30):
            hard_rejects.append("aspect_ratio")
        
        if metrics.relative_area_score < self.hard_reject_thresholds.get("relative_area", 0.05):
            hard_rejects.append("relative_area")
        
        return hard_rejects
    
    def evaluate(
        self,
        image: np.ndarray,
        detection_confidence: float = 1.0,
        box_aspect_ratio: float = None,
        box_area_ratio: float = 0.5,
    ) -> QualityResult:
        """Evaluate image quality on YOLO OBB crop.
        
        Args:
            image: Cropped card image (BGR, uint8).
            detection_confidence: YOLO detection confidence [0, 1].
            box_aspect_ratio: Aspect ratio of oriented box (height/width).
            box_area_ratio: Box area / frame area [0, 1].
            
        Returns:
            QualityResult with aggregate score, pass/fail, and per-metric breakdown.
            
        Examples:
            >>> gate = ImageQualityGate()
            >>> img = cv2.imread("crop.jpg")
            >>> result = gate.evaluate(img, detection_confidence=0.95)
            >>> if result.passed:
            ...     print(f"Quality: {result.score:.3f}")
        """
        if image is None or image.size == 0:
            return QualityResult(score=0.0, passed=False, reject_reasons=["Empty image"])
        
        metrics = QualityMetrics()
        metrics.sharpness = self._compute_sharpness(image)
        metrics.exposure = self._compute_exposure(image)
        metrics.glare = self._compute_glare(image)
        metrics.noise = self._compute_noise(image)
        metrics.resolution = self._compute_resolution(image)
        metrics.detection_confidence = detection_confidence
        
        if box_aspect_ratio is None:
            h, w = image.shape[:2]
            box_aspect_ratio = h / w if w > 0 else 1.0
        
        conf_score, aspect_score, area_score = self._compute_framing(
            detection_confidence, box_aspect_ratio, box_area_ratio
        )
        metrics.aspect_ratio_score = aspect_score
        metrics.relative_area_score = area_score
        
        metrics.hard_rejects = self._check_hard_rejects(metrics)
        
        score = (
            metrics.sharpness * self.weights.get("sharpness", 0.35)
            + metrics.exposure * self.weights.get("exposure", 0.20)
            + metrics.glare * self.weights.get("glare", 0.10)
            + metrics.noise * self.weights.get("noise", 0.15)
            + metrics.resolution * self.weights.get("resolution", 0.10)
            + metrics.detection_confidence * self.weights.get("detection_confidence", 0.05)
            + metrics.aspect_ratio_score * self.weights.get("aspect_ratio", 0.03)
            + metrics.relative_area_score * self.weights.get("relative_area", 0.02)
        )
        
        score = float(score)
        passed = bool(len(metrics.hard_rejects) == 0 and score >= 0.35)
        
        reject_reasons = []
        if metrics.hard_rejects:
            reject_reasons.extend([f"Hard reject: {m}" for m in metrics.hard_rejects])
        if score < 0.35:
            reject_reasons.append(f"Low aggregate score: {score:.3f}")
        
        return QualityResult(
            score=score,
            passed=passed,
            metrics=metrics,
            reject_reasons=reject_reasons,
        )

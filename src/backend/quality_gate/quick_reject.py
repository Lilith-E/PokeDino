"""Stage 1: Quick reject filter for raw images before YOLO inference."""

from typing import Tuple

import cv2
import numpy as np

from backend.quality_gate.helper import config, logger

class QuickRejectRaw:
    """Fast fail-filter for catastrophic frames before expensive YOLO inference.
    
    Attributes:
        min_pixel_std: Minimum global pixel standard deviation to reject near-uniform frames.
        min_laplacian_var: Minimum Laplacian variance to reject extremely blurred frames.
        min_width: Minimum width in pixels to reject ultra-low-resolution frames.
        min_height: Minimum height in pixels to reject ultra-low-resolution frames.
    """
    
    def __init__(
        self,
        min_pixel_std: float = None,
        min_laplacian_var: float = None,
        min_width: int = None,
        min_height: int = None,
    ):
        """Initialize with configurable thresholds.
        
        Args:
            min_pixel_std: Minimum pixel std dev. Defaults to config value.
            min_laplacian_var: Minimum Laplacian variance. Defaults to config value.
            min_width: Minimum width in pixels. Defaults to config value.
            min_height: Minimum height in pixels. Defaults to config value.
        """
        qg_config = config.get("quality_gate", {}).get("quick_reject", {})
        self.min_pixel_std = min_pixel_std if min_pixel_std is not None else qg_config.get("min_pixel_std", 10.0)
        self.min_laplacian_var = min_laplacian_var if min_laplacian_var is not None else qg_config.get("min_laplacian_var", 15.0)
        self.min_width = min_width if min_width is not None else int(qg_config.get("min_width", 300))
        self.min_height = min_height if min_height is not None else int(qg_config.get("min_height", 300))
    
    def evaluate(self, image: np.ndarray) -> Tuple[bool, str]:
        """Evaluate raw image for catastrophic quality issues.
        
        Args:
            image: Raw input image (BGR, uint8).
            
        Returns:
            Tuple of (pass: bool, reason: str). If pass is True, reason is empty.
        """
        if image is None or image.size == 0:
            return False, "Empty or invalid image"
        
        h, w = image.shape[:2]
        
        if w < self.min_width or h < self.min_height:
            logger.info(f"Quick reject: image resolution {w}x{h} < {self.min_width}x{self.min_height}")
            return False, f"Image resolution is too low ({w}x{h} px). Please retry with a higher resolution photograph."
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        pixel_std = float(np.std(gray))
        if pixel_std < self.min_pixel_std:
            logger.info(f"Quick reject: pixel std {pixel_std:.2f} < {self.min_pixel_std}")
            return False, f"Near-uniform frame (std={pixel_std:.2f})"
        
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < self.min_laplacian_var:
            logger.info(f"Quick reject: Laplacian var {laplacian_var:.2f} < {self.min_laplacian_var}")
            return False, f"Extreme blur (Laplacian={laplacian_var:.2f})"
        
        return True, ""

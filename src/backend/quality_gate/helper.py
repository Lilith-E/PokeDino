"""Shared utilities for the quality gate package."""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_config_path = _PROJECT_ROOT / "configs" / "default.yaml"
if not _config_path.exists():
    _config_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

with open(_config_path, "r") as _f:
    config: Dict[str, Any] = yaml.safe_load(_f)

logger: logging.Logger = logging.getLogger(__name__)

class Helper:
    """Utility class for common numeric operations."""

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp a value to [min_val, max_val] range."""
        return max(min_val, min(max_val, value))

    @staticmethod
    def normalize(value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to [0, 1] range."""
        if max_val <= min_val:
            return 0.0
        return Helper.clamp((value - min_val) / (max_val - min_val), 0.0, 1.0)

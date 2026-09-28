"""Module for managing and tracking experimental sessions.

Each run generates a dedicated directory in `experiments/` containing:
- Configuration snapshot (config.yaml).
- Serialized quantitative metrics (metrics.json).
- Execution log (log.txt).
- Visual figures and output artifacts.
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "configs").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

def load_config(path: Optional[str] = None) -> dict:
    """Load the central YAML config (optionally overridden by another file)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def resolve_path(cfg: dict, key_path: str) -> Path:
    """Resolve a path from config relative to the project root."""
    node: Any = cfg
    for part in key_path.split("."):
        node = node[part]
    p = Path(node)
    return p if p.is_absolute() else (PROJECT_ROOT / p)

class Experiment:
    """Creates and manages one experiment run folder."""

    def __init__(self, name: str, cfg: dict, config_path: Optional[str] = None):
        self.cfg = cfg
        self.name = name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = resolve_path(cfg, "paths.experiments_dir")
        self.run_dir = exp_dir / f"{self.timestamp}_{name}"
        self.plots_dir = self.run_dir / "plots"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        self.metrics: dict[str, Any] = {}
        self._t0 = time.time()

        self.logger = logging.getLogger(f"exp.{self.timestamp}.{name}")
        self.logger.setLevel(getattr(logging, cfg.get("experiment", {}).get("log_level", "INFO")))
        self.logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        fh = logging.FileHandler(self.run_dir / "log.txt")
        fh.setFormatter(fmt)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

        self.logger.info(f"Experiment '{name}' → {self.run_dir}")

    def log_metric(self, key: str, value: Any):
        """Record a numeric metric (nested keys with dots allowed)."""
        node = self.metrics
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
        self.logger.info(f"METRIC {key} = {value}")

    def log_metrics(self, d: dict, prefix: str = ""):
        """Records multiple metrics with an optional key prefix."""
        for k, v in d.items():
            self.log_metric(f"{prefix}{k}", v)

    def save_metrics(self):
        """Serializes metrics dictionary to metrics.json in the run directory."""
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)
        self.logger.info(f"Metrics saved → {self.run_dir / 'metrics.json'}")

    def save_plot(self, fig: plt.Figure, name: str):
        """Save a matplotlib figure into plots/ (and close it)."""
        out = self.plots_dir / f"{name}.png"
        fig.savefig(out, dpi=self.cfg.get("experiment", {}).get("plot_dpi", 150), bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Plot saved → {out}")
        return out

    def save_image(self, img_bgr_or_rgb, name: str, rgb: bool = True):
        """Save a numpy image (H,W,3) into plots/ via OpenCV."""
        import cv2

        arr = img_bgr_or_rgb
        if rgb:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        out = self.plots_dir / f"{name}.png"
        cv2.imwrite(str(out), arr)
        self.logger.info(f"Image saved → {out}")
        return out

    def artifact_path(self, name: str) -> Path:
        """Returns full filesystem path for a named file in the artifacts directory."""
        return self.artifacts_dir / name

    def copy_artifact(self, src: Path, name: Optional[str] = None):
        """Copies an external file into the experiment artifacts directory."""
        import shutil

        dst = self.artifacts_dir / (name or Path(src).name)
        shutil.copy2(src, dst)
        self.logger.info(f"Artifact copied → {dst}")
        return dst

    def finish(self):
        """Finalizes experiment run and records total execution duration."""
        elapsed = time.time() - self._t0
        self.metrics["_meta"] = {
            "name": self.name,
            "timestamp": self.timestamp,
            "duration_sec": round(elapsed, 1),
            "finished_at": datetime.now().isoformat(),
        }
        self.save_metrics()
        self.logger.info(f"Experiment finished in {elapsed:.1f}s")
        return self.run_dir

def set_seed(seed: int):
    """Reproducibility seeding for random, numpy and torch."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def get_device(cfg: dict) -> str:
    """Device selection honoring config (auto | cpu | mps | cuda).
    Env var POKE_DEVICE overrides the config (useful to pin scripts to CPU
    while another job holds the GPU)."""
    import os
    import torch

    pref = os.environ.get("POKE_DEVICE", cfg.get("project", {}).get("device", "auto"))
    if pref == "cpu":
        return "cpu"
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "mps" and torch.backends.mps.is_available():
        return "mps"
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"

"""Module for collectible card condition classification and grading.

Maps the computed visual anomaly score into three grading tiers:
- Near Mint/Mint (NM): Flawless to near-perfect condition.
- Good/Excellent (GE): Minor to visible wear.
- Poor/Played (PP): Moderate to heavy wear or structural defects.
"""

from dataclasses import dataclass
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DESCRIPTIONS = {
    "Near Mint/Mint": "Near Mint/Mint - Flawless to near-perfect: imperfections invisible or microscopic",
    "Good/Excellent": "Good/Excellent - Minor to visible wear: light edge whitening or light surface scratches",
    "Poor/Played": "Poor/Played - Moderate to heavy wear: creases, deep scratches, structural defects",
}

COLORS = {
    "Near Mint/Mint": "#22C55E",
    "Good/Excellent": "#EAB308",
    "Poor/Played": "#EF4444",
}

TAGS = {
    "Near Mint/Mint": "NM",
    "Good/Excellent": "GE",
    "Poor/Played": "PP",
}

ALIAS_TO_FULL = {
    "MT": "Near Mint/Mint",
    "MINT": "Near Mint/Mint",
    "NM": "Near Mint/Mint",
    "NEAR MINT": "Near Mint/Mint",
    "NEAR MINT/MINT": "Near Mint/Mint",
    "EX": "Good/Excellent",
    "EXCELLENT": "Good/Excellent",
    "GD": "Good/Excellent",
    "GOOD": "Good/Excellent",
    "GOOD/EXCELLENT": "Good/Excellent",
    "LP": "Poor/Played",
    "LIGHT PLAYED": "Poor/Played",
    "LIGHTLY PLAYED": "Poor/Played",
    "MP": "Poor/Played",
    "PL": "Poor/Played",
    "PLAYED": "Poor/Played",
    "HP": "Poor/Played",
    "HEAVILY PLAYED": "Poor/Played",
    "PO": "Poor/Played",
    "POOR": "Poor/Played",
    "POOR/PLAYED": "Poor/Played",
    "DMG": "Poor/Played",
    "DAMAGED": "Poor/Played",
}

LABELS = ["Near Mint/Mint", "Good/Excellent", "Poor/Played"]

@dataclass
class ConditionResult:
    """Result of qualitative condition grading."""
    label: str
    score: float
    band_low: float
    band_high: Optional[float]
    description: str
    color: str
    tag: str = ""

    @property
    def tier(self) -> str:
        """Alias for label to match condition tier terminology."""
        return self.label

def classify_condition(score: float, bands: Optional[dict] = None) -> ConditionResult:
    """Maps the anomaly score to the corresponding preservation grade.

    Supports both full labels (Near Mint/Mint, ...) and short tags (NM, ...),
    plus all legacy seven-tier labels and tags via ALIAS_TO_FULL.

    Args:
        score: Numerical anomaly score.
        bands: Dictionary containing upper thresholds for each tier
            (last tier must be unbounded / None). If None, loaded from default config.

    Returns:
        ConditionResult object containing label, tag, description, and tier boundaries.
    """
    if bands is None:
        try:
            from backend.experiment import load_config
            cfg = load_config()
            bands = cfg.get("condition", {}).get("bands", {})
        except Exception:
            bands = {"Near Mint/Mint": 0.450, "Good/Excellent": 0.750, "Poor/Played": None}
    prev = 0.0

    normalized_bands = {}
    for k, v in bands.items():
        canonical = ALIAS_TO_FULL.get(k.strip().upper(), k.strip())
        normalized_bands[canonical] = v
        tag = TAGS.get(canonical)
        if tag:
            normalized_bands[tag] = v

    for lab in LABELS[:-1]:
        hi = normalized_bands.get(lab)
        if hi is not None and score < hi:
            return ConditionResult(
                label=lab,
                score=score,
                band_low=prev,
                band_high=hi,
                description=DESCRIPTIONS.get(lab, ""),
                color=COLORS.get(lab, "#888"),
                tag=TAGS.get(lab, lab[:2].upper()),
            )
        prev = hi if hi is not None else prev

    top_lab = LABELS[-1]
    return ConditionResult(
        label=top_lab,
        score=score,
        band_low=prev,
        band_high=None,
        description=DESCRIPTIONS[top_lab],
        color=COLORS[top_lab],
        tag=TAGS[top_lab],
    )

def condition_figure(score: float, result: ConditionResult, bands: Optional[dict] = None):
    """Generates a horizontal bar chart displaying score placement relative to grading bands.

    Args:
        score: Probe anomaly score to position on the ladder.
        result: ConditionResult for the probe score.
        bands: Optional band thresholds. When omitted, default calibration
            bands are used.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 2.4))

    if bands:
        normalized = {}
        for k, v in bands.items():
            canonical = ALIAS_TO_FULL.get(k.strip().upper(), k.strip())
            normalized[canonical] = v
        edges = [0.0] + [normalized.get(lab) for lab in LABELS[:-1]]
        edges = [e for e in edges if e is not None]
        bands_list = []
        prev = 0.0
        for lab in LABELS:
            hi = normalized.get(lab)
            hi = hi if hi is not None else (max(1.5, edges[-1] * 1.2) if edges else 1.5)
            bands_list.append((lab, prev, hi, COLORS[lab]))
            prev = hi
    else:
        bands_list = [
            ("Near Mint/Mint", 0.0, 0.410, COLORS["Near Mint/Mint"]),
            ("Good/Excellent", 0.410, 0.710, COLORS["Good/Excellent"]),
            ("Poor/Played", 0.710, 1.5, COLORS["Poor/Played"]),
        ]
    for lab, lo, hi, col in bands_list:
        ax.barh(0, hi - lo, left=lo, color=col, alpha=0.45, edgecolor="white", height=0.5)
        tag = TAGS.get(lab, lab)
        ax.text((lo + hi) / 2, 0, f"{tag}\n{lab}", ha="center", va="center", fontsize=8, fontweight="bold", color="#111827")

    ax.axvline(score, color="#111827", linewidth=2.5, linestyle="--", zorder=4)
    ax.text(score, 0.38, f"Score: {score:.3f}\n({result.label})", ha="center", va="bottom",
            fontweight="bold", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlim(0, bands_list[-1][2])
    ax.set_ylim(-0.5, 0.85)
    ax.set_yticks([])
    ax.set_xlabel("Anomaly Score")
    ax.set_title("Card Condition Classification", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig

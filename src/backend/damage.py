"""Module for procedural generation of synthetic defects on collectible cards.

Enables controlled synthesis of realistic physical wear patterns (edge whitening,
scratches, creases, dents, stains, holes, liquid damage) with severity
parameterized in [0.0, 1.0], mapped to standard condition levels (LP, MP, HP, DMG)
for anomaly detection pipeline validation.
"""

import random
from typing import Optional

import cv2
import numpy as np

LEVELS = {"LP": 0.25, "MP": 0.50, "HP": 0.75, "DMG": 1.00}

DEFECT_TYPES = ["edge_wear", "scratches", "crease", "dent", "stain", "hole", "water"]

_WHITE = (238, 236, 230)
_LIGHT_GRAY = (215, 222, 232)
_DARK = (28, 28, 34)
_BROWN = (22, 160, 205)
_WATER = (35, 155, 195)

def _rand_state(rng: random.Random) -> np.random.RandomState:
    """Deterministic numpy RNG derived from the python RNG."""
    return np.random.RandomState(rng.randint(0, 2 ** 31 - 1))

def _blend(card_bgra: np.ndarray, color_bgr: tuple, alpha_map: np.ndarray) -> np.ndarray:
    """Blend a solid color onto the card's BGR channels with a per-pixel alpha map."""
    out = card_bgra.copy()
    a = np.clip(alpha_map, 0.0, 1.0)[:, :, None]
    base = out[:, :, :3].astype(np.float32)
    overlay = np.zeros_like(base)
    overlay[:] = np.array(color_bgr, np.float32)
    out[:, :, :3] = (overlay * a + base * (1.0 - a)).astype(np.uint8)
    return out

def _edge_wear(card: np.ndarray, s: float, rs: np.random.RandomState) -> np.ndarray:
    """Whitening/scuffing along card edges (sleeve wear)."""
    h, w = card.shape[:2]
    alpha = (card[:, :, 3] > 0).astype(np.uint8)
    dist = cv2.distanceTransform(alpha, cv2.DIST_L2, 5)
    depth = max(4, int((0.05 + 0.13 * s) * min(h, w)))
    band = (dist < depth) & (dist > 0)

    noise = rs.rand(h, w)
    speckle = (noise > (1.0 - 0.20 - 0.60 * s)) & band
    speckle = speckle.astype(np.uint8)
    k = max(1, int(0.012 * min(h, w))) * 2 + 1
    speckle = cv2.dilate(speckle, np.ones((k, k), np.uint8))
    a_map = speckle.astype(np.float32) * (0.55 + 0.45 * s)

    edge_band = ((dist < depth * 0.6) & (dist > 0)).astype(np.float32)
    a_map = np.maximum(a_map, edge_band * (0.35 + 0.55 * s))

    a_map = cv2.GaussianBlur(a_map, (3, 3), 0)
    return _blend(card, _WHITE, a_map)

def _scratches(card: np.ndarray, s: float, rs: np.random.RandomState) -> np.ndarray:
    """Thin light lines on the card surface."""
    h, w = card.shape[:2]
    a_map = np.zeros((h, w), np.float32)
    n = int(3 + 30 * s)
    diag = (h ** 2 + w ** 2) ** 0.5
    for _ in range(n):
        x1, y1 = rs.randint(0, w), rs.randint(0, h)
        ang = rs.uniform(0, np.pi)
        length = rs.uniform(0.10, 0.40) * diag * (0.5 + s)
        x2 = int(x1 + length * np.cos(ang))
        y2 = int(y1 + length * np.sin(ang))
        thick = 1 if s < 0.5 else rs.choice([1, 2, 2])
        cv2.line(a_map, (x1, y1), (x2, y2), rs.uniform(0.45, 0.95) * s + 0.10, thick)
    a_map = cv2.GaussianBlur(a_map, (3, 3), 0)
    return _blend(card, _LIGHT_GRAY, a_map)

def _crease(card: np.ndarray, s: float, rs: np.random.RandomState) -> np.ndarray:
    """Fold/bend line across the card (dark line + highlight + soft shading)."""
    h, w = card.shape[:2]
    diag = int((h ** 2 + w ** 2) ** 0.5)
    cx, cy = rs.randint(0, w), rs.randint(0, h)
    ang = rs.uniform(0, np.pi)
    dx, dy = np.cos(ang), np.sin(ang)
    p1 = (int(cx - dx * diag), int(cy - dy * diag))
    p2 = (int(cx + dx * diag), int(cy + dy * diag))

    shade = np.zeros((h, w), np.float32)
    cv2.line(shade, p1, p2, 1.0, max(4, int(0.14 * min(h, w))))
    shade = cv2.GaussianBlur(shade, (0, 0), 0.04 * min(h, w))
    card = _blend(card, _DARK, shade * 0.16 * (0.4 + 0.6 * s))

    line = np.zeros((h, w), np.float32)
    cv2.line(line, p1, p2, 1.0, 1 if s < 0.6 else 2)
    card = _blend(card, _DARK, line * (0.55 + 0.45 * s))

    nx, ny = -dy, dx
    hi = np.zeros((h, w), np.float32)
    off = 2 if s < 0.6 else 3
    cv2.line(hi, (int(p1[0] + nx * off), int(p1[1] + ny * off)),
             (int(p2[0] + nx * off), int(p2[1] + ny * off)), 1.0, 1)
    card = _blend(card, _WHITE, hi * 0.60 * (0.4 + 0.6 * s))
    return card

def _dent(card: np.ndarray, s: float, rs: np.random.RandomState) -> np.ndarray:
    """Localized indentation: radial darkening + highlight rim."""
    h, w = card.shape[:2]
    cx = rs.uniform(0.2, 0.8) * w
    cy = rs.uniform(0.2, 0.8) * h
    r = (0.10 + 0.16 * s) * min(h, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1.0)

    dark = np.clip(1.0 - d, 0, 1) ** 1.0 * (0.65 + 0.35 * s)
    card = _blend(card, _DARK, dark)

    rim = np.exp(-((d - 1.1) ** 2) / 0.10) * 0.55 * (0.3 + 0.7 * s)
    card = _blend(card, _WHITE, rim)
    return card

def _stain(card: np.ndarray, s: float, rs: np.random.RandomState) -> np.ndarray:
    """Yellow-brown blotches (dirt / aging)."""
    h, w = card.shape[:2]
    a_map = np.zeros((h, w), np.float32)
    cx = rs.uniform(0.15, 0.85) * w
    cy = rs.uniform(0.15, 0.85) * h
    spread = (0.10 + 0.18 * s) * min(h, w)
    n_blobs = int(5 + 14 * s)
    for _ in range(n_blobs):
        bx = int(cx + rs.uniform(-1, 1) * spread)
        by = int(cy + rs.uniform(-1, 1) * spread)
        ax_ = int(rs.uniform(0.3, 1.0) * spread)
        ay_ = int(rs.uniform(0.3, 1.0) * spread)
        angle = rs.uniform(0, 180)
        cv2.ellipse(a_map, (bx, by), (max(2, ax_), max(2, ay_)), angle,
                    0, 360, 1.0, -1)
    k = max(3, int(0.05 * min(h, w))) | 1
    a_map = cv2.GaussianBlur(a_map, (k, k), 0)
    a_map = np.clip(a_map, 0, 1) * (0.20 + 0.55 * s)
    return _blend(card, _BROWN, a_map)

def _hole(card: np.ndarray, s: float, rs: np.random.RandomState) -> np.ndarray:
    """Punched/torn missing material: alpha → 0 so the background shows through."""
    h, w = card.shape[:2]
    cx = int(rs.uniform(0.25, 0.75) * w)
    cy = int(rs.uniform(0.25, 0.75) * h)
    r = (0.04 + 0.12 * s) * min(h, w)
    n_ver = rs.randint(8, 15)
    pts = []
    for i in range(n_ver):
        ang = 2 * np.pi * i / n_ver + rs.uniform(-0.15, 0.15)
        rr = r * rs.uniform(0.65, 1.35)
        pts.append([int(cx + rr * np.cos(ang)), int(cy + rr * np.sin(ang))])
    pts = np.array(pts, np.int32)

    mask = np.zeros((h, w), np.float32)
    cv2.fillPoly(mask, [pts], 1.0)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    out = card.copy()
    out[:, :, 3] = (out[:, :, 3].astype(np.float32) * (1.0 - np.clip(mask, 0, 1))).astype(np.uint8)

    rim = np.clip(cv2.GaussianBlur(mask, (5, 5), 0) - mask, 0, 1)
    return _blend(out, _DARK, rim * 0.6)

def _water(card: np.ndarray, s: float, rs: np.random.RandomState) -> np.ndarray:
    """Wavy water-damage discoloration entering from one edge."""
    h, w = card.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = yy / h if rs.rand() < 0.5 else 1.0 - yy / h
    u = (xx / w if rs.rand() < 0.5 else 1.0 - xx / w)
    phase = rs.uniform(0, 2 * np.pi)
    amp = 0.06 + 0.10 * s
    freq = rs.uniform(1.5, 3.5)
    extent = 0.30 + 0.50 * s
    wave = amp * np.sin(freq * 2 * np.pi * u + phase)
    mask = np.clip((t - (1.0 - extent - wave)) / max(extent, 1e-6), 0, 1)
    k = max(3, int(0.04 * min(h, w))) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return _blend(card, _WATER, mask * (0.40 + 0.50 * s))

_DISPATCH = {
    "edge_wear": _edge_wear,
    "scratches": _scratches,
    "crease": _crease,
    "dent": _dent,
    "stain": _stain,
    "hole": _hole,
    "water": _water,
}

def apply_damage(card_bgra: np.ndarray, defect: str, severity: float,
                 rng: random.Random) -> np.ndarray:
    """
    Apply one defect type at the given severity to a BGRA card scan.

    Args:
        card_bgra: pristine card scan (BGRA, any resolution)
        defect: one of DEFECT_TYPES
        severity: continuous damage severity in [0, 1]
        rng: python random.Random for determinism

    Returns:
        Damaged card as BGRA uint8 (input not modified).
    """
    if defect not in _DISPATCH:
        raise ValueError(f"Unknown defect '{defect}'. Choose from {DEFECT_TYPES}")
    rs = _rand_state(rng)
    return _DISPATCH[defect](card_bgra, float(np.clip(severity, 0.0, 1.0)), rs)

def synthesize_damaged(card_bgra: np.ndarray, level: str, rng: random.Random,
                       defect: Optional[str] = None):
    """
    Synthesize a damaged card at a named condition level.

    Args:
        card_bgra: pristine card scan (BGRA)
        level: one of "LP" | "MP" | "HP" | "DMG"
        rng: python random.Random
        defect: optional fixed defect type; random if None

    Returns:
        (damaged_bgra, defect_type)
    """
    if level not in LEVELS:
        raise ValueError(f"Unknown level '{level}'. Choose from {list(LEVELS)}")
    d = defect if defect is not None else rng.choice(DEFECT_TYPES)
    return apply_damage(card_bgra, d, LEVELS[level], rng), d

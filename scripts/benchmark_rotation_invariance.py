"""Rotation stress test for the production detector, PBL, warp, and DINO scorer."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.card_detector import CardDetector  # noqa: E402
from backend.detector import PokemonAnomalyDetector  # noqa: E402

EXPERIMENTS = PROJECT_ROOT / "experiments"
OUTPUT = PROJECT_ROOT / "report/figures/03_comparative_benchmark/rotation_invariance.json"
CASES = [
    ("mew_eng", EXPERIMENTS / "different-languages/mew-eng.jpg",
     EXPERIMENTS / "different-languages/mew-reference.jpg"),
    ("dragonair_ita", EXPERIMENTS / "different-languages/dragonair-ita.jpg",
     EXPERIMENTS / "different-languages/dragonair-reference.webp"),
    ("gengar_prime", EXPERIMENTS / "synthetic-to-real/realgengar-1.jpg",
     EXPERIMENTS / "synthetic-to-real/reference-gengarprime.webp"),
]

def polygon_distance(first: np.ndarray, second: np.ndarray) -> float:
    best = float("inf")
    for order in (first, first[::-1]):
        for shift in range(4):
            aligned = np.roll(order, shift, axis=0)
            best = min(best, float(np.linalg.norm(aligned - second, axis=1).mean()))
    return best

def restore(points: np.ndarray, rotation: int, width: int, height: int) -> np.ndarray:
    if rotation == 0:
        return points.copy()
    if rotation == 1:
        return np.stack([width - 1 - points[:, 1], points[:, 0]], axis=1)
    if rotation == 2:
        return np.stack([width - 1 - points[:, 0], height - 1 - points[:, 1]], axis=1)
    return np.stack([points[:, 1], height - 1 - points[:, 0]], axis=1)

def main():
    card_detector = CardDetector(refine=False)
    anomaly_detector = PokemonAnomalyDetector()
    report = {"rotations_deg": [0, 90, 180, 270], "cases": {}}

    for name, image_path, reference_path in CASES:
        image = cv2.imread(str(image_path))
        if image is None or not reference_path.exists():
            raise FileNotFoundError(f"Missing rotation-test input or reference for {name}")
        anomaly_detector.load_reference_images([str(reference_path)], card_id=None)
        height, width = image.shape[:2]
        scores, corners_back, winning_rotation = [], [], []

        for k in range(4):
            rotated = np.ascontiguousarray(np.rot90(image, k)) if k else image
            detections = card_detector.detect(rotated)
            if not detections:
                raise RuntimeError(f"Detector missed {name} at input rotation {90*k} degrees")
            d = detections[0]
            result = anomaly_detector.analyze_patch(d.patch, use_few_shot=False)
            scores.append(float(result["score"]))
            winning_rotation.append(int(result.get("rotation_index", 0)) * 90)
            corners_back.append(restore(d.corners, k, width, height))

        corner_deltas = [polygon_distance(q, corners_back[0]) for q in corners_back]
        case = {
            "reference": str(reference_path.relative_to(PROJECT_ROOT)) if reference_path.is_relative_to(PROJECT_ROOT) else reference_path.name,
            "scores_by_input_rotation": scores,
            "winner_rotation_deg": winning_rotation,
            "score_range": float(max(scores) - min(scores)),
            "corner_delta_from_0deg_px": corner_deltas,
            "max_corner_delta_px": float(max(corner_deltas)),
        }
        report["cases"][name] = case
        print(name, case)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2))
    print(f"Saved {OUTPUT}")

if __name__ == "__main__":
    main()

"""Integration test suite for FastAPI multi-backend detection and grading endpoints."""

import json
import logging
import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.main import app  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    client = TestClient(app)

    # 1. Status endpoint
    r_status = client.get("/status")
    assert r_status.status_code == 200, f"/status failed: {r_status.text}"
    st = r_status.json()
    assert "detector_backend" in st
    assert "available_backends" in st
    print("✓ GET /status:", {"backend": st["detector_backend"], "available": st["available_backends"]})

    # 2. Load cached Charizard reference (base1-4)
    r_ref = client.post("/load-reference", json={"card_id": "base1-4"})
    assert r_ref.status_code == 200, f"/load-reference failed: {r_ref.text}"
    print("✓ POST /load-reference (base1-4):", r_ref.json()["reference"]["name"])

    # 3. Choose real test photo (Charizard damaged smartphone capture or synthetic test card)
    probe = PROJECT_ROOT / "experiments" / "synthetic-to-real" / "charizard-damaged.jpeg"
    if not probe.exists():
        probe = sorted((PROJECT_ROOT / "data" / "datasets" / "pokemon_seg" / "images" / "test").glob("*.jpg"))[0]

    raw_bytes = probe.read_bytes()
    backends_to_test = ["obb_pbl", "seg_zeroshot"]
    if (PROJECT_ROOT / "data" / "weights" / "yolo11s_seg_best.pt").exists():
        backends_to_test.append("seg_yolo")
    if (PROJECT_ROOT / "data" / "weights" / "seg_semantic_best.pt").exists():
        backends_to_test.append("seg_semantic")
    if (PROJECT_ROOT / "data" / "weights" / "seg_canonical_unet_best.pt").exists():
        backends_to_test.append("obb_unet")
    if (PROJECT_ROOT / "data" / "weights" / "corner_heatmap_best.pt").exists():
        backends_to_test.append("obb_corner")

    summary = []
    for bk in backends_to_test:
        r_det = client.post(
            f"/detect?backend={bk}",
            files={"file": (probe.name, raw_bytes, "image/jpeg")},
        )
        assert r_det.status_code == 200, f"/detect {bk} failed: {r_det.text}"
        det_json = r_det.json()
        assert det_json["detector_backend"] == bk
        assert det_json["n_cards"] >= 1, f"{bk} found 0 cards on {probe.name}"

        r_ana = client.post(
            f"/analyze?backend={bk}",
            files={"file": (probe.name, raw_bytes, "image/jpeg")},
        )
        assert r_ana.status_code == 200, f"/analyze {bk} failed ({r_ana.status_code}): {r_ana.text}"
        ana = r_ana.json()
        assert "score" in ana and "condition" in ana and "timing_ms" in ana
        row = {
            "backend": bk,
            "n_cards": det_json["n_cards"],
            "has_dynamic_mask": det_json["detections"][0]["has_dynamic_mask"],
            "score": round(float(ana["score"]), 4),
            "condition": ana["condition"]["label"],
            "quality_score": round(float(ana["quality"]["score"]), 4),
            "timing_ms": ana["timing_ms"],
        }
        summary.append(row)
        print(f"✓ /detect & /analyze ({bk}):", json.dumps(row))

    print("\nAll FastAPI multi-backend integration tests passed.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())

"""FastAPI Server for the PokéDINO Baseline Pipeline.

The operational pipeline includes:
1. Catalog search for official card reference scans and memory bank configuration.
2. Stage 1 quality gate (quick reject on raw image before YOLO).
3. Oriented card detection via YOLO11s-OBB and rigid Euclidean perspective rectification (630x448).
4. Stage 2 quality gate (detailed assessment on YOLO crop).
5. Preprocessing with CLAHE equalization and DINOv2 visual anomaly estimation.
6. High-resolution defect heat map generation and Cardmarket qualitative grading.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.pokemon_api import search_cards, get_reference_images_for_card, format_card_display
from backend.detector import PokemonAnomalyDetector
from backend.card_detector import CardDetector
from backend.quality_gate import QuickRejectRaw, ImageQualityGate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PokéDINO API - Baseline",
    description="Pokémon Card Condition Inspection: YOLO OBB + CLAHE Preprocessing + AnomalyDINO (DINOv2)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

detector = PokemonAnomalyDetector()
_card_detector: Optional[CardDetector] = None
_manual_reference_info: Optional[dict] = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def get_card_detector() -> CardDetector:
    """Initializes and returns the singleton CardDetector instance."""
    global _card_detector
    if _card_detector is None:
        cfg = _load_config()
        obb_inf = cfg.get("obb", {}).get("inference", {})
        conf = float(obb_inf.get("conf", 0.18))
        iou = float(obb_inf.get("iou", 0.45))
        max_det = int(obb_inf.get("max_det", 10))
        _card_detector = CardDetector(conf=conf, iou=iou, max_det=max_det)
    return _card_detector

from backend.experiment import load_config as _load_config

_quick_reject = QuickRejectRaw()
_quality_gate = ImageQualityGate()

class ThresholdRequest(BaseModel):
    """Request schema for updating the anomaly decision threshold."""
    threshold: float

class HeatmapParamsRequest(BaseModel):
    """Request schema for configuring heatmap visualization parameters."""
    dead_zone: float = 0.30
    gamma: float = 2.0
    few_shot: Optional[bool] = None

class LoadReferenceRequest(BaseModel):
    """Request schema for loading a reference card into the memory bank."""
    card_id: str
    api_key: Optional[str] = None
    card_data: Optional[dict] = None

from fastapi.responses import JSONResponse, FileResponse, RedirectResponse

@app.get("/", tags=["Info"])
async def root():
    """Redirects root path requests to the frontend web application."""
    return RedirectResponse(url="/app/")

@app.get("/status", tags=["Info"])
async def get_status():
    """Returns current operational status of the inspection pipeline and models."""
    status = dict(detector.status)
    status["obb_weights"] = _card_detector.weights_path if _card_detector else None
    status["active_reference"] = _manual_reference_info
    status["quality_gate"] = {
        "quick_reject": {
            "min_pixel_std": _quick_reject.min_pixel_std,
            "min_laplacian_var": _quick_reject.min_laplacian_var,
        },
        "image_quality": {
            "weights": _quality_gate.weights,
            "hard_reject_thresholds": _quality_gate.hard_reject_thresholds,
            "min_score": 0.35,
        },
    }
    return status

@app.get("/search-card", tags=["Pokémon TCG"])
async def search_card(
    q: str = Query(..., description="Card name to search (e.g. 'Dark Charizard', 'Pikachu')"),
    api_key: Optional[str] = Query(None, description="pokemontcg.io API key (optional)"),
    limit: int = Query(20, ge=1, le=100),
):
    """Searches the card catalog using tokenized substring matching."""
    cards = search_cards(q, api_key=api_key, page_size=limit)
    if not cards:
        return {"results": [], "count": 0, "message": f"No cards found for '{q}'"}
    formatted = [format_card_display(c) for c in cards]
    return {"results": formatted, "count": len(formatted)}

@app.post("/load-reference", tags=["Detector"])
async def load_reference(body: LoadReferenceRequest):
    """Loads official reference card scans and initializes the memory bank."""
    global _manual_reference_info
    logger.info(f"Loading manual reference for: {body.card_id}")
    image_paths = get_reference_images_for_card(
        body.card_id,
        api_key=body.api_key,
        card_data=body.card_data,
    )
    if not image_paths:
        raise HTTPException(
            status_code=404,
            detail=f"Unable to retrieve reference image for '{body.card_id}'. Please verify the card ID."
        )

    try:
        detector.load_reference_images(image_paths, card_id=body.card_id)
        card_name = body.card_id
        set_name = ""
        image_url = None
        if body.card_data:
            card_name = body.card_data.get("name", body.card_id)
            set_obj = body.card_data.get("set", {})
            set_name = body.card_data.get("set_name") or (set_obj.get("name", "") if isinstance(set_obj, dict) else str(set_obj or ""))
            images = body.card_data.get("images", {})
            if isinstance(images, dict):
                image_url = images.get("small") or images.get("large")
            if not image_url:
                image_url = body.card_data.get("image_small") or body.card_data.get("image_large")

        if not image_url or not set_name:
            from backend.pokemon_api import get_card_by_id
            c = get_card_by_id(body.card_id)
            if c:
                if not card_name or card_name == body.card_id:
                    card_name = c.get("name", card_name)
                if not set_name:
                    set_name = c.get("set_name") or c.get("set", {}).get("name", "")
                if not image_url:
                    imgs = c.get("images") or {}
                    image_url = imgs.get("small") or imgs.get("large") or c.get("image_small") or c.get("image_large")

        if not image_url:
            image_url = f"/reference-image/{body.card_id}"

        _manual_reference_info = {
            "card_id": body.card_id,
            "name": card_name,
            "set_name": set_name,
            "image_url": image_url,
            "local_path": image_paths[0],
            "manual": True,
            "few_shot": detector.use_few_shot,
            "reference_count": detector.reference_count,
            "variants": [Path(p).name for p in detector.few_shot_variants],
        }
    except Exception as e:
        logger.error(f"Error loading reference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "success": True,
        "message": f"Memory bank configured with {len(image_paths)} reference image(s).",
        "reference": _manual_reference_info,
    }

@app.post("/clear-reference", tags=["Detector"])
async def clear_reference():
    """Removes active manual reference and resets memory bank."""
    global _manual_reference_info
    _manual_reference_info = None
    detector.current_card_id = None
    detector.memory_bank_ready = False
    return {"success": True, "message": "Reference successfully removed."}

@app.post("/set-heatmap-params", tags=["Detector"])
async def set_heatmap_params(body: HeatmapParamsRequest):
    """Updates visualization hyper-parameters (dead zone, gamma, few-shot mode)."""
    detector.set_heatmap_params(
        dead_zone=body.dead_zone,
        gamma=body.gamma,
        use_few_shot=body.few_shot,
    )
    return {
        "success": True,
        "dead_zone": body.dead_zone,
        "gamma": body.gamma,
        "use_few_shot": detector.use_few_shot,
    }

@app.post("/re-render-heatmap", tags=["Detector"])
async def re_render_heatmap(body: HeatmapParamsRequest):
    """Instantly re-renders the heatmap of the last analyzed patch using new dead_zone, gamma, and few_shot values."""
    res = detector.re_render_heatmap(dead_zone=body.dead_zone, gamma=body.gamma, few_shot=body.few_shot)
    if not res:
        raise HTTPException(status_code=400, detail="No recently analyzed patch to re-render.")
    return res

@app.get("/reference-image/{card_id}", tags=["Pokémon TCG"])
async def get_reference_image(card_id: str):
    """Serve card image from local cache or redirect to CDN."""
    cache_dir = PROJECT_ROOT / "data" / "reference_cache"
    for ext in (".png", ".jpg", ".webp"):
        p = cache_dir / f"{card_id}{ext}"
        if p.exists():
            return FileResponse(str(p))
    from backend.pokemon_api import get_card_by_id
    from fastapi.responses import RedirectResponse
    c = get_card_by_id(card_id)
    if c and c.get("images"):
        url = c["images"].get("small") or c["images"].get("large")
        if url:
            return RedirectResponse(url)
    raise HTTPException(status_code=404, detail="Reference image not found")

@app.post("/detect", tags=["Pipeline"])
async def detect_cards(file: UploadFile = File(..., description="Scene photograph containing cards")):
    """Detects oriented cards in a raw scene photograph using YOLO11s-OBB."""
    tmp_path = UPLOAD_DIR / f"scene_{file.filename or 'scene.jpg'}"
    try:
        with open(tmp_path, "wb") as f:
            f.write(await file.read())
        img = cv2.imread(str(tmp_path), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Unreadable image file.")

        cd = get_card_detector()
        dets = cd.detect(img)
        from backend.detector import ndarray_to_base64

        warning_msg = None
        if len(dets) > 1:
            warning_msg = (
                f"Multiple cards detected in the image ({len(dets)} cards found). "
                "Make sure only one card is present for anomaly detection!"
            )
            logger.warning(warning_msg)

        out = []
        for d in dets:
            out.append({
                "confidence": d.confidence,
                "corners": d.corners.tolist(),
                "sharpness": d.patch_gray_score,
                "patch_b64": ndarray_to_base64(d.patch),
            })
        scene_b64 = ndarray_to_base64(cd.draw_detections(img, dets)) if dets else None
        return {"n_cards": len(out), "detections": out, "scene_b64": scene_b64, "warning": warning_msg}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

@app.post("/analyze", tags=["Pipeline"])
async def analyze_card(
    file: UploadFile = File(..., description="Card photograph to inspect"),
    dead_zone: Optional[float] = Query(None, ge=0.0, le=1.0, description="Dead zone noise threshold (0.0 - 1.0)"),
    gamma: Optional[float] = Query(None, ge=0.1, le=5.0, description="Gamma correction exponent (e.g. 2.0)"),
    few_shot: Optional[bool] = Query(True, description="Use Few-Shot (photometric variants) or One-Shot (single scan)"),
):
    """Executes the full card condition grading pipeline."""
    if not detector.memory_bank_ready or not _manual_reference_info:
        raise HTTPException(
            status_code=400,
            detail="No reference card set. Search and select an official card from the catalog first."
        )

    tmp_path = UPLOAD_DIR / f"test_{file.filename or 'card.jpg'}"
    try:
        with open(tmp_path, "wb") as f:
            f.write(await file.read())
        img = cv2.imread(str(tmp_path), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Unreadable image file.")

        passed, reason = _quick_reject.evaluate(img)
        if not passed:
            raise HTTPException(
                status_code=422,
                detail=reason,
            )

        cd = get_card_detector()
        dets = cd.detect(img)
        if not dets:
            raise HTTPException(
                status_code=422,
                detail="No card detected in photograph. Ensure the entire card is visible and retry.",
            )

        warning_msg = None
        if len(dets) > 1:
            warning_msg = (
                f"Multiple cards detected in the image ({len(dets)} cards found). "
                "Make sure only one card is present for anomaly detection!"
            )
            logger.warning(warning_msg)

        det = dets[0]

        h_frame, w_frame = img.shape[:2]
        box_area = cv2.contourArea(det.corners.astype(np.int32))
        box_area_ratio = box_area / (h_frame * w_frame)
        
        sides = []
        for i in range(4):
            p1 = det.corners[i]
            p2 = det.corners[(i + 1) % 4]
            sides.append(np.linalg.norm(p2 - p1))
        sides.sort()
        box_aspect_ratio = sides[0] / sides[3] if sides[3] > 0 else 1.0
        
        quality_result = _quality_gate.evaluate(
            image=det.patch,
            detection_confidence=det.confidence,
            box_aspect_ratio=box_aspect_ratio,
            box_area_ratio=box_area_ratio,
        )

        quality_summary = {
            "score": round(quality_result.score, 3),
            "percentage": round(quality_result.score * 100, 1),
            "passed": quality_result.passed,
            "threshold": _quality_gate.weights,
            "metrics": {
                "sharpness": round(quality_result.metrics.sharpness, 3),
                "exposure": round(quality_result.metrics.exposure, 3),
                "glare": round(quality_result.metrics.glare, 3),
                "noise": round(quality_result.metrics.noise, 3),
                "resolution": round(quality_result.metrics.resolution, 3),
                "detection_confidence": round(quality_result.metrics.detection_confidence, 3),
                "aspect_ratio": round(quality_result.metrics.aspect_ratio_score, 3),
                "relative_area": round(quality_result.metrics.relative_area_score, 3),
            },
            "hard_rejects": quality_result.metrics.hard_rejects,
            "status_label": "Excellent" if quality_result.score >= 0.75 else "Good" if quality_result.score >= 0.50 else "Acceptable" if quality_result.passed else "Rejected",
        }

        if not quality_result.passed:
            reject_details = "; ".join(quality_result.reject_reasons) if quality_result.reject_reasons else "Unknown"
            raise HTTPException(
                status_code=422,
                detail=f"Image quality insufficient: {reject_details}",
            )

        result = detector.analyze_patch(det.patch, dead_zone=dead_zone, gamma=gamma, use_few_shot=few_shot)
        result["detection"] = {
            "confidence": det.confidence,
            "corners": det.corners.tolist(),
        }
        result["quality"] = quality_summary
        result["reference"] = _manual_reference_info
        result["patch_b64"] = result.get("original_b64")
        result["warning"] = warning_msg
        result["n_cards_detected"] = len(dets)

        logger.info(
            f"Baseline inspection complete: card={_manual_reference_info['name']}, "
            f"score={result['score']:.4f}, condition={result['condition']['label']}, "
            f"quality={quality_result.score:.2f}"
        )
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

@app.post("/set-threshold", tags=["Detector"])
async def set_threshold(body: ThresholdRequest):
    """Updates decision threshold for anomaly classification."""
    if not 0.0 < body.threshold < 1.0:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1 (exclusive).")
    detector.set_threshold(body.threshold)
    return {"success": True, "threshold": body.threshold}

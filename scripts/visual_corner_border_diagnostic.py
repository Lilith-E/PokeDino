"""High-resolution visual diagnostic focused on borders and corners on real card captures.

Compares obb_pbl, seg_yolo_640, seg_yolo_960, and seg_semantic on all 7 real
smartphone photographs in experiments/synthetic-to-real/, visualizing:
1. Raw predicted mask overlay + extracted [TL, TR, BR, BL] corners in scene space.
2. 4x corner zoom crops (TL, TR, BR, BL) to isolate mask error vs quad-fit error.
3. Rectified (630x448) card crop with zoomed border strips to evaluate alignment precision.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.card_detector import CardDetector, build_card_detector  # noqa: E402
from backend.experiment import load_config  # noqa: E402
from backend.seg_detector import SegmentationCardDetector  # noqa: E402
from scripts.benchmark_segmentation_vs_obb import project_patch_mask_to_scene  # noqa: E402

logger = logging.getLogger(__name__)

REAL_IMAGES = [
    "charizard-damaged.jpeg",
    "giratina-kanda-damaged.jpg",
    "realgengar-1.jpg",
    "realgengar-2.jpg",
    "realgengar-3.jpg",
    "realgengar-4.jpg",
    "realgengar-5.jpg",
]

CORNER_COLORS_BGR = [
    (0, 255, 0),     # TL: green
    (255, 180, 0),   # TR: cyan-blue
    (0, 0, 255),     # BR: red
    (255, 0, 255),   # BL: magenta
]
CORNER_NAMES = ["TL", "TR", "BR", "BL"]


def _draw_quad_and_mask(
    img_bgr: np.ndarray,
    corners: np.ndarray | None,
    mask: np.ndarray | None,
) -> np.ndarray:
    vis = img_bgr.copy()
    if mask is not None:
        m = mask > 0.5
        overlay = vis.copy()
        overlay[m] = (0, 220, 120)
        vis = cv2.addWeighted(overlay, 0.28, vis, 0.72, 0)
        # Draw raw mask contour in yellow
        cnts, _ = cv2.findContours(m.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 230, 255), max(2, vis.shape[1] // 400))

    if corners is not None and len(corners) == 4:
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        thick = max(3, vis.shape[1] // 320)
        cv2.polylines(vis, [pts], True, (255, 80, 20), thick, cv2.LINE_AA)
        r = max(8, vis.shape[1] // 140)
        for i, (cx, cy) in enumerate(corners):
            c = CORNER_COLORS_BGR[i]
            cv2.circle(vis, (int(round(cx)), int(round(cy))), r, c, -1, cv2.LINE_AA)
            cv2.putText(
                vis,
                CORNER_NAMES[i],
                (int(round(cx)) + r + 4, int(round(cy)) + r),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.7, vis.shape[1] / 1400.0),
                c,
                max(2, thick // 2),
                cv2.LINE_AA,
            )
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def _corner_zoom_mosaic(
    img_bgr: np.ndarray,
    corners: np.ndarray | None,
    mask: np.ndarray | None,
    patch_half: int = 110,
) -> np.ndarray:
    """Create a 2x2 mosaic of zoomed crops around TL, TR, BR, BL corners."""
    vis_rgb = _draw_quad_and_mask(img_bgr, corners, mask)
    h, w = vis_rgb.shape[:2]
    tiles = []
    if corners is None or len(corners) != 4:
        blank = np.full((220, 220, 3), 235, dtype=np.uint8)
        cv2.putText(blank, "NO DET", (45, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 30, 30), 2)
        return np.vstack([np.hstack([blank, blank]), np.hstack([blank, blank])])

    half = max(60, int(min(h, w) * 0.085))
    for i, (cx, cy) in enumerate(corners):
        x0 = max(0, int(round(cx)) - half)
        y0 = max(0, int(round(cy)) - half)
        x1 = min(w, int(round(cx)) + half)
        y1 = min(h, int(round(cy)) + half)
        crop = vis_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            crop = np.full((180, 180, 3), 220, dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (180, 180), interpolation=cv2.INTER_LINEAR)
        cv2.putText(crop, CORNER_NAMES[i], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(crop, CORNER_NAMES[i], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 1)
        tiles.append(crop)

    return np.vstack([np.hstack([tiles[0], tiles[1]]), np.hstack([tiles[3], tiles[2]])])


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual corner/border diagnostic on real card photos.")
    parser.add_argument("--only-obb-variants", action="store_true", help="Compare only OBB+PBL vs OBB+Canon UNet.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config()
    real_dir = PROJECT_ROOT / "experiments" / "synthetic-to-real"
    out_dir = PROJECT_ROOT / "report" / "figures" / "05_segmentation_pipeline" / "corner_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    backends: list[tuple[str, object]] = [
        ("OBB + PBL", CardDetector(
            conf=float(cfg["obb"]["inference"]["conf"]),
            iou=float(cfg["obb"]["inference"]["iou"]),
            max_det=int(cfg["obb"]["inference"]["max_det"]),
        )),
    ]
    w_unet = PROJECT_ROOT / cfg.get("detector", {}).get("obb_unet", {}).get(
        "unet_weights", "data/weights/seg_canonical_unet_best.pt"
    )
    if w_unet.exists():
        backends.append(("OBB + Canon UNet", build_card_detector(cfg, backend="obb_unet")))

    if not args.only_obb_variants:
        w640 = PROJECT_ROOT / cfg["segmentation"]["yolo_seg"]["weights"]
        if w640.exists():
            backends.append(("YOLO11-Seg @ 640", SegmentationCardDetector(
                backend="seg_yolo", weights_path=str(w640), imgsz=640,
            )))
        w960 = PROJECT_ROOT / cfg["segmentation"]["yolo_seg"]["weights_960"]
        if w960.exists():
            backends.append(("YOLO11-Seg @ 960", SegmentationCardDetector(
                backend="seg_yolo", weights_path=str(w960), imgsz=960,
            )))
        wsem = PROJECT_ROOT / cfg["segmentation"]["semantic"]["weights"]
        if wsem.exists():
            backends.append(("MobileNetV3-UNet", SegmentationCardDetector(
                backend="seg_semantic", weights_path=str(wsem), imgsz=640,
            )))

    for fname in REAL_IMAGES:
        fpath = real_dir / fname
        if not fpath.exists():
            continue
        img_bgr = cv2.imread(str(fpath))
        if img_bgr is None:
            continue

        fig, axes = plt.subplots(3, len(backends), figsize=(4.8 * len(backends), 14.5), dpi=170)
        fig.patch.set_facecolor("white")

        for col, (bname, det_model) in enumerate(backends):
            dets = det_model.detect(img_bgr)
            top = dets[0] if dets else None
            corners = top.corners if top is not None else None
            raw_mask = getattr(top, "mask", None) if top is not None else None
            mask = (
                project_patch_mask_to_scene(corners, raw_mask, img_bgr.shape[:2])
                if (raw_mask is not None and corners is not None)
                else None
            )
            patch = top.patch if top is not None else None

            scene_vis = _draw_quad_and_mask(img_bgr, corners, mask)
            zoom_vis = _corner_zoom_mosaic(img_bgr, corners, mask)

            axes[0, col].imshow(scene_vis)
            status = f"conf={top.confidence:.2f}" if top is not None else "NO DETECTION"
            axes[0, col].set_title(f"{bname}\n{status}", fontsize=11, fontweight="bold")
            axes[0, col].axis("off")

            axes[1, col].imshow(zoom_vis)
            axes[1, col].set_title("Corner Zoom (TL / TR / BL / BR)\nYellow=Raw Mask, Blue=Fitted Quad", fontsize=9.5)
            axes[1, col].axis("off")

            if patch is not None:
                axes[2, col].imshow(patch)
                # Draw subtle 10px inner reference border to highlight edge alignment
                h_p, w_p = patch.shape[:2]
                axes[2, col].plot([0, w_p - 1, w_p - 1, 0, 0], [0, 0, h_p - 1, h_p - 1, 0], color="lime", lw=1.2, alpha=0.8)
                axes[2, col].set_title(f"Rectified Patch ({w_p}x{h_p})", fontsize=9.5)
            else:
                axes[2, col].text(0.5, 0.5, "NO DETECTION", ha="center", va="center", color="red", fontsize=12, fontweight="bold")
                axes[2, col].set_title("Rectified Patch", fontsize=9.5)
            axes[2, col].axis("off")

        fig.suptitle(f"Border & Corner Precision Diagnostic — {fname}", fontsize=14, fontweight="bold", y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.985])
        out_path = out_dir / f"diag_{fpath.stem}.png"
        fig.savefig(out_path)
        plt.close(fig)
        logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()

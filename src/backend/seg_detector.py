"""Object-segmentation card detectors and perspective rectification frontends.

Provides single-stage segmentation backends (`seg_yolo`, `seg_zeroshot`, and
`seg_semantic`) sharing the ``CardDetection`` output contract while emitting
warped dynamic validity masks aligned with the rectified 630x448 card patch.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.card_detector import CardDetection, CardDetector, TARGET_H, TARGET_W
from backend.experiment import get_device, load_config
from backend.rectifier import (
    CANON_H,
    CANON_W,
    rectify_from_canonical_prob,
    rectify_from_mask,
    warp_to_canonical_obb,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "configs").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def bgr_to_imagenet_tensor(
    img_bgr: np.ndarray,
    size_hw: tuple[int, int],
    device: str | torch.device,
) -> torch.Tensor:
    """Resize a BGR image to ``(H, W)`` and normalize with ImageNet channel statistics."""
    th, tw = size_hw
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (th, tw):
        rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0).to(device)


class _ConvBlock(nn.Module):
    """Double 3x3 convolution block with BatchNorm and SiLU activation."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MobileNetV3UNet(nn.Module):
    """Full-resolution semantic segmentation U-Net backed by MobileNetV3-Large."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        feats = mobilenet_v3_large(weights=weights).features
        self.enc1 = feats[:2]    # stride 2, 16 ch
        self.enc2 = feats[2:4]   # stride 4, 24 ch
        self.enc3 = feats[4:7]   # stride 8, 40 ch
        self.enc4 = feats[7:13]  # stride 16, 112 ch
        self.enc5 = feats[13:]   # stride 32, 960 ch

        self.up4 = _ConvBlock(960 + 112, 256)
        self.up3 = _ConvBlock(256 + 40, 128)
        self.up2 = _ConvBlock(128 + 24, 64)
        self.up1 = _ConvBlock(64 + 16, 32)
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_in, w_in = x.shape[-2:]
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        s5 = self.enc5(s4)

        d4 = self.up4(torch.cat([F.interpolate(s5, size=s4.shape[-2:], mode="bilinear", align_corners=False), s4], dim=1))
        d3 = self.up3(torch.cat([F.interpolate(d4, size=s3.shape[-2:], mode="bilinear", align_corners=False), s3], dim=1))
        d2 = self.up2(torch.cat([F.interpolate(d3, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2], dim=1))
        d1 = self.up1(torch.cat([F.interpolate(d2, size=s1.shape[-2:], mode="bilinear", align_corners=False), s1], dim=1))
        full = F.interpolate(d1, size=(h_in, w_in), mode="bilinear", align_corners=False)
        return self.head(full)


class SegmentationCardDetector:
    """Single-model object segmentation card detector with mask-driven rectification."""

    def __init__(
        self,
        backend: str = "seg_yolo",
        weights_path: Optional[str] = None,
        imgsz: Optional[int] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: int = 10,
        target_size: tuple[int, int] = (TARGET_H, TARGET_W),
        zeroshot_engine: Optional[str] = None,
        device: Optional[str] = None,
        refine_edges: Optional[bool] = None,
        crop_refine: Optional[bool] = None,
    ):
        cfg = load_config()
        seg_cfg = cfg.get("segmentation", {})
        rect_cfg = cfg.get("detector", {}).get("rectifier", {})

        self.backend = backend
        self.target_size = target_size
        self.max_det = max_det
        self.device = device or get_device(cfg)

        self.edge_fit_method = str(rect_cfg.get("edge_fit_method", "ransac"))
        self.corner_excl_frac = float(rect_cfg.get("corner_excl_frac", 0.14))
        self.min_side_inliers = int(rect_cfg.get("min_side_inliers", 12))
        self.ransac_thresh_px = float(rect_cfg.get("ransac_thresh_px", 2.5))
        self.ransac_iters = int(rect_cfg.get("ransac_iters", 120))
        self.erode_mask_px = int(rect_cfg.get("erode_mask_px", 2))
        self.feather_px = int(rect_cfg.get("feather_px", 3))
        self.refine_edges = bool(
            rect_cfg.get("refine_edges", True) if refine_edges is None else refine_edges
        )
        self.crop_refine = False

        self.model = None
        self.weights_path: Optional[str] = None
        self.zeroshot_engine = zeroshot_engine or str(seg_cfg.get("zeroshot", {}).get("engine", "yoloe"))

        if backend == "seg_yolo":
            ycfg = seg_cfg.get("yolo_seg", {})
            self.imgsz = int(imgsz or ycfg.get("imgsz", 640))
            self.conf = float(conf if conf is not None else ycfg.get("conf", 0.25))
            self.iou = float(iou if iou is not None else ycfg.get("iou", 0.45))
            self.retina_masks = bool(ycfg.get("retina_masks", True))
            self.crop_refine = bool(
                ycfg.get("crop_refine", True) if crop_refine is None else crop_refine
            )
            w_path = Path(weights_path or ycfg.get("weights", "data/weights/yolo11s_seg_best.pt"))
            if not w_path.is_absolute():
                w_path = PROJECT_ROOT / w_path
            if not w_path.exists():
                raise FileNotFoundError(f"YOLO-Seg weights not found at {w_path}. Run scripts/train_seg.py first.")
            from ultralytics import YOLO

            self.weights_path = str(w_path)
            logger.info("Loading YOLO-Seg detector from %s (imgsz=%d)", self.weights_path, self.imgsz)
            self.model = YOLO(self.weights_path)

        elif backend == "seg_zeroshot":
            zcfg = seg_cfg.get("zeroshot", {})
            self.imgsz = int(imgsz or zcfg.get("imgsz", 640))
            self.conf = float(conf if conf is not None else zcfg.get("conf", 0.15))
            self.iou = float(iou if iou is not None else 0.45)
            self.text_prompts = list(zcfg.get("text_prompts", ["trading card", "pokemon card", "playing card"]))
            if self.zeroshot_engine == "fastsam":
                from ultralytics import FastSAM

                model_ref = str(weights_path or zcfg.get("fastsam_model", "FastSAM-s.pt"))
                self.weights_path = model_ref
                logger.info("Loading FastSAM zero-shot segmenter: %s", model_ref)
                self.model = FastSAM(model_ref)
            else:
                from ultralytics import YOLOE

                model_ref = str(weights_path or zcfg.get("model", "yoloe-11s-seg.pt"))
                self.weights_path = model_ref
                logger.info("Loading YOLOE zero-shot segmenter: %s", model_ref)
                self.model = YOLOE(model_ref)
                self.model.set_classes(self.text_prompts)

        elif backend == "seg_semantic":
            scfg = seg_cfg.get("semantic", {})
            self.imgsz = int(imgsz or scfg.get("imgsz", 640))
            self.threshold = float(scfg.get("threshold", 0.50))
            self.min_area_frac = float(scfg.get("min_area_frac", 0.04))
            self.conf = float(conf if conf is not None else 0.50)
            w_path = Path(weights_path or scfg.get("weights", "data/weights/seg_semantic_best.pt"))
            if not w_path.is_absolute():
                w_path = PROJECT_ROOT / w_path
            if not w_path.exists():
                raise FileNotFoundError(f"Semantic weights not found at {w_path}. Run scripts/train_seg.py --model semantic first.")
            self.weights_path = str(w_path)
            logger.info("Loading MobileNetV3-UNet semantic segmenter from %s", self.weights_path)
            net = MobileNetV3UNet(pretrained=False)
            ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=True)
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            net.load_state_dict(state)
            net.to(self.device).eval()
            self.model = net
        else:
            raise ValueError(f"Unsupported segmentation backend: {backend}")

    def _rectify(self, img_bgr: np.ndarray, mask: np.ndarray, confidence: float) -> Optional[CardDetection]:
        res = rectify_from_mask(
            img_bgr=img_bgr,
            mask=mask,
            target_size=self.target_size,
            edge_fit_method=self.edge_fit_method,
            corner_excl_frac=self.corner_excl_frac,
            min_side_inliers=self.min_side_inliers,
            ransac_thresh_px=self.ransac_thresh_px,
            ransac_iters=self.ransac_iters,
            erode_mask_px=self.erode_mask_px,
            feather_px=self.feather_px,
            refine_edges=self.refine_edges,
        )
        if res is None:
            return None
        return CardDetection(
            corners=res.corners,
            confidence=float(confidence),
            patch=res.patch_rgb,
            patch_gray_score=res.sharpness,
            mask=res.warped_mask,
        )

    def _refine_mask_on_roi(
        self,
        img_bgr: np.ndarray,
        global_mask: np.ndarray,
        kwargs: dict,
    ) -> np.ndarray:
        """Re-run YOLO-Seg on a padded crop around the initial mask for higher edge resolution."""
        h_img, w_img = img_bgr.shape[:2]
        ys, xs = np.where(global_mask >= 0.5)
        if len(xs) == 0:
            return global_mask
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        # Skip ROI crop if the card already dominates the frame
        if (bw * bh) / max(float(h_img * w_img), 1.0) >= 0.72:
            return global_mask

        pad = max(16, int(0.18 * max(bw, bh)))
        rx0 = max(0, x0 - pad)
        ry0 = max(0, y0 - pad)
        rx1 = min(w_img, x1 + pad + 1)
        ry1 = min(h_img, y1 + pad + 1)
        roi = img_bgr[ry0:ry1, rx0:rx1]
        if roi.size == 0:
            return global_mask

        roi_kwargs = dict(kwargs)
        roi_kwargs["max_det"] = 3
        roi_res = self.model.predict(roi, **roi_kwargs)[0]
        if roi_res.masks is None or len(roi_res.masks) == 0:
            return global_mask

        rh, rw = roi.shape[:2]
        global_roi = global_mask[ry0:ry1, rx0:rx1] >= 0.5
        best_iou = 0.0
        best_roi_mask: Optional[np.ndarray] = None
        for r_mask in roi_res.masks.data.cpu().numpy():
            rm = (
                cv2.resize(r_mask.astype(np.float32), (rw, rh), interpolation=cv2.INTER_LINEAR)
                if r_mask.shape[:2] != (rh, rw)
                else r_mask.astype(np.float32)
            )
            rm_bin = rm >= 0.5
            inter = float(np.logical_and(rm_bin, global_roi).sum())
            union = float(np.logical_or(rm_bin, global_roi).sum())
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_roi_mask = rm_bin.astype(np.float32)

        if best_roi_mask is None or best_iou < 0.82:
            return global_mask

        out_mask = np.zeros((h_img, w_img), dtype=np.float32)
        out_mask[ry0:ry1, rx0:rx1] = best_roi_mask
        return out_mask

    def _detect_ultralytics(self, img_bgr: np.ndarray) -> list[CardDetection]:
        h_img, w_img = img_bgr.shape[:2]
        img_area = float(h_img * w_img)

        kwargs = dict(
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            imgsz=self.imgsz,
            retina_masks=True,
            verbose=False,
        )
        if self.backend == "seg_zeroshot" and self.zeroshot_engine == "fastsam":
            kwargs.pop("max_det", None)

        result = self.model.predict(img_bgr, **kwargs)[0]
        if result.masks is None or len(result.masks) == 0:
            return []

        masks_data = result.masks.data.cpu().numpy()
        confs = (
            result.boxes.conf.cpu().numpy()
            if result.boxes is not None and result.boxes.conf is not None
            else np.ones(len(masks_data), dtype=np.float32)
        )

        candidates: list[tuple[float, np.ndarray, float]] = []
        for idx, raw_mask in enumerate(masks_data):
            conf = float(confs[idx])
            m_resized = (
                cv2.resize(raw_mask.astype(np.float32), (w_img, h_img), interpolation=cv2.INTER_LINEAR)
                if raw_mask.shape[:2] != (h_img, w_img)
                else raw_mask.astype(np.float32)
            )
            bin_m = (m_resized >= 0.5).astype(np.uint8)
            area_frac = float(bin_m.sum()) / max(img_area, 1.0)
            if area_frac < 0.04 or area_frac > 0.985:
                continue

            cnts, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            largest = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest)
            rw, rh = rect[1]
            if min(rw, rh) < 20:
                continue
            ar = max(rw, rh) / max(min(rw, rh), 1e-3)
            if not (1.15 <= ar <= 1.65):
                continue

            # For class-agnostic FastSAM, prefer card-proportioned rectangular masks
            fill_ratio = float(cv2.contourArea(largest)) / max(rw * rh, 1.0)
            if fill_ratio < 0.72:
                continue

            rank_score = conf * fill_ratio / (1.0 + 2.0 * abs(ar - (88.0 / 63.0)))
            mask_bin_f = (m_resized >= 0.5).astype(np.float32)
            if self.crop_refine and self.backend == "seg_yolo":
                mask_bin_f = self._refine_mask_on_roi(img_bgr, mask_bin_f, kwargs)
            candidates.append((rank_score, mask_bin_f, conf))

        candidates.sort(key=lambda x: x[0], reverse=True)
        detections: list[CardDetection] = []
        for _, mask_f, conf in candidates[: self.max_det]:
            det = self._rectify(img_bgr, mask_f, conf)
            if det is not None:
                detections.append(det)
        return detections

    def _detect_semantic(self, img_bgr: np.ndarray) -> list[CardDetection]:
        h_img, w_img = img_bgr.shape[:2]
        img_area = float(h_img * w_img)

        tensor = bgr_to_imagenet_tensor(img_bgr, (self.imgsz, self.imgsz), self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        prob_full = cv2.resize(prob, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        bin_mask = (prob_full >= self.threshold).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

        candidates: list[tuple[float, np.ndarray, float]] = []
        for lbl in range(1, n_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            area_frac = area / max(img_area, 1.0)
            if area_frac < self.min_area_frac:
                continue
            inst_mask = (labels == lbl).astype(np.float32)
            mean_prob = float(prob_full[labels == lbl].mean())
            candidates.append((mean_prob * area_frac, inst_mask, mean_prob))

        candidates.sort(key=lambda x: x[0], reverse=True)
        detections: list[CardDetection] = []
        for _, inst_mask, conf in candidates[: self.max_det]:
            det = self._rectify(img_bgr, inst_mask, conf)
            if det is not None:
                detections.append(det)
        return detections

    def detect(self, img_bgr) -> list[CardDetection]:
        """Segment cards in a BGR scene and return rectified patches with dynamic masks."""
        if isinstance(img_bgr, (str, Path)):
            img_bgr = cv2.imread(str(img_bgr))
            if img_bgr is None:
                return []

        if self.backend == "seg_semantic":
            return self._detect_semantic(img_bgr)
        return self._detect_ultralytics(img_bgr)

    def detect_first(self, img_bgr: np.ndarray) -> Optional[CardDetection]:
        """Return highest-confidence detection or None."""
        dets = self.detect(img_bgr)
        return dets[0] if dets else None

    @staticmethod
    def draw_detections(img_bgr: np.ndarray, detections: list[CardDetection]) -> np.ndarray:
        """Draw rectified quad contours and confidence on an RGB copy of the scene."""
        vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
        for i, d in enumerate(detections):
            pts = d.corners.astype(np.int32).reshape(-1, 1, 2)
            color = (0, 230, 150) if i == 0 else (59, 158, 255)
            cv2.polylines(vis, [pts], True, color, 3)
            cx, cy = d.corners.mean(axis=0).astype(int)
            cv2.putText(
                vis, f"{d.confidence:.2f}", (cx - 20, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )
        return vis


class OBBCanonicalUNetDetector:
    """Two-stage detector: Stage 1 Heavy YOLO-OBB -> Stage 2 Canonical Boundary UNet."""

    def __init__(
        self,
        obb_weights: Optional[str] = None,
        unet_weights: Optional[str] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
        adaptive_tta: Optional[bool] = None,
        adaptive_tta_conf: Optional[float] = None,
        corridor_px: Optional[int] = None,
        canon_size: tuple[int, int] = (CANON_H, CANON_W),
        target_size: tuple[int, int] = (TARGET_H, TARGET_W),
        device: Optional[str] = None,
    ):
        cfg = load_config()
        ou_cfg = cfg.get("detector", {}).get("obb_unet", {})
        obb_inf = cfg.get("obb", {}).get("inference", {})
        rect_cfg = cfg.get("detector", {}).get("rectifier", {})

        self.device = device or get_device(cfg)
        self.canon_h, self.canon_w = canon_size
        self.target_size = target_size
        self.corridor_px = int(ou_cfg.get("corridor_px", 28) if corridor_px is None else corridor_px)
        self.corner_excl_frac = float(rect_cfg.get("corner_excl_frac", 0.14))
        self.erode_mask_px = int(rect_cfg.get("erode_mask_px", 2))
        self.feather_px = int(rect_cfg.get("feather_px", 3))

        resolved_obb = self._resolve_obb_weights(obb_weights, ou_cfg)
        self.obb_detector = CardDetector(
            weights_path=resolved_obb,
            conf=float(obb_inf.get("conf", 0.12) if conf is None else conf),
            iou=float(obb_inf.get("iou", 0.45) if iou is None else iou),
            max_det=int(obb_inf.get("max_det", 10) if max_det is None else max_det),
            adaptive_tta=bool(ou_cfg.get("adaptive_tta", True) if adaptive_tta is None else adaptive_tta),
            adaptive_tta_conf=float(
                ou_cfg.get("adaptive_tta_conf", 0.75) if adaptive_tta_conf is None else adaptive_tta_conf
            ),
            use_pbl=False,
        )

        w_unet = Path(unet_weights or ou_cfg.get("unet_weights", "data/weights/seg_canonical_unet_best.pt"))
        if not w_unet.is_absolute():
            w_unet = PROJECT_ROOT / w_unet
        if not w_unet.exists():
            raise FileNotFoundError(
                f"Canonical Boundary UNet weights missing at {w_unet}. "
                "Run scripts/train_canonical_refiner.py first."
            )

        self.unet_weights_path = str(w_unet)
        self.weights_path = f"{Path(resolved_obb).name} + {w_unet.name}"
        logger.info("Loading Canonical Boundary UNet from %s", self.unet_weights_path)
        net = MobileNetV3UNet(pretrained=False)
        ckpt = torch.load(self.unet_weights_path, map_location=self.device, weights_only=True)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        net.load_state_dict(state)
        net.to(self.device).eval()
        self.refiner = net

    @staticmethod
    def _resolve_obb_weights(explicit: Optional[str], ou_cfg: dict) -> str:
        candidates = [
            explicit,
            ou_cfg.get("obb_weights"),
            "data/weights/yolo11l_obb_heavy_best.pt",
            "data/weights/yolo11m_obb_paired_best.pt",
            "data/weights/yolo11s_obb_paired_best.pt",
            "data/weights/yolo11s_obb_finetuned_best.pt",
        ]
        for cand in candidates:
            if not cand:
                continue
            p = Path(cand)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                return str(p)
        raise FileNotFoundError("No YOLO-OBB weights found for OBBCanonicalUNetDetector.")

    def detect(self, img_bgr) -> list[CardDetection]:
        """Run Stage-1 OBB proposals then Stage-2 Canonical UNet border refinement."""
        if isinstance(img_bgr, (str, Path)):
            img_bgr = cv2.imread(str(img_bgr))
            if img_bgr is None:
                return []

        proposals = self.obb_detector.propose_quads(img_bgr)
        detections: list[CardDetection] = []
        for prop in proposals:
            canon_bgr, _, M_canon_inv = warp_to_canonical_obb(
                img_bgr,
                prop["corners"],
                canon_size=(self.canon_h, self.canon_w),
                target_size=self.target_size,
            )
            tensor = bgr_to_imagenet_tensor(canon_bgr, (self.canon_h, self.canon_w), self.device)
            with torch.inference_mode():
                logits = self.refiner(tensor)
                prob_canon = torch.sigmoid(logits)[0, 0].cpu().numpy()

            res = rectify_from_canonical_prob(
                img_bgr=img_bgr,
                obb_corners=prop["corners"],
                prob_canon=prob_canon,
                M_canon_inv=M_canon_inv,
                target_size=self.target_size,
                corridor_px=self.corridor_px,
                corner_excl_frac=self.corner_excl_frac,
                erode_mask_px=self.erode_mask_px,
                feather_px=self.feather_px,
            )
            if res is None:
                continue
            detections.append(
                CardDetection(
                    corners=res.corners,
                    confidence=float(prop["conf"]),
                    patch=res.patch_rgb,
                    patch_gray_score=res.sharpness,
                    mask=res.warped_mask,
                )
            )
        return detections

    def detect_first(self, img_bgr: np.ndarray) -> Optional[CardDetection]:
        """Return highest-confidence detection or None."""
        dets = self.detect(img_bgr)
        return dets[0] if dets else None

    @staticmethod
    def draw_detections(img_bgr: np.ndarray, detections: list[CardDetection]) -> np.ndarray:
        """Draw refined quad contours on an RGB copy of the scene."""
        return SegmentationCardDetector.draw_detections(img_bgr, detections)

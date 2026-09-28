"""Module for Pokémon card detection via YOLO OBB and perspective rectification.

Detects cards in arbitrary orientation (0-360 degrees) and executes
perspective warping to produce canonical upright patches (630x448 px)
suitable for surface anomaly inspection.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TARGET_H = 630
TARGET_W = 448

TARGET_AR = 88.0 / 63.0

@dataclass
class CardDetection:
    """One detected card in a scene."""
    corners: np.ndarray
    confidence: float
    patch: np.ndarray
    patch_gray_score: float

class CardDetector:
    """YOLO OBB-based Pokémon card detector with perspective-warp cropping."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf: float = 0.12,
        iou: float = 0.45,
        max_det: int = 10,
        target_size: tuple = (TARGET_H, TARGET_W),
        imgsz: int = 640,
        corner_shrink: float = 0.0,
        refine: bool = False,
    ):
        from ultralytics import YOLO

        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.imgsz = imgsz
        self.corner_shrink = corner_shrink
        self.refine = refine
        self.target_h, self.target_w = target_size

        if weights_path is None:
            weights_path = self._find_latest_weights()
        if weights_path is None:
            raise FileNotFoundError(
                "No OBB weights found. Train first: python scripts/train_obb.py"
            )
        logger.info(f"Loading OBB detector from {weights_path}")
        self.model = YOLO(str(weights_path))
        self.weights_path = str(weights_path)

    @staticmethod
    def _find_latest_weights() -> Optional[str]:
        """Locate the most recent best.pt across experiment runs (yolo11s then yolo11n)."""
        project_root = Path(__file__).resolve().parents[2]
        shared_ft = project_root / "data/weights/yolo11s_obb_finetuned_best.pt"
        if shared_ft.exists():
            return str(shared_ft)
        root = project_root / "experiments"
        if not root.exists():
            root = Path(__file__).resolve().parents[1] / "experiments"
        ft_cands = sorted(root.glob("*phase1c*/artifacts/yolo11s_obb_finetuned_best.pt"))
        if ft_cands:
            return str(ft_cands[-1])
        s_cands = sorted(root.glob("*phase1b*/artifacts/yolo11s_obb_best.pt"))
        if s_cands:
            return str(s_cands[-1])
        n_cands = sorted(root.glob("*phase1b*/artifacts/yolo11n_obb_best.pt"))
        if n_cands:
            return str(n_cands[-1])
        raw = sorted(root.glob("*phase1b*/artifacts/yolo_runs/obb_train/weights/best.pt"))
        return str(raw[-1]) if raw else None

    @staticmethod
    def _unrotate_quad(corners: np.ndarray, k: int, width: int, height: int) -> np.ndarray:
        """Map OBB corners from a ``np.rot90(image, k)`` view back to original pixels."""
        if k == 0:
            return corners.copy()
        if k == 1:
            return np.stack([width - 1 - corners[:, 1], corners[:, 0]], axis=1)
        if k == 2:
            return np.stack([width - 1 - corners[:, 0], height - 1 - corners[:, 1]], axis=1)
        return np.stack([corners[:, 1], height - 1 - corners[:, 0]], axis=1)

    @staticmethod
    def _align_quad(corners: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
        """Align cyclic/reversed OBB vertex order to a reference quad."""
        best, best_error = corners, float("inf")
        for order in (corners, corners[::-1]):
            for shift in range(4):
                candidate = np.roll(order, shift, axis=0)
                error = float(np.linalg.norm(candidate - reference, axis=1).mean())
                if error < best_error:
                    best, best_error = candidate, error
        return best, best_error

    def detect(self, img_bgr) -> list[CardDetection]:
        """
        Detect cards in a BGR scene (or image file path) and return warped upright patches.

        Returns a list of CardDetection sorted by confidence (descending).
        Enforces strict full-card geometry: discards partial sub-rectangles or split boxes.
        """
        if isinstance(img_bgr, (str, Path)):
            img_bgr = cv2.imread(str(img_bgr))
            if img_bgr is None:
                return []

        h_img, w_img = img_bgr.shape[:2]
        img_area = float(h_img * w_img)

        raw_candidates = []

        def collect_candidates(view: np.ndarray, rotation: int = 0, offset=(0, 0)):
            """Runs YOLO-OBB inference on a view and maps candidates back to unrotated coordinates."""
            result = self.model.predict(
                view, conf=self.conf, iou=self.iou, max_det=self.max_det,
                imgsz=self.imgsz, verbose=False
            )[0]
            if result.obb is None or len(result.obb) == 0:
                return
            boxes = result.obb.xyxyxyxy.cpu().numpy()
            confs = result.obb.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                corners = box.astype(np.float64) - np.asarray(offset, dtype=np.float64)
                corners = self._unrotate_quad(corners, rotation, w_img, h_img)
                rect = cv2.minAreaRect(corners.astype(np.float32))
                rw, rh = rect[1]
                side_short, side_long = min(rw, rh), max(rw, rh)
                ar = side_long / max(side_short, 1e-3)
                area_frac = (side_short * side_long) / max(img_area, 1.0)
                if 1.15 <= ar <= 1.65 and area_frac >= 0.06:
                    score = float(conf) / (1.0 + 3.0 * abs(ar - TARGET_AR))
                    raw_candidates.append({"corners": corners, "conf": float(conf), "ar": ar, "score": score})

        for rotation in range(4):
            view = np.ascontiguousarray(np.rot90(img_bgr, rotation)) if rotation else img_bgr
            collect_candidates(view, rotation)

        needs_context = not raw_candidates
        for cand in raw_candidates:
            q = cand["corners"]
            edge_margin = min(float(q[:, 0].min()), float(q[:, 1].min()),
                              float(w_img - 1 - q[:, 0].max()), float(h_img - 1 - q[:, 1].max()))
            rect = cv2.minAreaRect(q.astype(np.float32))
            area_frac = rect[1][0] * rect[1][1] / max(img_area, 1.0)
            if edge_margin < 12.0 or area_frac > 0.55:
                needs_context = True
                break
        if needs_context:
            for rotation in range(4):
                view = np.ascontiguousarray(np.rot90(img_bgr, rotation)) if rotation else img_bgr
                view_h, view_w = view.shape[:2]
                pad_h, pad_w = int(view_h * 0.08), int(view_w * 0.08)
                padded = cv2.copyMakeBorder(view, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
                collect_candidates(padded, rotation, (pad_w, pad_h))

        raw_candidates.sort(key=lambda x: x["score"], reverse=True)
        clusters = []
        for cand in raw_candidates:
            center = cand["corners"].mean(axis=0)
            diag = float(np.linalg.norm(cand["corners"][0] - cand["corners"][2]))
            cluster = next((c for c in clusters
                            if np.linalg.norm(center - c[0]["corners"].mean(axis=0)) < 0.35 * diag), None)
            if cluster is None:
                clusters.append([cand])
            else:
                ref = max(cluster, key=lambda x: x["score"])["corners"]
                aligned, alignment_error = self._align_quad(cand["corners"], ref)
                if alignment_error <= max(12.0, 0.05 * diag):
                    candidate = dict(cand)
                    candidate["corners"] = aligned
                    cluster.append(candidate)

        kept_candidates = []
        for cluster in clusters:
            reference = max(cluster, key=lambda x: x["score"])
            ref_corners = reference["corners"]
            fused_items = []
            for item in cluster:
                aligned, alignment_error = self._align_quad(item["corners"], ref_corners)
                diag = float(np.linalg.norm(ref_corners[0] - ref_corners[2]))
                if alignment_error <= max(12.0, 0.05 * diag):
                    fused_items.append((item, aligned))
            weights = np.array([max(item["score"], 1e-4) ** 2 for item, _ in fused_items])
            weights /= weights.sum()
            fused = np.sum(np.stack([q for _, q in fused_items]) * weights[:, None, None], axis=0)
            conf = float(np.average([item["conf"] for item, _ in fused_items], weights=weights))
            rect = cv2.minAreaRect(fused.astype(np.float32))
            rw, rh = rect[1]
            ar = max(rw, rh) / max(min(rw, rh), 1e-3)
            score = conf / (1.0 + 3.0 * abs(ar - TARGET_AR))
            kept_candidates.append({"corners": fused, "conf": conf, "ar": ar, "score": score})
        kept_candidates.sort(key=lambda x: x["score"], reverse=True)
        kept_candidates = kept_candidates[:self.max_det]

        detections: list[CardDetection] = []
        for cand in kept_candidates:
            corners = cand["corners"]
            conf = cand["conf"]
            if self.corner_shrink > 0:
                ctr = corners.mean(axis=0)
                corners = ctr + (corners - ctr) * (1.0 - self.corner_shrink)

            corners = self._lock_physical_card(img_bgr, corners)

            if self.refine:
                patch, corners = self._refine_warp(img_bgr, corners)
            else:
                patch = self.warp_card(img_bgr, corners)
            if patch is None:
                continue

            r_rect = cv2.minAreaRect(corners.astype(np.float32))
            rw_r, rh_r = r_rect[1]
            r_ar = max(rw_r, rh_r) / max(min(rw_r, rh_r), 1e-3)
            if not (1.15 <= r_ar <= 1.65):
                logger.warning(
                    "Discarded post-refine: aspect ratio %.2f not conforming to standard Pokémon card [1.15, 1.65]",
                    r_ar,
                )
                continue
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            detections.append(CardDetection(
                corners=corners,
                confidence=float(conf),
                patch=patch,
                patch_gray_score=sharp,
            ))

        return detections

    def detect_first(self, img_bgr: np.ndarray) -> Optional[CardDetection]:
        """Highest-confidence detection or None."""
        dets = self.detect(img_bgr)
        return dets[0] if dets else None

    def _lock_physical_card(self, img_bgr: np.ndarray, yolo_corners: np.ndarray) -> np.ndarray:
        """Lock onto the physical card border to discard protective sleeves, fingers, and background.

        Step 1 regularizes the YOLO quadrilateral. A self-calibrated, color-agnostic ring
        estimator then proposes a physical-border contour in Lab space. The proposal is
        accepted only when it is centered, non-expanding, card-shaped, and close to YOLO;
        otherwise the regularized detection is retained. No color class is hardcoded.
        """
        h_img, w_img = img_bgr.shape[:2]
        rect_yolo = cv2.minAreaRect(yolo_corners.astype(np.float32))
        (cx, cy), (yw, yh), y_ang = rect_yolo
        yolo_area = max(yw, yh) * min(yw, yh)
        if yolo_area < 1000:
            return yolo_corners

        short_y, long_y = min(yw, yh), max(yw, yh)
        adj_long = long_y * 0.985
        adj_short = adj_long / TARGET_AR
        reg_rect = ((cx, cy), (adj_long, adj_short), y_ang) if yw > yh else ((cx, cy), (adj_short, adj_long), y_ang)
        regularized_corners = cv2.boxPoints(reg_rect).astype(np.float64)

        border_color = self._estimate_border_color(img_bgr, regularized_corners)
        if border_color is not None:
            best_candidate, min_diff_ar = self._color_contour_search(
                img_bgr, border_color, (cx, cy), (yw, yh), y_ang, yolo_area
            )
            if best_candidate is not None:
                min_cand_dist = self._quad_distance(best_candidate, regularized_corners)
                max_correction = max(10.0, 0.025 * long_y)
                if min_cand_dist <= max_correction:
                    logger.info(
                        "Physical card boundary locked (color ring): AR dev %.4f, delta vs YOLO %.2f px",
                        min_diff_ar, min_cand_dist
                    )
                    return best_candidate
                logger.warning(
                    "Sanity Gate triggered: rejected color contour (deviation %.1f px > %.1f px). Fallback to regularized.",
                    min_cand_dist, max_correction
                )

        return regularized_corners

    def _quad_distance(self, cand: np.ndarray, ref: np.ndarray) -> float:
        """Order-agnostic mean nearest-corner distance between two quads."""
        return float(np.mean([np.min(np.linalg.norm(ref - p, axis=1)) for p in cand]))

    def _estimate_border_color(self, img_bgr: np.ndarray, reg_corners: np.ndarray) -> Optional[np.ndarray]:
        """Estimate the border color via per-side radial profile search (color-agnostic).

        A color qualifies as border only if, on each of the four sides, a thin strip at some
        inward depth is covered by that color (the border is a closed ring); among qualifying
        colors the one most distinctive from the surrounding background wins.
        """
        h_img, w_img = img_bgr.shape[:2]
        rect = cv2.minAreaRect(reg_corners.astype(np.float32))
        (cx, cy), (lw, sw), ang = rect
        ang = float(ang) if np.isfinite(ang) else 0.0
        rect_vertices = cv2.boxPoints(((cx, cy), (lw, sw), ang)).astype(np.float64)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)

        bg_ring, _, _ = self._rect_band(img_bgr, rect, grow_out=30, grow_in=-8)
        bys, bxs = np.nonzero(bg_ring)
        bg_color = np.median(lab[bys, bxs].astype(np.float64), axis=0) if len(bxs) > 50 else None

        wide, _, _ = self._rect_band(img_bgr, rect, grow_out=6, grow_in=18)
        wys, wxs = np.nonzero(wide)
        if len(wxs) < 500:
            return None
        wide_pts = lab[wys, wxs].astype(np.float64)
        if len(wide_pts) > 20000:
            idx = np.random.default_rng(0).choice(len(wide_pts), 20000, replace=False)
            wide_pts = wide_pts[idx]
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        cv2.setRNGSeed(7)
        _, _, centers = cv2.kmeans(wide_pts.astype(np.float32), 6, None, crit, 1, cv2.KMEANS_PP_CENTERS)

        depths = list(range(1, 21, 2))
        strip_cov = {}
        for s in range(4):
            for d in depths:
                r_out = ((cx, cy), (lw - 2 * d, sw - 2 * d), ang)
                r_in = ((cx, cy), (max(lw - 2 * (d + 2), 2), max(sw - 2 * (d + 2), 2)), ang)
                m = np.zeros((h_img, w_img), np.uint8)
                cv2.fillPoly(m, [cv2.boxPoints(r_out).astype(np.int32)], 255)
                cv2.fillPoly(m, [cv2.boxPoints(r_in).astype(np.int32)], 0)
                sys_, sxs = np.nonzero(m & (wide > 0))
                if len(sxs) < 30:
                    continue
                pts_rect = np.stack([sxs, sys_], 1).astype(np.float64)
                sides = self._side_index(pts_rect, rect_vertices)
                sel = sides == s
                if np.any(sel):
                    strip_cov[(s, d)] = (sys_[sel], sxs[sel])

        best, best_score = None, 0.0
        for ci in range(len(centers)):
            c = centers[ci].astype(np.float64)
            dist_bg = float(np.linalg.norm(c - bg_color)) if bg_color is not None else 100.0
            per_side_best = []
            for s in range(4):
                bcv = 0.0
                for d in depths:
                    if (s, d) not in strip_cov:
                        continue
                    sys_, sxs = strip_cov[(s, d)]
                    dd = np.linalg.norm(lab[sys_, sxs].astype(np.float64) - c, axis=1)
                    bcv = max(bcv, float((dd <= 32.0).mean()))
                per_side_best.append(bcv)
            min_cov = min(per_side_best)
            mean_cov = float(np.mean(per_side_best))
            if min_cov >= 0.35 and mean_cov >= 0.50:
                score = mean_cov * (0.4 + 0.6 * min(1.0, dist_bg / 60.0))
                if score > best_score:
                    best, best_score = c, score
        return best

    @staticmethod
    def _side_index(pts: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """Assign annulus pixels to the nearest rectangle side in image coordinates.

        Segment distance avoids angle-convention assumptions in ``minAreaRect`` and remains
        stable when the input photograph is rotated by any in-plane angle.
        """
        pts = np.asarray(pts, dtype=np.float64)
        quad = np.asarray(quad, dtype=np.float64)
        if len(pts) == 0 or not np.isfinite(pts).all() or not np.isfinite(quad).all():
            return np.zeros(len(pts), dtype=np.int64)
        distances = np.full((len(pts), 4), np.inf, dtype=np.float64)
        for i in range(4):
            a = quad[i]
            edge = quad[(i + 1) % 4] - a
            denom = float(np.dot(edge, edge))
            if denom <= 1e-9:
                continue
            rel = pts - a
            t = np.clip(np.sum(rel * edge[None, :], axis=1) / denom, 0.0, 1.0)
            nearest = a + t[:, None] * edge
            distances[:, i] = np.linalg.norm(pts - nearest, axis=1)
        return np.argmin(distances, axis=1)

    @staticmethod
    def _rect_band(img_bgr: np.ndarray, rect: tuple, grow_out: float, grow_in: float) -> tuple:
        """Mask of the ring between rect scaled outward and inward (rotated annulus)."""
        (cx, cy), (lw, sw), ang = rect
        outer = cv2.boxPoints(((cx, cy), (lw + 2 * grow_out, sw + 2 * grow_out), ang))
        inner = cv2.boxPoints(((cx, cy), (max(lw - 2 * grow_in, 2), max(sw - 2 * grow_in, 2)), ang))
        m = np.zeros(img_bgr.shape[:2], np.uint8)
        cv2.fillPoly(m, [outer.astype(np.int32)], 255)
        cv2.fillPoly(m, [inner.astype(np.int32)], 0)
        return m, outer, inner

    def _color_contour_search(
        self,
        img_bgr: np.ndarray,
        border_color: np.ndarray,
        center: tuple,
        dims: tuple,
        ang: float,
        yolo_area: float,
    ) -> tuple:
        """Run the 20-threshold contour sweep on the generic Lab likelihood of the border color."""
        h_img, w_img = img_bgr.shape[:2]
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float64)
        d_full = np.linalg.norm(lab - border_color, axis=2)
        s_map = np.exp(-(d_full / 55.0) ** 2)

        mask_yolo = np.zeros((h_img, w_img), np.uint8)
        yw, yh = dims
        cv2.fillPoly(
            mask_yolo,
            [cv2.boxPoints(((center[0], center[1]), (yw + 21, yh + 21), ang)).astype(np.int32)],
            255,
        )

        s_vals = s_map[mask_yolo > 0]
        p40, p98 = np.percentile(s_vals, 40), np.percentile(s_vals, 98)
        thresholds = np.linspace(p40, p98, 20)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        best_candidate, min_diff_ar, best_rect_area = None, 1e9, 0.0
        for th in thresholds:
            mask = ((s_map > th) & (mask_yolo > 0)).astype(np.uint8) * 255
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if not (0.90 * yolo_area <= area <= 1.00 * yolo_area):
                    continue
                rect = cv2.minAreaRect(c)
                rw, rh = rect[1]
                short_s, long_s = min(rw, rh), max(rw, rh)
                if short_s < 80:
                    continue
                rect_area = short_s * long_s
                if rect_area > 1.00 * yolo_area or rect_area < 0.88 * yolo_area:
                    continue
                center_shift = float(np.linalg.norm(np.asarray(rect[0], dtype=np.float64) - center))
                if center_shift > 0.02 * max(dims):
                    continue
                diff_ar = abs(long_s / short_s - TARGET_AR)
                if diff_ar < 0.040 and (
                    diff_ar < min_diff_ar - 1e-5
                    or (abs(diff_ar - min_diff_ar) <= 1e-5 and rect_area > best_rect_area)
                ):
                    min_diff_ar = diff_ar
                    best_rect_area = rect_area
                    best_candidate = cv2.boxPoints(rect).astype(np.float64)
        return best_candidate, min_diff_ar

    @staticmethod
    def _order_quad(q: np.ndarray) -> np.ndarray:
        """Cyclic TL,TR,BR,BL order (top = short edge, upright portrait orientation)."""
        q = np.asarray(q, np.float64)
        ctr = q.mean(axis=0)
        ang = np.arctan2(q[:, 1] - ctr[1], q[:, 0] - ctr[0])
        q = q[np.argsort(ang)]

        candidates = []
        for i in range(4):
            r = np.roll(q, -i, axis=0)
            if np.linalg.norm(r[1] - r[0]) < np.linalg.norm(r[3] - r[0]):
                mid_y = (r[0, 1] + r[1, 1]) / 2.0
                candidates.append((mid_y, r))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        return q

    @staticmethod
    def _fit_line(pts: np.ndarray) -> Optional[tuple[float, float, float, float]]:
        """Fit a 2D line (A*x + B*y + C = 0) with normalized normal vector and angle in deg."""
        if len(pts) < 15:
            return None
        line = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
        vx, vy, x0, y0 = [float(v) for v in line.flatten()]
        A = -vy
        B = vx
        C = vy * x0 - vx * y0
        norm = np.hypot(A, B)
        if norm < 1e-8:
            return None
        ang = float(np.degrees(np.arctan2(vy, vx)))
        return (A / norm, B / norm, C / norm, ang)

    @staticmethod
    def _intersect_lines(l1, l2) -> Optional[np.ndarray]:
        """Compute the 2D intersection point of two lines A1*x + B1*y + C1 = 0 and A2*x + B2*y + C2 = 0."""
        if l1 is None or l2 is None:
            return None
        A1, B1, C1, _ = l1
        A2, B2, C2, _ = l2
        det = A1 * B2 - A2 * B1
        if abs(det) < 0.1:
            return None
        x = (B1 * C2 - B2 * C1) / det
        y = (A2 * C1 - A1 * C2) / det
        return np.array([x, y], dtype=np.float64)

    def _refine_quad_lines(
        self,
        img_bgr: np.ndarray,
        corners: np.ndarray,
        pad: int = 50,
        search_in: int = 55,
        search_out: int = 25,
        n_samples: int = 60,
    ) -> tuple[Optional[np.ndarray], np.ndarray]:
        """Refine card quadrilateral by detecting the 4 physical card edges.

        Handles cards inside transparent protective sleeves/toploaders by scanning
        an expanded inward search window. Uses directional Sobel gradients
        (edge-normal component only) and Median Absolute Deviation filtering
        to lock onto the high-contrast physical card boundary while rejecting
        faint sleeve borders and background reflections.
        """
        q = self._order_quad(corners)
        target_w, target_h = self.target_w, self.target_h
        W_pad = target_w + 2 * pad
        H_pad = target_h + 2 * pad

        dst_pad = np.array([
            [pad, pad],
            [pad + target_w - 1, pad],
            [pad + target_w - 1, pad + target_h - 1],
            [pad, pad + target_h - 1]
        ], dtype=np.float32)

        try:
            M_pad = cv2.getPerspectiveTransform(q.astype(np.float32), dst_pad)
            patch = cv2.warpPerspective(img_bgr, M_pad, (W_pad, H_pad),
                                        borderMode=cv2.BORDER_REPLICATE)
        except cv2.error:
            return None, q

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gy = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3))
        gx = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))

        def _sample_edge(mag, expected_pos, axis, s_out, s_in):
            """Sample edge points along one border using directional gradient magnitude."""
            pts = []
            for t in np.linspace(0.18, 0.82, n_samples):
                if axis == 'h':
                    x = int(pad + t * (target_w - 1))
                    y_lo = max(2, expected_pos - s_out)
                    y_hi = min(H_pad - 3, expected_pos + s_in)
                    col = mag[y_lo:y_hi, x]
                    if len(col) > 0 and col.max() > 25:
                        pts.append([x, y_lo + int(np.argmax(col))])
                else:
                    y = int(pad + t * (target_h - 1))
                    x_lo = max(2, expected_pos - s_out)
                    x_hi = min(W_pad - 3, expected_pos + s_in)
                    row = mag[y, x_lo:x_hi]
                    if len(row) > 0 and row.max() > 25:
                        pts.append([x_lo + int(np.argmax(row)), y])
            return np.array(pts, dtype=np.float32) if pts else np.empty((0, 2), dtype=np.float32)

        top_pts = _sample_edge(gy, pad, 'h', search_out, search_in)
        bot_pts = _sample_edge(gy, pad + target_h - 1, 'h', search_in, search_out)
        left_pts = _sample_edge(gx, pad, 'v', search_out, search_in)
        right_pts = _sample_edge(gx, pad + target_w - 1, 'v', search_in, search_out)

        def _mad_filter(pts, axis_idx):
            """Remove outliers using Median Absolute Deviation on the scan axis."""
            if len(pts) < 10:
                return pts
            vals = pts[:, axis_idx]
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            if mad < 1.0:
                mad = 1.0
            mask = np.abs(vals - med) < 3.5 * mad
            return pts[mask]

        top_pts = _mad_filter(top_pts, 1)
        bot_pts = _mad_filter(bot_pts, 1)
        left_pts = _mad_filter(left_pts, 0)
        right_pts = _mad_filter(right_pts, 0)

        l_top = self._fit_line(top_pts) if len(top_pts) >= 15 else None
        l_bot = self._fit_line(bot_pts) if len(bot_pts) >= 15 else None
        l_left = self._fit_line(left_pts) if len(left_pts) >= 15 else None
        l_right = self._fit_line(right_pts) if len(right_pts) >= 15 else None

        def check_angle(line, expected, tol=18.0):
            """Verifies line angle against expected cardinal direction within tolerance."""
            if line is None:
                return False
            ang = line[3]
            diff = min(abs(ang - expected), abs(ang - (expected - 180)),
                       abs(ang - (expected + 180)))
            return diff <= tol

        if not (check_angle(l_top, 0) and check_angle(l_bot, 0) and
                check_angle(l_left, 90) and check_angle(l_right, 90)):
            return None, q

        tl = self._intersect_lines(l_top, l_left)
        tr = self._intersect_lines(l_top, l_right)
        br = self._intersect_lines(l_bot, l_right)
        bl = self._intersect_lines(l_bot, l_left)

        if tl is None or tr is None or br is None or bl is None:
            return None, q

        w_top = np.linalg.norm(tr - tl)
        w_bot = np.linalg.norm(br - bl)
        h_left = np.linalg.norm(bl - tl)
        h_right = np.linalg.norm(br - tr)

        if not (0.90 * target_w <= w_top <= 1.10 * target_w and
                0.90 * target_w <= w_bot <= 1.10 * target_w and
                0.90 * target_h <= h_left <= 1.10 * target_h and
                0.90 * target_h <= h_right <= 1.10 * target_h):
            return None, q

        if (abs(w_top - w_bot) / max(w_top, w_bot, 1.0) > 0.035 or
                abs(h_left - h_right) / max(h_left, h_right, 1.0) > 0.035):
            return None, q

        try:
            M_pad_inv = np.linalg.inv(M_pad).astype(np.float32)
            quad_pad = np.array([tl, tr, br, bl], dtype=np.float32).reshape(-1, 1, 2)
            refined_corners = cv2.perspectiveTransform(quad_pad, M_pad_inv).reshape(-1, 2)

            dst_final = np.array([
                [0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]
            ], dtype=np.float32)
            M_final = cv2.getPerspectiveTransform(refined_corners.astype(np.float32), dst_final)
            warped = cv2.warpPerspective(img_bgr, M_final, (target_w, target_h))
            if warped is None or warped.mean() < 1:
                return None, q
            return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), refined_corners
        except Exception:
            return None, q

    def _refine_warp(self, img_bgr: np.ndarray, corners: np.ndarray):
        """Snap the quad to edge evidence only when the correction stays within 5 px."""
        patch, refined_corners = self._refine_quad_lines(img_bgr, corners)
        if patch is not None and self._quad_distance(refined_corners, corners) <= 5.0:
            return patch, refined_corners
        return self.warp_card(img_bgr, corners), corners

    def warp_card(self, img_bgr: np.ndarray, corners: np.ndarray) -> Optional[np.ndarray]:
        """Perspective-warp a rotated card (4 corners) into an upright canonical RGB patch."""
        quad = self._order_quad(corners)
        dst = np.array(
            [[0, 0], [self.target_w - 1, 0],
             [self.target_w - 1, self.target_h - 1], [0, self.target_h - 1]],
            np.float32,
        )
        M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        warped = cv2.warpPerspective(img_bgr, M, (self.target_w, self.target_h))
        if warped is None or warped.mean() < 1:
            return None
        return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    @staticmethod
    def draw_detections(img_bgr: np.ndarray, detections: list[CardDetection]) -> np.ndarray:
        """Draw OBB contours + confidence on a copy of the scene (RGB out)."""
        vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
        for i, d in enumerate(detections):
            pts = d.corners.astype(np.int32).reshape(-1, 1, 2)
            color = (255, 215, 0) if i == 0 else (59, 158, 255)
            cv2.polylines(vis, [pts], True, color, 3)
            cx, cy = d.corners.mean(axis=0).astype(int)
            cv2.putText(vis, f"{d.confidence:.2f}", (cx - 20, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return vis

"""Train the SOTA 4-Corner Heatmap Regressor (MobileNetV3-UNet + Soft-Argmax) on pokemon_seg.

Uses exact ground-truth trapezoid corners (``meta/*/scene_*.json``) and pixel masks
(``masks/*``) with realistic detector-proposal perturbations and on-the-fly projective
perspective / blur / photometric augmentations.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.corner_regressor import (  # noqa: E402
    PAD_FRAC,
    PATCH_H,
    PATCH_W,
    CornerHeatmapUNet,
    extract_oriented_padded_patch,
    soft_argmax_2d,
)
from backend.experiment import Experiment, get_device, load_config, set_seed  # noqa: E402
from backend.rectifier import order_quad  # noqa: E402

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _load_split_samples(ds_root: Path, split: str) -> list[tuple[Path, Path, np.ndarray]]:
    img_dir = ds_root / "images" / split
    mask_dir = ds_root / "masks" / split
    meta_dir = ds_root / "meta" / split
    samples: list[tuple[Path, Path, np.ndarray]] = []
    for meta_p in sorted(meta_dir.glob("*.json")):
        data = json.loads(meta_p.read_text())
        cards = data.get("cards", [])
        if len(cards) != 1:
            # Train single-card corner regressor cleanly on single-card scenes
            continue
        corners = np.asarray(cards[0]["corners_gt"], dtype=np.float64)
        if corners.shape != (4, 2):
            continue
        stem = meta_p.stem
        img_p = next((img_dir / f"{stem}{ext}" for ext in (".jpg", ".jpeg", ".png") if (img_dir / f"{stem}{ext}").exists()), None)
        mask_p = mask_dir / f"{stem}.png"
        if img_p is not None and mask_p.exists():
            samples.append((img_p, mask_p, order_quad(corners)))
    return samples


class CornerPatchDataset(Dataset):
    """Generates similarity-rotated padded patches with detector perturbation + perspective boost."""

    def __init__(
        self,
        samples: list[tuple[Path, Path, np.ndarray]],
        augment: bool = True,
        patch_hw: tuple[int, int] = (PATCH_H, PATCH_W),
        pad_frac: float = PAD_FRAC,
        seed: int = 42,
    ):
        self.samples = samples
        self.augment = augment
        self.patch_hw = patch_hw
        self.pad_frac = pad_frac
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _random_projective_warp(
        self,
        img_bgr: np.ndarray,
        mask_u8: np.ndarray,
        corners: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = img_bgr.shape[:2]
        src = corners.astype(np.float32)
        span = float(max(np.ptp(src[:, 0]), np.ptp(src[:, 1]), 80.0))
        jitter = self.rng.uniform(-0.14 * span, 0.14 * span, size=(4, 2)).astype(np.float32)
        dst = src + jitter
        # Keep warped corners inside image frame
        if np.any(dst[:, 0] < 12) or np.any(dst[:, 0] > w - 12) or np.any(dst[:, 1] < 12) or np.any(dst[:, 1] > h - 12):
            return img_bgr, mask_u8, corners
        try:
            M_p = cv2.getPerspectiveTransform(src, dst)
        except cv2.error:
            return img_bgr, mask_u8, corners
        w_img = cv2.warpPerspective(img_bgr, M_p, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        w_msk = cv2.warpPerspective(mask_u8, M_p, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return w_img, w_msk, order_quad(dst.astype(np.float64))

    def _perturb_proposal(self, gt_corners: np.ndarray) -> np.ndarray:
        """Simulate noisy OBB / Seg detector box around GT corners."""
        q = gt_corners.copy()
        ctr = q.mean(axis=0)
        w_est = float(np.linalg.norm(q[1] - q[0]))
        h_est = float(np.linalg.norm(q[3] - q[0]))
        ang = np.radians(float(self.rng.uniform(-11.0, 11.0))) if self.augment else 0.0
        sc_x = float(self.rng.uniform(0.90, 1.14)) if self.augment else 1.0
        sc_y = float(self.rng.uniform(0.90, 1.14)) if self.augment else 1.0
        shift = self.rng.uniform(-0.08 * min(w_est, h_est), 0.08 * min(w_est, h_est), size=(1, 2)) if self.augment else np.zeros((1, 2))

        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=np.float64)
        rel = q - ctr
        rel[:, 0] *= sc_x
        rel[:, 1] *= sc_y
        prop = (rel @ R.T) + ctr + shift
        if self.augment:
            # Also occasionally use strict minAreaRect (simulating OBB rectangle without perspective)
            if self.rng.random() < 0.45:
                rect = cv2.minAreaRect(prop.astype(np.float32))
                prop = order_quad(cv2.boxPoints(rect).astype(np.float64))
        return prop

    def _photometric_aug(self, patch_bgr: np.ndarray) -> np.ndarray:
        out = patch_bgr.astype(np.float32)
        if self.rng.random() < 0.7:
            alpha = float(self.rng.uniform(0.72, 1.28))
            beta = float(self.rng.uniform(-28.0, 28.0))
            out = np.clip(out * alpha + beta, 0, 255)
        if self.rng.random() < 0.35:
            k = int(self.rng.choice([3, 5]))
            out = cv2.GaussianBlur(out, (k, k), 0)
        if self.rng.random() < 0.25:
            noise = self.rng.normal(0.0, float(self.rng.uniform(3.0, 10.0)), size=out.shape).astype(np.float32)
            out = np.clip(out + noise, 0, 255)
        return out.astype(np.uint8)

    def __getitem__(self, idx: int):
        img_p, mask_p, corners_gt = self.samples[idx]
        img_bgr = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        mask_u8 = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        h0, w0 = img_bgr.shape[:2]
        # Fast ROI pre-crop around card (+45% margin) before projective/affine warps
        span = float(max(np.ptp(corners_gt[:, 0]), np.ptp(corners_gt[:, 1]), 80.0))
        pad_c = int(0.45 * span)
        rx0 = max(0, int(np.floor(corners_gt[:, 0].min())) - pad_c)
        ry0 = max(0, int(np.floor(corners_gt[:, 1].min())) - pad_c)
        rx1 = min(w0, int(np.ceil(corners_gt[:, 0].max())) + pad_c)
        ry1 = min(h0, int(np.ceil(corners_gt[:, 1].max())) + pad_c)
        if rx1 - rx0 > 64 and ry1 - ry0 > 64:
            img_bgr = img_bgr[ry0:ry1, rx0:rx1]
            mask_u8 = mask_u8[ry0:ry1, rx0:rx1]
            corners_gt = corners_gt - np.array([rx0, ry0], dtype=np.float64)

        if self.augment and self.rng.random() < 0.40:
            img_bgr, mask_u8, corners_gt = self._random_projective_warp(img_bgr, mask_u8, corners_gt)

        prop = self._perturb_proposal(corners_gt)
        patch_bgr, M_sim, _ = extract_oriented_padded_patch(img_bgr, prop, patch_hw=self.patch_hw, pad_frac=self.pad_frac)
        ph, pw = self.patch_hw
        patch_mask = cv2.warpAffine(mask_u8, M_sim, (pw, ph), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        ones = np.ones((4, 1), dtype=np.float64)
        patch_corners = (np.hstack([corners_gt, ones]) @ M_sim.T).astype(np.float32)

        if self.augment:
            patch_bgr = self._photometric_aug(patch_bgr)

        rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_t = (rgb.transpose(2, 0, 1) - _IMAGENET_MEAN) / _IMAGENET_STD

        hm_h, hm_w = ph // 2, pw // 2
        mask_hm = cv2.resize((patch_mask > 127).astype(np.float32), (hm_w, hm_h), interpolation=cv2.INTER_NEAREST)

        # Build normalized 2D Gaussian target heatmaps (sigma=3.5 at stride-2)
        ys = np.arange(hm_h, dtype=np.float32)[:, None]
        xs = np.arange(hm_w, dtype=np.float32)[None, :]
        sigma = 3.5
        targets_hm = np.zeros((4, hm_h, hm_w), dtype=np.float32)
        for k in range(4):
            cx = patch_corners[k, 0] * (float(hm_w - 1) / float(pw - 1))
            cy = patch_corners[k, 1] * (float(hm_h - 1) / float(ph - 1))
            g = np.exp(-0.5 * (((xs - cx) ** 2 + (ys - cy) ** 2) / (sigma ** 2)))
            g_sum = float(g.sum())
            targets_hm[k] = g / g_sum if g_sum > 1e-8 else (1.0 / float(hm_h * hm_w))

        return (
            torch.from_numpy(img_t),
            torch.from_numpy(patch_corners),
            torch.from_numpy(targets_hm),
            torch.from_numpy(mask_hm).unsqueeze(0),
        )


def _wing_loss(pred: torch.Tensor, target: torch.Tensor, w: float = 10.0, eps: float = 2.0) -> torch.Tensor:
    diff = torch.abs(pred - target)
    c = w - w * np.log(1.0 + w / eps)
    small = w * torch.log1p(diff / eps)
    large = diff - c
    return torch.where(diff < w, small, large).mean()


def _soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(2, 3))
    denom = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-6
    return (1.0 - 2.0 * inter / denom).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 4-Corner Heatmap Regressor (MobileNetV3-UNet)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    set_seed(int(cfg["project"]["seed"]))
    device = get_device(cfg)

    ds_root = PROJECT_ROOT / "data" / "datasets" / "pokemon_seg"
    train_samples = _load_split_samples(ds_root, "train")
    val_samples = _load_split_samples(ds_root, "val")
    if args.max_train > 0:
        train_samples = train_samples[: args.max_train]
    logger.info("Loaded corner datasets: train=%d, val=%d on device=%s", len(train_samples), len(val_samples), device)

    cv2.setNumThreads(1)
    train_ds = CornerPatchDataset(train_samples, augment=True, seed=42)
    val_ds = CornerPatchDataset(val_samples, augment=False, seed=99)
    val_cache = [val_ds[i] for i in range(len(val_ds))]
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_dl = DataLoader(val_cache, batch_size=args.batch_size, shuffle=False, num_workers=0)

    exp = Experiment("train_corner_regressor", cfg)
    model = CornerHeatmapUNet(pretrained=not args.resume).to(device)
    out_weights = PROJECT_ROOT / "data" / "weights" / "corner_heatmap_best.pt"
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    best_val_err = float("inf")
    if args.resume and out_weights.exists():
        ckpt = torch.load(out_weights, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        best_val_err = float(ckpt.get("val_mean_px", float("inf")))
        logger.info("Resumed weights from %s (best_val_mean_px=%.2f px)", out_weights, best_val_err)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for imgs, gt_pts, gt_hm, gt_mask in train_dl:
            imgs = imgs.to(device)
            gt_pts = gt_pts.to(device)
            gt_hm = gt_hm.to(device)
            gt_mask = gt_mask.to(device)

            optimizer.zero_grad()
            c_logits, m_logits = model(imgs)
            pred_pts, probs = soft_argmax_2d(c_logits, out_hw=(PATCH_H, PATCH_W))

            loss_coord = _wing_loss(pred_pts, gt_pts, w=10.0, eps=2.0)
            log_probs = torch.log(probs.clamp(min=1e-8))
            loss_kl = F.kl_div(log_probs, gt_hm, reduction="batchmean")
            loss_mask = 0.5 * F.binary_cross_entropy_with_logits(m_logits, gt_mask) + 0.5 * _soft_dice_loss(m_logits, gt_mask)
            loss = loss_coord + 2.0 * loss_kl + 5.0 * loss_mask

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            tr_loss += float(loss.item()) * len(imgs)
            n_tr += len(imgs)
        scheduler.step()

        model.eval()
        val_errs: list[float] = []
        with torch.no_grad():
            for imgs, gt_pts, _, _ in val_dl:
                imgs = imgs.to(device)
                gt_pts = gt_pts.to(device)
                c_logits, _ = model(imgs)
                pred_pts, _ = soft_argmax_2d(c_logits, out_hw=(PATCH_H, PATCH_W))
                d = torch.linalg.norm(pred_pts - gt_pts, dim=-1).mean(dim=-1)
                val_errs.extend(d.cpu().numpy().tolist())

        val_mean = float(np.mean(val_errs))
        val_med = float(np.median(val_errs))
        dt = time.perf_counter() - t0
        logger.info(
            "Epoch %02d/%02d (%.1fs) | train_loss=%.4f | val_mean=%.2fpx | val_med=%.2fpx",
            epoch,
            args.epochs,
            dt,
            tr_loss / max(n_tr, 1),
            val_mean,
            val_med,
        )
        exp.log_metric(f"epoch_{epoch}_val_mean_px", val_mean)
        if val_mean < best_val_err:
            best_val_err = val_mean
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_mean_px": val_mean}, out_weights)

    exp.log_metric("best_val_mean_px", best_val_err)
    exp.finish()
    logger.info("Saved best weights to %s (best_val_mean_px=%.2f px)", out_weights, best_val_err)


if __name__ == "__main__":
    main()

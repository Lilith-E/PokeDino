"""Train Stage-2 Canonical Boundary MobileNetV3-UNet refiner on padded canonical card crops.

Expects ``data/datasets/pokemon_canon_refiner/`` produced by
``scripts/prepare_canonical_refiner_dataset.py`` with ``(768, 544)`` portrait
images, binary validity masks (occlusion-aware), and canonical corner labels.
Saves the best checkpoint to ``data/weights/seg_canonical_unet_best.pt`` with
``weights_only=True``-compatible tensors and scalar metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.experiment import Experiment, get_device, load_config, set_seed  # noqa: E402
from backend.rectifier import CANON_H, CANON_W, fit_quad_from_canonical_prob  # noqa: E402
from backend.seg_detector import MobileNetV3UNet, bgr_to_imagenet_tensor  # noqa: E402
from scripts.benchmark_segmentation_vs_obb import align_corner_error  # noqa: E402
from scripts.train_seg import _boundary_loss, _dice_loss  # noqa: E402

logger = logging.getLogger(__name__)


class CanonicalRefinerDataset(Dataset):
    """Dataset of canonical ``(768, 544)`` padded OBB crops and occlusion-aware masks."""

    def __init__(
        self,
        root: Path,
        split: str,
        augment: bool = False,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.root = root
        self.split = split
        self.augment = augment
        self.rng = random.Random(seed)
        img_paths = sorted((root / "images" / split).glob("canon_*.jpg"))
        if max_samples is not None and max_samples > 0:
            img_paths = img_paths[:max_samples]
        self.samples: list[tuple[Path, Path, Path]] = []
        for img_p in img_paths:
            mask_p = root / "masks" / split / f"{img_p.stem}.png"
            meta_p = root / "meta" / split / f"{img_p.stem}.json"
            if mask_p.exists() and meta_p.exists():
                self.samples.append((img_p, mask_p, meta_p))
        if not self.samples:
            raise FileNotFoundError(
                f"No canonical refiner samples found under {root / 'images' / split}. "
                "Run scripts/prepare_canonical_refiner_dataset.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_train_aug(
        self, bgr: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.rng.random() < 0.5:
            bgr = cv2.flip(bgr, 1)
            mask = cv2.flip(mask, 1)
        if self.rng.random() < 0.5:
            bgr = cv2.flip(bgr, -1)
            mask = cv2.flip(mask, -1)

        if self.rng.random() < 0.6:
            alpha = self.rng.uniform(0.84, 1.16)
            beta = self.rng.uniform(-16.0, 16.0)
            bgr = np.clip(bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if self.rng.random() < 0.35:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.rng.uniform(0.85, 1.18), 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * self.rng.uniform(0.88, 1.12), 0, 255)
            bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if self.rng.random() < 0.18:
            k = self.rng.choice((3, 5))
            bgr = cv2.GaussianBlur(bgr, (k, k), 0)
        return bgr, mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        img_p, mask_p, meta_p = self.samples[idx]
        bgr = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            raise IOError(f"Failed loading canonical sample {img_p}")
        if bgr.shape[:2] != (CANON_H, CANON_W):
            bgr = cv2.resize(bgr, (CANON_W, CANON_H), interpolation=cv2.INTER_LINEAR)
        if mask.shape[:2] != (CANON_H, CANON_W):
            mask = cv2.resize(mask, (CANON_W, CANON_H), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            bgr, mask = self._apply_train_aug(bgr, mask)

        img_t = bgr_to_imagenet_tensor(bgr, (CANON_H, CANON_W), "cpu").squeeze(0)
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        meta = json.loads(meta_p.read_text())
        corners = np.asarray(meta["corners_canon"], dtype=np.float32)
        return img_t, mask_t, corners


def _weighted_border_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Border-weighted BCE + Dice + Sobel boundary loss emphasizing card margins."""
    dil = F.max_pool2d(targets, kernel_size=9, stride=1, padding=4)
    ero = -F.max_pool2d(-targets, kernel_size=9, stride=1, padding=4)
    edge_band = (dil - ero).clamp(0.0, 1.0)
    weight = 1.0 + 3.0 * edge_band
    bce_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weighted_bce = (bce_map * weight).mean()
    return 0.45 * weighted_bce + 0.35 * _dice_loss(logits, targets) + 0.20 * _boundary_loss(logits, targets)


def _evaluate_split(
    model: MobileNetV3UNet,
    loader: DataLoader,
    device: str,
    corridor_px: int,
    max_corner_eval: Optional[int] = None,
) -> dict[str, float]:
    """Compute IoU, 9px-perimeter boundary IoU, and geometric corner error."""
    model.eval()
    inter_total = 0.0
    union_total = 0.0
    b_inter_total = 0.0
    b_union_total = 0.0
    loss_total = 0.0
    n_batches = 0
    corner_errs: list[float] = []

    with torch.no_grad():
        for imgs, masks, corners_gt in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(imgs)
            loss_total += float(_weighted_border_loss(logits, masks).item())
            n_batches += 1

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            inter_total += float((preds * masks).sum().item())
            union_total += float(((preds + masks) > 0).sum().item())

            dil_p = F.max_pool2d(preds, kernel_size=9, stride=1, padding=4)
            ero_p = -F.max_pool2d(-preds, kernel_size=9, stride=1, padding=4)
            band_p = (dil_p - ero_p) > 0.5
            dil_g = F.max_pool2d(masks, kernel_size=9, stride=1, padding=4)
            ero_g = -F.max_pool2d(-masks, kernel_size=9, stride=1, padding=4)
            band_g = (dil_g - ero_g) > 0.5
            b_inter_total += float((band_p & band_g).sum().item())
            b_union_total += float((band_p | band_g).sum().item())

            probs_np = probs[:, 0].cpu().numpy()
            gt_np = corners_gt.numpy()
            for prob_map, c_gt in zip(probs_np, gt_np):
                if max_corner_eval is not None and len(corner_errs) >= max_corner_eval:
                    break
                quad_pred = fit_quad_from_canonical_prob(
                    prob_map,
                    corridor_px=corridor_px,
                )
                corner_errs.append(align_corner_error(quad_pred, c_gt))

    iou = inter_total / union_total if union_total > 0 else 0.0
    b_iou = b_inter_total / b_union_total if b_union_total > 0 else 0.0
    mean_ce = float(np.mean(corner_errs)) if corner_errs else float("inf")
    med_ce = float(np.median(corner_errs)) if corner_errs else float("inf")
    p90_ce = float(np.percentile(corner_errs, 90)) if corner_errs else float("inf")
    return {
        "loss": loss_total / max(1, n_batches),
        "iou": iou,
        "boundary_iou": b_iou,
        "corner_err_mean": mean_ce,
        "corner_err_median": med_ce,
        "corner_err_p90": p90_ce,
    }


def _save_training_plots(history: list[dict], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val Loss")
    axes[0].set_title("Border-Weighted Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, [h["val_iou"] for h in history], label="Val Mask IoU")
    axes[1].plot(epochs, [h["val_boundary_iou"] for h in history], label="Val Boundary IoU (9px)")
    axes[1].set_title("Validation IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(epochs, [h["val_corner_err_mean"] for h in history], label="Mean Corner Err (px)")
    axes[2].plot(epochs, [h["val_corner_err_median"] for h in history], label="Median Corner Err (px)")
    axes[2].set_title("Canonical Corner Error (px @ 630x448)")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage-2 Canonical MobileNetV3-UNet refiner.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15, help="Early-stopping patience on validation score (0=off).")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap per split for smoke tests.")
    parser.add_argument("--out-weights", type=str, default=None, help="Optional override checkpoint path.")
    parser.add_argument("--no-pretrained", action="store_true", help="Skip torchvision ImageNet backbone download.")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("train_canonical_refiner", cfg)

    obb_unet_cfg = cfg["detector"]["obb_unet"]
    ds_name = obb_unet_cfg["dataset"]["name"]
    ds_root = PROJECT_ROOT / cfg["paths"]["datasets_dir"] / ds_name
    corridor_px = int(obb_unet_cfg.get("corridor_px", 28))
    device = get_device(cfg)
    pin_mem = device == "cuda"

    train_ds = CanonicalRefinerDataset(ds_root, "train", augment=True, max_samples=args.max_samples, seed=cfg["project"]["seed"])
    val_ds = CanonicalRefinerDataset(ds_root, "val", augment=False, max_samples=args.max_samples, seed=cfg["project"]["seed"])
    test_ds = CanonicalRefinerDataset(ds_root, "test", augment=False, max_samples=args.max_samples, seed=cfg["project"]["seed"])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=pin_mem, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=pin_mem
    )

    model = MobileNetV3UNet(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-6)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_weights = Path(args.out_weights) if args.out_weights else (PROJECT_ROOT / obb_unet_cfg["unet_weights"])
    out_weights.parent.mkdir(parents=True, exist_ok=True)

    hparams = {
        "architecture": "MobileNetV3UNet",
        "canon_size": [CANON_H, CANON_W],
        "corridor_px": corridor_px,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch,
        "lr": args.lr,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss": "0.45*BorderWeightedBCE(1+3*band9px) + 0.35*Dice + 0.20*SobelBoundary",
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "device": device,
        "seed": cfg["project"]["seed"],
    }
    (exp.artifacts_dir / "hyperparameters.json").write_text(json.dumps(hparams, indent=2))

    best_score = -1e9
    best_corner_err = float("inf")
    no_improve = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_steps = 0
        for imgs, masks, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs)
                loss = _weighted_border_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            n_steps += 1
        scheduler.step()

        train_loss = running_loss / max(1, n_steps)
        val_stats = _evaluate_split(model, val_loader, device=device, corridor_px=corridor_px)
        row = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_stats["loss"], 5),
            "val_iou": round(val_stats["iou"], 5),
            "val_boundary_iou": round(val_stats["boundary_iou"], 5),
            "val_corner_err_mean": round(val_stats["corner_err_mean"], 4),
            "val_corner_err_median": round(val_stats["corner_err_median"], 4),
            "val_corner_err_p90": round(val_stats["corner_err_p90"], 4),
        }
        history.append(row)
        logger.info(
            "Epoch %02d/%02d | train_loss=%.4f | val_iou=%.4f | val_b_iou=%.4f | val_ce=%.2f px (med=%.2f)",
            epoch,
            args.epochs,
            train_loss,
            val_stats["iou"],
            val_stats["boundary_iou"],
            val_stats["corner_err_mean"],
            val_stats["corner_err_median"],
        )

        # Combine geometric corner precision and boundary IoU for checkpoint selection.
        selection_score = val_stats["iou"] + 0.5 * val_stats["boundary_iou"] - 0.015 * val_stats["corner_err_mean"]
        if selection_score > best_score:
            best_score = selection_score
            best_corner_err = val_stats["corner_err_mean"]
            no_improve = 0
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            ckpt_payload = {
                "model": cpu_state,
                "epoch": int(epoch),
                "val_iou": float(val_stats["iou"]),
                "val_boundary_iou": float(val_stats["boundary_iou"]),
                "val_corner_err_px": float(val_stats["corner_err_mean"]),
                "canon_size": [CANON_H, CANON_W],
            }
            torch.save(ckpt_payload, out_weights)
            exp.log_metric("best_epoch", epoch)
            exp.log_metric("best_val_iou", val_stats["iou"])
            exp.log_metric("best_val_boundary_iou", val_stats["boundary_iou"])
            exp.log_metric("best_val_corner_err_px", val_stats["corner_err_mean"])
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                logger.info(
                    "Early stopping triggered after %d epochs without validation improvement (best corner_err=%.2f px).",
                    args.patience,
                    best_corner_err,
                )
                break

    # Evaluate final best checkpoint on held-out test split.
    best_ckpt = torch.load(out_weights, map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model"])
    test_stats = _evaluate_split(model, test_loader, device=device, corridor_px=corridor_px)
    logger.info(
        "TEST split -> iou=%.4f | boundary_iou=%.4f | corner_err_mean=%.2f px | med=%.2f px | p90=%.2f px",
        test_stats["iou"],
        test_stats["boundary_iou"],
        test_stats["corner_err_mean"],
        test_stats["corner_err_median"],
        test_stats["corner_err_p90"],
    )
    for k, v in test_stats.items():
        exp.log_metric(f"test_{k}", v)

    (exp.artifacts_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    _save_training_plots(history, exp.plots_dir / "canonical_unet_training.png")
    exp.copy_artifact(out_weights, out_weights.name)
    exp.finish()
    logger.info("Saved best canonical UNet weights (best val corner_err=%.2f px) -> %s", best_corner_err, out_weights)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    main()

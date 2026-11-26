"""Training pipeline for card segmentation models (YOLO11s-Seg and MobileNetV3-UNet)."""

import argparse
import json
import logging
import shutil
import sys
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

from backend.experiment import Experiment, get_device, load_config, set_seed  # noqa: E402
from backend.seg_detector import MobileNetV3UNet  # noqa: E402

logger = logging.getLogger(__name__)


class SemanticCardDataset(Dataset):
    """Paired RGB scene and binary mask dataset for semantic card segmentation."""

    def __init__(self, dataset_root: Path, split: str, imgsz: int = 640, augment: bool = False):
        self.img_dir = dataset_root / "images" / split
        self.mask_dir = dataset_root / "masks" / split
        self.img_paths = sorted(self.img_dir.glob("*.jpg"))
        self.imgsz = imgsz
        self.augment = augment
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_paths[idx]
        mask_path = self.mask_dir / f"{img_path.stem}.png"

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img_rgb.shape[:2] != (self.imgsz, self.imgsz):
            img_rgb = cv2.resize(img_rgb, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            k_rot = np.random.randint(0, 4)
            if k_rot:
                img_rgb = np.rot90(img_rgb, k_rot).copy()
                mask = np.rot90(mask, k_rot).copy()
            if np.random.rand() < 0.5:
                img_rgb = np.fliplr(img_rgb).copy()
                mask = np.fliplr(mask).copy()
            gain = np.random.uniform(0.85, 1.15)
            bias = np.random.uniform(-15.0, 15.0)
            img_rgb = np.clip(img_rgb.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

        img_f = (img_rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        img_t = torch.from_numpy(img_f.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return img_t, mask_t


def _dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def _boundary_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Edge-weighted L1 loss on Sobel gradient magnitude to sharpen card borders."""
    probs = torch.sigmoid(logits)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=logits.dtype, device=logits.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)

    def grad_mag(t: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(t, kx, padding=1)
        gy = F.conv2d(t, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    return F.l1_loss(grad_mag(probs), grad_mag(targets))


def train_yolo_seg(
    cfg: dict,
    imgsz: int,
    epochs: int,
    batch: int,
    out_weights: Path,
    workers: int = 2,
    mask_ratio: int = 1,
    base_model_override: str | None = None,
) -> None:
    """Train YOLO11-Seg on pokemon_seg and copy best weights to data/weights/."""
    from ultralytics import YOLO

    exp = Experiment(f"train_yolo_seg_{imgsz}", cfg)
    dataset_root = PROJECT_ROOT / cfg["paths"]["datasets_dir"] / cfg["segmentation"]["dataset"]["name"]
    dataset_yaml = dataset_root / "pokemon_seg.yaml"
    if not dataset_yaml.exists():
        exp.logger.error("Dataset YAML missing: %s. Run scripts/prepare_seg_dataset.py first.", dataset_yaml)
        sys.exit(1)

    import yaml

    yaml_payload = yaml.safe_load(dataset_yaml.read_text())
    yaml_payload["path"] = str(dataset_root.resolve())
    resolved_yaml = exp.artifacts_dir / "pokemon_seg_resolved.yaml"
    resolved_yaml.write_text(yaml.safe_dump(yaml_payload, sort_keys=False))

    device = get_device(cfg)
    base_model = base_model_override or cfg["segmentation"]["yolo_seg"].get("base_model", "yolo11s-seg.pt")
    hparams = {
        "model": base_model,
        "imgsz": imgsz,
        "epochs": epochs,
        "batch": batch,
        "workers": workers,
        "mask_ratio": mask_ratio,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "cos_lr": True,
        "patience": 12,
        "weight_decay": 0.0005,
        "warmup_epochs": 2.5,
        "box": 10.0,
        "overlap_mask": False,
        "mosaic": 0.35,
        "close_mosaic": 8,
        "degrees": 180,
        "fliplr": 0.5,
        "flipud": 0.5,
        "scale": 0.40,
        "translate": 0.10,
        "perspective": 0.0005,
        "hsv_h": 0.02,
        "hsv_s": 0.35,
        "hsv_v": 0.45,
        "seed": cfg["project"]["seed"],
        "device": str(device),
    }
    (exp.artifacts_dir / "hyperparameters.json").write_text(json.dumps(hparams, indent=2))
    exp.log_metric("model", base_model)
    exp.log_metric("imgsz", imgsz)
    exp.log_metric("epochs", epochs)
    exp.log_metric("device", device)

    model = YOLO(base_model)
    results = model.train(
        data=str(resolved_yaml),
        epochs=epochs,
        batch=batch,
        workers=workers,
        imgsz=imgsz,
        lr0=0.001,
        lrf=0.01,
        optimizer="AdamW",
        cos_lr=True,
        patience=12,
        weight_decay=0.0005,
        warmup_epochs=2.5,
        box=10.0,
        mask_ratio=mask_ratio,
        overlap_mask=False,
        mosaic=0.35,
        close_mosaic=8,
        mixup=0.0,
        degrees=180,
        fliplr=0.5,
        flipud=0.5,
        scale=0.40,
        translate=0.10,
        perspective=0.0005,
        hsv_h=0.02,
        hsv_s=0.35,
        hsv_v=0.45,
        seed=cfg["project"]["seed"],
        device=device,
        project=str(exp.artifacts_dir / "yolo_runs"),
        name=f"seg_{imgsz}",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (exp.artifacts_dir / "yolo_runs" / f"seg_{imgsz}")
    for png_file in sorted(run_dir.glob("*.png")):
        shutil.copy2(png_file, exp.plots_dir / png_file.name)
    for data_file in sorted(list(run_dir.glob("*.csv")) + list(run_dir.glob("*.yaml"))):
        shutil.copy2(data_file, exp.artifacts_dir / f"yolo_{data_file.name}")

    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        out_weights.parent.mkdir(parents=True, exist_ok=True)
        out_weights.write_bytes(best_pt.read_bytes())
        exp.copy_artifact(best_pt, out_weights.name)
        exp.logger.info("Saved best YOLO-Seg weights → %s", out_weights)
    else:
        exp.logger.error("Expected best.pt missing in %s", run_dir)
    exp.finish()


def train_semantic_unet(
    cfg: dict,
    imgsz: int,
    epochs: int,
    batch: int,
    out_weights: Path,
    workers: int = 0,
) -> None:
    """Train full-resolution MobileNetV3-UNet on pokemon_seg."""
    exp = Experiment("train_seg_semantic", cfg)
    dataset_root = PROJECT_ROOT / cfg["paths"]["datasets_dir"] / cfg["segmentation"]["dataset"]["name"]
    device = torch.device(get_device(cfg))

    train_ds = SemanticCardDataset(dataset_root, "train", imgsz=imgsz, augment=True)
    val_ds = SemanticCardDataset(dataset_root, "val", imgsz=imgsz, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers)

    model = MobileNetV3UNet(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    bce = nn.BCEWithLogitsLoss()
    grad_accum = max(1, 16 // max(1, batch))

    best_iou = 0.0
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    hparams = {
        "model": "MobileNetV3UNet",
        "pretrained": True,
        "imgsz": imgsz,
        "epochs": epochs,
        "batch": batch,
        "grad_accum": grad_accum,
        "effective_batch": batch * grad_accum,
        "workers": workers,
        "optimizer": "AdamW",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "loss": "0.5*BCE + 0.35*Dice + 0.15*Boundary",
        "seed": cfg["project"]["seed"],
        "device": str(device),
    }
    (exp.artifacts_dir / "hyperparameters.json").write_text(json.dumps(hparams, indent=2))
    history_rows = ["epoch,train_loss,val_iou,lr\n"]
    history_data: list[tuple[int, float, float, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        n_batches = len(train_loader)
        for step, (imgs, masks) in enumerate(train_loader, start=1):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = 0.5 * bce(logits, masks) + 0.35 * _dice_loss(logits, masks) + 0.15 * _boundary_loss(logits, masks)
            (loss / grad_accum).backward()
            if step % grad_accum == 0 or step == n_batches:
                optimizer.step()
                optimizer.zero_grad()
            train_loss += float(loss.item()) * imgs.size(0)
            if step % 150 == 0 or step == n_batches:
                print(
                    f"[semantic epoch {epoch}/{epochs}] batch {step}/{n_batches} loss={float(loss.item()):.4f}",
                    flush=True,
                )
        scheduler.step()
        train_loss /= max(1, len(train_ds))

        model.eval()
        inter_sum = 0.0
        union_sum = 0.0
        with torch.inference_mode():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = (torch.sigmoid(model(imgs)) >= 0.5).float()
                inter_sum += float((preds * masks).sum().item())
                union_sum += float(((preds + masks) > 0).sum().item())

        val_iou = inter_sum / max(union_sum, 1.0)
        lr_now = float(optimizer.param_groups[0]["lr"])
        exp.logger.info("Epoch %d/%d | train_loss=%.4f | val_iou=%.4f | lr=%.2e", epoch, epochs, train_loss, val_iou, lr_now)
        exp.log_metric(f"epoch_{epoch}.train_loss", round(train_loss, 5))
        exp.log_metric(f"epoch_{epoch}.val_iou", round(val_iou, 5))
        history_rows.append(f"{epoch},{train_loss:.6f},{val_iou:.6f},{lr_now:.8f}\n")
        history_data.append((epoch, train_loss, val_iou, lr_now))
        (exp.artifacts_dir / "semantic_history.csv").write_text("".join(history_rows))

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"model": model.state_dict(), "val_iou": val_iou, "imgsz": imgsz}, out_weights)
            exp.copy_artifact(out_weights, out_weights.name)
            exp.logger.info("New best semantic checkpoint saved → %s (IoU=%.4f)", out_weights, best_iou)

        if device.type == "mps":
            torch.mps.empty_cache()

    if history_data:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ep_arr = [r[0] for r in history_data]
            loss_arr = [r[1] for r in history_data]
            iou_arr = [r[2] for r in history_data]

            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=160)
            axes[0].plot(ep_arr, loss_arr, marker="o", color="#B91C1C", linewidth=2.0)
            axes[0].set_title("MobileNetV3-UNet Training Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("BCE + Dice + Boundary Loss")
            axes[0].grid(True, linestyle="--", alpha=0.4)

            axes[1].plot(ep_arr, iou_arr, marker="s", color="#0F766E", linewidth=2.0)
            axes[1].set_title(f"Validation IoU (Best = {best_iou:.4f})")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("IoU")
            axes[1].set_ylim(0.0, 1.0)
            axes[1].grid(True, linestyle="--", alpha=0.4)

            fig.tight_layout()
            curve_path = exp.plots_dir / "semantic_training_curves.png"
            fig.savefig(curve_path)
            fig.savefig(exp.artifacts_dir / "semantic_training_curves.png")
            plt.close(fig)
        except Exception as exc:
            exp.logger.warning("Could not save semantic training plot: %s", exc)

    exp.log_metric("best_val_iou", round(best_iou, 5))
    exp.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PokéDINO segmentation models.")
    parser.add_argument("--model", choices=["yolo640", "yolo960", "semantic", "all"], default="all")
    parser.add_argument("--epochs-yolo", type=int, default=12)
    parser.add_argument("--epochs-semantic", type=int, default=12)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mask-ratio", type=int, default=1)
    parser.add_argument("--base-model", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    if args.model in ("yolo640", "all"):
        w640 = PROJECT_ROOT / cfg["segmentation"]["yolo_seg"]["weights"]
        train_yolo_seg(
            cfg, imgsz=640, epochs=args.epochs_yolo, batch=args.batch,
            out_weights=w640, workers=args.workers, mask_ratio=args.mask_ratio,
            base_model_override=args.base_model,
        )

    if args.model in ("yolo960", "all"):
        w960 = PROJECT_ROOT / cfg["segmentation"]["yolo_seg"]["weights_960"]
        train_yolo_seg(
            cfg, imgsz=960, epochs=args.epochs_yolo, batch=max(4, args.batch // 2),
            out_weights=w960, workers=args.workers, mask_ratio=args.mask_ratio,
            base_model_override=args.base_model,
        )

    if args.model in ("semantic", "all"):
        wsem = PROJECT_ROOT / cfg["segmentation"]["semantic"]["weights"]
        sem_workers = 0 if sys.platform == "darwin" else args.workers
        train_semantic_unet(
            cfg, imgsz=640, epochs=args.epochs_semantic, batch=args.batch,
            out_weights=wsem, workers=sem_workers,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

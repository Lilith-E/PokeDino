"""Edge-focused YOLO11s-OBB fine-tuning procedure for boundary alignment.

Loads optimal weights from previous run (Epoch 10) and applies a loss
re-weighted heavily toward precise edge localization:
- box loss: 18.0 (2.4x default)
- dfl loss: 3.5 (2.3x default)
- angle loss: 2.5 (2.5x default)
- cls loss: 0.15 (de-emphasized, single class already saturated at 100%)
- mosaic: 0.0 (no artificial edge cuts during fine-tuning)
- mixup: 0.0
- lr0: 0.0003 (gentle learning rate with cosine decay over 5 epochs)
"""

import logging
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.experiment import Experiment, load_config, set_seed, get_device, resolve_path

logger = logging.getLogger(__name__)

def main():
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("phase1c_finetune_obb", cfg)

    from ultralytics import YOLO

    dataset_yaml = PROJECT_ROOT / cfg["paths"]["datasets_dir"] / cfg["obb"]["dataset"]["name"] / "pokemon_obb.yaml"
    if not dataset_yaml.exists():
        exp.logger.error(f"Dataset YAML missing: {dataset_yaml}. Run scripts/prepare_dataset.py first.")
        sys.exit(1)

    device = get_device(cfg)
    imgsz = cfg["obb"]["model"]["imgsz"]

    exp_dir = resolve_path(cfg, "paths.experiments_dir")
    cands = sorted(exp_dir.glob("*phase1b*/artifacts/yolo11s_obb_best.pt"))
    if cands:
        latest_weights = cands[-1]
    else:
        fallback = PROJECT_ROOT.parent / "data/weights/yolo11s_obb_best.pt"
        if fallback.exists():
            latest_weights = fallback
        else:
            exp.logger.error("Could not find yolo11s_obb_best.pt from Epoch 10.")
            sys.exit(1)

    exp.logger.info(f"Loading checkpoint from Epoch 10: {latest_weights}")
    model = YOLO(str(latest_weights))

    ft_cfg = {
        "epochs": 5,
        "batch": 16,
        "workers": 4,
        "imgsz": imgsz,
        "lr0": 0.0003,
        "lrf": 0.01,
        "optimizer": "AdamW",
        "cos_lr": True,
        "patience": 5,
        "weight_decay": 0.0005,
        "warmup_epochs": 0.5,
        "box": 18.0,
        "dfl": 3.5,
        "angle": 2.5,
        "cls": 0.15,
        "mosaic": 0.0,
        "mixup": 0.0,
        "close_mosaic": 0,
        "erasing": 0.05,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "degrees": 180,
        "fliplr": 0.5,
        "flipud": 0.5,
        "scale": 0.25,
        "translate": 0.05,
        "perspective": 0.0002,
    }

    exp.log_metric("base_weights", str(latest_weights))
    exp.log_metric("imgsz", imgsz)
    exp.log_metric("device", device)
    exp.log_metrics({f"ft.{k}": v for k, v in ft_cfg.items()})

    exp.logger.info("Starting Edge-Focused OBB Fine-Tuning for 5 epochs…")
    results = model.train(
        data=str(dataset_yaml),
        epochs=ft_cfg["epochs"],
        batch=ft_cfg["batch"],
        workers=ft_cfg["workers"],
        imgsz=imgsz,
        lr0=ft_cfg["lr0"],
        lrf=ft_cfg["lrf"],
        optimizer=ft_cfg["optimizer"],
        cos_lr=ft_cfg["cos_lr"],
        patience=ft_cfg["patience"],
        weight_decay=ft_cfg["weight_decay"],
        warmup_epochs=ft_cfg["warmup_epochs"],
        box=ft_cfg["box"],
        dfl=ft_cfg["dfl"],
        angle=ft_cfg["angle"],
        cls=ft_cfg["cls"],
        mosaic=ft_cfg["mosaic"],
        mixup=ft_cfg["mixup"],
        close_mosaic=ft_cfg["close_mosaic"],
        erasing=ft_cfg["erasing"],
        hsv_h=ft_cfg["hsv_h"],
        hsv_s=ft_cfg["hsv_s"],
        hsv_v=ft_cfg["hsv_v"],
        degrees=ft_cfg["degrees"],
        fliplr=ft_cfg["fliplr"],
        flipud=ft_cfg["flipud"],
        scale=ft_cfg["scale"],
        translate=ft_cfg["translate"],
        perspective=ft_cfg["perspective"],
        seed=cfg["project"]["seed"],
        device=device,
        project=str(exp.artifacts_dir / "yolo_runs"),
        name="obb_finetune",
        exist_ok=True,
        verbose=True,
    )

    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (exp.artifacts_dir / "yolo_runs" / "obb_finetune")
    best_pt = run_dir / "weights" / "best.pt"

    for fname in ["results.png", "results.csv", "confusion_matrix.png",
                  "confusion_matrix_normalized.png", "PR_curve.png", "F_curve.png",
                  "P_curve.png", "R_curve.png", "args.yaml", "train_batch0.jpg",
                  "val_batch0_labels.jpg", "val_batch0_pred.jpg"]:
        src = run_dir / fname
        if src.exists():
            exp.copy_artifact(src, f"ft_{fname}")

    model_target = "yolo11s_obb_finetuned_best.pt"
    if best_pt.exists():
        exp.copy_artifact(best_pt, model_target)
        shared_dest = PROJECT_ROOT.parent / "data/weights" / model_target
        shared_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, shared_dest)
        exp.logger.info(f"Finetuned best weights saved → {exp.artifact_path(model_target)} and {shared_dest}")

    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        import pandas as pd

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        last = df.iloc[-1]
        metric_map = {
            "metrics/mAP50(B)": "map50",
            "metrics/mAP50-95(B)": "map50_95",
            "metrics/precision(B)": "precision",
            "metrics/recall(B)": "recall",
        }
        for col, key in metric_map.items():
            if col in df.columns:
                exp.log_metric(f"final.{key}", float(last[col]))
        exp.log_metric("final.epoch", int(last["epoch"]))
        if "train/box_loss" in df.columns:
            exp.log_metric("final.train_box_loss", float(last["train/box_loss"]))
        if "train/angle_loss" in df.columns:
            exp.log_metric("final.train_angle_loss", float(last["train/angle_loss"]))

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor("#1e1e2e")
        for ax in axs:
            ax.set_facecolor("#181825")
            ax.grid(alpha=0.25, color="#585b70")
            ax.tick_params(colors="#cdd6f4")
            for sp in ax.spines.values():
                sp.set_color("#45475a")

        if "metrics/mAP50(B)" in df.columns:
            axs[0].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50", color="#89b4fa", linewidth=2)
            axs[0].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95", color="#f38ba8", linewidth=2)
            axs[0].set_title("mAP vs Epoch", color="#cdd6f4", fontweight="bold")
            axs[0].set_xlabel("Epoch", color="#a6adc8")
            axs[0].legend(facecolor="#1e1e2e", labelcolor="#cdd6f4")

        if "train/box_loss" in df.columns:
            axs[1].plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", color="#f9e2af", linewidth=2)
            if "val/box_loss" in df.columns:
                axs[1].plot(df["epoch"], df["val/box_loss"], label="Val Box Loss", color="#a6e3a1", linewidth=2)
            axs[1].set_title("Box Alignment Loss (18x)", color="#cdd6f4", fontweight="bold")
            axs[1].set_xlabel("Epoch", color="#a6adc8")
            axs[1].legend(facecolor="#1e1e2e", labelcolor="#cdd6f4")

        if "train/angle_loss" in df.columns:
            axs[2].plot(df["epoch"], df["train/angle_loss"], label="Train Angle Loss", color="#cba6f7", linewidth=2)
            if "val/angle_loss" in df.columns:
                axs[2].plot(df["epoch"], df["val/angle_loss"], label="Val Angle Loss", color="#fab387", linewidth=2)
            axs[2].set_title("Angle Loss (2.5x)", color="#cdd6f4", fontweight="bold")
            axs[2].set_xlabel("Epoch", color="#a6adc8")
            axs[2].legend(facecolor="#1e1e2e", labelcolor="#cdd6f4")

        fig.suptitle("Phase 1c - YOLO11s-OBB Edge & Alignment Fine-Tuning", color="#ffffff", fontweight="bold")
        fig.tight_layout()
        exp.save_plot(fig, "finetuning_curves")

    if best_pt.exists():
        exp.logger.info("Validating fine-tuned model on validation set…")
        best_model = YOLO(str(best_pt))
        metrics = best_model.val(
            data=str(dataset_yaml),
            split="val",
            imgsz=imgsz,
            device=device,
            project=str(exp.artifacts_dir / "yolo_runs"),
            name="obb_ft_val",
            exist_ok=True,
        )
        try:
            exp.log_metric("val.map50", float(metrics.box.map50))
            exp.log_metric("val.map50_95", float(metrics.box.map))
            exp.log_metric("val.precision", float(metrics.box.mp))
            exp.log_metric("val.recall", float(metrics.box.mr))
        except Exception as e:
            exp.logger.warning(f"Could not log val metrics: {e}")

    exp.logger.info(f"Phase 1c fine-tuning completed successfully. Run: {exp.run_id}")

if __name__ == "__main__":
    main()

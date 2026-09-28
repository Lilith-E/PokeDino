"""Training protocol for the oriented YOLO11s-OBB model.

Trains oriented bounding box detector for robust card localization
across arbitrary rotation angles (0-360 deg), archiving checkpoints, loss curves
and confusion matrices in the experiment directory.
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.experiment import Experiment, load_config, set_seed, get_device  # noqa: E402

logger = logging.getLogger(__name__)

def main():
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("phase1b_train_obb", cfg)

    from ultralytics import YOLO

    dataset_yaml = PROJECT_ROOT / cfg["paths"]["datasets_dir"] / cfg["obb"]["dataset"]["name"] / "pokemon_obb.yaml"
    if not dataset_yaml.exists():
        exp.logger.error(f"Dataset YAML missing: {dataset_yaml}. Run scripts/prepare_dataset.py first.")
        sys.exit(1)

    device = get_device(cfg)
    tcfg = cfg["obb"]["train"]
    model_name = cfg["obb"]["model"]["weights"]
    imgsz = cfg["obb"]["model"]["imgsz"]

    exp.log_metric("model", model_name)
    exp.log_metric("imgsz", imgsz)
    exp.log_metric("device", device)
    exp.log_metrics({f"train.{k}": v for k, v in tcfg.items()})

    exp.logger.info(f"Loading pretrained {model_name}…")
    model = YOLO(model_name)

    exp.logger.info("Starting OBB training…")
    results = model.train(
        data=str(dataset_yaml),
        epochs=tcfg["epochs"],
        batch=tcfg["batch"],
        workers=tcfg.get("workers", 2),
        imgsz=imgsz,
        lr0=tcfg["lr0"],
        lrf=tcfg["lrf"],
        optimizer=tcfg["optimizer"],
        cos_lr=tcfg["cos_lr"],
        patience=tcfg["patience"],
        weight_decay=tcfg["weight_decay"],
        warmup_epochs=tcfg["warmup_epochs"],
        mosaic=tcfg["mosaic"],
        close_mosaic=tcfg.get("close_mosaic", 5),
        mixup=tcfg.get("mixup", 0.15),
        erasing=tcfg.get("erasing", 0.20),
        hsv_h=tcfg["hsv_h"],
        hsv_s=tcfg["hsv_s"],
        hsv_v=tcfg["hsv_v"],
        degrees=tcfg["degrees"],
        fliplr=tcfg["fliplr"],
        flipud=tcfg["flipud"],
        scale=tcfg["scale"],
        translate=tcfg["translate"],
        perspective=tcfg["perspective"],
        seed=cfg["project"]["seed"],
        device=device,
        project=str(exp.artifacts_dir / "yolo_runs"),
        name="obb_train",
        exist_ok=True,
        verbose=True,
    )

    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (exp.artifacts_dir / "yolo_runs" / "obb_train")
    best_pt = run_dir / "weights" / "best.pt"

    for fname in ["results.png", "results.csv", "confusion_matrix.png",
                  "confusion_matrix_normalized.png", "PR_curve.png", "F_curve.png",
                  "P_curve.png", "R_curve.png", "args.yaml", "train_batch0.jpg",
                  "train_batch1.jpg", "val_batch0_labels.jpg", "val_batch0_pred.jpg"]:
        src = run_dir / fname
        if src.exists():
            exp.copy_artifact(src, f"train_{fname}")

    model_stem = Path(model_name).stem.replace("-", "_") + "_best.pt"
    if best_pt.exists():
        exp.copy_artifact(best_pt, model_stem)
        exp.logger.info(f"Best weights → {exp.artifact_path(model_stem)}")

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
        exp.log_metric("final.train_loss", float(last.get("train/box_loss", float("nan"))))

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        if "metrics/mAP50(B)" in df.columns:
            axs[0].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50", color="#3B9EFF")
            axs[0].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95", color="#FF4D6A")
            axs[0].set_title("mAP vs epoch"); axs[0].set_xlabel("epoch"); axs[0].legend(); axs[0].grid(alpha=0.3)
        if "train/box_loss" in df.columns:
            axs[1].plot(df["epoch"], df["train/box_loss"], label="box_loss", color="#FFD700")
            if "val/box_loss" in df.columns:
                axs[1].plot(df["epoch"], df["val/box_loss"], label="val_box_loss", color="#00E396")
            axs[1].set_title("Box loss"); axs[1].set_xlabel("epoch"); axs[1].legend(); axs[1].grid(alpha=0.3)
        if "metrics/precision(B)" in df.columns:
            axs[2].plot(df["epoch"], df["metrics/precision(B)"], label="precision", color="#A78BFA")
            axs[2].plot(df["epoch"], df["metrics/recall(B)"], label="recall", color="#FFA500")
            axs[2].set_title("Precision / Recall"); axs[2].set_xlabel("epoch"); axs[2].legend(); axs[2].grid(alpha=0.3)
        fig.suptitle(f"Phase 1b - {model_name} training curves")
        exp.save_plot(fig, "training_curves")

    if best_pt.exists():
        exp.logger.info("Running validation with best weights…")
        best_model = YOLO(str(best_pt))
        metrics = best_model.val(data=str(dataset_yaml), split="val", imgsz=imgsz,
                                 device=device.replace("mps", "cpu") if device == "mps" else device,
                                 project=str(exp.artifacts_dir / "yolo_runs"), name="obb_val", exist_ok=True)
        try:
            exp.log_metric("val.map50", float(metrics.box.map50))
            exp.log_metric("val.map50_95", float(metrics.box.map))
            exp.log_metric("val.precision", float(metrics.box.mp))
            exp.log_metric("val.recall", float(metrics.box.mr))
        except Exception as e:
            exp.logger.warning(f"Could not parse val metrics: {e}")

    exp.finish()
    exp.logger.info("DONE. Phase 1b training complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

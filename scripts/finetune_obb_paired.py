"""Fair paired YOLO-OBB training on the shadow/glare/full-art pokemon_seg scenes.

Links the shared scene images and ``labels_obb/`` exported by
``scripts/prepare_seg_dataset.py`` into ``data/datasets/pokemon_obb_paired`` and
trains YOLO11-OBB under identical data and strong photometric/shadow augmentation
to enable an apples-to-apples benchmark against YOLO11-Seg.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.experiment import Experiment, get_device, load_config, set_seed  # noqa: E402

logger = logging.getLogger(__name__)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_paired_obb_dataset(seg_root: Path, obb_root: Path) -> Path:
    """Materialize a zero-copy (hard-linked) YOLO-OBB dataset from pokemon_seg."""
    for split in ("train", "val", "test"):
        src_imgs = seg_root / "images" / split
        src_lbls = seg_root / "labels_obb" / split
        if not src_imgs.exists() or not src_lbls.exists():
            raise FileNotFoundError(
                f"Missing paired OBB labels in {src_lbls}. Run scripts/prepare_seg_dataset.py first."
            )
        dst_imgs = obb_root / "images" / split
        dst_lbls = obb_root / "labels" / split
        for f_dir in (dst_imgs, dst_lbls):
            f_dir.mkdir(parents=True, exist_ok=True)
            for old in f_dir.glob("*"):
                if old.is_file():
                    old.unlink()

        for img_p in sorted(src_imgs.glob("*.jpg")):
            _link_or_copy(img_p, dst_imgs / img_p.name)
            lbl_p = src_lbls / f"{img_p.stem}.txt"
            if lbl_p.exists():
                _link_or_copy(lbl_p, dst_lbls / lbl_p.name)

    yaml_path = obb_root / "pokemon_obb_paired.yaml"
    yaml_path.write_text(
        f"path: {obb_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n  0: card\n"
    )
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO11-OBB on paired shadow-augmented scenes.")
    parser.add_argument("--base-model", type=str, default="data/weights/yolo11s_obb_finetuned_best.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("finetune_obb_paired", cfg)

    seg_root = PROJECT_ROOT / cfg["paths"]["datasets_dir"] / cfg["segmentation"]["dataset"]["name"]
    obb_root = PROJECT_ROOT / cfg["paths"]["datasets_dir"] / "pokemon_obb_paired"
    yaml_path = build_paired_obb_dataset(seg_root, obb_root)

    base_ref = Path(args.base_model)
    if not base_ref.is_absolute() and (PROJECT_ROOT / base_ref).exists():
        base_ref_str = str(PROJECT_ROOT / base_ref)
    else:
        base_ref_str = args.base_model

    device = get_device(cfg)
    hparams = {
        "model": base_ref_str,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "optimizer": "AdamW",
        "lr0": 0.0008,
        "lrf": 0.01,
        "cos_lr": True,
        "patience": 12,
        "box": 14.0,
        "dfl": 3.0,
        "angle": 2.5,
        "mosaic": 0.35,
        "close_mosaic": 8,
        "degrees": 180,
        "hsv_h": 0.02,
        "hsv_s": 0.35,
        "hsv_v": 0.45,
        "seed": cfg["project"]["seed"],
        "device": str(device),
    }
    (exp.artifacts_dir / "hyperparameters.json").write_text(json.dumps(hparams, indent=2))
    exp.log_metric("model", base_ref_str)
    exp.log_metric("epochs", args.epochs)
    exp.log_metric("imgsz", args.imgsz)
    exp.log_metric("device", device)

    from ultralytics import YOLO

    model = YOLO(base_ref_str)
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        imgsz=args.imgsz,
        lr0=0.0008,
        lrf=0.01,
        optimizer="AdamW",
        cos_lr=True,
        patience=12,
        weight_decay=0.0005,
        warmup_epochs=2.5,
        box=14.0,
        dfl=3.0,
        angle=2.5,
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
        name=f"obb_paired_{args.imgsz}",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (exp.artifacts_dir / "yolo_runs" / f"obb_paired_{args.imgsz}")
    for png_file in sorted(run_dir.glob("*.png")):
        shutil.copy2(png_file, exp.plots_dir / png_file.name)
    for data_file in sorted(list(run_dir.glob("*.csv")) + list(run_dir.glob("*.yaml"))):
        shutil.copy2(data_file, exp.artifacts_dir / f"yolo_{data_file.name}")

    best_pt = run_dir / "weights" / "best.pt"
    default_out = "yolo11m_obb_paired_best.pt" if "yolo11m" in args.base_model.lower() else "yolo11s_obb_paired_best.pt"
    out_weights = PROJECT_ROOT / "data" / "weights" / (args.out_name or default_out)
    if best_pt.exists():
        out_weights.parent.mkdir(parents=True, exist_ok=True)
        out_weights.write_bytes(best_pt.read_bytes())
        exp.copy_artifact(best_pt, out_weights.name)
        exp.logger.info("Saved paired OBB weights -> %s", out_weights)
    exp.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

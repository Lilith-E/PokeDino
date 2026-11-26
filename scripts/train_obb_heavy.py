"""Heavy two-phase YOLO11l-OBB training on the unified 18,168-scene dataset.

Combines:
  - ``data/datasets/pokemon_obb`` (14,168 clean/damaged/multi-card/HN scenes)
  - ``data/datasets/pokemon_seg`` (4,000 shadow/glare/occlusion/sleeve/full-art scenes via labels_obb)
into a card-disjoint ``pokemon_obb_unified`` workspace using zero-copy hard links
(or file copies when across filesystems), then runs:
  - Phase 1: Full adaptation (default 80 epochs, patience=20, mosaic=0.30, close_mosaic=12)
  - Phase 2: Edge-focused sub-pixel corner fine-tuning (default 40 epochs, patience=15,
             mosaic=0.0, mixup=0.0, box=18.0, dfl=4.0, angle=3.0, cls=0.15)
Saves final weights to ``data/weights/yolo11l_obb_heavy_best.pt`` and archives
all plots, CSV curves, hyperparameters, and held-out test metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.experiment import Experiment, get_device, load_config, set_seed  # noqa: E402
from scripts.finetune_obb_paired import _link_or_copy  # noqa: E402

logger = logging.getLogger(__name__)


def _collect_split_pairs(img_dir: Path, lbl_dir: Path) -> list[tuple[Path, Optional[Path]]]:
    pairs: list[tuple[Path, Path | None]] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for img_p in sorted(img_dir.glob(ext)):
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            pairs.append((img_p, lbl_p if lbl_p.exists() else None))
    return pairs


def build_unified_obb_dataset(obb_dir: Path, seg_dir: Path, unified_dir: Path) -> tuple[Path, dict[str, int]]:
    """Materialize unified OBB dataset from pokemon_obb + pokemon_seg/labels_obb."""
    counts: dict[str, int] = {}
    sources: list[tuple[str, Path, Path]] = []
    if (obb_dir / "images" / "train").exists():
        sources.append(("obb", obb_dir / "images", obb_dir / "labels"))
    if (seg_dir / "images" / "train").exists() and (seg_dir / "labels_obb" / "train").exists():
        sources.append(("seg", seg_dir / "images", seg_dir / "labels_obb"))
    if not sources:
        raise FileNotFoundError(
            f"Neither {obb_dir} nor {seg_dir}/labels_obb was found."
        )

    for split in ("train", "val", "test"):
        dst_imgs = unified_dir / "images" / split
        dst_lbls = unified_dir / "labels" / split
        for d in (dst_imgs, dst_lbls):
            d.mkdir(parents=True, exist_ok=True)
            for old in d.glob("*"):
                if old.is_file():
                    old.unlink()

        split_count = 0
        for prefix, src_img_root, src_lbl_root in sources:
            img_s = src_img_root / split
            lbl_s = src_lbl_root / split
            if not img_s.exists():
                continue
            for img_p, lbl_p in _collect_split_pairs(img_s, lbl_s):
                out_name = f"{prefix}_{img_p.name}"
                _link_or_copy(img_p, dst_imgs / out_name)
                if lbl_p is not None and lbl_p.exists():
                    _link_or_copy(lbl_p, dst_lbls / f"{prefix}_{img_p.stem}.txt")
                split_count += 1
        counts[split] = split_count

    yaml_path = unified_dir / "pokemon_obb_unified.yaml"
    yaml_path.write_text(
        f"path: {unified_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n  0: card\n",
        encoding="utf-8",
    )
    return yaml_path, counts


def _archive_run_outputs(run_dir: Path, exp: Experiment, prefix: str) -> None:
    for png_file in sorted(run_dir.glob("*.png")):
        shutil.copy2(png_file, exp.plots_dir / f"{prefix}_{png_file.name}")
    for data_file in sorted(list(run_dir.glob("*.csv")) + list(run_dir.glob("*.yaml"))):
        shutil.copy2(data_file, exp.artifacts_dir / f"{prefix}_{data_file.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-phase heavy YOLO11l-OBB fine-tuning on unified 18k+ scenes.")
    parser.add_argument("--base-model", type=str, default="yolo11l-obb.pt")
    parser.add_argument("--epochs-p1", type=int, default=80)
    parser.add_argument("--patience-p1", type=int, default=20)
    parser.add_argument("--epochs-p2", type=int, default=40)
    parser.add_argument("--patience-p2", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8, help="Batch=8 is VRAM-safe for YOLO11l-OBB on RTX 3060 12GB.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Build unified dataset YAML and exit without training.")
    parser.add_argument("--out-name", type=str, default="yolo11l_obb_heavy_best.pt")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("train_obb_heavy", cfg)

    ds_root = PROJECT_ROOT / cfg["paths"]["datasets_dir"]
    obb_dir = ds_root / cfg["obb"]["dataset"]["name"]
    seg_dir = ds_root / cfg["segmentation"]["dataset"]["name"]
    unified_dir = ds_root / "pokemon_obb_unified"
    yaml_path, split_counts = build_unified_obb_dataset(obb_dir, seg_dir, unified_dir)
    logger.info("Unified OBB dataset ready (%s) -> splits=%s", yaml_path, split_counts)
    if args.prepare_only:
        exp.log_metric("unified_train_scenes", split_counts["train"])
        exp.log_metric("unified_val_scenes", split_counts["val"])
        exp.log_metric("unified_test_scenes", split_counts["test"])
        exp.finish()
        return

    base_ref = Path(args.base_model)
    base_ref_str = (
        str(PROJECT_ROOT / base_ref)
        if not base_ref.is_absolute() and (PROJECT_ROOT / base_ref).exists()
        else args.base_model
    )
    device = get_device(cfg)

    hparams = {
        "base_model": base_ref_str,
        "unified_splits": split_counts,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": str(device),
        "phase1": {
            "epochs": args.epochs_p1,
            "patience": args.patience_p1,
            "optimizer": "AdamW",
            "lr0": 0.0008,
            "lrf": 0.01,
            "cos_lr": True,
            "box": 12.0,
            "dfl": 2.5,
            "angle": 2.5,
            "cls": 0.35,
            "mosaic": 0.30,
            "close_mosaic": min(12, max(0, args.epochs_p1 // 4)),
        },
        "phase2": {
            "enabled": not args.skip_phase2,
            "epochs": args.epochs_p2,
            "patience": args.patience_p2,
            "optimizer": "AdamW",
            "lr0": 0.00025,
            "lrf": 0.01,
            "cos_lr": True,
            "box": 18.0,
            "dfl": 4.0,
            "angle": 3.0,
            "cls": 0.15,
            "mosaic": 0.0,
            "mixup": 0.0,
            "erasing": 0.0,
        },
        "seed": cfg["project"]["seed"],
    }
    (exp.artifacts_dir / "hyperparameters.json").write_text(json.dumps(hparams, indent=2))
    exp.log_metric("unified_train_scenes", split_counts["train"])
    exp.log_metric("unified_val_scenes", split_counts["val"])
    exp.log_metric("unified_test_scenes", split_counts["test"])

    from ultralytics import YOLO

    # Phase 1: Full-scale localization + shadow/glare/occlusion robustness.
    model = YOLO(base_ref_str)
    p1_res = model.train(
        data=str(yaml_path),
        epochs=args.epochs_p1,
        patience=args.patience_p1,
        batch=args.batch,
        workers=args.workers,
        imgsz=args.imgsz,
        optimizer="AdamW",
        lr0=0.0008,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        box=12.0,
        dfl=2.5,
        angle=2.5,
        cls=0.35,
        mosaic=0.30,
        close_mosaic=hparams["phase1"]["close_mosaic"],
        mixup=0.0,
        erasing=0.05,
        degrees=180,
        fliplr=0.5,
        flipud=0.5,
        scale=0.45,
        translate=0.10,
        perspective=0.0006,
        hsv_h=0.02,
        hsv_s=0.40,
        hsv_v=0.45,
        seed=cfg["project"]["seed"],
        device=device,
        project=str(exp.artifacts_dir / "yolo_runs"),
        name=f"p1_{args.imgsz}",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    p1_dir = Path(p1_res.save_dir) if hasattr(p1_res, "save_dir") else (exp.artifacts_dir / "yolo_runs" / f"p1_{args.imgsz}")
    _archive_run_outputs(p1_dir, exp, "phase1")

    p1_best = p1_dir / "weights" / "best.pt"
    p1_out = PROJECT_ROOT / "data" / "weights" / "yolo11l_obb_phase1_best.pt"
    if p1_best.exists():
        p1_out.parent.mkdir(parents=True, exist_ok=True)
        p1_out.write_bytes(p1_best.read_bytes())
        exp.copy_artifact(p1_best, p1_out.name)

    final_best = p1_best
    if not args.skip_phase2 and p1_best.exists():
        # Phase 2: Pure unsevered boundary corner refinement without mosaic cuts.
        model_p2 = YOLO(str(p1_best))
        p2_res = model_p2.train(
            data=str(yaml_path),
            epochs=args.epochs_p2,
            patience=args.patience_p2,
            batch=args.batch,
            workers=args.workers,
            imgsz=args.imgsz,
            optimizer="AdamW",
            lr0=0.00025,
            lrf=0.01,
            cos_lr=True,
            weight_decay=0.0005,
            warmup_epochs=1.0,
            box=18.0,
            dfl=4.0,
            angle=3.0,
            cls=0.15,
            mosaic=0.0,
            close_mosaic=0,
            mixup=0.0,
            erasing=0.0,
            degrees=180,
            fliplr=0.5,
            flipud=0.5,
            scale=0.30,
            translate=0.06,
            perspective=0.0004,
            hsv_h=0.015,
            hsv_s=0.35,
            hsv_v=0.35,
            seed=cfg["project"]["seed"],
            device=device,
            project=str(exp.artifacts_dir / "yolo_runs"),
            name=f"p2_edge_{args.imgsz}",
            exist_ok=True,
            plots=True,
            verbose=True,
        )
        p2_dir = Path(p2_res.save_dir) if hasattr(p2_res, "save_dir") else (exp.artifacts_dir / "yolo_runs" / f"p2_edge_{args.imgsz}")
        _archive_run_outputs(p2_dir, exp, "phase2")
        if (p2_dir / "weights" / "best.pt").exists():
            final_best = p2_dir / "weights" / "best.pt"

    out_weights = PROJECT_ROOT / "data" / "weights" / args.out_name
    if final_best.exists():
        out_weights.parent.mkdir(parents=True, exist_ok=True)
        out_weights.write_bytes(final_best.read_bytes())
        exp.copy_artifact(final_best, out_weights.name)
        logger.info("Saved final heavy OBB weights -> %s", out_weights)

        eval_model = YOLO(str(out_weights))
        test_res = eval_model.val(data=str(yaml_path), split="test", imgsz=args.imgsz, batch=args.batch, device=device, verbose=False)
        if hasattr(test_res, "results_dict"):
            for k, v in test_res.results_dict.items():
                clean_k = str(k).replace("(", "").replace(")", "").replace("/", "_")
                try:
                    exp.log_metric(f"test.{clean_k}", float(v))
                except (TypeError, ValueError):
                    pass

    exp.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    main()

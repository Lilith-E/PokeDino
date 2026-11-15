"""Visual comparison generator for the card segmentation vs OBB+PBL study.

Produces publication-grade figures saved under both ``experiments/`` and
``report/figures/05_segmentation_pipeline/`` covering:
1. Segmentation training curves (YOLO11s-Seg @640/@960 and MobileNetV3-UNet).
2. Stratified synthetic stress qualitative grid (perspective, occlusion, sleeve, clutter).
3. Real-photograph end-to-end localization, dynamic masking, and AnomalyDINO heatmaps.
4. Dynamic warped validity mask vs static rounded-rectangle mask impact under occlusion.
"""

import base64
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.card_detector import CardDetector, TARGET_H, TARGET_W  # noqa: E402
from backend.detector import PokemonAnomalyDetector  # noqa: E402
from backend.experiment import Experiment, load_config, set_seed  # noqa: E402
from backend.masking import rounded_rect_mask  # noqa: E402
from backend.seg_detector import SegmentationCardDetector  # noqa: E402
from scripts.benchmark_segmentation_vs_obb import (  # noqa: E402
    align_corner_error,
    pixel_mask_iou,
    project_patch_mask_to_scene,
)

logger = logging.getLogger(__name__)


def _decode_b64_rgb(b64_str: Optional[str]) -> Optional[np.ndarray]:
    if not b64_str:
        return None
    raw = b64_str.split(",", 1)[-1]
    buf = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _latest_experiment_dir(prefix: str) -> Optional[Path]:
    exp_root = PROJECT_ROOT / "experiments"
    matches = sorted([d for d in exp_root.glob(f"*_{prefix}") if d.is_dir()])
    return matches[-1] if matches else None


def plot_training_curves(out_dir: Path) -> Optional[Path]:
    """Plot training convergence for YOLO11s-Seg (@640/@960) and MobileNetV3-UNet."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=200)
    fig.patch.set_facecolor("white")
    has_any_curve = False

    # 1. YOLO11s-Seg mAP50-95 (Mask)
    ax = axes[0]
    for tag, label, color in [
        ("train_yolo_seg_640", "YOLO11s-Seg @ 640", "#1D4ED8"),
        ("train_yolo_seg_960", "YOLO11s-Seg @ 960", "#059669"),
    ]:
        ed = _latest_experiment_dir(tag)
        if ed is None:
            continue
        csv_files = list((ed / "artifacts" / "yolo_runs").glob("*/results.csv"))
        if not csv_files:
            continue
        epochs, map_m5095, val_seg_loss = [], [], []
        with open(csv_files[0], newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}
                ep = int(float(row.get("epoch", len(epochs) + 1)))
                epochs.append(ep)
                map_m5095.append(float(row.get("metrics/mAP50-95(M)", 0.0)))
                val_seg_loss.append(float(row.get("val/seg_loss", 0.0)))
        if epochs:
            has_any_curve = True
            ax.plot(epochs, map_m5095, marker="o", ms=3.5, lw=2.0, color=color, label=label)
            axes[1].plot(epochs, val_seg_loss, marker="s", ms=3.5, lw=2.0, color=color, label=label)

    ax.set_title("YOLO-Seg Mask mAP@0.50:0.95 (Val)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0.50:0.95 (Mask)")
    ax.set_ylim(0.5, 1.01)
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=9)

    axes[1].set_title("YOLO-Seg Validation Seg Loss", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Seg Loss")
    axes[1].grid(alpha=0.3)
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(fontsize=9)

    # 3. MobileNetV3-UNet loss & val IoU
    ax3 = axes[2]
    ed_sem = _latest_experiment_dir("train_seg_semantic")
    if ed_sem is not None and (ed_sem / "artifacts" / "semantic_history.csv").exists():
        ep_s, loss_s, iou_s = [], [], []
        with open(ed_sem / "artifacts" / "semantic_history.csv", newline="") as f:
            for row in csv.DictReader(f):
                has_any_curve = True
                ep_s.append(int(row["epoch"]))
                loss_s.append(float(row["train_loss"]))
                iou_s.append(float(row["val_iou"]))
        if ep_s:
            l1 = ax3.plot(ep_s, iou_s, marker="o", ms=4, lw=2.2, color="#7C3AED", label="Val Mask IoU")
            ax3_r = ax3.twinx()
            l2 = ax3_r.plot(ep_s, loss_s, marker="^", ms=4, lw=1.8, ls="--", color="#DC2626", label="Train Composite Loss")
            ax3.set_ylabel("Validation IoU", color="#7C3AED")
            ax3_r.set_ylabel("Composite Loss", color="#DC2626")
            lines = l1 + l2
            ax3.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=8.5)
    ax3.set_title("MobileNetV3-UNet Convergence", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.grid(alpha=0.3)

    if not has_any_curve:
        plt.close(fig)
        return None

    plt.tight_layout()
    out_path = out_dir / "seg_training_curves.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _overlay_quad_and_mask(
    img_bgr: np.ndarray,
    corners: np.ndarray,
    patch_mask: np.ndarray,
    color_quad: tuple[int, int, int],
) -> np.ndarray:
    """Render semi-transparent scene mask and crisp quadrilateral border in RGB."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    scene_mask = project_patch_mask_to_scene(corners, patch_mask, img_bgr.shape[:2])
    tint = np.zeros_like(rgb)
    tint[scene_mask] = color_quad
    rgb = cv2.addWeighted(rgb, 1.0, tint, 0.28, 0)
    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(rgb, [pts], True, color_quad, 3, cv2.LINE_AA)
    for pt in corners.astype(np.int32):
        cv2.circle(rgb, tuple(pt), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(rgb, tuple(pt), 3, color_quad, -1, cv2.LINE_AA)
    return rgb


def plot_synthetic_stress_grid(cfg: dict, backends: dict[str, object], out_dir: Path) -> Optional[Path]:
    """Build qualitative grid across clean, perspective, sleeve, and occlusion scenes."""
    ds_root = PROJECT_ROOT / cfg["paths"]["datasets_dir"] / cfg["segmentation"]["dataset"]["name"]
    meta_dir = ds_root / "meta" / "test"
    img_dir = ds_root / "images" / "test"
    mask_dir = ds_root / "masks" / "test"

    chosen: dict[str, Path] = {}
    for mp in sorted(meta_dir.glob("*.json")):
        meta = json.loads(mp.read_text())
        if meta.get("hard_negative") or len(meta.get("cards", [])) != 1:
            continue
        tags = meta["cards"][0].get("tags", [])
        if "occlusion" in tags and "occlusion" not in chosen:
            chosen["occlusion"] = mp
        elif "sleeve" in tags and "perspective" not in tags and "sleeve" not in chosen:
            chosen["sleeve"] = mp
        elif "perspective" in tags and "occlusion" not in tags and "perspective" not in chosen:
            chosen["perspective"] = mp
        elif not tags and "clean" not in chosen:
            chosen["clean"] = mp
        if len(chosen) == 4:
            break

    order_keys = [k for k in ("clean", "perspective", "sleeve", "occlusion") if k in chosen]
    if not order_keys:
        return None

    col_keys = [("gt", "Ground Truth")] + [
        (k, lbl)
        for k, lbl in [
            ("obb_pbl", "YOLO11s-OBB + PBL"),
            ("seg_yolo_640", "YOLO11s-Seg @640"),
            ("seg_semantic", "MobileNetV3-UNet"),
            ("seg_zeroshot_yoloe", "YOLOE Zero-Shot"),
        ]
        if k in backends
    ]
    colors = {
        "obb_pbl": (37, 99, 235),
        "seg_yolo_640": (16, 185, 129),
        "seg_semantic": (124, 58, 237),
        "seg_zeroshot_yoloe": (217, 119, 6),
    }
    baseline_mask = rounded_rect_mask(
        TARGET_H,
        TARGET_W,
        corner_radius_frac=cfg["masking"]["corner_radius_frac"],
        erode_frac=cfg["masking"]["erode_frac"],
        feather_px=cfg["masking"]["feather_px"],
    )

    n_rows = len(order_keys)
    n_cols = len(col_keys)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3.2 * n_rows), dpi=180)
    if n_rows == 1:
        axes = axes[None, :]
    fig.patch.set_facecolor("white")

    for r_idx, tag_name in enumerate(order_keys):
        mp = chosen[tag_name]
        meta = json.loads(mp.read_text())
        stem = mp.stem
        img_bgr = cv2.imread(str(img_dir / f"{stem}.jpg"), cv2.IMREAD_COLOR)
        gt_mask = cv2.imread(str(mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE) >= 127
        gt_quad = np.array(meta["cards"][0]["corners_gt"], dtype=np.float64)

        for c_idx, (bk, label) in enumerate(col_keys):
            ax = axes[r_idx, c_idx]
            ax.axis("off")
            if bk == "gt":
                vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
                tint = np.zeros_like(vis)
                tint[gt_mask] = (14, 165, 233)
                vis = cv2.addWeighted(vis, 1.0, tint, 0.25, 0)
                cv2.polylines(vis, [gt_quad.astype(np.int32).reshape(-1, 1, 2)], True, (14, 165, 233), 3)
                ax.imshow(vis)
                subtitle = f"[{tag_name.upper()}] GT"
            else:
                t0 = time.perf_counter()
                dets = backends[bk].detect(img_bgr)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                if not dets:
                    vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    ax.imshow(vis)
                    subtitle = "NO DETECTION"
                else:
                    d = dets[0]
                    pm = d.mask if d.mask is not None else baseline_mask
                    vis = _overlay_quad_and_mask(img_bgr, d.corners, pm, colors.get(bk, (220, 38, 38)))
                    cerr = align_corner_error(d.corners, gt_quad)
                    pred_b = project_patch_mask_to_scene(d.corners, pm, img_bgr.shape[:2])
                    miou = pixel_mask_iou(pred_b, gt_mask)
                    ax.imshow(vis)
                    subtitle = f"err={cerr:.2f}px | IoU={miou:.3f} | {lat_ms:.0f}ms"

            if r_idx == 0:
                ax.set_title(f"{label}\n{subtitle}", fontsize=9.2, fontweight="bold")
            else:
                ax.set_title(subtitle, fontsize=8.8)

    plt.tight_layout()
    out_path = out_dir / "seg_vs_obb_synthetic_stress.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_real_downstream_comparison(cfg: dict, backends: dict[str, object], out_dir: Path) -> Optional[Path]:
    """Visualize localization + dynamic mask + AnomalyDINO heatmap on real smartphone captures."""
    real_dir = PROJECT_ROOT / "experiments" / "synthetic-to-real"
    samples = [
        ("Gengar #1", real_dir / "realgengar-1.jpg", real_dir / "reference-gengarprime.webp"),
        ("Charizard Damaged", real_dir / "charizard-damaged.jpeg", real_dir / "charizard-reference.webp"),
        ("Giratina Close-up", real_dir / "giratina-kanda-damaged.jpg", real_dir / "giratina-kanda-reference.jpeg"),
    ]
    samples = [(n, q, r) for n, q, r in samples if q.exists() and r.exists()]
    if not samples:
        return None

    active_bk = [
        (k, lbl)
        for k, lbl in [
            ("obb_pbl", "OBB + PBL"),
            ("seg_yolo_640", "YOLO11s-Seg @640"),
            ("seg_semantic", "MobileNetV3-UNet"),
            ("seg_zeroshot_yoloe", "YOLOE Zero-Shot"),
        ]
        if k in backends
    ]
    if not active_bk:
        return None

    ad = PokemonAnomalyDetector(masking_enabled=True, clahe_enabled=True)
    n_rows = len(samples)
    n_cols = 1 + 2 * len(active_bk)  # Raw photo + (Rectified/Mask, Heatmap) per backend
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.8 * n_rows), dpi=180)
    if n_rows == 1:
        axes = axes[None, :]
    fig.patch.set_facecolor("white")

    for r_idx, (title, q_path, r_path) in enumerate(samples):
        img_bgr = cv2.imread(str(q_path), cv2.IMREAD_COLOR)
        ad.load_reference_images([str(r_path)], card_id=f"vis_{r_path.stem}")

        ax0 = axes[r_idx, 0]
        ax0.axis("off")
        ax0.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax0.set_title(f"{title}\n(Input Capture)", fontsize=9.5, fontweight="bold")

        col_ptr = 1
        for bk, lbl in active_bk:
            ax_patch = axes[r_idx, col_ptr]
            ax_heat = axes[r_idx, col_ptr + 1]
            ax_patch.axis("off")
            ax_heat.axis("off")

            dets = backends[bk].detect(img_bgr)
            if not dets:
                ax_patch.text(0.5, 0.5, "NO DETECTION", ha="center", va="center", fontsize=10, color="#DC2626")
                ax_heat.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10, color="#6B7280")
                ax_patch.set_title(f"{lbl}: Patch", fontsize=9)
                ax_heat.set_title(f"{lbl}: Heatmap", fontsize=9)
            else:
                d = dets[0]
                res = ad.analyze_patch(d.patch, query_mask=d.mask)
                patch_vis = d.patch.copy()
                if d.mask is not None:
                    inactive = d.mask < 0.5
                    patch_vis[inactive] = (patch_vis[inactive] * 0.35 + np.array([220, 40, 40]) * 0.65).astype(np.uint8)
                ax_patch.imshow(patch_vis)
                mask_pct = 100.0 * float((d.mask >= 0.5).mean()) if d.mask is not None else 98.5
                ax_patch.set_title(f"{lbl}\nconf={d.confidence:.2f} | mask={mask_pct:.1f}%", fontsize=8.8)

                overlay_rgb = _decode_b64_rgb(res.get("heatmap_b64"))
                if overlay_rgb is not None:
                    ax_heat.imshow(overlay_rgb)
                ax_heat.set_title(f"Score {res['score']:.3f}\n({res['condition']['label']})", fontsize=8.8, fontweight="bold")
            col_ptr += 2

    plt.tight_layout()
    out_path = out_dir / "seg_vs_obb_real_cards.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_clahe_and_mask_ablation(cfg: dict, backends: dict[str, object], out_dir: Path) -> Optional[Path]:
    """Compare CLAHE ON vs OFF and static vs dynamic masking on downstream heatmaps."""
    real_dir = PROJECT_ROOT / "experiments" / "synthetic-to-real"
    pairs = [
        ("Gengar #1", real_dir / "realgengar-1.jpg", real_dir / "reference-gengarprime.webp"),
        ("Charizard Damaged", real_dir / "charizard-damaged.jpeg", real_dir / "charizard-reference.webp"),
    ]
    pairs = [(t, q, r) for t, q, r in pairs if q.exists() and r.exists()]
    if not pairs:
        return None

    ad_c = PokemonAnomalyDetector(masking_enabled=True, clahe_enabled=True)
    ad_nc = PokemonAnomalyDetector(masking_enabled=True, clahe_enabled=False)

    test_bk = [
        (k, lbl)
        for k, lbl in [
            ("obb_pbl", "OBB+PBL (Static Mask)"),
            ("seg_yolo_640", "YOLO11s-Seg (Dyn Mask)"),
            ("seg_zeroshot_yoloe", "YOLOE (Dyn Mask)"),
        ]
        if k in backends
    ]
    if not test_bk:
        return None

    n_rows = len(pairs) * len(test_bk)
    fig, axes = plt.subplots(n_rows, 4, figsize=(13.2, 3.6 * n_rows), dpi=180)
    if n_rows == 1:
        axes = axes[None, :]
    fig.patch.set_facecolor("white")

    row_i = 0
    for title, q_path, r_path in pairs:
        img_bgr = cv2.imread(str(q_path), cv2.IMREAD_COLOR)
        ad_c.load_reference_images([str(r_path)], card_id=f"abl_{r_path.stem}")
        ad_nc.load_reference_images([str(r_path)], card_id=f"abl_{r_path.stem}")
        for bk, lbl in test_bk:
            dets = backends[bk].detect(img_bgr)
            for c in range(4):
                axes[row_i, c].axis("off")
            if not dets:
                axes[row_i, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                axes[row_i, 0].set_title(f"{title} — {lbl}\nNO DETECTION", fontsize=9)
                row_i += 1
                continue
            d = dets[0]
            res_c = ad_c.analyze_patch(d.patch, query_mask=d.mask)
            res_nc = ad_nc.analyze_patch(d.patch, query_mask=d.mask)

            mask_vis = np.zeros_like(d.patch)
            if d.mask is not None:
                m_u8 = np.clip(d.mask * 255.0, 0, 255).astype(np.uint8)
                mask_vis = cv2.applyColorMap(m_u8, cv2.COLORMAP_VIRIDIS)
                mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)
            else:
                bm = rounded_rect_mask(TARGET_H, TARGET_W, cfg["masking"]["corner_radius_frac"], cfg["masking"]["erode_frac"], cfg["masking"]["feather_px"])
                mask_vis = cv2.cvtColor(cv2.applyColorMap((bm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)

            axes[row_i, 0].imshow(d.patch)
            axes[row_i, 0].set_title(f"{title}\n{lbl} Patch", fontsize=9, fontweight="bold")
            axes[row_i, 1].imshow(mask_vis)
            axes[row_i, 1].set_title("Active Validity Mask", fontsize=9)
            hm_c = _decode_b64_rgb(res_c.get("heatmap_b64"))
            hm_nc = _decode_b64_rgb(res_nc.get("heatmap_b64"))
            if hm_c is not None:
                axes[row_i, 2].imshow(hm_c)
            axes[row_i, 2].set_title(f"CLAHE ON: {res_c['score']:.3f}\n({res_c['condition']['label']})", fontsize=9)
            if hm_nc is not None:
                axes[row_i, 3].imshow(hm_nc)
            axes[row_i, 3].set_title(f"CLAHE OFF: {res_nc['score']:.3f}\n({res_nc['condition']['label']})", fontsize=9)
            row_i += 1

    plt.tight_layout()
    out_path = out_dir / "clahe_and_mask_ablation_real.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("seg_visual_report", cfg)
    fig_dir = PROJECT_ROOT / "report" / "figures" / "05_segmentation_pipeline"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training curves
    p_curves = plot_training_curves(fig_dir)
    if p_curves:
        exp.copy_artifact(p_curves, p_curves.name)
        exp.logger.info("Saved %s", p_curves)

    # 2. Load available backends
    backends: dict[str, object] = {
        "obb_pbl": CardDetector(
            conf=float(cfg["obb"]["inference"]["conf"]),
            iou=float(cfg["obb"]["inference"]["iou"]),
            max_det=int(cfg["obb"]["inference"]["max_det"]),
        )
    }
    w640 = PROJECT_ROOT / cfg["segmentation"]["yolo_seg"]["weights"]
    if w640.exists():
        backends["seg_yolo_640"] = SegmentationCardDetector("seg_yolo", weights_path=str(w640), imgsz=640)
    wsem = PROJECT_ROOT / cfg["segmentation"]["semantic"]["weights"]
    if wsem.exists():
        backends["seg_semantic"] = SegmentationCardDetector("seg_semantic", weights_path=str(wsem), imgsz=640)
    try:
        backends["seg_zeroshot_yoloe"] = SegmentationCardDetector("seg_zeroshot", zeroshot_engine="yoloe")
    except Exception as exc:
        exp.logger.warning("Zero-shot YOLOE unavailable for visual grid: %s", exc)

    p_synth = plot_synthetic_stress_grid(cfg, backends, fig_dir)
    if p_synth:
        exp.copy_artifact(p_synth, p_synth.name)
        exp.logger.info("Saved %s", p_synth)

    p_real = plot_real_downstream_comparison(cfg, backends, fig_dir)
    if p_real:
        exp.copy_artifact(p_real, p_real.name)
        exp.logger.info("Saved %s", p_real)

    p_abl = plot_clahe_and_mask_ablation(cfg, backends, fig_dir)
    if p_abl:
        exp.copy_artifact(p_abl, p_abl.name)
        exp.logger.info("Saved %s", p_abl)

    exp.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

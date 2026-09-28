"""End-to-end evaluation of the Baseline pipeline.

The procedure evaluates on a set of test scenes:
1. Stage 1 quality gate (quick reject on raw image).
2. Oriented card detection via YOLO OBB and perspective warping.
3. Stage 2 quality gate (detailed assessment on YOLO crop).
4. Association with official reference scan.
5. Anomaly score computation and qualitative condition grading via AnomalyDINO.

Metrics calculated:
- Detection recall (percentage of scenes with card detected).
- Discard rate for insufficient quality.
- Anomaly score distribution on pristine cards.
- False positive rate (FPR) against calibrated threshold.
"""

import json
import logging
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.card_detector import CardDetector
from backend.detector import PokemonAnomalyDetector
from backend.experiment import Experiment, load_config, set_seed
from backend.quality_gate import ImageQualityGate

logger = logging.getLogger(__name__)

def find_best_reference(
    patch_rgb: np.ndarray,
    ref_paths: list[Path],
    detector: PokemonAnomalyDetector,
    ref_feat_cache: dict | None = None,
) -> tuple[Path, float]:
    """Individua la scansione di riferimento corrispondente mediante similarità del coseno su descrittori DINOv2."""
    detector.load_model()
    target_h, target_w = 630, 448
    p_resized = cv2.resize(patch_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    with torch.inference_mode():
        p_tensor, _ = detector.model.prepare_image(p_resized)
        p_feats = detector.model.extract_features(p_tensor)
        p_feat = torch.from_numpy(np.asarray(p_feats, dtype=np.float32).mean(axis=0, keepdims=True))
        p_feat = torch.nn.functional.normalize(p_feat, p=2, dim=1)

        p180 = np.ascontiguousarray(np.rot90(p_resized, 2))
        p180_tensor, _ = detector.model.prepare_image(p180)
        p180_feats = detector.model.extract_features(p180_tensor)
        p180_feat = torch.from_numpy(np.asarray(p180_feats, dtype=np.float32).mean(axis=0, keepdims=True))
        p180_feat = torch.nn.functional.normalize(p180_feat, p=2, dim=1)

        best_path = ref_paths[0]
        max_sim = -1.0

        for r_path in ref_paths:
            r_feat = None
            if ref_feat_cache is not None and str(r_path) in ref_feat_cache:
                r_feat = ref_feat_cache[str(r_path)]
            else:
                r_img = cv2.imread(str(r_path), cv2.IMREAD_COLOR)
                if r_img is None:
                    continue
                r_rgb = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
                r_resized = cv2.resize(r_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                r_tensor, _ = detector.model.prepare_image(r_resized)
                r_feats = detector.model.extract_features(r_tensor)
                r_feat = torch.from_numpy(np.asarray(r_feats, dtype=np.float32).mean(axis=0, keepdims=True))
                r_feat = torch.nn.functional.normalize(r_feat, p=2, dim=1)
                if ref_feat_cache is not None:
                    ref_feat_cache[str(r_path)] = r_feat

            sim = max(float(torch.mm(p_feat, r_feat.T).item()),
                      float(torch.mm(p180_feat, r_feat.T).item()))
            if sim > max_sim:
                max_sim = sim
                best_path = r_path

    return best_path, max_sim

def main():
    """Entry point for end-to-end evaluation of the Baseline pipeline."""
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("e2e_pipeline", cfg)

    n_scenes = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    weights = CardDetector._find_latest_weights()
    if weights is None:
        exp.logger.error("OBB model weights not found. Run train_obb.py first.")
        sys.exit(1)
    cd = CardDetector(weights_path=weights)

    det = PokemonAnomalyDetector(masking_enabled=True, clahe_enabled=True)

    data_root = PROJECT_ROOT / "data" / "datasets" / "pokemon_obb"
    labels_dir = data_root / "labels" / "test"
    all_scenes = sorted((data_root / "images" / "test").glob("*.jpg"))
    scenes = []
    for sp in all_scenes:
        lbl = labels_dir / (sp.stem + ".txt")
        if lbl.exists() and len(lbl.read_text().strip().splitlines()) == 1:
            scenes.append(sp)
        if len(scenes) >= n_scenes:
            break

    exp.log_metric("multi_card_scenes_skipped", len(all_scenes) - len(scenes))
    if not scenes:
        exp.logger.error("Nessuna scena di test a carta singola disponibile.")
        sys.exit(1)

    ref_pool_dir = (
        PROJECT_ROOT / "data" / "datasets" / "cards" / "pokemon_official_tcg" / "test"
    )
    ref_paths = sorted(ref_pool_dir.glob("*.webp")) if ref_pool_dir.exists() else []
    if not ref_paths:
        raw_cards_dir = (
            PROJECT_ROOT
            / "data"
            / "datasets"
            / "cards"
            / "pokemon_pocket_tcgp"
        )
        ref_paths = sorted(raw_cards_dir.glob("*.webp")) if raw_cards_dir.exists() else []
    if not ref_paths:
        ref_paths = sorted((PROJECT_ROOT / "data" / "reference_cache").glob("*.png"))

    n_detected = 0
    n_discarded = 0
    quality_gate = ImageQualityGate()
    scores = []
    per_scene = []
    sample_data = []
    ref_feat_cache: dict = {}

    for i, scene_path in enumerate(scenes):
        img = cv2.imread(str(scene_path))
        dets = cd.detect(img)
        if not dets:
            per_scene.append({"scene": scene_path.stem, "detected": 0})
            continue
        n_detected += 1

        d = dets[0]
        
        h_frame, w_frame = img.shape[:2]
        box_area = cv2.contourArea(d.corners.astype(np.int32))
        box_area_ratio = box_area / (h_frame * w_frame)
        
        sides = []
        for j in range(4):
            p1 = d.corners[j]
            p2 = d.corners[(j + 1) % 4]
            sides.append(np.linalg.norm(p2 - p1))
        sides.sort()
        box_aspect_ratio = sides[0] / sides[3] if sides[3] > 0 else 1.0
        
        quality_result = quality_gate.evaluate(
            image=d.patch,
            detection_confidence=d.confidence,
            box_aspect_ratio=box_aspect_ratio,
            box_area_ratio=box_area_ratio,
        )
        
        if not quality_result.passed:
            n_discarded += 1
            per_scene.append({
                "scene": scene_path.stem,
                "detected": len(dets),
                "conf": d.confidence,
                "quality": round(quality_result.score, 3),
                "discarded_low_quality": True,
            })
            exp.logger.info("  Scena %d/%d: Scartata per qualità insufficiente (%.2f)", i + 1, len(scenes), quality_result.score)
            continue

        ref_path, sim = find_best_reference(d.patch, ref_paths, det, ref_feat_cache)
        det.load_reference_images([str(ref_path)], card_id=ref_path.stem)
        out = det.analyze_patch(d.patch)
        scores.append(out["score"])

        per_scene.append({
            "scene": scene_path.stem,
            "detected": len(dets),
            "conf": d.confidence,
            "quality": round(quality_result.score, 3),
            "reference": ref_path.stem,
            "similarity": round(sim, 3),
            "flipped": out["flipped"],
            "score": round(out["score"], 4),
            "condition": out["condition"]["label"],
        })

        if len(sample_data) < 4:
            patch_disp = np.rot90(d.patch, 2) if out["flipped"] else d.patch
            sample_data.append((img, patch_disp, out))

        exp.logger.info(
            "  Scena %d/%d: det=%d conf=%.2f ref=%s score=%.4f -> %s",
            i + 1,
            len(scenes),
            len(dets),
            d.confidence,
            ref_path.stem,
            out["score"],
            out["condition"]["label"],
        )

    threshold = det.anomaly_threshold
    scores_arr = np.array(scores) if scores else np.array([0.0])
    metrics = {
        "scenes": len(scenes),
        "detection_recall": n_detected / len(scenes),
        "quality_discard_rate": n_discarded / n_detected if n_detected else 0.0,
        "quality_gate_min_score": 0.35,
        "pristine_scores_mean": float(scores_arr.mean()),
        "pristine_scores_max": float(scores_arr.max()),
        "threshold": threshold,
        "false_positive_rate": float((scores_arr > threshold).mean()),
    }
    for k, v in metrics.items():
        exp.log_metric(k, v)

    with open(exp.run_dir / "e2e_results.json", "w", encoding="utf-8") as f:
        json.dump(per_scene, f, indent=2, ensure_ascii=False)

    if sample_data:
        fig, axs = plt.subplots(len(sample_data), 3, figsize=(12, 4.2 * len(sample_data)))
        axs = axs.reshape(len(sample_data), 3)
        for r, (scene, patch, out) in enumerate(sample_data):
            axs[r, 0].imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
            axs[r, 0].set_title("Scena (input YOLO OBB)", fontsize=9)
            axs[r, 1].imshow(patch)
            axs[r, 1].set_title("Ritaglio canonico", fontsize=9)
            axs[r, 2].imshow(
                cv2.cvtColor(
                    cv2.imdecode(
                        np.frombuffer(
                            __import__("base64").b64decode(out["heatmap_b64"]),
                            np.uint8,
                        ),
                        cv2.IMREAD_COLOR,
                    ),
                    cv2.COLOR_BGR2RGB,
                )
            )
            axs[r, 2].set_title(
                f"Mappa difetti - score {out['score']:.3f} ({out['condition']['label']})",
                fontsize=9,
            )
            for c in range(3):
                axs[r, c].axis("off")
        fig.suptitle("End-to-end pipeline: detection -> rectification -> anomaly")
        fig.tight_layout()
        exp.save_plot(fig, "e2e_samples")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(scores_arr, bins=12, color="#3B9EFF", alpha=0.8, edgecolor="white")
    ax.axvline(threshold, color="#FF4D6A", linestyle="--", linewidth=2, label=f"Soglia ({threshold:.4f})")
    ax.set_title("Distribuzione dei punteggi di anomalia (carte integre)")
    ax.set_xlabel("Punteggio di anomalia")
    ax.set_ylabel("Frequenza")
    ax.legend()
    fig.tight_layout()
    exp.save_plot(fig, "e2e_scores")

if __name__ == "__main__":
    main()

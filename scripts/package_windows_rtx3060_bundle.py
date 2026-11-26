"""Builds portable Windows CUDA (RTX 3060 12GB, 64GB RAM) training & benchmark bundles.

Default profile (``heavy``) produces ``pokedino_rtx3060_heavy_obb_bundle.zip``:
  - Complete source (``src/``), training/benchmark scripts (``scripts/``),
    ``configs/default.yaml``, ``requirements.txt``, ``run_server.py``, and
    domain agent skills (``.agents/``).
  - Tailored ``AGENTS.md`` (workspace rules + Windows RTX 3060 coding-agent
    playbook for 18,168-scene ``YOLO11l-OBB`` two-phase fine-tuning, Stage-2
    Canonical UNet refiner training, and Masked CLAHE ON/OFF + corner benchmarks).
  - Offline datasets: ``pokemon_obb`` (14,168 scenes), ``pokemon_seg`` (4,000
    paired stress scenes), ``pokemon_canon_refiner`` (4,500 native 768x544
    Stage-2 crops), ``backgrounds`` (352 surfaces), ``reference_cache/*.png``,
    ``card_catalog.json``, baseline ``.pt`` checkpoints, and held-out real
    smartphone captures (``experiments/synthetic-to-real/*`` top-level files only).
  - One-command runner ``run_windows_rtx3060_heavy.py`` collecting everything into
    ``pokedino_trained_outputs_v4/``.
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)

AGENTS_WINDOWS_ADDENDUM = """

---

## Windows RTX 3060 Coding-Agent Playbook (Remote Training & Benchmarking)

This portable workspace targets a Windows desktop workstation with **NVIDIA RTX 3060 (12 GB VRAM)**, **64 GB System RAM**, and CUDA enabled. Always verify `POKE_DEVICE=cuda` (`$env:POKE_DEVICE = "cuda"` in PowerShell) before running training or benchmarks.

### 1. Pre-Bundled Offline Datasets & Weights

- `data/datasets/pokemon_obb/` — **14,168** OBB scenes (`9,921` train / `2,104` val / `2,143` test).
- `data/datasets/pokemon_seg/` — **4,000** shadow/glare/occlusion/sleeve/full-art stress scenes (`3,000` train / `500` val / `500` test) with synchronized Seg polygons (`labels/`), masks (`masks/`), and OBB labels (`labels_obb/`).
- `data/datasets/pokemon_canon_refiner/` — **4,500** canonical `768x544` Stage-2 refiner crops (`3,500` train / `500` val / `500` test) with occlusion-aware masks and `corners_canon` metadata.
- `data/datasets/backgrounds/` (352 surfaces), `data/reference_cache/*.png`, `data/card_catalog.json` — offline assets for AnomalyDINO defect & CLAHE benchmarks.
- `experiments/synthetic-to-real/` — Real held-out smartphone captures (**EVALUATION ONLY — NEVER INCLUDE IN TRAINING**).
- `data/weights/` — Includes baseline `yolo11s_obb_finetuned_best.pt`, `yolo11s_seg_best.pt`, and `seg_semantic_best.pt` for head-to-head benchmarking.

### 2. Recommended Heavy Training & Benchmark Protocol (RTX 3060 12 GB Safe)

Execute via `python run_windows_rtx3060_heavy.py` or step-by-step:

1. **Stage 1 — Heavy Two-Phase `YOLO11l-OBB` Fine-Tuning (`18,168` unified scenes)**:
   - Command:
     `python scripts/train_obb_heavy.py --base-model yolo11l-obb.pt --epochs-p1 80 --patience-p1 20 --epochs-p2 40 --patience-p2 15 --batch 8 --imgsz 640 --workers 4`
   - Automatically hard-links `pokemon_obb` + `pokemon_seg/labels_obb` into `data/datasets/pokemon_obb_unified` (`12,921` train / `2,604` val / `2,643` test) with portable YAML path.
   - Phase 1 trains `YOLO11l-OBB` (`mosaic=0.30, close_mosaic=12, box=12.0, dfl=2.5, angle=2.5`). Phase 2 disables mosaic/mixup completely (`mosaic=0.0, mixup=0.0, erasing=0.0`) and boosts geometric regression weights (`box=18.0, dfl=4.0, angle=3.0, cls=0.15, lr0=0.00025`) for sub-pixel corners.
   - Output: `data/weights/yolo11l_obb_heavy_best.pt` (`yolo11l_obb_phase1_best.pt` also saved).
   - VRAM note: Keep `--batch 8` at `imgsz=640` for `yolo11l-obb` (do not exceed batch 12 without monitoring `nvidia-smi`).

2. **Stage 2 — Canonical `MobileNetV3-UNet` Boundary Refiner (`768x544`)**:
   - Command:
     `python scripts/train_canonical_refiner.py --epochs 80 --patience 15 --batch 8 --workers 4`
   - Uses `0.45 * BorderWeightedBCE + 0.35 * Dice + 0.20 * SobelBoundary`, AMP on CUDA, early-stopping patience `15`, and evaluates geometric corner error (`val_corner_err_px`, `test_corner_err_mean`) via `fit_quad_from_canonical_prob`.
   - Output: `data/weights/seg_canonical_unet_best.pt` (`weights_only=True` safe).

3. **Corner Perfection & Masked CLAHE (`ON` vs `OFF`) Benchmark Suite**:
   - Commands:
     - `python scripts/benchmark_clahe_and_corners.py --mode all --max-synth-scenes 500 --n-defect-cards 8`
     - `python scripts/benchmark_segmentation_vs_obb.py`
     - `python scripts/visual_corner_border_diagnostic.py`
     - `python scripts/test_canonical_refiner.py`
     - `python scripts/test_webapp_backends.py`
   - Outputs JSON summaries (`clahe_and_corners_report.json`, `benchmark_summary.json`), corner CDF plots, stratified IoU bars, and side-by-side real-card AnomalyDINO heatmaps (`CLAHE ON` vs `CLAHE OFF` across `obb_pbl_heavy`, `obb_unet`, `seg_yolo_640`).

4. **Iteration Guidance for Coding Agent**:
   - Check corner precision first: inspect `corners.<backend>.corner_err_median`, `pct_lt_1px`, `pct_lt_2px`, and `report/figures/05_segmentation_pipeline/corner_diagnostics/diag_*.png`. If `obb_raw_heavy` still drifts >1.5 px on perspective/occluded cards, confirm `obb_unet` (`corridor_px=28`) fixes those corners cleanly.
   - Check Masked CLAHE decision: inspect `clahe.<backend>.clahe_on_luma_on` vs `clahe_off_luma_on` across `roc_auc_all`, `roc_auc_subtle`, `edge_wear_roc_auc`, `pristine_std`, and real Gengar/Charizard/Giratina separation before recommending final `configs/default.yaml` `clahe.enabled`.
   - Collect all `.pt` weights and `experiments/` folders into `pokedino_trained_outputs_v4/` when finished.
"""

INSTRUCTIONS_HEAVY_MD = """# PokéDINO — Heavy YOLO11l-OBB + Canonical UNet + CLAHE Benchmark (RTX 3060 12GB)

Questa cartella contiene tutto il workspace pronto per il **coding agent** sul PC fisso Windows (RTX 3060 12GB VRAM, 64GB RAM, CUDA).

## 1. Setup Ambiente (una volta sola)

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python-headless "numpy<2.0" scipy pyyaml matplotlib tqdm scikit-learn fastapi httpx pillow requests
```

Verifica CUDA:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Esecuzione Automatica Completa (`python run_windows_rtx3060_heavy.py`)

```powershell
$env:POKE_DEVICE = "cuda"
python run_windows_rtx3060_heavy.py
```

Cosa esegue in sequenza:
1. **`YOLO11l-OBB` Heavy Two-Phase Fine-Tuning** su dataset unificato **18.168 scene** (`pokemon_obb` 14.168 + `pokemon_seg/labels_obb` 4.000):
   - Phase 1: `80 epoche`, `patience=20`, `batch=8`, `imgsz=640`
   - Phase 2: `40 epoche`, `patience=15`, `mosaic=0.0`, `box=18.0, dfl=4.0, angle=3.0` → `data/weights/yolo11l_obb_heavy_best.pt`
2. **`Canonical MobileNetV3-UNet`** (`768x544`) con `80 epoche`, `patience=15` → `data/weights/seg_canonical_unet_best.pt`
3. **Corner + Masked CLAHE ON/OFF Benchmark Suite** (`benchmark_clahe_and_corners.py`, `benchmark_segmentation_vs_obb.py`, `visual_corner_border_diagnostic.py`, `test_canonical_refiner.py`, `test_webapp_backends.py`)
4. Raccolta automatica di tutti i pesi `.pt`, grafici, JSON benchmark e CSV in **`pokedino_trained_outputs_v4/`**.

> Consulta **`AGENTS.md`** dentro questa cartella: contiene tutte le istruzioni complete per far lavorare autonomamente il tuo coding agent su training, tuning dei corner e benchmark CLAHE ON/OFF.
"""

RUNNER_HEAVY_PY = '''"""One-command RTX 3060 Windows runner for Heavy YOLO11l-OBB + Canonical UNet + CLAHE Benchmarks."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "pokedino_trained_outputs_v4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RTX 3060 Heavy OBB + Stage-2 Refiner + CLAHE Benchmarks.")
    parser.add_argument("--with-yolo-seg", action="store_true", help="Also train YOLO11l-Seg @ 640 (50 epochs).")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip post-training benchmark scripts.")
    args = parser.parse_args()

    env = os.environ.copy()
    env["POKE_DEVICE"] = "cuda"
    env.setdefault("PYTHONUNBUFFERED", "1")

    commands = [
        [
            sys.executable, "scripts/train_obb_heavy.py",
            "--base-model", "yolo11l-obb.pt",
            "--epochs-p1", "80", "--patience-p1", "20",
            "--epochs-p2", "40", "--patience-p2", "15",
            "--batch", "8", "--imgsz", "640", "--workers", "4",
        ],
        [
            sys.executable, "scripts/train_canonical_refiner.py",
            "--epochs", "80", "--patience", "15",
            "--batch", "8", "--workers", "4",
        ],
    ]
    if args.with_yolo_seg:
        commands.append([
            sys.executable, "scripts/train_seg.py",
            "--model", "yolo640", "--base-model", "yolo11l-seg.pt",
            "--epochs-yolo", "60", "--batch", "8", "--workers", "4", "--mask-ratio", "1",
        ])

    if not args.skip_benchmarks:
        commands.extend([
            [sys.executable, "scripts/test_canonical_refiner.py"],
            [
                sys.executable, "scripts/benchmark_clahe_and_corners.py",
                "--mode", "all", "--max-synth-scenes", "500", "--n-defect-cards", "8",
            ],
            [sys.executable, "scripts/benchmark_segmentation_vs_obb.py"],
            [sys.executable, "scripts/visual_corner_border_diagnostic.py"],
            [sys.executable, "scripts/test_webapp_backends.py"],
        ])

    exp_root = ROOT / "experiments"
    before = set(p.name for p in exp_root.glob("*")) if exp_root.exists() else set()

    for cmd in commands:
        print("\\n>>> Running:", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if res.returncode != 0:
            raise SystemExit(f"Command failed ({res.returncode}): {' '.join(cmd)}")

    after = set(p.name for p in exp_root.glob("*")) if exp_root.exists() else set()
    new_exps = sorted(after - before)

    w_out = OUT_DIR / "data" / "weights"
    w_out.mkdir(parents=True, exist_ok=True)
    for w in (ROOT / "data" / "weights").glob("*.pt"):
        shutil.copy2(w, w_out / w.name)

    e_out = OUT_DIR / "experiments"
    e_out.mkdir(parents=True, exist_ok=True)
    for name in new_exps:
        src = exp_root / name
        dst = e_out / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    diag_src = ROOT / "report" / "figures" / "05_segmentation_pipeline" / "corner_diagnostics"
    if diag_src.exists():
        diag_dst = OUT_DIR / "corner_diagnostics"
        if diag_dst.exists():
            shutil.rmtree(diag_dst)
        shutil.copytree(diag_src, diag_dst)

    print(f"\\nALL DONE! Outputs archived in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
'''


def _add_tree(
    zf: zipfile.ZipFile,
    src_dir: Path,
    rel_base: Path,
    skip_globs: tuple[str, ...] = ("__pycache__", ".DS_Store"),
    top_level_only: bool = False,
) -> int:
    count = 0
    for p in sorted(src_dir.rglob("*") if not top_level_only else src_dir.iterdir()):
        if not p.is_file():
            continue
        if any(s in p.parts for s in skip_globs):
            continue
        arcname = str(rel_base / p.relative_to(src_dir))
        zf.write(p, arcname)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Windows RTX 3060 training & benchmark workspace zip.")
    parser.add_argument(
        "--out",
        type=str,
        default="pokedino_rtx3060_heavy_obb_bundle.zip",
        help="Output bundle filename in project root.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    canon_dir = PROJECT_ROOT / "data" / "datasets" / "pokemon_canon_refiner"
    obb_dir = PROJECT_ROOT / "data" / "datasets" / "pokemon_obb"
    seg_dir = PROJECT_ROOT / "data" / "datasets" / "pokemon_seg"
    for req_dir in (canon_dir, obb_dir, seg_dir):
        if not (req_dir / "images" / "train").exists():
            raise FileNotFoundError(f"Missing required dataset split under {req_dir}")

    zip_path = PROJECT_ROOT / args.out
    if zip_path.exists():
        zip_path.unlink()

    base_agents = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    combined_agents = base_agents.rstrip() + AGENTS_WINDOWS_ADDENDUM

    n_files = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
        zf.writestr("AGENTS.md", combined_agents)
        zf.writestr("INSTRUCTIONS_WINDOWS_RTX3060.md", INSTRUCTIONS_HEAVY_MD)
        zf.writestr("run_windows_rtx3060_heavy.py", RUNNER_HEAVY_PY)
        n_files += 3

        for single in (
            "configs/default.yaml",
            "requirements.txt",
            "run_server.py",
            "data/card_catalog.json",
            "data/weights/yolo11s_obb_finetuned_best.pt",
            "data/weights/yolo11s_seg_best.pt",
            "data/weights/seg_semantic_best.pt",
        ):
            sp = PROJECT_ROOT / single
            if sp.exists():
                zf.write(sp, single)
                n_files += 1

        # Rewrite portable pokemon_obb.yaml for Windows workspace
        zf.writestr(
            "data/datasets/pokemon_obb/pokemon_obb.yaml",
            "path: data/datasets/pokemon_obb\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: card\n",
        )
        n_files += 1

        if (PROJECT_ROOT / ".agents").exists():
            n_files += _add_tree(zf, PROJECT_ROOT / ".agents", Path(".agents"))
        n_files += _add_tree(zf, PROJECT_ROOT / "src", Path("src"))
        n_files += _add_tree(zf, PROJECT_ROOT / "scripts", Path("scripts"))

        # Package pokemon_obb (skip old yaml already written above)
        n_files += _add_tree(zf, obb_dir / "images", Path("data/datasets/pokemon_obb/images"))
        n_files += _add_tree(zf, obb_dir / "labels", Path("data/datasets/pokemon_obb/labels"))
        n_files += _add_tree(zf, seg_dir, Path("data/datasets/pokemon_seg"))
        n_files += _add_tree(zf, canon_dir, Path("data/datasets/pokemon_canon_refiner"))

        bg_dir = PROJECT_ROOT / "data" / "datasets" / "backgrounds"
        if bg_dir.exists():
            n_files += _add_tree(zf, bg_dir, Path("data/datasets/backgrounds"))

        ref_cache = PROJECT_ROOT / "data" / "reference_cache"
        if ref_cache.exists():
            n_files += _add_tree(zf, ref_cache, Path("data/reference_cache"), top_level_only=True)

        real_dir = PROJECT_ROOT / "experiments" / "synthetic-to-real"
        if real_dir.exists():
            n_files += _add_tree(zf, real_dir, Path("experiments/synthetic-to-real"), top_level_only=True)

    bad = zipfile.ZipFile(zip_path, "r").testzip()
    if bad is not None:
        raise RuntimeError(f"Corrupted zip entry: {bad}")

    size_mb = zip_path.stat().st_size / (1024.0 * 1024.0)
    logger.info("Created %s (%.1f MB, %d entries, testzip=OK)", zip_path, size_mb, n_files)


if __name__ == "__main__":
    main()

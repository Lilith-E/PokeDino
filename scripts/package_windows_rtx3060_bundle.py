"""Builds portable Windows CUDA (RTX 3060 12GB) training bundle zip (v3).

Packages source code, configs, both paired datasets (``pokemon_seg`` with
``labels_obb`` + ``pokemon_canon_refiner`` at native ``768x544`` resolution),
baseline OBB checkpoint, held-out real images (verification only), and a
one-command runner (``run_windows_rtx3060_v3.py``) saving trained weights,
plots, hyperparameters, and metrics into ``pokedino_trained_outputs_v3/``.
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)

INSTRUCTIONS_V3_MD = """# PokéDINO v3 — Training Two-Stage `obb_unet` su Windows (RTX 3060 12GB VRAM, 64GB RAM)

Questo pacchetto implementa la nuova architettura two-stage **`obb_unet`**:
1. **Stage 1 — `YOLO11m-OBB` Heavy Paired Detector** (`60 epoche @ 640`, `box=14.0, dfl=3.0, angle=2.5`) con **Adaptive TTA** (single-pass veloce su immagini pulite, 4-view TTA automatico solo su casi ambigui/bassi/al bordo).
2. **Stage 2 — `Canonical MobileNetV3-UNet`** sui crop canonici **`768×544`** (risoluzione nativa 1:1 intorno alla carta `630×448` con margine corridor), addestrato con **Border-Weighted BCE + Dice + Sobel Boundary Loss** su **`data/datasets/pokemon_canon_refiner`** (4.500 scene canoniche con perspective tilt, perturbazioni OBB realistiche, ombre proiettate, glare, full-art borderless, sleeve e occlusioni da dita), seguito da fitting geometrico **4-linee RANSAC outside-to-inside** nel corridor OBB (`±28 px`).

> **Nota metodologica**: `experiments/synthetic-to-real/` è incluso **solo** per eventuale test visuale post-training e non entra mai nel training.

## 1. Setup ambiente (una volta sola)

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python-headless "numpy<2.0" scipy pyyaml matplotlib tqdm scikit-learn fastapi httpx
```

Verifica CUDA:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Esecuzione automatica one-command (`pokedino_trained_outputs_v3/`)

Lancia il trainer sequenziale two-stage (~25-35 min su RTX 3060 12GB):

```powershell
$env:POKE_DEVICE = "cuda"
python run_windows_rtx3060_v3.py
```

Output principali raccolti automaticamente in **`pokedino_trained_outputs_v3/`**:
- `data/weights/yolo11m_obb_paired_best.pt` (Stage 1 OBB heavy)
- `data/weights/seg_canonical_unet_best.pt` (Stage 2 Canonical Boundary Refiner)
- `experiments/` (curve loss, mAP50-95 OBB, Val Boundary IoU, Val/Test Corner Error px @ 630×448, `hyperparameters.json`, `metrics.json`)

### Opzionale: rigenerazione dataset Stage-2 o re-training `YOLO11l-Seg`

Se desideri rigenerare un dataset canonico più grande o includere anche `YOLO11l-Seg @ 640` nella run:

```powershell
python run_windows_rtx3060_v3.py --with-yolo-seg
```

Oppure esegui i singoli stadi manualmente:

```powershell
$env:POKE_DEVICE = "cuda"
python scripts/finetune_obb_paired.py --base-model yolo11m-obb.pt --epochs 60 --batch 12 --imgsz 640 --workers 4 --out-name yolo11m_obb_paired_best.pt
python scripts/train_canonical_refiner.py --epochs 50 --batch 8 --workers 4
```

Al termine, copia la cartella **`pokedino_trained_outputs_v3`** sul Mac per il benchmark finale su carte reali (`obb_pbl` vs `obb_unet` vs `seg_yolo` vs `seg_semantic`) e l'aggiornamento dei grafici nella tesi!
"""

RUNNER_V3_PY = '''"""One-command RTX 3060 Windows runner for Stage-1 YOLO11m-OBB + Stage-2 Canonical UNet (v3)."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "pokedino_trained_outputs_v3"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RTX 3060 v3 two-stage training suite.")
    parser.add_argument("--with-yolo-seg", action="store_true", help="Also train YOLO11l-Seg @ 640 for comparison.")
    args = parser.parse_args()

    env = os.environ.copy()
    env["POKE_DEVICE"] = "cuda"
    env.setdefault("PYTHONUNBUFFERED", "1")

    commands = [
        [
            sys.executable, "scripts/finetune_obb_paired.py",
            "--base-model", "yolo11m-obb.pt",
            "--epochs", "60", "--batch", "12", "--imgsz", "640", "--workers", "4",
            "--out-name", "yolo11m_obb_paired_best.pt",
        ],
        [
            sys.executable, "scripts/train_canonical_refiner.py",
            "--epochs", "50", "--batch", "8", "--workers", "4",
        ],
    ]
    if args.with_yolo_seg:
        commands.append([
            sys.executable, "scripts/train_seg.py",
            "--model", "yolo640", "--base-model", "yolo11l-seg.pt",
            "--epochs-yolo", "50", "--batch", "8", "--workers", "4", "--mask-ratio", "1",
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

    print(f"\\nALL DONE! Copy {OUT_DIR} back to macOS.", flush=True)


if __name__ == "__main__":
    main()
'''


def _add_tree(
    zf: zipfile.ZipFile,
    src_dir: Path,
    rel_base: Path,
    skip_globs: tuple[str, ...] = ("__pycache__", ".DS_Store"),
) -> int:
    count = 0
    for p in sorted(src_dir.rglob("*")):
        if not p.is_file():
            continue
        if any(s in p.parts for s in skip_globs):
            continue
        arcname = str(rel_base / p.relative_to(src_dir))
        zf.write(p, arcname)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Windows RTX 3060 training bundle zip.")
    parser.add_argument(
        "--out",
        type=str,
        default="pokedino_windows_rtx3060_bundle_v3.zip",
        help="Output bundle filename in project root.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    canon_dir = PROJECT_ROOT / "data" / "datasets" / "pokemon_canon_refiner"
    if not (canon_dir / "images" / "train").exists():
        raise FileNotFoundError(
            f"Missing canonical refiner dataset under {canon_dir}. "
            "Run scripts/prepare_canonical_refiner_dataset.py first."
        )

    zip_path = PROJECT_ROOT / args.out
    if zip_path.exists():
        zip_path.unlink()

    n_files = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as zf:
        for single in ("AGENTS.md", "configs/default.yaml", "data/weights/yolo11s_obb_finetuned_best.pt"):
            sp = PROJECT_ROOT / single
            if sp.exists():
                zf.write(sp, single)
                n_files += 1

        n_files += _add_tree(zf, PROJECT_ROOT / "src", Path("src"))
        n_files += _add_tree(zf, PROJECT_ROOT / "scripts", Path("scripts"))
        n_files += _add_tree(zf, PROJECT_ROOT / "data" / "datasets" / "pokemon_seg", Path("data/datasets/pokemon_seg"))
        n_files += _add_tree(zf, canon_dir, Path("data/datasets/pokemon_canon_refiner"))
        if (PROJECT_ROOT / "experiments" / "synthetic-to-real").exists():
            n_files += _add_tree(zf, PROJECT_ROOT / "experiments" / "synthetic-to-real", Path("experiments/synthetic-to-real"))

        zf.writestr("INSTRUCTIONS_WINDOWS_RTX3060.md", INSTRUCTIONS_V3_MD)
        zf.writestr("run_windows_rtx3060_v3.py", RUNNER_V3_PY)
        n_files += 2

    bad = zipfile.ZipFile(zip_path, "r").testzip()
    if bad is not None:
        raise RuntimeError(f"Corrupted zip entry: {bad}")

    size_mb = zip_path.stat().st_size / (1024.0 * 1024.0)
    logger.info("Created %s (%.1f MB, %d entries, testzip=OK)", zip_path, size_mb, n_files)


if __name__ == "__main__":
    main()

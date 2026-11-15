"""Builds a portable Windows CUDA (RTX 3060 12GB) training bundle zip.

Packages source code, configs, shadow/glare-augmented paired dataset
(pokemon_seg + labels_obb), baseline OBB starting checkpoint, held-out real
images (for post-training verification only), and a one-command runner script
that saves all trained weights, plots, hyperparameters, and metrics into
pokedino_trained_outputs_v2/.
"""

from __future__ import annotations

import logging
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)

INSTRUCTIONS_MD = """# PokéDINO v2 — Istruzioni Training su Windows (RTX 3060 12GB VRAM, 64GB RAM CUDA)

Questo pacchetto contiene:
- Il nuovo dataset **`data/datasets/pokemon_seg`** (4.000 scene 640×640 con **ombre proiettate realistiche**, **glare su bordi/angoli**, **full-art borderless**, **tight close-up**, **tilt prospettico**, **sleeve** e **occlusioni da dita**), completo di **annotazioni appaiate sia YOLO-Seg (`labels/`) sia YOLO-OBB (`labels_obb/`)** sulle stesse identiche scene.
- Gli script di training pesante (`50 epoche`) e raccolta automatica di pesi, plot, `metrics.json`, CSV e iperparametri dentro `pokedino_trained_outputs_v2/`.
- **Nota metodologica**: `experiments/synthetic-to-real/` è incluso **solo** per eventuale test visuale post-training e NON viene MAI usato durante il training.

## 1. Setup ambiente (una volta sola)

Apri PowerShell / prompt nella cartella estratta (es. `C:\\PokeDino_v2`):

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python-headless "numpy<2.0" scipy pyyaml matplotlib tqdm scikit-learn
```

Verifica CUDA:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Esecuzione completa automatica (Consigliata — 1 comando)

Lancia il runner unificato che addestra in sequenza:
1. **`YOLO11l-Seg @ 640`** (50 epoche, `mask_ratio=1`, `overlap_mask=False`, `batch=8`) → `data/weights/yolo11s_seg_best.pt`
2. **`YOLO11l-Seg @ 960`** (50 epoche, `mask_ratio=1`, `overlap_mask=False`, `batch=4`) → `data/weights/yolo11s_seg_960_best.pt`
3. **`MobileNetV3-UNet @ 640`** (45 epoche, `BCE + Dice + Sobel Boundary`, `batch=8`) → `data/weights/seg_semantic_best.pt`
4. **`YOLO11s-OBB (Paired Fair Benchmark)`** sugli **identici dati con ombre** (`50 epoche`, `batch=16`, `box=14.0`, `dfl=3.0`, `angle=2.5`) → `data/weights/yolo11s_obb_paired_best.pt`

E copia automaticamente tutti i pesi, curve di loss/mAP, `hyperparameters.json` e `metrics.json` in **`pokedino_trained_outputs_v2/`**:

```powershell
$env:POKE_DEVICE = "cuda"
python run_windows_rtx3060_v2.py
```

### Oppure comandi manuali singoli

```powershell
$env:POKE_DEVICE = "cuda"
python scripts/train_seg.py --model yolo640 --base-model yolo11l-seg.pt --epochs-yolo 50 --batch 8 --workers 4 --mask-ratio 1
python scripts/train_seg.py --model yolo960 --base-model yolo11l-seg.pt --epochs-yolo 50 --batch 4 --workers 4 --mask-ratio 1
python scripts/train_seg.py --model semantic --epochs-semantic 45 --batch 8 --workers 4
python scripts/finetune_obb_paired.py --base-model data/weights/yolo11s_obb_finetuned_best.pt --epochs 50 --batch 16 --imgsz 640 --workers 4
```

Al termine, riporta sul Mac la cartella **`pokedino_trained_outputs_v2`** per il benchmark finale *apples-to-apples* (`OBB-Paired+PBL` vs `YOLO11-Seg + Hybrid Sobel/Lab Rectifier + Crop-Refine`) e l'aggiornamento del report LaTeX!
"""

RUNNER_PY = """\"\"\"One-command RTX 3060 Windows runner for paired Seg + OBB v2 heavy training.\"\"\"

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "pokedino_trained_outputs_v2"

ENV = os.environ.copy()
ENV["POKE_DEVICE"] = "cuda"
ENV.setdefault("PYTHONUNBUFFERED", "1")

COMMANDS = [
    [
        sys.executable, "scripts/train_seg.py",
        "--model", "yolo640", "--base-model", "yolo11l-seg.pt",
        "--epochs-yolo", "50", "--batch", "8", "--workers", "4", "--mask-ratio", "1",
    ],
    [
        sys.executable, "scripts/train_seg.py",
        "--model", "yolo960", "--base-model", "yolo11l-seg.pt",
        "--epochs-yolo", "50", "--batch", "4", "--workers", "4", "--mask-ratio", "1",
    ],
    [
        sys.executable, "scripts/train_seg.py",
        "--model", "semantic",
        "--epochs-semantic", "45", "--batch", "8", "--workers", "4",
    ],
    [
        sys.executable, "scripts/finetune_obb_paired.py",
        "--base-model", "data/weights/yolo11s_obb_finetuned_best.pt",
        "--epochs", "50", "--batch", "16", "--imgsz", "640", "--workers", "4",
    ],
]

def main() -> None:
    exp_root = ROOT / "experiments"
    before = set(p.name for p in exp_root.glob("*")) if exp_root.exists() else set()

    for cmd in COMMANDS:
        print("\\n>>> Running:", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, cwd=str(ROOT), env=ENV)
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
"""


def _add_tree(zf: zipfile.ZipFile, src_dir: Path, rel_base: Path, skip_globs: tuple[str, ...] = ("__pycache__", ".DS_Store")) -> int:
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    zip_path = PROJECT_ROOT / "pokedino_windows_rtx3060_bundle_v2.zip"
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
        if (PROJECT_ROOT / "experiments" / "synthetic-to-real").exists():
            n_files += _add_tree(zf, PROJECT_ROOT / "experiments" / "synthetic-to-real", Path("experiments/synthetic-to-real"))

        zf.writestr("INSTRUCTIONS_WINDOWS_RTX3060.md", INSTRUCTIONS_MD)
        zf.writestr("run_windows_rtx3060_v2.py", RUNNER_PY)
        n_files += 2

    bad = zipfile.ZipFile(zip_path, "r").testzip()
    if bad is not None:
        raise RuntimeError(f"Corrupted zip entry: {bad}")

    size_mb = zip_path.stat().st_size / (1024.0 * 1024.0)
    logger.info("Created %s (%.1f MB, %d entries, testzip=OK)", zip_path, size_mb, n_files)


if __name__ == "__main__":
    main()

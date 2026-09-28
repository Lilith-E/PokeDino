"""Download and curate realistic background images for PokéDINO scene generation.

Sources:
1. Hugging Face DTD (Describable Textures Dataset) via 'Multimodal-Fatima/DTD_parition1_train':
   Selected tabletop and fabric classes: marbled, fibrous, chequered, grid, grooved,
   lined, striped, woven, matted, crosshatched, veined, banded, stained, flecked.
2. Hugging Face CC0 PBR Architectural & Tabletop Materials via 'nyuuzyou/cc0-textures':
   Photographic PBR textures: wood, planks, marble, fabric, cloth, tiles, leather, metal.
3. Curated Workspace & Tabletop Photography:
   Real desk surfaces, keyboards, office setups, and gaming mats.

All images are resized/center-cropped to 640x640 px and saved in data/datasets/backgrounds/
along with an indexed backgrounds_manifest.json.
"""

import argparse
import io
import json
import logging
import random
from pathlib import Path

import requests
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BG_DIR = PROJECT_ROOT / "data" / "datasets" / "backgrounds"

IMG_W, IMG_H = 640, 640

TARGET_DTD_CLASSES = {
    "marbled", "fibrous", "chequered", "grid", "grooved", "lined",
    "striped", "woven", "matted", "crosshatched", "veined", "banded",
    "stained", "flecked", "wrinkled", "blotchy"
}

TARGET_PBR_TAGS = {
    "wood", "planks", "plank", "parquet", "marble", "fabric", "cloth",
    "leather", "concrete", "metal", "tiles", "countertop", "floor", "table"
}

CURATED_UNSPLASH_IDS = [
    "1518455027359-f3f8164ba6bd",
    "1527443224154-c4a3942d3acf",
    "1507207611509-ec012433ff52",
    "1497215728101-856f4ea42174",
    "1498050108023-c5249f4df085",
    "1524758631624-e2822e304c36",
    "1493934558415-9d19f0b2b4d2",
    "1533090161767-e6ffed986c88",
    "1486312338219-ce68d2c6f44d",
    "1471341971476-ae15ff5dd4ea",
    "1505330622279-bf7d7fc918f4",
    "1496171367470-9ed9a91ea931",
    "1519389950473-47ba0277781c",
    "1515378791036-0648a3ef77b2",
    "1461749280684-dccba630e2f6",
    "1501504905252-473c47e087f8",
    "1531403009284-440f080d1e12",
    "1517048676732-d65bc937f952",
    "1499750310107-5fef28a66643",
    "1487014679447-9f8336841d58",
    "1520607162513-77705c0f0d4a",
    "1488998427799-e3362cec87c3",
    "1455390582262-044cdead277a",
    "1516321318423-f06f85e504b3",
    "1517245386807-bb43f82c33c4",
    "1581291518633-83b4ebd1d83e",
    "1434030216411-0b793f4b4173",
    "1504384308090-c894fdcc538d",
    "1508873696983-2df5703bc248",
    "1519389950473-47ba0277781c",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Download and curate background images for PokéDINO.")
    parser.add_argument("--bg-dir", type=str, default=str(DEFAULT_BG_DIR), help="Output directory for backgrounds.")
    parser.add_argument("--dtd-max", type=int, default=180, help="Max backgrounds from Hugging Face DTD.")
    parser.add_argument("--pbr-max", type=int, default=160, help="Max backgrounds from Hugging Face CC0 PBR.")
    parser.add_argument("--desk-max", type=int, default=30, help="Max backgrounds from curated desk workspaces.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    return parser.parse_args()

def crop_and_resize(img: Image.Image, target_w: int = IMG_W, target_h: int = IMG_H) -> Image.Image:
    """Center-crop and resize an image to target dimensions while maintaining aspect ratio."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = max(target_w / w, target_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - target_w) // 2
    top = (nh - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))

def download_dtd_backgrounds(out_dir: Path, max_samples: int, seed: int) -> list[dict]:
    """Download tabletop and pattern textures from Hugging Face DTD."""
    records = []
    try:
        from datasets import load_dataset
        logger.info("Loading DTD dataset from 'Multimodal-Fatima/DTD_parition1_train'...")
        ds = load_dataset("Multimodal-Fatima/DTD_parition1_train", split="train")
        class_names = ds.features["label"].names

        class_indices = {}
        for idx, row in enumerate(ds):
            lbl_name = class_names[row["label"]]
            if lbl_name in TARGET_DTD_CLASSES:
                class_indices.setdefault(lbl_name, []).append(idx)

        logger.info(f"DTD matching classes found: {list(class_indices.keys())}")
        rng = random.Random(seed)

        selected_indices = []
        per_class_quota = max(1, max_samples // max(1, len(class_indices)))
        for cname, idxs in class_indices.items():
            rng.shuffle(idxs)
            selected_indices.extend([(idx, cname) for idx in idxs[:per_class_quota]])

        rng.shuffle(selected_indices)
        selected_indices = selected_indices[:max_samples]

        logger.info(f"Extracting {len(selected_indices)} DTD background images...")
        for count, (ds_idx, cname) in enumerate(selected_indices):
            out_file = out_dir / f"bg_dtd_{count:04d}.jpg"
            if not out_file.exists():
                item = ds[ds_idx]
                pil_img = item["image"]
                final_img = crop_and_resize(pil_img, IMG_W, IMG_H)
                final_img.save(str(out_file), "JPEG", quality=92)

            records.append({
                "filename": out_file.name,
                "source": "huggingface/dtd",
                "category": f"dtd/{cname}",
                "path": str(out_file)
            })

            if (count + 1) % 50 == 0 or (count + 1) == len(selected_indices):
                logger.info(f"  DTD: {count + 1}/{len(selected_indices)} saved")

    except Exception as e:
        logger.warning(f"DTD download encountered an error: {e}")

    return records

def download_pbr_backgrounds(out_dir: Path, max_samples: int, seed: int) -> list[dict]:
    """Download wood, marble, fabric, and surface textures from Hugging Face CC0 PBR dataset."""
    records = []
    try:
        from datasets import load_dataset
        logger.info("Loading CC0 PBR textures from 'nyuuzyou/cc0-textures'...")
        ds = load_dataset("nyuuzyou/cc0-textures", split="train", streaming=True)

        count = 0
        for item in ds:
            if count >= max_samples:
                break
            meta = item.get("json") or {}
            tags = [t.lower() for t in (meta.get("tags") or [])]
            matched = [t for t in tags if t in TARGET_PBR_TAGS]
            if not matched:
                continue

            category_name = matched[0]
            out_file = out_dir / f"bg_pbr_{count:04d}.jpg"
            if not out_file.exists():
                pil_img = item.get("jpg")
                if pil_img is None:
                    continue
                final_img = crop_and_resize(pil_img, IMG_W, IMG_H)
                final_img.save(str(out_file), "JPEG", quality=92)

            records.append({
                "filename": out_file.name,
                "source": "huggingface/cc0-textures",
                "category": f"pbr/{category_name}",
                "path": str(out_file)
            })
            count += 1
            if count % 40 == 0 or count == max_samples:
                logger.info(f"  CC0 PBR: {count}/{max_samples} saved")

    except Exception as e:
        logger.warning(f"CC0 PBR download encountered an error: {e}")

    return records

def download_curated_desks(out_dir: Path, max_samples: int) -> list[dict]:
    """Download curated workspace and tabletop photography from Unsplash."""
    records = []
    selected_ids = CURATED_UNSPLASH_IDS[:max_samples]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PokeDINOResearch/1.0)"}

    logger.info(f"Downloading {len(selected_ids)} curated desk/tabletop photos...")
    for idx, photo_id in enumerate(selected_ids):
        out_file = out_dir / f"bg_desk_{idx:04d}.jpg"
        if not out_file.exists():
            url = f"https://images.unsplash.com/photo-{photo_id}?w={IMG_W}&h={IMG_H}&fit=crop&q=90"
            try:
                r = requests.get(url, headers=headers, timeout=20)
                if r.status_code == 200:
                    pil_img = Image.open(io.BytesIO(r.content))
                    final_img = crop_and_resize(pil_img, IMG_W, IMG_H)
                    final_img.save(str(out_file), "JPEG", quality=92)
                else:
                    logger.warning(f"Could not download photo {photo_id} (HTTP {r.status_code})")
                    continue
            except Exception as e:
                logger.warning(f"Error downloading photo {photo_id}: {e}")
                continue

        records.append({
            "filename": out_file.name,
            "source": "unsplash/curated_desks",
            "category": "desk/workspace",
            "path": str(out_file)
        })

    logger.info(f"  Curated Desks: {len(records)}/{len(selected_ids)} saved")
    return records

def main():
    args = parse_args()
    bg_dir = Path(args.bg_dir)
    bg_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    dtd_records = download_dtd_backgrounds(bg_dir, args.dtd_max, args.seed)
    all_records.extend(dtd_records)

    pbr_records = download_pbr_backgrounds(bg_dir, args.pbr_max, args.seed)
    all_records.extend(pbr_records)

    desk_records = download_curated_desks(bg_dir, args.desk_max)
    all_records.extend(desk_records)

    picsum_files = sorted(bg_dir.glob("bg_[0-9][0-9][0-9][0-9].jpg"))
    for pf in picsum_files:
        all_records.append({
            "filename": pf.name,
            "source": "picsum/generic",
            "category": "generic/photo",
            "path": str(pf)
        })

    sources_count = {}
    categories_count = {}
    for r in all_records:
        src = r["source"]
        cat = r["category"]
        sources_count[src] = sources_count.get(src, 0) + 1
        categories_count[cat] = categories_count.get(cat, 0) + 1

    manifest = {
        "total_backgrounds": len(all_records),
        "target_resolution": f"{IMG_W}x{IMG_H}",
        "sources_count": sources_count,
        "categories_count": categories_count,
        "items": all_records
    }

    manifest_path = bg_dir / "backgrounds_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Background manifest saved → {manifest_path}")
    logger.info(f"Total available background images: {len(all_records)}")
    for src, cnt in sources_count.items():
        logger.info(f"  - {src}: {cnt} images")

if __name__ == "__main__":
    main()

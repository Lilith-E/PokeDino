"""Download and partition Pokémon cards from the Hugging Face dataset TheFusion21/PokemonCards.

Features:
1. Loads dataset metadata using Hugging Face datasets.
2. Deterministic train/val/test partition (70% / 15% / 15%).
3. High-performance asynchronous downloading with concurrency control.
4. Auto-converts to optimized WebP format (quality 92, max 1024px) for minimal disk usage (~80KB/card).
5. Resumes automatically: skips already downloaded files.
6. Exports a comprehensive split_manifest.json.
"""

import argparse
import asyncio
import io
import json
import logging
import random
import time
from pathlib import Path

import aiohttp
from datasets import load_dataset
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "datasets" / "cards" / "pokemon_official_tcg"

def parse_args():
    parser = argparse.ArgumentParser(description="Download and partition Hugging Face PokemonCards dataset.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory for raw cards.")
    parser.add_argument("--max-cards", type=int, default=None, help="Maximum number of cards to download (None = all).")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio (default 0.70).")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val split ratio (default 0.15).")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio (default 0.15).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split.")
    parser.add_argument("--concurrency", type=int, default=20, help="Number of concurrent download tasks.")
    parser.add_argument("--webp-quality", type=int, default=92, help="WebP compression quality (1-100).")
    parser.add_argument("--max-dim", type=int, default=1024, help="Max dimension of image (aspect ratio preserved).")
    return parser.parse_args()

def assign_splits(items: list[dict], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> list[dict]:
    """Deterministically assign each card to train, val, or test partition."""
    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    for i, item in enumerate(shuffled):
        if i < n_train:
            item["split"] = "train"
        elif i < n_train + n_val:
            item["split"] = "val"
        else:
            item["split"] = "test"

    return shuffled

async def download_card(
    session: aiohttp.ClientSession,
    item: dict,
    out_dir: Path,
    semaphore: asyncio.Semaphore,
    max_dim: int = 1024,
    webp_quality: int = 92,
) -> bool:
    """Download a single card image, resize if necessary, and save as optimized WebP."""
    card_id = item["id"]
    split = item["split"]
    url = item["image_url"]

    safe_id = card_id.replace("/", "_").replace("\\", "_")
    target_path = out_dir / split / f"{safe_id}.webp"

    if target_path.exists() and target_path.stat().st_size > 1000:
        item["local_path"] = str(target_path)
        return True

    async with semaphore:
        for attempt in range(3):
            try:
                headers = {"User-Agent": "Mozilla/5.0 (compatible; PokemonCardResearch/1.0)"}
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status != 200:
                        if attempt == 2:
                            logger.warning(f"Failed to download {card_id} (HTTP {resp.status}): {url}")
                        await asyncio.sleep(1.0)
                        continue

                    raw_bytes = await resp.read()
                    if len(raw_bytes) < 500:
                        continue

                    img = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
                    w, h = img.size

                    if max(w, h) > max_dim:
                        scale = max_dim / float(max(w, h))
                        new_w = max(1, int(w * scale))
                        new_h = max(1, int(h * scale))
                        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    img.save(str(target_path), "WEBP", quality=webp_quality, method=4)

                    item["local_path"] = str(target_path)
                    return True
            except Exception as e:
                if attempt == 2:
                    logger.debug(f"Error downloading {card_id}: {e}")
                await asyncio.sleep(1.0)

    return False

async def download_all(items: list[dict], out_dir: Path, concurrency: int, max_dim: int, webp_quality: int):
    semaphore = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=concurrency * 2, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [
            download_card(
                session, item, out_dir, semaphore, max_dim=max_dim, webp_quality=webp_quality
            )
            for item in items
        ]

        total = len(tasks)
        completed = 0
        success = 0
        t0 = time.time()

        for f in asyncio.as_completed(tasks):
            res = await f
            completed += 1
            if res:
                success += 1
            if completed % 100 == 0 or completed == total:
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-3)
                logger.info(
                    f"Progress: {completed}/{total} ({completed / total * 100:.1f}%) - "
                    f"Success: {success} - Speed: {rate:.1f} cards/s"
                )

        return success

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    for s in ["train", "val", "test"]:
        (out_dir / s).mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset 'TheFusion21/PokemonCards' via datasets library...")
    ds = load_dataset("TheFusion21/PokemonCards", split="train")
    logger.info(f"Loaded {len(ds)} cards from Hugging Face dataset.")

    items = []
    for row in ds:
        url = row.get("image_url")
        if not url or not url.startswith("http"):
            continue
        items.append({
            "id": row.get("id"),
            "image_url": url,
            "name": row.get("name"),
            "hp": row.get("hp"),
            "set_name": row.get("set_name"),
            "caption": row.get("caption"),
        })

    if args.max_cards is not None and args.max_cards < len(items):
        logger.info(f"Limiting to first {args.max_cards} cards as requested.")
        items = items[:args.max_cards]

    items = assign_splits(
        items,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    counts = {"train": 0, "val": 0, "test": 0}
    for item in items:
        counts[item["split"]] += 1
    logger.info(f"Partitioned cards: Train={counts['train']}, Val={counts['val']}, Test={counts['test']}")

    logger.info(f"Starting asynchronous download (concurrency={args.concurrency}, webp_quality={args.webp_quality})...")
    success_count = asyncio.run(
        download_all(
            items,
            out_dir=out_dir,
            concurrency=args.concurrency,
            max_dim=args.max_dim,
            webp_quality=args.webp_quality,
        )
    )

    logger.info(f"Download completed: {success_count}/{len(items)} cards saved successfully.")

    manifest_path = out_dir / "split_manifest.json"
    manifest = {
        "dataset_name": "TheFusion21/PokemonCards",
        "total_requested": len(items),
        "total_downloaded": success_count,
        "split_counts": counts,
        "seed": args.seed,
        "cards": [
            {
                "id": it["id"],
                "name": it["name"],
                "set_name": it["set_name"],
                "split": it["split"],
                "local_path": it.get("local_path"),
            }
            for it in items
            if "local_path" in it
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved split manifest to {manifest_path}")

if __name__ == "__main__":
    main()

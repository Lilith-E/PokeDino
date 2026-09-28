"""Module for accessing Pokémon card catalog and official reference scans.

Supports tokenized text search (name, card number, expansion set, card ID)
across the local catalog and manages downloading and caching high-resolution scans.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional
import requests

logger = logging.getLogger(__name__)

POKEMON_TCG_API = "https://api.pokemontcg.io/v2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "reference_cache"
CATALOG_PATH = PROJECT_ROOT / "data" / "card_catalog.json"

_card_catalog: Optional[list[dict]] = None
_catalog_by_id: dict[str, dict] = {}

def _get_catalog() -> list[dict]:
    global _card_catalog, _catalog_by_id
    if _card_catalog is None:
        if CATALOG_PATH.exists():
            try:
                with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                    _card_catalog = json.load(f)
                _catalog_by_id = {c["id"]: c for c in _card_catalog if c.get("id")}
                logger.info(f"Local card catalog loaded: {len(_card_catalog)} cards indexed.")
            except Exception as e:
                logger.error(f"Error loading catalog: {e}")
                _card_catalog = []
        else:
            _card_catalog = []
    return _card_catalog

def search_cards(query: str, api_key: Optional[str] = None, page_size: int = 24) -> list[dict]:
    """Searches Pokémon cards with relaxed substring matching (name, number, set, ID).

    Queries the local catalog (20,444 English cards) supporting zero-padding ('017' -> '17'),
    set-total formats ('017/172'), and partial word matching.
    
    Args:
        query: Search query string
        api_key: Optional pokemontcg.io API key
        page_size: Maximum number of results to return
    """
    q = query.strip()
    if not q:
        return []

    catalog = _get_catalog()
    q = query.strip()
    if not q:
        return []

    raw_tokens = re.split(r'[\s/#,-]+', q.lower())
    tokens = [t.strip(' -#') for t in raw_tokens if t.strip(' -#')]
    if not tokens:
        return []

    name_tokens = [t for t in tokens if not t.isdigit()]
    num_tokens = [t for t in tokens if t.isdigit()]

    results = []

    if catalog:
        scored = []
        q_lower = q.lower()
        for c in catalog:
            name = (c.get("name") or "").lower()
            num = str(c.get("number") or "").lower()
            num_unpadded = num.lstrip("0")
            set_name = (c.get("set_name") or "").lower()
            cid = (c.get("id") or "").lower()
            full_text = f"{name} {num} {set_name} {cid}"

            if name_tokens:
                if not any(nt in full_text for nt in name_tokens):
                    continue

            score = 0
            matched_tokens = 0
            for t in tokens:
                t_unpadded = t.lstrip("0") if t.isdigit() else t
                if t.isdigit():
                    if t == num or t_unpadded == num_unpadded:
                        score += 260
                        matched_tokens += 1
                    elif t in cid or t in num:
                        score += 90
                        matched_tokens += 1
                else:
                    if t in name:
                        score += 130
                        matched_tokens += 1
                    elif t in set_name:
                        score += 60
                        matched_tokens += 1
                    elif t in cid:
                        score += 40
                        matched_tokens += 1

            if matched_tokens == len(tokens):
                score += 300
            elif matched_tokens >= len(tokens) - 1 and len(tokens) > 1:
                score += 100
            elif matched_tokens > 0:
                score += 30
            else:
                continue

            if q_lower == name:
                score += 400
            elif q_lower in name:
                score += 200
            elif name.startswith(tokens[0]):
                score += 60

            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [c for s, c in scored[:page_size]]

    if not results:
        results = _search_remote_api(q, api_key, page_size)

    formatted_cards = []
    for c in results:
        if "images" not in c and ("image_small" in c or "image_large" in c):
            formatted_cards.append({
                "id": c.get("id"),
                "name": c.get("name"),
                "number": c.get("number"),
                "set_name": c.get("set_name", ""),
                "set": {"name": c.get("set_name", "")},
                "images": {
                    "small": c.get("image_small"),
                    "large": c.get("image_large"),
                },
                "rarity": c.get("rarity", ""),
                "language": c.get("language", "en"),
            })
        else:
            set_obj = c.get("set", {})
            s_name = set_obj.get("name", "") if isinstance(set_obj, dict) else str(set_obj or "")
            c_copy = dict(c)
            c_copy["set_name"] = s_name or c.get("set_name", "")
            if "language" not in c_copy:
                c_copy["language"] = "en"
            formatted_cards.append(c_copy)

    return formatted_cards

def _search_remote_api(query: str, api_key: Optional[str], page_size: int) -> list[dict]:
    """Queries the remote pokemontcg.io API.
    
    Args:
        query: Search query string
        api_key: Optional API key
        page_size: Maximum results
    
    Returns:
        List of card dictionaries from the remote API
    """
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    if api_key:
        headers["X-Api-Key"] = api_key

    clean_q = re.sub(r'[^a-zA-Z0-9\s]', '', query).strip()
    search_query = f'name:"{clean_q}"'
    
    try:
        resp = requests.get(
            f"{POKEMON_TCG_API}/cards",
            headers=headers,
            params={"q": search_query, "pageSize": page_size},
            timeout=10,
        )
        if resp.status_code == 200:
            api_cards = resp.json().get("data", [])
            return api_cards
    except Exception as e:
        logger.warning(f"Remote API lookup failed for '{query}': {e}")
    
    return []

def get_card_by_id(card_id: str, api_key: Optional[str] = None) -> Optional[dict]:
    """Retrieves card metadata by unique ID (e.g. 'base5-21', 'base1-4')."""
    _get_catalog()
    if card_id in _catalog_by_id:
        c = _catalog_by_id[card_id]
        return {
            "id": c.get("id"),
            "name": c.get("name"),
            "number": c.get("number"),
            "set": {"name": c.get("set_name", "")},
            "images": {
                "small": c.get("image_small"),
                "large": c.get("image_large"),
            },
            "rarity": c.get("rarity", ""),
        }

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    if api_key:
        headers["X-Api-Key"] = api_key

    try:
        resp = requests.get(f"{POKEMON_TCG_API}/cards/{card_id}", headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("data")
    except Exception as e:
        logger.error(f"Error retrieving card by ID {card_id}: {e}")

    return None

def download_card_image(card: dict, resolution: str = "large", card_id: Optional[str] = None) -> Optional[str]:
    """Downloads reference scan from URL and saves to local disk cache."""
    cid = card_id or card.get("id") or "card"
    images = card.get("images") or {}
    img_url = (images.get(resolution) if isinstance(images, dict) else None) or \
              (images.get("large") if isinstance(images, dict) else None) or \
              (images.get("small") if isinstance(images, dict) else None) or \
              card.get("image_large") or card.get("image_small")

    if not img_url:
        logger.warning(f"No image URL found for card {cid}")
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ext = img_url.split(".")[-1].split("?")[0]
    if ext not in ("png", "jpg", "jpeg", "webp"):
        ext = "png"
    local_path = CACHE_DIR / f"{cid}.{ext}"

    if local_path.exists() and local_path.stat().st_size > 1000:
        return str(local_path)

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(img_url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()
        with open(str(local_path), "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Image downloaded successfully for {cid}: {local_path}")
        return str(local_path)
    except Exception as e:
        logger.error(f"Error downloading image {cid}: {e}")
        return None

def get_reference_images_for_card(
    card_id: str,
    api_key: Optional[str] = None,
    card_data: Optional[dict] = None,
) -> list[str]:
    """Retrieves local cached path or downloads reference scan for a card ID."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        cached_file = CACHE_DIR / f"{card_id}{ext}"
        if cached_file.exists() and cached_file.stat().st_size > 1000:
            logger.info(f"Card scan found in local cache: {cached_file}")
            return [str(cached_file)]

    card = card_data if (card_data and (card_data.get("images") or card_data.get("image_small") or card_data.get("image_large"))) else get_card_by_id(card_id, api_key)
    if not card:
        logger.error(f"Card {card_id} not found.")
        return []

    for res in ["large", "small"]:
        path = download_card_image(card, resolution=res, card_id=card_id)
        if path and Path(path).exists() and Path(path).stat().st_size > 1000:
            return [path]

    return []

def format_card_display(card: dict) -> dict:
    """Formats card dictionary into clean frontend representation."""
    images = card.get("images") or {}
    set_obj = card.get("set") or {}
    set_name = card.get("set_name") or (set_obj.get("name", "") if isinstance(set_obj, dict) else str(set_obj or ""))

    img_small = images.get("small") if isinstance(images, dict) else None
    if not img_small:
        img_small = card.get("image_small")

    img_large = images.get("large") if isinstance(images, dict) else None
    if not img_large:
        img_large = card.get("image_large")

    return {
        "id": card.get("id"),
        "name": card.get("name"),
        "number": str(card.get("number") or ""),
        "set_name": set_name,
        "rarity": card.get("rarity", ""),
        "image_small": img_small,
        "image_large": img_large,
    }

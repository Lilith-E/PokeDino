"""Unit tests for spatial-aware k-NN, adaptive dead zone, and morphology scoring.

Verifies algebraic correctness of spatial primitives in backend/spatial_morphology.py
and feature caches in backend/few_shot.py independently of neural weights.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.few_shot import (  # noqa: E402
    EXPECTED_VARIANT_NAMES,
    hash_mask,
    load_cached_features,
    save_cached_features,
)
from backend.spatial_morphology import (  # noqa: E402
    MorphologyConfig,
    SpatialSearchConfig,
    analyze_defect_morphology,
    build_patch_coordinates,
    compute_spatial_knn_distances,
    spatial_distance_matrix,
    spatial_penalty,
)

def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)

def test_build_patch_coordinates() -> None:
    coords = build_patch_coordinates((3, 4))
    _assert(coords.shape == (12, 2), f"shape inattesa: {coords.shape}")
    _assert(coords.dtype == np.float32, f"dtype inatteso: {coords.dtype}")
    _assert(tuple(coords[0]) == (0.0, 0.0), f"primo patch inatteso: {coords[0]}")
    _assert(tuple(coords[-1]) == (2.0, 3.0), f"ultimo patch inatteso: {coords[-1]}")
    print("  ✓ build_patch_coordinates")

def test_spatial_distance_matrix_is_symmetric() -> None:
    a = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    b = np.array([[3.0, 4.0]], dtype=np.float32)
    d = spatial_distance_matrix(a, b, radius=2.0)
    _assert(d.shape == (2, 1), f"shape: {d.shape}")
    _assert(abs(d[0, 0] - 5.0) < 1e-5, f"distanza euclidea errata: {d[0, 0]}")
    _assert(abs(d[1, 0] - np.sqrt(13.0)) < 1e-5, f"distanza euclidea errata: {d[1, 0]}")
    print("  ✓ spatial_distance_matrix")

def test_spatial_penalty_grows_with_distance() -> None:
    pen = spatial_penalty(np.array([[0.0, 2.0, 3.0, 10.0]], dtype=np.float32), radius=2.0, weight=0.4)
    _assert(pen[0, 0] == 0.0, f"pen[0] deve essere 0: {pen[0, 0]}")
    _assert(pen[0, 1] == 0.0, f"pen[1] entro radius deve essere 0: {pen[0, 1]}")
    _assert(0.0 < pen[0, 2] < 0.4, f"pen[2] deve essere intermedio: {pen[0, 2]}")
    _assert(abs(pen[0, 2] - 0.4 * (1.0 / 2.0)) < 1e-5, f"pen[2] attesa ~0.2: {pen[0, 2]}")
    _assert(abs(pen[0, 3] - 0.4) < 1e-6, f"pen[3] deve essere al massimo: {pen[0, 3]}")
    print("  ✓ spatial_penalty")

def test_compute_spatial_knn_penalizes_cross_region() -> None:
    """Un test patch identico ad un reference molto lontano deve essere penalizzato."""
    import torch
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    test_f = rng.randn(1, 8).astype(np.float32)
    ref_f = rng.randn(5, 8).astype(np.float32)
    ref_f[0] = test_f[0]

    coords_test = np.array([[0.0, 0.0]], dtype=np.float32)
    coords_ref = np.array([[10.0, 10.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)

    test_tensor = torch.nn.functional.normalize(torch.from_numpy(test_f), p=2, dim=1)
    ref_tensor = torch.nn.functional.normalize(torch.from_numpy(ref_f), p=2, dim=1)

    cfg_no = SpatialSearchConfig(enabled=False, radius_patches=3.0, penalty_weight=0.0)
    cfg_yes = SpatialSearchConfig(enabled=True, radius_patches=3.0, penalty_weight=0.9)

    d_no = compute_spatial_knn_distances(test_tensor, ref_tensor, coords_test, coords_ref, cfg_no)
    d_yes = compute_spatial_knn_distances(test_tensor, ref_tensor, coords_test, coords_ref, cfg_yes)
    _assert(d_no[0] < 1e-4, f"senza vincolo la distanza dovrebbe essere ~0: {d_no[0]}")
    _assert(d_yes[0] > 0.5, f"con vincolo la distanza deve essere penalizzata: {d_yes[0]}")
    print(f"  ✓ compute_spatial_knn_distances (no={d_no[0]:.4f}, yes={d_yes[0]:.4f})")


def test_morphology_detects_linear_scratch() -> None:
    grid_mask = np.ones((10, 10), dtype=bool)
    d_map = np.zeros((10, 10), dtype=np.float32)
    d_map[5, 1:7] = 0.6

    cfg = MorphologyConfig(enabled=True, large_cluster_threshold=4, cluster_penalty=0.05, linear_eccentricity=1.5, linear_penalty=0.1)
    morph = analyze_defect_morphology(d_map, grid_mask, defect_threshold=0.28, config=cfg)
    _assert(morph.cluster_count == 1, f"cluster atteso 1, ottenuto {morph.cluster_count}")
    _assert(morph.max_cluster_area == 6, f"area attesa 6, ottenuta {morph.max_cluster_area}")
    _assert(morph.linear_cluster_count == 1, f"cluster lineare atteso 1, ottenuto {morph.linear_cluster_count}")
    _assert(morph.morphology_bonus > 0.0, f"bonus deve essere positivo: {morph.morphology_bonus}")
    _assert(morph.morphology_bonus >= cfg.cluster_penalty * (6 - cfg.large_cluster_threshold) + cfg.linear_penalty - 1e-6,
            f"bonus non conforme: {morph.morphology_bonus}")
    print(f"  ✓ morphology (linear cluster bonus={morph.morphology_bonus:.4f})")

def test_morphology_ignores_scattered_dust() -> None:
    grid_mask = np.ones((10, 10), dtype=bool)
    d_map = np.zeros((10, 10), dtype=np.float32)
    pts = [(1, 1), (1, 8), (8, 1), (8, 8), (3, 4)]
    for r, c in pts:
        d_map[r, c] = 0.5

    cfg = MorphologyConfig(enabled=True, large_cluster_threshold=10, cluster_penalty=0.5, linear_eccentricity=3.0, linear_penalty=0.5)
    morph = analyze_defect_morphology(d_map, grid_mask, defect_threshold=0.28, config=cfg)
    _assert(morph.cluster_count == 5, f"cluster atteso 5, ottenuto {morph.cluster_count}")
    _assert(morph.linear_cluster_count == 0, f"nessun cluster lineare atteso, ottenuto {morph.linear_cluster_count}")
    _assert(morph.morphology_bonus == 0.0, f"bonus deve essere nullo per dust, ottenuto {morph.morphology_bonus}")
    print(f"  ✓ morphology (scattered dust bonus=0)")

def test_morphology_disabled_returns_zero() -> None:
    grid_mask = np.ones((10, 10), dtype=bool)
    d_map = np.zeros((10, 10), dtype=np.float32)
    d_map[5, 1:7] = 0.6
    cfg = MorphologyConfig(enabled=False)
    morph = analyze_defect_morphology(d_map, grid_mask, defect_threshold=0.28, config=cfg)
    _assert(morph.cluster_count == 0 and morph.morphology_bonus == 0.0, "disabilitato deve restituire zero")
    print("  ✓ morphology (disabled returns empty)")

def test_feature_cache_roundtrip() -> None:
    rng = np.random.RandomState(123)
    feats = rng.randn(100, 768).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        save_cached_features(td, "test-card", "base", (45, 32), feats)
        loaded = load_cached_features(td, "test-card", "base", (45, 32))
        _assert(loaded is not None, "cache not found after saving")
        _assert(loaded.shape == feats.shape, f"shape rotonda: {loaded.shape}")
        _assert(np.allclose(loaded, feats), "i dati roundtrip non coincidono")

        mismatched = load_cached_features(td, "test-card", "base", (32, 32))
        _assert(mismatched is None, "cache con grid_size diverso deve essere rifiutata")
        miss = load_cached_features(td, "other-card", "base", (45, 32))
        _assert(miss is None, "cache inesistente deve restituire None")
    print("  ✓ feature_cache_roundtrip")

def test_hash_mask_deterministic() -> None:
    rng = np.random.RandomState(7)
    m1 = (rng.rand(20, 20) > 0.5).astype(np.uint8)
    m2 = m1.copy()
    h1 = hash_mask(m1)
    h2 = hash_mask(m2)
    _assert(h1 == h2, "hash deve essere deterministico per input identici")
    m3 = (rng.rand(20, 20) > 0.5).astype(np.uint8)
    h3 = hash_mask(m3)
    _assert(h1 != h3 or m1.tolist() == m3.tolist(), "maschere diverse devono (quasi sempre) produrre hash diversi")
    h_none = hash_mask(None)
    _assert(h_none == "nomask", f"hash(None) deve essere 'nomask', ottenuto '{h_none}'")
    print("  ✓ hash_mask")

def test_few_shot_variants_count() -> None:
    """Verifica che il numero di varianti Few-Shot corrisponda alle 20 previste."""
    _assert(len(EXPECTED_VARIANT_NAMES) == 20, f"attese 20 varianti, trovate {len(EXPECTED_VARIANT_NAMES)}")
    expected_keywords = [
        "warm_tungsten", "cool_fluorescent", "iso_sensor_noise",
        "defocus", "motion_blur", "vignette", "flash",
        "jpeg", "bilateral", "shadow", "window_light",
        "underexposed", "overexposed", "hdr", "washed",
        "vivid", "chromatic", "holo", "strong", "blowout"
    ]
    for name, kw in zip(EXPECTED_VARIANT_NAMES, expected_keywords):
        _assert(kw in name, f"variante {name} non contiene keyword '{kw}'")
    print("  ✓ few_shot_variants_count")

def main() -> None:
    print("\n=== Test spatial_morphology + few_shot.cache ===\n")
    test_build_patch_coordinates()
    test_spatial_distance_matrix_is_symmetric()
    test_spatial_penalty_grows_with_distance()
    test_compute_spatial_knn_penalizes_cross_region()
    test_morphology_detects_linear_scratch()
    test_morphology_ignores_scattered_dust()
    test_morphology_disabled_returns_zero()
    test_feature_cache_roundtrip()
    test_hash_mask_deterministic()
    test_few_shot_variants_count()
    print("\n Tutti i test sono passati.\n")

if __name__ == "__main__":
    main()
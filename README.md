# PokéDINO

> **Automated Visual Inspection and Condition Grading for Collectible Trading Cards**  
> *A Deep Learning & Computer Vision System*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Ultralytics](https://img.shields.io/badge/YOLO11-OBB-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Vision](https://img.shields.io/badge/DINOv2-ViT--B%2F14-FF6F00.svg)](https://github.com/facebookresearch/dinov2)

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#1-prerequisites)
  - [Environment Setup](#2-environment-setup)
  - [Model Weights](#3-model-weights)
  - [Running the Web Demo](#4-running-the-web-demo)
- [Evaluation & Benchmarks](#evaluation--benchmarks)
- [Technical Report & Documentation](#technical-report--documentation)
- [References](#references)

---

## Overview

PokéDINO is an end-to-end computer vision and deep learning system designed for the automated localization, rectification, anomaly detection, and condition grading of collectible trading cards from unconstrained camera captures.

By combining oriented bounding box localization (YOLO11s-OBB) with analytical edge refinement (Physical Boundary Locking, PBL), two-stage image quality screening, local contrast normalization (CLAHE), and self-supervised feature embeddings (DINOv2 ViT-B/14), the pipeline detects surface scratches, edge whitening, creases, and centering defects, mapping cards onto a calibrated 3-tier condition scale (*Near Mint/Mint*, *Good/Excellent*, *Poor/Played*).

---

## System Architecture

```
                            Camera Capture (Unconstrained Scene)
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ Stage 1: Quick-Reject Image Quality Gate     │
                    │   • Laplacian variance blur screening         │
                    │   • Global intensity std-dev filter           │
                    └───────────────────────┬───────────────────────┘
                                            │ (Pass)
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ Stage 2: Oriented Card Localization (YOLO OBB)│
                    │   • YOLO11s-OBB with Test-Time Augmentation   │
                    │   • Perspective rectification to 630×448 px   │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ Stage 3: Physical Boundary Locking (PBL)     │
                    │   • Morphological edge refinement             │
                    │   • Sub-pixel corner & border snap            │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ Stage 4: Stage-2 Detailed Quality Gate        │
                    │   • Sharpness, glare, exposure, noise scoring │
                    └───────────────────────┬───────────────────────┘
                                            │ (Score ≥ 0.35)
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ Stage 5: Photometric Preprocessing & Masking  │
                    │   • CLAHE local contrast normalization        │
                    │   • Rounded-corner feathered geometric mask   │
                    └───────────────────────┬───────────────────────┘
                                            │
                     Query Crop (630×448)   │   Reference Memory Bank
                                            ▼       ▼
                    ┌───────────────────────────────────────────────┐
                    │ Stage 6: Anomaly Inspection (DINOv2 ViT-B/14) │
                    │   • Few-shot photometric memory bank          │
                    │   • Spatially constrained k-NN patch matching │
                    │   • Dead-zone optical noise filtering         │
                    │   • Gamma-corrected anomaly distance map      │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ Stage 7: Defect Morphology & Condition Grade  │
                    │   • Connected-component cluster scoring       │
                    │   • Consolidated 3-tier grading scale:        │
                    │     - Near Mint / Mint (NM/M)                 │
                    │     - Good / Excellent (GD/EX)                │
                    │     - Poor / Played (PL/PO)                   │
                    └───────────────────────────────────────────────┘
```

---

## Repository Structure

```
.
├── configs/
│   └── default.yaml             # Centralized pipeline configuration
│
├── data/
│   ├── card_catalog.json        # Offline Pokémon TCG metadata catalog (>20k cards)
│   ├── test_charizard_damaged.png # Sample test image for quick inference
│   ├── datasets/                # Dataset directory placeholder (.gitkeep)
│   ├── weights/                 # Model checkpoints placeholder (.gitkeep)
│   ├── reference_cache/         # DINOv2 feature cache placeholder (.gitkeep)
│   └── uploads/                 # Runtime image upload folder (.gitkeep)
│
├── ReportFinale.pdf             # Compiled comprehensive academic report document (PDF)
│
├── scripts/                     # Scientific evaluation, calibration & benchmark scripts
│   ├── benchmark_3way_comparison.py     # 3-way corner error benchmark: Raw YOLO vs Fine-tuned vs PBL
│   ├── benchmark_corner_error.py        # Quantitative corner error distribution
│   ├── benchmark_pbl_pipeline.py        # Paired Wilcoxon validation of PBL border locking
│   ├── benchmark_rotation_invariance.py # 360-degree rotation invariance assessment
│   ├── calibrate_threshold.py           # Anomaly decision threshold statistical calibration
│   ├── download_hf_backgrounds.py       # Background asset downloader (Hugging Face)
│   ├── download_hf_cards.py             # Card scan downloader (Hugging Face / TCG API)
│   ├── eval_clahe.py                    # CLAHE local contrast enhancement evaluation
│   ├── eval_damage.py                   # Synthetic defect detection evaluation
│   ├── eval_e2e.py                      # End-to-end pipeline evaluation
│   ├── eval_few_shot.py                 # Photometric few-shot memory bank evaluation
│   ├── eval_masking.py                  # Geometric rounded mask ablation
│   ├── eval_obb.py                      # YOLO11s-OBB detection mAP evaluation
│   ├── finetune_obb.py                  # Edge-focused fine-tuning for YOLO OBB
│   ├── prepare_dataset.py               # Synthetic scene generator & dataset preparation
│   ├── test_ablation_render.py          # Visual ablation inspection test
│   ├── test_condition.py                # 3-tier condition grading threshold test
│   ├── test_dinov2_comparison.py        # Standard DINOv2 vs Register token comparison
│   ├── test_spatial_morphology.py       # Spatial morphology and scratch cluster test
│   └── train_obb.py                     # Initial training of YOLO11s-OBB on synthetic data
│
├── src/
│   ├── backend/                         # Core Python modules
│   │   ├── card_detector.py             # YOLO OBB + TTA + PBL detector
│   │   ├── condition.py                 # Condition grading logic & thresholds
│   │   ├── damage.py                    # Procedural defect synthesis
│   │   ├── detector.py                  # DINOv2 anomaly engine & memory bank
│   │   ├── experiment.py                # Configuration loader & logging utilities
│   │   ├── few_shot.py                  # Deterministic photometric variant generator
│   │   ├── main.py                      # FastAPI web server & endpoints
│   │   ├── masking.py                   # Rounded-corner feathered mask generator
│   │   ├── pokemon_api.py               # Card catalog client & reference fetcher
│   │   ├── preprocessing.py             # CLAHE & luminance alignment
│   │   ├── spatial_morphology.py        # Defect clustering & scratch eccentricity
│   │   └── quality_gate/                # Two-stage image quality gate
│   ├── frontend/
│   │   └── index.html                   # Single-page interactive web interface
│   ├── anomalydino_src/                 # Third-party AnomalyDINO reference implementation
│   └── requirements.txt                 # Backend dependency list
│
├── .gitignore                   # Excludes heavy datasets, checkpoints, and caches
├── requirements.txt             # Pinned project dependencies
└── run_server.py                # Primary entry point: starts FastAPI web demo
```

---

## Getting Started

### 1. Prerequisites

- Python 3.11+
- Compatible with macOS (Apple Silicon MPS), Linux (CUDA), and CPU fallback.

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Lilith-E/PokeDino.git
cd PokeDino

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Model Weights

The pipeline utilizes two model checkpoints expected in `data/weights/`:
- `yolo11s_obb_best.pt`: Base detector trained on synthetic card scenes.
- `yolo11s_obb_finetuned_best.pt`: Fine-tuned detector with edge-preserving loss weighting.

To train the detector from scratch:
```bash
python scripts/train_obb.py
```

To fine-tune with boundary-focused weighting:
```bash
python scripts/finetune_obb.py
```

### 4. Running the Web Demo

Launch the interactive FastAPI application from the project root:

```bash
python run_server.py
```

Once started, access the service via:
- **Interactive Web Interface**: `http://localhost:8000/app/`
- **Interactive API Documentation (Swagger)**: `http://localhost:8000/docs`

You can upload a card photograph or test using the sample provided in `data/test_charizard_damaged.png` to inspect the detected bounding box, rectified patch, DINOv2 anomaly heatmap, and predicted condition grade.

---

## Evaluation & Benchmarks

All evaluation and benchmark scripts use relative path resolution and can be run directly from the command line:

```bash
# Evaluate YOLO OBB detection accuracy
python scripts/eval_obb.py

# Benchmark Physical Boundary Locking (PBL) corner accuracy
python scripts/benchmark_corner_error.py

# Run paired Wilcoxon statistical test for border refinement
python scripts/benchmark_pbl_pipeline.py

# Evaluate rotation invariance across 360 degrees
python scripts/benchmark_rotation_invariance.py

# Run end-to-end validation on sample scenes
python scripts/eval_e2e.py

# Calibrate anomaly threshold on pristine samples
python scripts/calibrate_threshold.py
```

---

## Technical Report & Documentation

- **Compiled PDF**: The complete academic thesis and technical report is available directly in [`ReportFinale.pdf`](ReportFinale.pdf).

---

## References

1. Jocher, G., & Qiu, J. (2024). *Ultralytics YOLO11*. [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
2. Oquab, M., et al. (2024). *DINOv2: Learning Robust Visual Features without Supervision*. ICLR 2024.
3. Damm, S., Laszkiewicz, M., Lederer, J., & Fischer, A. (2025). *AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2*. WACV 2025.
4. Zuiderveld, K. (1994). *Contrast Limited Adaptive Histogram Equalization*. Graphics Gems IV.

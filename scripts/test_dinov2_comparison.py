#!/usr/bin/env python3
"""Comparative evaluation between DINOv2 ViT-B/14 and ViT-B/14 with registers.

Evaluates performance differences between:
- dinov2_vitb14 (standard, without registers)
- dinov2_vitb14_reg (with register tokens)

Generates comparative heatmaps and outputs results to data/dinov2_comparison/.
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.detector import PokemonAnomalyDetector, heatmap_overlay
from backend.pokemon_api import get_reference_images_for_card

def create_side_by_side_comparison(img1, img2, title1, title2, output_path):
    """Crea un'immagine side-by-side con titoli"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    max_h = max(h1, h2)
    max_w = max(w1, w2)
    
    canvas = np.ones((max_h + 40, max_w * 2 + 20, 3), dtype=np.uint8) * 255
    
    canvas[40:40+h1, 0:w1] = img1
    canvas[40:40+h2, max_w+20:max_w+20+w2] = img2
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title1, (10, 25), font, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, title2, (max_w+30, 25), font, 0.7, (0, 0, 0), 2)
    
    cv2.imwrite(str(output_path), canvas)
    print(f" Salvato: {output_path}")

def test_card_comparison(card_id, test_image_path, output_dir):
    """Testa una carta con entrambi i modelli DINOv2"""
    
    print(f"\n{'='*80}")
    print(f"Testing card: {card_id}")
    print(f"{'='*80}\n")
    
    print(f"📥 Downloading reference image for {card_id}...")
    ref_images = get_reference_images_for_card(card_id)
    if not ref_images:
        print(f" Errore: impossibile scaricare reference per {card_id}")
        return
    
    ref_image_path = ref_images[0]
    print(f" Reference scaricata: {ref_image_path}")
    
    if not os.path.exists(test_image_path):
        print(f" Errore: immagine di test non trovata: {test_image_path}")
        return
    
    test_img_bgr = cv2.imread(test_image_path)
    if test_img_bgr is None:
        print(f" Errore: impossibile caricare {test_image_path}")
        return
    
    test_img = cv2.cvtColor(test_img_bgr, cv2.COLOR_BGR2RGB)
    
    print(f" Immagine di test caricata: {test_image_path}")
    
    print("\n🔬 Testing with dinov2_vitb14 (standard)...")
    detector_standard = PokemonAnomalyDetector(
        model_name="dinov2_vitb14",
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )
    detector_standard.load_reference_images([ref_image_path])
    result_standard = detector_standard.analyze_card(test_image_path)
    
    score_standard = result_standard['score']
    condition_standard = result_standard['condition']
    
    test_img_processed = detector_standard._last_patch
    
    heatmap_standard = heatmap_overlay(
        test_img_processed,
        detector_standard._last_d_map,
        valid_mask=detector_standard._last_mask,
        dead_zone=detector_standard.dead_zone,
        gamma=detector_standard.gamma,
        adaptive_alpha=True,
        bilateral_smoothing=True,
    )
    
    print(f"  Score: {score_standard:.4f}")
    print(f"  Condition: {condition_standard}")
    
    print("\n🔬 Testing with dinov2_vitb14_reg (with registers)...")
    detector_reg = PokemonAnomalyDetector(
        model_name="dinov2_vitb14_reg",
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )
    detector_reg.load_reference_images([ref_image_path])
    result_reg = detector_reg.analyze_card(test_image_path)
    
    score_reg = result_reg['score']
    condition_reg = result_reg['condition']
    
    test_img_processed_reg = detector_reg._last_patch
    
    heatmap_reg = heatmap_overlay(
        test_img_processed_reg,
        detector_reg._last_d_map,
        valid_mask=detector_reg._last_mask,
        dead_zone=detector_reg.dead_zone,
        gamma=detector_reg.gamma,
        adaptive_alpha=True,
        bilateral_smoothing=True,
    )
    
    print(f"  Score: {score_reg:.4f}")
    print(f"  Condition: {condition_reg}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    heatmap_standard_path = output_dir / f"{card_id}_standard_heatmap.png"
    cv2.imwrite(str(heatmap_standard_path), cv2.cvtColor(heatmap_standard, cv2.COLOR_RGB2BGR))
    print(f" Heatmap standard salvata: {heatmap_standard_path}")
    
    heatmap_reg_path = output_dir / f"{card_id}_reg_heatmap.png"
    cv2.imwrite(str(heatmap_reg_path), cv2.cvtColor(heatmap_reg, cv2.COLOR_RGB2BGR))
    print(f" Heatmap reg salvata: {heatmap_reg_path}")
    
    comparison_path = output_dir / f"{card_id}_comparison.png"
    create_side_by_side_comparison(
        cv2.cvtColor(heatmap_standard, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(heatmap_reg, cv2.COLOR_RGB2BGR),
        f"Standard (score: {score_standard:.4f})",
        f"With Registers (score: {score_reg:.4f})",
        comparison_path
    )
    
    report_path = output_dir / f"{card_id}_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Test comparativo DINOv2 per carta: {card_id}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write("DINOv2 Standard (dinov2_vitb14):\n")
        f.write(f"  Score: {score_standard:.4f}\n")
        f.write(f"  Condition: {condition_standard}\n")
        f.write(f"  Peak Score: {result_standard.get('peak_score', 'N/A')}\n")
        f.write(f"  Defect Area: {result_standard.get('defect_area_pct', 'N/A')}%\n\n")
        
        f.write("DINOv2 with Registers (dinov2_vitb14_reg):\n")
        f.write(f"  Score: {score_reg:.4f}\n")
        f.write(f"  Condition: {condition_reg}\n")
        f.write(f"  Peak Score: {result_reg.get('peak_score', 'N/A')}\n")
        f.write(f"  Defect Area: {result_reg.get('defect_area_pct', 'N/A')}%\n\n")
        
        f.write(f"Differenza score: {abs(score_reg - score_standard):.4f}\n")
        f.write(f"Miglioramento: {'' if score_reg > score_standard else ''}\n")
    
    print(f" Report salvato: {report_path}")
    
    return {
        'card_id': card_id,
        'standard': {
            'score': score_standard,
            'condition': condition_standard,
            'peak_score': result_standard.get('peak_score'),
            'defect_area_pct': result_standard.get('defect_area_pct')
        },
        'reg': {
            'score': score_reg,
            'condition': condition_reg,
            'peak_score': result_reg.get('peak_score'),
            'defect_area_pct': result_reg.get('defect_area_pct')
        }
    }

def main():
    """Test principale"""
    print("\n" + "="*80)
    print("TEST COMPARATIVO DINOv2: Standard vs With Registers")
    print("="*80 + "\n")
    
    output_dir = PROJECT_ROOT / "data" / "dinov2_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_cases = [
        {
            'card_id': 'base1-4',
            'test_image': str(PROJECT_ROOT / "data" / "test_charizard_damaged.png")
        }
    ]
    
    results = []
    for test_case in test_cases:
        result = test_card_comparison(
            test_case['card_id'],
            test_case['test_image'],
            output_dir
        )
        if result:
            results.append(result)
    
    if results:
        final_report_path = output_dir / "FINAL_REPORT.txt"
        with open(final_report_path, 'w') as f:
            f.write("REPORT FINALE - TEST COMPARATIVO DINOv2\n")
            f.write("="*80 + "\n\n")
            
            f.write("Configurazione:\n")
            f.write("  - Modello 1: dinov2_vitb14 (standard)\n")
            f.write("  - Modello 2: dinov2_vitb14_reg (con register tokens)\n")
            f.write("  - Migliorie implementate:\n")
            f.write("    • Adaptive alpha based on defect severity\n")
            f.write("    • Bilateral filter per smoothing percettivo\n")
            f.write("    • Soglie di classificazione ricalibrate\n")
            f.write("    • Morphology bonus ridotto\n\n")
            
            f.write("Risultati:\n")
            f.write("-"*80 + "\n")
            
            for result in results:
                f.write(f"\nCarta: {result['card_id']}\n")
                f.write(f"  Standard:     score={result['standard']['score']:.4f}, condition={result['standard']['condition']}\n")
                f.write(f"  With Reg:     score={result['reg']['score']:.4f}, condition={result['reg']['condition']}\n")
                f.write(f"  Differenza:   {abs(result['reg']['score'] - result['standard']['score']):.4f}\n")
                f.write(f"  Miglioramento: {'' if result['reg']['score'] > result['standard']['score'] else ''}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("File generati:\n")
            f.write(f"  - Heatmap comparative: {output_dir}/*.png\n")
            f.write(f"  - Report individuali: {output_dir}/*_report.txt\n")
            f.write(f"  - Report finale: {final_report_path}\n")
        
        print(f"\n{'='*80}")
        print(f" TEST COMPLETATO")
        print(f"{'='*80}")
        print(f"Report finale: {final_report_path}")
        print(f"Immagini comparative: {output_dir}/*.png")
        print(f"\nRiepilogo:")
        for result in results:
            print(f"  {result['card_id']}: {result['standard']['score']:.4f} → {result['reg']['score']:.4f}")

if __name__ == "__main__":
    main()

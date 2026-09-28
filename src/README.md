# PokéDINO — Source Modules

Questo modulo contiene il codice sorgente dell'applicazione PokéDINO per l'ispezione visiva e la gradazione automatica delle carte collezionabili.

---

## Architettura dei Moduli (`src/backend`)

1. **`card_detector.py`**: Rilevamento della carta tramite YOLO11s-OBB, Test-Time Augmentation (TTA), Physical Boundary Locking (PBL) e raddrizzamento prospettico sul formato canonico verticale (630×448 px).
2. **`quality_gate/`**: Validazione a due stadi: Quick-Reject sul fotogramma grezzo (nitidezza Laplaciano e deviazione standard) e Quality Gate approfondito sul crop (esposizione, riflessi, rumore, proporzioni).
3. **`preprocessing.py`**: Normalizzazione fotometrica e bilanciamento locale del contrasto tramite CLAHE (Contrast Limited Adaptive Histogram Equalization).
4. **`masking.py`**: Generazione della maschera geometrica perimetrale ad angoli arrotondati con bordo sfumato (feathering) per isolare la carta dal contesto di sfondo.
5. **`detector.py`**: Modulo di anomaly detection basato su DINOv2 ViT-B/14, memory bank fotometrica Few-Shot e One-Shot, ricerca k-NN con vincolo di prossimità spaziale, sogliatura Dead Zone e correzione gamma della mappa di anomalia.
6. **`spatial_morphology.py`**: Analisi morfologica dei difetti, clustering spaziale delle componenti connesse e calcolo dell'eccentricità per l'individuazione di graffi lineari.
7. **`condition.py`**: Calcolo dello score composito di anomalia e classificazione nella scala di conservazione consolidata a tre fasce (*Near Mint/Mint*, *Good/Excellent*, *Poor/Played*).
8. **`damage.py`**: Generazione procedurale di difetti sintetici (graffi, pieghe, usura dei bordi) per la validazione quantitativa della pipeline.
9. **`few_shot.py`**: Generazione deterministica di varianti fotometriche per la costruzione della memory bank di riferimento.
10. **`experiment.py`**: Gestione centralizzata degli esperimenti, tracciamento delle metriche, salvataggio dei grafici e risoluzione dei percorsi di configurazione.
11. **`pokemon_api.py`**: Gestione del catalogo offline (>20.000 carte) e download/caching delle scansioni ufficiali di riferimento ad alta risoluzione.
12. **`main.py`**: Server web FastAPI con API RESTful per upload immagini, inferenza asincrona e serving dell'interfaccia frontend.

---

## Frontend Web (`src/frontend`)

Interfaccia grafica interattiva a pagina singola (`index.html`) per:
- Caricamento di immagini da fotocamera o file system.
- Visualizzazione in tempo reale del bounding box orientato e del ritaglio raddrizzato.
- Ispezione visiva della mappa termica delle anomalie e dei singoli difetti.
- Visualizzazione dei punteggi di qualità, metriche di integrità e grado di conservazione stimato.

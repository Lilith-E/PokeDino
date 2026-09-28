"""Integration tests for card condition classification.

Verifies correct classification across the three consolidated tiers
(Near Mint/Mint, Good/Excellent, Poor/Played) as a function of anomaly
scores and calibrated thresholds.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.experiment import Experiment, load_config, set_seed  # noqa: E402
from backend.condition import classify_condition, condition_figure  # noqa: E402

def main():
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    exp = Experiment("phase2c_condition", cfg)

    bands = cfg["condition"]["bands"]
    labels = cfg["condition"]["labels"]
    exp.logger.info(f"bands={bands}")

    cases = [
        (0.10, "Near Mint/Mint"),
        (0.39, "Near Mint/Mint"),
        (0.45, "Good/Excellent"),
        (0.69, "Good/Excellent"),
        (0.75, "Poor/Played"),
        (1.45, "Poor/Played"),
    ]

    n_pass = 0
    for score, expected in cases:
        res = classify_condition(score, bands)
        ok = res.label == expected
        n_pass += ok
        exp.logger.info(
            f"score={score:.2f} → {res.label} ({'OK' if ok else 'FAIL, expected ' + expected}) "
            f"band=[{res.band_low}, {res.band_high}] color={res.color}"
        )
        exp.log_metric(f"case_{score:.2f}_{expected}", 1 if ok else 0)

    edges = [bands[lab] for lab in labels[:-1]]
    mono = all(a < b for a, b in zip(edges, edges[1:]))
    exp.log_metric("bands_monotonic", 1 if mono else 0)

    fig = condition_figure(0.50, classify_condition(0.50, bands), bands=bands)
    exp.save_plot(fig, "condition_bands")

    accuracy = n_pass / len(cases)
    exp.log_metric("n_cases", len(cases))
    exp.log_metric("accuracy", accuracy)
    exp.finish()
    exp.logger.info(f"DONE. {n_pass}/{len(cases)} cases pass, bands_monotonic={mono}")

    if accuracy < 1.0 or not mono:
        sys.exit(1)

if __name__ == "__main__":
    main()

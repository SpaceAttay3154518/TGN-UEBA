# Role-Anchored Update Authorisation

**Defending Temporal Graph Intrusion Detection Against Slow Memory Drift**

This repository contains the source code, configuration, and paper for the role-anchored update authorisation defence — a zero-parameter complementary layer that detects and mitigates slow memory drift attacks against Temporal Graph Networks (TGN) used for intrusion detection.

## Repository Structure

```
├── paper/                  # LaTeX manuscript and figures
│   ├── main.tex           # Main paper source
│   ├── main.pdf           # Compiled paper (22 pages)
│   ├── references.bib     # Bibliography
│   └── figures/           # All figures (TikZ-generated + external)
│
├── src/emdd/              # Core Python package
│   ├── model.py           # Adapted TGN/KAIROS architecture
│   ├── defense.py         # Role-anchored defence implementation
│   ├── attack.py          # Slow-conditioning attack construction
│   ├── study.py           # CADETS single-target evaluation
│   ├── slow_conditioning_study.py  # Slow-conditioning experiments
│   ├── multi_target_study.py       # Multi-target CERT evaluation
│   ├── longitudinal_study.py       # Longitudinal study driver
│   ├── evaluation.py      # Metrics and evaluation utilities
│   ├── io.py              # Data loading and checkpointing
│   ├── cli.py             # Command-line interface
│   └── ...                # Supporting modules
│
├── config/                # Experiment configurations
│   ├── cert_r42_longitudinal.json  # CERT r4.2 multi-target config
│   ├── cert_r42.json               # CERT r4.2 base config
│   └── ...
│
└── experiments/           # Experiment protocols and scripts
    └── slow_drift_defense/
        ├── PROTOCOL.md    # Experimental protocol
        ├── EVIDENCE_LEDGER.md
        └── run_multi_target.sh
```

## Key Contributions

1. **Slow-conditioning attack**: A two-phase attack that injects sub-threshold interactions to gradually drift TGN memory, suppressing anomaly scores on subsequent malicious payloads.

2. **Role-anchored defence**: Five drift signals (role displacement, innovation, directional persistence, trajectory coherence, event surprise) monitored via CUSUM to detect and quarantine conditioning attempts.

3. **Difference-in-differences evaluation**: A four-branch causal design isolating the defence's contribution from confounds.

4. **Multi-target validation**: 6,650+ case-level evaluations across CERT r4.2 (70 insiders, 3 scenarios, 5 model seeds).

## Datasets

- **DARPA TC CADETS** (E3): Provenance graph from the DARPA Transparent Computing program
- **CERT r4.2**: CMU synthetic insider threat corpus (1,000 users, 15.6M events, 70 labelled insiders)

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- PyTorch Geometric
- pandas, numpy, scikit-learn

## License

This project is released for academic research purposes.

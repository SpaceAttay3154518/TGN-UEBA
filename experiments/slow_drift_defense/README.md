# Slow-drift attack and defense study

This folder is the canonical workspace for the corrected attack and defense
evaluation. It does not replace the completed MemFreezing discrimination test.
MemFreezing tests one-shot stable-state degradation. This study tests gradual,
identity-preserving, payoff-directed conditioning.

The study follows this order:

1. define realizable slow-drift attacks and their success criteria;
2. replay plain KAIROS and establish whether it misses the conditioning phase
   and whether the later payoff is suppressed;
3. tune the defense on development cases only;
4. freeze one complete configuration;
5. run four matched branches on held-out cases;
6. compare attack recovery and clean false-positive cost;
7. generate machine-readable tables and figures before updating the paper.

## Current claim status

The frozen official-KAIROS track establishes pre-payload blindness at all four
tested rates and a zero-clean-alert defense result at 20 interactions per hour.
The same monitor family misses the other three rates. The corrected 14-day CERT
development probe also alerts on clean activity before it alerts on conditioning
and therefore fails the prespecified gate. The reserved CERT incidents remain
unopened. The current defensible conclusion is a scoped feasibility result plus
a negative generalization result, not a universal slow-drift defense.

No model weights are trained or fine-tuned in this folder. The experiment uses
the five frozen adapted-KAIROS CERT checkpoints in `../../codebase/checkpoints/`.
A separate frozen-checkpoint control uses the authors' released CADETS E3
checkpoint and records that result as an external-validity experiment rather
than pretending that CADETS weights can encode CERT entities.

## Evidence states

- `PROTOCOL.md` is the design and claim boundary.
- `config.json` fixes the case split and candidate grid.
- `results/development/` may be used for selection.
- `results/test/` is written only after a configuration is frozen.
- `results/aggregate/` contains paper-ready tables, statistics, and figures.

An attack is not called an evasion merely because its mean loss decreases. A
valid report distinguishes three outcomes: a hidden conditioning phase, score
suppression, and an actual final decision flip.

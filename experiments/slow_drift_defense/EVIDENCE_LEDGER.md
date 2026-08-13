# Evidence ledger

Last updated: 2026-08-11

This file separates reproduced facts, exploratory results, pending tests, and
invalidated artifacts. It is the claim-control source for the paper.

## Reproduced baseline

The authors' frozen `cadets3_models.pt` checkpoint reproduces the published
KAIROS CADETS E3 result under the released daily-reset, 1024-event-batch,
15-minute queue protocol:

- true positives: 4;
- true negatives: 174;
- false positives: 1;
- false negatives: 0;
- precision: 0.8000;
- recall: 1.0000;
- F1: 0.8889;
- accuracy: 0.9944;
- hard-label ROC-AUC: 0.9971.

This is the only result that is directly comparable with the published KAIROS
CADETS E3 table. The adapted CERT checkpoints are a separate transfer study.

## Attack result established with frozen official KAIROS

The slow-conditioning stressor repeatedly applies a clean-validation-observed
`EVENT_RECVFROM` interaction between the real CADETS process `vUgefal` and the
observed support `128.55.12.10:53`. The conditioning phase starts before the
official malicious payoff. Original events retain their original KAIROS batch
and window identities.

At 1, 4, 20, and 100 interactions per hour:

- released KAIROS alerts zero of the 43 strictly pre-payload windows;
- 91.67, 97.78, 99.55, and 99.82 percent of injected events are below the
  clean-validation q99 event threshold;
- mean specific-payoff cross-entropy is lower by approximately 0.00350 to
  0.00356;
- all four later official payload windows remain detected.

The supported claim is therefore pre-payload blindness and payoff-score
suppression. Full payload evasion is not supported.

## Exploratory defense result

The grid-selected scoped result uses the frozen official KAIROS weights,
maximum positive evidence, a 30-minute inactivity-reset CUSUM, a threshold
equal to the maximum clean-validation CUSUM, daily-anchor rollback, and
memory-update quarantine until the daily reset. For the
20-interaction-per-hour stressor:

- the defense alerts once on `vUgefal` before the payload;
- 677 of 833,764 branch updates are quarantined, including 54.71 percent of
  synthetic conditioning updates after the alert;
- the clean paired branch produces zero defense alerts and zero quarantines;
- a separate clean day of 2,284,034 events produces zero defense alerts and
  zero quarantines;
- KAIROS has zero alerted windows on the clean day both with and without the
  defense;
- the specific-payoff recovery difference in differences is +0.13538, with a
  paired-event bootstrap interval of [+0.11251, +0.15949];
- the target-process recovery difference in differences is +0.23301, with a
  paired-event bootstrap interval of [+0.19417, +0.27160];
- the KAIROS queue alert count is unchanged in every paired branch.

This result is exploratory because the attack and clean day were inspected
during development. It proves that the mechanism can work for one frozen
stress condition. It is not a general or confirmatory robustness result.

## Exploratory ablation result

The fixed-trace ablation compares three evidence formulas, three sequential
monitors, three inactivity resets, and four clean-validation threshold
quantiles, for 108 candidates. Under the constraint of zero alerted nodes on
the inspected clean day, every feasible candidate detects only the
20-interaction-per-hour attack. The 1, 4, and 100 rates are missed. This rate
and batching sensitivity is a limitation, not a success.

## MemFreezing discrimination result

The completed CERT discrimination test uses 19 answer-key events and three
independently initialized adapted models, giving 57 paired model-event scores.
The solved target is reached at mean cosine similarity 0.9998. Mean anomaly
increases from 0.5758 to 0.7193, a difference of +0.1435. In this setting,
MemFreezing sensitizes the labelled payoff rather than normalizing it. This
distinguishes generic persistent disruption from payoff-directed slow
conditioning. It does not validate the proposed defense.

## Exact-current-transition CERT development result

The corrected 14-day development probe uses CERT case `s3:BBS0039`, seed 17,
and 140 conditioning events per policy. Role-common and role-diverse attacks
are fully below the clean q99 threshold, but their payoff changes are only
+0.000013 and -0.000218. The payoff-type-matched grammar has a larger payoff
change of -0.008469, with a descriptive event-bootstrap interval of
[-0.016741, -0.001674], but only 83.57 percent of its conditioning events are
below q99.

The controlled follow-up derives every feature from the current event's exact
uncommitted shadow transition and compares 31 one-factor candidates on the same
frozen seed-17 state. The paired clean branch contains 443 target-user
transitions before payoff. No candidate has zero clean alerts, so the feasible
set and selected candidate are both empty.

Key outcomes are:

- LDAP broad, LDAP fine, hybrid, and hierarchical roles each produce 4 clean
  alerts and incrementally detect all 3 attack grammars;
- behavioral roles produce 6 clean alerts and detect 3, while the global role
  produces 8 clean alerts and detects 1;
- mean, 10-percent trimmed mean, and median shrinkage all produce 4/3, as do
  cosine, normalized Euclidean, and diagonal Mahalanobis distances;
- across weighted mean, maximum, and RMS formulas, Shewhart produces 3/1, EMA
  produces 14/0, and CUSUM produces 4/3, where each pair is clean
  alerts/incremental attack grammars detected;
- raising the validation quantile to 0.999 or 0.9995 reduces clean alerts to one
  but eliminates every attack detection;
- the closest tested trade-off is CUSUM allowance quantile 0.90 at 2/3, which
  still fails the zero-clean-alert gate.

The role, reference, and distance ties do not establish equivalence in general;
they show no decision-level advantage on this development incident. The
chronologically reserved CERT incidents remain unopened. Opening them after the
empty development selection would not constitute a valid confirmatory test.

The broader role-aware defense claim is rejected by the current evidence.

## Invalidated evidence

The earlier CERT longitudinal defense grid measured a state transition that
had already occurred and then rejected the next batch. It did not authorize the
current candidate transition. Its defense comparisons and positive-looking
figures must not be used as proof.

The 2026-08-10 CERT four-branch response probe corrected the branch-time shadow
update but calibrated its monitor on lagged validation transitions. Its
unprotected attack effects remain admissible because those scores do not depend
on the guard. Its reported 46 clean rejections and recovery estimates are not
used as defense evidence. They are superseded by the exact-current-transition
31-candidate ablation, which found no clean-feasible candidate and therefore did
not authorize a protected held-out replay.

The earlier official-KAIROS four-hour rollback recovered payoff scores but
quarantined 1.3966 percent of a clean day after a false `find` alert. It fails
the clean-cost requirement and is retained only as a diagnostic that motivated
the inactivity reset.

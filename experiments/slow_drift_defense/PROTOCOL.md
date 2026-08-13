# Confirmatory protocol for slow memory drift

## 1. Research question

Can a sequence of ordinary-looking events from a compromised identity alter a
frozen KAIROS-style temporal detector so that a fixed later malicious payoff is
less anomalous, and can an auxiliary state-integrity layer prevent that effect
without a higher clean false-positive burden than plain KAIROS?

The neural weights are fixed throughout. "Training the defense" means choosing
its role definition, reference, distance, evidence formula, sequential monitor,
threshold, and remediation policy on development data. It never means changing
the KAIROS weights.

## 2. Threat model

The attacker controls the event-producing actions of one existing user. The
attacker cannot edit model tensors, audit records, directory roles, validation
data, labels, or thresholds. Injected events must use event types and
destinations observed for clean role peers and must pass through the ordinary
TGN message and update functions.

For target user \(u\), conditioning sequence \(C\), and fixed payoff events
\(P\), the plain-model payoff effect is

\[
\Delta_{\mathrm{pay}}(C)
= \frac{1}{|P|}\sum_{e\in P}
\left[a_e(C)-a_e(\varnothing)\right].
\]

Here \(a_e(C)\) is the pre-update cross-entropy of payoff event \(e\) after
replaying the conditioning sequence, and \(a_e(\varnothing)\) is the paired
clean replay score from the same checkpoint and temporal snapshot. Negative
\(\Delta_{\mathrm{pay}}\) is score suppression. It is not automatically a
detection evasion.

The stealth fraction is

\[
S(C)=\frac{1}{|C|}\sum_{c\in C}
\mathbb{1}\left[a_c < \tau_{K}\right],
\]

where \(\tau_K\) is plain KAIROS's clean-validation q99 event threshold. A
conditioning phase is missed at event level only when every injected event is
below \(\tau_K\). Window-level miss and final decision flip are reported
separately.

### Attack families

1. `role_common`: repeat the most frequent clean event template among LDAP
   peers.
2. `role_diverse`: cycle through the five most frequent peer templates.
3. `payoff_type_matched`: repeat the most frequent peer template with the modal
   event type of the fixed payoff.
4. `greedy_payoff`: choose from twelve peer-observed templates by development-
   only beam search to minimize paired payoff loss, subject to q99 stealth.

Budgets are 3, 7, or 14 days and 1, 3, 5, or 10 interactions per day. The
14-day boundary is a stress-test horizon motivated by the global median dwell
time in the M-Trends 2026 Executive Edition. It is not claimed to be the exact
dwell time of CERT scenario 3 or of every intrusion.

## 3. Attack evidence required before defense evaluation

For each attack family and budget, plain KAIROS must be evaluated first. The
paper reports:

- paired payoff score change and an incident-cluster bootstrap interval;
- fraction of conditioning events below the frozen event threshold;
- whether the conditioning phase caused an event or 15-minute-window alert;
- whether a payoff alert changed from detected to undetected;
- collateral score change on non-payoff target-user events and matched controls.

The defense is not credited when no tested sequence suppresses the payoff. In
that case the honest result is attack failure, and defense sensitivity is only
a stress-test result.

## 4. Development and test separation

CERT scenario 3 supplies ten compact collaborative incidents with the same
high-level behavior and explicit target-actor labels. They are divided by
chronology before attack outcomes are inspected.

Development incidents:

- `s3:CSC0217`
- `s3:GTD0219`
- `s3:JGT0221`
- `s3:JTM0223`
- `s3:BBS0039`

Held-out test incidents:

- `s3:BSS0369`
- `s3:CCA0046`
- `s3:MPM0220`
- `s3:MSO0222`
- `s3:JLM0364`

The five model seeds are optimization replicates, not five independent cyber
incidents. Statistical uncertainty is clustered by incident, with seed shown as
a repeated-measure factor.

## 5. Correct candidate-state authorization

For current state \(s_u^-\) and event batch \(b\), the detector first computes
the ordinary anomaly score from \(s_u^-\). A shadow branch then applies the
same TGN update to obtain candidate state

\[
\widetilde{s}_u^+ = \operatorname{UPDT}(s_u^-, b).
\]

Every defense feature is computed from \(s_u^-\) and
\(\widetilde{s}_u^+\). If the monitor accepts the candidate, the shadow state is
committed. If it rejects the candidate, that same transition is quarantined.
This ordering fixes the previous implementation, which measured a prior state
change and then rejected the next event.

## 6. Candidate defense grid

Reference definitions:

- personal start-of-window anchor;
- LDAP broad role;
- LDAP fine role;
- behavioral cluster fitted on clean training activity;
- hybrid LDAP plus behavioral cluster;
- hierarchical LDAP median with parent and global shrinkage.

Distances:

- cosine distance;
- norm-normalized Euclidean distance;
- diagonal robust Mahalanobis distance.

Evidence signals:

- candidate distance from the peer reference;
- candidate distance from the personal trusted anchor;
- candidate update magnitude relative to clean peer steps;
- directional persistence of successive candidate updates;
- current plain-KAIROS event surprise.

Evidence formulas:

- single distance threshold;
- nonnegative weighted mean of robust z-scores;
- maximum weighted positive z-score;
- positive root-mean-square score.

Sequential monitors:

- Shewhart;
- exponential moving average;
- one-sided CUSUM.

Clean validation estimates all centers, robust scales, and threshold
quantiles. Development attacks choose one complete configuration subject to the
false-positive constraint below. The test cases never choose a parameter.

## 7. Selection rule and false-positive constraint

All methods are evaluated at the same user-batch decision unit. Plain KAIROS
alerts a user-batch when its maximum event loss exceeds \(\tau_K\). A defense
false positive is a rejected clean user-batch. The selected defense must meet
both conditions on development data:

1. observed clean rejection rate no greater than plain KAIROS's clean alert
   rate;
2. upper one-sided 95 percent cluster-bootstrap bound for the rate difference
   no greater than 0.001 absolute.

Among feasible configurations, selection maximizes the number of successful
conditioning sequences detected before the payoff. Ties are resolved by lower
clean rejection, earlier detection, smaller computational cost, and then a
fixed lexical ordering of the configuration identifier. If no configuration is
feasible, the null option, plain KAIROS, is retained.

## 8. Four matched test branches

Each held-out incident, attack, budget, seed, and selected defense uses four
branches from an identical temporal snapshot:

1. clean;
2. conditioned;
3. clean plus defense;
4. conditioned plus defense.

Defense recovery is the difference in differences

\[
R = \left(\bar a_{D}-\bar a_{G}\right)
  - \left(\bar a_{A}-\bar a_{C}\right),
\]

where \(C\), \(A\), \(G\), and \(D\) denote clean, conditioned, clean plus
defense, and conditioned plus defense. Positive \(R\) means the defense removes
some conditioning effect after accounting for its clean main effect. The paper
also reports the clean main effect, attack detection rate, rejection burden,
latency, and operational detection flips.

## 9. Claim gates

The final paper may claim that the tested defense works only if all of the
following hold on held-out test incidents:

1. at least one predeclared realizable attack has negative
   \(\Delta_{\mathrm{pay}}\) with its incident-cluster interval below zero;
2. plain KAIROS misses the attack's conditioning phase at the stated event or
   window level;
3. the selected defense detects a majority of successful attacks before payoff;
4. defense recovery \(R\) is positive with its incident-cluster interval above
   zero;
5. the false-positive constraint is satisfied;
6. clean plus defense does not materially degrade KAIROS's payoff ranking or
   window decisions.

Failure of a gate is reported as a negative result. It is not repaired by
changing the test threshold, redefining the payoff, or selecting a different
formula after test inspection.

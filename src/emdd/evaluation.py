"""Paired research metrics, uncertainty summaries, and publication graphs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def paired_payoff_summary(scores: pd.DataFrame) -> dict:
    clean = scores[(scores["condition"] == "clean") & scores["attack_label"]]
    attacked = scores[(scores["condition"] == "conditioned") & scores["attack_label"]]
    paired = clean[["seed", "event_id", "anomaly_score"]].merge(
        attacked[["seed", "event_id", "anomaly_score"]],
        on=["seed", "event_id"],
        suffixes=("_clean", "_conditioned"),
        validate="one_to_one",
    )
    paired["delta"] = paired["anomaly_score_conditioned"] - paired["anomaly_score_clean"]
    return {
        "pairs": int(len(paired)),
        "clean_mean": float(paired["anomaly_score_clean"].mean()),
        "conditioned_mean": float(paired["anomaly_score_conditioned"].mean()),
        "paired_delta_mean": float(paired["delta"].mean()),
        "suppressed_fraction": float((paired["delta"] < 0).mean()),
        "seed_deltas": {
            str(seed): float(group["delta"].mean()) for seed, group in paired.groupby("seed")
        },
    }


def cluster_bootstrap_interval(
    paired: pd.DataFrame,
    cluster_column: str,
    value_column: str,
    samples: int,
    seed: int = 2026,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    clusters = paired[cluster_column].unique()
    estimates = []
    for _ in range(samples):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        values = [paired.loc[paired[cluster_column] == cluster, value_column].to_numpy() for cluster in selected]
        estimates.append(float(np.concatenate(values).mean()))
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def defense_summary(
    decisions: pd.DataFrame,
    payoff_start: pd.Timestamp,
    conditioning_start: pd.Timestamp,
    target_node: int,
) -> dict:
    clean = decisions[
        (decisions["condition"] == "clean") & ~decisions["attack_label"]
    ]
    poisoned = decisions[decisions["condition"] == "conditioned"]
    before = poisoned[
        (poisoned["timestamp"] >= conditioning_start)
        & (poisoned["timestamp"] < payoff_start)
        & (poisoned["node_id"] == target_node)
    ]
    detected = before[before["alert"]]
    if clean.empty:
        entity_days = 1e-8
    else:
        span_days = max(
            (clean["timestamp"].max() - clean["timestamp"].min()).total_seconds()
            / 86_400,
            1.0 / 86_400,
        )
        entity_days = max(float(clean["node_id"].nunique()) * span_days, 1e-8)
    payoff = poisoned[poisoned["attack_label"]]
    return {
        "detected_before_payoff": bool(not detected.empty),
        "first_detection": None if detected.empty else pd.Timestamp(detected["timestamp"].min()).isoformat(),
        "false_alerts_per_1000_entity_days": float(clean["alert"].sum() / entity_days * 1000),
        "clean_alerts": int(clean["alert"].sum()),
        "clean_entity_days": float(entity_days),
        "conditioned_payoff_anomaly_mean": float(payoff["anomaly_score"].mean()),
        "conditioned_payoff_updates_rejected": int((~payoff["update_accepted"]).sum()),
    }


def write_drift_plot(conditioning: pd.DataFrame, output: Path, threshold: float) -> None:
    selected = conditioning[
        conditioning["event_id"].astype(str).str.startswith("conditioning:")
    ].copy()
    if selected.empty:
        return
    selected = selected.sort_values("timestamp")
    shifts = selected["memory_shifts"].map(
        lambda item: float(next(iter(item.values()))) if item else 0.0
    )
    cumulative = shifts.cumsum()
    figure, left = plt.subplots(figsize=(7.2, 4.2))
    left.plot(selected["timestamp"], selected["anomaly_score"], color="#b23a48", label="event anomaly")
    left.axhline(threshold, color="#b23a48", linestyle="--", alpha=0.65, label="validation q99")
    left.set_ylabel("Pre-update event anomaly", color="#b23a48")
    right = left.twinx()
    right.plot(selected["timestamp"], cumulative, color="#286983", label="cumulative memory movement")
    right.set_ylabel("Cumulative target-memory step (L2)", color="#286983")
    left.set_title("Slow conditioning: stealth constraint and accumulated drift")
    left.tick_params(axis="x", rotation=25)
    handles = left.get_lines() + right.get_lines()
    left.legend(handles, [item.get_label() for item in handles], frameon=False, loc="best")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def write_payoff_plot(scores: pd.DataFrame, output: Path) -> None:
    attack = scores[scores["attack_label"]]
    means = attack.groupby(["seed", "condition"])["anomaly_score"].mean().unstack()
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    for seed, row in means.iterrows():
        axis.plot([0, 1], [row["clean"], row["conditioned"]], marker="o", label=f"seed {seed}")
    axis.set_xticks([0, 1], ["Clean", "Conditioned"])
    axis.set_ylabel("Mean payoff anomaly")
    axis.set_title("Paired payoff effect of slow conditioning")
    axis.legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))

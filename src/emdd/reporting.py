"""Aggregate completed post-training runs without overstating independence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import write_json
from .io import resolve_config_path


def _run_rows(artifact_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(artifact_root.glob("seed_*/d*_r*/summary.json")):
        payload = json.loads(path.read_text())
        seed_text = path.parents[1].name.removeprefix("seed_")
        row = {
            "seed": int(seed_text),
            "duration_days": int(payload["duration_days"]),
            "interactions_per_day": int(payload["interactions_per_day"]),
            "pairs": int(payload["pairs"]),
            "clean_mean": float(payload["clean_mean"]),
            "conditioned_mean": float(payload["conditioned_mean"]),
            "paired_delta_mean": float(payload["paired_delta_mean"]),
            "suppressed_fraction": float(payload["suppressed_fraction"]),
            "all_conditioning_subthreshold": bool(
                payload["all_conditioning_subthreshold"]
            ),
            "attack_success": bool(payload["attack_success"]),
        }
        for method in ("ema", "cusum"):
            defense = payload.get("defense", {}).get(method, {})
            row[f"{method}_detected_before_payoff"] = bool(
                defense.get("detected_before_payoff", False)
            )
            row[f"{method}_false_alerts_per_1000_entity_days"] = float(
                defense.get("false_alerts_per_1000_entity_days", np.nan)
            )
            row[f"{method}_payoff_anomaly_mean"] = float(
                defense.get("conditioned_payoff_anomaly_mean", np.nan)
            )
        rows.append(row)
    return rows


def _write_heatmap(frame: pd.DataFrame, output: Path) -> None:
    table = frame.pivot(
        index="duration_days",
        columns="interactions_per_day",
        values="paired_delta_mean",
    ).sort_index()
    figure, axis = plt.subplots(figsize=(6.8, 4.5))
    image = axis.imshow(table.to_numpy(), cmap="coolwarm", aspect="auto")
    axis.set_xticks(range(len(table.columns)), table.columns)
    axis.set_yticks(range(len(table.index)), table.index)
    axis.set_xlabel("Conditioning interactions per day")
    axis.set_ylabel("Conditioning duration (days)")
    axis.set_title("Mean paired payoff-anomaly change across seeds")
    for row in range(table.shape[0]):
        for column in range(table.shape[1]):
            value = table.iloc[row, column]
            label = "NA" if pd.isna(value) else f"{value:.3f}"
            axis.text(column, row, label, ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label="Conditioned minus clean anomaly")
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def aggregate_artifacts(config: dict) -> dict:
    artifact_root = resolve_config_path(config, config["paths"]["artifacts"])
    rows = _run_rows(artifact_root)
    if not rows:
        raise FileNotFoundError(
            f"No completed run summaries under {artifact_root}; run `emdd study` first"
        )
    runs = pd.DataFrame(rows).sort_values(
        ["duration_days", "interactions_per_day", "seed"]
    )
    aggregate = (
        runs.groupby(["duration_days", "interactions_per_day"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            paired_delta_mean=("paired_delta_mean", "mean"),
            paired_delta_sd=("paired_delta_mean", "std"),
            attack_success_fraction=("attack_success", "mean"),
            all_subthreshold_fraction=("all_conditioning_subthreshold", "mean"),
            ema_detection_fraction=("ema_detected_before_payoff", "mean"),
            cusum_detection_fraction=("cusum_detected_before_payoff", "mean"),
            ema_false_alert_rate=(
                "ema_false_alerts_per_1000_entity_days",
                "mean",
            ),
            cusum_false_alert_rate=(
                "cusum_false_alerts_per_1000_entity_days",
                "mean",
            ),
        )
        .sort_values(["duration_days", "interactions_per_day"])
    )
    output = artifact_root / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "seed_level_runs.csv", index=False)
    aggregate.to_csv(output / "budget_summary.csv", index=False)
    _write_heatmap(aggregate, output / "payoff_delta_heatmap.png")
    summary = {
        "completed_runs": int(len(runs)),
        "seeds_present": sorted(int(seed) for seed in runs["seed"].unique()),
        "budgets_present": int(
            runs[["duration_days", "interactions_per_day"]].drop_duplicates().shape[0]
        ),
        "claim_warning": (
            "Seeds are repeated model fits on one CERT incident, not independent "
            "datasets; aggregate values are descriptive."
        ),
        "files": {
            "seed_level": str(output / "seed_level_runs.csv"),
            "budget_summary": str(output / "budget_summary.csv"),
            "heatmap": str(output / "payoff_delta_heatmap.png"),
        },
    }
    write_json(output / "summary.json", summary)
    return summary

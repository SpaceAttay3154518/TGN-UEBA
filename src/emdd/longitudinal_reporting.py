"""Aggregate longitudinal CERT results and create publication figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .io import resolve_config_path


def _mean_ci(values: np.ndarray, samples: int = 5000, seed: int = 2026) -> dict:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"mean": None, "lower": None, "upper": None, "n": 0}
    rng = np.random.default_rng(seed)
    estimates = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(samples)]
    )
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "n": int(len(values)),
    }


def _save(figure, root: Path, name: str) -> None:
    figure.tight_layout()
    figure.savefig(root / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(root / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def aggregate_longitudinal(config: dict) -> dict:
    artifact_root = resolve_config_path(config, config["paths"]["artifacts"])
    bootstrap_samples = int(config.get("evaluation", {}).get("bootstrap_samples", 5000))
    training_summary_path = (
        resolve_config_path(config, config["paths"]["checkpoints"])
        / "training_summary.json"
    )
    training_summaries = (
        json.loads(training_summary_path.read_text())
        if training_summary_path.exists()
        else []
    )
    training_by_seed = {
        int(item["seed"]): item for item in training_summaries
    }
    summaries = []
    ablations = []
    cases = []
    samples = []
    conditioning = []
    for seed_dir in sorted(artifact_root.glob("seed_*")):
        summary_path = seed_dir / "longitudinal_summary.json"
        if not summary_path.exists():
            continue
        summaries.append(json.loads(summary_path.read_text()))
        frame = pd.read_csv(seed_dir / "defense_ablation.csv")
        frame["seed"] = int(seed_dir.name.split("_")[-1])
        ablations.append(frame)
        cases.append(pd.read_csv(seed_dir / "case_level_results.csv"))
        score_frame = pd.read_parquet(seed_dir / "event_score_sample.parquet")
        score_frame["seed"] = int(seed_dir.name.split("_")[-1])
        samples.append(score_frame[score_frame["event_metric_eligible"]])
        conditioning_path = seed_dir / "slow_conditioning" / "summary.csv"
        if conditioning_path.exists():
            conditioning.append(pd.read_csv(conditioning_path))
    if not summaries:
        raise FileNotFoundError("No completed longitudinal seed summaries exist")
    figures = artifact_root / "aggregate" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ablation = pd.concat(ablations, ignore_index=True)
    case = pd.concat(cases, ignore_index=True)
    score_sample = pd.concat(samples, ignore_index=True)
    primary_id = (
        f"primary|{config['defense']['primary_formula']}|"
        f"{config['defense']['primary_monitor']}"
    )

    metric_names = [
        "roc_auc",
        "average_precision_prevalence_corrected",
        "target_actor_recall_q99",
        "benign_event_fpr_q99",
    ]
    event_aggregate = {
        name: _mean_ci(
            np.asarray([item["event_metrics"][name] for item in summaries]),
            samples=bootstrap_samples,
        )
        for name in metric_names
    }
    baseline_aggregate = {
        name: _mean_ci(
            np.asarray([item["baseline_incidents"][name] for item in summaries]),
            samples=bootstrap_samples,
        )
        for name in ("attack_window_recall", "matched_control_rate", "net_detection_rate")
    }
    grouped_ablation = (
        ablation.groupby(
            ["variant_id", "base_id", "role", "reference", "distance", "formula", "monitor"],
            as_index=False,
        )
        .agg(
            attack_window_recall=("attack_window_recall", "mean"),
            matched_control_rate=("matched_control_rate", "mean"),
            net_detection_rate=("net_detection_rate", "mean"),
            attack_batch_recall=("attack_batch_recall", "mean"),
            false_alerts_per_1000_entity_days=(
                "benign_alerts_per_1000_entity_days", "mean"
            ),
            false_alert_decisions_per_1000_entity_days=(
                "benign_alert_decisions_per_1000_entity_days", "mean"
            ),
        )
    )
    grouped_ablation.to_csv(
        artifact_root / "aggregate" / "defense_ablation_aggregate.csv", index=False
    )
    scenario = (
        case.groupby(["method", "scenario"], as_index=False)
        .agg(
            attack_window_recall=("attack_window_alert", "mean"),
            matched_control_rate=("matched_control_alert", "mean"),
        )
    )
    scenario["net_detection_rate"] = (
        scenario["attack_window_recall"] - scenario["matched_control_rate"]
    )
    scenario.to_csv(
        artifact_root / "aggregate" / "scenario_results.csv", index=False
    )

    manifest = json.loads(
        (
            resolve_config_path(config, config["paths"]["prepared"])
            / "manifest.json"
        ).read_text()
    )
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    splits = ["train", "validation", "context", "evaluation"]
    values = [manifest["split_sizes"][name] / 1_000_000 for name in splits]
    axis.bar(splits, values, color=["#286983", "#6b8e23", "#d29f42", "#b23a48"])
    axis.set_ylabel("Events (millions)")
    axis.set_title("Leakage-free CERT r4.2 chronological split")
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:.2f}M", ha="center", va="bottom")
    _save(figure, figures, "dataset_splits")

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for summary in summaries:
        training = training_by_seed.get(
            int(summary["seed"]), summary["checkpoint_training"]
        )
        history = pd.DataFrame(training["history"])
        axis.plot(
            history["epoch"],
            history["validation_reconstruction_error"],
            marker="o",
            label=f"seed {summary['seed']}",
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation edge-type cross-entropy")
    axis.set_title("Adapted KAIROS training convergence")
    axis.legend(frameon=False, ncol=2)
    _save(figure, figures, "training_convergence")

    primary = grouped_ablation[grouped_ablation["variant_id"] == primary_id]
    primary_seed_rows = ablation[ablation["variant_id"] == primary_id]
    primary_aggregate = {
        name: _mean_ci(
            primary_seed_rows[name].to_numpy(float), samples=bootstrap_samples
        )
        for name in ("attack_window_recall", "matched_control_rate", "net_detection_rate")
    }
    primary_means = {name: values["mean"] for name, values in primary_aggregate.items()}
    labels = ["Attack-window\nalert", "Matched-control\nalert", "Net detection"]
    plain = [
        baseline_aggregate["attack_window_recall"]["mean"],
        baseline_aggregate["matched_control_rate"]["mean"],
        baseline_aggregate["net_detection_rate"]["mean"],
    ]
    defended = [
        primary_means["attack_window_recall"],
        primary_means["matched_control_rate"],
        primary_means["net_detection_rate"],
    ]
    x = np.arange(3)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(x - 0.19, plain, 0.38, label="Plain adapted KAIROS", color="#536878")
    axis.bar(
        x + 0.19, defended, 0.38,
        label="KAIROS + drift monitor (gate trigger)", color="#3a7d44"
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(min(0.0, min(plain + defended) - 0.05), 1.0)
    axis.set_ylabel("Rate across 70 incidents")
    axis.set_title("Real-case discrimination: attack vs matched pre-24 h control")
    axis.legend(frameon=False)
    _save(figure, figures, "baseline_vs_defense")

    plot_ablation = grouped_ablation.sort_values(
        ["net_detection_rate", "false_alerts_per_1000_entity_days"]
    ).reset_index(drop=True)
    labels = []
    for _, row in plot_ablation.iterrows():
        if row["base_id"] == "primary":
            formula = {
                "weighted_mean_positive_z": "weighted mean",
                "max_positive_z": "maximum",
                "l2_positive_z": "RMS",
            }.get(row["formula"], str(row["formula"]).replace("_", " "))
            labels.append(f"{formula} + {str(row['monitor']).upper()}")
        else:
            labels.append(str(row["base_id"]).replace("=", ": ").replace("_", " "))
    y = np.arange(len(plot_ablation))
    burden = plot_ablation["false_alerts_per_1000_entity_days"].to_numpy(float)
    figure, axis = plt.subplots(figsize=(8.2, 7.0))
    points = axis.scatter(
        plot_ablation["net_detection_rate"],
        y,
        c=burden,
        cmap="viridis",
        s=58,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    primary_position = np.flatnonzero(plot_ablation["variant_id"].eq(primary_id))
    if len(primary_position):
        row = plot_ablation.iloc[int(primary_position[0])]
        axis.scatter(
            row["net_detection_rate"],
            int(primary_position[0]),
            marker="*",
            s=190,
            facecolor="none",
            edgecolor="#c0392b",
            linewidth=1.4,
            zorder=4,
            label="Fixed comparison anchor",
        )
    axis.axvline(0.0, color="#a61b1b", linestyle="--", linewidth=1.0)
    axis.set_yticks(y, labels, fontsize=8)
    axis.set_xlabel("Net detection: incident alert rate minus matched-control rate")
    axis.set_title("All defense candidates remain below the null-option boundary")
    axis.grid(axis="x", color="#d7d7d7", linewidth=0.7, zorder=0)
    axis.legend(frameon=False, loc="lower right")
    figure.colorbar(points, ax=axis, label="Alerted benign entity-days per 1,000")
    _save(figure, figures, "defense_ablation_tradeoff")

    heat = grouped_ablation[grouped_ablation["base_id"] == "primary"].pivot(
        index="formula", columns="monitor", values="net_detection_rate"
    )
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    image = axis.imshow(heat.to_numpy(), cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(
        np.arange(len(heat.columns)),
        [str(value).upper() for value in heat.columns],
    )
    formula_names = {
        "l2_positive_z": "RMS",
        "max_positive_z": "Maximum",
        "weighted_mean_positive_z": "Weighted mean",
    }
    axis.set_yticks(
        np.arange(len(heat.index)),
        [formula_names.get(value, str(value).replace("_", " ")) for value in heat.index],
    )
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            axis.text(j, i, f"{heat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    axis.set_title("Net incident detection by evidence formula and monitor")
    figure.colorbar(image, ax=axis, label="Attack rate - matched-control rate")
    _save(figure, figures, "formula_monitor_heatmap")

    selected_scenario = scenario[
        scenario["method"].isin(["plain_adapted_kairos_q99", primary_id])
    ]
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    for offset, (method, label, color) in zip(
        (-0.18, 0.18),
        [
            ("plain_adapted_kairos_q99", "Plain adapted KAIROS", "#536878"),
            (primary_id, "KAIROS + drift monitor", "#3a7d44"),
        ],
        strict=True,
    ):
        part = selected_scenario[selected_scenario["method"] == method].set_index("scenario")
        axis.bar(
            np.arange(3) + offset,
            [part.loc[value, "net_detection_rate"] for value in (1, 2, 3)],
            0.36,
            label=label,
            color=color,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(3), ["Scenario 1", "Scenario 2", "Scenario 3"])
    axis.set_ylabel("Net 24 h detection rate")
    axis.set_title("Real-case results by CERT scenario")
    axis.legend(frameon=False)
    _save(figure, figures, "scenario_comparison")

    figure, axis = plt.subplots(figsize=(6.3, 5.0))
    for seed, group in score_sample.groupby("seed"):
        labels_seed = group["attack_label"].to_numpy(np.int8)
        values_seed = group["anomaly_score"].to_numpy(float)
        weights = 1.0 / group["benign_sampling_probability"].to_numpy(float)
        false_positive, true_positive, _ = roc_curve(
            labels_seed, values_seed, sample_weight=weights
        )
        axis.plot(false_positive, true_positive, alpha=0.75, label=f"seed {seed}")
    axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("Event-level ROC on all answer rows and sampled benign events")
    axis.legend(frameon=False)
    _save(figure, figures, "event_roc")

    conditioning_aggregate = None
    if conditioning:
        conditioning_frame = pd.concat(conditioning, ignore_index=True)
        conditioning_frame["conditioned_minus_clean"] = (
            conditioning_frame["conditioned_payoff_mean"]
            - conditioning_frame["clean_payoff_mean"]
        )
        conditioning_frame["conditioned_gated_minus_clean_gated"] = (
            conditioning_frame["gated_conditioning_delta"]
        )
        conditioning_grouped = (
            conditioning_frame.groupby(
                ["policy", "duration_days", "interactions_per_day"],
                as_index=False,
            )
            .agg(
                seeds=("seed", "nunique"),
                conditioning_events=("conditioning_events", "mean"),
                conditioning_subthreshold_fraction=(
                    "conditioning_subthreshold_fraction", "mean"
                ),
                conditioned_minus_clean=("conditioned_minus_clean", "mean"),
                conditioned_gated_minus_clean_gated=(
                    "conditioned_gated_minus_clean_gated", "mean"
                ),
                gating_main_effect=("gating_main_effect", "mean"),
                defense_recovery_difference_in_differences=(
                    "defense_recovery_difference_in_differences", "mean"
                ),
                payoff_suppression_rate=("payoff_suppression", "mean"),
                stealth_qualified_success_rate=(
                    "stealth_qualified_success", "mean"
                ),
                clean_prepayoff_alert_rate=("clean_prepayoff_alert", "mean"),
                conditioning_detection_rate=("detected_on_conditioning", "mean"),
                incremental_prepayoff_detection_rate=(
                    "incremental_prepayoff_detection", "mean"
                ),
                conditioning_updates_rejected=(
                    "conditioning_updates_rejected", "mean"
                ),
                conditioning_rejection_fraction=(
                    "conditioning_rejection_fraction", "mean"
                ),
                payoff_pairs=("payoff_pairs", "sum"),
            )
        )
        conditioning_grouped.to_csv(
            artifact_root / "aggregate" / "slow_conditioning_aggregate.csv",
            index=False,
        )
        policies = list(conditioning_grouped["policy"].drop_duplicates())
        rates = sorted(conditioning_grouped["interactions_per_day"].unique())
        durations = sorted(conditioning_grouped["duration_days"].unique())
        figure, axes = plt.subplots(
            2, len(policies), figsize=(4.0 * len(policies), 6.0), squeeze=False,
            sharex=True, sharey="row"
        )
        bounds = np.nanmax(
            np.abs(
                conditioning_grouped[
                    [
                        "conditioned_minus_clean",
                        "conditioned_gated_minus_clean_gated",
                    ]
                ].to_numpy(float)
            )
        )
        bounds = max(float(bounds), 1e-6)
        for column, policy in enumerate(policies):
            selected = conditioning_grouped[
                conditioning_grouped["policy"] == policy
            ]
            for row_index, (metric, title) in enumerate(
                (
                    ("conditioned_minus_clean", "Conditioned minus clean"),
                    (
                        "conditioned_gated_minus_clean_gated",
                        "Conditioned+gate minus clean+gate",
                    ),
                )
            ):
                matrix = (
                    selected.pivot(
                        index="duration_days",
                        columns="interactions_per_day",
                        values=metric,
                    )
                    .reindex(index=durations, columns=rates)
                )
                axis = axes[row_index, column]
                image = axis.imshow(
                    matrix.to_numpy(float),
                    cmap="RdBu_r",
                    vmin=-bounds,
                    vmax=bounds,
                    aspect="auto",
                )
                for i in range(len(durations)):
                    for j in range(len(rates)):
                        value = matrix.iloc[i, j]
                        axis.text(
                            j, i, f"{value:+.3f}", ha="center", va="center", fontsize=7
                        )
                axis.set_xticks(np.arange(len(rates)), rates)
                axis.set_yticks(np.arange(len(durations)), durations)
                axis.set_xlabel("Conditioning interactions/day")
                if column == 0:
                    axis.set_ylabel(f"{title}\nDuration (days)")
                if row_index == 0:
                    axis.set_title(policy.replace("_", " "))
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label="Paired payoff anomaly-score difference",
            shrink=0.84,
        )
        _save(figure, figures, "slow_conditioning_effect")
        conditioning_aggregate = {
            "completed_seeds": sorted(
                int(value) for value in conditioning_frame["seed"].unique()
            ),
            "runs": int(len(conditioning_frame)),
            "payoff_suppression_fraction": float(
                conditioning_frame["payoff_suppression"].mean()
            ),
            "stealth_qualified_success_fraction": float(
                conditioning_frame["stealth_qualified_success"].mean()
            ),
            "conditioning_detection_fraction": float(
                conditioning_frame["detected_on_conditioning"].mean()
            ),
            "incremental_prepayoff_detection_fraction": float(
                conditioning_frame["incremental_prepayoff_detection"].mean()
            ),
            "clean_prepayoff_alert_fraction": float(
                conditioning_frame["clean_prepayoff_alert"].mean()
            ),
            "mean_conditioning_subthreshold_fraction": float(
                conditioning_frame["conditioning_subthreshold_fraction"].mean()
            ),
            "mean_conditioned_minus_clean": float(
                conditioning_frame["conditioned_minus_clean"].mean()
            ),
            "mean_conditioned_gated_minus_clean_gated": float(
                conditioning_frame["conditioned_gated_minus_clean_gated"].mean()
            ),
            "mean_recovery_difference_in_differences": float(
                conditioning_frame[
                    "defense_recovery_difference_in_differences"
                ].mean()
            ),
        }

    aggregate = {
        "completed_seeds": [int(item["seed"]) for item in summaries],
        "event_metrics": event_aggregate,
        "baseline_incidents": baseline_aggregate,
        "primary_defense_variant": primary_id,
        "primary_defense": primary_aggregate,
        "slow_conditioning": conditioning_aggregate,
        "training": [
            training_by_seed.get(int(item["seed"]), item["checkpoint_training"])
            for item in summaries
        ],
        "dataset_manifest": manifest,
    }
    output = artifact_root / "aggregate" / "aggregate_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2))
    return aggregate

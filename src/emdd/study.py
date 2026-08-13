"""Post-training attack, defense, and explanation study orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from .attack import (
    build_action_library,
    interleave_conditioning,
    plan_slow_conditioning,
)
from .defense import (
    DriftFeatures,
    FeatureExtractor,
    RoleReferenceBank,
    SequentialMonitor,
    fit_calibration,
)
from .evaluation import (
    defense_summary,
    paired_payoff_summary,
    write_drift_plot,
    write_json,
    write_payoff_plot,
)
from .explain import (
    attention_explanation,
    counterfactual_deletion_explanation,
    mitre_candidates,
)
from .io import PreparedDataset, load_checkpoint, resolve_config_path
from .model import TGNDetector
from .replay import batch_from_row, replay_frame, score_batch


@torch.no_grad()
def calibrate_clean_validation(
    trained: TGNDetector,
    validation: pd.DataFrame,
    metadata: dict,
    defense_config: dict,
) -> tuple[TGNDetector, float, object, list[DriftFeatures], pd.DataFrame]:
    branch = trained.fork()
    profiles = metadata["role_profiles"]
    bank = RoleReferenceBank(
        metadata["node_to_id"],
        profiles,
        float(defense_config["shrinkage_kappa"]),
        int(defense_config["minimum_role_peers"]),
    )
    bank.update(branch.memory.memory.detach())
    extractor = FeatureExtractor()
    features: list[DriftFeatures] = []
    score_rows: list[dict] = []
    accepted = 0
    for _, row in validation.iterrows():
        batch = batch_from_row(row, metadata, branch.device_ref)
        anomaly = score_batch(branch, batch)
        source = int(row["src_id"])
        before = branch.memory.memory[source].detach().clone()
        branch.observe(batch)
        after = branch.memory.memory[source].detach().clone()
        try:
            role = bank.role(source)
            reference = bank.reference(source)
        except KeyError:
            role = "entity:unknown"
            reference = before
        if role != "entity:unknown":
            features.append(
                extractor.step(
                    source,
                    role,
                    float(row["timestamp_s"]),
                    before,
                    after,
                    reference,
                    anomaly,
                )
            )
        score_rows.append(
            {"event_id": row["event_id"], "timestamp": row["timestamp"], "anomaly_score": anomaly}
        )
        accepted += 1
        if accepted % 512 == 0:
            bank.update(branch.memory.memory.detach())
    scores = pd.DataFrame(score_rows)
    threshold = float(scores["anomaly_score"].quantile(0.99))
    calibration = fit_calibration(
        features,
        defense_config["weights"],
        float(defense_config["ema_alpha"]),
        float(defense_config["cusum_allowance_quantile"]),
        float(defense_config["alert_quantile"]),
    )
    return branch, threshold, calibration, features, scores


@torch.no_grad()
def monitored_replay(
    start: TGNDetector,
    stream: pd.DataFrame,
    metadata: dict,
    defense_config: dict,
    calibration,
    method: str,
    condition: str,
    target_node: int,
) -> tuple[pd.DataFrame, TGNDetector]:
    if method not in {"ema", "cusum"}:
        raise ValueError("method must be 'ema' or 'cusum'")
    branch = start.fork()
    bank = RoleReferenceBank(
        metadata["node_to_id"],
        metadata["role_profiles"],
        float(defense_config["shrinkage_kappa"]),
        int(defense_config["minimum_role_peers"]),
    )
    bank.update(branch.memory.memory.detach(), excluded={target_node})
    extractor = FeatureExtractor()
    monitor = SequentialMonitor(
        calibration,
        defense_config["weights"],
        float(defense_config["ema_alpha"]),
    )
    output: list[dict] = []
    accepted = 0
    for _, row in stream.sort_values(["time_rel", "event_id"]).iterrows():
        batch = batch_from_row(row, metadata, branch.device_ref)
        anomaly = score_batch(branch, batch)
        source = int(row["src_id"])
        candidate = branch.fork()
        before = candidate.memory.memory[source].detach().clone()
        candidate.observe(batch)
        after = candidate.memory.memory[source].detach().clone()
        role = bank.role(source)
        alert = False
        decision = None
        if role != "entity:unknown":
            try:
                reference = bank.reference(source)
            except KeyError:
                reference = before
            feature = extractor.step(
                source,
                role,
                float(row["timestamp_s"]),
                before,
                after,
                reference,
                anomaly,
                update_history=True,
                update_role_scale=source != target_node,
            )
            decision = monitor.step(feature)
            alert = decision.ema_alert if method == "ema" else decision.cusum_alert
        if not alert:
            branch = candidate
            accepted += 1
            if accepted % 512 == 0:
                bank.update(branch.memory.memory.detach(), excluded={target_node})
        output.append(
            {
                "condition": condition,
                "method": method,
                "event_id": row["event_id"],
                "timestamp": row["timestamp"],
                "node_id": source,
                "event_type": row["event_type"],
                "attack_label": bool(row["attack_label"]),
                "is_conditioning": str(row["event_id"]).startswith("conditioning:"),
                "anomaly_score": anomaly,
                "alert": alert,
                "combined_signal": None if decision is None else decision.combined_signal,
                "ema": None if decision is None else decision.ema,
                "cusum": None if decision is None else decision.cusum,
                "update_accepted": not alert,
            }
        )
    return pd.DataFrame(output), branch


def role_peer_names(metadata: dict, target_name: str) -> set[str]:
    profiles = metadata["role_profiles"]
    target = profiles[target_name]
    fine = {
        name for name, profile in profiles.items()
        if profile["fine_role"] == target["fine_role"] and name != target_name
    }
    if len(fine) >= 3:
        return fine
    parent = {
        name for name, profile in profiles.items()
        if profile["parent_role"] == target["parent_role"] and name != target_name
    }
    return parent or {name for name in profiles if name != target_name}


def run_seed_study(
    seed: int,
    events: pd.DataFrame | PreparedDataset,
    metadata: dict,
    config: dict,
    *,
    durations: list[int] | None = None,
    rates: list[int] | None = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = resolve_config_path(config, config["paths"]["checkpoints"])
    artifact_root = resolve_config_path(config, config["paths"]["artifacts"])
    checkpoint = checkpoint_dir / f"tgn_seed_{seed}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing {checkpoint}. Run the configured training stage first."
        )
    trained, payload = load_checkpoint(checkpoint, config, metadata, device)
    if isinstance(events, PreparedDataset):
        validation = events.load_splits("validation")
        context = events.load_splits("context")
        test = events.load_splits("test")
    else:
        validation = events[events["split"] == "validation"].copy()
        context = events[events["split"] == "context"].copy()
        test = events[events["split"] == "test"].copy()
    payoff = test[test["attack_label"]].copy()
    validated, threshold, calibration, _, validation_scores = calibrate_clean_validation(
        trained, validation, metadata, config["defense"]
    )
    target_name = metadata["target_user_node"]
    target_node = int(metadata["node_to_id"][target_name])
    peer_names = role_peer_names(metadata, target_name)
    library_source = (
        events.role_events(("train", "validation"), peer_names)
        if isinstance(events, PreparedDataset)
        else events[events["split"].isin(["train", "validation"])]
    )
    templates = build_action_library(
        library_source,
        metadata,
        peer_names,
        int(config["attack"]["candidate_pool_size"]),
    )

    clean_context_model = validated.fork()
    replay_frame(clean_context_model, context, metadata, "context")
    clean_result = replay_frame(
        clean_context_model, test, metadata, "clean"
    ).scores
    clean_result["seed"] = seed
    clean_full_stream = pd.concat([context, test], ignore_index=True).sort_values(
        ["time_rel", "event_id"]
    )

    chosen_durations = durations or [int(value) for value in config["attack"]["durations_days"]]
    chosen_rates = rates or [int(value) for value in config["attack"]["interactions_per_day"]]
    run_summaries = []
    all_scores = []
    for duration in chosen_durations:
        for rate in chosen_rates:
            run_name = f"seed_{seed}/d{duration}_r{rate}"
            output = artifact_root / run_name
            output.mkdir(parents=True, exist_ok=True)
            plan = plan_slow_conditioning(
                validated,
                context,
                payoff,
                templates,
                metadata,
                config["attack"],
                target_node=target_node,
                target_name=target_name,
                threshold=threshold,
                duration_days=duration,
                interactions_per_day=rate,
            )
            write_json(output / "plan.json", plan.to_dict())
            conditioned_context = interleave_conditioning(context, plan.conditioning_events)
            conditioned_model = validated.fork()
            conditioning_replay = replay_frame(
                conditioned_model,
                conditioned_context,
                metadata,
                "conditioning",
                record_memory_nodes={target_node},
            ).scores
            conditioned = replay_frame(
                conditioned_model,
                test,
                metadata,
                "conditioned",
            ).scores
            conditioned["seed"] = seed
            paired_scores = pd.concat([clean_result, conditioned], ignore_index=True)
            summary = paired_payoff_summary(paired_scores)
            summary.update(
                {
                    "duration_days": duration,
                    "interactions_per_day": rate,
                    "threshold_q99": threshold,
                    "all_conditioning_subthreshold": bool(
                        conditioning_replay.loc[
                            conditioning_replay["event_id"].astype(str).str.startswith("conditioning:"),
                            "anomaly_score",
                        ].lt(threshold).all()
                    ),
                    "attack_success": bool(
                        summary["paired_delta_mean"] < 0
                        and plan.feasible_events > 0
                    ),
                }
            )
            paired_scores.to_csv(output / "paired_scores.csv", index=False)
            conditioning_replay.to_csv(output / "conditioning_replay.csv", index=False)
            write_json(output / "summary.json", summary)
            write_payoff_plot(paired_scores, output / "paired_payoff.png")
            write_drift_plot(
                conditioning_replay,
                output / "conditioning_drift.png",
                threshold,
            )

            full_conditioned_stream = pd.concat(
                [conditioned_context, test], ignore_index=True
            ).sort_values(["time_rel", "event_id"])
            defense_frames = []
            defense_summaries = {}
            conditioning_start = (
                pd.Timestamp(min(event["timestamp"] for event in plan.conditioning_events))
                if plan.conditioning_events
                else pd.Timestamp(payoff["timestamp"].min())
            )
            for method in ("ema", "cusum"):
                clean_defended, _ = monitored_replay(
                    validated,
                    clean_full_stream,
                    metadata,
                    config["defense"],
                    calibration,
                    method,
                    "clean",
                    target_node,
                )
                defended, _ = monitored_replay(
                    validated,
                    full_conditioned_stream,
                    metadata,
                    config["defense"],
                    calibration,
                    method,
                    "conditioned",
                    target_node,
                )
                method_decisions = pd.concat(
                    [clean_defended, defended], ignore_index=True
                )
                defense_frames.append(method_decisions)
                defense_summaries[method] = defense_summary(
                    method_decisions,
                    pd.Timestamp(payoff["timestamp"].min()),
                    conditioning_start,
                    target_node,
                )
                defended_payoff = defense_summaries[method][
                    "conditioned_payoff_anomaly_mean"
                ]
                defense_summaries[method]["payoff_recovery_absolute"] = float(
                    defended_payoff - summary["conditioned_mean"]
                )
                suppression_gap = summary["clean_mean"] - summary["conditioned_mean"]
                defense_summaries[method]["payoff_recovery_fraction"] = (
                    None
                    if abs(suppression_gap) < 1e-12
                    else float(
                        (defended_payoff - summary["conditioned_mean"])
                        / suppression_gap
                    )
                )
            defense_decisions = pd.concat(defense_frames, ignore_index=True)
            defense_decisions.to_csv(
                output / "defense_decisions.csv", index=False
            )
            write_json(output / "defense_summary.json", defense_summaries)

            # One auditable explanation packet per run. Attention is clearly
            # marked diagnostic; deletion/replay is the fidelity check.
            first_payoff = payoff.sort_values(["time_rel", "event_id"]).iloc[0]
            prepay_test = test[test["time_rel"] < first_payoff["time_rel"]]
            prepay_history = pd.concat(
                [conditioned_context, prepay_test], ignore_index=True
            ).sort_values(["time_rel", "event_id"])
            explanation_model = validated.fork()
            replay_frame(
                explanation_model,
                prepay_history,
                metadata,
                "explanation_history",
            )
            attention_explanation(
                explanation_model,
                first_payoff,
                prepay_history,
                metadata,
            ).to_csv(output / "attention_diagnostic.csv", index=False)
            candidate_ids = [
                str(event["event_id"]) for event in plan.conditioning_events[-5:]
            ]
            counterfactual_deletion_explanation(
                validated,
                prepay_history,
                first_payoff,
                metadata,
                candidate_ids,
            ).to_csv(output / "counterfactual_deletion.csv", index=False)
            write_json(
                output / "mitre_candidates.json",
                {
                    "claim_level": "analyst-context candidates, not ground truth",
                    "payoff_event_id": str(first_payoff["event_id"]),
                    "candidates": [
                        candidate.__dict__ for candidate in mitre_candidates(first_payoff)
                    ],
                },
            )
            summary["defense"] = defense_summaries
            write_json(output / "summary.json", summary)
            run_summaries.append(summary)
            all_scores.append(paired_scores.assign(duration_days=duration, rate=rate))

    validation_scores.to_csv(artifact_root / f"seed_{seed}/validation_scores.csv", index=False)
    if all_scores:
        pd.concat(all_scores, ignore_index=True).to_csv(
            artifact_root / f"seed_{seed}/all_scores.csv", index=False
        )
    seed_summary = {
        "seed": seed,
        "checkpoint_training": payload["training_summary"],
        "validation_threshold_q99": threshold,
        "runs": run_summaries,
    }
    write_json(artifact_root / f"seed_{seed}/study_summary.json", seed_summary)
    return seed_summary

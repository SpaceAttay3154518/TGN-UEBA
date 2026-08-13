"""Batched, all-incident CERT evaluation for the adapted KAIROS detector.

The implementation keeps KAIROS's causal mini-batch semantics, evaluates every
event for window construction, and retains only a deterministic benign event
sample plus all incident/target-user rows.  Drift ablations are one-factor-at-
a-time around a preregistered primary configuration; attack labels never tune
their calibration thresholds.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .defense import (
    DriftFeatures,
    FeatureExtractor,
    RoleReferenceBank,
    SequentialMonitor,
    fit_calibration,
)
from .io import PreparedDataset, load_checkpoint, resolve_config_path
from .kairos_protocol import WindowRecord, edge_scores_to_windows, evaluate_darpa_windows
from .model import Batch, iter_batches
from .replay import tensors_from_frame


def _base_configurations(config: dict) -> list[dict[str, str]]:
    defense = config["defense"]
    primary = {
        "role": defense["primary_role"],
        "reference": defense["primary_reference"],
        "distance": defense["primary_distance"],
    }
    candidates = [("primary", primary)]
    candidates.extend(
        (f"role={value}", {**primary, "role": value})
        for value in defense["ablation"]["roles"]
        if value != primary["role"]
    )
    candidates.extend(
        (f"reference={value}", {**primary, "reference": value})
        for value in defense["ablation"]["references"]
        if value != primary["reference"]
    )
    candidates.extend(
        (f"distance={value}", {**primary, "distance": value})
        for value in defense["ablation"]["distances"]
        if value != primary["distance"]
    )
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for name, values in candidates:
        signature = (values["role"], values["reference"], values["distance"])
        if signature in seen:
            continue
        seen.add(signature)
        output.append({"base_id": name, **values})
    return output


def _monitor_variants(config: dict, bases: list[dict[str, str]]) -> list[dict[str, str]]:
    defense = config["defense"]
    variants: list[dict[str, str]] = []
    for base in bases:
        formulas = (
            defense["ablation"]["formulas"]
            if base["base_id"] == "primary"
            else [defense["primary_formula"]]
        )
        monitors = (
            defense["ablation"]["monitors"]
            if base["base_id"] == "primary"
            else [defense["primary_monitor"]]
        )
        for formula in formulas:
            for monitor in monitors:
                variants.append(
                    {
                        **base,
                        "formula": str(formula),
                        "monitor": str(monitor),
                        "variant_id": f"{base['base_id']}|{formula}|{monitor}",
                    }
                )
    return variants


def _decision_alert(decision, monitor: str) -> bool:
    if monitor == "shewhart":
        return bool(decision.shewhart_alert)
    if monitor == "ema":
        return bool(decision.ema_alert)
    if monitor == "cusum":
        return bool(decision.cusum_alert)
    raise ValueError(f"Unknown monitor: {monitor}")


def _sample_benign(frame: pd.DataFrame, denominator: int = 100) -> np.ndarray:
    hashes = pd.util.hash_pandas_object(frame["event_id"], index=False).to_numpy(np.uint64)
    return (hashes % np.uint64(denominator)) == 0


def _window_to_dict(window: WindowRecord, split: str) -> dict:
    return {
        "split": split,
        "window_id": str(window.window_id),
        "sequence_id": str(window.sequence_id),
        "anomaly_score": float(window.anomaly_score),
        "anomalous_nodes": ";".join(sorted(window.anomalous_nodes)),
        "attack_label": bool(window.attack_label),
    }


class _VariantStats:
    def __init__(self) -> None:
        self.decisions = 0
        self.alerts = 0
        self.attack_batches = 0
        self.attack_batch_alerts = 0
        self.benign_entity_days: set[tuple[int, str]] = set()
        self.benign_alerts = 0
        self.benign_alert_entity_days: set[tuple[int, str]] = set()
        self.incident_attack: set[str] = set()
        self.incident_control: set[str] = set()
        self.first_detection: dict[str, pd.Timestamp] = {}


def _incident_indexes(catalog: dict, metadata: dict):
    by_node: dict[int, list[dict]] = defaultdict(list)
    for row in catalog["incidents"]:
        node_name = f"usr:{row['target_user']}"
        if node_name not in metadata["node_to_id"]:
            continue
        start = pd.Timestamp(row["start"])
        end = pd.Timestamp(row["end"])
        by_node[int(metadata["node_to_id"][node_name])].append(
            {
                **row,
                "start_ts": start,
                "end_ts": end,
                "attack_window_end": start + pd.Timedelta(hours=24),
                "control_start": start - pd.Timedelta(hours=24),
            }
        )
    return by_node


def _control_nodes(metadata: dict, target_nodes: set[int], count: int = 100) -> set[int]:
    users = sorted(
        int(metadata["node_to_id"][name])
        for name in metadata["role_profiles"]
        if name in metadata["node_to_id"]
        and int(metadata["node_to_id"][name]) not in target_nodes
    )
    # Stable multiplicative ordering avoids relying on Python's randomized hash.
    users.sort(key=lambda value: ((value + 1) * 2654435761) % 2**32)
    return set(users[:count])


def _feature_record(
    runtime: dict,
    node: int,
    timestamp: pd.Timestamp,
    event_anomaly: float,
    attack_label: bool,
    target_actor_label: bool,
    case_ids: str,
    current: torch.Tensor,
    update_role_scale: bool = True,
) -> tuple[DriftFeatures, dict]:
    bank: RoleReferenceBank = runtime["bank"]
    extractor: FeatureExtractor = runtime["extractor"]
    current = current.detach().clone()
    previous = runtime["previous"].get(node, current)
    runtime["previous"][node] = current
    try:
        reference = bank.reference(node)
        scale = bank.reference_scale(node)
        role = bank.role(node)
    except KeyError:
        reference = current
        scale = torch.ones_like(current)
        role = "entity:unknown"
    feature = extractor.step(
        node,
        role,
        float(timestamp.value / 1_000_000_000),
        previous,
        current,
        reference,
        event_anomaly,
        reference_scale=scale,
        update_role_scale=update_role_scale,
    )
    return feature, {
        "node_id": node,
        "timestamp": timestamp,
        "attack_label": attack_label,
        "target_actor_label": target_actor_label,
        "case_ids": case_ids,
        "event_anomaly": event_anomaly,
    }


def _update_variant_stats(
    stats: _VariantStats,
    meta: dict,
    alert: bool,
    incident_by_node: dict[int, list[dict]],
    record_false_alerts: bool,
) -> None:
    node = int(meta["node_id"])
    timestamp = pd.Timestamp(meta["timestamp"])
    stats.decisions += 1
    stats.alerts += int(alert)
    if bool(meta["target_actor_label"]):
        stats.attack_batches += 1
        stats.attack_batch_alerts += int(alert)
    inside_incident = False
    for case in incident_by_node.get(node, []):
        if case["start_ts"] <= timestamp <= case["end_ts"]:
            inside_incident = True
        if case["start_ts"] <= timestamp < case["attack_window_end"] and alert:
            stats.incident_attack.add(case["case_id"])
            prior = stats.first_detection.get(case["case_id"])
            if prior is None or timestamp < prior:
                stats.first_detection[case["case_id"]] = timestamp
        if case["control_start"] <= timestamp < case["start_ts"] and alert:
            stats.incident_control.add(case["case_id"])
    if record_false_alerts and not inside_incident and not bool(meta["attack_label"]):
        entity_day = (node, timestamp.strftime("%Y-%m-%d"))
        stats.benign_entity_days.add(entity_day)
        stats.benign_alerts += int(alert)
        if alert:
            stats.benign_alert_entity_days.add(entity_day)


@torch.no_grad()
def run_longitudinal_seed(
    seed: int,
    events: PreparedDataset,
    metadata: dict,
    config: dict,
) -> dict:
    if not isinstance(events, PreparedDataset):
        raise TypeError("Longitudinal evaluation requires a partitioned dataset")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and "gpu_memory_fraction" in config["model"]:
        torch.cuda.set_per_process_memory_fraction(
            float(config["model"]["gpu_memory_fraction"]),
            device=torch.cuda.current_device(),
        )
    checkpoint = (
        resolve_config_path(config, config["paths"]["checkpoints"])
        / f"tgn_seed_{seed}.pt"
    )
    # The defense configuration (weights, signals) is independent of the
    # trained model parameters.  Skip the full-config hash check so that
    # defense-only changes do not invalidate existing checkpoints.
    detector, payload = load_checkpoint(
        checkpoint, config, metadata, device, enforce_hash=False
    )
    # Released KAIROS starts inference from fresh temporal state.
    detector.eval()
    detector.reset_temporal_state()

    prepared_root = events.root
    catalog = json.loads((prepared_root / metadata["incident_catalog_file"]).read_text())
    incident_by_node = _incident_indexes(catalog, metadata)
    target_nodes = set(incident_by_node)
    control_nodes = _control_nodes(metadata, target_nodes)
    monitored_nodes = target_nodes | control_nodes

    bases = _base_configurations(config)
    variants = _monitor_variants(config, bases)
    defense = config["defense"]
    runtimes = {}
    banks: dict[tuple[str, str], RoleReferenceBank] = {}
    for base in bases:
        bank_key = (base["role"], base["reference"])
        if bank_key not in banks:
            banks[bank_key] = RoleReferenceBank(
                metadata["node_to_id"],
                metadata["role_profiles"],
                float(defense["shrinkage_kappa"]),
                int(defense["minimum_role_peers"]),
                role_method=base["role"],
                estimator=base["reference"],
            )
        runtimes[base["base_id"]] = {
            "bank": banks[bank_key],
            "extractor": FeatureExtractor(base["distance"]),
            "previous": {},
        }
    validation_features: dict[str, list[DriftFeatures]] = defaultdict(list)
    calibrations = {}
    monitors: dict[str, SequentialMonitor] = {}
    stats = {variant["variant_id"]: _VariantStats() for variant in variants}
    variant_lookup = {variant["variant_id"]: variant for variant in variants}
    variants_by_base: dict[str, list[dict]] = defaultdict(list)
    for variant in variants:
        variants_by_base[variant["base_id"]].append(variant)

    validation_scores: list[float] = []
    sampled_events: list[pd.DataFrame] = []
    windows: dict[str, list[WindowRecord]] = defaultdict(list)
    baseline_attack: set[str] = set()
    baseline_control: set[str] = set()
    baseline_first: dict[str, pd.Timestamp] = {}
    baseline_attack_events = baseline_attack_alerts = 0
    baseline_benign_events = baseline_benign_alerts = 0
    event_threshold = None
    batch_size = int(config["model"]["batch_size"])
    study_started = time.perf_counter()
    partition_key = str(metadata.get("partition_key", "day"))
    split_partition_totals = {
        split: len(
            list(
                (events.root / "events" / f"split={split}").glob(
                    f"{partition_key}=*"
                )
            )
        )
        for split in ("validation", "context", "evaluation")
    }
    split_event_counts: dict[str, int] = defaultdict(int)

    for split in ("validation", "context", "evaluation"):
        for frame_index, frame in enumerate(events.iter_split(split)):
            memory_snapshot = detector.memory.memory.detach().cpu()
            for bank in banks.values():
                bank.update(memory_snapshot)
            src, dst, timestamp, message, edge_type = tensors_from_frame(frame, metadata)
            day_scores = np.empty(len(frame), dtype=np.float32)
            feature_rows: dict[str, list[tuple[DriftFeatures, dict]]] = defaultdict(list)
            for start in range(0, len(frame), batch_size):
                stop = min(start + batch_size, len(frame))
                batch = Batch(
                    src=src[start:stop],
                    dst=dst[start:stop],
                    t=timestamp[start:stop],
                    msg=message[start:stop],
                    edge_type=edge_type[start:stop],
                ).to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda" and bool(config["model"].get("mixed_precision", False)),
                ):
                    anomaly = detector.anomaly(batch)
                anomaly_np = anomaly.float().cpu().numpy()
                day_scores[start:stop] = anomaly_np
                batch_frame = frame.iloc[start:stop]
                unique_nodes = batch_frame["src_id"].astype(int).unique()
                if split == "validation":
                    queried_nodes = [int(node) for node in unique_nodes]
                else:
                    queried_nodes = [
                        int(node) for node in unique_nodes if int(node) in monitored_nodes
                    ]
                current_by_node: dict[int, torch.Tensor] = {}
                if queried_nodes:
                    query = torch.tensor(queried_nodes, dtype=torch.long, device=device)
                    updated, _ = detector.memory(query)
                    updated = updated.detach().cpu()
                    current_by_node = {
                        node: updated[position]
                        for position, node in enumerate(queried_nodes)
                    }
                source_ids = batch_frame["src_id"].to_numpy(np.int64)
                timestamps = batch_frame["timestamp"].to_numpy()
                attack_labels = batch_frame["attack_label"].to_numpy(bool)
                target_labels = batch_frame["target_actor_label"].to_numpy(bool)
                case_ids = batch_frame["case_id"].astype(str).to_numpy()
                node_inputs = {}
                for node in queried_nodes:
                    positions = np.flatnonzero(source_ids == node)
                    node_inputs[node] = {
                        "timestamp": pd.Timestamp(timestamps[positions].max()),
                        "event_anomaly": float(np.max(anomaly_np[positions])),
                        "attack_label": bool(attack_labels[positions].any()),
                        "target_actor_label": bool(target_labels[positions].any()),
                        "case_ids": ";".join(
                            sorted(
                                value
                                for value in np.unique(case_ids[positions])
                                if value
                            )
                        ),
                    }
                for base in bases:
                    if split == "validation":
                        selected = queried_nodes
                    else:
                        # All ablations use the identical target/control
                        # population. Restricting non-primary bases to targets
                        # would make their false-alert burden incomparable.
                        selected = [node for node in unique_nodes if node in monitored_nodes]
                    runtime = runtimes[base["base_id"]]
                    for node in selected:
                        values = node_inputs[int(node)]
                        feature, meta = _feature_record(
                            runtime,
                            int(node),
                            values["timestamp"],
                            values["event_anomaly"],
                            values["attack_label"],
                            values["target_actor_label"],
                            values["case_ids"],
                            current_by_node[int(node)],
                            update_role_scale=(
                                split == "validation" or int(node) not in target_nodes
                            ),
                        )
                        feature_rows[base["base_id"]].append((feature, meta))
                detector.observe(batch)
            detector.compact_history()

            scored = frame.copy()
            scored["anomaly_score"] = day_scores
            windows[split].extend(edge_scores_to_windows(scored))
            if split == "validation":
                validation_scores.extend(float(value) for value in day_scores)
                for base_id, rows in feature_rows.items():
                    validation_features[base_id].extend(feature for feature, _ in rows)
            else:
                if event_threshold is None:
                    raise RuntimeError("Event threshold was not calibrated")
                for base_id, rows in feature_rows.items():
                    for feature, meta in rows:
                        # The defense is COMPLEMENTARY to plain KAIROS: it can
                        # only add detections, never remove them.  The combined
                        # alert fires when either the KAIROS event-level score
                        # exceeds its calibrated threshold OR the drift monitor
                        # raises an alert.
                        kairos_alert = float(meta["event_anomaly"]) > float(event_threshold)
                        for variant in variants_by_base[base_id]:
                            decision = monitors[variant["variant_id"]].step(feature)
                            drift_alert = _decision_alert(decision, variant["monitor"])
                            alert = kairos_alert or drift_alert
                            _update_variant_stats(
                                stats[variant["variant_id"]],
                                meta,
                                alert,
                                incident_by_node,
                                record_false_alerts=split == "evaluation",
                            )

            if split in {"context", "evaluation"}:
                event_alerts = day_scores > float(event_threshold)
                target_mask = frame["src_id"].astype(int).isin(target_nodes).to_numpy()
                for position in np.flatnonzero(target_mask):
                    row = frame.iloc[position]
                    node = int(row["src_id"])
                    ts = pd.Timestamp(row["timestamp"])
                    alert = bool(event_alerts[position])
                    for case in incident_by_node.get(node, []):
                        if case["start_ts"] <= ts < case["attack_window_end"] and alert:
                            baseline_attack.add(case["case_id"])
                            prior = baseline_first.get(case["case_id"])
                            if prior is None or ts < prior:
                                baseline_first[case["case_id"]] = ts
                        if case["control_start"] <= ts < case["start_ts"] and alert:
                            baseline_control.add(case["case_id"])

            if split == "evaluation":
                target_actor = frame["target_actor_label"].to_numpy(bool)
                benign = ~frame["attack_label"].to_numpy(bool)
                baseline_attack_events += int(target_actor.sum())
                baseline_attack_alerts += int((event_alerts & target_actor).sum())
                baseline_benign_events += int(benign.sum())
                baseline_benign_alerts += int((event_alerts & benign).sum())
                benign_sample = _sample_benign(frame)
                keep = (
                    frame["attack_label"].to_numpy(bool)
                    | target_mask
                    | (benign_sample & benign)
                )
                sample = scored.loc[
                    keep,
                    [
                        "event_id", "timestamp", "src", "dst", "src_id",
                        "event_type", "attack_label", "target_actor_label",
                        "case_id", "attack_scenario", "anomaly_score",
                    ],
                ].copy()
                sample["event_metric_eligible"] = (
                    sample["attack_label"].to_numpy(bool)
                    | benign_sample[keep]
                )
                sample["benign_sampling_probability"] = np.where(
                    sample["attack_label"],
                    1.0,
                    0.01,
                )
                sampled_events.append(sample)

            split_event_counts[split] += len(frame)
            partitions_complete = frame_index + 1
            if (
                split != "evaluation"
                or partitions_complete % 5 == 0
                or partitions_complete == split_partition_totals[split]
            ):
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "stage": "longitudinal_replay",
                            "split": split,
                            "partitions_complete": partitions_complete,
                            "partitions_total": split_partition_totals[split],
                            "split_events_complete": split_event_counts[split],
                            "elapsed_seconds": time.perf_counter() - study_started,
                        }
                    ),
                    flush=True,
                )

        if split == "validation":
            event_threshold = float(np.quantile(validation_scores, 0.99))
            for variant in variants:
                key = (variant["base_id"], variant["formula"])
                if key not in calibrations:
                    calibrations[key] = fit_calibration(
                        validation_features[variant["base_id"]],
                        defense["weights"],
                        float(defense["ema_alpha"]),
                        float(defense["cusum_allowance_quantile"]),
                        float(defense["alert_quantile"]),
                        formula=variant["formula"],
                    )
                monitors[variant["variant_id"]] = SequentialMonitor(
                    calibrations[key],
                    defense["weights"],
                    float(defense["ema_alpha"]),
                    formula=variant["formula"],
                )

    kairos_windows = evaluate_darpa_windows(
        windows["validation"], windows["context"], windows["evaluation"]
    )
    event_sample = pd.concat(sampled_events, ignore_index=True)
    metric_sample = event_sample[event_sample["event_metric_eligible"]]
    labels = metric_sample["attack_label"].to_numpy(np.int8)
    values = metric_sample["anomaly_score"].to_numpy(float)
    probabilities = metric_sample["benign_sampling_probability"].to_numpy(float)
    sample_weight = 1.0 / probabilities
    event_metrics = {
        "roc_auc": float(roc_auc_score(labels, values, sample_weight=sample_weight)),
        "average_precision_prevalence_corrected": float(
            average_precision_score(labels, values, sample_weight=sample_weight)
        ),
        "sampled_rows": int(len(metric_sample)),
        "positive_rows": int(labels.sum()),
        "benign_sampling_probability": 0.01,
        "threshold_q99_validation": float(event_threshold),
        "target_actor_recall_q99": baseline_attack_alerts / max(baseline_attack_events, 1),
        "benign_event_fpr_q99": baseline_benign_alerts / max(baseline_benign_events, 1),
    }
    all_cases = {row["case_id"] for row in catalog["incidents"]}
    baseline_incidents = {
        "attack_windows_detected": len(baseline_attack),
        "attack_windows_total": len(all_cases),
        "attack_window_recall": len(baseline_attack) / len(all_cases),
        "matched_pre24h_control_alerts": len(baseline_control),
        "matched_pre24h_controls": len(all_cases),
        "matched_control_rate": len(baseline_control) / len(all_cases),
        "net_detection_rate": (len(baseline_attack) - len(baseline_control)) / len(all_cases),
        "first_detection": {key: value.isoformat() for key, value in baseline_first.items()},
    }
    variant_rows = []
    for variant_id, item in stats.items():
        definition = variant_lookup[variant_id]
        entity_days = len(item.benign_entity_days)
        variant_rows.append(
            {
                **definition,
                "decisions": item.decisions,
                "alerts": item.alerts,
                "attack_batch_recall": item.attack_batch_alerts / max(item.attack_batches, 1),
                "benign_alerts_per_1000_entity_days": len(item.benign_alert_entity_days) / max(entity_days, 1) * 1000,
                "benign_alert_decisions_per_1000_entity_days": item.benign_alerts / max(entity_days, 1) * 1000,
                "benign_monitored_entity_days": entity_days,
                "attack_windows_detected": len(item.incident_attack),
                "attack_window_recall": len(item.incident_attack) / len(all_cases),
                "matched_control_alerts": len(item.incident_control),
                "matched_control_rate": len(item.incident_control) / len(all_cases),
                "net_detection_rate": (len(item.incident_attack) - len(item.incident_control)) / len(all_cases),
                "first_detection": json.dumps(
                    {key: value.isoformat() for key, value in item.first_detection.items()},
                    sort_keys=True,
                ),
            }
        )
    case_rows = []
    for case in catalog["incidents"]:
        case_id = case["case_id"]
        base_row = {
            "seed": seed,
            "case_id": case_id,
            "scenario": int(case["scenario"]),
            "target_user": case["target_user"],
            "method": "plain_adapted_kairos_q99",
            "attack_window_alert": case_id in baseline_attack,
            "matched_control_alert": case_id in baseline_control,
            "first_detection": (
                None
                if case_id not in baseline_first
                else baseline_first[case_id].isoformat()
            ),
        }
        case_rows.append(base_row)
        for variant_id, item in stats.items():
            case_rows.append(
                {
                    **{key: value for key, value in base_row.items() if key not in {"method", "attack_window_alert", "matched_control_alert", "first_detection"}},
                    "method": variant_id,
                    "attack_window_alert": case_id in item.incident_attack,
                    "matched_control_alert": case_id in item.incident_control,
                    "first_detection": (
                        None
                        if case_id not in item.first_detection
                        else item.first_detection[case_id].isoformat()
                    ),
                }
            )
    output = resolve_config_path(config, config["paths"]["artifacts"]) / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    event_sample.to_parquet(output / "event_score_sample.parquet", index=False, compression="zstd")
    pd.DataFrame(
        [_window_to_dict(item, split) for split, items in windows.items() for item in items]
    ).to_parquet(output / "kairos_windows.parquet", index=False, compression="zstd")
    pd.DataFrame(variant_rows).to_csv(output / "defense_ablation.csv", index=False)
    pd.DataFrame(case_rows).to_csv(output / "case_level_results.csv", index=False)
    result = {
        "seed": seed,
        "checkpoint_training": payload["training_summary"],
        "event_metrics": event_metrics,
        "baseline_incidents": baseline_incidents,
        "kairos_window_protocol": {
            key: value for key, value in kairos_windows.items() if key != "graph_scores"
        },
        "defense_variants": variant_rows,
        "evaluation_scope": {
            "incidents": len(all_cases),
            "target_users": len(target_nodes),
            "benign_control_users": len(control_nodes),
            "all_events_used_for_window_evaluation": True,
            "inference_temporal_state": "reset then validation -> context -> evaluation",
            "drift_decision_unit": "entity within causal KAIROS mini-batch",
        },
    }
    (output / "longitudinal_summary.json").write_text(json.dumps(result, indent=2))
    return result


def run_longitudinal_study(
    events: PreparedDataset,
    metadata: dict,
    config: dict,
    seeds: list[int] | None = None,
) -> list[dict]:
    selected = seeds or [int(value) for value in config["model_seeds"]]
    return [run_longitudinal_seed(seed, events, metadata, config) for seed in selected]

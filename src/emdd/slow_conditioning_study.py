"""Realizable slow-conditioning stress test around a fixed CERT incident.

The stressors replay role-observed actions through the ordinary KAIROS message
path.  They are deliberately simple and auditable (common-role and payoff-type
matched schedules), not claimed to be optimal attacks.  Clean, conditioned,
and gated branches start from the same temporal snapshot.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .attack import (
    build_action_library,
    build_payoff_matched_library,
    conditioning_slots,
    materialize_action,
)
from .defense import FeatureExtractor, RoleReferenceBank, SequentialMonitor, fit_calibration
from .io import PreparedDataset, load_checkpoint, resolve_config_path
from .longitudinal_study import _decision_alert, _feature_record
from .model import Batch
from .replay import tensors_from_frame
from .study import role_peer_names


def _runtime(metadata: dict, defense: dict) -> dict:
    return {
        "bank": RoleReferenceBank(
            metadata["node_to_id"],
            metadata["role_profiles"],
            float(defense["shrinkage_kappa"]),
            int(defense["minimum_role_peers"]),
            role_method=defense["primary_role"],
            estimator=defense["primary_reference"],
        ),
        "extractor": FeatureExtractor(defense["primary_distance"]),
        "previous": {},
    }


def _batch(
    src, dst, timestamp, message, edge_type, start: int, stop: int, device
) -> Batch:
    return Batch(
        src=src[start:stop],
        dst=dst[start:stop],
        t=timestamp[start:stop],
        msg=message[start:stop],
        edge_type=edge_type[start:stop],
    ).to(device)


@torch.no_grad()
def _calibrate_and_advance(
    detector,
    events: PreparedDataset,
    metadata: dict,
    config: dict,
    target_node: int,
    branch_start: pd.Timestamp,
):
    defense = config["defense"]
    runtime = _runtime(metadata, defense)
    validation_features = []
    validation_scores = []
    batch_size = int(config["model"]["batch_size"])
    for frame in events.iter_split("validation"):
        runtime["bank"].update(detector.memory.memory.detach().cpu())
        src, dst, timestamp, message, edge_type = tensors_from_frame(frame, metadata)
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            batch = _batch(src, dst, timestamp, message, edge_type, start, stop, detector.device_ref)
            with torch.autocast(
                device_type=detector.device_ref.type,
                dtype=torch.bfloat16,
                enabled=detector.device_ref.type == "cuda" and bool(config["model"].get("mixed_precision", False)),
            ):
                anomaly = detector.anomaly(batch)
            anomaly_np = anomaly.float().cpu().numpy()
            validation_scores.extend(float(value) for value in anomaly_np)
            rows = frame.iloc[start:stop]
            nodes = [int(value) for value in rows["src_id"].unique()]
            query = torch.tensor(nodes, dtype=torch.long, device=detector.device_ref)
            current, _ = detector.memory(query)
            current = current.detach().cpu()
            for position, node in enumerate(nodes):
                mask = rows["src_id"].to_numpy(np.int64) == node
                selected = rows.loc[mask]
                feature, _ = _feature_record(
                    runtime,
                    node,
                    pd.Timestamp(selected["timestamp"].max()),
                    float(np.max(anomaly_np[mask])),
                    False,
                    False,
                    "",
                    current[position],
                )
                validation_features.append(feature)
            detector.observe(batch)
        detector.compact_history()
    threshold = float(np.quantile(validation_scores, 0.99))
    calibration = fit_calibration(
        validation_features,
        defense["weights"],
        float(defense["ema_alpha"]),
        float(defense["cusum_allowance_quantile"]),
        float(defense["alert_quantile"]),
        formula=defense["primary_formula"],
    )
    monitor = SequentialMonitor(
        calibration,
        defense["weights"],
        float(defense["ema_alpha"]),
        formula=defense["primary_formula"],
    )

    for split in ("context", "evaluation"):
        finished = False
        for frame in events.iter_split(split):
            selected = frame[frame["timestamp"] < branch_start]
            if selected.empty:
                if pd.Timestamp(frame["timestamp"].min()) >= branch_start:
                    finished = True
                    break
                continue
            runtime["bank"].update(detector.memory.memory.detach().cpu())
            src, dst, timestamp, message, edge_type = tensors_from_frame(selected, metadata)
            for start in range(0, len(selected), batch_size):
                stop = min(start + batch_size, len(selected))
                batch = _batch(src, dst, timestamp, message, edge_type, start, stop, detector.device_ref)
                with torch.autocast(
                    device_type=detector.device_ref.type,
                    dtype=torch.bfloat16,
                    enabled=detector.device_ref.type == "cuda" and bool(config["model"].get("mixed_precision", False)),
                ):
                    anomaly = detector.anomaly(batch)
                rows = selected.iloc[start:stop]
                target = rows["src_id"].to_numpy(np.int64) == target_node
                if bool(target.any()):
                    query = torch.tensor([target_node], device=detector.device_ref)
                    current, _ = detector.memory(query)
                    current = current.detach().cpu()
                    target_rows = rows.loc[target]
                    feature, _ = _feature_record(
                        runtime,
                        target_node,
                        pd.Timestamp(target_rows["timestamp"].max()),
                        float(anomaly.float().cpu().numpy()[target].max()),
                        False,
                        False,
                        "",
                        current[0],
                        update_role_scale=False,
                    )
                    monitor.step(feature)
                detector.observe(batch)
            detector.compact_history()
        if finished:
            break
    runtime["bank"].update(detector.memory.memory.detach().cpu())
    return detector, runtime, monitor, threshold


def _load_range(
    events: PreparedDataset, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    frames = []
    for frame in events.iter_split("evaluation"):
        minimum = pd.Timestamp(frame["timestamp"].min())
        maximum = pd.Timestamp(frame["timestamp"].max())
        if maximum < start:
            continue
        if minimum > end:
            break
        selected = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
        if not selected.empty:
            frames.append(selected)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["timestamp_s", "event_id"], kind="stable"
    ).reset_index(drop=True)


def _injections(
    template,
    policy: str,
    duration: int,
    rate: int,
    payoff_start: pd.Timestamp,
    target_node: int,
    target_name: str,
    metadata: dict,
    columns: list[str],
) -> pd.DataFrame:
    rows = []
    for ordinal, timestamp in enumerate(conditioning_slots(payoff_start, duration, rate)):
        selected = template[ordinal % len(template)]
        row = materialize_action(
            selected,
            target_node,
            target_name,
            timestamp,
            int(metadata["time_origin_s"]),
            metadata,
            ordinal,
        ).to_dict()
        row.update(
            {
                "target_actor_label": False,
                "case_id": "",
                "attack_scenario": 0,
                "case_target_user": "",
                "answer_actor": "",
                "policy": policy,
            }
        )
        rows.append(row)
    output = pd.DataFrame(rows)
    for column in columns:
        if column not in output:
            output[column] = False if column in {"attack_label", "target_actor_label"} else ""
    return output[columns]


@torch.no_grad()
def _replay_branch(
    start_detector,
    start_runtime: dict,
    start_monitor,
    stream: pd.DataFrame,
    metadata: dict,
    config: dict,
    target_node: int,
    case_id: str,
    defended: bool,
) -> pd.DataFrame:
    detector = start_detector.fork()
    runtime = copy.deepcopy(start_runtime)
    monitor = copy.deepcopy(start_monitor)
    defense = config["defense"]
    monitor_name = defense["primary_monitor"]
    batch_size = int(config["model"]["batch_size"])
    output = []
    ordered = stream.sort_values(["timestamp_s", "event_id"], kind="stable")
    ordered["day"] = pd.to_datetime(ordered["timestamp"]).dt.strftime("%Y-%m-%d")
    for _, frame in ordered.groupby("day", sort=True):
        runtime["bank"].update(detector.memory.memory.detach().cpu())
        src, dst, timestamp, message, edge_type = tensors_from_frame(frame, metadata)
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            batch = _batch(src, dst, timestamp, message, edge_type, start, stop, detector.device_ref)
            with torch.autocast(
                device_type=detector.device_ref.type,
                dtype=torch.bfloat16,
                enabled=detector.device_ref.type == "cuda" and bool(config["model"].get("mixed_precision", False)),
            ):
                anomaly = detector.anomaly(batch)
            anomaly_np = anomaly.float().cpu().numpy()
            rows = frame.iloc[start:stop]
            target = rows["src_id"].to_numpy(np.int64) == target_node
            alert = False
            decision = None
            candidate = None
            if bool(target.any()):
                query = torch.tensor([target_node], device=detector.device_ref)
                candidate, before, after = detector.candidate_transition(batch, query)
                target_rows = rows.loc[target]
                bank = runtime["bank"]
                extractor = runtime["extractor"]
                try:
                    reference = bank.reference(target_node)
                    reference_scale = bank.reference_scale(target_node)
                    role = bank.role(target_node)
                except KeyError:
                    reference = before[0].detach().cpu()
                    reference_scale = torch.ones_like(reference)
                    role = "entity:unknown"
                feature = extractor.step(
                    target_node,
                    role,
                    float(pd.Timestamp(target_rows["timestamp"].max()).value / 1_000_000_000),
                    before[0].detach().cpu(),
                    after[0].detach().cpu(),
                    reference,
                    float(anomaly_np[target].max()),
                    reference_scale=reference_scale,
                    update_role_scale=False,
                )
                decision = monitor.step(feature)
                alert = _decision_alert(decision, monitor_name)
            accepted = np.ones(len(rows), dtype=bool)
            if defended and alert:
                accepted[target] = False
            if candidate is not None and (not defended or not alert):
                detector = candidate
            elif bool(accepted.any()):
                mask = torch.from_numpy(accepted).to(detector.device_ref)
                detector.observe(
                    Batch(
                        src=batch.src[mask],
                        dst=batch.dst[mask],
                        t=batch.t[mask],
                        msg=batch.msg[mask],
                        edge_type=batch.edge_type[mask] if batch.edge_type is not None else None,
                    )
                )
            retained = (
                rows["event_id"].astype(str).str.startswith("conditioning:")
                | (rows["case_id"] == case_id)
                | target
            )
            for position in np.flatnonzero(retained.to_numpy()):
                row = rows.iloc[position]
                output.append(
                    {
                        "event_id": row["event_id"],
                        "timestamp": row["timestamp"],
                        "case_id": row["case_id"],
                        "target_actor_label": bool(row["target_actor_label"]),
                        "is_conditioning": str(row["event_id"]).startswith("conditioning:"),
                        "anomaly_score": float(anomaly_np[position]),
                        "alert": alert and bool(target[position]),
                        "update_accepted": bool(accepted[position]),
                        "combined_signal": None if decision is None else decision.combined_signal,
                    }
                )
        detector.compact_history()
    return pd.DataFrame(output)


def run_slow_conditioning_seed(
    seed: int,
    events: PreparedDataset,
    metadata: dict,
    config: dict,
) -> dict:
    study_started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and "gpu_memory_fraction" in config["model"]:
        torch.cuda.set_per_process_memory_fraction(
            float(config["model"]["gpu_memory_fraction"]), torch.cuda.current_device()
        )
    checkpoint = resolve_config_path(config, config["paths"]["checkpoints"]) / f"tgn_seed_{seed}.pt"
    detector, _ = load_checkpoint(checkpoint, config, metadata, device)
    detector.eval()
    detector.reset_temporal_state()
    catalog = json.loads((events.root / metadata["incident_catalog_file"]).read_text())
    case_id = config["attack"]["sentinel_case"]
    case = next(row for row in catalog["incidents"] if row["case_id"] == case_id)
    target_name = f"usr:{case['target_user']}"
    target_node = int(metadata["node_to_id"][target_name])
    payoff_start = pd.Timestamp(case["start"])
    payoff_end = pd.Timestamp(case["end"])
    branch_start = payoff_start - pd.Timedelta(days=max(config["attack"]["durations_days"]))
    detector, runtime, monitor, threshold = _calibrate_and_advance(
        detector, events, metadata, config, target_node, branch_start
    )
    real_stream = _load_range(events, branch_start, payoff_end)
    payoff = real_stream[
        (real_stream["case_id"] == case_id) & real_stream["target_actor_label"]
    ]
    if payoff.empty:
        raise RuntimeError(f"No target-actor payoff rows found for {case_id}")

    peers = role_peer_names(metadata, target_name)
    peer_events = events.role_events(("train", "validation"), peers)
    templates = build_action_library(
        peer_events,
        metadata,
        peers,
        int(config["attack"]["candidate_pool_size"]),
    )
    payoff_matched = build_payoff_matched_library(
        peer_events,
        metadata,
        peers,
        payoff,
        int(config["attack"]["candidate_pool_size"]),
    )
    policies = {
        "role_common": templates[:1],
        "role_diverse": templates[: min(5, len(templates))],
        "payoff_type_matched": payoff_matched,
    }
    clean = _replay_branch(
        detector, runtime, monitor, real_stream, metadata, config, target_node, case_id, False
    )
    clean_gated = _replay_branch(
        detector, runtime, monitor, real_stream, metadata, config, target_node, case_id, True
    )
    clean_payoff = clean[clean["target_actor_label"]].set_index("event_id")["anomaly_score"]
    clean_gated_payoff = clean_gated[clean_gated["target_actor_label"]].set_index(
        "event_id"
    )["anomaly_score"]
    clean_pre_payoff = clean[
        (pd.to_datetime(clean["timestamp"]) < payoff_start)
        & ~clean["target_actor_label"]
    ]
    clean_gated_pre_payoff = clean_gated[
        (pd.to_datetime(clean_gated["timestamp"]) < payoff_start)
        & ~clean_gated["target_actor_label"]
    ]
    clean_alert_rows = clean_pre_payoff[clean_pre_payoff["alert"]]
    first_clean_alert = (
        None
        if clean_alert_rows.empty
        else pd.to_datetime(clean_alert_rows["timestamp"]).min()
    )
    rows = []
    detailed = [
        clean.assign(
            condition="clean", seed=seed, policy="none", duration_days=0,
            interactions_per_day=0,
        ),
        clean_gated.assign(
            condition="clean_gated", seed=seed, policy="none", duration_days=0,
            interactions_per_day=0,
        ),
    ]
    for policy, policy_templates in policies.items():
        for duration in config["attack"]["durations_days"]:
            for rate in config["attack"]["interactions_per_day"]:
                injected = _injections(
                    policy_templates,
                    policy,
                    int(duration),
                    int(rate),
                    payoff_start,
                    target_node,
                    target_name,
                    metadata,
                    list(real_stream.columns),
                )
                stream = pd.concat([real_stream, injected], ignore_index=True).sort_values(
                    ["timestamp_s", "event_id"], kind="stable"
                )
                attacked = _replay_branch(
                    detector, runtime, monitor, stream, metadata, config, target_node, case_id, False
                )
                defended = _replay_branch(
                    detector, runtime, monitor, stream, metadata, config, target_node, case_id, True
                )
                attack_payoff = attacked[attacked["target_actor_label"]].set_index("event_id")["anomaly_score"]
                defense_payoff = defended[defended["target_actor_label"]].set_index("event_id")["anomaly_score"]
                paired = pd.concat(
                    [
                        clean_payoff.rename("clean"),
                        clean_gated_payoff.rename("clean_gated"),
                        attack_payoff.rename("conditioned"),
                        defense_payoff.rename("defended"),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()
                conditioning = attacked[attacked["is_conditioning"]]
                defended_conditioning = defended[defended["is_conditioning"]]
                conditioning_alert_rows = defended_conditioning[
                    defended_conditioning["alert"]
                ]
                first_conditioning_alert = (
                    None
                    if conditioning_alert_rows.empty
                    else pd.to_datetime(conditioning_alert_rows["timestamp"]).min()
                )
                conditioning_delta = float(
                    (paired["conditioned"] - paired["clean"]).mean()
                )
                stealth_fraction = float(
                    (conditioning["anomaly_score"] < threshold).mean()
                )
                payoff_suppression = conditioning_delta < 0
                gating_main_effect = float(
                    (paired["clean_gated"] - paired["clean"]).mean()
                )
                gated_conditioning_delta = float(
                    (paired["defended"] - paired["clean_gated"]).mean()
                )
                defense_recovery = gated_conditioning_delta - conditioning_delta
                result = {
                    "seed": seed,
                    "policy": policy,
                    "duration_days": int(duration),
                    "interactions_per_day": int(rate),
                    "conditioning_events": int(len(conditioning)),
                    "conditioning_subthreshold_fraction": stealth_fraction,
                    "clean_payoff_mean": float(paired["clean"].mean()),
                    "clean_gated_payoff_mean": float(paired["clean_gated"].mean()),
                    "conditioned_payoff_mean": float(paired["conditioned"].mean()),
                    "defended_payoff_mean": float(paired["defended"].mean()),
                    "conditioning_delta": conditioning_delta,
                    "gating_main_effect": gating_main_effect,
                    "gated_conditioning_delta": gated_conditioning_delta,
                    "defense_recovery_difference_in_differences": defense_recovery,
                    "payoff_suppression": bool(payoff_suppression),
                    "stealth_qualified_success": bool(
                        payoff_suppression and stealth_fraction == 1.0
                    ),
                    "clean_prepayoff_alert": first_clean_alert is not None,
                    "first_clean_prepayoff_alert": (
                        None if first_clean_alert is None else first_clean_alert.isoformat()
                    ),
                    "detected_on_conditioning": first_conditioning_alert is not None,
                    "first_conditioning_alert": (
                        None
                        if first_conditioning_alert is None
                        else first_conditioning_alert.isoformat()
                    ),
                    "incremental_prepayoff_detection": bool(
                        first_conditioning_alert is not None
                        and (
                            first_clean_alert is None
                            or first_conditioning_alert < first_clean_alert
                        )
                    ),
                    "conditioning_updates_rejected": int((~defended_conditioning["update_accepted"]).sum()),
                    "conditioning_rejection_fraction": float(
                        (~defended_conditioning["update_accepted"]).mean()
                    ),
                    "clean_updates_rejected_before_payoff": int(
                        (~clean_gated_pre_payoff["update_accepted"]).sum()
                    ),
                    "validation_event_q99": threshold,
                    "payoff_pairs": int(len(paired)),
                }
                rows.append(result)
                detailed.append(attacked.assign(condition="conditioned", **{key: result[key] for key in ("seed", "policy", "duration_days", "interactions_per_day")}))
                detailed.append(defended.assign(condition="defended", **{key: result[key] for key in ("seed", "policy", "duration_days", "interactions_per_day")}))
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "stage": "slow_conditioning",
                            "policy": policy,
                            "duration_days": int(duration),
                            "interactions_per_day": int(rate),
                            "configurations_complete": len(rows),
                            "configurations_total": (
                                len(policies)
                                * len(config["attack"]["durations_days"])
                                * len(config["attack"]["interactions_per_day"])
                            ),
                            "elapsed_seconds": time.perf_counter() - study_started,
                        }
                    ),
                    flush=True,
                )
    output = resolve_config_path(config, config["paths"]["artifacts"]) / f"seed_{seed}" / "slow_conditioning"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    pd.concat(detailed, ignore_index=True).to_parquet(
        output / "retained_event_scores.parquet", index=False, compression="zstd"
    )
    summary = {
        "seed": seed,
        "case_id": case_id,
        "target_actor_payoff_events": int(len(payoff)),
        "policies": {
            name: [
                {
                    "event_type": item.event_type,
                    "destination": item.dst,
                    "training_support": item.support,
                }
                for item in values
            ]
            for name, values in policies.items()
        },
        "runs": rows,
        "claim": (
            "auditable role-observed conditioning stressors; payoff suppression "
            "and all-injection q99 stealth are reported separately; four matched "
            "branches identify the clean gate main effect; not an optimal attack"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_slow_conditioning_study(events, metadata, config, seeds=None):
    selected = seeds or [int(value) for value in config["model_seeds"]]
    return [run_slow_conditioning_seed(seed, events, metadata, config) for seed in selected]

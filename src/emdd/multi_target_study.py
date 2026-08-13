"""Multi-target slow-conditioning study.

Extends ``slow_conditioning_study`` to iterate over multiple CERT insider
cases and (optionally) multiple attack scenarios.  The per-seed, per-case
core logic is identical; only the target resolution and output paths differ.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .attack import (
    build_action_library,
    build_payoff_matched_library,
)
from .io import PreparedDataset, load_checkpoint, resolve_config_path
from .slow_conditioning_study import (
    _calibrate_and_advance,
    _injections,
    _load_range,
    _replay_branch,
)
from .study import role_peer_names


def _resolve_case(catalog: dict, case_id: str) -> dict:
    """Look up an incident in the catalog by case_id."""
    for entry in catalog["incidents"]:
        if entry["case_id"] == case_id:
            return entry
    raise KeyError(f"Case {case_id!r} not found in incident catalog")


def run_case_seed(
    seed: int,
    case_id: str,
    events: PreparedDataset,
    metadata: dict,
    config: dict,
    catalog: dict,
) -> dict:
    """Run the slow-conditioning study for one seed × one case."""
    study_started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and "gpu_memory_fraction" in config["model"]:
        torch.cuda.set_per_process_memory_fraction(
            float(config["model"]["gpu_memory_fraction"]), torch.cuda.current_device()
        )

    case = _resolve_case(catalog, case_id)
    target_name = f"usr:{case['target_user']}"
    if target_name not in metadata["node_to_id"]:
        return {
            "seed": seed,
            "case_id": case_id,
            "status": "skipped",
            "reason": f"Target node {target_name!r} not in prepared graph",
        }
    target_node = int(metadata["node_to_id"][target_name])
    payoff_start = pd.Timestamp(case["start"])
    payoff_end = pd.Timestamp(case["end"])

    # Load model checkpoint
    checkpoint = resolve_config_path(config, config["paths"]["checkpoints"]) / f"tgn_seed_{seed}.pt"
    detector, _ = load_checkpoint(checkpoint, config, metadata, device)
    detector.eval()
    detector.reset_temporal_state()

    branch_start = payoff_start - pd.Timedelta(days=max(config["attack"]["durations_days"]))

    # Calibrate and advance to branch point
    detector, runtime, monitor, threshold = _calibrate_and_advance(
        detector, events, metadata, config, target_node, branch_start
    )

    # Load the event range for replay
    real_stream = _load_range(events, branch_start, payoff_end)
    payoff = real_stream[
        (real_stream["case_id"] == case_id) & real_stream["target_actor_label"]
    ]

    if payoff.empty:
        return {
            "seed": seed,
            "case_id": case_id,
            "status": "skipped",
            "reason": f"No target-actor payoff rows found for {case_id}",
        }

    # Build action library from role peers
    peers = role_peer_names(metadata, target_name)
    if not peers:
        return {
            "seed": seed,
            "case_id": case_id,
            "status": "skipped",
            "reason": f"No role peers found for {target_name}",
        }

    peer_events = events.role_events(("train", "validation"), peers)
    if peer_events.empty:
        return {
            "seed": seed,
            "case_id": case_id,
            "status": "skipped",
            "reason": f"No clean peer events found for {target_name}",
        }

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

    # Clean branches
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
        None if clean_alert_rows.empty
        else pd.to_datetime(clean_alert_rows["timestamp"]).min()
    )

    rows = []
    detailed = [
        clean.assign(
            condition="clean", seed=seed, case_id=case_id,
            policy="none", duration_days=0, interactions_per_day=0,
        ),
        clean_gated.assign(
            condition="clean_gated", seed=seed, case_id=case_id,
            policy="none", duration_days=0, interactions_per_day=0,
        ),
    ]

    for policy, policy_templates in policies.items():
        for duration in config["attack"]["durations_days"]:
            for rate in config["attack"]["interactions_per_day"]:
                try:
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
                except ValueError as exc:
                    # Some cases may not have enough room for the conditioning window
                    print(
                        json.dumps({
                            "seed": seed, "case_id": case_id, "policy": policy,
                            "duration_days": int(duration), "rate": int(rate),
                            "status": "skipped", "reason": str(exc),
                        }),
                        flush=True,
                    )
                    continue

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
                    None if conditioning_alert_rows.empty
                    else pd.to_datetime(conditioning_alert_rows["timestamp"]).min()
                )

                conditioning_delta = float((paired["conditioned"] - paired["clean"]).mean())
                stealth_fraction = float((conditioning["anomaly_score"] < threshold).mean())
                payoff_suppression = conditioning_delta < 0
                gating_main_effect = float((paired["clean_gated"] - paired["clean"]).mean())
                gated_conditioning_delta = float((paired["defended"] - paired["clean_gated"]).mean())
                defense_recovery = gated_conditioning_delta - conditioning_delta

                result = {
                    "seed": seed,
                    "case_id": case_id,
                    "scenario": case["scenario"],
                    "target_user": case["target_user"],
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
                        None if first_conditioning_alert is None
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
                    "n_role_peers": len(peers),
                    "target_actor_payoff_events": int(case["target_actor_events"]),
                }
                rows.append(result)
                detailed.append(attacked.assign(
                    condition="conditioned", **{
                        key: result[key] for key in
                        ("seed", "case_id", "policy", "duration_days", "interactions_per_day")
                    }
                ))
                detailed.append(defended.assign(
                    condition="defended", **{
                        key: result[key] for key in
                        ("seed", "case_id", "policy", "duration_days", "interactions_per_day")
                    }
                ))
                print(
                    json.dumps({
                        "seed": seed,
                        "case_id": case_id,
                        "stage": "multi_target_conditioning",
                        "policy": policy,
                        "duration_days": int(duration),
                        "interactions_per_day": int(rate),
                        "configurations_complete": len(rows),
                        "elapsed_seconds": time.perf_counter() - study_started,
                    }),
                    flush=True,
                )

    # Save per-case results
    output = (
        resolve_config_path(config, config["paths"]["artifacts"])
        / f"seed_{seed}" / "multi_target" / case_id.replace(":", "_")
    )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    pd.concat(detailed, ignore_index=True).to_parquet(
        output / "retained_event_scores.parquet", index=False, compression="zstd"
    )
    summary = {
        "seed": seed,
        "case_id": case_id,
        "scenario": case["scenario"],
        "target_user": case["target_user"],
        "target_actor_payoff_events": int(case["target_actor_events"]),
        "n_role_peers": len(peers),
        "status": "complete",
        "configurations": len(rows),
        "policies": {
            name: [
                {"event_type": item.event_type, "destination": item.dst, "support": item.support}
                for item in values
            ]
            for name, values in policies.items()
        },
        "runs": rows,
        "elapsed_seconds": time.perf_counter() - study_started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_multi_target_study(
    events: PreparedDataset,
    metadata: dict,
    config: dict,
    cases: list[str] | None = None,
    seeds: list[int] | None = None,
    scenarios: list[int] | None = None,
) -> list[dict]:
    """Run the slow-conditioning study across multiple targets and seeds.

    Parameters
    ----------
    cases : list of str, optional
        Explicit case IDs to run (e.g. ["s3:BBS0039", "s1:KPC0073"]).
        If None, uses development_cases + test_cases from config.
    seeds : list of int, optional
        Model seeds. If None, uses all from config.
    scenarios : list of int, optional
        If set, additionally include all incidents from these scenarios.
    """
    catalog = json.loads((events.root / metadata["incident_catalog_file"]).read_text())
    selected_seeds = seeds or [int(s) for s in config["model_seeds"]]

    # Resolve case list
    if cases is not None:
        selected_cases = cases
    else:
        selected_cases = list(config.get("development_cases", []))
        selected_cases.extend(config.get("test_cases", []))

    # Add scenario-based cases if requested
    if scenarios:
        for incident in catalog["incidents"]:
            if incident["scenario"] in scenarios and incident["case_id"] not in selected_cases:
                selected_cases.append(incident["case_id"])

    print(
        json.dumps({
            "stage": "multi_target_study_start",
            "seeds": selected_seeds,
            "cases": selected_cases,
            "total_runs": len(selected_seeds) * len(selected_cases),
        }),
        flush=True,
    )

    results = []
    for seed in selected_seeds:
        for case_id in selected_cases:
            print(
                json.dumps({
                    "stage": "starting_case",
                    "seed": seed,
                    "case_id": case_id,
                }),
                flush=True,
            )
            try:
                result = run_case_seed(seed, case_id, events, metadata, config, catalog)
                results.append(result)
            except Exception as exc:
                error = {
                    "seed": seed,
                    "case_id": case_id,
                    "status": "error",
                    "reason": str(exc),
                }
                results.append(error)
                print(json.dumps(error), flush=True)

    # Save aggregate summary
    output = resolve_config_path(config, config["paths"]["artifacts"]) / "multi_target_aggregate"
    output.mkdir(parents=True, exist_ok=True)
    (output / "all_results.json").write_text(json.dumps(results, indent=2, default=str))

    # Also save a flat CSV of all configurations across all cases
    all_runs = []
    for result in results:
        if result.get("status") in ("skipped", "error"):
            continue
        all_runs.extend(result.get("runs", []))
    if all_runs:
        pd.DataFrame(all_runs).to_csv(output / "all_configurations.csv", index=False)

    return results

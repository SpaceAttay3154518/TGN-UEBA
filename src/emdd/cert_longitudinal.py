"""Leakage-free longitudinal preparation of the complete CERT r4.2 corpus.

The split boundary is fixed from the answer-key catalogue, not from model
performance: training, validation, and warm-up all end before the first known
malicious event.  Every raw event is retained and every answer-key event is
annotated with its incident and scenario so later evaluation can cover all 70
CERT r4.2 cases rather than a single hand-picked user.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .cert_full import (
    RAW_TABLES,
    _assign_nodes,
    _transform_chunk,
    _update_digest,
    hierarchical_hash_features,
    load_directory_roles,
)
from .data import EVENT_TYPES


def read_answer_catalog(answers: Path) -> tuple[dict[str, dict], list[dict]]:
    """Read all r4.2 ground truth and return event and incident catalogues."""
    events: dict[str, dict] = {}
    incidents: list[dict] = []
    pattern = re.compile(r"r4\.2-(\d+)-(.+)\.csv$")
    for path in sorted(answers.glob("r4.2-*/*.csv")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        scenario = int(match.group(1))
        target_user = match.group(2)
        case_id = f"s{scenario}:{target_user}"
        rows: list[dict] = []
        with path.open(newline="", errors="replace") as stream:
            for raw in csv.reader(stream):
                if len(raw) < 4:
                    continue
                event_id = str(raw[1])
                if event_id in events:
                    raise ValueError(f"Duplicate answer-key event ID: {event_id}")
                timestamp = pd.to_datetime(
                    raw[2], format="%m/%d/%Y %H:%M:%S", errors="raise"
                )
                actor = str(raw[3])
                record = {
                    "case_id": case_id,
                    "scenario": scenario,
                    "case_target_user": target_user,
                    "answer_actor": actor,
                    "target_actor_label": actor == target_user,
                    "timestamp": timestamp,
                    "kind": str(raw[0]),
                }
                events[event_id] = record
                rows.append(record)
        if not rows:
            raise ValueError(f"Empty answer file: {path}")
        incidents.append(
            {
                "case_id": case_id,
                "scenario": scenario,
                "target_user": target_user,
                "start": min(row["timestamp"] for row in rows).isoformat(),
                "end": max(row["timestamp"] for row in rows).isoformat(),
                "answer_events": len(rows),
                "target_actor_events": sum(
                    bool(row["target_actor_label"]) for row in rows
                ),
                "actors": sorted({str(row["answer_actor"]) for row in rows}),
                "source": str(path),
            }
        )
    if not incidents:
        raise FileNotFoundError(f"No CERT r4.2 answer files found below {answers}")
    return events, sorted(incidents, key=lambda row: (row["start"], row["case_id"]))


def _behavioural_roles(
    profiles: dict[str, dict[str, str]],
    event_counts: dict[str, Counter],
    unique_destinations: dict[str, set[int]],
    off_hours: Counter,
    weekends: Counter,
    clusters: int,
    seed: int,
) -> dict:
    """Derive unsupervised roles using only clean training-period behaviour."""
    users = sorted(user for user, counts in event_counts.items() if sum(counts.values()))
    matrix: list[list[float]] = []
    for user in users:
        counts = event_counts[user]
        total = float(sum(counts.values()))
        matrix.append(
            [
                *[float(counts[name]) / total for name in EVENT_TYPES],
                float(off_hours[user]) / total,
                float(weekends[user]) / total,
                np.log1p(total),
                float(len(unique_destinations[user])) / total,
            ]
        )
    if not matrix:
        raise ValueError("No clean training events were available for behavioural roles")
    count = min(int(clusters), len(users))
    standardized = StandardScaler().fit_transform(np.asarray(matrix, dtype=float))
    assignments = KMeans(n_clusters=count, random_state=seed, n_init=20).fit_predict(
        standardized
    )
    for user, cluster in zip(users, assignments, strict=True):
        node = f"usr:{user}"
        if node in profiles:
            profiles[node]["behavior_role"] = f"behavior:{int(cluster):02d}"
            profiles[node]["hybrid_role"] = (
                f"{profiles[node]['broad_role']}|behavior:{int(cluster):02d}"
            )
    for profile in profiles.values():
        profile.setdefault("behavior_role", "behavior:unknown")
        profile.setdefault(
            "hybrid_role", f"{profile['broad_role']}|behavior:unknown"
        )
    return {
        "method": "KMeans on standardized clean-training user behaviour",
        "seed": int(seed),
        "clusters": count,
        "features": [
            *[f"proportion_{name}" for name in EVENT_TYPES],
            "off_hours_fraction",
            "weekend_fraction",
            "log1p_event_count",
            "unique_destination_ratio",
        ],
        "fit_users": len(users),
        "fit_period_only": True,
    }


def prepare_longitudinal_cert(
    config: dict, cert_root: Path, answers: Path, prepared: Path
) -> dict:
    dataset = config["dataset"]
    answer_events, incidents = read_answer_catalog(answers)
    first_attack = min(pd.Timestamp(row["start"]) for row in incidents)
    validation_days = int(dataset.get("validation_days", 7))
    context_days = int(dataset.get("conditioning_context_days", 14))
    context_start = first_attack - pd.Timedelta(days=context_days)
    validation_start = context_start - pd.Timedelta(days=validation_days)
    profiles, role_provenance = load_directory_roles(cert_root, validation_start)

    if prepared.exists():
        shutil.rmtree(prepared)
    event_root = prepared / "events"
    event_root.mkdir(parents=True)

    event_type_to_id = {name: index for index, name in enumerate(EVENT_TYPES)}
    dst_type_to_id = {
        name: index
        for index, name in enumerate(("device", "domain", "fileext", "maildomain", "pc"))
    }
    node_to_id: dict[str, int] = {}
    split_counts: Counter = Counter()
    split_min: dict[str, pd.Timestamp] = {}
    split_max: dict[str, pd.Timestamp] = {}
    source_counts: Counter = Counter()
    labelled_by_scenario: Counter = Counter()
    matched_answer_ids: set[str] = set()
    training_destinations: dict[int, set[int]] = defaultdict(set)
    behaviour_counts: dict[str, Counter] = defaultdict(Counter)
    behaviour_destinations: dict[str, set[int]] = defaultdict(set)
    behaviour_off_hours: Counter = Counter()
    behaviour_weekends: Counter = Counter()
    digest = hashlib.sha256()
    chunk_rows = int(dataset.get("read_chunk_rows", 200_000))

    case_map = {key: value["case_id"] for key, value in answer_events.items()}
    scenario_map = {key: value["scenario"] for key, value in answer_events.items()}
    target_map = {
        key: value["case_target_user"] for key, value in answer_events.items()
    }
    actor_map = {key: value["answer_actor"] for key, value in answer_events.items()}
    actor_label_map = {
        key: value["target_actor_label"] for key, value in answer_events.items()
    }

    for table in RAW_TABLES:
        path = cert_root / f"{table}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        for chunk_index, raw in enumerate(
            pd.read_csv(path, dtype=str, chunksize=chunk_rows, on_bad_lines="skip")
        ):
            frame = _transform_chunk(table, raw)
            if frame.empty:
                continue
            frame["attack_label"] = frame["event_id"].isin(answer_events)
            frame["case_id"] = frame["event_id"].map(case_map).fillna("")
            frame["attack_scenario"] = (
                frame["event_id"].map(scenario_map).fillna(0).astype("int8")
            )
            frame["case_target_user"] = frame["event_id"].map(target_map).fillna("")
            frame["answer_actor"] = frame["event_id"].map(actor_map).fillna("")
            frame["target_actor_label"] = (
                frame["event_id"].map(actor_label_map).fillna(False).astype(bool)
            )
            frame["split"] = np.select(
                [
                    frame["timestamp"] < validation_start,
                    (frame["timestamp"] >= validation_start)
                    & (frame["timestamp"] < context_start),
                    (frame["timestamp"] >= context_start)
                    & (frame["timestamp"] < first_attack),
                    frame["timestamp"] >= first_attack,
                ],
                ["train", "validation", "context", "evaluation"],
                default="discard",
            )
            frame = frame[frame["split"] != "discard"].copy()
            if frame.empty:
                continue
            frame["src_id"] = _assign_nodes(frame["src"], node_to_id)
            frame["dst_id"] = _assign_nodes(frame["dst"], node_to_id)
            frame["event_type_id"] = (
                frame["event_type"].map(event_type_to_id).astype("int16")
            )
            frame["dst_type_id"] = (
                frame["dst_type"].map(dst_type_to_id).astype("int8")
            )
            frame["timestamp_s"] = (
                frame["timestamp"].astype("datetime64[ns]").astype("int64")
                // 1_000_000_000
            ).astype("int64")
            frame["day"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
            output_columns = [
                "event_id",
                "timestamp",
                "timestamp_s",
                "src",
                "dst",
                "dst_type",
                "src_id",
                "dst_id",
                "dst_type_id",
                "event_type",
                "event_type_id",
                "attack_label",
                "case_id",
                "attack_scenario",
                "case_target_user",
                "answer_actor",
                "target_actor_label",
                "split",
            ]
            for (split, day), group in frame.groupby(["split", "day"], sort=True):
                group = group[output_columns].sort_values(
                    ["timestamp_s", "event_id"], kind="stable"
                )
                directory = event_root / f"split={split}" / f"day={day}"
                directory.mkdir(parents=True, exist_ok=True)
                group.to_parquet(
                    directory / f"{table}-{chunk_index:05d}.parquet",
                    index=False,
                    compression="zstd",
                )
                count = int(len(group))
                split_counts[str(split)] += count
                source_counts[table] += count
                split_min[str(split)] = min(
                    split_min.get(str(split), group["timestamp"].min()),
                    group["timestamp"].min(),
                )
                split_max[str(split)] = max(
                    split_max.get(str(split), group["timestamp"].max()),
                    group["timestamp"].max(),
                )
                if split == "train":
                    for type_id, values in group.groupby("dst_type_id")["dst_id"]:
                        training_destinations[int(type_id)].update(
                            int(value) for value in values.unique()
                        )
                    hours = pd.to_datetime(group["timestamp"]).dt.hour
                    weekend = pd.to_datetime(group["timestamp"]).dt.dayofweek >= 5
                    for user, user_rows in group.groupby("src", sort=False):
                        bare_user = str(user).split(":", 1)[-1]
                        behaviour_counts[bare_user].update(
                            user_rows["event_type"].value_counts().to_dict()
                        )
                        behaviour_destinations[bare_user].update(
                            int(value) for value in user_rows["dst_id"].unique()
                        )
                        positions = group.index.get_indexer(user_rows.index)
                        behaviour_off_hours[bare_user] += int(
                            ((hours.iloc[positions] < 7) | (hours.iloc[positions] >= 19)).sum()
                        )
                        behaviour_weekends[bare_user] += int(weekend.iloc[positions].sum())
                if bool(group["attack_label"].any()):
                    malicious = group[group["attack_label"]]
                    matched_answer_ids.update(str(value) for value in malicious["event_id"])
                    labelled_by_scenario.update(
                        int(value) for value in malicious["attack_scenario"]
                    )
                _update_digest(digest, group)

    missing_answer_ids = set(answer_events) - matched_answer_ids
    if missing_answer_ids:
        sample = sorted(missing_answer_ids)[:5]
        raise RuntimeError(
            f"{len(missing_answer_ids)} answer-key IDs were not found in raw tables; sample={sample}"
        )
    if any(
        pd.Timestamp(value) >= first_attack
        for value in (split_max.get("train"), split_max.get("validation"), split_max.get("context"))
        if value is not None
    ):
        raise RuntimeError("A clean-development split crosses the first-attack boundary")

    unknown_users: set[str] = set()
    for node in node_to_id:
        if node.startswith("usr:") and node not in profiles:
            user = node.split(":", 1)[1]
            unknown_users.add(user)
            profiles[node] = {
                "user_id": user,
                "role": "unknown",
                "department": "unknown",
                "team": "unknown",
                "functional_unit": "unknown",
                "business_unit": "unknown",
                "fine_role": "unknown|unknown|unknown",
                "parent_role": "unknown|unknown",
                "broad_role": "unknown",
                "entity_type": "usr",
                "ldap_snapshot": "missing_before_cutoff",
            }
    behaviour_provenance = _behavioural_roles(
        profiles,
        behaviour_counts,
        behaviour_destinations,
        behaviour_off_hours,
        behaviour_weekends,
        int(dataset.get("behavior_role_clusters", 20)),
        int(dataset.get("behavior_role_seed", 2026)),
    )

    feature_dim = int(config["model"]["node_feature_dim"])
    node_features = hierarchical_hash_features(node_to_id, profiles, feature_dim)
    np.save(prepared / "node_features.npy", node_features)
    time_origin_s = int(
        min(value.value for value in split_min.values()) // 1_000_000_000
    )
    incident_catalog = {
        "dataset": "CERT r4.2",
        "incidents": incidents,
        "incident_count": len(incidents),
        "answer_event_count": len(answer_events),
        "scenario_counts": {
            str(scenario): sum(row["scenario"] == scenario for row in incidents)
            for scenario in sorted({row["scenario"] for row in incidents})
        },
    }
    (prepared / "incident_catalog.json").write_text(
        json.dumps(incident_catalog, indent=2)
    )
    metadata = {
        "format_version": 3,
        "storage": "partitioned_parquet_by_split_and_day",
        "partition_key": "day",
        "dataset": dataset["name"],
        "scope": dataset["scope"],
        "first_attack": first_attack.isoformat(),
        "development_cutoffs": {
            "validation_start": validation_start.isoformat(),
            "context_start": context_start.isoformat(),
            "evaluation_start": first_attack.isoformat(),
        },
        "node_to_id": node_to_id,
        "id_to_node": {str(value): key for key, value in node_to_id.items()},
        "node_types": {
            str(node_id): node.split(":", 1)[0]
            for node, node_id in node_to_id.items()
        },
        "role_profiles": profiles,
        "role_provenance": role_provenance,
        "behavior_role_provenance": behaviour_provenance,
        "unknown_ldap_users": sorted(unknown_users),
        "event_types": EVENT_TYPES,
        "event_type_to_id": event_type_to_id,
        "dst_type_to_id": dst_type_to_id,
        "training_destination_ids_by_type": {
            str(type_id): sorted(values)
            for type_id, values in training_destinations.items()
        },
        "node_feature_dim": feature_dim,
        "node_features_file": "node_features.npy",
        "raw_msg_dim": 2 * feature_dim + len(EVENT_TYPES),
        "message_features": [
            *[f"src_hash_{index}" for index in range(feature_dim)],
            *[f"edge_type_{name}" for name in EVENT_TYPES],
            *[f"dst_hash_{index}" for index in range(feature_dim)],
        ],
        "message_columns": [],
        "split_sizes": dict(split_counts),
        "split_time_ranges": {
            split: {
                "min": split_min[split].isoformat(),
                "max": split_max[split].isoformat(),
            }
            for split in split_counts
        },
        "source_event_counts": dict(source_counts),
        "labelled_attack_events": len(matched_answer_ids),
        "labelled_by_scenario": {
            str(key): int(value) for key, value in sorted(labelled_by_scenario.items())
        },
        "incident_catalog_file": "incident_catalog.json",
        "events_sha256": digest.hexdigest(),
        "time_origin_s": time_origin_s,
        "cert_root": str(cert_root),
    }
    (prepared / "metadata.json").write_text(json.dumps(metadata, indent=2))
    manifest = {
        "format_version": 3,
        "scope": "all CERT r4.2 users and events; no sampling or event cap",
        "rows": int(sum(split_counts.values())),
        "nodes": len(node_to_id),
        "users": sum(node.startswith("usr:") for node in node_to_id),
        "incidents": len(incidents),
        "labelled_attack_events": len(matched_answer_ids),
        "split_sizes": dict(split_counts),
        "split_time_ranges": metadata["split_time_ranges"],
        "source_event_counts": dict(source_counts),
        "chronological": True,
        "pre_attack_development": True,
        "no_answer_rows_removed": True,
        "events_sha256": metadata["events_sha256"],
        "unknown_ldap_users": len(unknown_users),
        "prepared": str(event_root),
        "source_root": str(cert_root),
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def validate_longitudinal_prepared(prepared: Path) -> dict:
    metadata = json.loads((prepared / "metadata.json").read_text())
    manifest = json.loads((prepared / "manifest.json").read_text())
    catalog = json.loads((prepared / metadata["incident_catalog_file"]).read_text())
    if int(metadata.get("format_version", -1)) != 3:
        raise ValueError("Expected longitudinal format version 3")
    if catalog["incident_count"] != 70 or catalog["answer_event_count"] != 7323:
        raise ValueError("CERT r4.2 ground-truth catalogue is incomplete")
    for split in ("train", "validation", "context", "evaluation"):
        directory = prepared / "events" / f"split={split}"
        if not directory.exists() or not list(directory.rglob("*.parquet")):
            raise FileNotFoundError(f"Prepared split is empty: {split}")
    if int(metadata["labelled_attack_events"]) != 7323:
        raise ValueError("Not every CERT r4.2 answer-key event was matched")
    evaluation_start = pd.Timestamp(metadata["development_cutoffs"]["evaluation_start"])
    for split in ("train", "validation", "context"):
        if pd.Timestamp(metadata["split_time_ranges"][split]["max"]) >= evaluation_start:
            raise ValueError(f"{split} crosses the first known attack")
    features = np.load(prepared / metadata["node_features_file"], mmap_mode="r")
    if features.shape != (len(metadata["node_to_id"]), metadata["node_feature_dim"]):
        raise ValueError("Node feature dimensions do not match metadata")
    return manifest

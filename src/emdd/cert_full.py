"""Streaming preparation of the complete eligible CERT r4.2 event corpus.

All users and all events before the declared test boundary are processed.  The
only exclusions are answer-key events belonging to non-target incidents.  No
per-user sampling, per-day cap, or global training cap is applied.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data import EVENT_TYPES, read_answer_events


RAW_TABLES = ("logon", "device", "file", "http", "email")


def _snapshot_date(path: Path) -> pd.Timestamp:
    match = re.search(r"(20\d{2})[-_](\d{1,2})", path.stem)
    if not match:
        return pd.Timestamp.min
    return pd.Timestamp(year=int(match.group(1)), month=int(match.group(2)), day=1)


def load_directory_roles(
    cert_root: Path, attack_start: pd.Timestamp
) -> tuple[dict[str, dict[str, str]], dict]:
    """Return each user's latest pre-incident LDAP record.

    CERT supplies explicit organizational attributes.  Iterating every eligible
    monthly snapshot preserves users who left before the final snapshot while
    allowing later records to replace stale ones.
    """
    ldap_dirs = [path for path in cert_root.rglob("LDAP") if path.is_dir()]
    if not ldap_dirs:
        raise FileNotFoundError("CERT LDAP directory was not found")
    cutoff = attack_start.normalize().replace(day=1)
    snapshots = [
        path
        for path in sorted(ldap_dirs[0].glob("*.csv"), key=_snapshot_date)
        if _snapshot_date(path) <= cutoff
    ]
    if not snapshots:
        raise FileNotFoundError("No LDAP snapshot exists before the incident")
    latest: dict[str, dict[str, str]] = {}
    for snapshot in snapshots:
        frame = pd.read_csv(snapshot, dtype=str).fillna("")
        frame.columns = [column.strip().lower() for column in frame.columns]
        user_column = next(
            (name for name in ("user_id", "userid", "user") if name in frame),
            None,
        )
        if user_column is None:
            raise ValueError(f"No user ID column in {snapshot}")
        for row in frame.to_dict("records"):
            user = str(row[user_column]).strip()
            if not user:
                continue
            role = str(row.get("role", "unknown")).strip() or "unknown"
            department = str(row.get("department", "unknown")).strip() or "unknown"
            team = str(row.get("team", "unknown")).strip() or "unknown"
            functional = (
                str(row.get("functional_unit", "unknown")).strip() or "unknown"
            )
            business = str(row.get("business_unit", "unknown")).strip() or "unknown"
            latest[f"usr:{user}"] = {
                "user_id": user,
                "role": role,
                "department": department,
                "team": team,
                "functional_unit": functional,
                "business_unit": business,
                "fine_role": f"{role}|{department}|{team}",
                "parent_role": f"{role}|{functional}",
                "broad_role": role,
                "entity_type": "usr",
                "ldap_snapshot": snapshot.name,
            }
    return latest, {
        "method": "latest available LDAP record on or before the incident month",
        "snapshots": [str(path) for path in snapshots],
        "attributes": [
            "role",
            "department",
            "functional_unit",
            "business_unit",
            "team",
        ],
    }


def _host_from_url(values: pd.Series) -> pd.Series:
    hosts = (
        values.fillna("unknown")
        .str.replace(r"^[A-Za-z][A-Za-z0-9+.-]*://", "", regex=True)
        .str.split("/", n=1)
        .str[0]
        .str.split(":", n=1)
        .str[0]
        .str.lower()
        .replace("", "unknown")
    )
    # Preserve the registrable-looking suffix used by the pilot.  This is a
    # semantic reduction, not event sampling: every HTTP row remains present.
    return hosts.map(
        lambda host: ".".join(str(host).split(".")[-2:])
        if str(host).count(".") >= 1
        else str(host)
    )


def _email_domain(values: pd.Series) -> pd.Series:
    domains = values.fillna("").str.extract(r"@([^;,\s]+)", expand=False)
    return domains.fillna("unknown").str.lower()


def _transform_chunk(table: str, chunk: pd.DataFrame) -> pd.DataFrame:
    chunk.columns = [column.strip().lower() for column in chunk.columns]
    chunk["timestamp"] = pd.to_datetime(
        chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce"
    )
    chunk = chunk.dropna(subset=["timestamp", "id", "user"]).copy()
    chunk["event_id"] = chunk["id"]
    chunk["src"] = "usr:" + chunk["user"].astype(str)
    if table == "logon":
        activity = chunk["activity"].str.lower()
        chunk["event_type"] = np.where(activity == "logon", "logon", "logoff")
        chunk["dst"] = "pc:" + chunk["pc"].fillna("unknown")
        chunk["dst_type"] = "pc"
    elif table == "device":
        activity = chunk["activity"].str.lower()
        chunk["event_type"] = np.where(
            activity == "connect", "device_connect", "device_disconnect"
        )
        chunk["dst"] = "device:removable"
        chunk["dst_type"] = "device"
    elif table == "file":
        extension = chunk["filename"].fillna("").map(
            lambda value: Path(value).suffix.lower() or "none"
        )
        chunk["event_type"] = "file"
        chunk["dst"] = "fileext:" + extension
        chunk["dst_type"] = "fileext"
    elif table == "http":
        chunk["event_type"] = "http"
        chunk["dst"] = "domain:" + _host_from_url(chunk["url"])
        chunk["dst_type"] = "domain"
    elif table == "email":
        chunk["event_type"] = "email"
        chunk["dst"] = "maildomain:" + _email_domain(chunk["to"])
        chunk["dst_type"] = "maildomain"
    else:
        raise ValueError(f"Unsupported CERT table: {table}")
    return chunk


def _semantic_tokens(node: str, profile: dict[str, str] | None) -> list[str]:
    kind, value = node.split(":", 1)
    if kind == "usr" and profile is not None:
        role = profile["role"]
        department = profile["department"]
        team = profile["team"]
        functional = profile["functional_unit"]
        return [
            "usr",
            f"usr/role/{role}",
            f"usr/role/{role}/functional/{functional}",
            f"usr/role/{role}/department/{department}",
            f"usr/role/{role}/department/{department}/team/{team}",
            f"usr/id/{value}",
        ]
    if kind in {"domain", "maildomain"}:
        labels = value.split(".")
        hierarchy = [f"{kind}/{'.'.join(labels[index:])}" for index in range(len(labels) - 1, -1, -1)]
        return [kind, *hierarchy]
    if kind == "pc":
        prefix = value.split("-", 1)[0]
        return ["pc", f"pc/{prefix}", f"pc/{value}"]
    return [kind, f"{kind}/{value}"]


def hierarchical_hash_features(
    node_to_id: dict[str, int],
    profiles: dict[str, dict[str, str]],
    dimension: int,
) -> np.ndarray:
    """Signed feature hashing over hierarchical semantic tokens, as in KAIROS."""
    output = np.zeros((len(node_to_id), dimension), dtype=np.float32)
    for node, node_id in node_to_id.items():
        for token in _semantic_tokens(node, profiles.get(node)):
            digest = hashlib.blake2b(token.encode(), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            output[node_id, index] += sign
    return output


def _assign_nodes(values: pd.Series, node_to_id: dict[str, int]) -> pd.Series:
    for value in pd.unique(values):
        text = str(value)
        if text not in node_to_id:
            node_to_id[text] = len(node_to_id)
    return values.map(node_to_id).astype("int64")


def _update_digest(digest: "hashlib._Hash", frame: pd.DataFrame) -> None:
    selected = frame[["event_id", "timestamp_s", "src", "dst", "event_type", "split"]]
    hashed = pd.util.hash_pandas_object(selected, index=False).to_numpy(np.uint64)
    digest.update(hashed.tobytes())


def prepare_full_cert(config: dict, cert_root: Path, answers: Path, prepared: Path) -> dict:
    dataset = config["dataset"]
    all_attack_ids, target_attack_ids, target_answers = read_answer_events(
        answers, dataset["target_user"], int(dataset["target_scenario"])
    )
    attack_start = pd.Timestamp(target_answers["timestamp"].min())
    attack_end = pd.Timestamp(target_answers["timestamp"].max())
    validation_start = attack_start - pd.Timedelta(
        days=int(dataset["conditioning_context_days"]) + int(dataset["validation_days"])
    )
    context_start = attack_start - pd.Timedelta(
        days=int(dataset["conditioning_context_days"])
    )
    test_end = attack_end + pd.Timedelta(hours=int(dataset["post_payoff_hours"]))
    profiles, role_provenance = load_directory_roles(cert_root, attack_start)

    if prepared.exists():
        shutil.rmtree(prepared)
    event_root = prepared / "events"
    event_root.mkdir(parents=True)
    node_to_id: dict[str, int] = {}
    split_counts: dict[str, int] = defaultdict(int)
    split_min: dict[str, pd.Timestamp] = {}
    split_max: dict[str, pd.Timestamp] = {}
    source_counts: dict[str, int] = defaultdict(int)
    training_destinations: dict[int, set[int]] = defaultdict(set)
    digest = hashlib.sha256()
    labelled = 0
    unknown_users: set[str] = set()
    event_to_id = {name: index for index, name in enumerate(EVENT_TYPES)}
    dst_type_to_id = {
        name: index
        for index, name in enumerate(("device", "domain", "fileext", "maildomain", "pc"))
    }
    chunk_rows = int(dataset.get("read_chunk_rows", 200_000))

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
            if frame["timestamp"].min() > test_end:
                break
            frame = frame[frame["timestamp"] <= test_end].copy()
            if frame.empty:
                continue
            frame["is_any_attack"] = frame["event_id"].isin(all_attack_ids)
            frame["attack_label"] = frame["event_id"].isin(target_attack_ids)
            frame = frame[~(frame["is_any_attack"] & ~frame["attack_label"])].copy()
            if frame.empty:
                continue
            conditions = [
                (frame["timestamp"] < validation_start) & ~frame["is_any_attack"],
                (frame["timestamp"] >= validation_start)
                & (frame["timestamp"] < context_start)
                & ~frame["is_any_attack"],
                (frame["timestamp"] >= context_start)
                & (frame["timestamp"] < attack_start)
                & ~frame["is_any_attack"],
                (frame["timestamp"] >= attack_start) & (frame["timestamp"] <= test_end),
            ]
            frame["split"] = np.select(
                conditions, ["train", "validation", "context", "test"], default="discard"
            )
            frame = frame[frame["split"] != "discard"].copy()
            if frame.empty:
                continue
            frame["src_id"] = _assign_nodes(frame["src"], node_to_id)
            frame["dst_id"] = _assign_nodes(frame["dst"], node_to_id)
            frame["event_type_id"] = frame["event_type"].map(event_to_id).astype("int16")
            frame["dst_type_id"] = frame["dst_type"].map(dst_type_to_id).astype("int8")
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
                if split == "train":
                    for type_id, values in group.groupby("dst_type_id")["dst_id"]:
                        training_destinations[int(type_id)].update(
                            int(value) for value in values.unique()
                        )
                split_min[str(split)] = min(
                    split_min.get(str(split), group["timestamp"].min()),
                    group["timestamp"].min(),
                )
                split_max[str(split)] = max(
                    split_max.get(str(split), group["timestamp"].max()),
                    group["timestamp"].max(),
                )
                labelled += int(group["attack_label"].sum())
                _update_digest(digest, group)

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
                "ldap_snapshot": "missing",
            }
    if labelled != len(target_attack_ids):
        raise RuntimeError(
            f"Expected {len(target_attack_ids)} target events, prepared {labelled}"
        )
    feature_dim = int(config["model"]["node_feature_dim"])
    node_features = hierarchical_hash_features(node_to_id, profiles, feature_dim)
    np.save(prepared / "node_features.npy", node_features)
    events_sha256 = digest.hexdigest()
    time_origin_s = int(min(value.value for value in split_min.values()) // 1_000_000_000)
    metadata = {
        "format_version": 2,
        "storage": "partitioned_parquet_by_split_and_day",
        "dataset": dataset["name"],
        "scope": dataset["scope"],
        "target_user": dataset["target_user"],
        "target_user_node": f"usr:{dataset['target_user']}",
        "target_scenario": int(dataset["target_scenario"]),
        "attack_start": attack_start.isoformat(),
        "attack_end": attack_end.isoformat(),
        "test_end": test_end.isoformat(),
        "node_to_id": node_to_id,
        "id_to_node": {str(value): key for key, value in node_to_id.items()},
        "node_types": {
            str(node_id): node.split(":", 1)[0]
            for node, node_id in node_to_id.items()
        },
        "role_profiles": profiles,
        "role_provenance": role_provenance,
        "unknown_ldap_users": sorted(unknown_users),
        "event_types": EVENT_TYPES,
        "event_type_to_id": event_to_id,
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
        "target_attack_events_in_stream": labelled,
        "events_sha256": events_sha256,
        "time_origin_s": time_origin_s,
        "cert_root": str(cert_root),
    }
    (prepared / "metadata.json").write_text(json.dumps(metadata, indent=2))
    target_profile = profiles[f"usr:{dataset['target_user']}"]
    manifest = {
        "format_version": 2,
        "scope": "all users; every eligible event; no sampling or training cap",
        "rows": int(sum(split_counts.values())),
        "nodes": int(len(node_to_id)),
        "users": int(sum(node.startswith("usr:") for node in node_to_id)),
        "labelled_payoff_events": labelled,
        "split_sizes": dict(split_counts),
        "split_time_ranges": metadata["split_time_ranges"],
        "source_event_counts": dict(source_counts),
        "chronological_partitions": True,
        "leakage_free": True,
        "events_sha256": events_sha256,
        "target_role": {
            name: target_profile[name]
            for name in ("role", "department", "functional_unit", "business_unit", "team")
        },
        "unknown_ldap_users": len(unknown_users),
        "prepared": str(event_root),
        "source_root": str(cert_root),
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def validate_full_prepared(prepared: Path) -> dict:
    metadata = json.loads((prepared / "metadata.json").read_text())
    manifest = json.loads((prepared / "manifest.json").read_text())
    if metadata.get("format_version") != 2:
        raise ValueError("Expected partitioned prepared-data format version 2")
    if not (prepared / metadata["node_features_file"]).exists():
        raise FileNotFoundError("Node feature matrix is missing")
    features = np.load(prepared / metadata["node_features_file"], mmap_mode="r")
    if features.shape != (
        len(metadata["node_to_id"]),
        int(metadata["node_feature_dim"]),
    ):
        raise ValueError("Node feature matrix shape does not match metadata")
    pools = metadata.get("training_destination_ids_by_type", {})
    if not pools or any(not values for values in pools.values()):
        raise ValueError("Training-only negative destination pools are missing")
    if any(
        int(node) >= len(metadata["node_to_id"])
        for values in pools.values()
        for node in values
    ):
        raise ValueError("A training negative-pool node exceeds the vocabulary")
    for split in ("train", "validation", "context", "test"):
        directory = prepared / "events" / f"split={split}"
        if not directory.exists() or not list(directory.rglob("*.parquet")):
            raise FileNotFoundError(f"Prepared split is empty: {split}")
    if int(manifest["labelled_payoff_events"]) <= 0:
        raise ValueError("No labelled payoff events were prepared")
    if "no sampling or training cap" not in manifest["scope"]:
        raise ValueError("Manifest does not declare the no-sampling full scope")
    return manifest

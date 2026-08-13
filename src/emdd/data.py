"""Prepare the leakage-controlled CERT r4.2 stream for the main study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd


EVENT_TYPES = [
    "logon",
    "logoff",
    "device_connect",
    "device_disconnect",
    "file",
    "http",
    "email",
]


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    config["_config_path"] = str(path.resolve())
    return config


def resolve(base: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def read_answer_events(answers: Path, target_user: str, scenario: int) -> tuple[set[str], set[str], pd.DataFrame]:
    all_ids: set[str] = set()
    target_rows = []
    target_path = answers / f"r4.2-{scenario}" / f"r4.2-{scenario}-{target_user}.csv"
    if not target_path.exists():
        raise FileNotFoundError(target_path)
    for path in answers.glob("r4.2-*/*.csv"):
        with path.open(newline="", errors="replace") as stream:
            for row in csv.reader(stream):
                if len(row) < 3:
                    continue
                all_ids.add(row[1])
                if path == target_path:
                    target_rows.append(
                        {
                            "kind": row[0],
                            "event_id": row[1],
                            "date": row[2],
                            "user": row[3] if len(row) > 3 else target_user,
                        }
                    )
    target = pd.DataFrame(target_rows)
    target["timestamp"] = pd.to_datetime(target["date"], format="%m/%d/%Y %H:%M:%S")
    return all_ids, set(target["event_id"]), target


def load_role_peers(
    cert_root: Path,
    target_user: str,
    peer_count: int,
    attack_start: pd.Timestamp,
) -> tuple[list[str], dict, dict[str, dict[str, str]]]:
    ldap_dir_candidates = [p for p in cert_root.rglob("LDAP") if p.is_dir()]
    if not ldap_dir_candidates:
        raise RuntimeError("CERT LDAP directory not found")
    ldap_files = sorted(ldap_dir_candidates[0].glob("*.csv"))
    if not ldap_files:
        raise RuntimeError("No LDAP snapshots found")

    # Snapshot names are chronological in the CERT release. Prefer the latest
    # snapshot whose filename date does not exceed the incident month.
    eligible = []
    for path in ldap_files:
        match = re.search(r"(20\d{2})[-_](\d{1,2})", path.stem)
        if match:
            stamp = pd.Timestamp(year=int(match.group(1)), month=int(match.group(2)), day=1)
            if stamp <= attack_start.normalize().replace(day=1):
                eligible.append(path)
    snapshot_path = eligible[-1] if eligible else ldap_files[-1]
    ldap = pd.read_csv(snapshot_path, dtype=str).fillna("")
    ldap.columns = [column.strip().lower() for column in ldap.columns]
    user_column = next(
        (column for column in ["user_id", "userid", "user"] if column in ldap.columns),
        None,
    )
    if user_column is None:
        raise RuntimeError(f"No user identifier in LDAP columns: {ldap.columns.tolist()}")
    target_match = ldap[ldap[user_column] == target_user]
    if target_match.empty:
        # Employees can disappear in the snapshot immediately after an
        # incident. Search backwards for the most recent containing snapshot.
        for candidate in reversed(ldap_files):
            frame = pd.read_csv(candidate, dtype=str).fillna("")
            frame.columns = [column.strip().lower() for column in frame.columns]
            candidate_user = next(
                (column for column in ["user_id", "userid", "user"] if column in frame.columns),
                None,
            )
            if candidate_user and (frame[candidate_user] == target_user).any():
                ldap, user_column, snapshot_path = frame, candidate_user, candidate
                target_match = ldap[ldap[user_column] == target_user]
                break
    if target_match.empty:
        raise RuntimeError(f"Target {target_user} not present in LDAP snapshots")

    target = target_match.iloc[0]
    role_columns = [
        column
        for column in ["role", "department", "functional_unit", "business_unit", "team"]
        if column in ldap.columns and str(target[column]).strip()
    ]
    ranked = []
    for _, row in ldap.iterrows():
        user = row[user_column]
        if not user or user == target_user:
            continue
        agreement = sum(row[column] == target[column] for column in role_columns)
        ranked.append((-agreement, user))
    ranked.sort()
    peers = [target_user, *[user for _, user in ranked[:peer_count]]]
    role_info = {
        "ldap_snapshot": str(snapshot_path),
        "role_columns": role_columns,
        "target_attributes": {column: target[column] for column in role_columns},
        "peer_selection": "descending exact agreement across available role columns",
    }
    profiles: dict[str, dict[str, str]] = {}
    for user in peers:
        match = ldap[ldap[user_column] == user]
        if match.empty:
            continue
        row = match.iloc[0]
        role = str(row.get("role", "unknown")).strip() or "unknown"
        department = str(row.get("department", "unknown")).strip() or "unknown"
        team = str(row.get("team", "unknown")).strip() or "unknown"
        functional = str(row.get("functional_unit", "unknown")).strip() or "unknown"
        profiles[f"usr:{user}"] = {
            "user_id": user,
            "fine_role": f"{role}|{department}|{team}",
            "parent_role": f"{role}|{functional}",
            "broad_role": role,
            "entity_type": "usr",
        }
    return peers, role_info, profiles


def domain_from_url(value: str) -> str:
    parsed = urlparse(value if "://" in value else f"http://{value}")
    host = (parsed.hostname or "unknown").lower()
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def email_domain(value: str) -> str:
    addresses = re.split(r"[;,]", str(value))
    for address in addresses:
        if "@" in address:
            return address.rsplit("@", 1)[1].strip().lower()
    return "unknown"


def cap_events(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    frame["day"] = frame["timestamp"].dt.floor("D")
    rank = frame.groupby(["src", "day", "event_type"], sort=False).cumcount()
    keep = (rank < cap) | frame["is_any_attack"]
    return frame.loc[keep].drop(columns="day")


def read_filtered_csv(path: Path, users: set[str], all_attack_ids: set[str], target_attack_ids: set[str], chunk_size: int = 200_000):
    for chunk in pd.read_csv(path, dtype=str, chunksize=chunk_size, on_bad_lines="skip"):
        chunk.columns = [column.strip().lower() for column in chunk.columns]
        if "user" not in chunk.columns:
            continue
        chunk = chunk[chunk["user"].isin(users)].copy()
        if chunk.empty:
            continue
        chunk["timestamp"] = pd.to_datetime(
            chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce"
        )
        chunk = chunk.dropna(subset=["timestamp"])
        chunk["event_id"] = chunk["id"]
        chunk["src"] = "usr:" + chunk["user"]
        chunk["is_any_attack"] = chunk["event_id"].isin(all_attack_ids)
        chunk["attack_label"] = chunk["event_id"].isin(target_attack_ids)
        yield chunk


def build_events(cert_root: Path, users: list[str], all_attack_ids: set[str], target_attack_ids: set[str], cap: int) -> pd.DataFrame:
    user_set = set(users)
    frames = []

    for chunk in read_filtered_csv(cert_root / "logon.csv", user_set, all_attack_ids, target_attack_ids):
        activity = chunk["activity"].str.lower()
        chunk["event_type"] = np.where(activity == "logon", "logon", "logoff")
        chunk["dst"] = "pc:" + chunk["pc"].fillna("unknown")
        chunk["dst_type"] = "pc"
        chunk["external"] = 0.0
        chunk["magnitude"] = 0.0
        frames.append(cap_events(chunk, cap))

    for chunk in read_filtered_csv(cert_root / "device.csv", user_set, all_attack_ids, target_attack_ids):
        activity = chunk["activity"].str.lower()
        chunk["event_type"] = np.where(
            activity == "connect", "device_connect", "device_disconnect"
        )
        chunk["dst"] = "device:removable"
        chunk["dst_type"] = "device"
        chunk["external"] = 1.0
        chunk["magnitude"] = 0.0
        frames.append(cap_events(chunk, cap))

    for chunk in read_filtered_csv(cert_root / "file.csv", user_set, all_attack_ids, target_attack_ids):
        extension = chunk["filename"].fillna("").map(lambda value: Path(value).suffix.lower() or "none")
        chunk["event_type"] = "file"
        chunk["dst"] = "fileext:" + extension
        chunk["dst_type"] = "fileext"
        chunk["external"] = 0.0
        chunk["magnitude"] = chunk.get("content", pd.Series("", index=chunk.index)).fillna("").str.len().astype(float)
        frames.append(cap_events(chunk, cap))

    for chunk in read_filtered_csv(cert_root / "http.csv", user_set, all_attack_ids, target_attack_ids):
        domain = chunk["url"].fillna("unknown").map(domain_from_url)
        chunk["event_type"] = "http"
        chunk["dst"] = "domain:" + domain
        chunk["dst_type"] = "domain"
        chunk["external"] = 1.0
        chunk["magnitude"] = chunk.get("content", pd.Series("", index=chunk.index)).fillna("").str.len().astype(float)
        frames.append(cap_events(chunk, cap))

    for chunk in read_filtered_csv(cert_root / "email.csv", user_set, all_attack_ids, target_attack_ids):
        domain = chunk["to"].fillna("unknown").map(email_domain)
        chunk["event_type"] = "email"
        chunk["dst"] = "maildomain:" + domain
        chunk["dst_type"] = "maildomain"
        chunk["external"] = (domain != "dtaa.com").astype(float)
        size = pd.to_numeric(chunk.get("size", 0), errors="coerce").fillna(0)
        attachments = pd.to_numeric(chunk.get("attachments", 0), errors="coerce").fillna(0)
        chunk["magnitude"] = size + 10_000 * attachments
        frames.append(cap_events(chunk, cap))

    columns = [
        "event_id",
        "timestamp",
        "src",
        "dst",
        "dst_type",
        "event_type",
        "external",
        "magnitude",
        "is_any_attack",
        "attack_label",
    ]
    events = pd.concat([frame[columns] for frame in frames], ignore_index=True)
    events = events.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)
    # Other insiders must not be silently counted as benign controls.
    events = events[~(events["is_any_attack"] & ~events["attack_label"])].copy()
    return events


def assign_splits(events: pd.DataFrame, attack_start: pd.Timestamp, attack_end: pd.Timestamp, config: dict) -> pd.DataFrame:
    context_start = attack_start - pd.Timedelta(days=int(config["conditioning_context_days"]))
    validation_start = context_start - pd.Timedelta(days=int(config["validation_days"]))
    test_end = attack_end + pd.Timedelta(hours=int(config["post_payoff_hours"]))

    events["split"] = "discard"
    events.loc[(events["timestamp"] < validation_start) & ~events["is_any_attack"], "split"] = "train"
    events.loc[
        (events["timestamp"] >= validation_start)
        & (events["timestamp"] < context_start)
        & ~events["is_any_attack"],
        "split",
    ] = "validation"
    events.loc[
        (events["timestamp"] >= context_start)
        & (events["timestamp"] < attack_start)
        & ~events["is_any_attack"],
        "split",
    ] = "context"
    events.loc[
        (events["timestamp"] >= attack_start) & (events["timestamp"] <= test_end),
        "split",
    ] = "test"

    train = events[events["split"] == "train"]
    cap = int(config["train_event_cap"])
    if len(train) > cap:
        retained = set(train.tail(cap).index)
        events.loc[(events["split"] == "train") & ~events.index.isin(retained), "split"] = "discard"
    return events[events["split"] != "discard"].copy()


def vectorize(events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    event_to_id = {name: index for index, name in enumerate(EVENT_TYPES)}
    nodes = sorted(set(events["src"]) | set(events["dst"]))
    node_to_id = {node: index for index, node in enumerate(nodes)}
    dst_types = sorted(events["dst_type"].unique())
    dst_type_to_id = {name: index for index, name in enumerate(dst_types)}

    events["src_id"] = events["src"].map(node_to_id).astype("int64")
    events["dst_id"] = events["dst"].map(node_to_id).astype("int64")
    events["event_type_id"] = events["event_type"].map(event_to_id).astype("int64")
    events["dst_type_id"] = events["dst_type"].map(dst_type_to_id).astype("int64")
    # Pandas 3 may infer datetime64[us], for which astype("int64") yields
    # microseconds rather than nanoseconds.  Normalize the storage unit before
    # conversion so sub-minute event order is never collapsed.
    events["timestamp_s"] = (
        events["timestamp"].astype("datetime64[ns]").astype("int64")
        // 1_000_000_000
    ).astype("int64")

    hour = events["timestamp"].dt.hour + events["timestamp"].dt.minute / 60.0
    events["hour_sin"] = np.sin(2 * math.pi * hour / 24)
    events["hour_cos"] = np.cos(2 * math.pi * hour / 24)
    events["weekend"] = (events["timestamp"].dt.dayofweek >= 5).astype(float)
    events["after_hours"] = ((hour < 7) | (hour >= 19)).astype(float)
    events["log_magnitude"] = np.log1p(events["magnitude"].clip(lower=0))
    train_magnitude = events.loc[events["split"] == "train", "log_magnitude"]
    magnitude_mean = float(train_magnitude.mean())
    magnitude_std = float(train_magnitude.std()) or 1.0
    events["log_magnitude"] = (events["log_magnitude"] - magnitude_mean) / magnitude_std

    message_columns = []
    for event_type in EVENT_TYPES:
        column = f"msg_type_{event_type}"
        events[column] = (events["event_type"] == event_type).astype(float)
        message_columns.append(column)
    for column in ["hour_sin", "hour_cos", "weekend", "after_hours", "external", "log_magnitude"]:
        message_columns.append(column)
    for index, column in enumerate(message_columns):
        events[f"msg_{index}"] = events[column].astype("float32")
    output_message_columns = [f"msg_{index}" for index in range(len(message_columns))]

    node_types = {}
    for node, node_id in node_to_id.items():
        node_types[node_id] = node.split(":", 1)[0]
    metadata = {
        "event_types": EVENT_TYPES,
        "message_features": message_columns,
        "raw_msg_dim": len(message_columns),
        "node_to_id": node_to_id,
        "id_to_node": {str(value): key for key, value in node_to_id.items()},
        "node_types": {str(key): value for key, value in node_types.items()},
        "dst_type_to_id": dst_type_to_id,
        "magnitude_standardization": {"mean": magnitude_mean, "std": magnitude_std},
        "message_columns": output_message_columns,
    }
    keep = [
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
        *output_message_columns,
    ]
    output = events[keep].sort_values(["timestamp_s", "event_id"], kind="stable")
    if not output["timestamp_s"].is_monotonic_increasing:
        raise RuntimeError("Prepared event stream is not chronologically ordered")
    return output.reset_index(drop=True), metadata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_prepared(events: pd.DataFrame, metadata: dict) -> dict:
    required = {
        "event_id", "timestamp", "timestamp_s", "src", "dst", "src_id",
        "dst_id", "dst_type", "dst_type_id", "event_type", "event_type_id",
        "attack_label", "split",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Prepared events miss columns: {missing}")
    if events["event_id"].duplicated().any():
        raise ValueError("Event identifiers are not unique")
    if not events["timestamp_s"].is_monotonic_increasing:
        raise ValueError("Events are not chronological")
    attack = events[events["attack_label"]]
    if attack.empty or set(attack["split"]) != {"test"}:
        raise ValueError("Labelled payoff events must exist only in test")
    if events.loc[events["split"].isin(["train", "validation", "context"]), "attack_label"].any():
        raise ValueError("Attack-label leakage before test")
    expected = set(metadata["message_columns"])
    if not expected.issubset(events.columns):
        raise ValueError("Message-feature contract is incomplete")
    observed_nodes = set(events["src_id"]) | set(events["dst_id"])
    if max(observed_nodes) >= len(metadata["node_to_id"]):
        raise ValueError("Node identifiers exceed the declared vocabulary")
    return {
        "rows": int(len(events)),
        "nodes": int(len(metadata["node_to_id"])),
        "labelled_payoff_events": int(attack.shape[0]),
        "split_sizes": {k: int(v) for k, v in events["split"].value_counts().items()},
        "chronological": True,
        "leakage_free": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    base = config_path.parent
    config = load_config(config_path)
    paths = config["paths"]
    dataset = config["dataset"]
    cert_root = resolve(base, dataset["raw_root"])
    answers = resolve(base, dataset["answers_root"])
    prepared = resolve(base, paths["prepared"])
    prepared.mkdir(parents=True, exist_ok=True)

    if dataset.get("scope") == "full_longitudinal_all_incidents":
        from .cert_longitudinal import (
            prepare_longitudinal_cert,
            validate_longitudinal_prepared,
        )

        if args.validate_only:
            print(json.dumps(validate_longitudinal_prepared(prepared), indent=2))
            return
        print(
            json.dumps(
                prepare_longitudinal_cert(config, cert_root, answers, prepared),
                indent=2,
            )
        )
        return

    if dataset.get("scope") == "full_preincident_corpus":
        from .cert_full import prepare_full_cert, validate_full_prepared

        if args.validate_only:
            print(json.dumps(validate_full_prepared(prepared), indent=2))
            return
        print(
            json.dumps(
                prepare_full_cert(config, cert_root, answers, prepared), indent=2
            )
        )
        return

    if args.validate_only:
        events = pd.read_parquet(prepared / "events.parquet")
        metadata = json.loads((prepared / "metadata.json").read_text())
        print(json.dumps(validate_prepared(events, metadata), indent=2))
        return
    required_sources = [
        cert_root / "logon.csv",
        cert_root / "device.csv",
        cert_root / "file.csv",
        cert_root / "http.csv",
        cert_root / "email.csv",
    ]
    missing_sources = [str(path) for path in required_sources if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(f"CERT raw files are missing: {missing_sources}")
    all_attack_ids, target_attack_ids, target_answers = read_answer_events(
        answers,
        dataset["target_user"],
        int(dataset["target_scenario"]),
    )
    attack_start = target_answers["timestamp"].min()
    attack_end = target_answers["timestamp"].max()
    peers, role_info, role_profiles = load_role_peers(
        cert_root,
        dataset["target_user"],
        int(dataset["peer_count"]),
        attack_start,
    )
    events = build_events(
        cert_root,
        peers,
        all_attack_ids,
        target_attack_ids,
        int(dataset["per_user_day_type_cap"]),
    )
    events = assign_splits(events, attack_start, attack_end, dataset)
    events, metadata = vectorize(events)
    if int(events["attack_label"].sum()) == 0:
        raise RuntimeError("No target answer-key events survived preparation")
    if set(events.loc[events["attack_label"], "split"]) != {"test"}:
        raise RuntimeError("Target answer-key events leaked outside the test split")
    metadata.update(
        {
            "dataset": dataset["name"],
            "target_user": dataset["target_user"],
            "target_user_node": f"usr:{dataset['target_user']}",
            "target_scenario": int(dataset["target_scenario"]),
            "attack_start": attack_start.isoformat(),
            "attack_end": attack_end.isoformat(),
            "selected_users": peers,
            "role_info": role_info,
            "role_profiles": role_profiles,
            "split_sizes": {key: int(value) for key, value in events["split"].value_counts().items()},
            "target_attack_events_in_stream": int(events["attack_label"].sum()),
            "cert_root": str(cert_root),
            "time_origin_s": int(events["timestamp_s"].min()),
        }
    )
    events_path = prepared / "events.parquet"
    metadata_path = prepared / "metadata.json"
    events.to_parquet(events_path, index=False)
    metadata["events_sha256"] = sha256_file(events_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    validation = validate_prepared(events, metadata)
    manifest = {
        **validation,
        "prepared": str(events_path),
        "events_sha256": metadata["events_sha256"],
        "config": str(config_path),
        "target_role": role_info["target_attributes"],
        "source_root": str(cert_root),
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

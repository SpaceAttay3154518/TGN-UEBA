"""Prepare the public StreamSpot ALL corpus using KAIROS's exact graph split.

The source archive is streamed directly; its 2.3 GB TSV member is never
materialized.  Output is partitioned per graph so validation and test can reset
TGN state at the same graph boundary used by KAIROS.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


NODE_TYPES = ["a", "b", "c", "d", "e", "f", "g", "h"]
EVENT_TYPES = [*list("ijklmnopqrstuvwxyz"), *list("ABCDEFGH")]
NODE_TYPE_MEANINGS = {
    "a": "process",
    "b": "thread",
    "c": "file",
    "d": "MAP_ANONYMOUS",
    "e": "NA",
    "f": "stdin",
    "g": "stdout",
    "h": "stderr",
}
EVENT_TYPE_MEANINGS = {
    code: name
    for name, code in {
        "accept": "i", "access": "j", "bind": "k", "chmod": "l",
        "clone": "m", "close": "n", "connect": "o", "execve": "p",
        "fstat": "q", "ftruncate": "r", "listen": "s", "mmap2": "t",
        "open": "u", "read": "v", "recv": "w", "recvfrom": "x",
        "recvmsg": "y", "send": "z", "sendmsg": "A", "sendto": "B",
        "stat": "C", "truncate": "D", "unlink": "E", "waitpid": "F",
        "write": "G", "writev": "H",
    }.items()
}
BENIGN_BASES = (0, 100, 200, 400, 500)


def graph_split(graph_id: int) -> str:
    if graph_id in BENIGN_BASES:
        return "train"
    if any(base + 1 <= graph_id <= base + 24 for base in BENIGN_BASES):
        return "validation"
    if 300 <= graph_id <= 399:
        return "test"
    if any(base + 25 <= graph_id <= base + 99 for base in BENIGN_BASES):
        return "test"
    raise ValueError(f"Graph ID outside the StreamSpot contract: {graph_id}")


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _stable_frame_digest(digest: "hashlib._Hash", frame: pd.DataFrame) -> None:
    selected = frame[
        ["event_id", "timestamp_s", "src_id", "dst_id", "event_type_id", "split"]
    ]
    values = pd.util.hash_pandas_object(selected, index=False).to_numpy(np.uint64)
    digest.update(values.tobytes())


def prepare_streamspot(
    archive: Path,
    output: Path,
    *,
    chunk_rows: int = 500_000,
) -> dict:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if output.exists():
        shutil.rmtree(output)
    (output / "events").mkdir(parents=True)

    node_type_to_id = {value: index for index, value in enumerate(NODE_TYPES)}
    event_type_to_id = {value: index for index, value in enumerate(EVENT_TYPES)}
    split_rows: dict[str, int] = defaultdict(int)
    graph_rows: dict[int, int] = defaultdict(int)
    graph_fragments: dict[int, int] = defaultdict(int)
    observed_graphs: set[int] = set()
    max_node_id = -1
    offset = 0
    digest = hashlib.sha256()

    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.getmember("all.tsv")
        stream = bundle.extractfile(member)
        if stream is None:
            raise RuntimeError("all.tsv is missing from StreamSpot archive")
        reader = pd.read_csv(
            stream,
            sep="\t",
            header=None,
            names=["src_id", "src_type", "dst_id", "dst_type", "event_type", "graph_id"],
            dtype={
                "src_id": "int64",
                "src_type": "string",
                "dst_id": "int64",
                "dst_type": "string",
                "event_type": "string",
                "graph_id": "int16",
            },
            chunksize=int(chunk_rows),
        )
        for raw in reader:
            if raw.empty:
                continue
            unknown_nodes = (set(raw["src_type"]) | set(raw["dst_type"])) - set(NODE_TYPES)
            unknown_events = set(raw["event_type"]) - set(EVENT_TYPES)
            if unknown_nodes or unknown_events:
                raise ValueError(
                    f"Unknown StreamSpot codes: nodes={unknown_nodes}, events={unknown_events}"
                )
            frame = raw.copy()
            length = len(frame)
            frame["event_id"] = np.arange(offset, offset + length, dtype=np.int64)
            frame["timestamp_s"] = frame["event_id"] + 1
            frame["timestamp"] = pd.to_datetime(frame["timestamp_s"], unit="s", utc=True)
            frame["src_type_id"] = frame["src_type"].map(node_type_to_id).astype("int8")
            frame["dst_type_id"] = frame["dst_type"].map(node_type_to_id).astype("int8")
            frame["event_type_id"] = frame["event_type"].map(event_type_to_id).astype("int8")
            frame["attack_label"] = frame["graph_id"].between(300, 399)
            frame["window_id"] = frame["graph_id"].map(lambda value: f"graph-{int(value):03d}")
            frame["split"] = frame["graph_id"].map(lambda value: graph_split(int(value)))
            offset += length
            max_node_id = max(
                max_node_id,
                int(frame["src_id"].max()),
                int(frame["dst_id"].max()),
            )
            output_columns = [
                "event_id", "timestamp", "timestamp_s", "src_id", "dst_id",
                "src_type", "dst_type", "src_type_id", "dst_type_id",
                "event_type", "event_type_id", "attack_label", "window_id", "graph_id",
                "split",
            ]
            for graph_id, group in frame.groupby("graph_id", sort=True):
                gid = int(graph_id)
                split = graph_split(gid)
                observed_graphs.add(gid)
                directory = output / "events" / f"split={split}" / f"partition=graph-{gid:03d}"
                directory.mkdir(parents=True, exist_ok=True)
                fragment = graph_fragments[gid]
                selected = group[output_columns].sort_values("timestamp_s", kind="stable")
                selected.to_parquet(
                    directory / f"part-{fragment:05d}.parquet",
                    index=False,
                    compression="zstd",
                )
                graph_fragments[gid] += 1
                graph_rows[gid] += len(selected)
                split_rows[split] += len(selected)
                _stable_frame_digest(digest, selected)

    expected_graphs = set(range(600))
    if observed_graphs != expected_graphs:
        missing = sorted(expected_graphs - observed_graphs)
        extra = sorted(observed_graphs - expected_graphs)
        raise RuntimeError(f"StreamSpot graph contract failed: missing={missing}, extra={extra}")

    graph_units = {
        "train": sorted(BENIGN_BASES),
        "validation": sorted(
            graph for base in BENIGN_BASES for graph in range(base + 1, base + 25)
        ),
        "test_benign": sorted(
            graph for base in BENIGN_BASES for graph in range(base + 25, base + 100)
        ),
        "test_attack": list(range(300, 400)),
    }
    metadata = {
        "format_version": 3,
        "storage": "partitioned_parquet_by_split_and_graph",
        "partition_key": "partition",
        "dataset": "Manzoor et al. StreamSpot ALL",
        "dataset_id": "streamspot",
        "num_nodes": max_node_id + 1,
        "implicit_node_ids": True,
        "node_types": NODE_TYPES,
        "node_type_to_id": node_type_to_id,
        "node_type_meanings": NODE_TYPE_MEANINGS,
        "event_types": EVENT_TYPES,
        "event_type_to_id": event_type_to_id,
        "event_type_meanings": EVENT_TYPE_MEANINGS,
        "dst_type_to_id": node_type_to_id,
        "raw_msg_dim": len(NODE_TYPES) + len(EVENT_TYPES) + len(NODE_TYPES),
        "message_encoding": "src_type_one_hot+edge_type_one_hot+dst_type_one_hot",
        "message_columns": [],
        "time_origin_s": 1,
        "split_sizes": dict(split_rows),
        "graph_rows": {str(key): int(value) for key, value in sorted(graph_rows.items())},
        "graph_units": graph_units,
        "model_selection_graphs": [505],
        "reset_temporal_state": {"train": False, "validation": True, "test": True},
        "negative_sampling": {
            "strategy": "global_uniform_kairos_compatibility",
            "minimum_node_id": 0,
            "maximum_node_id": max_node_id,
            "note": "Matches KAIROS StreamSpot code, which samples over its declared global node range."
        },
        "evaluation_granularity": "graph",
        "kairos_reference_units": {"TP": 100, "TN": 375, "FP": 0, "FN": 0},
        "source_archive": str(archive.resolve()),
        "source_sha256": _sha256(archive),
        "events_sha256": digest.hexdigest(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    manifest = {
        "format_version": 3,
        "dataset_id": "streamspot",
        "rows": int(offset),
        "nodes": int(max_node_id + 1),
        "graphs": len(observed_graphs),
        "split_sizes": dict(split_rows),
        "graph_unit_counts": {
            "train": len(graph_units["train"]),
            "validation": len(graph_units["validation"]),
            "test_benign": len(graph_units["test_benign"]),
            "test_attack": len(graph_units["test_attack"]),
        },
        "chronological_within_graph": True,
        "kairos_split_exact": True,
        "training_started": False,
        "source_sha256": metadata["source_sha256"],
        "events_sha256": metadata["events_sha256"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def validate_streamspot(output: Path) -> dict:
    metadata = json.loads((output / "metadata.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    if metadata.get("dataset_id") != "streamspot":
        raise ValueError("Not a StreamSpot prepared directory")
    if int(manifest.get("graphs", 0)) != 600:
        raise ValueError("StreamSpot must contain all 600 graphs")
    counts = manifest.get("graph_unit_counts", {})
    if counts != {"train": 5, "validation": 120, "test_benign": 375, "test_attack": 100}:
        raise ValueError(f"Wrong KAIROS graph split: {counts}")
    for split in ("train", "validation", "test"):
        directory = output / "events" / f"split={split}"
        if not directory.is_dir() or not next(directory.rglob("*.parquet"), None):
            raise FileNotFoundError(f"Empty StreamSpot split: {split}")
    return manifest

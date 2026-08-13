"""Configuration, prepared-data, and checkpoint I/O."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch

from .model import TGNDetector


def load_config(path: Path) -> dict:
    path = path.resolve()
    config = json.loads(path.read_text())
    config["_config_path"] = str(path)
    return config


def resolve_config_path(config: dict, value: str) -> Path:
    base = Path(config["_config_path"]).parent
    alias = config.get("_path_aliases", {}).get(value)
    if alias is not None:
        return (base / alias).resolve()
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def canonical_config_hash(config: dict) -> str:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


class PreparedDataset:
    """Chronological reader for partitioned full-corpus Parquet data."""

    def __init__(self, root: Path, metadata: dict) -> None:
        self.root = root
        self.metadata = metadata
        feature_file = metadata.get("node_features_file")
        self.node_features = None
        if feature_file:
            feature_path = root / feature_file
            self.node_features = np.load(feature_path, mmap_mode="r")
            self.metadata["_node_features"] = self.node_features

    def _day_directories(self, split: str) -> list[Path]:
        directory = self.root / "events" / f"split={split}"
        if not directory.exists():
            raise FileNotFoundError(f"Prepared split does not exist: {directory}")
        partition_key = str(self.metadata.get("partition_key", "day"))
        return sorted(
            path for path in directory.glob(f"{partition_key}=*") if path.is_dir()
        )

    def iter_split(self, split: str, rows_per_frame: int | None = None) -> Iterator[pd.DataFrame]:
        origin = int(self.metadata["time_origin_s"])
        for directory in self._day_directories(split):
            files = sorted(directory.glob("*.parquet"))
            if not files:
                continue
            frame = pd.concat(
                [pd.read_parquet(path) for path in files], ignore_index=True
            ).sort_values(["timestamp_s", "event_id"], kind="stable")
            frame["time_rel"] = frame["timestamp_s"] - origin + 1
            frame = frame.reset_index(drop=True)
            step = len(frame) if not rows_per_frame else int(rows_per_frame)
            for start in range(0, len(frame), max(step, 1)):
                yield frame.iloc[start : start + step].copy()

    def load_splits(self, *splits: str) -> pd.DataFrame:
        frames = [frame for split in splits for frame in self.iter_split(split)]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(
            ["timestamp_s", "event_id"], kind="stable"
        ).reset_index(drop=True)

    def role_events(self, splits: tuple[str, ...], sources: set[str]) -> pd.DataFrame:
        frames = []
        for split in splits:
            for frame in self.iter_split(split):
                selected = frame[frame["src"].isin(sources)]
                if not selected.empty:
                    frames.append(selected)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(
            ["timestamp_s", "event_id"], kind="stable"
        )


def load_prepared(config: dict) -> tuple[pd.DataFrame | PreparedDataset, dict]:
    prepared = resolve_config_path(config, config["paths"]["prepared"])
    metadata_path = prepared / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Prepared data not found in {prepared}; run `emdd prepare --config ...` first"
        )
    metadata = json.loads(metadata_path.read_text())
    if int(metadata.get("format_version", 1)) >= 2:
        return PreparedDataset(prepared, metadata), metadata
    events_path = prepared / "events.parquet"
    if not events_path.exists():
        raise FileNotFoundError(f"Prepared event file not found: {events_path}")
    events = pd.read_parquet(events_path)
    origin = int(events["timestamp_s"].min())
    metadata.setdefault("time_origin_s", origin)
    events["time_rel"] = events["timestamp_s"] - origin + 1
    return events, metadata


def build_model(config: dict, metadata: dict, device: torch.device) -> TGNDetector:
    model = config["model"]
    num_nodes = int(metadata.get("num_nodes", len(metadata.get("node_to_id", {}))))
    if num_nodes <= 0:
        raise ValueError("Prepared metadata does not declare a positive node count")
    return TGNDetector(
        num_nodes=num_nodes,
        raw_msg_dim=int(metadata["raw_msg_dim"]),
        memory_dim=int(model["memory_dim"]),
        time_dim=int(model["time_dim"]),
        embedding_dim=int(model["embedding_dim"]),
        neighbor_size=int(model["neighbor_size"]),
        device=device,
        num_edge_types=len(metadata["event_types"]),
        attention_heads=int(model.get("attention_heads", 2)),
        edge_type_loss_weight=float(model.get("edge_type_loss_weight", 0.0)),
        link_loss_weight=float(model.get("link_loss_weight", 1.0)),
        decoder_dropout=float(model.get("decoder_dropout", 0.0)),
        guard_temperature=float(model.get("guard_temperature", 0.0)),
        memory_decay_beta=float(model.get("memory_decay_beta", 0.0)),
        embedding_layers=int(model.get("embedding_layers", 2)),
        kairos_attention_layout=bool(model.get("kairos_attention_layout", False)),
    )


def save_checkpoint(
    path: Path,
    detector: TGNDetector,
    config: dict,
    metadata: dict,
    seed: int,
    training_summary: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "seed": int(seed),
            "config_hash": canonical_config_hash(config),
            "events_sha256": metadata.get("events_sha256"),
            "model_state_dict": {
                key: value.detach().cpu() for key, value in detector.state_dict().items()
            },
            "temporal_state": detector.temporal_state_dict(),
            "training_summary": training_summary,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    config: dict,
    metadata: dict,
    device: torch.device,
    *,
    enforce_hash: bool = True,
) -> tuple[TGNDetector, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format_version')}")
    if enforce_hash and payload["config_hash"] != canonical_config_hash(config):
        raise ValueError("Checkpoint configuration hash does not match this run")
    expected_data = metadata.get("events_sha256")
    if expected_data and payload.get("events_sha256") != expected_data:
        raise ValueError("Checkpoint was trained on a different prepared event stream")
    detector = build_model(config, metadata, device)
    # Change mode while the freshly constructed temporal stores are empty.
    # TGNMemory flushes pending messages on a normal train->eval transition.
    detector.train(bool(payload["temporal_state"].get("training", False)))
    detector.load_state_dict(payload["model_state_dict"])
    detector.load_temporal_state_dict(payload["temporal_state"])
    return detector, payload

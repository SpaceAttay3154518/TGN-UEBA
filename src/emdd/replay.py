"""Shared causal replay utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .model import Batch, TGNDetector


def tensors_from_frame(
    frame: pd.DataFrame, metadata: dict
) -> tuple[torch.Tensor, ...]:
    messages = messages_from_frame(frame, metadata)
    return (
        torch.from_numpy(frame["src_id"].to_numpy(np.int64).copy()),
        torch.from_numpy(frame["dst_id"].to_numpy(np.int64).copy()),
        torch.from_numpy(frame["time_rel"].to_numpy(np.int64).copy()),
        torch.from_numpy(messages),
        torch.from_numpy(frame["event_type_id"].to_numpy(np.int64).copy()),
    )


def messages_from_frame(frame: pd.DataFrame, metadata: dict) -> np.ndarray:
    columns = metadata.get("message_columns", [])
    if columns:
        return frame[columns].to_numpy(np.float32).copy()
    encoding = metadata.get("message_encoding")
    if encoding == "src_type_one_hot+edge_type_one_hot+dst_type_one_hot":
        node_type_count = len(metadata["node_types"])
        event_type_count = len(metadata["event_types"])
        src_type = frame["src_type_id"].to_numpy(np.int64)
        dst_type = frame["dst_type_id"].to_numpy(np.int64)
        edge_type = frame["event_type_id"].to_numpy(np.int64)
        src_one_hot = np.zeros((len(frame), node_type_count), dtype=np.float32)
        edge_one_hot = np.zeros((len(frame), event_type_count), dtype=np.float32)
        dst_one_hot = np.zeros((len(frame), node_type_count), dtype=np.float32)
        rows = np.arange(len(frame))
        src_one_hot[rows, src_type] = 1.0
        edge_one_hot[rows, edge_type] = 1.0
        dst_one_hot[rows, dst_type] = 1.0
        return np.concatenate([src_one_hot, edge_one_hot, dst_one_hot], axis=1)
    features = np.asarray(metadata["_node_features"], dtype=np.float32)
    src = features[frame["src_id"].to_numpy(np.int64)]
    dst = features[frame["dst_id"].to_numpy(np.int64)]
    edge_type = frame["event_type_id"].to_numpy(np.int64)
    one_hot = np.zeros((len(frame), len(metadata["event_types"])), dtype=np.float32)
    one_hot[np.arange(len(frame)), edge_type] = 1.0
    return np.concatenate([src, one_hot, dst], axis=1).astype(np.float32, copy=False)


def destination_candidates(events: pd.DataFrame) -> dict[int, np.ndarray]:
    pools = {
        int(kind): np.sort(group["dst_id"].unique().astype(np.int64))
        for kind, group in events.groupby("dst_type_id")
    }
    pools[-1] = np.sort(events["dst_id"].unique().astype(np.int64))
    return pools


def sample_negatives(
    frame: pd.DataFrame, pools: dict[int, np.ndarray], seed: int
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    output = np.empty(len(frame), dtype=np.int64)
    for position, row in enumerate(frame.itertuples(index=False)):
        pool = pools.get(int(row.dst_type_id), pools[-1])
        if len(pool) <= 1:
            pool = pools[-1]
        choice = int(pool[rng.integers(0, len(pool))])
        if choice == int(row.dst_id) and len(pool) > 1:
            where = int(np.searchsorted(pool, choice))
            choice = int(pool[(where + 1) % len(pool)])
        output[position] = choice
    return torch.from_numpy(output)


def batch_from_row(row: pd.Series, metadata: dict, device: torch.device) -> Batch:
    message = messages_from_frame(pd.DataFrame([row]), metadata)
    return Batch(
        src=torch.tensor([int(row["src_id"])], dtype=torch.long, device=device),
        dst=torch.tensor([int(row["dst_id"])], dtype=torch.long, device=device),
        t=torch.tensor([int(row["time_rel"])], dtype=torch.long, device=device),
        msg=torch.from_numpy(message).to(device),
        edge_type=torch.tensor(
            [int(row["event_type_id"])], dtype=torch.long, device=device
        ),
    )


@torch.no_grad()
def score_batch(detector: TGNDetector, batch: Batch) -> float:
    score = detector.anomaly(batch)
    return float(score.squeeze().cpu())


@dataclass
class ReplayResult:
    scores: pd.DataFrame
    diagnostics: dict


@torch.no_grad()
def replay_frame(
    detector: TGNDetector,
    frame: pd.DataFrame,
    metadata: dict,
    condition: str,
    *,
    negative_pools: dict[int, np.ndarray] | None = None,
    seed: int = 0,
    record_memory_nodes: set[int] | None = None,
) -> ReplayResult:
    detector.eval()
    device = detector.device_ref
    negatives = (
        None if negative_pools is None else sample_negatives(frame, negative_pools, seed)
    )
    positive_probabilities: list[float] = []
    negative_probabilities: list[float] = []
    predicted_types: list[int] = []
    observed_types: list[int] = []
    rows: list[dict] = []
    memory_nodes = record_memory_nodes or set()
    for position, (_, row) in enumerate(frame.iterrows()):
        batch = batch_from_row(row, metadata, device)
        before = {
            node: detector.memory.memory[node].detach().clone() for node in memory_nodes
        }
        if negatives is None:
            positive, _ = detector.logits(
                batch.src, batch.dst, query_time=batch.t
            )
        else:
            negative = negatives[position : position + 1].to(device)
            positive, negative_logit = detector.logits(
                batch.src, batch.dst, negative, query_time=batch.t
            )
            assert negative_logit is not None
            negative_probabilities.append(float(torch.sigmoid(negative_logit).cpu()))
        anomaly = float(detector.anomaly(batch).mean().cpu())
        if detector.edge_decoder is not None and batch.edge_type is not None:
            predicted_types.append(int(detector.edge_type_logits(batch).argmax(dim=-1).cpu()))
            observed_types.append(int(batch.edge_type.cpu()))
        positive_probabilities.append(float(torch.sigmoid(positive).cpu()))
        detector.observe(batch)
        shifts = {
            str(node): float(
                torch.linalg.vector_norm(detector.memory.memory[node] - state).cpu()
            )
            for node, state in before.items()
        }
        rows.append(
            {
                "condition": condition,
                "event_id": row["event_id"],
                "timestamp": row["timestamp"],
                "src": row.get("src", f"node:{int(row['src_id'])}"),
                "dst": row.get("dst", f"node:{int(row['dst_id'])}"),
                "src_id": int(row["src_id"]),
                "dst_id": int(row["dst_id"]),
                "event_type": row["event_type"],
                "attack_label": bool(row["attack_label"]),
                "anomaly_score": anomaly,
                "memory_shifts": shifts,
            }
        )
    diagnostics: dict[str, float | int] = {"events": len(rows)}
    if negatives is not None:
        labels = np.r_[
            np.ones(len(positive_probabilities)), np.zeros(len(negative_probabilities))
        ]
        values = np.r_[positive_probabilities, negative_probabilities]
        diagnostics["link_ap"] = float(average_precision_score(labels, values))
        diagnostics["link_auc"] = float(roc_auc_score(labels, values))
    if observed_types:
        diagnostics["edge_type_accuracy"] = float(
            np.mean(np.asarray(predicted_types) == np.asarray(observed_types))
        )
        diagnostics["edge_type_macro_f1"] = float(
            f1_score(observed_types, predicted_types, average="macro", zero_division=0)
        )
        diagnostics["mean_reconstruction_error"] = float(
            pd.DataFrame(rows)["anomaly_score"].mean()
        )
    return ReplayResult(pd.DataFrame(rows), diagnostics)

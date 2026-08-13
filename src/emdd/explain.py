"""Diagnostic attention, state attribution, and counterfactual fidelity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .model import TGNDetector
from .replay import batch_from_row, score_batch


@torch.no_grad()
def attention_explanation(
    detector: TGNDetector,
    event: pd.Series,
    history: pd.DataFrame,
    metadata: dict,
    top_k: int = 10,
) -> pd.DataFrame:
    batch = batch_from_row(event, metadata, detector.device_ref)
    trace = detector.attention_trace(batch.src, batch.dst, batch.t)
    event_ids = trace["event_id"].detach().cpu().numpy()
    weights = trace["weights"].detach().cpu().numpy()
    # TransformerConv can append self-loops. Only original neighbor edges map
    # to historical event IDs and are retained here.
    count = min(len(event_ids), len(weights))
    if count == 0:
        return pd.DataFrame(columns=["event_id", "attention", "diagnostic_only"])
    mean_weight = weights[:count].mean(axis=-1) if weights.ndim == 2 else weights[:count]
    rows = []
    history_aligned = len(history) == int(detector.history_t.numel())
    for internal_id, weight in zip(event_ids[:count], mean_weight):
        slot = int(internal_id)
        source = history.iloc[slot] if history_aligned and slot < len(history) else None
        rows.append(
            {
                "internal_event_slot": slot,
                "event_id": None if source is None else source["event_id"],
                "timestamp": (
                    int(detector.history_t[slot].cpu())
                    if source is None and slot < detector.history_t.numel()
                    else source["timestamp"]
                ),
                "src": None if source is None else source["src"],
                "dst": None if source is None else source["dst"],
                "event_type": None if source is None else source["event_type"],
                "attention": float(weight),
                "history_mapping_available": source is not None,
                "diagnostic_only": True,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "event_id",
                "internal_event_slot",
                "timestamp",
                "src",
                "dst",
                "event_type",
                "attention",
                "history_mapping_available",
                "diagnostic_only",
            ]
        )
    return pd.DataFrame(rows).sort_values("attention", ascending=False).head(top_k)


@torch.no_grad()
def score_after_history(
    start: TGNDetector,
    history: pd.DataFrame,
    target: pd.Series,
    metadata: dict,
    omitted_event_ids: set[str] | None = None,
) -> float:
    branch = start.fork()
    omitted = omitted_event_ids or set()
    for _, row in history.sort_values(["time_rel", "event_id"]).iterrows():
        if str(row["event_id"]) in omitted:
            continue
        branch.observe(batch_from_row(row, metadata, branch.device_ref))
    return score_batch(branch, batch_from_row(target, metadata, branch.device_ref))


@torch.no_grad()
def counterfactual_deletion_explanation(
    start: TGNDetector,
    history: pd.DataFrame,
    target: pd.Series,
    metadata: dict,
    candidate_event_ids: list[str],
) -> pd.DataFrame:
    baseline = score_after_history(start, history, target, metadata)
    rows = []
    lookup = history.set_index("event_id", drop=False)
    for event_id in candidate_event_ids:
        without = score_after_history(
            start, history, target, metadata, {str(event_id)}
        )
        source = lookup.loc[event_id] if event_id in lookup.index else None
        rows.append(
            {
                "event_id": event_id,
                "timestamp": None if source is None else source["timestamp"],
                "event_type": None if source is None else source["event_type"],
                "baseline_anomaly": baseline,
                "anomaly_without_event": without,
                "necessity_effect": baseline - without,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "event_id",
                "timestamp",
                "event_type",
                "baseline_anomaly",
                "anomaly_without_event",
                "necessity_effect",
            ]
        )
    return pd.DataFrame(rows).sort_values("necessity_effect", ascending=False)


def state_movement_attribution(
    before: torch.Tensor,
    after: torch.Tensor,
    role_reference: torch.Tensor,
) -> dict[str, float]:
    movement = after - before
    away = before - role_reference
    denominator = float(torch.linalg.vector_norm(movement).cpu()) * float(
        torch.linalg.vector_norm(away).cpu()
    )
    alignment = 0.0
    if denominator > 1e-8:
        alignment = float(torch.dot(movement, away).cpu()) / denominator
    return {
        "memory_step_l2": float(torch.linalg.vector_norm(movement).cpu()),
        "alignment_away_from_role": alignment,
        "before_role_distance_l2": float(torch.linalg.vector_norm(away).cpu()),
        "after_role_distance_l2": float(
            torch.linalg.vector_norm(after - role_reference).cpu()
        ),
    }


@dataclass(frozen=True)
class MitreCandidate:
    technique_id: str
    technique_name: str
    basis: str
    confidence: str = "candidate"


def mitre_candidates(event: pd.Series) -> list[MitreCandidate]:
    """Return transparent CERT-proxy mappings, never ground-truth labels."""
    event_type = str(event["event_type"])
    destination = str(event["dst"])
    candidates: list[MitreCandidate] = []
    if event_type == "logon":
        candidates.append(MitreCandidate("T1078", "Valid Accounts", "successful logon event"))
    if event_type.startswith("device_"):
        candidates.append(
            MitreCandidate("T1091", "Replication Through Removable Media", "removable-device activity")
        )
    if event_type == "file" and destination.endswith(".exe"):
        candidates.append(MitreCandidate("T1105", "Ingress Tool Transfer", "executable file activity"))
    if event_type == "http":
        candidates.append(MitreCandidate("T1071.001", "Web Protocols", "HTTP destination"))
    return candidates

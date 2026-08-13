"""Discrete, realizable slow-conditioning attack planner.

The planner never writes arbitrary latent vectors. Candidate actions are clean
event templates observed for role peers and are passed through the normal TGN
message/update path. A lightweight payoff-only proxy ranks candidates; the
final plan is always evaluated by exact chronological replay of all real events.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .model import TGNDetector
from .replay import batch_from_row, score_batch


@dataclass(frozen=True)
class ActionTemplate:
    event_type: str
    event_type_id: int
    dst: str
    dst_id: int
    dst_type: str
    dst_type_id: int
    message: tuple[float, ...]
    role_penalty: float
    support: int


@dataclass
class PlanResult:
    target_node: int
    duration_days: int
    interactions_per_day: int
    attempted_slots: int
    feasible_events: int
    missed_slots: int
    objective: float
    proxy_payoff_anomaly: float
    conditioning_events: list[dict]
    success_candidate: bool

    def to_dict(self) -> dict:
        return asdict(self)


def build_action_library(
    clean_events: pd.DataFrame,
    metadata: dict,
    role_user_nodes: set[str],
    pool_size: int,
) -> list[ActionTemplate]:
    frame = clean_events[
        clean_events["src"].isin(role_user_nodes) & ~clean_events["attack_label"]
    ].copy()
    if frame.empty:
        raise ValueError("No clean role-peer events exist for the action grammar")
    columns = metadata["message_columns"]
    counts = frame.groupby(["event_type", "dst"], sort=False).size().rename("count")
    total = int(counts.sum())
    groups = frame.groupby(["event_type", "dst"], sort=False)
    templates: list[ActionTemplate] = []
    for (event_type, destination), group in groups:
        count = int(counts.loc[(event_type, destination)])
        row = group.iloc[0]
        # Smoothed negative log-frequency. Frequent role actions cost less.
        penalty = -math.log((count + 1) / (total + len(counts)))
        message = tuple(float(value) for value in group[columns].mean().to_numpy())
        templates.append(
            ActionTemplate(
                event_type=str(event_type),
                event_type_id=int(row["event_type_id"]),
                dst=str(destination),
                dst_id=int(row["dst_id"]),
                dst_type=str(row["dst_type"]),
                dst_type_id=int(row["dst_type_id"]),
                message=message,
                role_penalty=penalty,
                support=count,
            )
        )
    templates.sort(key=lambda item: (-item.support, item.role_penalty, item.dst))
    return templates[:pool_size]


def build_payoff_matched_library(
    clean_events: pd.DataFrame,
    metadata: dict,
    role_user_nodes: set[str],
    payoff: pd.DataFrame,
    pool_size: int,
) -> list[ActionTemplate]:
    """Build a peer-observed grammar that covers the payoff's event types.

    Exact event-type and destination pairs are preferred when a clean role peer
    performed them. For unseen payoff destinations, the most frequent peer
    template of the same event type is used. The attacker therefore conditions
    the detector with realizable role activity without copying an event that was
    absent from clean development data.
    """
    frame = clean_events[
        clean_events["src"].isin(role_user_nodes) & ~clean_events["attack_label"]
    ].copy()
    if frame.empty:
        raise ValueError("No clean role-peer events exist for the action grammar")
    if payoff.empty:
        raise ValueError("A non-empty payoff is required for payoff matching")
    columns = metadata["message_columns"]
    grouped = frame.groupby(["event_type", "dst"], sort=False)
    counts = grouped.size().rename("count")
    total = int(counts.sum())
    group_count = len(counts)

    def make(key: tuple[str, str]) -> ActionTemplate:
        group = grouped.get_group(key)
        row = group.iloc[0]
        count = int(counts.loc[key])
        penalty = -math.log((count + 1) / (total + group_count))
        message = tuple(float(value) for value in group[columns].mean().to_numpy())
        return ActionTemplate(
            event_type=str(key[0]),
            event_type_id=int(row["event_type_id"]),
            dst=str(key[1]),
            dst_id=int(row["dst_id"]),
            dst_type=str(row["dst_type"]),
            dst_type_id=int(row["dst_type_id"]),
            message=message,
            role_penalty=penalty,
            support=count,
        )

    payoff_counts = payoff.groupby(["event_type", "dst"], sort=False).size()
    selected: list[tuple[str, str]] = []
    for key, _ in sorted(
        payoff_counts.items(), key=lambda item: (-int(item[1]), str(item[0]))
    ):
        normalized = (str(key[0]), str(key[1]))
        if normalized in counts.index and normalized not in selected:
            selected.append(normalized)
    payoff_types = list(dict.fromkeys(str(value) for value in payoff["event_type"]))
    for event_type in payoff_types:
        same_type = counts.loc[event_type] if event_type in counts.index.levels[0] else None
        if same_type is None or same_type.empty:
            continue
        destination = str(same_type.sort_values(ascending=False).index[0])
        key = (event_type, destination)
        if key not in selected:
            selected.append(key)
    for key, _ in counts.sort_values(ascending=False).items():
        normalized = (str(key[0]), str(key[1]))
        if normalized not in selected:
            selected.append(normalized)
        if len(selected) >= pool_size:
            break
    return [make(key) for key in selected[:pool_size]]


def conditioning_slots(
    attack_start: pd.Timestamp, duration_days: int, interactions_per_day: int
) -> list[pd.Timestamp]:
    window_start = attack_start - pd.Timedelta(days=duration_days)
    start_day = window_start.normalize()
    slots: list[pd.Timestamp] = []
    for day in range(duration_days):
        current = start_day + pd.Timedelta(days=day)
        # Spread actions through a normal workday.  On the first day, begin no
        # earlier than the exact D-day branch snapshot; otherwise a normalized
        # calendar start can silently make replay time run backwards.
        earliest = current + pd.Timedelta(hours=8, minutes=30)
        latest = current + pd.Timedelta(hours=17, minutes=30)
        if day == 0:
            earliest = max(earliest, window_start)
        if latest < earliest:
            latest = min(
                current + pd.Timedelta(hours=23, minutes=59),
                earliest + pd.Timedelta(minutes=max(interactions_per_day - 1, 1)),
            )
        slots.extend(
            pd.date_range(earliest, latest, periods=interactions_per_day).tolist()
        )
    output = [slot for slot in slots if window_start <= slot < attack_start]
    if len(output) != duration_days * interactions_per_day:
        raise ValueError(
            "The payoff timestamp leaves too little room for the requested "
            "conditioning schedule"
        )
    return output


def materialize_action(
    template: ActionTemplate,
    target_node: int,
    target_name: str,
    timestamp: pd.Timestamp,
    time_origin_s: int,
    metadata: dict,
    ordinal: int,
) -> pd.Series:
    values = np.asarray(template.message, dtype=np.float32).copy()
    features = metadata["message_features"]
    hour = timestamp.hour + timestamp.minute / 60.0
    replacements = {
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "weekend": float(timestamp.dayofweek >= 5),
        "after_hours": float(hour < 7 or hour >= 19),
    }
    for name, value in replacements.items():
        if name in features:
            values[features.index(name)] = value
    # NumPy's naive datetime arithmetic is timezone-independent, unlike
    # ``datetime.timestamp()`` which consults the host timezone.
    timestamp_s = int(
        timestamp.to_datetime64().astype("datetime64[s]").astype(np.int64)
    )
    row = {
        "event_id": f"conditioning:{ordinal:04d}",
        "timestamp": timestamp,
        "timestamp_s": timestamp_s,
        "time_rel": timestamp_s - time_origin_s + 1,
        "src": target_name,
        "dst": template.dst,
        "src_id": int(target_node),
        "dst_id": template.dst_id,
        "dst_type": template.dst_type,
        "dst_type_id": template.dst_type_id,
        "event_type": template.event_type,
        "event_type_id": template.event_type_id,
        "attack_label": False,
        "split": "conditioning",
        "role_penalty": template.role_penalty,
    }
    row.update({column: float(values[i]) for i, column in enumerate(metadata["message_columns"])})
    return pd.Series(row)


@torch.no_grad()
def payoff_proxy(
    detector: TGNDetector,
    payoff: pd.DataFrame,
    metadata: dict,
) -> float:
    branch = detector.fork()
    scores = []
    for _, row in payoff.sort_values(["time_rel", "event_id"]).iterrows():
        batch = batch_from_row(row, metadata, branch.device_ref)
        scores.append(score_batch(branch, batch))
        branch.observe(batch)
    return float(np.mean(scores)) if scores else float("inf")


@torch.no_grad()
def advance_real_events(
    detector: TGNDetector,
    rows: Iterable[tuple[int, pd.Series]],
    metadata: dict,
) -> None:
    for _, row in rows:
        detector.observe(batch_from_row(row, metadata, detector.device_ref))


@dataclass
class _Beam:
    detector: TGNDetector
    events: list[dict]
    objective: float
    proxy: float
    accumulated_role_penalty: float
    accumulated_step_penalty: float


def plan_slow_conditioning(
    start_detector: TGNDetector,
    context: pd.DataFrame,
    payoff: pd.DataFrame,
    templates: list[ActionTemplate],
    metadata: dict,
    attack_config: dict,
    *,
    target_node: int,
    target_name: str,
    threshold: float,
    duration_days: int,
    interactions_per_day: int,
) -> PlanResult:
    if payoff.empty:
        raise ValueError("A labelled payoff is required")
    attack_start = pd.Timestamp(payoff["timestamp"].min())
    slots = conditioning_slots(attack_start, duration_days, interactions_per_day)
    window_start = attack_start - pd.Timedelta(days=duration_days)
    prefix = context[context["timestamp"] < window_start].sort_values(
        ["timestamp", "event_id"]
    )
    real = context[context["timestamp"] >= window_start].sort_values(
        ["timestamp", "event_id"]
    )
    # Synthetic events must share the exact integer time origin used by the
    # prepared stream.  This avoids both timezone-dependent conversion and a
    # conditioning-window-relative clock.
    origin = int(metadata.get("time_origin_s", 0))
    if not origin:
        origin = int(min(context["timestamp_s"].min(), payoff["timestamp_s"].min()))
        origin -= int(min(context["time_rel"].min(), payoff["time_rel"].min())) - 1
    base = start_detector.fork()
    advance_real_events(base, prefix.iterrows(), metadata)
    beams = [_Beam(base, [], float("inf"), float("inf"), 0.0, 0.0)]
    cursor = 0
    real_rows = list(real.iterrows())
    missed = 0

    for ordinal, slot in enumerate(slots):
        next_cursor = cursor
        while next_cursor < len(real_rows) and real_rows[next_cursor][1]["timestamp"] <= slot:
            next_cursor += 1
        segment = real_rows[cursor:next_cursor]
        for beam in beams:
            advance_real_events(beam.detector, segment, metadata)
        cursor = next_cursor

        expansions: list[_Beam] = []
        for beam in beams:
            for template in templates:
                row = materialize_action(
                    template,
                    target_node,
                    target_name,
                    slot,
                    origin,
                    metadata,
                    ordinal,
                )
                candidate = beam.detector.fork()
                batch = batch_from_row(row, metadata, candidate.device_ref)
                anomaly = score_batch(candidate, batch)
                if anomaly >= threshold:
                    continue
                before = candidate.memory.memory[target_node].detach().clone()
                candidate.observe(batch)
                step = float(
                    torch.linalg.vector_norm(
                        candidate.memory.memory[target_node] - before
                    ).cpu()
                )
                accumulated_role = beam.accumulated_role_penalty + template.role_penalty
                accumulated_step = beam.accumulated_step_penalty + step**2
                proxy = payoff_proxy(candidate, payoff, metadata)
                objective = (
                    float(attack_config["payoff_weight"]) * proxy
                    + float(attack_config["role_penalty_weight"]) * accumulated_role
                    + float(attack_config["step_penalty_weight"]) * accumulated_step
                )
                event = row.to_dict()
                event.update(
                    {
                        "pre_update_anomaly": anomaly,
                        "memory_step_l2": step,
                        "proxy_payoff_anomaly": proxy,
                        "objective": objective,
                    }
                )
                expansions.append(
                    _Beam(
                        candidate,
                        [*beam.events, event],
                        objective,
                        proxy,
                        accumulated_role,
                        accumulated_step,
                    )
                )
        if expansions:
            expansions.sort(key=lambda beam: beam.objective)
            beams = expansions[: int(attack_config["beam_width"])]
        else:
            missed += 1

    for beam in beams:
        advance_real_events(beam.detector, real_rows[cursor:], metadata)
        beam.proxy = payoff_proxy(beam.detector, payoff, metadata)
    best = min(beams, key=lambda beam: (beam.proxy, beam.objective))
    return PlanResult(
        target_node=target_node,
        duration_days=duration_days,
        interactions_per_day=interactions_per_day,
        attempted_slots=len(slots),
        feasible_events=len(best.events),
        missed_slots=missed,
        objective=float(best.objective),
        proxy_payoff_anomaly=float(best.proxy),
        conditioning_events=best.events,
        success_candidate=len(best.events) > 0,
    )


def interleave_conditioning(
    context: pd.DataFrame, conditioning_events: list[dict]
) -> pd.DataFrame:
    if not conditioning_events:
        return context.copy()
    injected = pd.DataFrame(conditioning_events)
    all_columns = list(context.columns)
    for column in all_columns:
        if column not in injected:
            injected[column] = False if column == "attack_label" else np.nan
    return (
        pd.concat([context, injected[all_columns]], ignore_index=True)
        .sort_values(["timestamp", "event_id"], kind="stable")
        .reset_index(drop=True)
    )

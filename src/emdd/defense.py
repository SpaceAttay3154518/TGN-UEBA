"""Role-relative memory-drift signals, EMA/CUSUM, and update gating."""

from __future__ import annotations

from bisect import bisect_left, insort
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


SIGNALS = (
    "role_displacement",
    "innovation",
    "directional_persistence",
    "trajectory_coherence",
    "event_surprise",
)


def robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    # 1.4826 makes MAD comparable to standard deviation under a normal model.
    return median, max(1.4826 * mad, 1e-6)


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    similarity = F.cosine_similarity(
        left.reshape(1, -1), right.reshape(1, -1), dim=-1, eps=1e-8
    )
    return float((1.0 - similarity).cpu())


def memory_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    method: str,
    scale: torch.Tensor | None = None,
) -> float:
    """Distance between an entity state and its role reference.

    ``normalized_euclidean`` removes arbitrary memory scale.  The diagonal
    Mahalanobis variant uses a robust, peer-derived per-coordinate scale and
    averages across dimensions, avoiding an unstable full covariance inverse.
    """
    if method == "cosine":
        return cosine_distance(left, right)
    difference = left - right
    if method == "normalized_euclidean":
        denominator = max(
            float(torch.linalg.vector_norm(right).cpu()),
            float(torch.linalg.vector_norm(left).cpu()),
            1e-6,
        )
        return float(torch.linalg.vector_norm(difference).cpu()) / denominator
    if method == "diagonal_mahalanobis":
        if scale is None:
            raise ValueError("Diagonal Mahalanobis distance requires peer scale")
        standardized = difference / scale.clamp_min(1e-6)
        return float(torch.sqrt(torch.mean(standardized.square())).cpu())
    raise ValueError(f"Unknown memory distance: {method}")


class RoleReferenceBank:
    """Coordinate-median role references with hierarchical shrinkage."""

    def __init__(
        self,
        node_to_id: dict[str, int],
        profiles: dict[str, dict[str, str]],
        kappa: float,
        minimum_peers: int,
        role_method: str = "hierarchical_ldap",
        estimator: str = "median_shrinkage",
    ) -> None:
        self.node_to_id = node_to_id
        self.profiles = {
            node: profile for node, profile in profiles.items() if node in node_to_id
        }
        self.kappa = float(kappa)
        self.minimum_peers = int(minimum_peers)
        self.role_method = str(role_method)
        self.estimator = str(estimator)
        self.profile_by_id = {
            int(self.node_to_id[name]): profile
            for name, profile in self.profiles.items()
        }
        self.role_by_id = {
            node_id: self._label(profile)
            for node_id, profile in self.profile_by_id.items()
        }
        self.references: dict[int, torch.Tensor] = {}
        self.scales: dict[int, torch.Tensor] = {}

    def _label(self, profile: dict[str, str]) -> str:
        mapping = {
            "global": "__global__",
            "ldap_broad": profile.get("broad_role", "unknown"),
            "ldap_fine": profile.get("fine_role", "unknown|unknown|unknown"),
            "behavior": profile.get("behavior_role", "behavior:unknown"),
            "hybrid": profile.get(
                "hybrid_role",
                f"{profile.get('broad_role', 'unknown')}|behavior:unknown",
            ),
            "hierarchical_ldap": profile.get("fine_role", "unknown|unknown|unknown"),
        }
        if self.role_method not in mapping:
            raise ValueError(f"Unknown role method: {self.role_method}")
        return str(mapping[self.role_method])

    def role(self, node_id: int) -> str:
        return self.role_by_id.get(int(node_id), "entity:unknown")

    @staticmethod
    def _median(memory: torch.Tensor, ids: list[int]) -> torch.Tensor:
        return memory[torch.tensor(ids, dtype=torch.long, device=memory.device)].median(dim=0).values

    def _aggregate(self, memory: torch.Tensor, ids: list[int]) -> torch.Tensor:
        selected = memory[torch.tensor(ids, dtype=torch.long, device=memory.device)]
        if self.estimator in {"median", "median_shrinkage"}:
            return selected.median(dim=0).values
        if self.estimator == "mean":
            return selected.mean(dim=0)
        if self.estimator == "trimmed_mean":
            ordered = selected.sort(dim=0).values
            trim = int(np.floor(0.1 * len(ids)))
            if trim and 2 * trim < len(ids):
                ordered = ordered[trim:-trim]
            return ordered.mean(dim=0)
        raise ValueError(f"Unknown reference estimator: {self.estimator}")

    @staticmethod
    def _scale(memory: torch.Tensor, ids: list[int]) -> torch.Tensor:
        selected = memory[torch.tensor(ids, dtype=torch.long, device=memory.device)]
        center = selected.median(dim=0).values
        mad = (selected - center).abs().median(dim=0).values
        return (1.4826 * mad).clamp_min(1e-6)

    def update(self, memory: torch.Tensor, excluded: set[int] | None = None) -> None:
        excluded = excluded or set()
        grouped: dict[str, list[int]] = defaultdict(list)
        fine: dict[str, list[int]] = defaultdict(list)
        parent: dict[str, list[int]] = defaultdict(list)
        broad: dict[str, list[int]] = defaultdict(list)
        profile_by_id: dict[int, dict[str, str]] = {}
        all_profile_by_id: dict[int, dict[str, str]] = {}
        for name, profile in self.profiles.items():
            node = int(self.node_to_id[name])
            all_profile_by_id[node] = profile
            if node in excluded:
                continue
            profile_by_id[node] = profile
            grouped[self._label(profile)].append(node)
            fine[profile["fine_role"]].append(node)
            parent[profile["parent_role"]].append(node)
            broad[profile["broad_role"]].append(node)
        trusted_ids = sorted(profile_by_id)
        if not trusted_ids:
            raise ValueError("No trusted role peers remain for reference construction")
        if len(trusted_ids) < 2:
            raise ValueError("Role references require at least two trusted entities")

        def group_statistics(ids: list[int]) -> dict:
            """One sort per peer group yields all exact mean/median LOO centers."""
            tensor_ids = torch.tensor(ids, dtype=torch.long, device=memory.device)
            selected = memory[tensor_ids]
            count, dimension = selected.shape
            ordered, order = selected.sort(dim=0)
            ranks = torch.empty_like(order)
            rank_values = torch.arange(count, device=memory.device).reshape(-1, 1)
            ranks.scatter_(0, order, rank_values.expand(count, dimension))
            if self.estimator == "mean":
                full = selected.mean(dim=0)
                loo = (
                    (selected.sum(dim=0).reshape(1, -1) - selected) / (count - 1)
                    if count > 1
                    else full.reshape(1, -1)
                )
            elif self.estimator == "trimmed_mean":
                trim = int(np.floor(0.1 * count))
                central = ordered[trim : count - trim if trim else count]
                full = central.mean(dim=0)
                if count > 1:
                    remaining = count - 1
                    loo_trim = int(np.floor(0.1 * remaining))
                    stop = remaining - loo_trim
                    # Exact coordinate-wise trimmed mean after removing each
                    # row. Prefix sums handle the rank shift without an
                    # entity-by-entity sort.
                    prefix = torch.cat(
                        [
                            torch.zeros(
                                (1, dimension),
                                dtype=selected.dtype,
                                device=selected.device,
                            ),
                            ordered.cumsum(dim=0),
                        ],
                        dim=0,
                    )
                    low = loo_trim + (ranks <= loo_trim).to(torch.long)
                    high = stop + (ranks < stop).to(torch.long)
                    sums = prefix.gather(0, high) - prefix.gather(0, low)
                    removed_inside = (ranks > loo_trim) & (ranks < stop)
                    sums = sums - torch.where(
                        removed_inside, selected, torch.zeros_like(selected)
                    )
                    loo = sums / max(remaining - 2 * loo_trim, 1)
                else:
                    loo = full.reshape(1, -1).expand_as(selected)
            else:
                full = ordered[(count - 1) // 2]
                if count > 1:
                    middle = (count - 2) // 2
                    indices = middle + (ranks <= middle).to(torch.long)
                    loo = ordered.gather(0, indices)
                else:
                    loo = full.reshape(1, -1)
            center = ordered[(count - 1) // 2]
            scale = (1.4826 * (selected - center).abs().median(dim=0).values).clamp_min(1e-6)
            return {
                "ids": ids,
                "index": {node: position for position, node in enumerate(ids)},
                "full": full,
                "loo": loo,
                "scale": scale,
            }

        def select(statistics: dict, node: int) -> tuple[torch.Tensor, int]:
            position = statistics["index"].get(node)
            if position is None:
                return statistics["full"], len(statistics["ids"])
            return statistics["loo"][position], len(statistics["ids"]) - 1

        global_stats = group_statistics(trusted_ids)
        grouped_stats = {
            key: group_statistics(sorted(ids)) for key, ids in grouped.items() if ids
        }
        broad_stats = {
            key: group_statistics(sorted(ids)) for key, ids in broad.items() if ids
        }
        parent_stats = {
            key: group_statistics(sorted(ids)) for key, ids in parent.items() if ids
        }
        fine_stats = {
            key: group_statistics(sorted(ids)) for key, ids in fine.items() if ids
        }

        self.references = {}
        self.scales = {}
        for node, profile in all_profile_by_id.items():
            global_reference, global_count = select(global_stats, node)
            if self.role_method != "hierarchical_ldap":
                statistics = grouped_stats.get(self._label(profile))
                if statistics is None:
                    statistics = global_stats
                role_reference, role_count = select(statistics, node)
                if role_count < self.minimum_peers:
                    statistics = global_stats
                    role_reference, _ = select(statistics, node)
                self.references[node] = role_reference.detach().clone()
                self.scales[node] = statistics["scale"].detach().clone()
                continue

            broad_group = broad_stats.get(profile["broad_role"])
            if broad_group is not None:
                broad_empirical, broad_count = select(broad_group, node)
                broad_rho = broad_count / (broad_count + self.kappa)
                broad_reference = (
                    broad_rho * broad_empirical + (1.0 - broad_rho) * global_reference
                )
            else:
                broad_count = 0
                broad_reference = global_reference
            parent_group = parent_stats.get(profile["parent_role"])
            if parent_group is not None:
                parent_empirical, parent_count = select(parent_group, node)
                parent_rho = parent_count / (parent_count + self.kappa)
                parent_reference = (
                    parent_rho * parent_empirical + (1.0 - parent_rho) * broad_reference
                )
            else:
                parent_count = 0
                parent_reference = broad_reference
            fine_group = fine_stats.get(profile["fine_role"])
            if fine_group is not None:
                fine_empirical, fine_count = select(fine_group, node)
            else:
                fine_count = 0
                fine_empirical = parent_reference
            if fine_count < self.minimum_peers:
                self.references[node] = parent_reference.detach().clone()
                scale_group = parent_group or broad_group or global_stats
                self.scales[node] = scale_group["scale"].detach().clone()
                continue
            rho = fine_count / (fine_count + self.kappa)
            self.references[node] = (
                rho * fine_empirical + (1.0 - rho) * parent_reference
            ).detach().clone()
            self.scales[node] = fine_group["scale"].detach().clone()

    def reference(self, node_id: int) -> torch.Tensor:
        if node_id not in self.references:
            raise KeyError(f"No role reference for node {node_id}")
        return self.references[node_id]

    def reference_scale(self, node_id: int) -> torch.Tensor:
        if node_id not in self.scales:
            raise KeyError(f"No role scale for node {node_id}")
        return self.scales[node_id]


@dataclass
class DriftFeatures:
    node_id: int
    role: str
    timestamp: float
    role_displacement: float
    innovation: float
    directional_persistence: float
    trajectory_coherence: float
    event_surprise: float

    def values(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in SIGNALS}


class _RollingMedian:
    """Exact median of a bounded insertion-ordered window in O(log n)."""

    def __init__(self, window: int = 1000) -> None:
        self.window = int(window)
        self.order: deque[float] = deque()
        self.sorted: list[float] = []

    def __len__(self) -> int:
        return len(self.order)

    def median(self) -> float:
        count = len(self.sorted)
        if not count:
            raise ValueError("Cannot compute the median of an empty window")
        middle = count // 2
        if count % 2:
            return float(self.sorted[middle])
        return float((self.sorted[middle - 1] + self.sorted[middle]) / 2.0)

    def append(self, value: float) -> None:
        value = float(value)
        if len(self.order) == self.window:
            expired = self.order.popleft()
            position = bisect_left(self.sorted, expired)
            if position == len(self.sorted) or self.sorted[position] != expired:
                raise RuntimeError("Rolling-median window lost synchronization")
            self.sorted.pop(position)
        self.order.append(value)
        insort(self.sorted, value)


class FeatureExtractor:
    def __init__(self, distance: str = "cosine") -> None:
        self.distance = str(distance)
        self.role_step_history: dict[str, _RollingMedian] = defaultdict(_RollingMedian)
        self.direction_ema: dict[int, torch.Tensor] = {}
        self.cumulative_delta: dict[int, torch.Tensor] = {}
        self.path_length: dict[int, float] = {}

    def step(
        self,
        node_id: int,
        role: str,
        timestamp: float,
        before: torch.Tensor,
        after: torch.Tensor,
        reference: torch.Tensor,
        event_anomaly: float,
        reference_scale: torch.Tensor | None = None,
        *,
        update_history: bool = True,
        update_role_scale: bool = True,
    ) -> DriftFeatures:
        delta = (after - before).detach()
        magnitude = float(torch.linalg.vector_norm(delta).cpu())
        peer_steps = self.role_step_history[role]
        peer_scale = peer_steps.median() if len(peer_steps) else max(magnitude, 1e-6)
        innovation = magnitude / max(peer_scale, 1e-6)
        previous = self.direction_ema.get(node_id)
        persistence = 0.0
        if previous is not None and magnitude > 0 and float(previous.norm()) > 0:
            persistence = max(
                0.0,
                float(F.cosine_similarity(delta.reshape(1, -1), previous.reshape(1, -1)).cpu()),
            )
        # Trajectory coherence: ||cumulative_delta|| / path_length.
        # Values near 1.0 indicate persistent unidirectional drift; near 0.0
        # indicates random-walk movement that cancels out.
        cum = self.cumulative_delta.get(node_id)
        pl = self.path_length.get(node_id, 0.0)
        if cum is not None:
            new_cum = cum + delta
            new_pl = pl + magnitude
        else:
            new_cum = delta.clone()
            new_pl = magnitude
        if new_pl > 0:
            coherence = float(torch.linalg.vector_norm(new_cum).cpu()) / new_pl
        else:
            coherence = 0.0
        if update_history:
            self.direction_ema[node_id] = (
                delta if previous is None else 0.2 * delta + 0.8 * previous
            ).detach()
            self.cumulative_delta[node_id] = new_cum.detach()
            self.path_length[node_id] = new_pl
        if update_role_scale:
            peer_steps.append(magnitude)
        return DriftFeatures(
            node_id=node_id,
            role=role,
            timestamp=float(timestamp),
            role_displacement=memory_distance(
                after, reference, self.distance, reference_scale
            ),
            innovation=innovation,
            directional_persistence=persistence,
            trajectory_coherence=coherence,
            event_surprise=float(event_anomaly),
        )


@dataclass
class Calibration:
    signal_location: dict[str, dict[str, float]]
    signal_scale: dict[str, dict[str, float]]
    allowance: dict[str, float]
    ema_threshold: dict[str, float]
    cusum_threshold: dict[str, float]
    shewhart_threshold: dict[str, float]
    global_role: str = "__global__"

    def parameters(self, role: str, signal: str) -> tuple[float, float]:
        selected = role if role in self.signal_location else self.global_role
        return self.signal_location[selected][signal], self.signal_scale[selected][signal]


def _combined_signal(
    feature: DriftFeatures,
    location: dict[str, float],
    scale: dict[str, float],
    weights: dict[str, float],
    formula: str = "weighted_mean_positive_z",
) -> float:
    weighted: list[float] = []
    for name in SIGNALS:
        weight = max(float(weights.get(name, 0.0)), 0.0)
        standardized = (getattr(feature, name) - location[name]) / scale[name]
        if weight > 0:
            weighted.append(weight * max(standardized, 0.0))
    if not weighted:
        return 0.0
    if formula == "weighted_mean_positive_z":
        weight_sum = sum(max(float(weights.get(name, 0.0)), 0.0) for name in SIGNALS)
        return float(sum(weighted) / max(weight_sum, 1e-8))
    if formula == "max_positive_z":
        return float(max(weighted))
    if formula == "l2_positive_z":
        return float(np.sqrt(np.mean(np.square(weighted))))
    raise ValueError(f"Unknown evidence formula: {formula}")


def fit_calibration(
    records: Iterable[DriftFeatures],
    weights: dict[str, float],
    ema_alpha: float,
    allowance_quantile: float,
    alert_quantile: float,
    formula: str = "weighted_mean_positive_z",
) -> Calibration:
    rows = list(records)
    if not rows:
        raise ValueError("Cannot calibrate drift monitoring without clean records")
    grouped_rows: dict[str, list[DriftFeatures]] = defaultdict(list)
    for row in rows:
        grouped_rows[row.role].append(row)
    roles = sorted(role for role, role_rows in grouped_rows.items() if len(role_rows) >= 30)
    by_role = {role: grouped_rows[role] for role in roles}
    by_role["__global__"] = rows
    locations: dict[str, dict[str, float]] = {}
    scales: dict[str, dict[str, float]] = {}
    allowances: dict[str, float] = {}
    ema_thresholds: dict[str, float] = {}
    cusum_thresholds: dict[str, float] = {}
    shewhart_thresholds: dict[str, float] = {}

    for role, role_rows in by_role.items():
        locations[role] = {}
        scales[role] = {}
        for signal in SIGNALS:
            locations[role][signal], scales[role][signal] = robust_center_scale(
                np.array([getattr(row, signal) for row in role_rows])
            )
        combined = np.array(
            [
                _combined_signal(
                    row, locations[role], scales[role], weights, formula
                )
                for row in role_rows
            ]
        )
        allowance = float(np.quantile(combined, allowance_quantile))
        allowances[role] = allowance
        shewhart_thresholds[role] = float(np.quantile(combined, alert_quantile))
        ema_by_node: dict[int, float] = defaultdict(float)
        cusum_by_node: dict[int, float] = defaultdict(float)
        ema_values: list[float] = []
        cusum_values: list[float] = []
        for row, value in zip(role_rows, combined):
            ema_by_node[row.node_id] = (
                ema_alpha * float(value) + (1.0 - ema_alpha) * ema_by_node[row.node_id]
            )
            cusum_by_node[row.node_id] = max(
                0.0, cusum_by_node[row.node_id] + float(value) - allowance
            )
            ema_values.append(ema_by_node[row.node_id])
            cusum_values.append(cusum_by_node[row.node_id])
        ema_thresholds[role] = float(np.quantile(ema_values, alert_quantile))
        cusum_thresholds[role] = float(np.quantile(cusum_values, alert_quantile))
    return Calibration(
        locations,
        scales,
        allowances,
        ema_thresholds,
        cusum_thresholds,
        shewhart_thresholds,
    )


@dataclass
class DriftDecision:
    combined_signal: float
    ema: float
    cusum: float
    ema_alert: bool
    cusum_alert: bool
    shewhart_alert: bool


class SequentialMonitor:
    def __init__(
        self,
        calibration: Calibration,
        weights: dict[str, float],
        ema_alpha: float,
        formula: str = "weighted_mean_positive_z",
        gap_reset_seconds: float = 1800.0,
    ) -> None:
        self.calibration = calibration
        self.weights = weights
        self.ema_alpha = float(ema_alpha)
        self.formula = str(formula)
        self.gap_reset_seconds = float(gap_reset_seconds)
        self.ema: dict[int, float] = defaultdict(float)
        self.cusum: dict[int, float] = defaultdict(float)
        self.last_seen: dict[int, float] = {}

    def step(self, feature: DriftFeatures) -> DriftDecision:
        role = feature.role if feature.role in self.calibration.signal_location else "__global__"
        value = _combined_signal(
            feature,
            self.calibration.signal_location[role],
            self.calibration.signal_scale[role],
            self.weights,
            self.formula,
        )
        node = feature.node_id
        # Gap-reset: if the node has been inactive for longer than the
        # configured threshold, reset its sequential state so stale evidence
        # from unrelated episodes does not accumulate indefinitely.
        prev_ts = self.last_seen.get(node)
        if prev_ts is not None and (feature.timestamp - prev_ts) > self.gap_reset_seconds:
            self.ema[node] = 0.0
            self.cusum[node] = 0.0
        self.last_seen[node] = feature.timestamp
        self.ema[node] = self.ema_alpha * value + (1.0 - self.ema_alpha) * self.ema[node]
        self.cusum[node] = max(
            0.0, self.cusum[node] + value - self.calibration.allowance[role]
        )
        return DriftDecision(
            combined_signal=value,
            ema=self.ema[node],
            cusum=self.cusum[node],
            ema_alert=self.ema[node] > self.calibration.ema_threshold[role],
            cusum_alert=self.cusum[node] > self.calibration.cusum_threshold[role],
            shewhart_alert=value > self.calibration.shewhart_threshold[role],
        )

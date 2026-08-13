"""KAIROS-compatible aggregation, calibration, and metric reporting.

KAIROS reports detection at graph level for StreamSpot and at 15-minute-window
level for DARPA provenance data.  Its published AUC is computed from binary
predictions, so this module reports that value as ``auc_hard_kairos`` and also
reports the conventional continuous-score ROC AUC separately.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


KAIROS_REFERENCE = {
    "streamspot": {"TP": 100, "TN": 375, "FP": 0, "FN": 0, "precision": 1.000, "recall": 1.000, "accuracy": 1.000, "AUC": 1.000},
    "darpa_e3_theia": {"TP": 9, "TN": 216, "FP": 2, "FN": 0, "precision": 0.818, "recall": 1.000, "accuracy": 0.991, "AUC": 0.995},
    "darpa_e3_cadets": {"TP": 4, "TN": 174, "FP": 1, "FN": 0, "precision": 0.800, "recall": 1.000, "accuracy": 0.994, "AUC": 0.997},
    "darpa_e3_clearscope": {"TP": 5, "TN": 112, "FP": 2, "FN": 0, "precision": 0.714, "recall": 1.000, "accuracy": 0.983, "AUC": 0.991},
    "darpa_e5_theia": {"TP": 2, "TN": 173, "FP": 1, "FN": 0, "precision": 0.667, "recall": 1.000, "accuracy": 0.994, "AUC": 0.997},
    "darpa_e5_cadets": {"TP": 7, "TN": 238, "FP": 9, "FN": 0, "precision": 0.438, "recall": 1.000, "accuracy": 0.965, "AUC": 0.982},
    "darpa_e5_clearscope": {"TP": 10, "TN": 217, "FP": 5, "FN": 0, "precision": 0.667, "recall": 1.000, "accuracy": 0.978, "AUC": 0.989},
    "optc": {"TP": 22, "TN": 1210, "FP": 16, "FN": 0, "precision": 0.579, "recall": 1.000, "accuracy": 0.987, "AUC": 0.993},
}


def binary_metrics(
    labels: Iterable[bool | int],
    predictions: Iterable[bool | int],
    scores: Iterable[float] | None = None,
) -> dict[str, float | int | None]:
    y_true = np.asarray(list(labels), dtype=np.int8)
    y_pred = np.asarray(list(predictions), dtype=np.int8)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("Labels and predictions must be equal-length vectors")
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    two_classes = len(np.unique(y_true)) == 2
    hard_auc = float(roc_auc_score(y_true, y_pred)) if two_classes else None
    score_auc = None
    if scores is not None and two_classes:
        values = np.asarray(list(scores), dtype=float)
        if values.shape != y_true.shape:
            raise ValueError("Scores must match labels")
        score_auc = float(roc_auc_score(y_true, values))
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "auc_hard_kairos": hard_auc,
        "roc_auc_continuous": score_auc,
        "evaluation_units": int(len(y_true)),
    }


def evaluate_streamspot_graph_scores(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    score_column: str = "anomaly_score",
) -> dict:
    required = {"graph_id", score_column}
    if missing := required - set(validation):
        raise ValueError(f"Validation scores missing columns: {sorted(missing)}")
    if missing := (required | {"attack_label"}) - set(test):
        raise ValueError(f"Test scores missing columns: {sorted(missing)}")
    validation_graphs = validation.groupby("graph_id", sort=True)[score_column].mean()
    if validation_graphs.empty:
        raise ValueError("Validation graph scores are empty")
    threshold = float(validation_graphs.max())
    grouped = test.groupby("graph_id", sort=True).agg(
        anomaly_score=(score_column, "mean"),
        attack_label=("attack_label", "max"),
    )
    predictions = grouped["anomaly_score"] > threshold
    metrics = binary_metrics(
        grouped["attack_label"], predictions, grouped["anomaly_score"]
    )
    return {
        "protocol": "KAIROS StreamSpot graph-level",
        "threshold_source": "maximum mean graph reconstruction loss over validation graphs",
        "threshold": threshold,
        "validation_graphs": int(len(validation_graphs)),
        "metrics": metrics,
        "graph_scores": grouped.assign(prediction=predictions).reset_index(),
    }


@dataclass(frozen=True)
class WindowRecord:
    window_id: Hashable
    sequence_id: Hashable
    anomaly_score: float
    anomalous_nodes: frozenset[str]
    attack_label: bool


def edge_scores_to_windows(
    scores: pd.DataFrame,
    *,
    multiplier: float = 1.5,
    window_minutes: int = 15,
) -> list[WindowRecord]:
    required = {"timestamp", "src", "dst", "anomaly_score", "attack_label"}
    if missing := required - set(scores):
        raise ValueError(f"Edge scores missing columns: {sorted(missing)}")
    frame = scores.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if "window_id" not in frame:
        frame["window_id"] = frame["timestamp"].dt.floor(f"{int(window_minutes)}min")
    if "sequence_id" not in frame:
        frame["sequence_id"] = frame["timestamp"].dt.date.astype(str)
    records: list[WindowRecord] = []
    for (sequence_id, window_id), group in frame.groupby(
        ["sequence_id", "window_id"], sort=True
    ):
        values = group["anomaly_score"].to_numpy(float)
        threshold = float(values.mean() + multiplier * values.std(ddof=0))
        anomalous = group[group["anomaly_score"] > threshold]
        nodes = frozenset(
            str(node) for node in pd.concat([anomalous["src"], anomalous["dst"]]).unique()
        )
        anomaly_score = float(anomalous["anomaly_score"].mean()) if len(anomalous) else 0.0
        records.append(
            WindowRecord(
                window_id=window_id,
                sequence_id=sequence_id,
                anomaly_score=anomaly_score,
                anomalous_nodes=nodes,
                attack_label=bool(group["attack_label"].any()),
            )
        )
    return records


def reference_node_idf(windows: Iterable[WindowRecord]) -> tuple[dict[str, float], int]:
    documents = list(windows)
    counts: Counter[str] = Counter()
    for window in documents:
        counts.update(window.anomalous_nodes)
    total = len(documents)
    if total == 0:
        raise ValueError("Cannot compute IDF without reference windows")
    return {
        node: math.log(total / (count + 1)) for node, count in counts.items()
    }, total


def _rare_shared_nodes(
    left: WindowRecord,
    right: WindowRecord,
    node_idf: dict[str, float],
    reference_windows: int,
) -> set[str]:
    rarity_threshold = math.log(reference_windows * 0.9)
    maximum_idf = math.log(reference_windows)
    return {
        node
        for node in left.anomalous_nodes & right.anomalous_nodes
        if node_idf.get(node, maximum_idf) > rarity_threshold
    }


def construct_queues(
    windows: Iterable[WindowRecord],
    node_idf: dict[str, float],
    reference_windows: int,
) -> list[list[WindowRecord]]:
    queues: list[list[WindowRecord]] = []
    for window in sorted(windows, key=lambda item: (str(item.sequence_id), str(item.window_id))):
        correlated = None
        for queue in queues:
            if queue[-1].sequence_id != window.sequence_id:
                continue
            if any(
                _rare_shared_nodes(window, prior, node_idf, reference_windows)
                for prior in queue
            ):
                correlated = queue
                break
        if correlated is None:
            queues.append([window])
        else:
            correlated.append(window)
    return queues


def queue_score(queue: Iterable[WindowRecord]) -> float:
    # The released KAIROS artifact adds one to every window score before taking
    # the product, although the paper's displayed equation omits that detail.
    return float(np.prod([1.0 + window.anomaly_score for window in queue]))


def evaluate_darpa_windows(
    reference: Iterable[WindowRecord],
    validation: Iterable[WindowRecord],
    test: Iterable[WindowRecord],
) -> dict:
    reference_list = list(reference)
    validation_list = list(validation)
    test_list = list(test)
    node_idf, reference_count = reference_node_idf(reference_list)
    validation_queues = construct_queues(validation_list, node_idf, reference_count)
    if not validation_queues:
        raise ValueError("Validation windows produced no queues")
    beta = max(queue_score(queue) for queue in validation_queues)
    test_queues = construct_queues(test_list, node_idf, reference_count)
    score_by_window: dict[tuple[Hashable, Hashable], float] = {}
    for queue in test_queues:
        score = queue_score(queue)
        for window in queue:
            key = (window.sequence_id, window.window_id)
            score_by_window[key] = max(score_by_window.get(key, 0.0), score)
    ordered_scores = [
        score_by_window[(window.sequence_id, window.window_id)] for window in test_list
    ]
    predictions = [value > beta for value in ordered_scores]
    metrics = binary_metrics(
        [window.attack_label for window in test_list], predictions, ordered_scores
    )
    return {
        "protocol": "KAIROS DARPA 15-minute-window",
        "edge_threshold": "per-window mean + 1.5 population standard deviations",
        "rarity": "IDF(v)=ln(N/(Nv+1)); rare when IDF>ln(0.9N)",
        "queue_score": "product of (1 + mean above-threshold edge error)",
        "beta_source": "maximum validation queue score",
        "beta": float(beta),
        "reference_windows": reference_count,
        "validation_queues": len(validation_queues),
        "test_queues": len(test_queues),
        "metrics": metrics,
    }


def compare_with_kairos(dataset_id: str, metrics: dict) -> dict:
    if dataset_id not in KAIROS_REFERENCE:
        raise KeyError(f"No KAIROS reference for {dataset_id}")
    reference = KAIROS_REFERENCE[dataset_id]
    mapping = {
        "TP": "TP", "TN": "TN", "FP": "FP", "FN": "FN",
        "precision": "precision", "recall": "recall", "accuracy": "accuracy",
        "AUC": "auc_hard_kairos",
    }
    return {
        "dataset_id": dataset_id,
        "reference": reference,
        "observed": {key: metrics.get(target) for key, target in mapping.items()},
        "delta": {
            key: None if metrics.get(target) is None else float(metrics[target] - value)
            for key, value in reference.items()
            for target in [mapping[key]]
        },
    }


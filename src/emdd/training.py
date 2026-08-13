"""Resource-bounded, resumable streaming multi-seed TGN training."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .io import PreparedDataset, build_model, resolve_config_path, save_checkpoint
from .model import TGNDetector, iter_batches, seed_everything
from .replay import sample_negatives, tensors_from_frame


def _frames(
    events: pd.DataFrame | PreparedDataset,
    split: str,
    graph_ids: set[int] | None = None,
) -> Iterator[pd.DataFrame]:
    if isinstance(events, PreparedDataset):
        for frame in events.iter_split(split):
            if graph_ids is not None and (
                "graph_id" not in frame
                or not set(frame["graph_id"].astype(int).unique()) <= graph_ids
            ):
                continue
            yield frame
        return
    selected = events[events["split"] == split].copy()
    if graph_ids is not None:
        selected = selected[selected["graph_id"].astype(int).isin(graph_ids)]
    yield selected


def _destination_pools(metadata: dict) -> dict[int, np.ndarray]:
    negative_sampling = metadata.get("negative_sampling", {})
    if negative_sampling.get("strategy") == "global_uniform_kairos_compatibility":
        lower = int(negative_sampling["minimum_node_id"])
        upper = int(negative_sampling["maximum_node_id"])
        global_pool = np.arange(lower, upper + 1, dtype=np.int64)
        return {-1: global_pool, **{
            int(type_id): global_pool for type_id in metadata["dst_type_to_id"].values()
        }}
    declared = metadata.get("training_destination_ids_by_type")
    if declared:
        pools = {
            int(type_id): np.asarray(values, dtype=np.int64)
            for type_id, values in declared.items()
        }
        pools[-1] = np.asarray(
            sorted({int(item) for values in pools.values() for item in values}),
            dtype=np.int64,
        )
        return pools
    reverse_type = {value: key for key, value in metadata["dst_type_to_id"].items()}
    prefix = {
        "device": "device:",
        "domain": "domain:",
        "fileext": "fileext:",
        "maildomain": "maildomain:",
        "pc": "pc:",
    }
    pools: dict[int, np.ndarray] = {}
    for type_id, name in reverse_type.items():
        ids = [
            int(node_id)
            for node, node_id in metadata["node_to_id"].items()
            if node.startswith(prefix.get(name, f"{name}:"))
        ]
        pools[int(type_id)] = np.asarray(sorted(ids), dtype=np.int64)
    pools[-1] = np.asarray(
        sorted({int(item) for values in pools.values() for item in values}),
        dtype=np.int64,
    )
    return pools


def _perturbed_loss(
    detector: TGNDetector,
    batch,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bounded stochastic memory perturbation used only by secondary baselines."""
    node_parts = [batch.src, batch.dst]
    if batch.neg_dst is not None:
        node_parts.append(batch.neg_dst)
    nodes = torch.cat(node_parts).unique()
    original = detector.memory.memory[nodes].detach().clone()
    clean = detector.training_loss(batch)
    with torch.no_grad():
        detector.memory.memory[nodes].add_(
            torch.empty_like(original).uniform_(-epsilon, epsilon)
        )
    perturbed = detector.training_loss(batch)
    with torch.no_grad():
        detector.memory.memory[nodes].copy_(original)
    sensitivity = (perturbed - clean).abs() / max(epsilon, 1e-8)
    return perturbed, sensitivity


def _autocast_context(device: torch.device, model_config: dict):
    enabled = device.type == "cuda" and bool(model_config.get("mixed_precision", False))
    if not enabled:
        return nullcontext()
    name = str(model_config.get("mixed_precision_dtype", "bfloat16"))
    dtype = torch.bfloat16 if name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def _validate(
    detector: TGNDetector,
    frames: Iterable[pd.DataFrame],
    metadata: dict,
    pools: dict[int, np.ndarray],
    seed: int,
    batch_size: int,
    *,
    reset_per_partition: bool = False,
    model_config: dict | None = None,
) -> dict:
    branch = detector.fork()
    total = 0
    reconstruction_sum = 0.0
    positive_probabilities: list[np.ndarray] = []
    negative_probabilities: list[np.ndarray] = []
    predicted_types: list[np.ndarray] = []
    observed_types: list[np.ndarray] = []
    for frame_index, frame in enumerate(frames):
        if reset_per_partition:
            branch.reset_temporal_state()
        src, dst, timestamp, message, edge_type = tensors_from_frame(frame, metadata)
        negatives = (
            sample_negatives(frame, pools, seed + frame_index)
            if branch.link_loss_weight > 0
            else None
        )
        for batch in iter_batches(
            src, dst, timestamp, message, batch_size, negatives, edge_type
        ):
            batch = batch.to(branch.device_ref)
            with _autocast_context(branch.device_ref, model_config or {}):
                positive, negative, type_logits = branch.reconstruction_outputs(batch)
            assert type_logits is not None and batch.edge_type is not None
            type_error = torch.nn.functional.cross_entropy(
                type_logits, batch.edge_type, reduction="none"
            )
            anomaly = (
                branch.edge_type_loss_weight * type_error
                + branch.link_loss_weight
                * torch.nn.functional.softplus(-positive)
            )
            reconstruction_sum += float(anomaly.sum().cpu())
            total += batch.size
            positive_probabilities.append(
                torch.sigmoid(positive).float().cpu().numpy()
            )
            if negative is not None:
                negative_probabilities.append(
                    torch.sigmoid(negative).float().cpu().numpy()
                )
            predicted_types.append(type_logits.argmax(dim=-1).cpu().numpy())
            observed_types.append(batch.edge_type.cpu().numpy())
            branch.observe(batch)
        branch.compact_history()
    positive = np.concatenate(positive_probabilities)
    negative = (
        np.concatenate(negative_probabilities)
        if negative_probabilities
        else np.asarray([], dtype=float)
    )
    observed = np.concatenate(observed_types)
    predicted = np.concatenate(predicted_types)
    return {
        "events": total,
        "mean_reconstruction_error": reconstruction_sum / max(total, 1),
        "link_ap": (
            float("nan")
            if not len(negative)
            else float(
                average_precision_score(
                    np.r_[np.ones(len(positive)), np.zeros(len(negative))],
                    np.r_[positive, negative],
                )
            )
        ),
        "link_auc": (
            float("nan")
            if not len(negative)
            else float(
                roc_auc_score(
                    np.r_[np.ones(len(positive)), np.zeros(len(negative))],
                    np.r_[positive, negative],
                )
            )
        ),
        "edge_type_accuracy": float(np.mean(observed == predicted)),
        "edge_type_macro_f1": float(
            f1_score(observed, predicted, average="macro", zero_division=0)
        ),
    }


def train_one_seed(
    seed: int,
    events: pd.DataFrame | PreparedDataset,
    metadata: dict,
    config: dict,
    device: torch.device,
) -> dict:
    seed_everything(seed)
    model_config = config["model"]
    pools = _destination_pools(metadata)
    detector = build_model(config, metadata, device)
    optimizer = torch.optim.Adam(
        detector.parameters(),
        lr=float(model_config["learning_rate"]),
        eps=float(model_config.get("optimizer_epsilon", 1e-8)),
        weight_decay=float(model_config["weight_decay"]),
    )
    use_fp16_scaler = (
        device.type == "cuda"
        and bool(model_config.get("mixed_precision", False))
        and str(model_config.get("mixed_precision_dtype", "bfloat16")) == "float16"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    throttle_seconds = max(float(model_config.get("batch_throttle_ms", 0.0)), 0.0) / 1000.0
    best_error = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    checkpoint_dir = resolve_config_path(config, config["paths"]["checkpoints"])
    checkpoint_path = checkpoint_dir / f"tgn_seed_{seed}.pt"
    started = time.perf_counter()
    checkpoint_selection = str(model_config.get("checkpoint_selection", "best_validation"))
    if checkpoint_selection not in {"best_validation", "final_epoch"}:
        raise ValueError(f"Unknown checkpoint selection: {checkpoint_selection}")
    model_selection_graphs = metadata.get("model_selection_graphs")
    selected_validation_graphs = (
        None if model_selection_graphs is None else {int(value) for value in model_selection_graphs}
    )

    for epoch in range(1, int(model_config["epochs"]) + 1):
        detector.train()
        detector.reset_temporal_state()
        total = 0.0
        count = 0
        frame_count = 0
        for frame_index, frame in enumerate(_frames(events, "train")):
            src, dst, timestamp, message, edge_type = tensors_from_frame(
                frame, metadata
            )
            negatives = (
                sample_negatives(frame, pools, seed + epoch * 1000 + frame_index)
                if detector.link_loss_weight > 0
                else None
            )
            for batch in iter_batches(
                src,
                dst,
                timestamp,
                message,
                int(model_config["batch_size"]),
                negatives,
                edge_type,
            ):
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                mode = str(model_config.get("baseline", "plain"))
                with _autocast_context(device, model_config):
                    if mode in {"memory_noise_training", "lipschitz"}:
                        robust_loss, sensitivity = _perturbed_loss(
                            detector,
                            batch,
                            float(model_config["memory_noise_epsilon"]),
                        )
                        loss = robust_loss
                        if mode == "lipschitz":
                            loss = loss + float(model_config["lipschitz_weight"]) * sensitivity
                    else:
                        loss = detector.training_loss(batch)
                detector.observe(batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    detector.parameters(), float(model_config["gradient_clip"])
                )
                scaler.step(optimizer)
                scaler.update()
                detector.memory.detach()
                total += float(loss.detach().cpu()) * batch.size
                count += batch.size
                if throttle_seconds:
                    time.sleep(throttle_seconds)
            # The neighbor sampler references at most K events per node. Drop
            # older messages so full-corpus training remains bounded in memory.
            detector.compact_history()
            frame_count += 1
            progress_days = int(model_config.get("progress_every_partitions", 0))
            if progress_days > 0 and frame_count % progress_days == 0:
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "epoch": epoch,
                            "train_partitions_complete": frame_count,
                            "train_events_complete": count,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )

        detector.eval()
        validation = _validate(
            detector,
            _frames(events, "validation", selected_validation_graphs),
            metadata,
            pools,
            seed + 10_000 + epoch * 100,
            int(model_config["batch_size"]),
            reset_per_partition=bool(
                metadata.get("reset_temporal_state", {}).get("validation", False)
            ),
            model_config=model_config,
        )
        validation_error = float(validation["mean_reconstruction_error"])
        record = {
            "epoch": epoch,
            "train_loss": total / max(count, 1),
            "train_events": count,
            "train_day_partitions": frame_count,
            "validation_reconstruction_error": validation_error,
            "validation_link_ap": float(validation.get("link_ap", float("nan"))),
            "validation_link_auc": float(validation.get("link_auc", float("nan"))),
            "validation_edge_type_accuracy": float(
                validation.get("edge_type_accuracy", float("nan"))
            ),
            "validation_edge_type_macro_f1": float(
                validation.get("edge_type_macro_f1", float("nan"))
            ),
        }
        history.append(record)
        print(json.dumps({"seed": seed, **record}), flush=True)
        minimum_delta = max(
            float(model_config.get("early_stopping_min_delta", 0.0)), 0.0
        )
        improved = validation_error < best_error - minimum_delta
        should_checkpoint = (
            improved
            if checkpoint_selection == "best_validation"
            else epoch == int(model_config["epochs"])
        )
        if improved:
            best_error = validation_error
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if should_checkpoint:
            if checkpoint_selection == "final_epoch":
                best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                detector,
                config,
                metadata,
                seed,
                {"best_epoch": best_epoch, "history": history},
            )
        if (
            checkpoint_selection == "best_validation"
            and stale >= int(model_config["early_stopping_patience"])
        ):
            break

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_reconstruction_error": best_error,
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }


def train_all(
    events: pd.DataFrame | PreparedDataset,
    metadata: dict,
    config: dict,
    seeds: list[int] | None = None,
) -> list[dict]:
    requested_device = str(config.get("model", {}).get("device", "auto"))
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configuration requests CUDA, but CUDA is unavailable")
    else:
        device = torch.device(requested_device)
    if device.type == "cuda" and "gpu_memory_fraction" in config.get("model", {}):
        torch.cuda.set_per_process_memory_fraction(
            float(config["model"]["gpu_memory_fraction"]),
            device=torch.cuda.current_device(),
        )
    selected = seeds or [int(seed) for seed in config["model_seeds"]]
    summaries = [
        train_one_seed(seed, events, metadata, config, device) for seed in selected
    ]
    output = resolve_config_path(config, config["paths"]["checkpoints"])
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "training_summary.json"
    existing: list[dict] = []
    if summary_path.exists():
        loaded = json.loads(summary_path.read_text())
        if isinstance(loaded, list):
            existing = loaded
    merged = {int(item["seed"]): item for item in existing}
    merged.update({int(item["seed"]): item for item in summaries})
    summary_path.write_text(
        json.dumps([merged[seed] for seed in sorted(merged)], indent=2)
    )
    return summaries

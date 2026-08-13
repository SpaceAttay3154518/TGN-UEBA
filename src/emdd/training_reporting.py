"""Validate completed training and persist standalone training figures."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .io import canonical_config_hash, resolve_config_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_figure(figure: plt.Figure, output: Path) -> None:
    figure.tight_layout()
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def aggregate_training(config: dict) -> dict:
    """Validate all configured seeds and write a durable training-only report."""

    checkpoint_root = resolve_config_path(config, config["paths"]["checkpoints"])
    artifact_root = resolve_config_path(config, config["paths"]["artifacts"])
    output = artifact_root / "training"

    summary_path = checkpoint_root / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Training summary does not exist: {summary_path}")
    summaries = json.loads(summary_path.read_text())
    if not isinstance(summaries, list):
        raise ValueError("training_summary.json must contain a list")

    expected_seeds = [int(seed) for seed in config["model_seeds"]]
    by_seed = {int(item["seed"]): item for item in summaries}
    if len(by_seed) != len(summaries):
        raise ValueError("training_summary.json contains duplicate seed entries")
    missing = [seed for seed in expected_seeds if seed not in by_seed]
    if missing:
        raise ValueError(f"Training summary is missing configured seeds: {missing}")

    epoch_rows: list[dict] = []
    selected_rows: list[dict] = []
    checkpoints: list[dict] = []
    for seed in expected_seeds:
        summary = by_seed[seed]
        history = summary.get("history")
        if not isinstance(history, list) or not history:
            raise ValueError(f"Seed {seed} has no epoch history")
        best_epoch = int(summary["best_epoch"])
        selected = next(
            (record for record in history if int(record["epoch"]) == best_epoch),
            None,
        )
        if selected is None:
            raise ValueError(f"Seed {seed} best epoch {best_epoch} is absent from history")
        checkpoint = checkpoint_root / f"tgn_seed_{seed}.pt"
        if not checkpoint.exists() or checkpoint.stat().st_size < 1_000_000:
            raise ValueError(f"Seed {seed} checkpoint is missing or incomplete: {checkpoint}")

        for record in history:
            epoch_rows.append(
                {
                    "seed": seed,
                    "selected": int(record["epoch"]) == best_epoch,
                    **record,
                }
            )
        selected_rows.append(
            {
                "seed": seed,
                "selected_epoch": best_epoch,
                "epochs_completed": len(history),
                "train_loss": selected["train_loss"],
                "validation_reconstruction_error": selected[
                    "validation_reconstruction_error"
                ],
                "validation_edge_type_accuracy": selected[
                    "validation_edge_type_accuracy"
                ],
                "validation_edge_type_macro_f1": selected[
                    "validation_edge_type_macro_f1"
                ],
                "checkpoint": str(checkpoint),
            }
        )
        checkpoints.append(
            {
                "seed": seed,
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "sha256": _sha256(checkpoint),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    epoch_frame = pd.DataFrame(epoch_rows).sort_values(["seed", "epoch"])
    selected_frame = pd.DataFrame(selected_rows).sort_values("seed")
    epoch_csv = output / "training_epoch_metrics.csv"
    selected_csv = output / "selected_checkpoint_metrics.csv"
    epoch_frame.to_csv(epoch_csv, index=False)
    selected_frame.to_csv(selected_csv, index=False)

    durable_results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_name": config["study_name"],
        "config_sha256": canonical_config_hash(config),
        "training_summary_sha256": _sha256(summary_path),
        "expected_seeds": expected_seeds,
        "all_seeds_complete": True,
        "selected_checkpoints": selected_rows,
        "checkpoints": checkpoints,
        "full_training_history": [by_seed[seed] for seed in expected_seeds],
    }
    results_json = output / "training_results.json"
    results_json.write_text(
        json.dumps(_json_safe(durable_results), indent=2, allow_nan=False)
    )

    colors = ["#286983", "#6b8e23", "#d29f42", "#b23a48", "#6b4c9a"]
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    for color, seed in zip(colors, expected_seeds, strict=True):
        rows = epoch_frame[epoch_frame["seed"] == seed]
        axes[0, 0].plot(
            rows["epoch"],
            rows["validation_reconstruction_error"],
            marker="o",
            color=color,
            label=f"seed {seed}",
        )
        axes[0, 1].plot(
            rows["epoch"], rows["train_loss"], marker="o", color=color
        )
    axes[0, 0].set_title("Validation reconstruction cross-entropy")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Cross-entropy")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    axes[0, 1].set_title("Training cross-entropy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Cross-entropy")

    x = list(range(len(expected_seeds)))
    axes[1, 0].bar(
        x,
        selected_frame["validation_reconstruction_error"],
        color=colors,
    )
    axes[1, 0].set_title("Selected-checkpoint validation loss")
    axes[1, 0].set_ylabel("Cross-entropy")
    axes[1, 1].bar(
        x,
        selected_frame["validation_edge_type_macro_f1"],
        color=colors,
    )
    axes[1, 1].set_title("Selected-checkpoint validation macro-F1")
    axes[1, 1].set_ylabel("Macro-F1")
    for axis in axes[1, :]:
        axis.set_xticks(x, [str(seed) for seed in expected_seeds])
        axis.set_xlabel("Seed")
    figure.suptitle(
        "Adapted KAIROS training summary, CERT r4.2",
        fontsize=14,
        fontweight="bold",
    )
    graph_base = output / "training_summary"
    _save_figure(figure, graph_base)

    manifest = {
        "status": "complete",
        "seeds": expected_seeds,
        "results_json": str(results_json),
        "epoch_metrics_csv": str(epoch_csv),
        "selected_metrics_csv": str(selected_csv),
        "graph_png": str(graph_base.with_suffix(".png")),
        "graph_pdf": str(graph_base.with_suffix(".pdf")),
        "checkpoint_hashes": checkpoints,
    }
    manifest_path = output / "training_report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {**manifest, "manifest": str(manifest_path)}

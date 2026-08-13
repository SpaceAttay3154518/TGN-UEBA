#!/usr/bin/env python3
"""Aggregate multi-target study results across seeds and scenarios.

Reads per-seed ``case_level_results.csv`` from each seed directory under
``artifacts/cert_r42_longitudinal_kairos/`` and (when available) the
``multi_target_aggregate/`` directory.  Produces:

  1. Per-scenario detection-rate summary (plain KAIROS vs combined system).
  2. Per-case detail for S3 insiders across all seeds.
  3. Cross-scenario generalisation summary.
  4. Aggregate statistics: mean recovery R, detection rates, FP rates.

Outputs are written as CSV and Markdown tables to a configurable directory.

Usage:
    python aggregate_multi_target.py                        # defaults
    python aggregate_multi_target.py --output-dir ./reports
    python aggregate_multi_target.py --artifacts-root /path/to/artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "cert_r42_longitudinal_kairos"
SEEDS = [17, 29, 43, 59, 71]

# Methods to compare: baseline KAIROS vs best combined detector
BASELINE_METHOD = "plain_adapted_kairos_q99"
COMBINED_PATTERN = "primary|"  # prefix for combined-system methods

# S3 case IDs (dev + test)
S3_DEV_CASES = [
    "s3:CSC0217", "s3:GTD0219", "s3:JGT0221", "s3:JTM0223", "s3:BBS0039",
]
S3_TEST_CASES = [
    "s3:BSS0369", "s3:CCA0046", "s3:MPM0220", "s3:MSO0222", "s3:JLM0364",
]
S3_ALL_CASES = S3_DEV_CASES + S3_TEST_CASES


# ===================================================================
# Data loading
# ===================================================================

def load_case_level_results(artifacts_root: Path, seeds: list[int]) -> pd.DataFrame:
    """Load and concatenate case_level_results.csv from each seed directory."""
    frames = []
    for seed in seeds:
        csv_path = artifacts_root / f"seed_{seed}" / "case_level_results.csv"
        if not csv_path.exists():
            print(f"  Warning: missing {csv_path}", file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)
        # Ensure seed column is present and correct
        df["seed"] = seed
        frames.append(df)
    if not frames:
        print("Error: no case_level_results.csv files found.", file=sys.stderr)
        sys.exit(1)
    combined = pd.concat(frames, ignore_index=True)
    # Normalise boolean columns
    for col in ("attack_window_alert", "matched_control_alert"):
        if col in combined.columns:
            combined[col] = combined[col].map(
                {True: True, False: False, "True": True, "False": False}
            ).astype(bool)
    return combined


def load_aggregate_summary(artifacts_root: Path) -> Optional[dict]:
    """Load the aggregate summary JSON if present."""
    path = artifacts_root / "aggregate" / "aggregate_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_scenario_results(artifacts_root: Path) -> Optional[pd.DataFrame]:
    """Load pre-computed scenario_results.csv from aggregate/."""
    path = artifacts_root / "aggregate" / "scenario_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_defense_ablation(artifacts_root: Path) -> Optional[pd.DataFrame]:
    """Load defense_ablation_aggregate.csv from aggregate/."""
    path = artifacts_root / "aggregate" / "defense_ablation_aggregate.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_multi_target_aggregate(artifacts_root: Path) -> Optional[dict]:
    """Load results from multi_target_aggregate/ directory if it exists."""
    agg_dir = artifacts_root / "multi_target_aggregate"
    if not agg_dir.exists():
        return None
    result: dict = {}
    for f in agg_dir.iterdir():
        if f.suffix == ".json":
            with open(f) as fh:
                result[f.stem] = json.load(fh)
        elif f.suffix == ".csv":
            result[f.stem] = pd.read_csv(f)
    return result if result else None


# ===================================================================
# Analysis
# ===================================================================

def compute_detection_rates(
    df: pd.DataFrame, method: str, group_col: str = "scenario"
) -> pd.DataFrame:
    """Compute per-group detection rate for a given method."""
    subset = df[df["method"] == method].copy()
    if subset.empty:
        return pd.DataFrame()
    grouped = subset.groupby(group_col).agg(
        n_cases=("attack_window_alert", "size"),
        n_detected=("attack_window_alert", "sum"),
        n_control_alert=("matched_control_alert", "sum"),
    ).reset_index()
    grouped["detection_rate"] = grouped["n_detected"] / grouped["n_cases"]
    grouped["false_positive_rate"] = grouped["n_control_alert"] / grouped["n_cases"]
    grouped["net_detection_rate"] = grouped["detection_rate"] - grouped["false_positive_rate"]
    grouped["method"] = method
    return grouped


def best_combined_method(df: pd.DataFrame) -> str:
    """Select the best combined-system method by net detection rate on S3."""
    combined = df[
        df["method"].str.startswith(COMBINED_PATTERN)
        & (df["scenario"] == 3)
    ]
    if combined.empty:
        # Fall back to any shewhart method
        combined = df[
            df["method"].str.contains("shewhart")
            & (df["scenario"] == 3)
        ]
    if combined.empty:
        return BASELINE_METHOD

    rates = combined.groupby("method").apply(
        lambda g: g["attack_window_alert"].mean() - g["matched_control_alert"].mean(),
        include_groups=False,
    )
    return str(rates.idxmax())


def per_scenario_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Table (a): per-scenario detection rates for baseline vs combined."""
    best_combined = best_combined_method(df)
    baseline_rates = compute_detection_rates(df, BASELINE_METHOD)
    combined_rates = compute_detection_rates(df, best_combined)
    summary = pd.concat([baseline_rates, combined_rates], ignore_index=True)
    cols = ["scenario", "method", "n_cases", "detection_rate",
            "false_positive_rate", "net_detection_rate"]
    return summary[[c for c in cols if c in summary.columns]]


def per_case_s3_detail(df: pd.DataFrame) -> pd.DataFrame:
    """Table (b): per-case detail for S3 insiders across all seeds."""
    s3 = df[df["scenario"] == 3].copy()
    best_combined = best_combined_method(df)

    rows = []
    for case_id in sorted(s3["case_id"].unique()):
        case_df = s3[s3["case_id"] == case_id]
        target_user = case_df["target_user"].iloc[0] if "target_user" in case_df.columns else ""

        # Baseline stats
        bl = case_df[case_df["method"] == BASELINE_METHOD]
        bl_detected = int(bl["attack_window_alert"].sum()) if not bl.empty else 0
        bl_total = len(bl)
        bl_fp = int(bl["matched_control_alert"].sum()) if not bl.empty else 0

        # Combined stats
        cb = case_df[case_df["method"] == best_combined]
        cb_detected = int(cb["attack_window_alert"].sum()) if not cb.empty else 0
        cb_total = len(cb)
        cb_fp = int(cb["matched_control_alert"].sum()) if not cb.empty else 0

        rows.append({
            "case_id": case_id,
            "target_user": target_user,
            "split": "dev" if case_id in S3_DEV_CASES else "test",
            "seeds_run": bl_total,
            "baseline_detected": bl_detected,
            "baseline_rate": bl_detected / bl_total if bl_total else np.nan,
            "baseline_fp": bl_fp,
            "combined_method": best_combined,
            "combined_detected": cb_detected,
            "combined_rate": cb_detected / cb_total if cb_total else np.nan,
            "combined_fp": cb_fp,
        })
    return pd.DataFrame(rows)


def cross_scenario_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Table (c): cross-scenario generalisation summary."""
    best_combined = best_combined_method(df)
    methods_to_check = [BASELINE_METHOD, best_combined]

    rows = []
    for method in methods_to_check:
        for scenario in sorted(df["scenario"].unique()):
            sub = df[(df["method"] == method) & (df["scenario"] == scenario)]
            if sub.empty:
                continue
            rows.append({
                "scenario": int(scenario),
                "method": method,
                "n_observations": len(sub),
                "n_unique_cases": sub["case_id"].nunique(),
                "n_seeds": sub["seed"].nunique(),
                "detection_rate": sub["attack_window_alert"].mean(),
                "false_positive_rate": sub["matched_control_alert"].mean(),
                "net_detection_rate": (
                    sub["attack_window_alert"].mean()
                    - sub["matched_control_alert"].mean()
                ),
            })
    return pd.DataFrame(rows)


def aggregate_statistics(df: pd.DataFrame, summary_json: Optional[dict]) -> dict:
    """Compute top-level aggregate statistics."""
    best_combined = best_combined_method(df)
    stats: dict = {"best_combined_method": best_combined}

    for label, method in [("baseline", BASELINE_METHOD), ("combined", best_combined)]:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        det = sub["attack_window_alert"].values.astype(float)
        fp = sub["matched_control_alert"].values.astype(float)
        net = det - fp
        stats[f"{label}_mean_detection_rate"] = float(np.mean(det))
        stats[f"{label}_std_detection_rate"] = float(np.std(det, ddof=1)) if len(det) > 1 else 0.0
        stats[f"{label}_mean_false_positive_rate"] = float(np.mean(fp))
        stats[f"{label}_mean_net_detection_rate"] = float(np.mean(net))

        # Per-scenario breakdown
        for sc in sorted(sub["scenario"].unique()):
            sc_sub = sub[sub["scenario"] == sc]
            sc_det = sc_sub["attack_window_alert"].mean()
            sc_fp = sc_sub["matched_control_alert"].mean()
            stats[f"{label}_s{int(sc)}_detection_rate"] = float(sc_det)
            stats[f"{label}_s{int(sc)}_false_positive_rate"] = float(sc_fp)

    # Pull recovery R from aggregate summary if available
    if summary_json and "baseline_incidents" in summary_json:
        bi = summary_json["baseline_incidents"]
        if "net_detection_rate" in bi:
            r = bi["net_detection_rate"]
            stats["mean_recovery_R"] = r.get("mean")
            stats["recovery_R_ci_lower"] = r.get("lower")
            stats["recovery_R_ci_upper"] = r.get("upper")

    return stats


# ===================================================================
# Output formatting
# ===================================================================

def df_to_markdown(df: pd.DataFrame, title: str = "") -> str:
    """Convert a DataFrame to a Markdown table with optional title."""
    lines = []
    if title:
        lines.append(f"## {title}\n")

    # Format float columns to 4 decimal places
    fmt_df = df.copy()
    for col in fmt_df.select_dtypes(include=[np.floating]).columns:
        fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    header = "| " + " | ".join(str(c) for c in fmt_df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in fmt_df.columns) + " |"
    lines.append(header)
    lines.append(separator)
    for _, row in fmt_df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    lines.append("")
    return "\n".join(lines)


def stats_to_markdown(stats: dict) -> str:
    """Format aggregate statistics as Markdown."""
    lines = ["## Aggregate Statistics\n"]
    for key, val in sorted(stats.items()):
        if isinstance(val, float):
            lines.append(f"- **{key}**: {val:.4f}")
        else:
            lines.append(f"- **{key}**: {val}")
    lines.append("")
    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-target study results across seeds and scenarios."
    )
    parser.add_argument(
        "--artifacts-root", type=Path, default=ARTIFACTS_ROOT,
        help="Root of the cert_r42_longitudinal_kairos artifacts tree.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for output files.  Defaults to <artifacts-root>/multi_target_aggregate/.",
    )
    parser.add_argument(
        "--seeds", type=str, default=",".join(str(s) for s in SEEDS),
        help="Comma-separated seed list (default: 17,29,43,59,71).",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    artifacts_root = args.artifacts_root.resolve()
    output_dir = (args.output_dir or artifacts_root / "multi_target_aggregate").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Artifacts root : {artifacts_root}")
    print(f"Seeds          : {seeds}")
    print(f"Output dir     : {output_dir}")
    print()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading case-level results...")
    df = load_case_level_results(artifacts_root, seeds)
    print(f"  Loaded {len(df):,} rows across {df['seed'].nunique()} seeds, "
          f"{df['case_id'].nunique()} cases, {df['method'].nunique()} methods.")

    summary_json = load_aggregate_summary(artifacts_root)
    if summary_json:
        print("  Loaded aggregate_summary.json.")
    else:
        print("  Warning: aggregate_summary.json not found; some stats may be incomplete.")

    multi_target_data = load_multi_target_aggregate(artifacts_root)
    if multi_target_data:
        print(f"  Loaded multi_target_aggregate/ ({len(multi_target_data)} files).")

    scenario_results = load_scenario_results(artifacts_root)
    defense_ablation = load_defense_ablation(artifacts_root)

    print()

    # ------------------------------------------------------------------
    # (a) Per-scenario detection rates
    # ------------------------------------------------------------------
    print("Computing per-scenario detection rates...")
    scenario_table = per_scenario_summary(df)
    scenario_table.to_csv(output_dir / "per_scenario_detection_rates.csv", index=False)
    print("  Saved per_scenario_detection_rates.csv")

    # ------------------------------------------------------------------
    # (b) Per-case S3 detail
    # ------------------------------------------------------------------
    print("Computing per-case S3 detail...")
    s3_detail = per_case_s3_detail(df)
    s3_detail.to_csv(output_dir / "s3_per_case_detail.csv", index=False)
    print("  Saved s3_per_case_detail.csv")

    # ------------------------------------------------------------------
    # (c) Cross-scenario generalisation
    # ------------------------------------------------------------------
    print("Computing cross-scenario generalisation summary...")
    cross_table = cross_scenario_summary(df)
    cross_table.to_csv(output_dir / "cross_scenario_summary.csv", index=False)
    print("  Saved cross_scenario_summary.csv")

    # ------------------------------------------------------------------
    # (d) Aggregate statistics
    # ------------------------------------------------------------------
    print("Computing aggregate statistics...")
    stats = aggregate_statistics(df, summary_json)
    with open(output_dir / "aggregate_statistics.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print("  Saved aggregate_statistics.json")

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    md_parts = [
        "# Multi-Target Study Aggregate Report\n",
        f"Seeds: {seeds}\n",
        f"Total observations: {len(df):,}\n",
    ]
    md_parts.append(df_to_markdown(scenario_table, "Per-Scenario Detection Rates"))
    md_parts.append(df_to_markdown(s3_detail, "S3 Per-Case Detail (All 10 Insiders, 5 Seeds)"))
    md_parts.append(df_to_markdown(cross_table, "Cross-Scenario Generalisation"))
    md_parts.append(stats_to_markdown(stats))

    # Include pre-computed scenario_results if available
    if scenario_results is not None:
        md_parts.append(df_to_markdown(scenario_results, "Pre-computed Scenario Results (from aggregate/)"))
    if defense_ablation is not None:
        md_parts.append(df_to_markdown(
            defense_ablation.head(20),
            "Defense Ablation Aggregate (top 20 rows)",
        ))

    report_text = "\n".join(md_parts)
    report_path = output_dir / "multi_target_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"  Saved {report_path.name}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    best = stats.get("best_combined_method", "N/A")
    print(f"  Best combined method : {best}")
    for label in ("baseline", "combined"):
        dr = stats.get(f"{label}_mean_detection_rate")
        fp = stats.get(f"{label}_mean_false_positive_rate")
        net = stats.get(f"{label}_mean_net_detection_rate")
        if dr is not None:
            print(f"  {label:>10s} | detect={dr:.4f}  FP={fp:.4f}  net={net:.4f}")
    if "mean_recovery_R" in stats:
        r = stats["mean_recovery_R"]
        lo = stats.get("recovery_R_ci_lower", "?")
        hi = stats.get("recovery_R_ci_upper", "?")
        print(f"  Recovery R           : {r:.4f}  [{lo:.4f}, {hi:.4f}]")
    print("=" * 60)
    print(f"  Reports in: {output_dir}")
    print()


if __name__ == "__main__":
    main()

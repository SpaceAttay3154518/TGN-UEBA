"""Stage-oriented command line interface."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from .io import load_config, load_prepared, resolve_config_path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _versions() -> dict[str, str]:
    names = ["torch", "torch-geometric", "numpy", "pandas", "pyarrow", "scikit-learn"]
    output = {}
    for name in names:
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = "MISSING"
    return output


def preflight(config_path: Path) -> int:
    config = load_config(config_path)
    dataset = config["dataset"]
    raw = resolve_config_path(config, dataset["raw_root"])
    answers = resolve_config_path(config, dataset["answers_root"])
    required = [raw / name for name in ("logon.csv", "device.csv", "file.csv", "http.csv", "email.csv")]
    target = answers / f"r4.2-{dataset['target_scenario']}" / (
        f"r4.2-{dataset['target_scenario']}-{dataset['target_user']}.csv"
    )
    required.append(target)
    versions = _versions()
    missing_packages = [name for name, version in versions.items() if version == "MISSING"]
    missing_files = [str(path) for path in required if not path.exists()]
    report = {
        "config": str(config_path.resolve()),
        "python": sys.version.split()[0],
        "packages": versions,
        "raw_dataset": str(raw),
        "required_files_present": not missing_files,
        "missing_files": missing_files,
        "missing_packages": missing_packages,
        "training_started": False,
        "ready_for_preparation": not missing_files and not missing_packages,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ready_for_preparation"] else 2


def status(config_path: Path) -> int:
    config = load_config(config_path)
    prepared = resolve_config_path(config, config["paths"]["prepared"])
    checkpoints = resolve_config_path(config, config["paths"]["checkpoints"])
    artifacts = resolve_config_path(config, config["paths"]["artifacts"])
    expected = [checkpoints / f"tgn_seed_{seed}.pt" for seed in config["model_seeds"]]
    report = {
        "prepared_manifest": str(prepared / "manifest.json"),
        "prepared": (prepared / "manifest.json").exists(),
        "expected_checkpoints": [str(path) for path in expected],
        "completed_checkpoints": [str(path) for path in expected if path.exists()],
        "training_complete": all(path.exists() for path in expected),
        "artifact_files": (
            len(
                [
                    path
                    for path in artifacts.rglob("*")
                    if path.is_file() and path.name != ".gitkeep"
                ]
            )
            if artifacts.exists()
            else 0
        ),
    }
    print(json.dumps(report, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="emdd")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "preflight",
        "prepare",
        "validate",
        "train",
        "training-report",
        "study",
        "report",
        "status",
        "longitudinal-study",
        "longitudinal-report",
        "slow-conditioning-study",
        "multi-target-study",
    ):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", type=Path, required=True)
        if command in {"train", "study", "longitudinal-study", "slow-conditioning-study", "multi-target-study"}:
            sub.add_argument("--seed", type=int, action="append")
        if command == "study":
            sub.add_argument("--duration", type=int, action="append")
            sub.add_argument("--rate", type=int, action="append")
        if command == "multi-target-study":
            sub.add_argument("--case", type=str, action="append",
                             help="Specific case IDs (e.g. s3:BBS0039). Repeatable. Default: all dev+test cases.")
            sub.add_argument("--scenario", type=int, action="append",
                             help="Include all cases from these scenarios (1-5). Repeatable.")
    datasets_parser = subparsers.add_parser("datasets")
    datasets_parser.add_argument(
        "--catalog", type=Path, default=PACKAGE_ROOT / "config" / "benchmark_catalog.json"
    )
    datasets_parser.add_argument(
        "--raw-root", type=Path, default=PACKAGE_ROOT / "data" / "raw_benchmarks"
    )
    datasets_parser.add_argument(
        "--prepared-root", type=Path, default=PACKAGE_ROOT / "data" / "training_ready"
    )
    prepare_parser = subparsers.add_parser("prepare-dataset")
    prepare_parser.add_argument("--dataset", required=True)
    prepare_parser.add_argument("--raw-root", type=Path, required=True)
    prepare_parser.add_argument(
        "--prepared-root", type=Path, default=PACKAGE_ROOT / "data" / "training_ready"
    )
    prepare_parser.add_argument(
        "--catalog", type=Path, default=PACKAGE_ROOT / "config" / "benchmark_catalog.json"
    )
    subparsers.add_parser("kairos-reference")
    args = parser.parse_args()

    if args.command == "datasets":
        from .benchmarks import write_audit

        print(
            json.dumps(
                write_audit(args.catalog, args.raw_root, args.prepared_root), indent=2
            )
        )
        return
    if args.command == "prepare-dataset":
        from .benchmarks import load_catalog, write_audit

        catalog = load_catalog(args.catalog)
        if args.dataset not in catalog["datasets"]:
            raise SystemExit(f"Unknown dataset: {args.dataset}")
        if args.dataset == "streamspot":
            from .streamspot import prepare_streamspot, validate_streamspot

            output = args.prepared_root.resolve() / "streamspot"
            manifest = prepare_streamspot(
                args.raw_root.resolve() / "streamspot" / "all.tar.gz", output
            )
            validate_streamspot(output)
            write_audit(args.catalog, args.raw_root, args.prepared_root)
            print(json.dumps(manifest, indent=2))
            return
        raise SystemExit(
            f"The {args.dataset} adapter is registered but cannot run until its raw "
            "release and ground truth are present; see `emdd datasets`."
        )
    if args.command == "kairos-reference":
        from .kairos_protocol import KAIROS_REFERENCE

        print(json.dumps(KAIROS_REFERENCE, indent=2))
        return

    if args.command == "preflight":
        raise SystemExit(preflight(args.config))
    if args.command == "status":
        raise SystemExit(status(args.config))
    if args.command in {"prepare", "validate"}:
        command = [sys.executable, "-m", "emdd.data", "--config", str(args.config)]
        if args.command == "validate":
            command.append("--validate-only")
        raise SystemExit(subprocess.call(command))

    config = load_config(args.config)
    if args.command == "report":
        from .reporting import aggregate_artifacts

        print(json.dumps(aggregate_artifacts(config), indent=2))
        return
    if args.command == "training-report":
        from .training_reporting import aggregate_training

        print(json.dumps(aggregate_training(config), indent=2))
        return
    if args.command == "longitudinal-report":
        from .longitudinal_reporting import aggregate_longitudinal

        print(json.dumps(aggregate_longitudinal(config), indent=2))
        return
    events, metadata = load_prepared(config)
    if args.command == "train":
        from .training import train_all

        summaries = train_all(events, metadata, config, args.seed)
        print(json.dumps(summaries, indent=2))
        return
    if args.command == "study":
        from .study import run_seed_study

        seeds = args.seed or [int(seed) for seed in config["model_seeds"]]
        summaries = [
            run_seed_study(
                seed,
                events,
                metadata,
                config,
                durations=args.duration,
                rates=args.rate,
            )
            for seed in seeds
        ]
        from .reporting import aggregate_artifacts

        aggregate = aggregate_artifacts(config)
        print(json.dumps({"seeds": summaries, "aggregate": aggregate}, indent=2, default=str))
        return
    if args.command == "longitudinal-study":
        from .longitudinal_study import run_longitudinal_study

        summaries = run_longitudinal_study(events, metadata, config, args.seed)
        print(json.dumps(summaries, indent=2, default=str))
        return
    if args.command == "slow-conditioning-study":
        from .slow_conditioning_study import run_slow_conditioning_study

        summaries = run_slow_conditioning_study(events, metadata, config, args.seed)
        print(json.dumps(summaries, indent=2, default=str))
    if args.command == "multi-target-study":
        from .multi_target_study import run_multi_target_study

        summaries = run_multi_target_study(
            events, metadata, config,
            cases=args.case,
            seeds=args.seed,
            scenarios=args.scenario,
        )
        print(json.dumps(summaries, indent=2, default=str))


if __name__ == "__main__":
    main()

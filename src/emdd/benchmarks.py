"""Benchmark registry and strict training-ready dataset audit.

The registry separates three different claims that are easy to conflate:

* a dataset named in a survey table;
* a dataset actually evaluated by the GIDS state-of-the-art study; and
* a dataset used by KAIROS with a reproducible split and metric protocol.

A directory is marked READY only after a preparer writes both ``metadata.json``
and ``manifest.json`` and the contract validator succeeds.
"""

from __future__ import annotations

import fnmatch
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "config" / "benchmark_catalog.json"


def load_catalog(path: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    catalog = json.loads(path.resolve().read_text())
    if int(catalog.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported benchmark catalog schema")
    dataset_names = set(catalog.get("datasets", {}))
    for suite, members in catalog.get("suites", {}).items():
        unknown = set(members) - dataset_names
        if unknown:
            raise ValueError(f"Suite {suite} names unknown datasets: {sorted(unknown)}")
    return catalog


def _matches(root: Path, pattern: str) -> list[str]:
    """Return relative files matching a required literal or recursive glob."""
    if not root.exists():
        return []
    if any(symbol in pattern for symbol in "*?["):
        return sorted(
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and fnmatch.fnmatch(str(path.relative_to(root)), pattern)
        )
    candidate = root / pattern
    return [pattern] if candidate.is_file() else []


def prepared_contract_status(directory: Path) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for filename in ("metadata.json", "manifest.json"):
        if not (directory / filename).is_file():
            reasons.append(f"missing {filename}")
    if reasons:
        return False, reasons
    try:
        metadata = json.loads((directory / "metadata.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        return False, [f"invalid metadata or manifest: {error}"]
    if not (directory / "events").is_dir() and not (directory / "events.parquet").is_file():
        reasons.append("missing prepared events")
    if int(manifest.get("rows", 0)) <= 0:
        reasons.append("manifest has no rows")
    if int(metadata.get("num_nodes", len(metadata.get("node_to_id", {})))) <= 0:
        reasons.append("metadata has no nodes")
    if not metadata.get("event_types"):
        reasons.append("metadata has no event types")
    return not reasons, reasons


def audit_datasets(catalog: dict[str, Any], raw_root: Path, prepared_root: Path) -> dict:
    usage = shutil.disk_usage(prepared_root.parent if prepared_root.parent.exists() else Path.cwd())
    rows: dict[str, dict[str, Any]] = {}
    for name, spec in catalog["datasets"].items():
        prepared = prepared_root / spec["prepared_directory"]
        ready, prepared_reasons = prepared_contract_status(prepared)
        raw_directory = raw_root / name
        required = spec.get("raw_required", [])
        matches = {pattern: _matches(raw_directory, pattern) for pattern in required}
        missing = [pattern for pattern, found in matches.items() if not found]
        declared = str(spec["status"])
        if ready:
            effective = "READY"
            reasons: list[str] = []
        elif declared == "unavailable_private":
            effective = "BLOCKED_PRIVATE"
            reasons = [str(spec.get("blocker", "private corpus is unavailable"))]
        elif declared == "external_storage_required":
            effective = "BLOCKED_STORAGE"
            reasons = [str(spec.get("blocker", "external storage is required")), *prepared_reasons]
        elif declared in {"survey_only", "excluded_by_study"}:
            effective = "NOT_IN_BENCHMARK"
            reasons = [str(spec["benchmark_role"])]
        elif missing:
            effective = "RAW_REQUIRED"
            reasons = [f"missing raw input: {pattern}" for pattern in missing]
        else:
            effective = "PREPARATION_REQUIRED"
            reasons = prepared_reasons
        rows[name] = {
            "display_name": spec["display_name"],
            "effective_status": effective,
            "prepared_directory": str(prepared.resolve()),
            "raw_directory": str(raw_directory.resolve()),
            "raw_matches": matches,
            "reasons": reasons,
            "benchmark_role": spec["benchmark_role"],
        }
    return {
        "schema_version": 1,
        "prepared_root": str(prepared_root.resolve()),
        "raw_root": str(raw_root.resolve()),
        "disk_free_bytes": usage.free,
        "suites": catalog["suites"],
        "datasets": rows,
        "counts": {
            status: sum(row["effective_status"] == status for row in rows.values())
            for status in sorted({row["effective_status"] for row in rows.values()})
        },
    }


def write_audit(catalog_path: Path, raw_root: Path, prepared_root: Path) -> dict:
    catalog = load_catalog(catalog_path)
    prepared_root.mkdir(parents=True, exist_ok=True)
    report = audit_datasets(catalog, raw_root.resolve(), prepared_root.resolve())
    (prepared_root / "INDEX.json").write_text(json.dumps(report, indent=2))
    return report


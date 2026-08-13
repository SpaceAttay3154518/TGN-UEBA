#!/bin/bash
# run_multi_target.sh -- Runner for emdd multi-target-study across seeds and scenarios.
#
# Invokes `emdd multi-target-study` with configurable seeds, scenarios, and
# case filters.  Logs are written to experiments/slow_drift_defense/logs/ with
# a timestamped filename.
#
# Usage examples:
#   ./run_multi_target.sh                          # all 5 seeds, S3 dev+test cases
#   ./run_multi_target.sh --seeds 17,29            # two seeds only
#   ./run_multi_target.sh --cross-scenario          # adds S1 and S2 cases
#   ./run_multi_target.sh --cases s3:BBS0039,s3:CSC0217
#   ./run_multi_target.sh --scenarios 1,3           # scenario 1 + scenario 3

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${EMDD_CONFIG:-$PROJECT_ROOT/codebase/config/cert_r42_longitudinal.json}"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ALL_SEEDS="17,29,43,59,71"
SEEDS=""
SCENARIOS=""
CASES=""
CROSS_SCENARIO=0

# GPU settings
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --seeds SEED_LIST       Comma-separated seed list (default: $ALL_SEEDS)
  --scenarios SCEN_LIST   Comma-separated scenario numbers, e.g. 1,3
  --cases CASE_LIST       Comma-separated case IDs, e.g. s3:BBS0039,s3:CSC0217
  --cross-scenario        Include S1 and S2 scenarios alongside default S3
  --config PATH           Override config file (default: cert_r42_longitudinal.json)
  --gpu DEVICE_ID         CUDA_VISIBLE_DEVICES value (default: 0)
  -h, --help              Show this help message
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)
            SEEDS="$2"; shift 2 ;;
        --scenarios)
            SCENARIOS="$2"; shift 2 ;;
        --cases)
            CASES="$2"; shift 2 ;;
        --cross-scenario)
            CROSS_SCENARIO=1; shift ;;
        --config)
            CONFIG="$2"; shift 2 ;;
        --gpu)
            export CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
        -h|--help)
            usage 0 ;;
        *)
            echo "Error: unknown argument '$1'" >&2
            usage 1 ;;
    esac
done

# Apply defaults
if [[ -z "$SEEDS" ]]; then
    SEEDS="$ALL_SEEDS"
fi

# ---------------------------------------------------------------------------
# Validate config
# ---------------------------------------------------------------------------
if [[ ! -f "$CONFIG" ]]; then
    echo "Error: config file not found: $CONFIG" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build the emdd command
# ---------------------------------------------------------------------------
CMD=(emdd multi-target-study --config "$CONFIG")

# Seeds
IFS=',' read -ra SEED_ARR <<< "$SEEDS"
for s in "${SEED_ARR[@]}"; do
    CMD+=(--seed "$s")
done

# Explicit cases
if [[ -n "$CASES" ]]; then
    IFS=',' read -ra CASE_ARR <<< "$CASES"
    for c in "${CASE_ARR[@]}"; do
        CMD+=(--case "$c")
    done
fi

# Explicit scenarios
if [[ -n "$SCENARIOS" ]]; then
    IFS=',' read -ra SCEN_ARR <<< "$SCENARIOS"
    for sc in "${SCEN_ARR[@]}"; do
        CMD+=(--scenario "$sc")
    done
fi

# Cross-scenario flag: add S1 and S2
if [[ "$CROSS_SCENARIO" -eq 1 ]]; then
    # Only add scenarios if not already specified via --scenarios
    if [[ -z "$SCENARIOS" ]]; then
        CMD+=(--scenario 1 --scenario 2 --scenario 3)
    fi
fi

# If neither --cases, --scenarios, nor --cross-scenario is given, the CLI
# defaults to all dev+test S3 cases (the emdd built-in default).

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOGFILE="$LOG_DIR/multi_target_${TIMESTAMP}.log"

echo "============================================================"
echo "  emdd multi-target-study"
echo "============================================================"
echo "  Timestamp : $TIMESTAMP"
echo "  Config    : $CONFIG"
echo "  Seeds     : $SEEDS"
echo "  Cases     : ${CASES:-<default: all S3>}"
echo "  Scenarios : ${SCENARIOS:-<default>}${CROSS_SCENARIO:+ (+cross-scenario)}"
echo "  GPU       : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Log file  : $LOGFILE"
echo "  Command   : ${CMD[*]}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Run with timing
# ---------------------------------------------------------------------------
START_EPOCH="$(date +%s)"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting multi-target study..."

# Tee to both stdout and log file; preserve exit code via pipefail
if "${CMD[@]}" 2>&1 | tee "$LOGFILE"; then
    STATUS=0
else
    STATUS=$?
fi

END_EPOCH="$(date +%s)"
ELAPSED=$(( END_EPOCH - START_EPOCH ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS_REM=$(( ELAPSED % 60 ))

echo ""
echo "============================================================"
if [[ "$STATUS" -eq 0 ]]; then
    echo "  COMPLETED SUCCESSFULLY"
else
    echo "  FAILED (exit code $STATUS)"
fi
echo "  Wall time : ${HOURS}h ${MINUTES}m ${SECONDS_REM}s  (${ELAPSED}s total)"
echo "  Log       : $LOGFILE"
echo "============================================================"

exit "$STATUS"

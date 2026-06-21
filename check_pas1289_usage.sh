#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-PAS1289}"
START_DATE="${START_DATE:-$(date +%Y-%m-01)}"
END_DATE="${END_DATE:-$(date +%F)}"
PERIOD="${PERIOD:-month}"
MODE="range"
SHOW_ME="0"
SHOW_RAW="0"
SHOW_HOURS="0"
JOB_NAME=""
SYSTEM_NAME=""

usage() {
    cat <<'EOF'
Usage:
  ./check_pas1289_usage.sh [options]

Default:
  Shows PAS1289 dollar usage from the first day of this month through today.

Options:
  -P, --project PROJECT      Project/account code (default: PAS1289)
  -s, --start YYYY-MM-DD     Start date for range mode
  -e, --end YYYY-MM-DD       End date for range mode
  -C, --current              Use OSC current-unbilled mode first
  -p, --period PERIOD        current-unbilled period: month, quarter, annual
  -q, --me                   Show current user's usage only
  -S, --system SYSTEM        OSC system: pitzer, cardinal, ascend, ...
  -N, --job-name PATTERN     Filter by job name
  -H, --hours                Ask OSCusage to show hours
  -r, --raw                  Ask OSCusage to show raw units
  -h, --help                 Show this help

Environment overrides:
  PROJECT=PAS1289 START_DATE=2026-06-01 END_DATE=2026-06-19 ./check_pas1289_usage.sh
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -P|--project)
            PROJECT="$2"
            shift 2
            ;;
        -s|--start)
            START_DATE="$2"
            shift 2
            ;;
        -e|--end)
            END_DATE="$2"
            shift 2
            ;;
        -C|--current)
            MODE="current"
            shift
            ;;
        -p|--period)
            PERIOD="$2"
            shift 2
            ;;
        -q|--me)
            SHOW_ME="1"
            shift
            ;;
        -S|--system)
            SYSTEM_NAME="$2"
            shift 2
            ;;
        -N|--job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        -H|--hours)
            SHOW_HOURS="1"
            shift
            ;;
        -r|--raw)
            SHOW_RAW="1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v OSCusage >/dev/null 2>&1; then
    if [[ -x /opt/osc/bin/OSCusage ]]; then
        OSCUSAGE=/opt/osc/bin/OSCusage
    else
        echo "ERROR: OSCusage was not found. This script needs OSC's OSCusage command for dollar accounting." >&2
        if command -v sreport >/dev/null 2>&1; then
            echo >&2
            echo "Fallback Slurm utilization report (not dollars):" >&2
            sreport -n -P cluster AccountUtilizationByUser \
                "Accounts=${PROJECT}" "Start=${START_DATE}" "End=${END_DATE}" || true
        fi
        exit 1
    fi
else
    OSCUSAGE="$(command -v OSCusage)"
fi

run_oscusage() {
    local -a args=()
    args+=("-P" "${PROJECT}")
    [[ -n "${SYSTEM_NAME}" ]] && args+=("-s" "${SYSTEM_NAME}")
    [[ "${SHOW_ME}" == "1" ]] && args+=("-q")
    [[ "${SHOW_HOURS}" == "1" ]] && args+=("-H")
    [[ "${SHOW_RAW}" == "1" ]] && args+=("-r")
    [[ -n "${JOB_NAME}" ]] && args+=("-N" "${JOB_NAME}")

    if [[ "${MODE}" == "current" ]]; then
        args+=("-C" "-p" "${PERIOD}")
    else
        args+=("${START_DATE}" "${END_DATE}")
    fi

    "${OSCUSAGE}" "${args[@]}" 2>&1
}

print_summary() {
    local output="$1"
    local remaining total used_budget pct

    remaining="$(awk '/Remaining Budget/ {print $NF; exit}' <<<"${output}")"
    total="$(awk '$1 == "TOTAL" {print $3; found=1} END {if (!found) exit 1}' <<<"${output}" 2>/dev/null || true)"

    if [[ -z "${remaining}" && -z "${total}" ]]; then
        return 0
    fi

    echo
    echo "Quick summary"
    echo "-------------"
    [[ -n "${total}" ]] && printf "Range dollars used : $%s\n" "${total}"
    [[ -n "${remaining}" ]] && printf "Remaining budget   : $%s\n" "${remaining}"

    if [[ -n "${remaining}" && -n "${total}" ]]; then
        read -r used_budget pct < <(
            awk -v used="${total}" -v rem="${remaining}" 'BEGIN {
                budget = used + rem
                if (budget > 0) {
                    printf "%.4f %.2f\n", budget, 100.0 * used / budget
                }
            }'
        )
        if [[ -n "${used_budget:-}" ]]; then
            printf "Implied budget     : $%s\n" "${used_budget}"
            printf "Used share         : %s%%\n" "${pct}"
        fi
    fi
}

echo "Project: ${PROJECT}"
if [[ "${MODE}" == "current" ]]; then
    echo "Mode   : current unbilled (${PERIOD})"
else
    echo "Range  : ${START_DATE} to ${END_DATE}"
fi
echo "Command: ${OSCUSAGE}"
echo

set +e
OUTPUT="$(run_oscusage)"
STATUS=$?
set -e

if [[ "${MODE}" == "current" ]] && grep -q "Unexpected HTTP status 204" <<<"${OUTPUT}"; then
    echo "${OUTPUT}"
    echo
    echo "Current-unbilled query returned HTTP 204; falling back to date range."
    MODE="range"
    set +e
    OUTPUT="$(run_oscusage)"
    STATUS=$?
    set -e
fi

echo "${OUTPUT}"
print_summary "${OUTPUT}"

if [[ "${STATUS}" -ne 0 ]] || grep -q "^ERROR:" <<<"${OUTPUT}"; then
    echo >&2
    echo "WARNING: OSCusage reported an error. The output above may be incomplete." >&2
    exit "${STATUS}"
fi

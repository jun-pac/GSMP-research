#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-PAS1289}"
START_DATE="${START_DATE:-$(date +%Y-%m-01)}"
END_DATE="${END_DATE:-$(date +%F)}"
PERIOD="${PERIOD:-month}"
MODE="range"
SHOW_ALL="0"
SORT_BY="dollars"
SYSTEM_NAME=""
JOB_NAME=""

usage() {
    cat <<'EOF'
Usage:
  ./check_pas1289_usage_by_account.sh [options]

Default:
  Shows PAS1289 usage by user/account from the first day of this month through today.
  Zero-dollar accounts are hidden unless --all is passed.

Options:
  -P, --project PROJECT      Project/account code (default: PAS1289)
  -s, --start YYYY-MM-DD     Start date for range mode
  -e, --end YYYY-MM-DD       End date for range mode
  -C, --current              Try OSC current-unbilled mode first
  -p, --period PERIOD        current-unbilled period: month, quarter, annual
  -a, --all                  Include zero-dollar / zero-job accounts
  --sort KEY                 Sort by dollars, jobs, or name (default: dollars)
  -S, --system SYSTEM        OSC system: pitzer, cardinal, ascend, ...
  -N, --job-name PATTERN     Filter by job name
  -h, --help                 Show this help

Examples:
  ./check_pas1289_usage_by_account.sh
  ./check_pas1289_usage_by_account.sh --all
  ./check_pas1289_usage_by_account.sh --start 2026-06-01 --end 2026-06-19
  ./check_pas1289_usage_by_account.sh --sort jobs
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
        -a|--all)
            SHOW_ALL="1"
            shift
            ;;
        --sort)
            SORT_BY="$2"
            shift 2
            ;;
        -S|--system)
            SYSTEM_NAME="$2"
            shift 2
            ;;
        -N|--job-name)
            JOB_NAME="$2"
            shift 2
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

case "${SORT_BY}" in
    dollars|jobs|name) ;;
    *)
        echo "ERROR: --sort must be dollars, jobs, or name" >&2
        exit 2
        ;;
esac

if command -v OSCusage >/dev/null 2>&1; then
    OSCUSAGE="$(command -v OSCusage)"
elif [[ -x /opt/osc/bin/OSCusage ]]; then
    OSCUSAGE=/opt/osc/bin/OSCusage
else
    echo "ERROR: OSCusage was not found. This script needs OSC's OSCusage command." >&2
    exit 1
fi

run_oscusage() {
    local -a args=("-P" "${PROJECT}")
    [[ -n "${SYSTEM_NAME}" ]] && args+=("-s" "${SYSTEM_NAME}")
    [[ -n "${JOB_NAME}" ]] && args+=("-N" "${JOB_NAME}")

    if [[ "${MODE}" == "current" ]]; then
        args+=("-C" "-p" "${PERIOD}")
    else
        args+=("${START_DATE}" "${END_DATE}")
    fi

    "${OSCUSAGE}" "${args[@]}" 2>&1
}

set +e
OUTPUT="$(run_oscusage)"
STATUS=$?
set -e

if [[ "${MODE}" == "current" ]] && grep -q "Unexpected HTTP status 204" <<<"${OUTPUT}"; then
    echo "OSCusage current-unbilled returned HTTP 204; falling back to date range." >&2
    MODE="range"
    set +e
    OUTPUT="$(run_oscusage)"
    STATUS=$?
    set -e
fi

if [[ "${STATUS}" -ne 0 ]] || grep -q "^ERROR:" <<<"${OUTPUT}"; then
    echo "${OUTPUT}" >&2
    exit "${STATUS}"
fi

TIME_RANGE="$(awk '/^Time[[:space:]]/ {sub(/^Time[[:space:]]+/, ""); print; exit}' <<<"${OUTPUT}")"
REMAINING="$(awk '/Remaining Budget/ {print $NF; exit}' <<<"${OUTPUT}")"
TOTAL_JOBS="$(awk '$1 == "TOTAL" {print $2; exit}' <<<"${OUTPUT}")"
TOTAL_DOLLARS="$(awk '$1 == "TOTAL" {print $3; exit}' <<<"${OUTPUT}")"

if [[ -z "${TOTAL_DOLLARS}" ]]; then
    echo "ERROR: Could not parse TOTAL dollars from OSCusage output." >&2
    echo "${OUTPUT}" >&2
    exit 1
fi

ROWS="$(
    awk -v total="${TOTAL_DOLLARS}" -v show_all="${SHOW_ALL}" '
        $1 == "--" {in_table=0}
        in_table && $1 != "User" && $1 !~ /^-+$/ && $1 != "TOTAL" {
            user=$1
            jobs=$2 + 0
            dollars=$3 + 0
            status=$4
            if (show_all == "1" || jobs > 0 || dollars > 0) {
                pct = (total > 0 ? dollars * 100.0 / total : 0)
                printf "%s\t%d\t%.6f\t%s\t%.2f\n", user, jobs, dollars, status, pct
            }
        }
        $1 == "----------" && $2 == "------" {in_table=1}
    ' <<<"${OUTPUT}"
)"

case "${SORT_BY}" in
    dollars)
        SORTED_ROWS="$(sort -t $'\t' -k3,3gr -k2,2gr <<<"${ROWS}")"
        ;;
    jobs)
        SORTED_ROWS="$(sort -t $'\t' -k2,2gr -k3,3gr <<<"${ROWS}")"
        ;;
    name)
        SORTED_ROWS="$(sort -t $'\t' -k1,1 <<<"${ROWS}")"
        ;;
esac

echo "Project          : ${PROJECT}"
if [[ "${MODE}" == "current" ]]; then
    echo "Mode             : current unbilled (${PERIOD})"
else
    echo "Range            : ${TIME_RANGE:-${START_DATE} to ${END_DATE}}"
fi
echo "Total jobs       : ${TOTAL_JOBS}"
printf "Total dollars    : $%.4f\n" "${TOTAL_DOLLARS}"
if [[ -n "${REMAINING}" ]]; then
    printf "Remaining budget : $%.4f\n" "${REMAINING}"
fi
echo

printf "%-14s %8s %14s %10s %9s\n" "Account" "Jobs" "Dollars" "Share" "Status"
printf "%-14s %8s %14s %10s %9s\n" "--------------" "--------" "--------------" "----------" "---------"
awk -F '\t' '
    NF {
        printf "%-14s %8d %14.4f %9.2f%% %9s\n", $1, $2, $3, $5, $4
    }
' <<<"${SORTED_ROWS}"

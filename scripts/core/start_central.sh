#!/bin/bash
# Start all central-computer processes for multi-robot exploration.
#
# This is the central entry point for both:
#   - shared-domain mode (legacy)
#   - bridged multi-domain mode (per-robot domains + central domain)
# It starts:
#   1. TF relay        (/<robot>/tf -> /tf with consistent frame prefixing)
#   2. Single-robot helpers (only when one robot is selected):
#      - static TF map -> <robot>/map (identity bridge)
#      - relay /<robot>/map(_wire_z) -> /map for central consumers
#   3. Map merge       (multi-robot only; skipped when one robot)
#   4. Explorer        (detect frontiers, send Nav2 action goals)
#   5. Safe-node planner (optional, off by default: --safe-node-planner or
#      SAFE_NODE_PLANNER_ENABLE=true)
#
# Prerequisites:
#   - shared_domain mode:
#       all robots and central use the same ROS_DOMAIN_ID (typically 50)
#   - bridged_domains mode:
#       central uses ROS_DOMAIN_ID you set (or fleet_domain_map.central_domain_id); robots use
#       fleet_domain_map robot_domain_ids; domain_bridge connects them.
#   - Each robot is powered on and running:
#       * bringup       (robot.launch.py)
#       * SLAM + Nav2   (navigation2_slam.launch.py fleet_mode:=True with this script)
#   - Central PC can reach robots (same Azure LAN, or Tailscale+Zenoh on TAMU_WiFi)
#
# Usage:
#   Default comms mode is bridged_domains (override: --comms-mode shared_domain or CENTRAL_COMMS_MODE).
#   export ROS_DOMAIN_ID=64    # optional; default is fleet_domain_map.central_domain_id when unset
#   ./scripts/core/start_central.sh
#   ./scripts/core/start_central.sh --comms-mode shared_domain   # legacy single-domain fleet
#   ./scripts/core/start_central.sh -b      # only Blinky
#   ./scripts/core/start_central.sh -pi     # only Pinky + Inky
#   ./scripts/core/start_central.sh -bpic   # Blinky, Pinky, Inky, Clyde (subset must appear on the graph)
#   # TAMU + Zenoh: start ./scripts/comms/start_zenoh_central.sh first, then this script.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
EXPLORE_PKG_DIR="${WORKSPACE_DIR}/src/m-explore-ros2/explore"
EXPLORE_SCRIPT_DIR="${EXPLORE_PKG_DIR}/scripts"
MAP_MERGE_PKG_DIR="${WORKSPACE_DIR}/src/m-explore-ros2/map_merge"
MAP_MERGE_CONFIG_FILE="${MAP_MERGE_PKG_DIR}/config/multirobot_params_unknown_poses.yaml"
EXPLORE_CONFIG_FILE="${EXPLORE_PKG_DIR}/config/multi_robot_explorer.yaml"
BRIDGE_CONTRACT_FILE="${WORKSPACE_DIR}/config/fleet_bridge_contract.yaml"
DOMAIN_MAP_FILE="${WORKSPACE_DIR}/config/fleet_domain_map.yaml"
BRIDGE_CONFIG_GENERATOR="${WORKSPACE_DIR}/scripts/bridging/generate_domain_bridge_configs.py"
GENERATED_BRIDGE_CONFIG_DIR="${WORKSPACE_DIR}/config/generated_domain_bridge"
TF_RELAY_SCRIPT="${EXPLORE_SCRIPT_DIR}/tf_relay_multirobot.py"
EXPLORER_SCRIPT="${EXPLORE_SCRIPT_DIR}/multi_robot_explorer.py"
SAFE_NODE_PLANNER_SCRIPT="${EXPLORE_SCRIPT_DIR}/safe_node_planner.py"
SINGLE_ROBOT_WORLD_TF_SCRIPT="${EXPLORE_SCRIPT_DIR}/single_robot_world_tf_bridge.py"
SINGLE_ROBOT_MAP_RELAY_SCRIPT="${EXPLORE_SCRIPT_DIR}/single_robot_map_relay.py"
ACTION_RELAY_SCRIPT="${WORKSPACE_DIR}/scripts/bridging/fleet_navigate_to_pose_service_relay.py"

cd "$WORKSPACE_DIR"
source /opt/ros/humble/setup.bash
source install/setup.bash 2>/dev/null

# Zenoh bridges publish robot-domain traffic onto localhost only. When a local
# zenoh-bridge-ros2dds is running (or CENTRAL_ZENOH=1), force Cyclone + localhost
# discovery so start_central / domain_bridge can see those topics.
zenoh_transport_active() {
    [[ "${CENTRAL_ZENOH:-}" == "1" || "${CENTRAL_ZENOH:-}" == "true" ]] && return 0
    pgrep -f 'zenoh-bridge-ros2dds' >/dev/null 2>&1
}

USING_ZENOH_TRANSPORT=0
if zenoh_transport_active; then
    USING_ZENOH_TRANSPORT=1
    export ROS_DISTRO="${ROS_DISTRO:-humble}"
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export ROS_LOCALHOST_ONLY=1
    # The Zenoh bridge uses this localhost mode and no CYCLONEDDS_URI.
    # A custom URI, even one that also selects 127.0.0.1, cannot see the
    # topics the bridge injects, so /<robot>/(tf|map|map_wire_z) stay empty.
    # RViz raises the participant cap on its own; this process must not.
    unset CYCLONEDDS_URI
fi

if [[ ! -f "$TF_RELAY_SCRIPT" ]]; then
    echo "ERROR: TF relay script not found: $TF_RELAY_SCRIPT"
    exit 1
fi
if [[ ! -f "$EXPLORER_SCRIPT" ]]; then
    echo "ERROR: Explorer script not found: $EXPLORER_SCRIPT"
    exit 1
fi
if [[ ! -f "$EXPLORE_CONFIG_FILE" ]]; then
    echo "ERROR: Explorer config not found: $EXPLORE_CONFIG_FILE"
    exit 1
fi
if [[ ! -f "$SINGLE_ROBOT_WORLD_TF_SCRIPT" ]]; then
    echo "ERROR: Single-robot world TF script not found: $SINGLE_ROBOT_WORLD_TF_SCRIPT"
    exit 1
fi
if [[ ! -f "$SINGLE_ROBOT_MAP_RELAY_SCRIPT" ]]; then
    echo "ERROR: Single-robot map relay script not found: $SINGLE_ROBOT_MAP_RELAY_SCRIPT"
    exit 1
fi
if [[ ! -f "$MAP_MERGE_CONFIG_FILE" ]]; then
    echo "ERROR: Map merge config not found: $MAP_MERGE_CONFIG_FILE"
    exit 1
fi

DEFAULT_CENTRAL_DOMAIN_ID="$(
    python3 - <<'PY' "$DOMAIN_MAP_FILE"
import sys, yaml
path, fallback = sys.argv[1], 50
try:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cid = data.get("fleet_domain_map", {}).get("central_domain_id")
    print(int(cid) if cid is not None else fallback)
except Exception:
    print(fallback)
PY
)"
COMMS_MODE="${CENTRAL_COMMS_MODE:-bridged_domains}"
CENTRAL_TERMINAL_PROFILE="${CENTRAL_TERMINAL_PROFILE:-clean}"
CENTRAL_MAP_MERGE_RAW_STDOUT="${CENTRAL_MAP_MERGE_RAW_STDOUT:-false}"
USER_DOMAIN_ARG=""
DOMAIN_SOURCE=""
SELECTION=""
declare -A LETTER_TO_ROBOT=(
    [b]="blinky"
    [p]="pinky"
    [i]="inky"
    [c]="clyde"
)
declare -A SELECTED_ROBOT_SET=()
parse_selection_letters() {
    local letters="$1"
    SELECTED_ROBOT_SET=()
    [[ -z "$letters" ]] && return 0
    for ((idx=0; idx<${#letters}; idx++)); do
        local ch="${letters:$idx:1}"
        if [[ -z "${LETTER_TO_ROBOT[$ch]:-}" ]]; then
            echo "ERROR: invalid robot selector '${ch}' in '-${letters}'"
            echo "Use any combination of: b=blinky, p=pinky, i=inky, c=clyde"
            exit 1
        fi
        SELECTED_ROBOT_SET["${LETTER_TO_ROBOT[$ch]}"]=1
    done
}
while (($# > 0)); do
    case "$1" in
        --comms-mode)
            shift
            if (($# == 0)); then
                echo "ERROR: --comms-mode requires value shared_domain|bridged_domains"
                exit 1
            fi
            COMMS_MODE="$1"
            ;;
        --domain-id)
            shift
            if (($# == 0)); then
                echo "ERROR: --domain-id requires an integer value."
                exit 1
            fi
            USER_DOMAIN_ARG="$1"
            ;;
        -[bpic]*)
            SELECTION="${1#-}"
            ;;
        --safe-node-planner)
            SAFE_NODE_PLANNER_ENABLE=true
            ;;
        -h|--help)
            cat <<'EOF'
Usage:
  ./scripts/core/start_central.sh [robot_filter] [--comms-mode <shared_domain|bridged_domains>] [--domain-id <id>] [--safe-node-planner]

  --safe-node-planner   also start safe_node_planner.py (publishes /<robot>/safe_node
                        and /<robot>/connection_risk). Same as SAFE_NODE_PLANNER_ENABLE=true.

Examples:
  ./scripts/core/start_central.sh
  ./scripts/core/start_central.sh -pi
  ./scripts/core/start_central.sh --domain-id 64
  ./scripts/core/start_central.sh --comms-mode bridged_domains
  ./scripts/core/start_central.sh -b --comms-mode bridged_domains --domain-id 64
EOF
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument '$1'"
            exit 1
            ;;
    esac
    shift
done

if [[ "$COMMS_MODE" != "shared_domain" && "$COMMS_MODE" != "bridged_domains" ]]; then
    echo "ERROR: invalid --comms-mode '$COMMS_MODE' (use shared_domain|bridged_domains)"
    exit 1
fi
if [[ -n "$USER_DOMAIN_ARG" && ! "$USER_DOMAIN_ARG" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid --domain-id '$USER_DOMAIN_ARG' (must be integer)."
    exit 1
fi
parse_selection_letters "$SELECTION"

is_clean_profile() {
    [[ "$CENTRAL_TERMINAL_PROFILE" != "verbose" ]]
}

# Domain precedence (both comms modes):
#   1) --domain-id
#   2) existing ROS_DOMAIN_ID in shell
#   3) default central domain (fleet_domain_map.central_domain_id, else 50)
if [[ -n "$USER_DOMAIN_ARG" ]]; then
    export ROS_DOMAIN_ID="$USER_DOMAIN_ARG"
    DOMAIN_SOURCE="cli(--domain-id)"
elif [[ -n "${ROS_DOMAIN_ID:-}" ]]; then
    DOMAIN_SOURCE="env(ROS_DOMAIN_ID)"
else
    export ROS_DOMAIN_ID="${DEFAULT_CENTRAL_DOMAIN_ID}"
    DOMAIN_SOURCE="default(${DEFAULT_CENTRAL_DOMAIN_ID})"
fi

echo "=========================================="
echo "  Central Computer — Multi-Robot Exploration"
echo "=========================================="
echo ""
if [[ "$COMMS_MODE" == "shared_domain" ]]; then
    echo "  Comms mode      = shared_domain"
    echo "  ROS_DOMAIN_ID   = $ROS_DOMAIN_ID"
else
    echo "  Comms mode      = bridged_domains"
    if [[ ! -f "$DOMAIN_MAP_FILE" ]]; then
        echo "ERROR: missing domain map file for bridged mode: ${DOMAIN_MAP_FILE}"
        exit 1
    fi
    echo "  ROS_DOMAIN_ID   = ${ROS_DOMAIN_ID}"
    echo "  Domain map      = ${DOMAIN_MAP_FILE}"
    if (( USING_ZENOH_TRANSPORT == 1 )); then
        echo "  Zenoh transport = yes (RMW=rmw_cyclonedds_cpp, ROS_LOCALHOST_ONLY=1)"
    fi
fi
echo "  Domain source   = ${DOMAIN_SOURCE}"
if [[ -n "$SELECTION" ]]; then
    echo "  Robot filter    = -$SELECTION  (b=blinky, p=pinky, i=inky, c=clyde)"
else
    echo "  Robot filter    = (all detected robots)"
fi
echo ""

# Control whether the explorer falls back to publishing PoseStamped goals on
# /<robot>/goal_pose when the NavigateToPose action server is unavailable.
# Default is "false" so that Nav2 action usage is required and failures are
# surfaced clearly. Override by setting EXPLORER_USE_GOAL_POSE_FALLBACK=true
# before running this script if you need the legacy behaviour.
EXPLORER_USE_GOAL_POSE_FALLBACK="${EXPLORER_USE_GOAL_POSE_FALLBACK:-false}"
SAFE_NODE_PLANNER_ENABLE="${SAFE_NODE_PLANNER_ENABLE:-false}"
EXPLORER_FREQUENCY_HZ="${EXPLORER_FREQUENCY_HZ:-}"
EXPLORER_TERMINAL_SUMMARY_ENABLE="${EXPLORER_TERMINAL_SUMMARY_ENABLE:-}"
EXPLORER_TERMINAL_SUMMARY_PERIOD_SEC="${EXPLORER_TERMINAL_SUMMARY_PERIOD_SEC:-}"
EXPLORER_TERMINAL_EVENT_LOG_MODE="${EXPLORER_TERMINAL_EVENT_LOG_MODE:-}"
if is_clean_profile; then
    echo "  Explorer fallback = ${EXPLORER_USE_GOAL_POSE_FALLBACK}"
else
    echo "  Explorer fallback = use_pose_goal_fallback=${EXPLORER_USE_GOAL_POSE_FALLBACK}"
fi
echo "  Safe-node planner = ${SAFE_NODE_PLANNER_ENABLE}"
if [[ -n "$EXPLORER_FREQUENCY_HZ" ]]; then
    echo "  Explorer frequency = ${EXPLORER_FREQUENCY_HZ} Hz (override)"
elif ! is_clean_profile; then
    echo "  Explorer frequency override = (using YAML default)"
fi
if [[ -f "$BRIDGE_CONTRACT_FILE" ]]; then
    echo "  Bridge contract   = ${BRIDGE_CONTRACT_FILE}"
fi
if is_clean_profile; then
    echo "  Robot mode hint   = prefer fleet_mode:=false unless global /tf + /map is required"
else
    echo "  Recommended robot mode for stability: fleet_mode:=false (local-first)"
    echo "  Use fleet_mode:=true only when central-coupled global /tf + /map is needed."
fi
if [[ -n "$EXPLORER_TERMINAL_SUMMARY_ENABLE" || -n "$EXPLORER_TERMINAL_SUMMARY_PERIOD_SEC" || -n "$EXPLORER_TERMINAL_EVENT_LOG_MODE" || "$CENTRAL_MAP_MERGE_RAW_STDOUT" == "true" ]]; then
    echo "  Runtime knobs:"
    [[ -n "$EXPLORER_TERMINAL_SUMMARY_ENABLE" ]] && echo "    summary_enable=${EXPLORER_TERMINAL_SUMMARY_ENABLE}"
    [[ -n "$EXPLORER_TERMINAL_SUMMARY_PERIOD_SEC" ]] && echo "    summary_period_sec=${EXPLORER_TERMINAL_SUMMARY_PERIOD_SEC}"
    [[ -n "$EXPLORER_TERMINAL_EVENT_LOG_MODE" ]] && echo "    event_log_mode=${EXPLORER_TERMINAL_EVENT_LOG_MODE}"
    [[ "$CENTRAL_MAP_MERGE_RAW_STDOUT" == "true" ]] && echo "    map_merge_raw_stdout=true"
fi
echo ""

# Prevent multiple central stacks from running at once. If you start this
# script multiple times (e.g., in different terminals), you'll end up with
# duplicate nodes with the same name publishing conflicting status and
# sending duplicate Nav2 goals.
#
# We validate this using OS processes (authoritative) rather than only the
# ROS graph, because the ROS graph can temporarily retain nodes after
# crashes/network hiccups.
ensure_no_central_stack_running() {
    # Match only central-computer processes launched by this script:
    #  - tf_relay_multirobot.py
    #  - single_robot_world_tf_bridge.py (single-robot mode)
    #  - single_robot_map_relay.py (single-robot mode)
    #  - multi_robot_explorer.py using the central params file
    #  - fleet_navigate_to_pose_service_relay.py (bridged_domains only)
    local pattern_tf="python3 .*tf_relay_multirobot\\.py"
    local pattern_bridge="python3 .*single_robot_world_tf_bridge\\.py"
    local pattern_map_relay="python3 .*single_robot_map_relay\\.py"
    local pattern_explorer="python3 .*multi_robot_explorer\\.py .*--params-file .*multi_robot_explorer\\.yaml"
    local pattern_domain_bridge="domain_bridge .*\\.domain_bridge\\.yaml"
    local pattern_action_relay="python3 .*fleet_navigate_to_pose_service_relay\\.py"
    local pattern_safe_node="python3 .*safe_node_planner\\.py"

    # Collect matching PIDs + commands (skip the grep processes themselves).
    local existing
    existing="$(ps aux | grep -E "$pattern_tf|$pattern_bridge|$pattern_map_relay|$pattern_explorer|$pattern_domain_bridge|$pattern_action_relay|$pattern_safe_node" | grep -v grep || true)"

    if [[ -z "$existing" ]]; then
        return 0
    fi

    echo "ERROR: A central stack appears to already be running (found existing processes):"
    echo "$existing"
    echo ""
    echo "Typical causes:"
    echo "  - An earlier run of this script is still active in another terminal or tmux pane."
    echo "  - A previous run crashed or the terminal was closed without stopping the processes."
    echo ""

    # In non-interactive shells (e.g., launched from another script), do not
    # attempt to prompt; just fail safe unless explicitly overridden.
    if [[ ! -t 0 && "${CENTRAL_AUTO_KILL:-false}" != "true" ]]; then
        echo "This shell is non-interactive; not killing processes automatically."
        echo "To clean up manually, you can run for example:"
        echo "  ps aux | grep multi_robot_explorer.py | grep -v grep"
        echo "  kill <pid>"
        echo ""
        echo "Alternatively, re-run this script from an interactive terminal"
        echo "to be prompted to kill the detected processes, or set"
        echo "CENTRAL_AUTO_KILL=true if you understand the risks."
        exit 1
    fi

    echo "The following PID(s) and command lines were detected:"
    echo ""
    # Pretty-print: PID and full command.
    # ps aux format: USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND
    echo "$existing" | while read -r user pid rest; do
        # 'rest' starts with %CPU; we want the command portion at the end.
        # For clarity we just echo the original line prefixed with PID.
        echo "  PID ${pid}: ${user} ${rest}"
    done
    echo ""

    # PIDs from ps aux (column 2). Sort -u in case multiple lines match one process.
    local pids_raw
    pids_raw="$(echo "$existing" | awk '{print $2}' | sort -u)"

    kill_matching_central_pids() {
        # Multiprocessing relays (e.g. fleet_navigate_to_pose_service_relay) may
        # ignore SIGTERM while blocked in rclpy; escalate to SIGKILL. Kill known
        # children first so the parent can reap where possible.
        local pid
        for pid in $pids_raw; do
            [[ -n "$pid" ]] || continue
            local child
            for child in $(pgrep -P "$pid" 2>/dev/null || true); do
                kill -TERM "$child" 2>/dev/null || true
            done
            kill -TERM "$pid" 2>/dev/null || true
        done
        sleep 2
        for pid in $pids_raw; do
            [[ -n "$pid" ]] || continue
            if kill -0 "$pid" 2>/dev/null; then
                for child in $(pgrep -P "$pid" 2>/dev/null || true); do
                    kill -KILL "$child" 2>/dev/null || true
                done
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
    }

    # If non-interactive auto-kill is explicitly enabled, skip prompt.
    if [[ "${CENTRAL_AUTO_KILL:-false}" == "true" && ! -t 0 ]]; then
        echo "CENTRAL_AUTO_KILL=true and non-interactive shell detected — killing matching processes without prompting..."
        kill_matching_central_pids
    else
        read -r -p "Kill these processes and continue? [y/N] " reply
        if [[ "$reply" != "y" && "$reply" != "Y" ]]; then
            echo "Aborting without killing any processes."
            exit 1
        fi
        echo "Killing matching processes (SIGTERM, then SIGKILL if needed)..."
        kill_matching_central_pids
    fi

    # Give the OS a moment, then re-check.
    sleep 1
    local remaining
    remaining="$(ps aux | grep -E "$pattern_tf|$pattern_bridge|$pattern_map_relay|$pattern_explorer|$pattern_domain_bridge|$pattern_action_relay|$pattern_safe_node" | grep -v grep || true)"
    if [[ -n "$remaining" ]]; then
        echo "WARNING: Some central processes still appear to be running:"
        echo "$remaining"
        echo "You may need to inspect and kill these manually before re-running."
        exit 1
    fi

    echo "All matching central processes have been terminated. Continuing startup..."
    echo ""
}

ensure_no_central_stack_running

# If the graph still shows these nodes, warn but continue.
NODES_RAW="$(ros2 node list 2>/dev/null || true)"
if [[ -n "$NODES_RAW" ]]; then
    explorer_count="$(echo "$NODES_RAW" | grep -c '^/multi_robot_explorer$' || true)"
    tf_relay_count="$(echo "$NODES_RAW" | grep -c '^/tf_relay_multirobot$' || true)"
    if (( explorer_count > 0 || tf_relay_count > 0 )); then
        echo "WARNING: ROS graph currently includes central node names already:"
        echo "  /multi_robot_explorer count = ${explorer_count}"
        echo "  /tf_relay_multirobot count  = ${tf_relay_count}"
        echo "Continuing because no matching OS processes were found."
        echo ""
    fi
fi

# Detect/select robots.
if [[ "$COMMS_MODE" == "shared_domain" ]]; then
    echo "Detecting robots from ROS topics..."
else
    echo "Selecting robots from domain map (bridged mode)..."
fi

# Cross-domain DDS discovery from the central PC often needs >5s on Wi‑Fi.
CENTRAL_DISCOVERY_TIMEOUT_SEC="${CENTRAL_DISCOVERY_TIMEOUT_SEC:-25}"
CENTRAL_DISCOVERY_POLL_SEC="${CENTRAL_DISCOVERY_POLL_SEC:-1}"
CENTRAL_MIN_ROBOTS_DEFAULT="${CENTRAL_MIN_ROBOTS_DEFAULT:-2}"
if ! [[ "$CENTRAL_DISCOVERY_TIMEOUT_SEC" =~ ^[0-9]+$ ]]; then
    CENTRAL_DISCOVERY_TIMEOUT_SEC=20
fi
if ! [[ "$CENTRAL_DISCOVERY_POLL_SEC" =~ ^[0-9]+$ ]]; then
    CENTRAL_DISCOVERY_POLL_SEC=1
fi
if (( CENTRAL_DISCOVERY_POLL_SEC < 1 )); then
    CENTRAL_DISCOVERY_POLL_SEC=1
fi

detect_robots_once() {
    local topics_raw="$1"
    local detected=()
    if [[ -n "$topics_raw" ]]; then
        while IFS= read -r line; do
            name="${line#/}"
            name="${name%%/*}"
            if [[ -n "$name" ]]; then
                detected+=("$name")
            fi
        done < <(echo "$topics_raw" | grep '^/[^/]\+/tf$' || true)
    fi

    if ((${#detected[@]} == 0)); then
        if [[ -n "$topics_raw" ]]; then
            while IFS= read -r line; do
                name="${line#/}"
                name="${name%%/*}"
                if [[ -n "$name" ]]; then
                    detected+=("$name")
                fi
            done < <(echo "$topics_raw" | grep '^/[^/]\+/map$' || true)
        fi
    fi

    # De-duplicate + sort for stable ordering.
    local uniq=()
    for r in "${detected[@]}"; do
        local skip=false
        for u in "${uniq[@]}"; do
            if [[ "$u" == "$r" ]]; then
                skip=true
                break
            fi
        done
        if [[ "$skip" == false ]]; then
            uniq+=("$r")
        fi
    done
    if ((${#uniq[@]} > 0)); then
        printf '%s\n' "${uniq[@]}" | sort
    fi
}

detect_active_robots_bridged() {
    local selection_letters="$1"
    local timeout_sec="$2"
    local poll_sec="$3"
    local robot_domain_lines

    robot_domain_lines="$(python3 - <<'PY' "$DOMAIN_MAP_FILE"
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
robot_map = data.get("fleet_domain_map", {}).get("robot_domain_ids", {})
for name, domain in sorted(robot_map.items()):
    print(f"{name} {domain}")
PY
)"

    local candidates=()
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        local name domain
        name="$(echo "$line" | awk '{print $1}')"
        domain="$(echo "$line" | awk '{print $2}')"
        if [[ -n "$selection_letters" ]]; then
            [[ -n "${SELECTED_ROBOT_SET[$name]:-}" ]] || continue
        fi
        candidates+=("${name}:${domain}")
    done <<< "$robot_domain_lines"

    if ((${#candidates[@]} == 0)); then
        return 0
    fi

    local active=()
    local elapsed=0
    while :; do
        active=()
        for entry in "${candidates[@]}"; do
            local name="${entry%%:*}"
            local domain="${entry##*:}"
            local topics_raw
            # Keep discovery env explicit so Zenoh localhost injection is visible.
            topics_raw="$(
                ROS_DOMAIN_ID="$domain" \
                RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-}" \
                ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-}" \
                env -u CYCLONEDDS_URI \
                ros2 topic list 2>/dev/null || true
            )"
            # tf/map are canonical; map_wire_z is the fleet bridge side channel and
            # appears as soon as SLAM+map_wire republisher are up on the robot.
            if echo "$topics_raw" | grep -Eq "^/${name}/(tf|map|map_wire_z)$"; then
                active+=("$name")
            fi
        done

        if ((${#active[@]} >= MIN_EXPECTED_ROBOTS)); then
            if ((${#active[@]} > 0)); then
                printf '%s\n' "${active[@]}" | sort -u
            fi
            return 0
        fi
        if (( elapsed >= timeout_sec )); then
            if ((${#active[@]} > 0)); then
                printf '%s\n' "${active[@]}" | sort -u
            fi
            return 0
        fi
        if (( elapsed == 0 )); then
            if [[ -n "$selection_letters" ]]; then
                echo "Waiting up to ${timeout_sec}s for selected bridged robots (-${selection_letters}) to publish /<robot>/(tf|map|map_wire_z)..." >&2
            else
                echo "Waiting up to ${timeout_sec}s for bridged robots to publish /<robot>/(tf|map|map_wire_z)..." >&2
            fi
        fi
        sleep "$poll_sec"
        elapsed=$((elapsed + poll_sec))
    done
}

# Determine how many robots should be visible before finalizing mode.
if [[ -n "$SELECTION" ]]; then
    MIN_EXPECTED_ROBOTS="${#SELECTED_ROBOT_SET[@]}"
else
    MIN_EXPECTED_ROBOTS="${CENTRAL_MIN_ROBOTS_DEFAULT}"
fi
if ! [[ "$MIN_EXPECTED_ROBOTS" =~ ^[0-9]+$ ]]; then
    MIN_EXPECTED_ROBOTS=1
fi
if (( MIN_EXPECTED_ROBOTS < 1 )); then
    MIN_EXPECTED_ROBOTS=1
fi

DETECTED_ROBOTS=()
if [[ "$COMMS_MODE" == "shared_domain" ]]; then
    elapsed=0
    while :; do
        TOPICS_RAW="$(ros2 topic list 2>/dev/null || true)"
        readarray -t DETECTED_ROBOTS < <(detect_robots_once "$TOPICS_RAW")

        if ((${#DETECTED_ROBOTS[@]} >= MIN_EXPECTED_ROBOTS)); then
            break
        fi

        if (( elapsed >= CENTRAL_DISCOVERY_TIMEOUT_SEC )); then
            break
        fi

        if (( elapsed == 0 )); then
            if [[ -n "$SELECTION" ]]; then
                echo "Waiting up to ${CENTRAL_DISCOVERY_TIMEOUT_SEC}s for selected robots (-${SELECTION}) to appear..."
            else
                echo "Waiting up to ${CENTRAL_DISCOVERY_TIMEOUT_SEC}s for >=${MIN_EXPECTED_ROBOTS} robots to appear..."
            fi
        fi
        sleep "$CENTRAL_DISCOVERY_POLL_SEC"
        elapsed=$((elapsed + CENTRAL_DISCOVERY_POLL_SEC))
    done
else
    readarray -t DETECTED_ROBOTS < <(
        detect_active_robots_bridged \
            "$SELECTION" \
            "$CENTRAL_DISCOVERY_TIMEOUT_SEC" \
            "$CENTRAL_DISCOVERY_POLL_SEC"
    )
fi

if ((${#DETECTED_ROBOTS[@]} == 0)); then
    echo "ERROR: Could not detect any robot namespaces from topics."
    echo "Make sure at least one robot is running its namespaced bringup + Nav2/SLAM."
    if [[ "$COMMS_MODE" == "bridged_domains" ]]; then
        echo ""
        echo "Bridged mode: the central PC runs \`ros2 topic list\` on each robot's ROS_DOMAIN_ID"
        echo "from ${DOMAIN_MAP_FILE} and looks for /<robot>/(tf|map|map_wire_z)."
        if (( USING_ZENOH_TRANSPORT == 1 )); then
            echo "Zenoh transport detected: discovery uses RMW=rmw_cyclonedds_cpp and ROS_LOCALHOST_ONLY=1."
            echo "Quick check (example Clyde domain 80 via Zenoh):"
            echo "  ROS_DOMAIN_ID=80 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOCALHOST_ONLY=1 ros2 topic list | grep clyde"
            echo "If empty: start bringup+SLAM on the robot, keep Zenoh central + robot bridges running,"
            echo "then re-check. Zenoh peer link alone does not create /<robot>/map."
        else
            echo "If robots are up but this fails, DDS on the central machine cannot see those domains"
            echo "(Wi‑Fi client isolation, VLANs, firewall, or mismatched ROS_LOCALHOST_ONLY)."
            echo "On TAMU_WiFi start Zenoh first: ./scripts/comms/start_zenoh_central.sh"
            echo "Quick check from central (example pinky on domain 22):"
            echo "  ROS_DOMAIN_ID=22 ros2 topic list | grep -E '^/pinky/(tf|map|map_wire_z)\$'"
        fi
        echo "Increase wait: CENTRAL_DISCOVERY_TIMEOUT_SEC=60 ./scripts/core/start_central.sh --comms-mode bridged_domains"
    fi
    exit 1
fi

# Apply selection filter from selector letters to full robot names.
SELECTED_ROBOTS=()
for r in "${DETECTED_ROBOTS[@]}"; do
    [[ -n "$r" ]] || continue
    if [[ -z "$SELECTION" ]]; then
        SELECTED_ROBOTS+=("$r")
    else
        if [[ -n "${SELECTED_ROBOT_SET[$r]:-}" ]]; then
            SELECTED_ROBOTS+=("$r")
        fi
    fi
done

if ((${#SELECTED_ROBOTS[@]} == 0)); then
    echo "ERROR: No robots matched selection '-$SELECTION'."
    echo "Detected robots: ${DETECTED_ROBOTS[*]}"
    exit 1
fi

echo "Detected robots: ${DETECTED_ROBOTS[*]}"
echo "Using robots   : ${SELECTED_ROBOTS[*]}"
echo ""

if ((${#SELECTED_ROBOTS[@]} == 1)); then
    MODE_DESC="single-robot (Nav2 on robot, no map_merge)"
else
    MODE_DESC="multi-robot (Nav2 on robots, map_merge enabled)"
fi
echo "  Mode          = ${MODE_DESC}"
echo ""

# Build a ROS 2 list parameter value like "[blinky,pinky]".
# This is passed via --ros-args -p to Python nodes that expect string arrays.
ROBOT_LIST_PARAM=""
for r in "${SELECTED_ROBOTS[@]}"; do
    if [[ -n "$ROBOT_LIST_PARAM" ]]; then
        ROBOT_LIST_PARAM+=","
    fi
    ROBOT_LIST_PARAM+="$r"
done

PIDS=()
CLEANUP_DONE=false
cleanup() {
    if [[ "$CLEANUP_DONE" == "true" ]]; then
        return
    fi
    CLEANUP_DONE=true
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait 2>/dev/null
    echo "Done."
}
trap cleanup SIGHUP SIGINT SIGTERM EXIT

start_domain_bridges() {
    if [[ "$COMMS_MODE" != "bridged_domains" ]]; then
        return 0
    fi
    if ! command -v ros2 >/dev/null 2>&1; then
        echo "ERROR: ros2 CLI not found for bridged mode"
        exit 1
    fi
    if [[ ! -f "$BRIDGE_CONFIG_GENERATOR" ]]; then
        echo "ERROR: bridge config generator missing: $BRIDGE_CONFIG_GENERATOR"
        exit 1
    fi

    # Central domain must not collide with any robot-assigned domain in the map.
    collision_robot="$(
        python3 - <<'PY' "$DOMAIN_MAP_FILE" "$ROS_DOMAIN_ID"
import sys, yaml
domain_map_path, central_domain = sys.argv[1], int(sys.argv[2])
with open(domain_map_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
robot_map = data.get("fleet_domain_map", {}).get("robot_domain_ids", {}) or {}
for name, dom in robot_map.items():
    if int(dom) == central_domain:
        print(name)
        break
PY
    )"
    if [[ -n "$collision_robot" ]]; then
        echo "ERROR: Central ROS_DOMAIN_ID=${ROS_DOMAIN_ID} collides with robot '${collision_robot}' domain."
        echo "Choose a non-robot domain (or pass --domain-id <id>)."
        exit 1
    fi

    echo "Generating domain bridge configs..."
    python3 "$BRIDGE_CONFIG_GENERATOR" \
        --domain-map "$DOMAIN_MAP_FILE" \
        --bridge-contract "$BRIDGE_CONTRACT_FILE" \
        --robots "${SELECTED_ROBOTS[@]}" \
        --output-dir "$GENERATED_BRIDGE_CONFIG_DIR" \
        --central-domain "$ROS_DOMAIN_ID"

    echo "Starting domain_bridge processes..."
    for robot in "${SELECTED_ROBOTS[@]}"; do
        cfg="${GENERATED_BRIDGE_CONFIG_DIR}/${robot}.domain_bridge.yaml"
        if [[ ! -f "$cfg" ]]; then
            echo "ERROR: generated bridge config missing: $cfg"
            exit 1
        fi
        ros2 run domain_bridge domain_bridge "$cfg" &
        PIDS+=($!)
    done

    bridge_wait_timeout="${BRIDGE_READY_TIMEOUT_SEC:-15}"
    if ! [[ "$bridge_wait_timeout" =~ ^[0-9]+$ ]]; then
        bridge_wait_timeout=15
    fi
    echo "Waiting up to ${bridge_wait_timeout}s for bridge outputs..."
    elapsed=0
    while (( elapsed <= bridge_wait_timeout )); do
        topics_now="$(ros2 topic list 2>/dev/null || true)"
        all_ready=true
        for robot in "${SELECTED_ROBOTS[@]}"; do
            # local-first robots may only expose /<robot>/map (no map_wire_z).
            if ! echo "$topics_now" | grep -Eq "^/${robot}/(map_wire_z|map|tf)$"; then
                all_ready=false
                break
            fi
        done
        if [[ "$all_ready" == "true" ]]; then
            echo "Bridge ready."
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    echo "WARNING: bridge readiness timeout reached; continuing startup."
}

start_domain_bridges

# ---- 0c. Nav2 action *services* relay (bridged_domains only) ----
# domain_bridge YAML loads topic bridges only; action clients need
# send_goal/get_result/cancel_goal on central for navigate_to_pose and
# compute_path_to_pose (explorer path precheck).
start_nav_action_relays() {
    if [[ "$COMMS_MODE" != "bridged_domains" ]]; then
        return 0
    fi
    if [[ ! -f "$ACTION_RELAY_SCRIPT" ]]; then
        echo "ERROR: NavigateToPose relay missing: ${ACTION_RELAY_SCRIPT}"
        exit 1
    fi
    echo "[0c] Starting Nav2 action service relays (navigate_to_pose + compute_path_to_pose)..."
    for robot in "${SELECTED_ROBOTS[@]}"; do
        robot_domain_id="$(
            python3 -c "
import sys, yaml
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f) or {}
rid = data.get('fleet_domain_map', {}).get('robot_domain_ids', {}).get(sys.argv[2])
if rid is None:
    print('ERROR: robot not found in fleet_domain_map.yaml', file=sys.stderr)
    sys.exit(1)
print(int(rid))
" "$DOMAIN_MAP_FILE" "$robot"
        )"
        python3 "$ACTION_RELAY_SCRIPT" \
            --robot "$robot" \
            --robot-domain "$robot_domain_id" \
            --central-domain "$ROS_DOMAIN_ID" &
        PIDS+=($!)
    done
    sleep 1
}

start_nav_action_relays

append_explorer_log_args() {
    # Keep logging overrides centralized so both single-robot and multi-robot
    # explorer launches share exactly the same env-driven behavior.
    if [[ -n "$EXPLORER_TERMINAL_SUMMARY_ENABLE" ]]; then
        EXPLORER_ARGS+=(-p "terminal_summary_enable:=${EXPLORER_TERMINAL_SUMMARY_ENABLE}")
    fi
    if [[ -n "$EXPLORER_TERMINAL_SUMMARY_PERIOD_SEC" ]]; then
        EXPLORER_ARGS+=(-p "terminal_summary_period_sec:=${EXPLORER_TERMINAL_SUMMARY_PERIOD_SEC}")
    fi
    if [[ -n "$EXPLORER_TERMINAL_EVENT_LOG_MODE" ]]; then
        EXPLORER_ARGS+=(-p "terminal_event_log_mode:=${EXPLORER_TERMINAL_EVENT_LOG_MODE}")
    fi
}

start_map_merge_filtered() {
    if [[ "$CENTRAL_MAP_MERGE_RAW_STDOUT" == "true" ]]; then
        ros2 run multirobot_map_merge map_merge --ros-args \
            --params-file "${MAP_MERGE_CONFIG_FILE}" &
        PIDS+=($!)
        return
    fi

    # Suppress non-ROS matrix/feature dumps emitted on stdout by OpenCV stages
    # in map_merge while retaining regular ROS logger lines.
    ros2 run multirobot_map_merge map_merge --ros-args \
        --params-file "${MAP_MERGE_CONFIG_FILE}" 2>&1 | \
        python3 -u -c '
import re, sys
drop = [
    re.compile(r"^[0-9]+\s+[0-9]+$"),
    re.compile(r"^features:\s"),
    re.compile(r"^matches:\s"),
    re.compile(r"^inliers:\s"),
    re.compile(r"^inliers/matches ratio:\s"),
    re.compile(r"^confidence:\s"),
    re.compile(r"^\[\s*[-0-9.]"),
    re.compile(r"^\s*[-0-9.]+,\s*[-0-9.]+,\s*[-0-9.]+;?$"),
    re.compile(r"^\s*0,\s*0,\s*1\]$"),
]
for line in sys.stdin:
    txt = line.rstrip("\n")
    if any(p.search(txt) for p in drop):
        continue
    print(txt, flush=True)
' &
    PIDS+=($!)
}

# ---- 1. TF relay (same policy for 1 or N robots) ----
echo "[1/3] Starting TF relay (prefix_frames=true)..."
python3 "${TF_RELAY_SCRIPT}" --ros-args \
    -p "robot_prefixes:=[${ROBOT_LIST_PARAM}]" \
    -p "prefix_frames:=true" &
PIDS+=($!)
sleep 1

if ((${#SELECTED_ROBOTS[@]} == 1)); then
    SINGLE_ROBOT_BRIDGE="${SELECTED_ROBOTS[0]}"
    echo "[1b/3] Single-robot mode — static TF map -> ${SINGLE_ROBOT_BRIDGE}/map (identity)"
    python3 "${SINGLE_ROBOT_WORLD_TF_SCRIPT}" --ros-args \
        -p "robot_name:=${SINGLE_ROBOT_BRIDGE}" &
    PIDS+=($!)
    sleep 0.5
    echo "[1c/3] Single-robot mode — relay /${SINGLE_ROBOT_BRIDGE}/map_wire_z -> /map (zlib wire)"
    python3 "${SINGLE_ROBOT_MAP_RELAY_SCRIPT}" --ros-args \
        -p "source_map:=/${SINGLE_ROBOT_BRIDGE}/map" \
        -p "source_map_wire_z:=/${SINGLE_ROBOT_BRIDGE}/map_wire_z" &
    PIDS+=($!)
    sleep 0.5
fi

if ((${#SELECTED_ROBOTS[@]} == 1)); then
    # ------------------------------------------------------------------
    # Single-robot setup:
    #   - Skip map_merge.
    #   - Treat the robot's /map in frame 'map' as the world map
    #     (world_frame='map').
    # ------------------------------------------------------------------
    SINGLE_ROBOT="${SELECTED_ROBOTS[0]}"
    echo "[2/3] Single-robot setup detected (${SINGLE_ROBOT}) — skipping map merge."
    echo "[3/3] Starting single-robot explorer (Nav2 offloaded to robot)..."
    EXPLORER_ARGS=(
        "${EXPLORER_SCRIPT}"
        --ros-args
        --params-file "${EXPLORE_CONFIG_FILE}"
        -p "robot_names:=[${SINGLE_ROBOT}]"
        -p "map_topic:=/${SINGLE_ROBOT}/map"
        -p "world_frame:=map"
        -p "single_robot_offloaded_nav2:=true"
        -p "use_pose_goal_fallback:=${EXPLORER_USE_GOAL_POSE_FALLBACK}"
    )
    if [[ -n "$EXPLORER_FREQUENCY_HZ" ]]; then
        EXPLORER_ARGS+=(-p "explore_frequency:=${EXPLORER_FREQUENCY_HZ}")
    fi
    append_explorer_log_args
    python3 "${EXPLORER_ARGS[@]}" &
    PIDS+=($!)
else
    # ---- 2. Map merge ----
    echo "[2/3] Starting map merge (unknown poses)..."
    start_map_merge_filtered
    sleep 2
    echo ""
    echo "  map_merge: a sparse scan can skip feature matching for that cycle."
    echo "  Pose estimation keeps running and retries on the next map."
    echo ""

    # ---- 2b. Map merge state monitor ----
    # This helper publishes a coarse merge state string (NO_OVERLAP, PARTIAL,
    # MERGED) on map_merge/merge_state so the explorer can automatically switch
    # from per-robot local exploration to global merged exploration.
    python3 "${WORKSPACE_DIR}/scripts/diagnostics/map_merge_state_monitor.py" --ros-args \
        -p "robot_names:=[${ROBOT_LIST_PARAM}]" \
        -p "world_frame:=map" \
        -p "stability_pos_threshold:=0.28" \
        -p "merged_dwell_time:=20.0" \
        -p "motion_jitter_confirm_ticks:=3" \
        -p "min_known_cells_per_robot:=450" \
        -p "min_known_area_m2_per_robot:=1.1" &
    PIDS+=($!)

    # ---- 3. Explorer ----
    echo "[3/3] Starting multi-robot explorer..."
    EXPLORER_ARGS=(
        "${EXPLORER_SCRIPT}"
        --ros-args
        --params-file "${EXPLORE_CONFIG_FILE}"
        -p "robot_names:=[${ROBOT_LIST_PARAM}]"
        -p "use_pose_goal_fallback:=${EXPLORER_USE_GOAL_POSE_FALLBACK}"
    )
    if [[ -n "$EXPLORER_FREQUENCY_HZ" ]]; then
        EXPLORER_ARGS+=(-p "explore_frequency:=${EXPLORER_FREQUENCY_HZ}")
    fi
    append_explorer_log_args
    python3 "${EXPLORER_ARGS[@]}" &
    PIDS+=($!)
fi

# ---- 4. Safe-node planner (optional) ----
# Reads /map, /<robot>/wifi_quality and TF; publishes /<robot>/safe_node
# (link heartbeat for the robot's link_watchdog.py) and /<robot>/connection_risk.
if [[ "$SAFE_NODE_PLANNER_ENABLE" == "true" ]]; then
    echo "[4] Starting safe-node planner..."
    SAFE_NODE_ARGS=(-p "robot_names:=[${ROBOT_LIST_PARAM}]")
    if ((${#SELECTED_ROBOTS[@]} == 1)); then
        # Same map as the single-robot explorer (map -> <robot>/map is identity).
        SAFE_NODE_ARGS+=(-p "map_topic:=/${SELECTED_ROBOTS[0]}/map")
    fi
    python3 "${SAFE_NODE_PLANNER_SCRIPT}" --ros-args "${SAFE_NODE_ARGS[@]}" &
    PIDS+=($!)
fi

echo ""
echo "=========================================="
echo "  All services running.  Press Ctrl+C to stop."
echo "=========================================="
echo ""
echo "  To visualise: rviz2  (add /map display, set frame to 'map')"
echo ""

wait

#!/usr/bin/env python3
"""
Multi-robot frontier exploration coordinator.

Subscribes to the merged global map, detects frontiers, and assigns
exploration waypoints to multiple robots via their Nav2 action servers.

Each planning cycle:
  1. Detect frontier regions on the merged occupancy grid.
  2. For every idle robot, pick the best unassigned frontier using a
     utility function that balances proximity vs information gain.
  3. Send a NavigateToPose goal: frontiers are chosen in the merged world
     ``map`` frame, then (by default) TF-transformed into ``<robot>/map``
     for Nav2 when ``dispatch_nav_goals_in_robot_map_frame`` is enabled
     (Path 2: local SLAM costmaps on each robot).

Handles the no-overlap case transparently: the merged map already
contains all robot maps (placed side-by-side by map_merge), so each
robot sees frontiers on its own portion and explores independently
until map overlap is detected.
"""

import math
import re
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

from action_msgs.msg import GoalStatus
from std_msgs.msg import Bool, String
from rcl_interfaces.msg import Log
from lifecycle_msgs.msg import TransitionEvent
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import String as StringMsg
from nav2_msgs.action import NavigateToPose, ComputePathToPose
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PoseStamped for Buffer.transform()
from central_explorer_event_logger import ExplorerEventLogger


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Frontier:
    centroid_world: Tuple[float, float]  # (x, y) in pose_frame_id (metres)
    size: int                            # number of frontier cells
    size_m: float                        # size * resolution (metres)
    cells: int = 0                       # alias for size
    indices: Optional[np.ndarray] = None  # (N, 2) array of (y, x) cell indices
    # TF frame for centroid_world / goal points (merged map or /<robot>/map).
    pose_frame_id: Optional[str] = None


@dataclass
class RobotState:
    name: str
    action_client: ActionClient = None
    goal_handle: object = None
    goal_position: Optional[Tuple[float, float]] = None
    last_goal_time: float = 0.0
    goal_active: bool = False
    goal_pending: bool = False
    goal_status: str = 'none'
    position: Optional[Tuple[float, float]] = None  # current (x,y) in world
    # Yaw (rad) from map -> base_footprint TF; used for arrival path probes.
    yaw: Optional[float] = None
    goals_reached: int = 0
    goals_failed: int = 0
    blacklist: list = field(default_factory=list)
    server_unavailable_logged: bool = False
    last_distance_remaining: Optional[float] = None
    last_progress_distance: Optional[float] = None
    last_progress_time: float = 0.0
    active_goal_sent_time: float = 0.0
    # Monotonic per-robot sequence for NavigateToPose goals sent by explorer.
    goal_seq_counter: int = 0
    # Sequence id for the currently active/pending goal callbacks.
    active_goal_seq: int = 0
    last_pose_when_goal_sent: Optional[Tuple[float, float]] = None
    # Straight-line distance robot→goal; used to detect motion toward goal when
    # Nav2 feedback is sparse so the stall watchdog does not cancel healthy drives.
    last_dist_to_goal: Optional[float] = None
    # Target frontier (centroid in map frame) for the most recent send attempt.
    # Used after Nav2 success to avoid immediately re-choosing the same frontier
    # while the map has not yet changed.
    last_target_frontier_centroid: Optional[Tuple[float, float]] = None
    # Track behaviour around repeated / stagnant goals. last_goal_world stores
    # the most recent goal position in world coordinates (x, y).
    last_goal_world: Optional[Tuple[float, float]] = None
    repeat_goal_count: int = 0
    # Short history of recent goals and associated frontier sizes for
    # stagnation detection. Each entry is (x, y, frontier_size_m).
    recent_goals: deque = field(default_factory=lambda: deque(maxlen=10))
    last_retarget_time: float = 0.0
    # Silence new assignments until this time (unix sec) after abort/cancel watchdog.
    next_assign_allowed_time: float = 0.0
    # When True, the next CANCELED result from Nav2 will not arm post-failure
    # cooldown (used when we cancel for retarget and immediately send a replacement
    # goal). ABORTED-as-cancel is gated on retarget_abort_blacklist_suppress_seq.
    ignore_cooldown_on_next_cancel: bool = False
    # NavigateToPose goal_seq for the replacement goal after retarget; Nav2 may
    # report ABORTED while tearing down the cancelled goal — do not blacklist.
    retarget_abort_blacklist_suppress_seq: Optional[int] = None
    # Reason for the most recent explorer-issued cancel request.
    last_cancel_reason: str = ''
    # First NavigateToPose of the current leg; retargets must stay near this.
    retarget_anchor: Optional[Tuple[float, float]] = None
    # Set when we cancel Nav2 after tacit "close enough" success (see plan_tick).
    tacit_success_pending_cancel: bool = False
    # First time (unix s) the robot was within tacit_goal_success_radius_m of goal.
    near_goal_since: Optional[float] = None
    # Time when NavigateToPose action server was first seen available.
    action_server_ready_time: float = 0.0
    # Last sampled world-frame pose used to detect physical motion.
    last_motion_sample_pose: Optional[Tuple[float, float]] = None
    # Last time (unix sec) TF pose was updated successfully.
    last_tf_update_time: float = 0.0
    # Throttle repeated TF stale warnings.
    tf_stale_warned: bool = False
    # Time of last_motion_sample_pose.
    last_motion_sample_time: float = 0.0
    # Last time meaningful physical movement (>= motion_progress_min_delta_m)
    # was observed from TF poses.
    last_motion_time: float = 0.0
    # True while Nav2 costmap marks the robot position as lethal/in-collision.
    nav2_lethal_active: bool = False
    # First timestamp when nav2_lethal_active became True.
    nav2_lethal_since: float = 0.0
    # True while robot-side nav2_retrace_escape is executing retreat.
    nav2_retrace_active: bool = False
    # First timestamp when nav2_retrace_active became True.
    nav2_retrace_since: float = 0.0
    # True while /<robot>/link_retreat_active (link_watchdog) is true.
    link_retreat_active: bool = False
    # Last goal blacklisted because it was cancelled far from target.
    last_cancel_blacklist_time: float = 0.0
    last_cancel_blacklist_point: Optional[Tuple[float, float]] = None
    # Goal to preserve while robot is in lethal space; resent after recovery.
    held_goal_position: Optional[Tuple[float, float]] = None
    # Robot-side topic: RegulatedPurePursuitController "collision ahead" (debounced).
    nav2_collision_ahead: bool = False
    # Async Nav2 path preflight (compute_path_to_pose before navigate_to_pose).
    path_precheck_client: Optional[object] = None
    path_precheck_goal_handle: Optional[object] = None
    path_precheck_in_progress: bool = False
    path_precheck_frontier: Optional[Frontier] = None
    path_precheck_from_retarget: bool = False
    path_precheck_goal_xy: Optional[Tuple[float, float]] = None
    # World-frame goal (merged map) for blacklists / NavigateToPose after precheck.
    path_precheck_world_goal_xy: Optional[Tuple[float, float]] = None
    path_precheck_started_time: float = 0.0
    # Bumped when a precheck is started or abandoned so a late Nav2 result
    # cannot clear or blacklist a newer attempt.
    path_precheck_epoch: int = 0
    # First time compute_path_to_pose wait_for_server succeeded (Nav2 warm-up).
    path_precheck_server_seen_time: float = 0.0
    # Session start pose in world frame (recorded on first TF once maps are ready).
    start_pose_xy: Optional[Tuple[float, float]] = None
    # True while a return-home NavigateToPose is active or pending.
    homing_active: bool = False
    # True after successful home, skip (already near start), or max failures.
    return_home_done: bool = False
    # Count of failed homing attempts (precheck abort, Nav2 failure, stall cancel).
    return_home_failures: int = 0
    # Degraded-data hold mode: true when explorer should pause new assignments
    # and avoid treating recent failures as map/frontier issues.
    degraded_active: bool = False
    degraded_since: float = 0.0
    degraded_reason: str = ''
    # Rolling counters for transient degradation signals.
    tf_degraded_events: int = 0
    precheck_fail_events: int = 0
    precheck_timeout_events: int = 0
    precheck_server_unavailable_events: int = 0
    # Last known planner lifecycle state label ("active", "inactive", ...).
    planner_state_label: str = ''
    planner_state_update_time: float = 0.0
    # Track repeated transient ABORTED precheck outcomes for the same goal.
    precheck_transient_goal_xy: Optional[Tuple[float, float]] = None
    precheck_transient_repeat_count: int = 0
    # Near-goal arrival feasibility: ComputePathToPose(use_start=True) + map score.
    arrival_probe_in_progress: bool = False
    arrival_probe_goal_handle: Optional[object] = None
    arrival_probe_started_time: float = 0.0
    arrival_probe_tacit_followup: bool = False
    # Set after first tacit-time attempt to run arrival probe (success or skip).
    arrival_tacit_probe_attempted: bool = False
    # active_goal_seq for which we already ran the stagnation-band probe once.
    arrival_stagnation_probe_fired_seq: int = -1
    arrival_probe_for_goal_seq: int = 0
    arrival_probe_map_ref: Optional[OccupancyGrid] = None


# ---------------------------------------------------------------------------
# Frontier detection
# ---------------------------------------------------------------------------

FREE = 0
UNKNOWN = -1
OCCUPIED_THRESH = 50  # cells >= this are considered occupied


def _label_frontiers_8c(mask: np.ndarray):
    """Connected-component labelling for frontier regions (8-connectivity).

    Returns (labelled_array, n_labels) similar to scipy.ndimage.label.
    """
    h, w = mask.shape
    labels = np.zeros_like(mask, dtype=np.int32)
    current = 0

    neighbours = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x] != 0:
                continue

            current += 1
            labels[y, x] = current
            q = deque([(y, x)])

            while q:
                cy, cx = q.popleft()
                for dy, dx in neighbours:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            q.append((ny, nx))

    return labels, current


def detect_frontiers(
    map_data: List[int],
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    min_size_m: float,
    pose_frame_id: Optional[str] = None,
) -> List[Frontier]:
    """Find frontier regions on an OccupancyGrid.

    A *frontier cell* is a free cell (value 0) that has at least one
    unknown neighbour (value -1) in 4-connectivity.  Adjacent frontier
    cells (8-connectivity) are clustered into frontier regions.
    """
    grid = np.array(map_data, dtype=np.int8).reshape((height, width))

    free = grid == FREE
    unknown = grid == UNKNOWN

    # frontier = free cells with at least one unknown 4-neighbour
    padded_unk = np.pad(unknown, 1, constant_values=False)
    has_unk_neighbour = (
        padded_unk[:-2, 1:-1]  # above
        | padded_unk[2:, 1:-1]  # below
        | padded_unk[1:-1, :-2]  # left
        | padded_unk[1:-1, 2:]   # right
    )
    frontier_mask = free & has_unk_neighbour

    if not np.any(frontier_mask):
        return []

    # cluster with 8-connectivity (pure NumPy implementation)
    labelled, n_labels = _label_frontiers_8c(frontier_mask)

    # Interpret min_size_m as a characteristic linear size and require at
    # least the corresponding area in cells. This better matches the
    # cluster-size semantics used in costmap-based frontier search.
    min_cells = max(1, int((min_size_m / resolution) ** 2))
    frontiers: List[Frontier] = []
    for label_id in range(1, n_labels + 1):
        ys, xs = np.where(labelled == label_id)
        n = len(ys)
        if n < min_cells:
            continue
        cy = float(np.mean(ys))
        cx = float(np.mean(xs))
        wx = origin_x + (cx + 0.5) * resolution
        wy = origin_y + (cy + 0.5) * resolution
        indices = np.stack((ys, xs), axis=1)
        frontiers.append(Frontier(
            centroid_world=(wx, wy),
            size=n,
            size_m=n * resolution,
            cells=n,
            indices=indices,
            pose_frame_id=pose_frame_id,
        ))

    return frontiers


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def link_retreat_blocks_assignment(enabled: bool, retreat_active: bool) -> bool:
    """True while the robot's link_watchdog is stopping or retreating."""
    return bool(enabled and retreat_active)


def cancel_blacklist_to_undo(
    now: float,
    cancel_time: float,
    point: Optional[Tuple[float, float]],
    window_sec: float,
) -> Optional[Tuple[float, float]]:
    """Point to un-blacklist when a retreat starts just after a cancel.

    The watchdog cancels the goal and raises link_retreat_active in the same
    tick, but the cancel result can reach central first and get blacklisted.
    """
    if point is None or cancel_time <= 0.0:
        return None
    if 0.0 <= now - cancel_time <= window_sec:
        return point
    return None


def assign_frontiers(
    robots: Dict[str, RobotState],
    frontiers: List[Frontier],
    potential_scale: float,
    gain_scale: float,
    nearby_penalty_dist: float = 2.0,
    min_goal_separation: float = 0.25,
    blacklist_radius: float = 0.5,
    utility_penalty_fn: Optional[Callable[[str, Frontier], float]] = None,
    skip_frontier_fn: Optional[Callable[[str, Frontier], bool]] = None,
    robot_ready_fn: Optional[Callable[[str], bool]] = None,
    goal_dist_fn: Optional[Callable[[str, Frontier], float]] = None,
    assignment_robot_order_fn: Optional[Callable[[List[str]], List[str]]] = None,
) -> Tuple[Dict[str, int], bool]:
    """Assign one frontier to each idle robot.

    Returns ({robot_name: frontier_index}, relaxed_min_separation).

    If every candidate is closer than ``min_goal_separation`` (e.g. robot
    just reached a goal on that frontier), the matrix would be empty. A
    second pass uses separation 0 so exploration can continue; the node also
    blacklists a just-visited frontier on Nav2 success (and clears stale
    blacklists near the success pose) so the same point is not re-selected
    on the next map tick. Repeat-goal blacklisting is an additional backstop
    (see parameters ``repeat_goal_dist_threshold`` / ``repeat_goal_count_limit``).

    When ``assignment_robot_order_fn`` is set, each robot (in that order)
    picks the best still-unassigned frontier. This avoids starvation when
    global greedy assigns the only frontier to another namespaced robot that
    is idle but not actually exploring.

    Utility for (robot, frontier):
        gain_scale * frontier.size_m  -  potential_scale * distance
        minus a penalty if another robot already targets a nearby frontier.
    """
    # robots that need a new goal
    idle_robots = {
        name: rs for name, rs in robots.items()
        if not rs.goal_active and not rs.goal_pending and rs.position is not None
        and (robot_ready_fn is None or robot_ready_fn(name))
    }
    if not idle_robots or not frontiers:
        return {}, False

    # already-assigned goal positions (from robots that ARE active)
    active_goals = [
        rs.goal_position
        for rs in robots.values()
        if rs.goal_active and rs.goal_position is not None
    ]

    robot_list = list(idle_robots.keys())
    relaxed = False
    utilities = np.full((len(robot_list), len(frontiers)), -np.inf)

    for min_sep in (min_goal_separation, 0.0):
        utilities.fill(-np.inf)
        for ri, rname in enumerate(robot_list):
            rpos = idle_robots[rname].position
            blacklist = idle_robots[rname].blacklist
            for fi, fr in enumerate(frontiers):
                if any(
                    _dist(fr.centroid_world, bl) < blacklist_radius
                    for bl in blacklist
                ):
                    continue
                if skip_frontier_fn is not None and skip_frontier_fn(rname, fr):
                    continue

                if goal_dist_fn is not None:
                    dist = float(goal_dist_fn(rname, fr))
                else:
                    dist = _dist(rpos, fr.centroid_world)
                # Prefer goals at least min_sep away; allow closer only on
                # relaxed pass after a successful leg onto the frontier.
                if dist < min_sep:
                    continue
                gain = gain_scale * fr.size_m
                cost = potential_scale * dist
                penalty = 0.0
                for ag in active_goals:
                    ad = _dist(fr.centroid_world, ag)
                    if ad < nearby_penalty_dist:
                        penalty += (nearby_penalty_dist - ad) * 2.0
                if utility_penalty_fn is not None:
                    penalty += float(utility_penalty_fn(rname, fr))
                utilities[ri, fi] = gain - cost - penalty

        if np.any(np.isfinite(utilities)):
            relaxed = min_sep < min_goal_separation
            break

    assignments: Dict[str, int] = {}
    assigned_f: set = set()
    name_to_ri = {n: i for i, n in enumerate(robot_list)}

    if assignment_robot_order_fn is not None:
        for rname in assignment_robot_order_fn(list(robot_list)):
            if rname not in name_to_ri:
                continue
            ri = name_to_ri[rname]
            best_fi: Optional[int] = None
            best_u = -1e9
            for fi in range(len(frontiers)):
                if fi in assigned_f:
                    continue
                u = utilities[ri, fi]
                if u > best_u:
                    best_u = u
                    best_fi = fi
            if best_fi is not None and best_u > -1e9:
                assignments[rname] = best_fi
                assigned_f.add(best_fi)
    else:
        assigned_r: set = set()
        flat = np.argsort(utilities.ravel())[::-1]
        for idx in flat:
            ri = int(idx // len(frontiers))
            fi = int(idx % len(frontiers))
            if ri in assigned_r or fi in assigned_f:
                continue
            if utilities[ri, fi] <= -1e9:
                continue
            assignments[robot_list[ri]] = fi
            assigned_r.add(ri)
            assigned_f.add(fi)
            if len(assigned_r) == len(robot_list):
                break

    return assignments, relaxed


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------

class MultiRobotExplorer(Node):
    """Central coordinator for frontier detection, assignment, and Nav2 dispatch.

    The node can operate in local, global, or auto-switch mode. In auto mode it
    starts from per-robot local maps and switches to merged global planning when
    map_merge state is sufficiently stable.
    """

    def __init__(self):
        super().__init__('multi_robot_explorer')

        # -- parameters --
        self.declare_parameter('robot_names', ['blinky', 'pinky', 'inky', 'clyde'])
        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('world_frame', 'map')
        # Path 2: send NavigateToPose / compute_path_to_pose in each robot's map
        # frame (e.g. pinky/map) by TF-transforming from frontier pose_frame_id.
        self.declare_parameter('dispatch_nav_goals_in_robot_map_frame', True)
        self.declare_parameter('nav_goal_frame_pattern', '{robot}/map')
        self.declare_parameter('nav_goal_tf_transform_timeout_sec', 0.5)
        # Parameters controlling local-vs-global planning behavior.
        # mode:
        #   - 'auto'        : start in local-per-robot mode and switch to global
        #                     once the merge_state reports MERGED.
        #   - 'local_only'  : always use per-robot local maps.
        #   - 'global_only' : always use the merged global map (back-compat).
        self.declare_parameter('mode', 'auto')
        # Topic where a helper node (or map_merge) can publish a simple
        # merge state string: NO_OVERLAP, PARTIAL, MERGED.
        self.declare_parameter('merge_state_topic', 'map_merge/merge_state')
        # Terminal-noise controls for long fleet runs.
        self.declare_parameter('terminal_summary_enable', True)
        self.declare_parameter('terminal_summary_period_sec', 10.0)
        self.declare_parameter('terminal_event_log_mode', 'summary')
        # Match m-explore-ros2 planner_frequency semantics; default tuned in YAML.
        self.declare_parameter('explore_frequency', 1.0)
        self.declare_parameter('min_frontier_size', 0.15)
        # Extra filter on top of min_frontier_size: do not send Nav2 goals to
        # very small clusters (wall slivers / noise); they often fail the
        # local controller with "Failed to make progress". Set to 0 to disable.
        self.declare_parameter('min_frontier_cells_for_goal', 32)
        self.declare_parameter('potential_scale', 3.0)
        self.declare_parameter('gain_scale', 1.0)
        # Progress timeout is interpreted as "no meaningful distance progress"
        # for this many seconds before a goal is considered stalled.
        self.declare_parameter('progress_timeout', 45.0)
        self.declare_parameter('progress_min_delta', 0.05)
        self.declare_parameter('motion_progress_min_delta_m', 0.03)
        self.declare_parameter('motion_progress_window_sec', 2.0)
        self.declare_parameter('nearby_penalty_dist', 2.0)
        self.declare_parameter('blacklist_radius', 0.5)
        self.declare_parameter('blacklist_clear_radius', 0.5)
        # On exploration success, drop blacklist points near the goal (revives
        # stale failure flags), then (if true) add the just-visited frontier
        # centroid to reduce instant re-send while the map is unchanged.
        self.declare_parameter('post_success_frontier_avoidance', True)
        # Re-send safety net: N consecutive selected goals within dist (m) of
        # the previous one trigger blacklist+skip. Set count_limit to 0 to disable
        # this backstop.
        self.declare_parameter('repeat_goal_dist_threshold', 0.3)
        self.declare_parameter('repeat_goal_count_limit', 5)
        self.declare_parameter('visualize', True)
        self.declare_parameter('use_pose_goal_fallback', True)
        self.declare_parameter('single_robot_offloaded_nav2', False)
        # Minimum separation between robot and candidate goal used as a final
        # safety net; most shaping is done via min_goal_distance.
        self.declare_parameter('min_goal_separation', 0.5)
        # Match Nav2 goal-check tolerance (e.g. TurtleBot3 burger.yaml
        # controller_server.xy_goal_tolerance: 0.25). Suspicious-success only
        # triggers when last distance_remaining is strictly greater than this.
        self.declare_parameter('suspicious_success_distance', 0.25)
        # SUCCEEDED + distance_remaining at/below this: trust Nav2 terminal feedback
        # even when map-frame leg progress is tiny (bridged TF lags the robot).
        self.declare_parameter('trust_nav2_terminal_distance_remaining_m', 0.30)
        # If NavigateToPose ends in ABORTED (e.g. controller "Failed to make
        # progress") but the robot is within this distance of the goal in the
        # map frame, count as reached and do not blacklist — Nav2 often aborts
        # near the target without a clean SUCCEEDED. Use a generous value; TF
        # vs goal can disagree with Nav2's notion of "close".
        self.declare_parameter('tacit_goal_success_radius_m', 0.15)
        # Also treat ABORT / tacit hold as success if Nav2 feedback
        # distance_remaining is at or below this (metres), but only when map-frame
        # pose is within tacit_goal_success_radius_m + this value of the goal
        # (avoids stale distance_remaining=0 while the robot is still far away).
        self.declare_parameter('tacit_abort_max_distance_remaining_m', 0.15)
        # While Nav2 still has an active NavigateToPose, cancel and count
        # success after this many seconds where pose and/or feedback indicate
        # the goal is essentially reached (see _at_nav_goal_for_tacit_hold).
        # This catches Nav2 never publishing SUCCEEDED while the robot is on
        # the target. Set 0 to disable.
        self.declare_parameter('tacit_goal_success_hold_sec', 1.0)
        # On STATUS_CANCELED, optionally blacklist when goal ended far from target
        # and the cancel was not an intentional administrative action.
        self.declare_parameter('blacklist_on_canceled_when_far', True)
        self.declare_parameter('canceled_blacklist_min_goal_dist_m', 0.30)
        # For very small initial maps, Nav2 can legitimately report success
        # with a relatively large distance_remaining because the frontier
        # centroid may still be outside the tiny explored region. To avoid
        # blacklisting useful early frontiers, the "suspicious success"
        # heuristic is only applied once the map has grown beyond a
        # configurable size.
        self.declare_parameter('strict_success_min_map_size', 3.0)
        self.declare_parameter('robot_base_frame', 'base_footprint')
        # Status + control topics (multi-robot counterpart of explore/status and explore/resume).
        self.declare_parameter('status_topic', 'explore_multi/status')
        self.declare_parameter('control_topic', 'explore_multi/resume')
        self.declare_parameter('return_to_start_enable', True)
        self.declare_parameter('return_to_start_skip_if_within_m', 0.25)
        self.declare_parameter('return_to_start_max_attempts', 5)
        # Goal selection strategy within each frontier region.
        # - 'centroid': centroid of the frontier; if the robot is already very
        #   close to that centroid, pick a cell farther out but within
        #   centroid_fallback_max_offset_from_centroid_m of the centroid.
        # - 'farthest_cell': choose the frontier cell farthest from the robot
        #   (in world distance), falling back to centroid when needed.
        self.declare_parameter('goal_point_strategy', 'centroid')
        # When using centroid: if the robot is already this close to the
        # centroid, pick a frontier cell farther out, but only within
        # centroid_fallback_max_offset_from_centroid_m of the centroid (avoids
        # ping-pong across the whole frontier like unbounded farthest_cell).
        self.declare_parameter('centroid_fallback_if_robot_closer_than_m', 0.55)
        self.declare_parameter('centroid_fallback_max_offset_from_centroid_m', 2.5)
        # Minimum desired distance from robot to goal (metres). This is used
        # when selecting a representative cell inside a frontier region so
        # that the initial goals are not almost inside Nav2's goal tolerance.
        self.declare_parameter('min_goal_distance', 0.6)
        # Goal-clearance shaping: discourage assigning goals too close to
        # occupied cells (likely walls/obstacles in the current map).
        self.declare_parameter('goal_clearance_radius_m', 0.45)
        self.declare_parameter('goal_unknown_neighbor_penalty_m', 0.0)
        self.declare_parameter('goal_clearance_weight', 4.0)
        # Hard safety gate for candidate goal cells:
        # - reject goals in high-cost cells
        # - reject goals that do not have enough local obstacle clearance
        self.declare_parameter('goal_max_cell_cost', 80)
        self.declare_parameter('goal_min_clearance_gate_m', 0.30)
        # Re-check selected goals against the latest map right before sending
        # to Nav2. This avoids dispatching goals that became unsafe between
        # frontier assignment and NavigateToPose dispatch.
        self.declare_parameter('goal_revalidate_before_send', True)
        # Mid-route retargeting controls.
        self.declare_parameter('retarget_enable', True)
        self.declare_parameter('retarget_cooldown_sec', 12.0)
        self.declare_parameter('retarget_min_goal_shift_m', 0.5)
        self.declare_parameter('retarget_min_utility_gain', 0.8)
        self.declare_parameter('retarget_stagnation_sec', 20.0)
        self.declare_parameter('retarget_clearance_threshold_m', 0.25)
        self.declare_parameter('retarget_opportunity_enable', True)
        self.declare_parameter('retarget_opportunity_min_utility_gain', 1.2)
        # Retargets only pick goals within this radius (m) of the leg's first
        # goal (retarget_anchor). Set to 0 to disable (allow distant retargets).
        self.declare_parameter('retarget_max_offset_from_anchor_m', 3.0)
        # Goal replacement gate controls to avoid rapid preempt/replan loops.
        self.declare_parameter('min_goal_replan_interval_s', 4.0)
        self.declare_parameter('min_goal_change_dist_m', 0.75)
        self.declare_parameter('min_progress_before_replan_m', 0.25)
        self.declare_parameter('max_stuck_time_s', 12.0)
        self.declare_parameter('allow_replan_when_no_progress', True)
        self.declare_parameter('post_failure_cooldown_sec', 2.0)
        # Grace window after action server appears: early goal rejections are
        # treated as Nav2 warm-up (lifecycle still activating) instead of
        # blacklisting goals.
        self.declare_parameter('goal_reject_warmup_sec', 12.0)
        # Hold active goals while robot is marked "in lethal space" by Nav2
        # and suppress watchdog/blacklisting for planner-start-in-lethal cases.
        self.declare_parameter('lethal_hold_enabled', True)
        self.declare_parameter(
            'nav2_lethal_topic_pattern', '/{robot}/nav2_lethal_inflation')
        self.declare_parameter(
            'nav2_retrace_topic_pattern', '/{robot}/nav2_retrace_active')
        self.declare_parameter('skip_blacklist_on_nav2_lethal', True)
        self.declare_parameter('lethal_hold_max_sec', 30.0)
        self.declare_parameter('lethal_retry_delay_sec', 0.5)
        self.declare_parameter('retrace_hold_enabled', True)
        self.declare_parameter('retrace_hold_max_sec', 30.0)
        # Pause new goals for a robot while its link_watchdog stops or retreats
        # (/<robot>/link_retreat_active), so explorer goals don't preempt the
        # retreat. Off by default.
        self.declare_parameter('pause_on_link_retreat', False)
        self.declare_parameter(
            'link_retreat_topic_pattern', '/{robot}/link_retreat_active')
        self.declare_parameter('link_retreat_unblacklist_window_sec', 3.0)
        # Debounced Bool from turtlebot3_navigation2/nav2_controller_collision_watch.py
        self.declare_parameter(
            'nav2_collision_ahead_topic_pattern', '/{robot}/nav2_collision_ahead')
        # When true, shorten tacit/stagnation waits for arrival probe if collision is reported.
        self.declare_parameter('goal_arrival_probe_use_collision_ahead', True)
        self.declare_parameter('goal_arrival_collision_tacit_hold_sec', 0.15)
        self.declare_parameter('goal_arrival_probe_stagnation_sec_collision', 0.5)
        # Per-robot local map topic pattern. By default we assume the standard
        # namespaced SLAM layout '/<robot>/map'.
        self.declare_parameter('local_map_topic_pattern', '/{robot}/map')
        # Bridge contract: allow disabling per-robot local map ingestion when
        # operating strictly on merged global map to reduce fleet DDS load.
        self.declare_parameter('local_map_subscriptions_enable', True)
        # Call /<robot>/compute_path_to_pose before NavigateToPose so goals
        # Nav2 cannot plan through the global costmap are skipped here.
        self.declare_parameter('nav2_path_precheck_enable', True)
        self.declare_parameter('nav2_path_precheck_required', True)
        self.declare_parameter('nav2_path_precheck_server_wait_sec', 0.5)
        # First connection to NavigateToPose can exceed ~200 ms across
        # domain_bridge or lossy links; ros2 action list may show the server
        # before ActionClient.wait_for_server succeeds.
        self.declare_parameter('navigate_to_pose_server_wait_sec', 5.0)
        self.declare_parameter('nav2_path_precheck_timeout_sec', 3.0)
        self.declare_parameter('nav2_path_min_poses', 2)
        # If ABORTED precheck has no explicit error code, treat repeated failures
        # for the same goal as hard-unreachable after this many repeats.
        self.declare_parameter('nav2_path_precheck_transient_abort_repeat_limit', 3)
        # Parse planner_server /rosout no-valid-path hints and map them back
        # onto precheck goals to force hard blacklist when Nav2 result fields
        # are missing detail.
        self.declare_parameter('nav2_path_precheck_use_rosout_no_path_hint', True)
        self.declare_parameter('nav2_path_precheck_rosout_no_path_window_sec', 6.0)
        self.declare_parameter('nav2_path_precheck_rosout_goal_match_tol_m', 0.30)
        # If true, immediately treat matched planner no-path rosout hints as
        # hard failures (blacklist + precheck abort) instead of waiting for
        # subsequent retries/result callbacks.
        self.declare_parameter('nav2_path_precheck_rosout_immediate_blacklist', True)
        # Enrichment for ambiguous ABORTED precheck reasons.
        self.declare_parameter('nav2_path_precheck_use_planner_lifecycle_hint', True)
        self.declare_parameter('nav2_path_precheck_planner_state_stale_sec', 8.0)
        # If precheck ABORTED is non-explicit but planner is confirmed active
        # (and data path looks healthy), treat it as hard no-path immediately.
        self.declare_parameter('nav2_path_precheck_abort_active_planner_is_hard_fail', True)
        # Must match planner_server.planner_plugins id (e.g. GridBased in burger.yaml).
        self.declare_parameter('nav2_path_precheck_planner_id', 'GridBased')
        # Near-goal arrival probe: replan from current TF pose, score path on merged map.
        self.declare_parameter('goal_arrival_probe_enable', True)
        self.declare_parameter('goal_arrival_probe_max_pose_dist_m', 1.0)
        self.declare_parameter('goal_arrival_probe_max_distance_remaining_m', 0.35)
        self.declare_parameter('goal_arrival_probe_stagnation_sec', 3.0)
        self.declare_parameter('goal_arrival_probe_timeout_sec', 3.0)
        self.declare_parameter('goal_arrival_probe_before_tacit_success', True)
        self.declare_parameter('goal_arrival_path_max_length_ratio', 2.5)
        self.declare_parameter('goal_arrival_max_cell_cost', 80)
        self.declare_parameter('goal_arrival_max_bad_fraction', 0.35)
        self.declare_parameter('goal_arrival_sample_step_scale', 0.5)
        # Degraded-data hold: suppress blacklist / stall cancel when data quality
        # is clearly bad (stale TF or repeated precheck issues) rather than true
        # map/frontier failure.
        self.declare_parameter('degraded_hold_enabled', True)
        self.declare_parameter('degraded_tf_hold_sec', 2.0)
        self.declare_parameter('degraded_precheck_fail_threshold', 3)
        self.declare_parameter('degraded_precheck_window_sec', 10.0)
        self.declare_parameter('degraded_recovery_stable_sec', 4.0)
        # Merge-state hysteresis (auto mode only): require MERGED to remain
        # stable for a short hold before switching to global, and fall back to
        # local if merge degrades for long enough. This avoids thrashing when
        # map_merge briefly reports MERGED on weak feature matches.
        self.declare_parameter('merge_stable_hold_sec', 8.0)
        self.declare_parameter('merge_degrade_grace_sec', 8.0)

        self.robot_names: List[str] = (
            self.get_parameter('robot_names').value)
        map_topic = self.get_parameter('map_topic').value
        self.map_topic = map_topic
        self.world_frame = self.get_parameter('world_frame').value
        self.dispatch_nav_goals_in_robot_map_frame: bool = bool(
            self.get_parameter('dispatch_nav_goals_in_robot_map_frame').value
        )
        self.nav_goal_frame_pattern: str = str(
            self.get_parameter('nav_goal_frame_pattern').value
        )
        self.nav_goal_tf_transform_timeout_sec: float = max(
            0.05,
            float(self.get_parameter('nav_goal_tf_transform_timeout_sec').value),
        )
        self.mode: str = (
            self.get_parameter('mode').value or 'auto'
        ).lower()
        self.merge_state_topic: str = (
            self.get_parameter('merge_state_topic').value
        )
        self.terminal_summary_enable: bool = bool(
            self.get_parameter('terminal_summary_enable').value
        )
        self.terminal_summary_period_sec: float = max(
            2.0, float(self.get_parameter('terminal_summary_period_sec').value)
        )
        self.terminal_event_log_mode: str = str(
            self.get_parameter('terminal_event_log_mode').value or 'summary'
        ).strip().lower()
        if self.terminal_event_log_mode not in ('summary', 'verbose'):
            self.terminal_event_log_mode = 'summary'
        self._last_summary_counts: Dict[str, Tuple[int, int]] = {}
        freq = self.get_parameter('explore_frequency').value
        self.min_frontier_size = self.get_parameter('min_frontier_size').value
        self.min_frontier_cells_for_goal = int(
            self.get_parameter('min_frontier_cells_for_goal').value
        )
        self.potential_scale = self.get_parameter('potential_scale').value
        self.gain_scale = self.get_parameter('gain_scale').value
        self.progress_timeout = self.get_parameter('progress_timeout').value
        self.progress_min_delta = (
            self.get_parameter('progress_min_delta').value)
        self.motion_progress_min_delta_m = float(
            self.get_parameter('motion_progress_min_delta_m').value
        )
        self.motion_progress_window_sec = float(
            self.get_parameter('motion_progress_window_sec').value
        )
        self.nearby_penalty_dist = (
            self.get_parameter('nearby_penalty_dist').value)
        self.blacklist_radius = self.get_parameter('blacklist_radius').value
        self.blacklist_clear_radius = float(
            self.get_parameter('blacklist_clear_radius').value)
        self.post_success_frontier_avoidance: bool = bool(
            self.get_parameter('post_success_frontier_avoidance').value)
        self.repeat_goal_dist_threshold: float = max(
            0.0, float(self.get_parameter('repeat_goal_dist_threshold').value))
        self.repeat_goal_count_limit: int = int(
            self.get_parameter('repeat_goal_count_limit').value
        )
        self.visualize = self.get_parameter('visualize').value
        self.use_pose_goal_fallback = (
            self.get_parameter('use_pose_goal_fallback').value)
        self.single_robot_offloaded_nav2 = (
            self.get_parameter('single_robot_offloaded_nav2').value)
        self.min_goal_separation = (
            self.get_parameter('min_goal_separation').value)
        self.suspicious_success_distance = (
            self.get_parameter('suspicious_success_distance').value)
        self.trust_nav2_terminal_distance_remaining_m = float(
            self.get_parameter('trust_nav2_terminal_distance_remaining_m').value
        )
        self.tacit_goal_success_radius_m = float(
            self.get_parameter('tacit_goal_success_radius_m').value
        )
        self.tacit_abort_max_distance_remaining_m = float(
            self.get_parameter('tacit_abort_max_distance_remaining_m').value
        )
        self.tacit_goal_success_hold_sec = float(
            self.get_parameter('tacit_goal_success_hold_sec').value
        )
        self.blacklist_on_canceled_when_far: bool = bool(
            self.get_parameter('blacklist_on_canceled_when_far').value
        )
        self.canceled_blacklist_min_goal_dist_m: float = max(
            0.0,
            float(self.get_parameter('canceled_blacklist_min_goal_dist_m').value),
        )
        self.strict_success_min_map_size = (
            self.get_parameter('strict_success_min_map_size').value)
        self.robot_base_frame = self.get_parameter('robot_base_frame').value
        self.status_topic = self.get_parameter('status_topic').value
        self.control_topic = self.get_parameter('control_topic').value
        self.return_to_start_enable = bool(
            self.get_parameter('return_to_start_enable').value
        )
        self.return_to_start_skip_if_within_m = float(
            self.get_parameter('return_to_start_skip_if_within_m').value
        )
        self.return_to_start_max_attempts = max(
            1, int(self.get_parameter('return_to_start_max_attempts').value)
        )
        self.goal_point_strategy = (
            self.get_parameter('goal_point_strategy').value or 'centroid'
        )
        self.centroid_fallback_if_robot_closer_than_m = float(
            self.get_parameter('centroid_fallback_if_robot_closer_than_m').value
        )
        self.centroid_fallback_max_offset_from_centroid_m = float(
            self.get_parameter('centroid_fallback_max_offset_from_centroid_m').value
        )
        self.min_goal_distance = float(
            self.get_parameter('min_goal_distance').value
        )
        self.goal_clearance_radius_m = float(
            self.get_parameter('goal_clearance_radius_m').value
        )
        self.goal_unknown_neighbor_penalty_m = float(
            self.get_parameter('goal_unknown_neighbor_penalty_m').value
        )
        self.goal_clearance_weight = float(
            self.get_parameter('goal_clearance_weight').value
        )
        self.goal_max_cell_cost = int(
            self.get_parameter('goal_max_cell_cost').value
        )
        self.goal_min_clearance_gate_m = float(
            self.get_parameter('goal_min_clearance_gate_m').value
        )
        self.goal_revalidate_before_send = bool(
            self.get_parameter('goal_revalidate_before_send').value
        )
        self.retarget_enable = bool(
            self.get_parameter('retarget_enable').value
        )
        self.retarget_cooldown_sec = float(
            self.get_parameter('retarget_cooldown_sec').value
        )
        self.retarget_min_goal_shift_m = float(
            self.get_parameter('retarget_min_goal_shift_m').value
        )
        self.retarget_min_utility_gain = float(
            self.get_parameter('retarget_min_utility_gain').value
        )
        self.retarget_stagnation_sec = float(
            self.get_parameter('retarget_stagnation_sec').value
        )
        self.retarget_clearance_threshold_m = float(
            self.get_parameter('retarget_clearance_threshold_m').value
        )
        self.retarget_opportunity_enable = bool(
            self.get_parameter('retarget_opportunity_enable').value
        )
        self.retarget_opportunity_min_utility_gain = float(
            self.get_parameter('retarget_opportunity_min_utility_gain').value
        )
        self.retarget_max_offset_from_anchor_m = float(
            self.get_parameter('retarget_max_offset_from_anchor_m').value
        )
        self.min_goal_replan_interval_s = float(
            self.get_parameter('min_goal_replan_interval_s').value
        )
        self.min_goal_change_dist_m = float(
            self.get_parameter('min_goal_change_dist_m').value
        )
        self.min_progress_before_replan_m = float(
            self.get_parameter('min_progress_before_replan_m').value
        )
        self.max_stuck_time_s = float(
            self.get_parameter('max_stuck_time_s').value
        )
        self.allow_replan_when_no_progress = bool(
            self.get_parameter('allow_replan_when_no_progress').value
        )
        self.post_failure_cooldown_sec = float(
            self.get_parameter('post_failure_cooldown_sec').value
        )
        self.goal_reject_warmup_sec = float(
            self.get_parameter('goal_reject_warmup_sec').value
        )
        self.lethal_hold_enabled = bool(
            self.get_parameter('lethal_hold_enabled').value
        )
        self.nav2_lethal_topic_pattern = str(
            self.get_parameter('nav2_lethal_topic_pattern').value
        )
        self.nav2_retrace_topic_pattern = str(
            self.get_parameter('nav2_retrace_topic_pattern').value
        )
        self.skip_blacklist_on_nav2_lethal = bool(
            self.get_parameter('skip_blacklist_on_nav2_lethal').value
        )
        self.lethal_hold_max_sec = float(
            self.get_parameter('lethal_hold_max_sec').value
        )
        self.lethal_retry_delay_sec = float(
            self.get_parameter('lethal_retry_delay_sec').value
        )
        self.retrace_hold_enabled = bool(
            self.get_parameter('retrace_hold_enabled').value
        )
        self.pause_on_link_retreat = bool(
            self.get_parameter('pause_on_link_retreat').value
        )
        self.link_retreat_topic_pattern = str(
            self.get_parameter('link_retreat_topic_pattern').value
        )
        self.link_retreat_unblacklist_window_sec = float(
            self.get_parameter('link_retreat_unblacklist_window_sec').value
        )
        self.retrace_hold_max_sec = float(
            self.get_parameter('retrace_hold_max_sec').value
        )
        self.nav2_collision_ahead_topic_pattern: str = str(
            self.get_parameter('nav2_collision_ahead_topic_pattern').value
        )
        self.goal_arrival_probe_use_collision_ahead: bool = bool(
            self.get_parameter('goal_arrival_probe_use_collision_ahead').value
        )
        self.goal_arrival_collision_tacit_hold_sec: float = float(
            self.get_parameter('goal_arrival_collision_tacit_hold_sec').value
        )
        self.goal_arrival_probe_stagnation_sec_collision: float = float(
            self.get_parameter('goal_arrival_probe_stagnation_sec_collision').value
        )
        self.local_map_topic_pattern: str = (
            self.get_parameter('local_map_topic_pattern').value
        )
        self.local_map_subscriptions_enable: bool = bool(
            self.get_parameter('local_map_subscriptions_enable').value
        )
        self.nav2_path_precheck_enable: bool = bool(
            self.get_parameter('nav2_path_precheck_enable').value
        )
        self.nav2_path_precheck_required: bool = bool(
            self.get_parameter('nav2_path_precheck_required').value
        )
        self.nav2_path_precheck_server_wait_sec: float = float(
            self.get_parameter('nav2_path_precheck_server_wait_sec').value
        )
        self.navigate_to_pose_server_wait_sec: float = max(
            0.05,
            float(self.get_parameter('navigate_to_pose_server_wait_sec').value),
        )
        self.nav2_path_precheck_timeout_sec: float = float(
            self.get_parameter('nav2_path_precheck_timeout_sec').value
        )
        self.tf_stale_timeout_sec: float = float(
            self.declare_parameter('tf_stale_timeout_sec', 4.0).value
        )
        self.degraded_hold_enabled: bool = bool(
            self.get_parameter('degraded_hold_enabled').value
        )
        self.degraded_tf_hold_sec: float = float(
            self.get_parameter('degraded_tf_hold_sec').value
        )
        self.degraded_precheck_fail_threshold: int = max(
            1, int(self.get_parameter('degraded_precheck_fail_threshold').value)
        )
        self.degraded_precheck_window_sec: float = float(
            self.get_parameter('degraded_precheck_window_sec').value
        )
        self.degraded_recovery_stable_sec: float = float(
            self.get_parameter('degraded_recovery_stable_sec').value
        )
        self.nav2_path_min_poses: int = max(
            1, int(self.get_parameter('nav2_path_min_poses').value)
        )
        self.nav2_path_precheck_transient_abort_repeat_limit: int = max(
            1, int(self.get_parameter(
                'nav2_path_precheck_transient_abort_repeat_limit').value)
        )
        self.nav2_path_precheck_use_rosout_no_path_hint: bool = bool(
            self.get_parameter('nav2_path_precheck_use_rosout_no_path_hint').value
        )
        self.nav2_path_precheck_rosout_no_path_window_sec: float = max(
            0.0,
            float(self.get_parameter(
                'nav2_path_precheck_rosout_no_path_window_sec').value),
        )
        self.nav2_path_precheck_rosout_goal_match_tol_m: float = max(
            0.0,
            float(self.get_parameter(
                'nav2_path_precheck_rosout_goal_match_tol_m').value),
        )
        self.nav2_path_precheck_rosout_immediate_blacklist: bool = bool(
            self.get_parameter(
                'nav2_path_precheck_rosout_immediate_blacklist').value
        )
        self.nav2_path_precheck_use_planner_lifecycle_hint: bool = bool(
            self.get_parameter(
                'nav2_path_precheck_use_planner_lifecycle_hint').value
        )
        self.nav2_path_precheck_planner_state_stale_sec: float = max(
            0.0,
            float(self.get_parameter(
                'nav2_path_precheck_planner_state_stale_sec').value),
        )
        self.nav2_path_precheck_abort_active_planner_is_hard_fail: bool = bool(
            self.get_parameter(
                'nav2_path_precheck_abort_active_planner_is_hard_fail').value
        )
        _pid = self.get_parameter('nav2_path_precheck_planner_id').value
        self.nav2_path_precheck_planner_id: str = (
            str(_pid).strip() if _pid is not None else ''
        )
        self.goal_arrival_probe_enable: bool = bool(
            self.get_parameter('goal_arrival_probe_enable').value
        )
        self.goal_arrival_probe_max_pose_dist_m: float = float(
            self.get_parameter('goal_arrival_probe_max_pose_dist_m').value
        )
        self.goal_arrival_probe_max_distance_remaining_m: float = float(
            self.get_parameter('goal_arrival_probe_max_distance_remaining_m').value
        )
        self.goal_arrival_probe_stagnation_sec: float = float(
            self.get_parameter('goal_arrival_probe_stagnation_sec').value
        )
        self.goal_arrival_probe_timeout_sec: float = float(
            self.get_parameter('goal_arrival_probe_timeout_sec').value
        )
        self.goal_arrival_probe_before_tacit_success: bool = bool(
            self.get_parameter('goal_arrival_probe_before_tacit_success').value
        )
        self.goal_arrival_path_max_length_ratio: float = float(
            self.get_parameter('goal_arrival_path_max_length_ratio').value
        )
        self.goal_arrival_max_cell_cost: int = int(
            self.get_parameter('goal_arrival_max_cell_cost').value
        )
        self.goal_arrival_max_bad_fraction: float = float(
            self.get_parameter('goal_arrival_max_bad_fraction').value
        )
        self.goal_arrival_sample_step_scale: float = float(
            self.get_parameter('goal_arrival_sample_step_scale').value
        )
        self.merge_stable_hold_sec: float = max(
            0.0, float(self.get_parameter('merge_stable_hold_sec').value)
        )
        self.merge_degrade_grace_sec: float = max(
            0.0, float(self.get_parameter('merge_degrade_grace_sec').value)
        )

        # -- state --
        self.robots: Dict[str, RobotState] = {}
        # Global merged map (from map_merge in multi-robot mode, or a single
        # robot's map in single-robot mode).
        self.latest_global_map: Optional[OccupancyGrid] = None
        # Per-robot local maps. In many setups these are the same as the maps
        # consumed by map_merge (/<robot>/map).
        self.latest_local_maps: Dict[str, OccupancyGrid] = {}
        # Map message currently associated with goal selection; set by the
        # frontier-step helpers before assigning goals so _select_goal_point
        # can convert frontier cells into world coordinates without assuming
        # a single global map source.
        self._current_goal_map: Optional[OccupancyGrid] = None
        self._map_size_x: float = 0.0
        self._map_size_y: float = 0.0
        self.exploration_complete = False
        # Set true once past waiting-for-map so start poses can be recorded.
        self._explorer_planning_ready: bool = False
        self._return_home_status_published: str = ''
        # Rotate fair frontier assignment so this robot is tried first when idle.
        self._last_goal_finished_robot: Optional[str] = None
        self._last_no_frontier_idle_log_time: float = 0.0
        self._last_no_assignment_log_time: float = 0.0
        self._return_home_no_pose_logged: set = set()
        self.goal_pubs: Dict[str, object] = {}
        self.lethal_subs: Dict[str, object] = {}
        self.retrace_subs: Dict[str, object] = {}
        self.link_retreat_subs: Dict[str, object] = {}
        self.collision_ahead_subs: Dict[str, object] = {}
        self.planner_state_subs: Dict[str, object] = {}
        self.paused: bool = False
        self.in_global_phase: bool = False
        self._merge_state_last: str = 'NO_OVERLAP'
        self._merge_merged_since: float = 0.0
        self._merge_nonmerged_since: float = 0.0
        self.event_logger = ExplorerEventLogger()
        self._planner_no_path_hints: Dict[str, Tuple[Tuple[float, float], float]] = {}
        self._planner_no_path_re = re.compile(
            r'failed to generate a valid path to \(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)',
            re.IGNORECASE,
        )
        self._planner_no_path_plain_re = re.compile(
            r'failed to create plan,\s*no valid path found',
            re.IGNORECASE,
        )

        # -- status publishing & control --
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.resume_sub = self.create_subscription(
            Bool,
            self.control_topic,
            self._resume_callback,
            10,
        )

        # -- TF --
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # -- global map subscription --
        map_qos_transient = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.map_sub_transient = self.create_subscription(
            OccupancyGrid,
            map_topic,
            self._global_map_callback,
            map_qos_transient,
        )

        map_qos_compatible = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.map_sub_compatible = self.create_subscription(
            OccupancyGrid,
            map_topic,
            self._global_map_callback,
            map_qos_compatible,
        )

        # -- per-robot local map subscriptions --
        self._local_map_subs = []
        use_local_maps = (
            self.local_map_subscriptions_enable and self.mode != 'global_only'
        )
        if use_local_maps:
            for name in self.robot_names:
                topic = self.local_map_topic_pattern.format(robot=name)
                sub_t = self.create_subscription(
                    OccupancyGrid,
                    topic,
                    lambda msg, r=name: self._local_map_callback(msg, r),
                    map_qos_transient,
                )
                sub_c = self.create_subscription(
                    OccupancyGrid,
                    topic,
                    lambda msg, r=name: self._local_map_callback(msg, r),
                    map_qos_compatible,
                )
                self._local_map_subs.append(sub_t)
                self._local_map_subs.append(sub_c)

        # -- merge state subscription (optional) --
        self.merge_state_sub = self.create_subscription(
            StringMsg,
            self.merge_state_topic,
            self._merge_state_callback,
            10,
        )
        if self.nav2_path_precheck_use_rosout_no_path_hint:
            self.rosout_sub = self.create_subscription(
                Log,
                '/rosout',
                self._rosout_callback,
                100,
            )

        # -- visualisation --
        if self.visualize:
            self.marker_pub = self.create_publisher(
                MarkerArray, 'explore/frontiers', 10)

        # -- per-robot setup --
        for name in self.robot_names:
            rs = RobotState(name=name)
            rs.action_client = ActionClient(
                self, NavigateToPose, f'/{name}/navigate_to_pose')
            rs.path_precheck_client = ActionClient(
                self, ComputePathToPose, f'/{name}/compute_path_to_pose')
            self.robots[name] = rs
            if self.lethal_hold_enabled:
                topic = self.nav2_lethal_topic_pattern.format(robot=name)
                self.lethal_subs[name] = self.create_subscription(
                    Bool,
                    topic,
                    lambda msg, rname=name: self._nav2_lethal_callback(msg, rname),
                    10,
                )
            if self.retrace_hold_enabled:
                rtopic = self.nav2_retrace_topic_pattern.format(robot=name)
                self.retrace_subs[name] = self.create_subscription(
                    Bool,
                    rtopic,
                    lambda msg, rname=name: self._nav2_retrace_callback(msg, rname),
                    10,
                )
            if self.pause_on_link_retreat:
                ltopic = self.link_retreat_topic_pattern.format(robot=name)
                self.link_retreat_subs[name] = self.create_subscription(
                    Bool,
                    ltopic,
                    lambda msg, rname=name: self._link_retreat_callback(msg, rname),
                    10,
                )
            if self.goal_arrival_probe_use_collision_ahead:
                ctopic = self.nav2_collision_ahead_topic_pattern.format(
                    robot=name)
                self.collision_ahead_subs[name] = self.create_subscription(
                    Bool,
                    ctopic,
                    lambda msg, rname=name: self._nav2_collision_ahead_callback(
                        msg, rname),
                    10,
                )
            if self.nav2_path_precheck_use_planner_lifecycle_hint:
                ptopic = f'/{name}/planner_server/transition_event'
                self.planner_state_subs[name] = self.create_subscription(
                    TransitionEvent,
                    ptopic,
                    lambda msg, rname=name: self._planner_lifecycle_callback(
                        msg, rname),
                    10,
                )
            if self.use_pose_goal_fallback:
                self.goal_pubs[name] = self.create_publisher(
                    PoseStamped, f'/{name}/goal_pose', 10)

        self._logged_waiting_for_map = False

        # -- planning timer --
        period = 1.0 / freq if freq > 0 else 3.0
        self.plan_timer = self.create_timer(period, self._plan_tick)
        if self.terminal_summary_enable:
            self.summary_timer = self.create_timer(
                self.terminal_summary_period_sec, self._emit_terminal_summary
            )
        else:
            self.summary_timer = None

        mode = 'single_robot_offloaded_nav2' if self.single_robot_offloaded_nav2 else 'multi_robot'
        self.get_logger().info(
            f'Multi-robot explorer started: robots={self.robot_names}, '
            f'map_topic={map_topic}, world_frame={self.world_frame}, '
            f'freq={freq:.2f} Hz, '
            f'local_map_subs={use_local_maps}, '
            f'use_pose_goal_fallback={self.use_pose_goal_fallback}, '
            f'mode={mode}, retarget_enable={self.retarget_enable}, '
            f'retarget_opportunity={self.retarget_opportunity_enable}, '
            f'nav2_path_precheck={self.nav2_path_precheck_enable}, '
            f'retrace_hold={self.retrace_hold_enabled}, '
            f'pause_on_link_retreat={self.pause_on_link_retreat}, '
            f'goal_arrival_probe={self.goal_arrival_probe_enable}, '
            f'collision_ahead_for_probe={self.goal_arrival_probe_use_collision_ahead}, '
            f'navigate_to_pose_wait_sec={self.navigate_to_pose_server_wait_sec:.1f}, '
            f'dispatch_nav_goals_in_robot_map_frame='
            f'{self.dispatch_nav_goals_in_robot_map_frame}')
        self._publish_status('STARTED')

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def _global_map_callback(self, msg: OccupancyGrid):
        self.latest_global_map = msg
        # Track current map physical size in metres for heuristics that behave
        # differently on tiny initial maps vs. larger, more mature maps.
        res = msg.info.resolution
        self._map_size_x = msg.info.width * res
        self._map_size_y = msg.info.height * res

    def _local_map_callback(self, msg: OccupancyGrid, robot: str):
        # Store the most recent local map per robot. These are only used in
        # local-per-robot mode.
        self.latest_local_maps[robot] = msg

    def _rosout_callback(self, msg: Log):
        try:
            if msg.level < Log.WARN:
                return
            name = str(msg.name or '')
            if '.planner_server' not in name:
                return
            text = str(msg.msg or '')
            text_lower = text.lower()
            has_coord_no_path = 'failed to generate a valid path to' in text_lower
            has_plain_no_path = bool(self._planner_no_path_plain_re.search(text))
            if not has_coord_no_path and not has_plain_no_path:
                return
            robot = name.split('.', 1)[0].strip().lstrip('/')
            if robot not in self.robots:
                # rosout logger names are not always exactly "<robot>.<node>".
                # Try matching by namespace token before giving up.
                for candidate in self.robots.keys():
                    if f'/{candidate}/' in name or f'{candidate}.' in name:
                        robot = candidate
                        break
            if robot not in self.robots:
                # Single-robot mode fallback: treat planner warnings as belonging
                # to the only managed robot namespace.
                if len(self.robots) == 1:
                    robot = next(iter(self.robots.keys()))
                else:
                    return
            rs = self.robots[robot]
            m = self._planner_no_path_re.search(text)
            if m is not None:
                try:
                    gx = float(m.group(1))
                    gy = float(m.group(2))
                    hinted_goal = (gx, gy)
                except Exception:
                    hinted_goal = rs.path_precheck_goal_xy
            else:
                hinted_goal = rs.path_precheck_goal_xy
            if hinted_goal is None:
                return
            now = self.get_clock().now().nanoseconds / 1e9
            self._planner_no_path_hints[robot] = (hinted_goal, now)
            if not self.nav2_path_precheck_rosout_immediate_blacklist:
                return
            # Fast path: if this matches the currently prechecked goal, do the
            # blacklist now rather than waiting for repeated ABORTED retries.
            if (
                rs.path_precheck_in_progress
                and rs.path_precheck_goal_xy is not None
                and _dist(rs.path_precheck_goal_xy, hinted_goal)
                <= self.nav2_path_precheck_rosout_goal_match_tol_m
            ):
                self.get_logger().warn(
                    f'[{rs.name}] Planner reported no valid path via rosout '
                    f'for current precheck goal ({hinted_goal[0]:.2f}, {hinted_goal[1]:.2f}); '
                    'blacklisting immediately'
                )
                self._abort_path_precheck(
                    rs,
                    cancel_action=True,
                    blacklist_xy=rs.path_precheck_goal_xy,
                    cooldown=not rs.degraded_active,
                )
                rs.precheck_transient_goal_xy = None
                rs.precheck_transient_repeat_count = 0
                return
            # Secondary path: if precheck already ended but hint is fresh, mark
            # this goal region as bad so it is not reselected immediately.
            if not rs.degraded_active:
                self._append_blacklist_if_distant(rs, hinted_goal)
        except Exception:
            # Never let rosout parsing faults tear down the explorer.
            return

    def _matches_recent_planner_no_path_hint(
        self, rs: RobotState, goal_xy: Optional[Tuple[float, float]]
    ) -> bool:
        if goal_xy is None:
            return False
        hint = self._planner_no_path_hints.get(rs.name)
        if hint is None:
            return False
        hinted_xy, ts = hint
        now = self.get_clock().now().nanoseconds / 1e9
        if (
            self.nav2_path_precheck_rosout_no_path_window_sec > 0.0
            and (now - ts) > self.nav2_path_precheck_rosout_no_path_window_sec
        ):
            return False
        return (
            _dist(goal_xy, hinted_xy)
            <= self.nav2_path_precheck_rosout_goal_match_tol_m
        )

    def _planner_lifecycle_callback(self, msg: TransitionEvent, robot: str):
        rs = self.robots.get(robot)
        if rs is None:
            return
        try:
            label = str(msg.goal_state.label or '').strip().lower()
        except Exception:
            label = ''
        rs.planner_state_label = label
        rs.planner_state_update_time = self.get_clock().now().nanoseconds / 1e9

    def _enrich_transient_precheck_reason(
        self, rs: RobotState, goal_xy: Optional[Tuple[float, float]], reason: str
    ) -> str:
        now = self.get_clock().now().nanoseconds / 1e9
        if (
            self.nav2_path_precheck_use_planner_lifecycle_hint
            and rs.planner_state_label
            and rs.planner_state_update_time > 0.0
            and (
                self.nav2_path_precheck_planner_state_stale_sec <= 0.0
                or (now - rs.planner_state_update_time)
                <= self.nav2_path_precheck_planner_state_stale_sec
            )
            and rs.planner_state_label != 'active'
        ):
            return (
                'precheck_transient_planner_lifecycle_'
                f'{rs.planner_state_label} ({reason})'
            )
        if (
            rs.last_tf_update_time > 0.0
            and (now - rs.last_tf_update_time) > self.tf_stale_timeout_sec
        ):
            return (
                'precheck_transient_tf_stale '
                f'(tf_age={now - rs.last_tf_update_time:.2f}s, '
                f'threshold={self.tf_stale_timeout_sec:.2f}s, base={reason})'
            )
        if (
            self.nav2_path_precheck_use_rosout_no_path_hint
            and goal_xy is not None
            and rs.name in self._planner_no_path_hints
        ):
            hinted_xy, ts = self._planner_no_path_hints[rs.name]
            age = now - ts
            return (
                'precheck_transient_rosout_no_match '
                f'(hint_xy={hinted_xy}, hint_age={age:.2f}s, goal={goal_xy}, base={reason})'
            )
        return reason

    def _planner_state_recently_active(self, rs: RobotState) -> bool:
        if not rs.planner_state_label:
            return False
        if rs.planner_state_label != 'active':
            return False
        if rs.planner_state_update_time <= 0.0:
            return False
        if self.nav2_path_precheck_planner_state_stale_sec <= 0.0:
            return True
        now = self.get_clock().now().nanoseconds / 1e9
        return (
            now - rs.planner_state_update_time
            <= self.nav2_path_precheck_planner_state_stale_sec
        )

    def _publish_status(self, state: str):
        msg = String()
        # Simple text state with per-robot summary; consumer tools can parse if desired.
        per_robot = ', '.join(
            f'{rs.name}:reached={rs.goals_reached},failed={rs.goals_failed},'
            f'goal_status={rs.goal_status},goal_active={rs.goal_active}'
            for rs in self.robots.values()
        )
        msg.data = f'state={state}; robots=[{per_robot}]'
        self.status_pub.publish(msg)

    def _resume_callback(self, msg: Bool):
        if msg.data:
            if self.paused:
                self.get_logger().info('Resume requested on control topic')
            self.paused = False
            self._publish_status('IN_PROGRESS')
        else:
            if not self.paused:
                self.get_logger().info('Pause requested on control topic — cancelling active goals')
                for rs in self.robots.values():
                    if rs.goal_active:
                        self._cancel_goal(rs, reason='pause')
            self.paused = True
            self._publish_status('PAUSED')

    def _event_info(self, message: str) -> None:
        if self.terminal_event_log_mode == 'verbose':
            self.get_logger().info(message)

    def _emit_terminal_summary(self) -> None:
        if not self.terminal_summary_enable:
            return
        parts: List[str] = []
        for rs in self.robots.values():
            prev_reached, prev_failed = self._last_summary_counts.get(
                rs.name, (rs.goals_reached, rs.goals_failed)
            )
            d_reached = rs.goals_reached - prev_reached
            d_failed = rs.goals_failed - prev_failed
            self._last_summary_counts[rs.name] = (rs.goals_reached, rs.goals_failed)

            state = 'idle'
            if rs.degraded_active:
                state = f'degraded:{rs.degraded_reason or "unknown"}'
            elif rs.path_precheck_in_progress:
                state = 'precheck'
            elif rs.goal_active:
                state = 'goal_active'
            elif rs.goal_pending:
                state = 'goal_pending'

            pos = 'na'
            if rs.position is not None:
                pos = f'({rs.position[0]:.2f},{rs.position[1]:.2f})'

            parts.append(
                f'{rs.name}[{state}] reached={rs.goals_reached}(+{d_reached}) '
                f'failed={rs.goals_failed}(+{d_failed}) status={rs.goal_status} '
                f'cancel={rs.last_cancel_reason or "-"} pos={pos}'
            )
        if parts:
            self.get_logger().info('Summary: ' + ' | '.join(parts))

    # -----------------------------------------------------------------------
    # Main planning loop
    # -----------------------------------------------------------------------

    def _plan_tick(self):
        # Decide whether we should operate in global or local mode on this tick.
        use_global = False
        if self.mode == 'global_only':
            use_global = True
        elif self.mode == 'local_only':
            use_global = False
        else:
            # auto mode: follow merge_state flag
            use_global = self.in_global_phase

        if use_global and self.latest_global_map is None:
            if not self._logged_waiting_for_map:
                mode = (
                    'single_robot_offloaded_nav2'
                    if self.single_robot_offloaded_nav2
                    else 'multi_robot'
                )
                self.get_logger().info(
                    f'Waiting for map on {self.map_topic} (mode={mode})...')
                self._logged_waiting_for_map = True
                self._publish_status('WAITING_FOR_MAP')
            return
        if not use_global and not self.latest_local_maps:
            if not self._logged_waiting_for_map:
                self.get_logger().info(
                    'Waiting for per-robot local maps in local exploration mode...')
                self._logged_waiting_for_map = True
                self._publish_status('WAITING_FOR_MAP')
            return

        self._explorer_planning_ready = True

        if self.paused:
            # Still keep TF and map up to date, but do not assign new goals.
            self._update_robot_positions()
            return

        # update robot positions from TF
        self._update_robot_positions()

        now = self.get_clock().now().nanoseconds / 1e9
        # Physical motion progress: update when TF pose displacement exceeds
        # threshold, regardless of whether motion is directly goal-seeking.
        for rs in self.robots.values():
            if rs.position is None:
                continue
            if rs.last_motion_sample_pose is None:
                rs.last_motion_sample_pose = rs.position
                rs.last_motion_sample_time = now
                continue
            motion_delta = _dist(rs.position, rs.last_motion_sample_pose)
            if motion_delta >= self.motion_progress_min_delta_m:
                rs.last_motion_time = now
                rs.last_motion_sample_pose = rs.position
                rs.last_motion_sample_time = now

        # Pose-based progress: advance last_progress_time when the robot moves toward
        # the goal in the map frame, only when Nav2 path-distance feedback is absent.
        # Straight-line distance often *increases* during a necessary detour (corners,
        # doorways); in that case distance_remaining (see _goal_feedback_callback)
        # is the correct progress signal.
        for rs in self.robots.values():
            if not rs.goal_active or rs.goal_position is None or rs.position is None:
                continue
            if rs.last_distance_remaining is not None:
                continue
            d = _dist(rs.position, rs.goal_position)
            if rs.last_dist_to_goal is None:
                rs.last_dist_to_goal = d
            elif rs.last_dist_to_goal - d >= self.progress_min_delta:
                rs.last_progress_time = now
                rs.last_dist_to_goal = d
            elif d < rs.last_dist_to_goal:
                rs.last_dist_to_goal = d

        # Tacit success: Nav2 often keeps an active goal while the robot is at
        # the pose (no SUCCEEDED). Use pose + optional distance_remaining from
        # feedback; after hold_sec, cancel Nav2 and count success in result cb.
        if self.tacit_goal_success_hold_sec > 0.0:
            for rs in self.robots.values():
                if (
                    not rs.goal_active
                    or rs.goal_pending
                    or rs.goal_position is None
                    or rs.position is None
                    or rs.goal_handle is None
                ):
                    rs.near_goal_since = None
                    continue
                d = _dist(rs.position, rs.goal_position)
                if self._at_nav_goal_for_tacit_hold(rs, d):
                    rs.last_progress_time = now
                    hold_needed = float(self.tacit_goal_success_hold_sec)
                    if (
                        self.goal_arrival_probe_use_collision_ahead
                        and rs.nav2_collision_ahead
                        and self.goal_arrival_probe_enable
                        and self.goal_arrival_probe_before_tacit_success
                        and not rs.homing_active
                    ):
                        hold_needed = min(
                            hold_needed,
                            float(self.goal_arrival_collision_tacit_hold_sec),
                        )
                    if rs.near_goal_since is None:
                        rs.near_goal_since = now
                    elif now - rs.near_goal_since >= hold_needed:
                        probe_gate = (
                            self.goal_arrival_probe_enable
                            and not rs.homing_active
                            and self.goal_arrival_probe_before_tacit_success
                        )
                        if probe_gate:
                            if rs.arrival_probe_in_progress:
                                continue
                            if not rs.arrival_tacit_probe_attempted:
                                rs.arrival_tacit_probe_attempted = True
                                if self._begin_arrival_probe(rs, tacit_followup=True):
                                    continue
                        dr = rs.last_distance_remaining
                        self.get_logger().info(
                            f'[{rs.name}] At goal for {self.tacit_goal_success_hold_sec:.1f}s '
                            f'(pose_dist={d:.2f}m remain={dr}) — canceling Nav2 and '
                            'counting success'
                        )
                        rs.tacit_success_pending_cancel = True
                        rs.last_cancel_reason = 'tacit_success'
                        rs.ignore_cooldown_on_next_cancel = True
                        try:
                            rs.goal_handle.cancel_goal_async()
                        except Exception:
                            pass
                        rs.near_goal_since = None
                else:
                    rs.near_goal_since = None

        # Near-goal stagnation: outside tacit radius, low remaining distance signal,
        # but no planner progress — run arrival probe once per NavigateToPose leg.
        if self.goal_arrival_probe_enable:
            for rs in self.robots.values():
                if not rs.goal_active or rs.goal_pending or rs.homing_active:
                    continue
                if rs.goal_position is None or rs.position is None:
                    continue
                if rs.arrival_probe_in_progress or rs.path_precheck_in_progress:
                    continue
                d = _dist(rs.position, rs.goal_position)
                if d <= self.tacit_goal_success_radius_m:
                    continue
                if d > self.goal_arrival_probe_max_pose_dist_m:
                    continue
                dr = rs.last_distance_remaining
                if (
                    dr is not None
                    and math.isfinite(dr)
                    and dr > self.goal_arrival_probe_max_distance_remaining_m
                ):
                    continue
                eff_stag = float(self.goal_arrival_probe_stagnation_sec)
                if (
                    self.goal_arrival_probe_use_collision_ahead
                    and rs.nav2_collision_ahead
                ):
                    eff_stag = min(
                        eff_stag,
                        float(self.goal_arrival_probe_stagnation_sec_collision),
                    )
                stagnation_age = now - rs.last_progress_time
                if stagnation_age < eff_stag:
                    continue
                if rs.arrival_stagnation_probe_fired_seq == rs.active_goal_seq:
                    continue
                if self._begin_arrival_probe(rs, tacit_followup=False):
                    rs.arrival_stagnation_probe_fired_seq = rs.active_goal_seq

        # Stall watchdog: cancel only when movement is expected but TF pose has
        # remained effectively stagnant for progress_timeout seconds. Skip while
        # degraded hold is active so transient data problems don't cause
        # blacklisting or unnecessary cancels.
        for rs in self.robots.values():
            if not rs.goal_active or rs.last_goal_time <= 0:
                continue
            if self.degraded_hold_enabled and rs.degraded_active:
                continue
            if self.retrace_hold_enabled and rs.nav2_retrace_active:
                hold_age = (
                    now - rs.nav2_retrace_since
                    if rs.nav2_retrace_since > 0.0
                    else 0.0
                )
                if (
                    self.retrace_hold_max_sec > 0.0
                    and hold_age > self.retrace_hold_max_sec
                ):
                    self.get_logger().warn(
                        f'[{rs.name}] retrace hold timeout after {hold_age:.1f}s; '
                        'resuming normal watchdog behavior'
                    )
                    rs.nav2_retrace_active = False
                    rs.nav2_retrace_since = 0.0
                else:
                    if rs.goal_position is not None and rs.held_goal_position is None:
                        rs.held_goal_position = rs.goal_position
                    continue
            if self.lethal_hold_enabled and rs.nav2_lethal_active:
                hold_age = now - rs.nav2_lethal_since if rs.nav2_lethal_since > 0.0 else 0.0
                if self.lethal_hold_max_sec > 0.0 and hold_age > self.lethal_hold_max_sec:
                    self.get_logger().warn(
                        f'[{rs.name}] lethal hold timeout after {hold_age:.1f}s; '
                        'resuming normal watchdog behavior'
                    )
                    rs.nav2_lethal_active = False
                    rs.nav2_lethal_since = 0.0
                else:
                    if rs.goal_position is not None and rs.held_goal_position is None:
                        rs.held_goal_position = rs.goal_position
                    continue

            if rs.goal_pending or rs.goal_position is None or rs.position is None:
                continue
            if rs.arrival_probe_in_progress:
                continue

            dist_to_goal = _dist(rs.position, rs.goal_position)
            if dist_to_goal <= self.tacit_goal_success_radius_m:
                # Close enough to goal: let tacit success / Nav2 result handling
                # decide completion instead of watchdog cancellation.
                continue

            pose_stagnation_age = (
                (now - rs.last_motion_time)
                if rs.last_motion_time > 0.0 else float('inf')
            )
            if pose_stagnation_age > self.progress_timeout:
                self.get_logger().warn(
                    f'[{rs.name}] Goal appears stalled for '
                    f'{self.progress_timeout:.0f}s without physical movement '
                    f'(pose_stagnation_age={pose_stagnation_age:.1f}s); '
                    'cancelling and blacklisting'
                )
                self._cancel_goal(rs, reason='stall_watchdog')
                if rs.homing_active:
                    self._return_home_note_failure(rs)
                elif rs.goal_position:
                    rs.blacklist.append(rs.goal_position)
                self._arm_post_failure_cooldown(rs)

        if self.lethal_hold_enabled or self.retrace_hold_enabled:
            for rs in self.robots.values():
                if rs.nav2_lethal_active or rs.nav2_retrace_active:
                    continue
                if rs.held_goal_position is None:
                    continue
                if rs.goal_active or rs.goal_pending:
                    continue
                if not self._robot_may_assign_now(rs, now):
                    continue
                self._resend_held_goal(rs)

        for rs in self.robots.values():
            if not rs.path_precheck_in_progress:
                continue
            if rs.path_precheck_started_time <= 0.0:
                continue
            if (
                now - rs.path_precheck_started_time
                < self.nav2_path_precheck_timeout_sec
            ):
                continue
            goal_xy = rs.path_precheck_world_goal_xy or rs.path_precheck_goal_xy
            goal_txt = (
                f' at ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})'
                if goal_xy is not None else ''
            )
            self.get_logger().warn(
                f'[{rs.name}] Nav2 path precheck timed out after '
                f'{self.nav2_path_precheck_timeout_sec:.1f}s{goal_txt}; '
                'cancelling and skipping goal'
            )
            rs.precheck_timeout_events += 1
            hard, reason = self._note_transient_precheck_abort_and_maybe_escalate(
                rs,
                goal_xy,
                f'timed out after {self.nav2_path_precheck_timeout_sec:.1f}s',
            )
            if hard:
                self.get_logger().warn(
                    f'[{rs.name}] Path precheck timed out repeatedly; '
                    f'blacklisting goal and continuing ({reason})'
                )
                self._abort_path_precheck(
                    rs,
                    cancel_action=True,
                    blacklist_xy=goal_xy,
                    cooldown=True,
                )
                rs.precheck_transient_goal_xy = None
                rs.precheck_transient_repeat_count = 0
            else:
                self._path_precheck_transient_fail(rs, reason)

        for rs in self.robots.values():
            if not rs.arrival_probe_in_progress:
                continue
            if rs.arrival_probe_started_time <= 0.0:
                continue
            if (
                now - rs.arrival_probe_started_time
                < self.goal_arrival_probe_timeout_sec
            ):
                continue
            self.get_logger().warn(
                f'[{rs.name}] Arrival path probe timed out after '
                f'{self.goal_arrival_probe_timeout_sec:.1f}s'
            )
            self._arrival_probe_fail(rs, False)

        # Degraded recovery: if robot has been healthy again for a while,
        # clear degraded flag so normal behaviour resumes.
        if self.degraded_hold_enabled:
            now_recover = self.get_clock().now().nanoseconds / 1e9
            for rs in self.robots.values():
                if not rs.degraded_active:
                    # decay counters slowly over time
                    if (
                        rs.precheck_fail_events > 0
                        and (now_recover - rs.degraded_since)
                        > self.degraded_precheck_window_sec
                    ):
                        rs.precheck_fail_events = 0
                    if (
                        rs.precheck_server_unavailable_events > 0
                        and (now_recover - rs.degraded_since)
                        > self.degraded_precheck_window_sec
                    ):
                        rs.precheck_server_unavailable_events = 0
                    continue
                # require some time since degraded_enter and fresh TF again
                tf_fresh = (
                    rs.last_tf_update_time > 0.0
                    and (now_recover - rs.last_tf_update_time)
                    <= self.tf_stale_timeout_sec
                )
                if (
                    tf_fresh
                    and (now_recover - rs.degraded_since)
                    >= self.degraded_recovery_stable_sec
                ):
                    self.get_logger().info(
                        f'[{rs.name}] Clearing degraded hold '
                        f'(reason={rs.degraded_reason}, '
                        f'stable_for={now_recover - rs.degraded_since:.1f}s)'
                    )
                    rs.degraded_active = False
                    rs.degraded_since = 0.0
                    rs.degraded_reason = ''
                    rs.precheck_fail_events = 0
                    rs.precheck_timeout_events = 0
                    rs.precheck_server_unavailable_events = 0

        if use_global:
            self._global_frontier_step()
        else:
            self._local_frontier_step()

        self._return_home_tick(use_global, now)
        self._maybe_publish_return_home_status()

    def _reset_return_home_state_when_exploration_resumes(self):
        self._return_home_status_published = ''
        self._return_home_no_pose_logged.clear()
        for rs in self.robots.values():
            if rs.homing_active and (rs.goal_active or rs.goal_pending):
                self._cancel_goal(rs, reason='return_home_reset')
            rs.return_home_done = False
            rs.homing_active = False
            rs.return_home_failures = 0

    def _return_home_tick(self, use_global: bool, now: float):
        if not self.return_to_start_enable or not self.exploration_complete:
            return

        for rs in self.robots.values():
            if rs.return_home_done:
                continue
            if rs.start_pose_xy is None:
                if rs.name not in self._return_home_no_pose_logged:
                    self._return_home_no_pose_logged.add(rs.name)
                    self.get_logger().warn(
                        f'[{rs.name}] Return home skipped: start pose was never recorded'
                    )
                rs.return_home_done = True
                continue
            if (
                rs.goal_active
                or rs.goal_pending
                or rs.path_precheck_in_progress
                or rs.arrival_probe_in_progress
            ):
                continue
            if not self._robot_may_assign_now(rs, now):
                continue
            if rs.return_home_failures >= self.return_to_start_max_attempts:
                self.get_logger().warn(
                    f'[{rs.name}] Return home abandoned after '
                    f'{self.return_to_start_max_attempts} failed attempt(s)'
                )
                rs.return_home_done = True
                continue
            sx, sy = rs.start_pose_xy
            if rs.position is not None:
                if _dist(rs.position, rs.start_pose_xy) <= self.return_to_start_skip_if_within_m:
                    self.get_logger().info(
                        f'[{rs.name}] Within {self.return_to_start_skip_if_within_m:.2f}m of '
                        'start — skipping NavigateToPose for return home'
                    )
                    rs.return_home_done = True
                    continue
            if use_global:
                self._current_goal_map = self.latest_global_map
            else:
                self._current_goal_map = self.latest_local_maps.get(rs.name)
            if self._current_goal_map is None:
                continue

            rs.repeat_goal_count = 0
            rs.homing_active = True
            proxy = Frontier(
                centroid_world=(sx, sy),
                size=max(1, self.min_frontier_cells_for_goal),
                size_m=max(self.min_frontier_size, 0.01),
                cells=max(1, self.min_frontier_cells_for_goal),
                indices=None,
                pose_frame_id=str(self.world_frame),
            )
            self.get_logger().info(
                f'[{rs.name}] Return home: sending goal to start '
                f'({sx:.2f}, {sy:.2f})'
            )
            self._send_goal(rs, proxy)
            if (
                not rs.goal_active
                and not rs.goal_pending
                and not rs.path_precheck_in_progress
                and not rs.arrival_probe_in_progress
            ):
                rs.homing_active = False

    def _maybe_publish_return_home_status(self):
        if not self.return_to_start_enable or not self.exploration_complete:
            return
        if all(rs.return_home_done for rs in self.robots.values()):
            if self._return_home_status_published != 'HOME_COMPLETE':
                self._return_home_status_published = 'HOME_COMPLETE'
                self._publish_status('HOME_COMPLETE')
            return
        finishing_explore = any(
            rs.goal_active and not rs.homing_active and not rs.return_home_done
            for rs in self.robots.values()
        )
        if finishing_explore:
            return
        if self._return_home_status_published not in ('HOMING', 'HOME_COMPLETE'):
            self._return_home_status_published = 'HOMING'
            self._publish_status('HOMING')

    def _global_frontier_step(self):
        m = self.latest_global_map
        # Use the current global map for all goal selection in this tick.
        self._current_goal_map = m
        now = self.get_clock().now().nanoseconds / 1e9
        n_idle = sum(
            1 for rs in self.robots.values()
            if not rs.goal_active and not rs.goal_pending and rs.position is not None
            and self._robot_may_assign_now(rs, now))

        gframe = (m.header.frame_id or '').strip() or str(self.world_frame)
        frontiers = detect_frontiers(
            m.data,
            m.info.width,
            m.info.height,
            m.info.resolution,
            m.info.origin.position.x,
            m.info.origin.position.y,
            self.min_frontier_size,
            pose_frame_id=gframe,
        )

        if not frontiers:
            if not self.exploration_complete:
                self.get_logger().info(
                    'No frontiers remaining — exploration complete!')
                self.exploration_complete = True
                self._publish_status('COMPLETE')
            elif n_idle > 0:
                if now - self._last_no_frontier_idle_log_time >= 15.0:
                    self._last_no_frontier_idle_log_time = now
                    self.get_logger().warning(
                        'Still no frontier regions on merged map while '
                        f'{n_idle} robot(s) are idle. If unexplored space '
                        'remains in RViz, confirm the map uses value -1 for '
                        'unknown and 0 for free, and that free cells border '
                        'unknown (frontier definition).'
                    )
            return

        if self.exploration_complete:
            self.get_logger().info('New frontiers appeared — resuming exploration')
            self._reset_return_home_state_when_exploration_resumes()
        self.exploration_complete = False

        self.get_logger().debug(
            f'{len(frontiers)} frontiers, {n_idle} idle robot(s)')
        if n_idle > 0:
            self._publish_status('IN_PROGRESS')

        # publish visualisation
        if self.visualize:
            self._publish_frontier_markers(frontiers)

        for rs in self.robots.values():
            self._maybe_retarget_active_robot(rs, frontiers, now)

        # assign frontiers to idle robots
        assignments, relaxed_sep = assign_frontiers(
            self.robots,
            frontiers,
            self.potential_scale,
            self.gain_scale,
            self.nearby_penalty_dist,
            self.min_goal_separation,
            self.blacklist_radius,
            utility_penalty_fn=lambda rname, fr: self._frontier_penalty(self.robots[rname], fr),
            skip_frontier_fn=lambda rname, fr: self._should_skip_frontier_for_assignment(
                self.robots[rname], fr),
            robot_ready_fn=lambda rname: self._robot_may_assign_now(
                self.robots[rname], now),
            goal_dist_fn=self._goal_dist_for_assignment,
            assignment_robot_order_fn=self._assignment_robot_order,
        )
        if not assignments and n_idle > 0 and frontiers:
            if now - self._last_no_assignment_log_time >= 15.0:
                self._last_no_assignment_log_time = now
                self._log_why_no_assignment(frontiers, n_idle, now)
        if relaxed_sep and assignments:
            self.get_logger().info(
                'Relaxed min_goal_separation for assignment (robot on/near '
                'frontier after a goal); repeat-goal blacklist still applies'
            )

        for rname, fi in assignments.items():
            fr = frontiers[fi]
            self._send_goal(self.robots[rname], fr)

    def _local_frontier_step(self):
        # Per-robot frontier detection: each robot plans only on its own local
        # map. This keeps exploration local while maps have not yet been
        # confidently merged.
        any_frontiers = False
        any_idle = False

        for name, rs in self.robots.items():
            if rs.position is None:
                continue
            lm = self.latest_local_maps.get(name)
            if lm is None:
                continue
            # Use this robot's local map for goal selection while we process
            # its frontiers in this tick.
            self._current_goal_map = lm
            self._map_size_x = lm.info.width * lm.info.resolution
            self._map_size_y = lm.info.height * lm.info.resolution
            lframe = (lm.header.frame_id or '').strip() or f'{name}/map'
            frontiers = detect_frontiers(
                lm.data,
                lm.info.width,
                lm.info.height,
                lm.info.resolution,
                lm.info.origin.position.x,
                lm.info.origin.position.y,
                self.min_frontier_size,
                pose_frame_id=lframe,
            )
            if not frontiers:
                continue
            any_frontiers = True

            if self.visualize:
                self._publish_frontier_markers(frontiers)

            now = self.get_clock().now().nanoseconds / 1e9
            self._maybe_retarget_active_robot(rs, frontiers, now)

            if rs.goal_active or rs.goal_pending:
                continue
            if not self._robot_may_assign_now(rs, now):
                continue
            any_idle = True

            # Assign a frontier for this single robot using the same utility
            # function but restricted to this robot and these frontiers.
            sub_assignments, relaxed_sep = assign_frontiers(
                {name: rs},
                frontiers,
                self.potential_scale,
                self.gain_scale,
                self.nearby_penalty_dist,
                self.min_goal_separation,
                self.blacklist_radius,
                utility_penalty_fn=lambda _rname, fr: self._frontier_penalty(rs, fr),
                skip_frontier_fn=lambda _rname, fr: self._should_skip_frontier_for_assignment(rs, fr),
                robot_ready_fn=lambda _rname: self._robot_may_assign_now(rs, now),
                goal_dist_fn=lambda rname, fr: self._goal_dist_for_assignment(rname, fr),
            )
            if relaxed_sep and name in sub_assignments:
                self.get_logger().info(
                    f'[{name}] Relaxed min_goal_separation for assignment (robot '
                    'on/near frontier after a goal); repeat-goal blacklist still applies'
                )
            if name not in sub_assignments and frontiers:
                # Mirror global-mode diagnostics so local mode does not appear
                # "stuck" when frontiers exist but all candidates are gated.
                self._log_why_no_assignment(frontiers, 1, now)
            if name in sub_assignments:
                fr = frontiers[sub_assignments[name]]
                self._send_goal(rs, fr)

        if any_frontiers:
            if self.exploration_complete:
                self.get_logger().info(
                    'New frontiers appeared — resuming exploration (local mode)')
                self._reset_return_home_state_when_exploration_resumes()
            self.exploration_complete = False

        if not any_frontiers:
            if not self.exploration_complete and any_idle:
                self.get_logger().info(
                    'No frontiers remaining on any local map — exploration complete!')
                self.exploration_complete = True
                self._publish_status('COMPLETE')

    def _merge_state_callback(self, msg: StringMsg):
        state = (msg.data or '').strip().upper()
        self._merge_state_last = state or 'NO_OVERLAP'
        # Hysteresis only applies in auto mode; explicit local/global modes
        # should honor user intent directly.
        if self.mode != 'auto':
            if state == 'MERGED' and not self.in_global_phase:
                self.get_logger().info(
                    'Merge state reported MERGED — switching to global exploration on merged map')
                self.in_global_phase = True
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if state == 'MERGED':
            self._merge_nonmerged_since = 0.0
            if self._merge_merged_since <= 0.0:
                self._merge_merged_since = now
                if not self.in_global_phase and self.merge_stable_hold_sec > 0.0:
                    self.get_logger().info(
                        f'Merge state reported MERGED; waiting '
                        f'{self.merge_stable_hold_sec:.1f}s before global switch '
                        '(stability hold)')
            if (
                not self.in_global_phase
                and (now - self._merge_merged_since) >= self.merge_stable_hold_sec
            ):
                self.get_logger().info(
                    'Merge state remained MERGED — switching to global exploration on merged map')
                self.in_global_phase = True
            return

        # Non-merged states (PARTIAL / NO_OVERLAP / unknown).
        self._merge_merged_since = 0.0
        if self._merge_nonmerged_since <= 0.0:
            self._merge_nonmerged_since = now
        if (
            self.in_global_phase
            and (now - self._merge_nonmerged_since) >= self.merge_degrade_grace_sec
        ):
            self.in_global_phase = False
            self.get_logger().warning(
                f'Merge state degraded to {self._merge_state_last}; '
                'falling back to local exploration until MERGED is stable again')

    # -----------------------------------------------------------------------
    # TF helpers
    # -----------------------------------------------------------------------

    def _update_robot_positions(self):
        now = self.get_clock().now().nanoseconds / 1e9
        for rs in self.robots.values():
            # In all modes, treat the robot base frame as "<robot>/base_footprint".
            # TF chain on global /tf:
            #   map -> <robot>/map -> <robot>/odom -> <robot>/base_footprint
            # (map_merge in multi-robot; single_robot_world_tf_bridge + SLAM in
            # single-robot central mode.)
            base_frame = f'{rs.name}/{self.robot_base_frame}'
            try:
                t = self.tf_buffer.lookup_transform(
                    self.world_frame, base_frame, rclpy.time.Time(),
                    timeout=Duration(seconds=0.5))
                rs.position = (
                    t.transform.translation.x,
                    t.transform.translation.y,
                )
                q = t.transform.rotation
                siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                rs.yaw = math.atan2(siny_cosp, cosy_cosp)
                rs.last_tf_update_time = now
                rs.tf_stale_warned = False
                if self._explorer_planning_ready and rs.start_pose_xy is None:
                    rs.start_pose_xy = (
                        t.transform.translation.x,
                        t.transform.translation.y,
                    )
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                # Keep last known position only briefly; stale TF should not
                # continue driving assignment/blacklist decisions.
                if (
                    rs.last_tf_update_time > 0.0
                    and (now - rs.last_tf_update_time) > self.tf_stale_timeout_sec
                ):
                    rs.position = None
                    rs.yaw = None
                    if not rs.tf_stale_warned:
                        tf_age = now - rs.last_tf_update_time
                        self.get_logger().warn(
                            f'[{rs.name}] TF stale -> robot temporarily unassignable '
                            f'(tf_age={tf_age:.2f}s, threshold={self.tf_stale_timeout_sec:.2f}s)'
                        )
                        rs.tf_stale_warned = True
                    if (
                        self.degraded_hold_enabled
                        and not rs.degraded_active
                        and (now - rs.last_tf_update_time) >= self.degraded_tf_hold_sec
                    ):
                        rs.degraded_active = True
                        rs.degraded_since = now
                        rs.degraded_reason = 'tf_stale'
                        self.get_logger().warn(
                            f'[{rs.name}] Entering degraded hold due to stale TF '
                            f'(age={now - rs.last_tf_update_time:.2f}s)'
                        )

    # -----------------------------------------------------------------------
    # Nav2 dispatch frame (Path 2: goals in <robot>/map)
    # -----------------------------------------------------------------------

    def _nav_dispatch_frame_id(self, rs: RobotState) -> str:
        if not self.dispatch_nav_goals_in_robot_map_frame:
            return str(self.world_frame)
        return self.nav_goal_frame_pattern.format(robot=rs.name)

    def _goal_pose_source_frame(self, frontier: Optional[Frontier]) -> str:
        if frontier is not None and frontier.pose_frame_id:
            return str(frontier.pose_frame_id)
        return str(self.world_frame)

    def _transform_plan_xy(
        self,
        rs: RobotState,
        from_frame: str,
        x: float,
        y: float,
        to_frame: str,
    ) -> Optional[Tuple[float, float]]:
        """Transform a 2D plan point between TF frames (latest stamp)."""
        if not from_frame or not to_frame or from_frame == to_frame:
            return (float(x), float(y))
        ps = PoseStamped()
        ps.header.frame_id = from_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0
        try:
            out = self.tf_buffer.transform(
                ps,
                to_frame,
                timeout=Duration(seconds=self.nav_goal_tf_transform_timeout_sec),
            )
            return (float(out.pose.position.x), float(out.pose.position.y))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(
                f'[{rs.name}] TF plan point {from_frame!r} -> {to_frame!r} '
                f'failed: {exc}'
            )
            return None

    def _pose_stamped_for_nav_dispatch(
        self,
        rs: RobotState,
        source_frame: str,
        x: float,
        y: float,
        yaw: float,
    ) -> Optional[PoseStamped]:
        """Build a PoseStamped for Nav2 in the robot dispatch frame (Path 2)."""
        dst = self._nav_dispatch_frame_id(rs)
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.x = 0.0
        ps.pose.orientation.y = 0.0
        ps.pose.orientation.z = math.sin(yaw * 0.5)
        ps.pose.orientation.w = math.cos(yaw * 0.5)
        if not self.dispatch_nav_goals_in_robot_map_frame:
            ps.header.frame_id = str(self.world_frame)
            return ps
        ps.header.frame_id = source_frame
        if source_frame == dst:
            ps.header.frame_id = dst
            return ps
        try:
            return self.tf_buffer.transform(
                ps,
                dst,
                timeout=Duration(seconds=self.nav_goal_tf_transform_timeout_sec),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(
                f'[{rs.name}] TF goal transform {source_frame!r} -> {dst!r} failed: {exc}'
            )
            return None

    # -----------------------------------------------------------------------
    # Nav2 goal management
    # -----------------------------------------------------------------------

    def _world_to_cell(self, m: OccupancyGrid, x: float, y: float) -> Optional[Tuple[int, int]]:
        res = m.info.resolution
        ox = m.info.origin.position.x
        oy = m.info.origin.position.y
        cx = int((x - ox) / res)
        cy = int((y - oy) / res)
        if cx < 0 or cy < 0 or cx >= m.info.width or cy >= m.info.height:
            return None
        return cx, cy

    def _goal_cell_cost(self, m: Optional[OccupancyGrid], x: float, y: float) -> int:
        if m is None:
            return 0
        cell = self._world_to_cell(m, x, y)
        if cell is None:
            return 100
        cx, cy = cell
        idx = cy * m.info.width + cx
        return int(m.data[idx])

    def _goal_clearance_m(self, m: Optional[OccupancyGrid], x: float, y: float) -> float:
        if m is None:
            return self.goal_clearance_radius_m
        cell = self._world_to_cell(m, x, y)
        if cell is None:
            return 0.0
        cx, cy = cell
        res = m.info.resolution
        radius_cells = max(1, int(self.goal_clearance_radius_m / res))
        grid = np.array(m.data, dtype=np.int16).reshape((m.info.height, m.info.width))
        min_clearance = self.goal_clearance_radius_m
        y0 = max(0, cy - radius_cells)
        y1 = min(m.info.height - 1, cy + radius_cells)
        x0 = max(0, cx - radius_cells)
        x1 = min(m.info.width - 1, cx + radius_cells)
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                if grid[yy, xx] >= OCCUPIED_THRESH:
                    d = math.hypot((xx - cx) * res, (yy - cy) * res)
                    if d < min_clearance:
                        min_clearance = d
        return min_clearance

    def _frontier_unknown_adjacency_counts(
        self, grid: np.ndarray, ys: np.ndarray, xs: np.ndarray
    ) -> np.ndarray:
        """Per frontier cell: count of unknown 4-neighbours (prefers deeper free cells)."""
        h, w = grid.shape
        unk = grid < 0
        yi = ys.astype(np.int32)
        xi = xs.astype(np.int32)
        n = len(yi)
        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            cy, cx = int(yi[i]), int(xi[i])
            c = 0
            if cy > 0 and unk[cy - 1, cx]:
                c += 1
            if cy < h - 1 and unk[cy + 1, cx]:
                c += 1
            if cx > 0 and unk[cy, cx - 1]:
                c += 1
            if cx < w - 1 and unk[cy, cx + 1]:
                c += 1
            out[i] = float(c)
        return out

    def _clearance_penalty(self, m: Optional[OccupancyGrid], x: float, y: float) -> float:
        clearance = self._goal_clearance_m(m, x, y)
        shortfall = max(0.0, self.goal_clearance_radius_m - clearance)
        return self.goal_clearance_weight * shortfall

    def _goal_passes_safety_gate(self, m: Optional[OccupancyGrid], x: float, y: float) -> bool:
        cost = self._goal_cell_cost(m, x, y)
        # Frontier targets can lie on cells that are still marked unknown due to
        # map discretization around the free/unknown boundary. Do not reject
        # solely for unknown cost; keep hard rejection only for high-cost cells.
        if cost >= self.goal_max_cell_cost:
            return False
        clearance = self._goal_clearance_m(m, x, y)
        return clearance >= self.goal_min_clearance_gate_m

    def _log_goal_safety_rejection(
        self,
        rs: RobotState,
        x: float,
        y: float,
        context: str,
    ) -> None:
        m = self._current_goal_map
        cost = self._goal_cell_cost(m, x, y)
        clearance = self._goal_clearance_m(m, x, y)
        reasons: List[str] = []
        if cost < 0:
            reasons.append('unknown_cell')
        elif cost >= self.goal_max_cell_cost:
            reasons.append('high_cost')
        if clearance < self.goal_min_clearance_gate_m:
            reasons.append('low_clearance')
        if not reasons:
            reasons.append('unknown')
        self.get_logger().debug(
            f'[{rs.name}] reject_goal_candidate ({context}): '
            f'goal=({x:.2f}, {y:.2f}); cost={cost}; '
            f'clearance_m={clearance:.2f}; '
            f'thresholds(cost<{self.goal_max_cell_cost}, '
            f'clearance>={self.goal_min_clearance_gate_m:.2f}); '
            f'reason={",".join(reasons)}'
        )

    def _return_home_note_failure(self, rs: RobotState) -> None:
        if not rs.homing_active:
            return
        rs.homing_active = False
        rs.return_home_failures += 1
        self.get_logger().warn(
            f'[{rs.name}] Return home attempt failed '
            f'({rs.return_home_failures}/{self.return_to_start_max_attempts})'
        )

    def _blacklist_or_home_fail(
        self,
        rs: RobotState,
        xy: Optional[Tuple[float, float]],
    ) -> None:
        if xy is None:
            return
        if rs.homing_active:
            self._return_home_note_failure(rs)
            return
        if not self._should_skip_blacklist_due_to_lethal(rs):
            rs.blacklist.append(xy)
        else:
            rs.held_goal_position = xy

    def _clear_blacklist_points_within_radius(
        self,
        rs: RobotState,
        center: Tuple[float, float],
        radius: float,
    ) -> int:
        """Remove blacklist points within ``radius`` of ``center`` (e.g. success
        goal pose). Returns the number of entries removed.
        """
        r = max(0.0, float(radius))
        if r <= 0.0 or not rs.blacklist:
            return 0
        n_before = len(rs.blacklist)
        cx, cy = float(center[0]), float(center[1])
        rs.blacklist = [
            bl for bl in rs.blacklist
            if math.hypot(bl[0] - cx, bl[1] - cy) >= r
        ]
        return n_before - len(rs.blacklist)

    def _append_blacklist_if_distant(
        self,
        rs: RobotState,
        xy: Tuple[float, float],
    ) -> None:
        """Append a blacklist point if nothing already blocks this region
        (within ``blacklist_radius`` of ``xy``).
        """
        if any(_dist((xy[0], xy[1]), bl) < self.blacklist_radius for bl in rs.blacklist):
            return
        if not self._should_skip_blacklist_due_to_lethal(rs):
            rs.blacklist.append((float(xy[0]), float(xy[1])))

    def _apply_post_success_blacklist_policy(self, rs: RobotState) -> None:
        """After a successful exploration leg, relax stale blacklists near the
        success pose, then (optionally) blacklist the visited frontier so the
        same assignment is not re-emitted on the next tick.
        """
        if rs.homing_active:
            return
        if rs.goal_position is None:
            return
        gpx, gpy = rs.goal_position
        if self.blacklist_clear_radius > 0.0:
            removed = self._clear_blacklist_points_within_radius(
                rs, (gpx, gpy), self.blacklist_clear_radius
            )
            if removed > 0:
                self.get_logger().debug(
                    f'[{rs.name}] Cleared {removed} stale blacklist entrie(s) within '
                    f'{self.blacklist_clear_radius:.2f}m of success pose'
                )
        if self.post_success_frontier_avoidance:
            c = rs.last_target_frontier_centroid
            if c is not None:
                self._append_blacklist_if_distant(rs, c)
            # Also avoid immediately re-selecting the exact goal pose, which can
            # differ from centroid due to fallback goal-point strategies.
            self._append_blacklist_if_distant(rs, (gpx, gpy))
        rs.last_target_frontier_centroid = None

    def _arm_post_failure_cooldown(self, rs: RobotState) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        rs.next_assign_allowed_time = now + max(0.0, self.post_failure_cooldown_sec)

    def _should_skip_blacklist_due_to_lethal(self, rs: RobotState) -> bool:
        lethal_skip = (
            self.lethal_hold_enabled
            and self.skip_blacklist_on_nav2_lethal
            and rs.nav2_lethal_active
        )
        retrace_skip = (
            self.retrace_hold_enabled
            and rs.nav2_retrace_active
        )
        return lethal_skip or retrace_skip

    def _resend_held_goal(self, rs: RobotState) -> None:
        goal = rs.held_goal_position
        if goal is None:
            return
        gx, gy = goal
        self.get_logger().info(
            f'[{rs.name}] Nav2 lethal cleared; resending held goal ({gx:.2f}, {gy:.2f})'
        )
        rs.repeat_goal_count = 0
        rs.last_goal_world = None
        proxy_frontier = Frontier(
            centroid_world=(gx, gy),
            size=max(1, self.min_frontier_cells_for_goal),
            size_m=max(self.min_frontier_size, 0.01),
            cells=max(1, self.min_frontier_cells_for_goal),
            indices=None,
            pose_frame_id=str(self.world_frame),
        )
        self._send_goal(rs, proxy_frontier)
        if rs.goal_active or rs.goal_pending:
            rs.held_goal_position = None

    def _nav2_lethal_callback(self, msg: Bool, robot_name: str):
        rs = self.robots.get(robot_name)
        if rs is None:
            return
        lethal_now = bool(msg.data)
        now = self.get_clock().now().nanoseconds / 1e9
        if lethal_now == rs.nav2_lethal_active:
            return
        rs.nav2_lethal_active = lethal_now
        if lethal_now:
            rs.nav2_lethal_since = now
            if rs.goal_position is not None:
                rs.held_goal_position = rs.goal_position
            self.get_logger().warn(
                f'[{rs.name}] Nav2 reports lethal-space state; holding current goal'
            )
            return
        rs.nav2_lethal_since = 0.0
        if rs.held_goal_position is not None:
            rs.next_assign_allowed_time = min(
                rs.next_assign_allowed_time,
                now + max(0.0, self.lethal_retry_delay_sec),
            )
            self.get_logger().info(
                f'[{rs.name}] Nav2 lethal-space state cleared'
            )

    def _nav2_collision_ahead_callback(self, msg: Bool, robot_name: str):
        rs = self.robots.get(robot_name)
        if rs is None:
            return
        rs.nav2_collision_ahead = bool(msg.data)

    def _link_retreat_callback(self, msg: Bool, robot_name: str):
        rs = self.robots.get(robot_name)
        if rs is None:
            return
        active = bool(msg.data)
        if active == rs.link_retreat_active:
            return
        rs.link_retreat_active = active
        now = self.get_clock().now().nanoseconds / 1e9
        if not active:
            self.get_logger().info(f'[{rs.name}] link retreat cleared; resuming goals')
            return
        self.get_logger().warn(f'[{rs.name}] link retreat active; pausing new goals')
        undo = cancel_blacklist_to_undo(
            now, rs.last_cancel_blacklist_time, rs.last_cancel_blacklist_point,
            self.link_retreat_unblacklist_window_sec)
        if undo is not None and undo in rs.blacklist:
            rs.blacklist.remove(undo)
            self.get_logger().info(
                f'[{rs.name}] un-blacklisted {undo}: cancelled by link_watchdog')
        rs.last_cancel_blacklist_point = None

    def _nav2_retrace_callback(self, msg: Bool, robot_name: str):
        rs = self.robots.get(robot_name)
        if rs is None:
            return
        retrace_now = bool(msg.data)
        now = self.get_clock().now().nanoseconds / 1e9
        if retrace_now == rs.nav2_retrace_active:
            return
        rs.nav2_retrace_active = retrace_now
        if retrace_now:
            rs.nav2_retrace_since = now
            if rs.goal_position is not None:
                rs.held_goal_position = rs.goal_position
            self.get_logger().warn(
                f'[{rs.name}] Nav2 retrace active; holding current goal'
            )
            return
        rs.nav2_retrace_since = 0.0
        if rs.held_goal_position is not None:
            rs.next_assign_allowed_time = min(
                rs.next_assign_allowed_time,
                now + max(0.0, self.lethal_retry_delay_sec),
            )
        self.get_logger().info(f'[{rs.name}] Nav2 retrace cleared')

    def _robot_may_assign_now(self, rs: RobotState, now: float) -> bool:
        if self.retrace_hold_enabled and rs.nav2_retrace_active:
            return False
        if link_retreat_blocks_assignment(
                self.pause_on_link_retreat, rs.link_retreat_active):
            return False
        return now >= rs.next_assign_allowed_time

    def _goal_point_blacklisted(
        self, rs: RobotState, gx: float, gy: float
    ) -> bool:
        return any(
            _dist((gx, gy), bl) < self.blacklist_radius for bl in rs.blacklist)

    def _frontier_too_small_for_nav_goal(self, fr: Frontier) -> bool:
        if self.min_frontier_cells_for_goal <= 0:
            return False
        return fr.size < self.min_frontier_cells_for_goal

    def _should_skip_frontier_for_assignment(
        self, rs: RobotState, fr: Frontier
    ) -> bool:
        """True if the pose Nav2 would receive for this frontier is unusable."""
        if self._frontier_too_small_for_nav_goal(fr):
            self.get_logger().debug(
                f'[{rs.name}] skip frontier: {fr.size} cells < '
                f'min_frontier_cells_for_goal ({self.min_frontier_cells_for_goal})'
            )
            return True
        gx, gy = self._select_goal_point(rs, fr)
        if self._goal_point_blacklisted(rs, gx, gy):
            return True
        if not self._goal_passes_safety_gate(self._current_goal_map, gx, gy):
            return True
        return False

    def _frontier_penalty(self, rs: RobotState, fr: Frontier) -> float:
        gx, gy = self._select_goal_point(rs, fr)
        return self._clearance_penalty(self._current_goal_map, gx, gy)

    def _frontier_utility(self, rs: RobotState, fr: Frontier) -> float:
        if self._frontier_too_small_for_nav_goal(fr):
            return -1e9
        gx, gy = self._select_goal_point(rs, fr)
        if rs.position is None:
            return -1e9
        if not self._goal_passes_safety_gate(self._current_goal_map, gx, gy):
            self._log_goal_safety_rejection(
                rs, gx, gy, context='frontier_utility'
            )
            return -1e9
        d = _dist(rs.position, (gx, gy))
        # Degenerate only when essentially same cell; otherwise penalize
        # closeness so retargeting still works after the robot reaches a goal
        # on/near the frontier (same situation as assign_frontiers relaxed pass).
        degenerate_eps = 0.02
        if d < degenerate_eps:
            return -1e9
        close_penalty = 0.0
        if d < self.min_goal_separation:
            close_penalty = (self.min_goal_separation - d) * 3.0
        gain = self.gain_scale * fr.size_m
        cost = self.potential_scale * d
        penalty = self._clearance_penalty(self._current_goal_map, gx, gy)
        return gain - cost - penalty - close_penalty

    def _best_frontier_for_robot(self, rs: RobotState, frontiers: List[Frontier]) -> Tuple[Optional[Frontier], float]:
        best = None
        best_u = -1e9
        for fr in frontiers:
            if any(_dist(fr.centroid_world, bl) < self.blacklist_radius for bl in rs.blacklist):
                continue
            if self._should_skip_frontier_for_assignment(rs, fr):
                continue
            u = self._frontier_utility(rs, fr)
            if u > best_u:
                best_u = u
                best = fr
        return best, best_u

    def _best_frontier_for_robot_within_anchor(
        self, rs: RobotState, frontiers: List[Frontier]
    ) -> Tuple[Optional[Frontier], float]:
        """Like _best_frontier_for_robot but only frontiers whose goal pose lies
        within retarget_max_offset_from_anchor_m of the leg anchor (first goal),
        or the current goal if the anchor was not set.
        """
        if self.retarget_max_offset_from_anchor_m <= 0.0:
            return self._best_frontier_for_robot(rs, frontiers)
        anchor = rs.retarget_anchor or rs.goal_position
        if anchor is None:
            return self._best_frontier_for_robot(rs, frontiers)
        ax, ay = anchor
        rmax = self.retarget_max_offset_from_anchor_m
        best = None
        best_u = -1e9
        for fr in frontiers:
            if any(
                _dist(fr.centroid_world, bl) < self.blacklist_radius
                for bl in rs.blacklist
            ):
                continue
            if self._should_skip_frontier_for_assignment(rs, fr):
                continue
            gx, gy = self._select_goal_point(rs, fr)
            if _dist((gx, gy), (ax, ay)) > rmax:
                continue
            u = self._frontier_utility(rs, fr)
            if u > best_u:
                best_u = u
                best = fr
        return best, best_u

    def _robot_is_making_progress(self, rs: RobotState, now: float) -> bool:
        if (
            rs.last_motion_time > 0.0
            and (now - rs.last_motion_time) <= self.motion_progress_window_sec
        ):
            return True
        if not rs.goal_active or rs.goal_position is None or rs.position is None:
            return False
        if rs.last_pose_when_goal_sent is None:
            return False
        start_dist = _dist(rs.last_pose_when_goal_sent, rs.goal_position)
        curr_dist = _dist(rs.position, rs.goal_position)
        return (start_dist - curr_dist) >= self.min_progress_before_replan_m

    def _is_robot_stuck(self, rs: RobotState, now: float) -> bool:
        if not self.allow_replan_when_no_progress:
            return False
        if not rs.goal_active:
            return False
        if (
            rs.last_motion_time > 0.0
            and (now - rs.last_motion_time) <= self.motion_progress_window_sec
        ):
            return False
        if rs.last_progress_time <= 0:
            return (now - rs.active_goal_sent_time) > self.max_stuck_time_s
        return (now - rs.last_progress_time) > self.max_stuck_time_s

    def _log_replan_decision(
        self,
        rs: RobotState,
        reason: str,
        allow: bool,
        now: float,
        candidate_goal: Optional[Tuple[float, float]] = None,
    ) -> None:
        elapsed = now - (rs.active_goal_sent_time or now)
        since_progress = now - (rs.last_progress_time or rs.active_goal_sent_time or now)
        since_motion = (
            (now - rs.last_motion_time)
            if rs.last_motion_time > 0.0 else float('inf')
        )
        goal_delta = None
        if candidate_goal is not None and rs.goal_position is not None:
            goal_delta = _dist(candidate_goal, rs.goal_position)
        prefix = 'replan' if allow else 'skip_replan'
        details = (
            f'[{rs.name}] {prefix}: {reason}; elapsed_since_goal={elapsed:.1f}s; '
            f'since_progress={since_progress:.1f}s; '
            f'since_motion={since_motion:.1f}s; '
            f'goal_delta_m={(f"{goal_delta:.2f}" if goal_delta is not None else "n/a")}'
        )
        if allow:
            self.get_logger().info(details)
        else:
            self.get_logger().debug(details)

    def _should_replace_goal(
        self,
        rs: RobotState,
        candidate_goal_xy: Tuple[float, float],
        now: float,
    ) -> Tuple[bool, str]:
        if rs.goal_position is None or not rs.goal_active:
            return True, 'no_active_goal'
        if rs.goal_status in ('aborted', 'canceled', 'failed', 'succeeded'):
            return True, f'current_goal_{rs.goal_status}'
        if (now - rs.active_goal_sent_time) < self.min_goal_replan_interval_s:
            return False, 'cooldown'
        if _dist(candidate_goal_xy, rs.goal_position) < self.min_goal_change_dist_m:
            return False, 'goal_too_similar'
        if self._robot_is_making_progress(rs, now):
            return False, 'robot_making_progress'
        if self._is_robot_stuck(rs, now):
            return True, 'stuck_timeout'
        return False, 'no_replan_condition'

    def _should_replace_goal_opportunity(
        self,
        rs: RobotState,
        candidate_goal_xy: Tuple[float, float],
        now: float,
    ) -> Tuple[bool, str]:
        """Allow mid-route diversion to a better frontier while the robot is
        still advancing; does not use the 'robot_making_progress' block from
        _should_replace_goal (that gate is for *stagnation* retarget only).
        """
        if rs.goal_position is None or not rs.goal_active:
            return True, 'no_active_goal'
        if (now - rs.active_goal_sent_time) < self.min_goal_replan_interval_s:
            return False, 'cooldown'
        if _dist(candidate_goal_xy, rs.goal_position) < self.min_goal_change_dist_m:
            return False, 'goal_too_similar'
        return True, 'opportunity'

    def _robot_recently_showed_progress(self, rs: RobotState, now: float) -> bool:
        """True if either goal-distance or physical motion updated recently."""
        if (
            rs.last_motion_time > 0.0
            and (now - rs.last_motion_time) <= self.motion_progress_window_sec
        ):
            return True
        if rs.last_progress_time <= 0:
            return False
        return (now - rs.last_progress_time) < 4.0

    def _maybe_retarget_active_robot(self, rs: RobotState, frontiers: List[Frontier], now: float):
        if not self.retarget_enable:
            return
        # In auto mode, avoid aggressive in-motion retarget churn while merge
        # state is degraded after we already entered global phase.
        if (
            self.mode == 'auto'
            and self.in_global_phase
            and self._merge_state_last != 'MERGED'
        ):
            return
        if rs.homing_active:
            return
        if self.retrace_hold_enabled and rs.nav2_retrace_active:
            return
        if rs.arrival_probe_in_progress:
            return
        if not rs.goal_active or rs.goal_pending or rs.position is None or rs.goal_position is None:
            return
        if not self._robot_may_assign_now(rs, now):
            return
        if (now - rs.last_retarget_time) < self.retarget_cooldown_sec:
            return

        current_goal = rs.goal_position
        best_fr, best_u = self._best_frontier_for_robot_within_anchor(rs, frontiers)
        if best_fr is None:
            if self.retarget_max_offset_from_anchor_m > 0.0:
                ax = rs.retarget_anchor or rs.goal_position
                if ax is not None:
                    self.get_logger().debug(
                        f'[{rs.name}] Retarget skipped: no frontier with goal within '
                        f'{self.retarget_max_offset_from_anchor_m:.1f}m of anchor '
                        f'({ax[0]:.2f}, {ax[1]:.2f})'
                    )
            return
        new_goal = self._select_goal_point(rs, best_fr)
        if _dist(new_goal, current_goal) < self.retarget_min_goal_shift_m:
            return

        curr_u = -(
            self.potential_scale * _dist(rs.position, current_goal)
            + self._clearance_penalty(self._current_goal_map, current_goal[0], current_goal[1])
        )

        # While the robot is moving toward the current goal, divert if the map
        # now favors a frontier with substantially higher exploration utility.
        if self.retarget_opportunity_enable:
            advancing = (
                self._robot_is_making_progress(rs, now)
                or self._robot_recently_showed_progress(rs, now)
            )
            if advancing and best_u >= (
                curr_u + self.retarget_opportunity_min_utility_gain
            ):
                ok, opp_reason = self._should_replace_goal_opportunity(
                    rs, new_goal, now)
                if ok:
                    self._log_replan_decision(
                        rs, opp_reason, allow=True, now=now, candidate_goal=new_goal)
                    self.get_logger().info(
                        f'[{rs.name}] Retargeting for better frontier while in motion; '
                        f'old=({current_goal[0]:.2f}, {current_goal[1]:.2f}) '
                        f'new=({new_goal[0]:.2f}, {new_goal[1]:.2f})'
                    )
                    rs.ignore_cooldown_on_next_cancel = True
                    self._cancel_goal(rs, reason='retarget_opportunity')
                    rs.last_retarget_time = now
                    self._send_goal(rs, best_fr, from_retarget=True)
                    return
                self._log_replan_decision(
                    rs, opp_reason, allow=False, now=now, candidate_goal=new_goal)

        curr_clearance = self._goal_clearance_m(
            self._current_goal_map, current_goal[0], current_goal[1])
        no_progress_for = now - (rs.last_progress_time or rs.last_goal_time or now)
        risky = curr_clearance < self.retarget_clearance_threshold_m
        stale = no_progress_for > self.retarget_stagnation_sec
        if not risky and not stale:
            return

        if best_u < (curr_u + self.retarget_min_utility_gain):
            return

        should_replace, gate_reason = self._should_replace_goal(rs, new_goal, now)
        if not should_replace:
            self._log_replan_decision(
                rs, gate_reason, allow=False, now=now, candidate_goal=new_goal)
            return

        reason = 'clearance risk' if risky else 'stagnation'
        self._log_replan_decision(
            rs, gate_reason, allow=True, now=now, candidate_goal=new_goal)
        self.get_logger().info(
            f'[{rs.name}] Retargeting active goal due to {reason}; '
            f'old=({current_goal[0]:.2f}, {current_goal[1]:.2f}) '
            f'new=({new_goal[0]:.2f}, {new_goal[1]:.2f})'
        )
        rs.ignore_cooldown_on_next_cancel = True
        self._cancel_goal(rs, reason='retarget_stagnation')
        rs.blacklist.append(current_goal)
        rs.last_retarget_time = now
        self._send_goal(rs, best_fr, from_retarget=True)

    def _pick_frontier_cell_farthest_from_robot(
        self,
        rs: RobotState,
        frontier: Frontier,
        centroid_xy: Tuple[float, float],
        max_offset_from_centroid: Optional[float],
    ) -> Optional[Tuple[float, float]]:
        """Pick a frontier cell that is far from the robot (for exploration).

        If max_offset_from_centroid is set, only cells within that radius of
        the frontier centroid are considered (centroid strategy fallback).
        If None, all frontier cells are considered (farthest_cell strategy).
        Returns None if no suitable cell exists.
        """
        if frontier.indices is None or rs.position is None:
            return None
        m = self._current_goal_map
        if m is None:
            return None
        rx, ry = rs.position
        ys = frontier.indices[:, 0].astype(np.float32)
        xs = frontier.indices[:, 1].astype(np.float32)
        res = m.info.resolution
        ox = m.info.origin.position.x
        oy = m.info.origin.position.y
        wx = ox + (xs + 0.5) * res
        wy = oy + (ys + 0.5) * res

        dx = wx - rx
        dy = wy - ry
        dists = np.hypot(dx, dy)
        grid = np.array(m.data, dtype=np.int16).reshape((m.info.height, m.info.width))
        costs = grid[ys.astype(np.int32), xs.astype(np.int32)]
        unk_adj = self._frontier_unknown_adjacency_counts(grid, ys, xs)
        penalty = max(0.0, self.goal_unknown_neighbor_penalty_m)

        def _score_pick(mask: np.ndarray) -> int:
            if penalty > 0.0:
                scores = dists - penalty * unk_adj
            else:
                scores = dists
            if not np.any(mask):
                return int(np.argmax(scores))
            return int(np.argmax(np.where(mask, scores, -np.inf)))

        valid_cost = (costs >= 0) & (costs < self.goal_max_cell_cost)
        if np.any(valid_cost):
            clearance_ok = np.zeros_like(valid_cost, dtype=bool)
            valid_indices = np.flatnonzero(valid_cost)
            for i in valid_indices:
                clearance = self._goal_clearance_m(m, float(wx[i]), float(wy[i]))
                if clearance >= self.goal_min_clearance_gate_m:
                    clearance_ok[i] = True
            valid_mask = valid_cost & clearance_ok
        else:
            valid_mask = valid_cost

        if max_offset_from_centroid is not None:
            cdx = wx - centroid_xy[0]
            cdy = wy - centroid_xy[1]
            mask_near_centroid = np.hypot(cdx, cdy) <= max_offset_from_centroid
            valid_mask = valid_mask & mask_near_centroid
            if not np.any(valid_mask):
                return None

        mask_far_enough = dists >= max(0.0, self.min_goal_distance)
        safe_far = valid_mask & mask_far_enough
        if np.any(safe_far):
            idx = _score_pick(safe_far)
        elif np.any(valid_mask):
            idx = _score_pick(valid_mask)
        elif np.any(mask_far_enough):
            idx = _score_pick(mask_far_enough)
        else:
            idx = _score_pick(np.ones_like(dists, dtype=bool))

        gx = float(wx[idx])
        gy = float(wy[idx])
        if not np.any(valid_mask):
            self._log_goal_safety_rejection(
                rs, gx, gy, context='select_goal_point_fallback'
            )
        return gx, gy

    def _select_goal_point(self, rs: RobotState, frontier: Frontier) -> Tuple[float, float]:
        """Choose a goal point within a frontier region according to the
        configured strategy. Falls back to the centroid when needed.
        """
        gx, gy = frontier.centroid_world

        if self.goal_point_strategy == 'centroid':
            if (
                rs.position is not None
                and frontier.indices is not None
                and _dist(rs.position, (gx, gy))
                < self.centroid_fallback_if_robot_closer_than_m
            ):
                picked = self._pick_frontier_cell_farthest_from_robot(
                    rs,
                    frontier,
                    (gx, gy),
                    self.centroid_fallback_max_offset_from_centroid_m,
                )
                if picked is not None:
                    gx, gy = picked

        elif self.goal_point_strategy == 'farthest_cell':
            if frontier.indices is not None and rs.position is not None:
                picked = self._pick_frontier_cell_farthest_from_robot(
                    rs, frontier, (gx, gy), None
                )
                if picked is not None:
                    gx, gy = picked

        return gx, gy

    def _goal_dist_for_assignment(self, rname: str, fr: Frontier) -> float:
        """Straight-line distance from robot to the goal pose for this frontier."""
        rs = self.robots[rname]
        if rs.position is None:
            return float('inf')
        gx, gy = self._select_goal_point(rs, fr)
        return _dist(rs.position, (gx, gy))

    def _assignment_robot_order(self, idle_names: List[str]) -> List[str]:
        """Idle robots in ``robot_names`` order, rotated so the last robot
        that completed a goal is first — avoids greedy starvation of the
        active robot when other names are idle without Nav2.
        """
        ordered = [n for n in self.robot_names if n in idle_names]
        anchor = self._last_goal_finished_robot
        if anchor and anchor in ordered:
            i = ordered.index(anchor)
            return ordered[i:] + ordered[:i]
        return ordered

    def _at_nav_goal_for_tacit_hold(self, rs: RobotState, d: float) -> bool:
        """True if map-frame distance or Nav2 feedback says we're at the goal."""
        if d <= self.tacit_goal_success_radius_m:
            return True
        dr = rs.last_distance_remaining
        # Nav2 can report distance_remaining≈0 while map-frame pose is still far
        # (stale/initial feedback). Only use the remaining-distance hint when pose
        # is within a small combined slack of the goal.
        max_d_when_using_remaining = (
            self.tacit_goal_success_radius_m
            + self.tacit_abort_max_distance_remaining_m
        )
        if (
            self.tacit_abort_max_distance_remaining_m > 0.0
            and dr is not None
            and math.isfinite(dr)
            and dr >= 0.0
            and dr <= self.tacit_abort_max_distance_remaining_m
            and d <= max_d_when_using_remaining
        ):
            return True
        return False

    def _nav2_abort_tacit_success(self, rs: RobotState) -> Tuple[bool, str]:
        """True if Nav2 ABORTED (e.g. controller gave up) but we count the leg
        as reached: same criteria as plan-loop tacit hold.
        """
        if rs.goal_position is None:
            return False, ''
        d = (
            _dist(rs.position, rs.goal_position)
            if rs.position is not None
            else float('inf')
        )
        if not self._at_nav_goal_for_tacit_hold(rs, d):
            return False, ''
        if rs.position is not None and d <= self.tacit_goal_success_radius_m:
            return True, f'pose_dist={d:.2f}m'
        dr = rs.last_distance_remaining
        return True, f'distance_remaining={dr:.2f}m'

    def _goal_leg_progress_m(self, rs: RobotState) -> float:
        """Estimated geometric progress made during the current goal leg."""
        if (
            rs.goal_position is None
            or rs.last_pose_when_goal_sent is None
            or rs.position is None
        ):
            return 0.0
        start_dist = _dist(rs.last_pose_when_goal_sent, rs.goal_position)
        now_dist = _dist(rs.position, rs.goal_position)
        return max(0.0, start_dist - now_dist)

    def _has_goal_leg_progress(self, rs: RobotState) -> bool:
        """True when the robot moved enough to count this leg as meaningful."""
        min_progress = max(self.motion_progress_min_delta_m, 0.05)
        return self._goal_leg_progress_m(rs) >= min_progress

    def _log_why_no_assignment(
        self,
        frontiers: List[Frontier],
        n_idle: int,
        now: float,
    ) -> None:
        """Throttled diagnostic when frontiers exist but assign_frontiers is empty."""
        idle_names = [
            n for n in self.robot_names
            if n in self.robots
            and not self.robots[n].goal_active
            and not self.robots[n].goal_pending
            and self.robots[n].position is not None
            and self._robot_may_assign_now(self.robots[n], now)
        ]
        if not idle_names:
            self.get_logger().warning(
                'No assignment: idle count mismatch (robots in cooldown or '
                'missing TF pose?)'
            )
            return
        order = self._assignment_robot_order(idle_names)
        rs = self.robots[order[0]]
        fr = frontiers[0]
        gx, gy = self._select_goal_point(rs, fr)
        parts: List[str] = []
        if self._frontier_too_small_for_nav_goal(fr):
            parts.append(
                f'cells<{self.min_frontier_cells_for_goal}'
            )
        if any(
            _dist(fr.centroid_world, bl) < self.blacklist_radius
            for bl in rs.blacklist
        ):
            parts.append('centroid_near_blacklist')
        if self._goal_point_blacklisted(rs, gx, gy):
            parts.append('goal_blacklisted')
        if not self._goal_passes_safety_gate(self._current_goal_map, gx, gy):
            c = self._goal_clearance_m(self._current_goal_map, gx, gy)
            cc = self._goal_cell_cost(self._current_goal_map, gx, gy)
            parts.append(f'safety(clear={c:.2f}m cost={cc})')
        d = _dist(rs.position, (gx, gy)) if rs.position else None
        self.get_logger().warning(
            f'No frontier assigned: {n_idle} idle, {len(frontiers)} frontier(s). '
            f'Sample robot={rs.name} vs frontier[0]: goal=({gx:.2f},{gy:.2f}) '
            f'dist_to_goal={d:.3f}m min_sep={self.min_goal_separation} '
            f'issues=[{",".join(parts) if parts else "none — check utility matrix"}]'
        )

    def _repeat_goal_blocks_send(
        self,
        rs: RobotState,
        goal_x: float,
        goal_y: float,
        frontier: Frontier,
    ) -> bool:
        """If True, skip sending: blacklist exhausted goal and try elsewhere."""
        if self.repeat_goal_count_limit <= 0:
            return False
        dthresh = self.repeat_goal_dist_threshold
        if dthresh <= 0.0:
            return False
        if rs.last_goal_world is not None:
            lx, ly = rs.last_goal_world
            if math.hypot(goal_x - lx, goal_y - ly) < dthresh:
                rs.repeat_goal_count += 1
            else:
                rs.repeat_goal_count = 0
        else:
            rs.repeat_goal_count = 0
        # Prevent A<->B ping-pong by checking against a short recent-goal window,
        # not just the immediately previous goal.
        recent_goal_pairs = [(gx, gy) for gx, gy, _ in list(rs.recent_goals)[-6:]]
        near_recent_hits = sum(
            1 for gx, gy in recent_goal_pairs
            if math.hypot(goal_x - gx, goal_y - gy) < dthresh
        )
        if near_recent_hits > 0:
            rs.repeat_goal_count = max(rs.repeat_goal_count, near_recent_hits)

        if rs.repeat_goal_count < self.repeat_goal_count_limit:
            return False

        if rs.homing_active:
            self.get_logger().warn(
                f'[{rs.name}] Return-home goal repeated too often; counting failure'
            )
            self._return_home_note_failure(rs)
            rs.repeat_goal_count = 0
            return True

        self.get_logger().warn(
            f'[{rs.name}] Repeatedly selecting nearly-identical goal '
            f'({goal_x:.2f}, {goal_y:.2f}); blacklisting and searching elsewhere'
        )
        rs.blacklist.append((goal_x, goal_y))
        rs.blacklist.append(frontier.centroid_world)
        rs.repeat_goal_count = 0
        return True

    def _abort_path_precheck(
        self,
        rs: RobotState,
        *,
        cancel_action: bool,
        blacklist_xy: Optional[Tuple[float, float]],
        cooldown: bool,
    ) -> None:
        if cancel_action and rs.path_precheck_goal_handle is not None:
            try:
                rs.path_precheck_goal_handle.cancel_goal_async()
            except Exception:
                pass
        # Invalidate callbacks for this attempt before clearing its fields.
        rs.path_precheck_epoch += 1
        rs.path_precheck_goal_handle = None
        rs.path_precheck_in_progress = False
        rs.path_precheck_frontier = None
        rs.path_precheck_from_retarget = False
        rs.path_precheck_goal_xy = None
        rs.path_precheck_world_goal_xy = None
        rs.path_precheck_started_time = 0.0
        rs.goal_pending = False
        if blacklist_xy is not None:
            self._blacklist_or_home_fail(rs, blacklist_xy)
        if cooldown:
            self._arm_post_failure_cooldown(rs)

    def _path_precheck_transient_fail(
        self,
        rs: RobotState,
        detail: str,
        *,
        count_for_degraded: bool = True,
    ) -> None:
        """Planner compute_path_to_pose rejected the goal or send failed.

        Nav2's planner server rejects goals while server_active_ is false
        (lifecycle still activating), even if wait_for_server() is true.
        Do not blacklist — retry shortly like NavigateToPose warm-up.
        Cancel any in-flight ComputePathToPose so a hung get_result cannot
        wedge the action relay.
        """
        self._abort_path_precheck(
            rs,
            cancel_action=True,
            blacklist_xy=None,
            cooldown=False,
        )
        now = self.get_clock().now().nanoseconds / 1e9
        retry_s = 1.0
        if (
            rs.path_precheck_server_seen_time > 0.0
            and (now - rs.path_precheck_server_seen_time)
            < self.goal_reject_warmup_sec
        ):
            retry_s = 0.5
        rs.next_assign_allowed_time = max(
            rs.next_assign_allowed_time, now + retry_s)
        if count_for_degraded:
            rs.precheck_fail_events += 1
        self._event_info(
            f'[{rs.name}] Path precheck {detail}; '
            f'retry in {retry_s:.1f}s (planner may still be activating)'
        )
        if (
            count_for_degraded
            and self.degraded_hold_enabled
            and not rs.degraded_active
            and rs.precheck_fail_events >= self.degraded_precheck_fail_threshold
        ):
            rs.degraded_active = True
            rs.degraded_since = now
            rs.degraded_reason = 'precheck_failures'
            self.get_logger().warn(
                f'[{rs.name}] Entering degraded hold due to repeated path precheck '
                f'failures (count={rs.precheck_fail_events})'
            )

    def _compute_path_error_name(self, error_code: Optional[int]) -> str:
        """Return symbolic ComputePathToPose result error name if available."""
        if error_code is None:
            return 'NONE'
        try:
            code = int(error_code)
        except Exception:
            return f'code_{error_code}'
        for attr in dir(ComputePathToPose.Result):
            if not attr.isupper():
                continue
            try:
                val = getattr(ComputePathToPose.Result, attr)
            except Exception:
                continue
            if isinstance(val, int) and int(val) == code:
                return attr
        return f'code_{code}'

    def _classify_precheck_failure(
        self, status: int, result_obj: Optional[object]
    ) -> Tuple[bool, str]:
        """Return (hard_unreachable, reason)."""
        status_name = {
            GoalStatus.STATUS_ABORTED: 'ABORTED',
            GoalStatus.STATUS_CANCELED: 'CANCELED',
            GoalStatus.STATUS_UNKNOWN: 'UNKNOWN',
        }.get(int(status), f'status_{int(status)}')
        error_code = getattr(result_obj, 'error_code', None) if result_obj is not None else None
        error_msg = str(getattr(result_obj, 'error_msg', '') or '').lower()
        error_name = self._compute_path_error_name(error_code)
        reason_base = f'status={status_name}, error={error_name}'

        # Only blacklist on explicit no-valid-path signals.
        if error_name == 'NO_VALID_PATH':
            return True, f'precheck_hard_no_valid_path ({reason_base})'
        if 'no valid path' in error_msg or 'failed to generate a valid path' in error_msg:
            return True, f'precheck_hard_no_valid_path_msg ({reason_base}, msg={error_msg})'

        # Explicit lethal/start occupied is treated as transient per policy.
        if (
            'starting point in lethal space' in error_msg
            or 'start occupied' in error_msg
            or error_name in ('START_OCCUPIED', 'START_IN_COLLISION')
        ):
            return False, f'precheck_transient_lethal_start ({reason_base})'
        return False, f'precheck_transient_non_explicit ({reason_base})'

    def _note_transient_precheck_abort_and_maybe_escalate(
        self, rs: RobotState, goal_xy: Optional[Tuple[float, float]], reason: str
    ) -> Tuple[bool, str]:
        """Escalate repeated same-goal transient ABORTED prechecks to hard fail."""
        if goal_xy is None:
            return False, reason
        if (
            rs.precheck_transient_goal_xy is None
            or _dist(rs.precheck_transient_goal_xy, goal_xy) > 0.05
        ):
            rs.precheck_transient_goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
            rs.precheck_transient_repeat_count = 1
            return False, reason
        rs.precheck_transient_repeat_count += 1
        if (
            rs.precheck_transient_repeat_count
            >= self.nav2_path_precheck_transient_abort_repeat_limit
        ):
            return True, (
                'precheck_hard_repeated_transient_abort '
                f'(count={rs.precheck_transient_repeat_count}, base={reason})'
            )
        return False, reason

    def _begin_nav2_path_precheck(
        self,
        rs: RobotState,
        frontier: Frontier,
        goal_x: float,
        goal_y: float,
        from_retarget: bool,
    ) -> None:
        """goal_x, goal_y are in ``world_frame`` (merged map coordinates)."""
        nav_goal = self._pose_stamped_for_nav_dispatch(
            rs, str(self.world_frame), goal_x, goal_y, 0.0)
        if nav_goal is None:
            self.get_logger().warn(
                f'[{rs.name}] Path precheck skipped: TF to Nav2 dispatch frame failed'
            )
            self._path_precheck_transient_fail(
                rs, 'tf_nav_dispatch_transform_failed', count_for_degraded=False)
            return

        rs.path_precheck_epoch += 1
        precheck_epoch = rs.path_precheck_epoch
        rs.path_precheck_world_goal_xy = (float(goal_x), float(goal_y))
        rs.goal_pending = True
        rs.goal_active = False
        rs.path_precheck_in_progress = True
        rs.path_precheck_frontier = frontier
        rs.path_precheck_from_retarget = from_retarget
        rs.path_precheck_goal_xy = (
            float(nav_goal.pose.position.x),
            float(nav_goal.pose.position.y),
        )
        rs.path_precheck_started_time = (
            self.get_clock().now().nanoseconds / 1e9
        )

        cp_goal = ComputePathToPose.Goal()
        cp_goal.use_start = False
        cp_goal.planner_id = self.nav2_path_precheck_planner_id
        cp_goal.goal = nav_goal

        send_future = rs.path_precheck_client.send_goal_async(cp_goal)
        send_future.add_done_callback(
            lambda f, r=rs, e=precheck_epoch: (
                self._path_precheck_goal_response_callback(f, r, e)))
        self._event_info(
            f'[{rs.name}] Nav2 path precheck '
            f'{nav_goal.header.frame_id} '
            f'({nav_goal.pose.position.x:.2f}, {nav_goal.pose.position.y:.2f}) '
            f'[world {goal_x:.2f}, {goal_y:.2f}]'
        )

    def _path_precheck_goal_response_callback(self, future, rs: RobotState, epoch: int):
        if epoch != rs.path_precheck_epoch:
            try:
                goal_handle = future.result()
            except Exception:
                return
            if goal_handle is not None and goal_handle.accepted:
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass
            return
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().warn(
                f'[{rs.name}] Path precheck send failed: {exc}')
            self._path_precheck_transient_fail(rs, f'send failed ({exc})')
            return

        if not goal_handle.accepted:
            # Warmup rejects are expected; do not push toward degraded hold.
            self._path_precheck_transient_fail(
                rs,
                'goal rejected (planner server inactive during lifecycle)',
                count_for_degraded=False,
            )
            return

        rs.path_precheck_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f, r=rs, e=epoch: self._path_precheck_result_callback(f, r, e))

    def _path_precheck_result_callback(self, future, rs: RobotState, epoch: int):
        if epoch != rs.path_precheck_epoch:
            return
        rs.path_precheck_goal_handle = None
        rs.path_precheck_in_progress = False
        goal_xy = rs.path_precheck_goal_xy
        world_goal_xy = rs.path_precheck_world_goal_xy
        frontier = rs.path_precheck_frontier
        from_retarget = rs.path_precheck_from_retarget
        rs.path_precheck_frontier = None
        rs.path_precheck_goal_xy = None
        rs.path_precheck_world_goal_xy = None
        rs.path_precheck_from_retarget = False
        rs.path_precheck_started_time = 0.0
        rs.goal_pending = False
        bl_xy = world_goal_xy if world_goal_xy is not None else goal_xy

        try:
            wrapped = future.result()
        except Exception as exc:
            self.get_logger().warn(
                f'[{rs.name}] Path precheck result error: {exc}')
            self._blacklist_or_home_fail(rs, bl_xy)
            self._arm_post_failure_cooldown(rs)
            return

        status = wrapped.status
        if status != GoalStatus.STATUS_SUCCEEDED:
            labels = {
                GoalStatus.STATUS_ABORTED: 'ABORTED',
                GoalStatus.STATUS_CANCELED: 'CANCELED',
                GoalStatus.STATUS_UNKNOWN: 'UNKNOWN',
            }
            st = labels.get(int(status), f'status_{int(status)}')
            self._event_info(
                f'[{rs.name}] Path precheck failed ({st}); skipping NavigateToPose'
            )
            now_ts = self.get_clock().now().nanoseconds / 1e9
            in_warmup = (
                rs.path_precheck_server_seen_time > 0.0
                and (now_ts - rs.path_precheck_server_seen_time)
                < self.goal_reject_warmup_sec
            )
            # ABORTED often means planner/costmap still settling; treat like reject.
            if (
                in_warmup
                and int(status) == GoalStatus.STATUS_ABORTED
                and goal_xy is not None
            ):
                self._path_precheck_transient_fail(
                    rs,
                    f'failed ({st}) during planner warmup',
                    count_for_degraded=False,
                )
                return
            hard_unreachable, reason = self._classify_precheck_failure(
                int(status), wrapped.result
            )
            if (
                not hard_unreachable
                and self.nav2_path_precheck_use_rosout_no_path_hint
                and self._matches_recent_planner_no_path_hint(rs, goal_xy)
            ):
                hard_unreachable = True
                reason = (
                    'precheck_hard_no_valid_path_rosout_hint '
                    f'(goal={goal_xy})'
                )
            if (
                not hard_unreachable
                and int(status) == GoalStatus.STATUS_ABORTED
                and 'precheck_transient_non_explicit' in reason
            ):
                tf_fresh = (
                    rs.last_tf_update_time <= 0.0
                    or (now_ts - rs.last_tf_update_time) <= self.tf_stale_timeout_sec
                )
                planner_active_hint = self._planner_state_recently_active(rs)
                # transition_event hints are sometimes absent even when the planner
                # is healthy; once compute_path_to_pose server has been observed
                # outside warmup, treat that as an "active enough" signal.
                planner_server_seen = rs.path_precheck_server_seen_time > 0.0
                if (
                    self.nav2_path_precheck_abort_active_planner_is_hard_fail
                    and tf_fresh
                    and (planner_active_hint or planner_server_seen)
                ):
                    hard_unreachable = True
                    planner_hint = (
                        'lifecycle_active'
                        if planner_active_hint else
                        'server_seen'
                    )
                    reason = (
                        'precheck_hard_aborted_with_active_planner '
                        f'(planner_hint={planner_hint}, '
                        f'planner_state={rs.planner_state_label}, '
                        f'planner_state_age={now_ts - rs.planner_state_update_time:.2f}s, '
                        f'tf_age={now_ts - rs.last_tf_update_time:.2f}s)'
                    )
                else:
                    reason = self._enrich_transient_precheck_reason(rs, goal_xy, reason)
            if (
                not hard_unreachable
                and int(status) == GoalStatus.STATUS_ABORTED
                and 'precheck_transient_non_explicit' in reason
            ):
                hard_unreachable, reason = (
                    self._note_transient_precheck_abort_and_maybe_escalate(
                        rs, goal_xy, reason
                    )
                )
            if hard_unreachable:
                self.get_logger().warn(
                    f'[{rs.name}] Path precheck hard failure -> blacklist ({reason})'
                )
                if not rs.degraded_active:
                    self._blacklist_or_home_fail(rs, bl_xy)
                    self._arm_post_failure_cooldown(rs)
                    rs.precheck_transient_goal_xy = None
                    rs.precheck_transient_repeat_count = 0
                else:
                    self.get_logger().info(
                        f'[{rs.name}] Degraded hold active; suppressing precheck blacklist ({reason})'
                    )
                return
            reason = self._enrich_transient_precheck_reason(rs, goal_xy, reason)
            self._path_precheck_transient_fail(rs, reason)
            return

        result = wrapped.result
        path = result.path if result is not None else None
        nposes = len(path.poses) if path is not None else 0
        if nposes < self.nav2_path_min_poses:
            self.get_logger().info(
                f'[{rs.name}] Path precheck: plan too short '
                f'({nposes} < {self.nav2_path_min_poses}); skipping goal'
            )
            if not rs.degraded_active:
                self._blacklist_or_home_fail(rs, bl_xy)
                self._arm_post_failure_cooldown(rs)
                rs.precheck_transient_goal_xy = None
                rs.precheck_transient_repeat_count = 0
            return

        if frontier is None or world_goal_xy is None:
            return
        gwx, gwy = world_goal_xy
        rs.precheck_transient_goal_xy = None
        rs.precheck_transient_repeat_count = 0
        self._dispatch_navigate_to_pose(
            rs, frontier, gwx, gwy, from_retarget)

    def _abort_arrival_probe(
        self,
        rs: RobotState,
        *,
        cancel_action: bool,
    ) -> None:
        if cancel_action and rs.arrival_probe_goal_handle is not None:
            try:
                rs.arrival_probe_goal_handle.cancel_goal_async()
            except Exception:
                pass
        rs.arrival_probe_goal_handle = None
        rs.arrival_probe_in_progress = False
        rs.arrival_probe_started_time = 0.0
        rs.arrival_probe_tacit_followup = False
        rs.arrival_probe_for_goal_seq = 0
        rs.arrival_probe_map_ref = None

    def _sample_nav_path_points(self, path: Path, step_m: float) -> List[Tuple[float, float]]:
        """Sample (x, y) along path poses at approximately ``step_m`` spacing."""
        poses = path.poses
        if not poses:
            return []
        pts: List[Tuple[float, float]] = [
            (float(p.pose.position.x), float(p.pose.position.y))
            for p in poses
        ]
        if len(pts) == 1:
            return pts
        out: List[Tuple[float, float]] = [pts[0]]
        cum = 0.0
        target = step_m
        for i in range(1, len(pts)):
            ax, ay = pts[i - 1]
            bx, by = pts[i]
            dx, dy = bx - ax, by - ay
            seg = math.hypot(dx, dy)
            if seg < 1e-9:
                continue
            ux, uy = dx / seg, dy / seg
            seg_start = cum
            cum += seg
            while target <= cum + 1e-9:
                back = target - seg_start
                out.append((ax + ux * back, ay + uy * back))
                target += step_m
        lx, ly = pts[-1]
        if math.hypot(out[-1][0] - lx, out[-1][1] - ly) > step_m * 0.25:
            out.append((lx, ly))
        return out

    def _score_arrival_path_on_map(
        self,
        m: Optional[OccupancyGrid],
        path: Optional[Path],
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        rs: Optional[RobotState] = None,
    ) -> Tuple[bool, str]:
        """Return (ok, reason) for merged-map feasibility of a Nav2 global path.

        ``start_xy`` / ``goal_xy`` are in ``world_frame``. Path poses are
        typically in Nav2's global frame (e.g. ``<robot>/map``); when that
        differs from the OccupancyGrid frame, sample points are transformed
        into the grid frame before cell lookup.
        """
        if m is None:
            return True, 'no_map_skip'
        if path is None or not path.poses:
            return False, 'empty_path'
        nposes = len(path.poses)
        if nposes < self.nav2_path_min_poses:
            return False, f'too_few_poses({nposes})'
        plen = 0.0
        for i in range(1, nposes):
            ax = path.poses[i - 1].pose.position.x
            ay = path.poses[i - 1].pose.position.y
            bx = path.poses[i].pose.position.x
            by = path.poses[i].pose.position.y
            plen += math.hypot(bx - ax, by - ay)
        eu = math.hypot(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1])
        if eu > 0.05 and plen / eu > self.goal_arrival_path_max_length_ratio:
            return False, (
                f'length_ratio={plen / eu:.2f}>'
                f'{self.goal_arrival_path_max_length_ratio:.2f}'
            )
        res = m.info.resolution
        step = max(res * self.goal_arrival_sample_step_scale, 0.05)
        samples = self._sample_nav_path_points(path, step)
        if not samples:
            samples = [start_xy, goal_xy]
        path_frame = (path.header.frame_id or '').strip()
        map_frame = (m.header.frame_id or '').strip() or str(self.world_frame)
        if (
            path_frame
            and map_frame
            and path_frame != map_frame
            and rs is not None
        ):
            mapped: List[Tuple[float, float]] = []
            for px, py in samples:
                conv = self._transform_plan_xy(rs, path_frame, px, py, map_frame)
                if conv is None:
                    return True, 'path_score_tf_skip'
                mapped.append(conv)
            samples = mapped
        bad = 0
        max_c = 0
        for x, y in samples:
            c = self._goal_cell_cost(m, x, y)
            max_c = max(max_c, c)
            if c >= self.goal_arrival_max_cell_cost:
                bad += 1
        frac = bad / max(len(samples), 1)
        if frac > self.goal_arrival_max_bad_fraction:
            return False, (
                f'bad_frac={frac:.2f}>{self.goal_arrival_max_bad_fraction:.2f} '
                f'max_cell={max_c}'
            )
        return (
            True,
            f'ok len_ratio={plen / eu if eu > 1e-6 else 0.0:.2f} '
            f'samples={len(samples)} max_cell={max_c}',
        )

    def _begin_arrival_probe(
        self,
        rs: RobotState,
        *,
        tacit_followup: bool,
    ) -> bool:
        """Start ComputePathToPose(use_start=True) for near-goal feasibility.

        Returns True if an async probe was started.
        """
        if not self.goal_arrival_probe_enable:
            return False
        if rs.homing_active:
            return False
        if rs.path_precheck_in_progress or rs.arrival_probe_in_progress:
            return False
        if rs.goal_position is None or rs.position is None:
            return False
        if rs.yaw is None:
            return False
        if rs.path_precheck_client is None:
            return False
        if not rs.path_precheck_client.wait_for_server(
            timeout_sec=self.nav2_path_precheck_server_wait_sec
        ):
            return False

        gx, gy = rs.goal_position
        rs.arrival_probe_in_progress = True
        rs.arrival_probe_tacit_followup = tacit_followup
        rs.arrival_probe_started_time = (
            self.get_clock().now().nanoseconds / 1e9
        )
        rs.arrival_probe_for_goal_seq = rs.active_goal_seq
        rs.arrival_probe_map_ref = self._current_goal_map

        start_pose = self._pose_stamped_for_nav_dispatch(
            rs,
            str(self.world_frame),
            float(rs.position[0]),
            float(rs.position[1]),
            float(rs.yaw),
        )
        goal_pose = self._pose_stamped_for_nav_dispatch(
            rs, str(self.world_frame), float(gx), float(gy), 0.0)
        if start_pose is None or goal_pose is None:
            self.get_logger().warn(
                f'[{rs.name}] Arrival probe skipped: TF to Nav2 dispatch frame failed'
            )
            self._abort_arrival_probe(rs, cancel_action=False)
            return

        cp_goal = ComputePathToPose.Goal()
        cp_goal.use_start = True
        cp_goal.planner_id = self.nav2_path_precheck_planner_id
        cp_goal.start = start_pose
        cp_goal.goal = goal_pose

        send_future = rs.path_precheck_client.send_goal_async(cp_goal)
        send_future.add_done_callback(
            lambda f, r=rs: self._arrival_probe_goal_response_callback(f, r))
        self._event_info(
            f'[{rs.name}] Arrival path probe (tacit_followup={tacit_followup}) '
            f'start={start_pose.header.frame_id} '
            f'({start_pose.pose.position.x:.2f},{start_pose.pose.position.y:.2f}) '
            f'goal={goal_pose.header.frame_id} '
            f'({goal_pose.pose.position.x:.2f},{goal_pose.pose.position.y:.2f}) '
            f'[world goal {gx:.2f},{gy:.2f}]'
        )
        return True

    def _arrival_probe_goal_response_callback(self, future, rs: RobotState):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().warn(
                f'[{rs.name}] Arrival probe send failed: {exc}')
            self._abort_arrival_probe(rs, cancel_action=False)
            return

        if not goal_handle.accepted:
            self.get_logger().info(
                f'[{rs.name}] Arrival probe goal rejected; treating as inconclusive'
            )
            self._abort_arrival_probe(rs, cancel_action=False)
            return

        rs.arrival_probe_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f, r=rs: self._arrival_probe_result_callback(f, r))

    def _arrival_probe_result_callback(self, future, rs: RobotState):
        saved_seq = rs.arrival_probe_for_goal_seq
        tacit_follow = rs.arrival_probe_tacit_followup
        map_snapshot = rs.arrival_probe_map_ref
        rs.arrival_probe_goal_handle = None
        rs.arrival_probe_in_progress = False
        rs.arrival_probe_tacit_followup = False
        rs.arrival_probe_started_time = 0.0
        rs.arrival_probe_for_goal_seq = 0
        rs.arrival_probe_map_ref = None

        if saved_seq != rs.active_goal_seq:
            self.get_logger().debug(
                f'[{rs.name}] Ignoring stale arrival probe result '
                f'(seq={saved_seq} active={rs.active_goal_seq})'
            )
            return

        now = self.get_clock().now().nanoseconds / 1e9
        try:
            wrapped = future.result()
        except Exception as exc:
            self.get_logger().warn(
                f'[{rs.name}] Arrival probe result error: {exc}')
            self._arrival_probe_fail(rs, tacit_follow)
            return

        status = wrapped.status
        if status != GoalStatus.STATUS_SUCCEEDED:
            labels = {
                GoalStatus.STATUS_ABORTED: 'ABORTED',
                GoalStatus.STATUS_CANCELED: 'CANCELED',
                GoalStatus.STATUS_UNKNOWN: 'UNKNOWN',
            }
            st = labels.get(int(status), f'status_{int(status)}')
            self.get_logger().info(
                f'[{rs.name}] Arrival probe planner failed ({st})'
            )
            self._arrival_probe_fail(rs, tacit_follow)
            return

        result = wrapped.result
        nav_path = result.path if result is not None else None
        gp = rs.goal_position
        sp = rs.position
        if gp is None or sp is None:
            return
        ok, why = self._score_arrival_path_on_map(
            map_snapshot, nav_path, sp, gp, rs)
        if ok:
            self.get_logger().info(
                f'[{rs.name}] Arrival probe passed ({why})'
            )
            if tacit_follow and rs.goal_handle is not None:
                rs.tacit_success_pending_cancel = True
                rs.last_cancel_reason = 'tacit_success'
                rs.ignore_cooldown_on_next_cancel = True
                try:
                    rs.goal_handle.cancel_goal_async()
                except Exception:
                    pass
                rs.near_goal_since = None
            else:
                rs.last_progress_time = now
                rs.last_goal_time = now
        else:
            self.get_logger().warn(
                f'[{rs.name}] Arrival probe failed map check: {why}'
            )
            self._arrival_probe_fail(rs, tacit_follow)

    def _arrival_probe_fail(
        self,
        rs: RobotState,
        tacit_follow: bool,
    ) -> None:
        """Blacklist and cancel after a failed arrival probe (exploration only)."""
        _ = tacit_follow
        gp = rs.goal_position
        if rs.homing_active:
            return
        self.get_logger().warn(
            f'[{rs.name}] Arrival probe failed — blacklisting goal and cancelling'
        )
        self._cancel_goal(rs, reason='arrival_probe_fail')
        if gp is not None and not (
            self.degraded_hold_enabled and rs.degraded_active
        ):
            self._blacklist_or_home_fail(rs, gp)
        self._arm_post_failure_cooldown(rs)

    def _dispatch_navigate_to_pose(
        self,
        rs: RobotState,
        frontier: Frontier,
        goal_x: float,
        goal_y: float,
        from_retarget: bool,
    ) -> None:
        """goal_x, goal_y are in ``world_frame`` (merged map). Nav2 receives
        ``PoseStamped`` transformed into the robot dispatch frame when enabled.
        """
        rs.last_goal_world = (goal_x, goal_y)
        rs.recent_goals.append((goal_x, goal_y, float(frontier.size_m)))

        nav_pose = self._pose_stamped_for_nav_dispatch(
            rs, str(self.world_frame), goal_x, goal_y, 0.0)
        if nav_pose is None:
            self.get_logger().warn(
                f'[{rs.name}] NavigateToPose skipped: TF to Nav2 dispatch frame failed'
            )
            rs.goal_active = False
            rs.goal_pending = False
            rs.goal_status = 'failed'
            if rs.homing_active:
                self._return_home_note_failure(rs)
            else:
                self._blacklist_or_home_fail(rs, (goal_x, goal_y))
                self._arm_post_failure_cooldown(rs)
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = nav_pose

        rs.goal_active = True
        rs.goal_pending = True
        rs.goal_status = 'pending'
        rs.last_cancel_reason = ''
        rs.goal_seq_counter += 1
        goal_seq = rs.goal_seq_counter
        rs.active_goal_seq = goal_seq
        if from_retarget:
            rs.retarget_abort_blacklist_suppress_seq = goal_seq
        rs.goal_position = (goal_x, goal_y)
        now = self.get_clock().now().nanoseconds / 1e9
        rs.last_goal_time = now
        rs.active_goal_sent_time = now
        rs.last_pose_when_goal_sent = rs.position
        rs.last_distance_remaining = None
        rs.last_progress_distance = None
        rs.last_progress_time = now
        rs.last_motion_sample_pose = rs.position
        rs.last_motion_sample_time = now
        rs.last_motion_time = now
        if rs.position is not None:
            rs.last_dist_to_goal = _dist(rs.position, (goal_x, goal_y))
        else:
            rs.last_dist_to_goal = None
        rs.near_goal_since = None
        rs.tacit_success_pending_cancel = False
        rs.arrival_tacit_probe_attempted = False
        rs.arrival_stagnation_probe_fired_seq = -1
        if not from_retarget:
            rs.retarget_anchor = (goal_x, goal_y)

        send_future = rs.action_client.send_goal_async(
            goal_msg,
            feedback_callback=(
                lambda f, r=rs, seq=goal_seq:
                self._goal_feedback_callback(f, r, seq)
            ),
        )
        try:
            self.event_logger.log_goal_sent(
                robot=rs.name,
                goal_x=float(nav_pose.pose.position.x),
                goal_y=float(nav_pose.pose.position.y),
                frame=str(nav_pose.header.frame_id),
                action_name=f'/{rs.name}/navigate_to_pose',
                t_ros_ns=int(self.get_clock().now().nanoseconds),
            )
        except Exception:
            pass
        send_future.add_done_callback(
            lambda f, r=rs, seq=goal_seq: self._goal_response_callback(f, r, seq))
        self._event_info(
            f'[{rs.name}] Sending NavigateToPose '
            f'{nav_pose.header.frame_id} '
            f'({nav_pose.pose.position.x:.2f}, {nav_pose.pose.position.y:.2f}) '
            f'[world {goal_x:.2f}, {goal_y:.2f}] '
            f'— frontier size {frontier.size_m:.2f}m ({frontier.size} cells)')

    def _send_goal(
        self,
        rs: RobotState,
        frontier: Frontier,
        from_retarget: bool = False,
    ):
        if rs.action_client is None:
            self.get_logger().warn(f'[{rs.name}] No action client available')
            rs.goal_active = False
            if rs.homing_active:
                self._return_home_note_failure(rs)
            return

        if not rs.action_client.wait_for_server(
                timeout_sec=self.navigate_to_pose_server_wait_sec):
            if not rs.server_unavailable_logged:
                self.get_logger().warn(
                    f'[{rs.name}] NavigateToPose action server not available '
                    f'on /{rs.name}/navigate_to_pose')
                rs.server_unavailable_logged = True
            if self.use_pose_goal_fallback:
                self._publish_pose_fallback_goal(rs, frontier, from_retarget)
                return
            rs.goal_active = False
            rs.goal_pending = False
            if rs.homing_active:
                self._return_home_note_failure(rs)
            return
        if rs.server_unavailable_logged:
            self.get_logger().info(
                f'[{rs.name}] NavigateToPose action server is now available')
            rs.server_unavailable_logged = False
            rs.action_server_ready_time = (
                self.get_clock().now().nanoseconds / 1e9
            )
        elif rs.action_server_ready_time <= 0.0:
            # Server was already up when first checked.
            rs.action_server_ready_time = (
                self.get_clock().now().nanoseconds / 1e9
            )

        goal_x, goal_y = self._select_goal_point(rs, frontier)
        src_frame = self._goal_pose_source_frame(frontier)
        world_goal = self._transform_plan_xy(
            rs, src_frame, goal_x, goal_y, str(self.world_frame))
        if world_goal is None:
            self.get_logger().warn(
                f'[{rs.name}] Skipping goal: TF {src_frame!r} -> '
                f'{self.world_frame!r} failed for ({goal_x:.2f}, {goal_y:.2f})'
            )
            rs.goal_active = False
            rs.goal_pending = False
            if rs.homing_active:
                self._return_home_note_failure(rs)
            else:
                self._arm_post_failure_cooldown(rs)
            return
        wx, wy = world_goal
        if not rs.homing_active:
            rs.last_target_frontier_centroid = frontier.centroid_world
        if self._repeat_goal_blocks_send(rs, wx, wy, frontier):
            return
        if self.goal_revalidate_before_send:
            m = self._current_goal_map
            if m is not None:
                map_frame = (m.header.frame_id or '').strip() or str(
                    self.world_frame)
                mxy = self._transform_plan_xy(
                    rs, src_frame, goal_x, goal_y, map_frame)
                if mxy is None:
                    self.get_logger().warn(
                        f'[{rs.name}] Skipping goal: TF {src_frame!r} -> '
                        f'{map_frame!r} failed for safety gate'
                    )
                    rs.goal_active = False
                    rs.goal_pending = False
                    if rs.homing_active:
                        self._return_home_note_failure(rs)
                    else:
                        self._arm_post_failure_cooldown(rs)
                    return
                mx, my = mxy
                gate_xy = (mx, my)
            else:
                gate_xy = (wx, wy)
            if not self._goal_passes_safety_gate(self._current_goal_map, gate_xy[0], gate_xy[1]):
                self._log_goal_safety_rejection(
                    rs, gate_xy[0], gate_xy[1], context='pre_send_revalidate'
                )
                self._blacklist_or_home_fail(rs, (wx, wy))
                self._arm_post_failure_cooldown(rs)
                self.get_logger().info(
                    f'[{rs.name}] Goal rejected by pre-send safety revalidation; '
                    'selecting another frontier'
                )
                return

        try:
            m = self._current_goal_map
            if m is not None:
                map_meta = {
                    'width': int(m.info.width),
                    'height': int(m.info.height),
                    'resolution': float(m.info.resolution),
                    'origin': {
                        'x': float(m.info.origin.position.x),
                        'y': float(m.info.origin.position.y),
                    },
                }
            else:
                map_meta = {}
            dist_to_goal = None
            if rs.position is not None:
                dist_to_goal = float(
                    math.hypot(wx - rs.position[0], wy - rs.position[1])
                )
            self.event_logger.log_goal_selected(
                robot=rs.name,
                goal_x=float(wx),
                goal_y=float(wy),
                frame=str(self.world_frame),
                map_topic=self.map_topic,
                map_meta=map_meta,
                merge_state=self._merge_state_last,
                score={
                    'frontier_size_cells': int(frontier.size),
                    'frontier_size_m': float(frontier.size_m),
                    'distance_to_goal_m': dist_to_goal,
                    'utility': float(self._frontier_utility(rs, frontier)),
                },
                reason='frontier_assignment',
                robot_pose_used=(
                    {'x': float(rs.position[0]), 'y': float(rs.position[1])}
                    if rs.position is not None else None
                ),
                t_ros_ns=int(self.get_clock().now().nanoseconds),
            )
        except Exception:
            pass

        if (
            self.nav2_path_precheck_enable
            and rs.path_precheck_client is not None
        ):
            if rs.path_precheck_in_progress or rs.arrival_probe_in_progress:
                return
            if rs.path_precheck_client.wait_for_server(
                timeout_sec=self.nav2_path_precheck_server_wait_sec
            ):
                now_ts = self.get_clock().now().nanoseconds / 1e9
                if rs.path_precheck_server_seen_time <= 0.0:
                    rs.path_precheck_server_seen_time = now_ts
                self._begin_nav2_path_precheck(
                    rs, frontier, wx, wy, from_retarget)
                return
            if self.nav2_path_precheck_required:
                self.get_logger().warn(
                    f'[{rs.name}] compute_path_to_pose unavailable; '
                    'deferring NavigateToPose because precheck is required '
                    f'(server_wait_sec={self.nav2_path_precheck_server_wait_sec:.2f}, '
                    f'precheck_required={self.nav2_path_precheck_required})')
                rs.goal_active = False
                rs.goal_pending = False
                rs.precheck_server_unavailable_events += 1
                if (
                    self.degraded_hold_enabled
                    and not rs.degraded_active
                    and rs.precheck_server_unavailable_events
                    >= self.degraded_precheck_fail_threshold
                ):
                    now_ts = self.get_clock().now().nanoseconds / 1e9
                    rs.degraded_active = True
                    rs.degraded_since = now_ts
                    rs.degraded_reason = 'precheck_unavailable'
                    self.get_logger().warn(
                        f'[{rs.name}] Entering degraded hold due to compute_path_to_pose '
                        f'server unavailability (count={rs.precheck_server_unavailable_events})'
                    )
                return
            self.get_logger().debug(
                f'[{rs.name}] compute_path_to_pose unavailable; '
                'NavigateToPose without precheck')

        self._dispatch_navigate_to_pose(
            rs, frontier, wx, wy, from_retarget)

    def _publish_pose_fallback_goal(
        self,
        rs: RobotState,
        frontier: Frontier,
        from_retarget: bool = False,
    ):
        goal_pub = self.goal_pubs.get(rs.name)
        if goal_pub is None:
            rs.goal_active = False
            rs.goal_pending = False
            return

        goal_x, goal_y = self._select_goal_point(rs, frontier)
        src_frame = self._goal_pose_source_frame(frontier)
        world_goal = self._transform_plan_xy(
            rs, src_frame, goal_x, goal_y, str(self.world_frame))
        if world_goal is None:
            self.get_logger().warn(
                f'[{rs.name}] PoseStamped fallback skipped: TF to world failed'
            )
            rs.goal_active = False
            rs.goal_pending = False
            return
        wx, wy = world_goal
        if self._repeat_goal_blocks_send(rs, wx, wy, frontier):
            rs.goal_active = False
            rs.goal_pending = False
            return

        rs.last_goal_world = (wx, wy)
        rs.recent_goals.append((wx, wy, float(frontier.size_m)))

        nav_pose = self._pose_stamped_for_nav_dispatch(
            rs, str(self.world_frame), wx, wy, 0.0)
        if nav_pose is None:
            self.get_logger().warn(
                f'[{rs.name}] PoseStamped fallback skipped: TF to Nav2 frame failed'
            )
            rs.goal_active = False
            rs.goal_pending = False
            return

        goal_msg = nav_pose
        goal_pub.publish(goal_msg)
        rs.goal_seq_counter += 1
        rs.active_goal_seq = rs.goal_seq_counter
        rs.goal_position = (wx, wy)
        now = self.get_clock().now().nanoseconds / 1e9
        rs.last_goal_time = now
        rs.active_goal_sent_time = now
        rs.last_pose_when_goal_sent = rs.position
        rs.last_progress_time = now
        rs.last_progress_distance = None
        rs.last_motion_sample_pose = rs.position
        rs.last_motion_sample_time = now
        rs.last_motion_time = now
        if rs.position is not None:
            rs.last_dist_to_goal = _dist(rs.position, (wx, wy))
        else:
            rs.last_dist_to_goal = None
        if not from_retarget:
            rs.retarget_anchor = (wx, wy)
        rs.goal_active = True
        rs.goal_pending = False
        rs.goal_status = 'executing'
        rs.goal_handle = None
        self.get_logger().warn(
            f'[{rs.name}] Falling back to PoseStamped goal_pose '
            f'{goal_msg.header.frame_id} '
            f'({goal_msg.pose.position.x:.2f}, {goal_msg.pose.position.y:.2f}) '
            f'[world {wx:.2f}, {wy:.2f}]')

    def _goal_response_callback(self, future, rs: RobotState, goal_seq: int):
        if goal_seq != rs.active_goal_seq:
            self.get_logger().debug(
                f'[{rs.name}] Ignoring stale goal response (seq={goal_seq}, '
                f'active_seq={rs.active_goal_seq})'
            )
            return
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().warn(
                f'[{rs.name}] Failed to send NavigateToPose goal: {exc}')
            rs.goal_active = False
            rs.goal_pending = False
            rs.goal_status = 'failed'
            rs.goal_handle = None
            rs.active_goal_seq = 0
            if rs.homing_active:
                self._return_home_note_failure(rs)
            elif rs.goal_position is not None and not self._should_skip_blacklist_due_to_lethal(rs):
                rs.blacklist.append(rs.goal_position)
            elif rs.goal_position is not None:
                rs.held_goal_position = rs.goal_position
            self._arm_post_failure_cooldown(rs)
            return

        rs.goal_pending = False
        if not goal_handle.accepted:
            now = self.get_clock().now().nanoseconds / 1e9
            in_warmup = (
                rs.action_server_ready_time > 0.0
                and (now - rs.action_server_ready_time) < self.goal_reject_warmup_sec
            )
            if in_warmup:
                elapsed = now - rs.action_server_ready_time
                self.get_logger().warn(
                    f'[{rs.name}] Goal rejected by Nav2 during warm-up '
                    f'({elapsed:.1f}s since action server became available); '
                    'will retry without blacklisting.'
                )
            else:
                self.get_logger().warn(f'[{rs.name}] Goal rejected by Nav2')
            rs.goal_active = False
            rs.goal_status = 'failed'
            rs.active_goal_seq = 0
            if rs.goal_position is not None and not in_warmup:
                if rs.homing_active:
                    self._return_home_note_failure(rs)
                elif not self._should_skip_blacklist_due_to_lethal(rs):
                    rs.blacklist.append(rs.goal_position)
                else:
                    rs.held_goal_position = rs.goal_position
            if in_warmup:
                # Retry quickly while Nav2 finishes lifecycle activation.
                rs.next_assign_allowed_time = now + 1.0
            else:
                self._arm_post_failure_cooldown(rs)
            return

        rs.goal_handle = goal_handle
        rs.goal_status = 'executing'
        now = self.get_clock().now().nanoseconds / 1e9
        rs.last_goal_time = now
        rs.last_progress_time = now
        self.get_logger().debug(f'[{rs.name}] Goal accepted')

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f, r=rs, seq=goal_seq: self._goal_result_callback(f, r, seq))

    def _goal_feedback_callback(self, feedback_msg, rs: RobotState, goal_seq: int):
        if goal_seq != rs.active_goal_seq:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        rs.last_goal_time = now
        feedback = feedback_msg.feedback
        if hasattr(feedback, 'distance_remaining'):
            dist = feedback.distance_remaining
            rs.last_distance_remaining = dist
            # Track last time we saw a meaningful decrease in remaining
            # distance, in metres. This lets the watchdog in _plan_tick
            # reason about "no progress" rather than just elapsed time.
            if rs.last_progress_distance is None:
                rs.last_progress_distance = float(dist)
                rs.last_progress_time = now
            elif (rs.last_progress_distance - dist) > self.progress_min_delta:
                rs.last_progress_distance = float(dist)
                rs.last_progress_time = now

    def _goal_result_callback(self, future, rs: RobotState, goal_seq: int):
        if goal_seq != rs.active_goal_seq:
            self.get_logger().debug(
                f'[{rs.name}] Ignoring stale goal result (seq={goal_seq}, '
                f'active_seq={rs.active_goal_seq})'
            )
            return
        # Fresh pose for tacit near-goal checks on ABORTED/CANCELED.
        self._update_robot_positions()
        was_homing = rs.homing_active
        now = self.get_clock().now().nanoseconds / 1e9
        status_text_map = {
            GoalStatus.STATUS_UNKNOWN: 'UNKNOWN',
            GoalStatus.STATUS_ACCEPTED: 'ACCEPTED',
            GoalStatus.STATUS_EXECUTING: 'EXECUTING',
            GoalStatus.STATUS_CANCELING: 'CANCELING',
            GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
            GoalStatus.STATUS_CANCELED: 'CANCELED',
            GoalStatus.STATUS_ABORTED: 'ABORTED',
        }
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().warn(
                f'[{rs.name}] Failed while waiting for NavigateToPose result: {exc}')
            result = None

        if result is None:
            try:
                self.event_logger.log_goal_result(
                    robot=rs.name,
                    status_code=-1,
                    status_text='RESULT_NONE',
                    result={
                        'goals_reached': int(rs.goals_reached),
                        'goals_failed': int(rs.goals_failed + 1),
                        'goal_position': (
                            {'x': float(rs.goal_position[0]), 'y': float(rs.goal_position[1])}
                            if rs.goal_position is not None else None
                        ),
                    },
                    t_ros_ns=int(self.get_clock().now().nanoseconds),
                )
            except Exception:
                pass
            rs.goals_failed += 1
            if was_homing:
                self._return_home_note_failure(rs)
            elif rs.goal_position and not self._should_skip_blacklist_due_to_lethal(rs):
                if not (self.degraded_hold_enabled and rs.degraded_active):
                    rs.blacklist.append(rs.goal_position)
            elif rs.goal_position:
                rs.held_goal_position = rs.goal_position
            rs.goal_active = False
            rs.goal_pending = False
            rs.goal_status = 'failed'
            rs.goal_handle = None
            rs.last_distance_remaining = None
            rs.last_progress_distance = None
            rs.last_progress_time = 0.0
            rs.last_dist_to_goal = None
            rs.last_motion_time = 0.0
            rs.last_motion_sample_pose = rs.position
            rs.last_motion_sample_time = now
            self._arm_post_failure_cooldown(rs)
            return

        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            if was_homing:
                rs.homing_active = False
                rs.return_home_done = True
                rs.goals_reached += 1
                rs.goal_status = 'succeeded'
                self.get_logger().info(
                    f'[{rs.name}] Return home succeeded (NavigateToPose)'
                )
                self._last_goal_finished_robot = rs.name
            else:
                # Only apply the "suspicious success" heuristic once the map has
                # grown beyond a minimum physical size. On tiny initial maps,
                # SLAM + Nav2 can legitimately succeed while still reporting a
                # non-trivial distance_remaining because the frontier centroid
                # lies just outside the currently known free space.
                small_map = (
                    self._map_size_x < self.strict_success_min_map_size
                    and self._map_size_y < self.strict_success_min_map_size
                )
                suspicious_thresh = self.suspicious_success_distance
                if self.mode == 'auto' and self._merge_state_last != 'MERGED':
                    # Be slightly looser before MERGED, but do not fully disable.
                    suspicious_thresh = max(
                        suspicious_thresh, self.suspicious_success_distance * 2.0
                    )
                suspicious = (
                    rs.last_distance_remaining is not None
                    and rs.last_distance_remaining > suspicious_thresh
                )
                made_progress = self._has_goal_leg_progress(rs)
                # Nav2 runs on the robot while pose comes from bridged TF/map; the
                # explorer can miss centroid motion on the central leg even when Nav2
                # feedback shows a clean terminal approach (distance_remaining).
                if (
                    not made_progress
                    and rs.last_distance_remaining is not None
                    and math.isfinite(rs.last_distance_remaining)
                    and rs.last_distance_remaining
                    <= self.trust_nav2_terminal_distance_remaining_m
                ):
                    made_progress = True
                elif (
                    self.single_robot_offloaded_nav2
                    and not made_progress
                    and rs.last_distance_remaining is not None
                    and math.isfinite(rs.last_distance_remaining)
                    and rs.last_distance_remaining
                    <= max(
                        self.tacit_abort_max_distance_remaining_m * 2.0,
                        self.suspicious_success_distance,
                    )
                ):
                    made_progress = True
                # Bridged feedback is often missing, so distance_remaining stays
                # None and the central pose can lag the robot. A multi-second
                # Nav2 success is real motion. An instant success with no
                # feedback is still rejected below.
                goal_runtime = (
                    max(0.0, now - rs.active_goal_sent_time)
                    if rs.active_goal_sent_time > 0.0 else 0.0
                )
                if (
                    not made_progress
                    and rs.last_distance_remaining is None
                    and goal_runtime >= 4.0
                ):
                    made_progress = True
                # Nav2's goal checker is 0.25 m. A success from inside that
                # radius is the robot already standing on the frontier, not a
                # failed leg. Counting it as success blacklists the frontier
                # in the map frame the assigner actually compares.
                already_there = (
                    rs.last_dist_to_goal is not None
                    and math.isfinite(rs.last_dist_to_goal)
                    and rs.last_dist_to_goal <= max(
                        0.35, self.suspicious_success_distance
                    )
                )
                if not made_progress and already_there and not suspicious:
                    made_progress = True
                if (suspicious and not small_map) or (not made_progress and not small_map):
                    rs.goals_failed += 1
                    rs.goal_status = 'failed'
                    self.get_logger().warn(
                        f'[{rs.name}] Goal reported success but failed validation '
                        f'(distance_remaining={rs.last_distance_remaining}, '
                        f'suspicious_thresh={suspicious_thresh:.2f}m, '
                        f'progress_m={self._goal_leg_progress_m(rs):.2f}); '
                        'treating as failure and blacklisting'
                    )
                    if rs.goal_position and not (
                        self.degraded_hold_enabled and rs.degraded_active
                    ):
                        self._append_blacklist_if_distant(rs, rs.goal_position)
                        # goal_position is in the world frame. Frontier
                        # assignment compares map-frame centroids, so a world
                        # point alone does not stop the same frontier.
                        if rs.last_target_frontier_centroid is not None:
                            self._append_blacklist_if_distant(
                                rs, rs.last_target_frontier_centroid
                            )
                    self._arm_post_failure_cooldown(rs)
                else:
                    rs.goals_reached += 1
                    rs.goal_status = 'succeeded'
                    if suspicious and small_map:
                        self.get_logger().info(
                            f'[{rs.name}] Goal reported success with '
                            f'distance_remaining={rs.last_distance_remaining:.2f}m '
                            f'on a small map (~{self._map_size_x:.1f}x'
                            f'{self._map_size_y:.1f}m); accepting as success to '
                            f'grow the initial map')
                    else:
                        self.get_logger().info(
                            f'[{rs.name}] Goal reached '
                            f'(total: {rs.goals_reached})')
                    self._last_goal_finished_robot = rs.name
                    self._apply_post_success_blacklist_policy(rs)
        elif status == GoalStatus.STATUS_ABORTED:
            tacit_ok, tacit_why = self._nav2_abort_tacit_success(rs)
            tacit_progress_ok = self._has_goal_leg_progress(rs)
            if tacit_ok and (was_homing or tacit_progress_ok):
                if was_homing:
                    rs.homing_active = False
                    rs.return_home_done = True
                rs.goals_reached += 1
                rs.goal_status = 'succeeded'
                if was_homing:
                    self.get_logger().info(
                        f'[{rs.name}] Return home succeeded (Nav2 tacit: {tacit_why}; '
                        f'total: {rs.goals_reached})'
                    )
                else:
                    self.get_logger().info(
                        f'[{rs.name}] Goal reached (Nav2 ABORTED; tacit success '
                        f'{tacit_why}; total: {rs.goals_reached})'
                    )
                self._last_goal_finished_robot = rs.name
                if not was_homing:
                    self._apply_post_success_blacklist_policy(rs)
            elif tacit_ok and not was_homing:
                rs.goals_failed += 1
                rs.goal_status = 'aborted'
                self.get_logger().warn(
                    f'[{rs.name}] Nav2 ABORTED met tacit criteria ({tacit_why}) but '
                    f'robot progress was too small ({self._goal_leg_progress_m(rs):.2f}m); '
                    'treating as failure'
                )
                if rs.goal_position and not (
                    self.degraded_hold_enabled and rs.degraded_active
                ):
                    self._blacklist_or_home_fail(rs, rs.goal_position)
                self._arm_post_failure_cooldown(rs)
            elif rs.tacit_success_pending_cancel:
                # Client cancel for tacit "at goal"; bt_navigator may end as ABORTED.
                rs.tacit_success_pending_cancel = False
                rs.ignore_cooldown_on_next_cancel = False
                if was_homing:
                    rs.homing_active = False
                    rs.return_home_done = True
                rs.goals_reached += 1
                rs.goal_status = 'succeeded'
                if was_homing:
                    self.get_logger().info(
                        f'[{rs.name}] Return home succeeded (tacit cancel as ABORTED; '
                        f'total: {rs.goals_reached})'
                    )
                else:
                    self.get_logger().info(
                        f'[{rs.name}] Goal reached (tacit cancel as ABORTED near pose; '
                        f'total: {rs.goals_reached})'
                    )
                self._last_goal_finished_robot = rs.name
                if not was_homing:
                    self._apply_post_success_blacklist_policy(rs)
            elif (
                not was_homing
                and rs.retarget_abort_blacklist_suppress_seq is not None
                and goal_seq == rs.retarget_abort_blacklist_suppress_seq
            ):
                # Replacement goal right after retarget: Nav2 often reports ABORTED
                # ("Aborting handle") while tearing down the cancelled goal. Scoped
                # to this goal_seq so a later real failure is not misclassified.
                rs.ignore_cooldown_on_next_cancel = False
                rs.goal_status = 'canceled'
                self.get_logger().info(
                    f'[{rs.name}] NavigateToPose ended as ABORTED after our cancel '
                    f'(retarget); not blacklisting'
                )
            else:
                rs.goals_failed += 1
                rs.goal_status = 'aborted'
                pd = (
                    _dist(rs.position, rs.goal_position)
                    if rs.position is not None and rs.goal_position is not None
                    else None
                )
                if self._should_skip_blacklist_due_to_lethal(rs):
                    self.get_logger().warn(
                        f'[{rs.name}] Goal aborted while Nav2 lethal hold active; '
                        'keeping goal and skipping blacklist'
                    )
                    if rs.goal_position:
                        rs.held_goal_position = rs.goal_position
                else:
                    self.get_logger().warn(
                        f'[{rs.name}] Goal aborted — blacklisting '
                        f'(pose_dist_m={pd}, distance_remaining={rs.last_distance_remaining}; '
                        f'tacit: pose≤{self.tacit_goal_success_radius_m:.2f}m or '
                        f'remain≤{self.tacit_abort_max_distance_remaining_m:.2f}m)'
                    )
                    if rs.goal_position and not (
                        self.degraded_hold_enabled and rs.degraded_active
                    ):
                        self._blacklist_or_home_fail(rs, rs.goal_position)
                self._arm_post_failure_cooldown(rs)
        elif status == GoalStatus.STATUS_CANCELED:
            tacit_progress_ok = self._has_goal_leg_progress(rs)
            if rs.tacit_success_pending_cancel and (was_homing or tacit_progress_ok):
                rs.tacit_success_pending_cancel = False
                rs.ignore_cooldown_on_next_cancel = False
                if was_homing:
                    rs.homing_active = False
                    rs.return_home_done = True
                rs.goals_reached += 1
                rs.goal_status = 'succeeded'
                if was_homing:
                    self.get_logger().info(
                        f'[{rs.name}] Return home succeeded (tacit cancel; '
                        f'total: {rs.goals_reached})'
                    )
                else:
                    self.get_logger().info(
                        f'[{rs.name}] Goal reached (tacit cancel near pose; '
                        f'total: {rs.goals_reached})'
                    )
                self._last_goal_finished_robot = rs.name
                if not was_homing:
                    self._apply_post_success_blacklist_policy(rs)
            elif rs.tacit_success_pending_cancel and not was_homing:
                rs.tacit_success_pending_cancel = False
                rs.ignore_cooldown_on_next_cancel = False
                rs.goals_failed += 1
                rs.goal_status = 'canceled'
                self.get_logger().warn(
                    f'[{rs.name}] Tacit cancel near-goal ignored due to low progress '
                    f'({self._goal_leg_progress_m(rs):.2f}m); treating as canceled/failure'
                )
                self._arm_post_failure_cooldown(rs)
            else:
                rs.goal_status = 'canceled'
                admin_cancel_reasons = {
                    'retarget_opportunity',
                    'retarget_stagnation',
                    'pause',
                    'shutdown',
                    'return_home_reset',
                    'tacit_success',
                    'arrival_probe_fail',
                    'stall_watchdog',
                    'retrace_escape',
                }
                cancel_reason = rs.last_cancel_reason or 'unspecified'
                pose_dist = (
                    _dist(rs.position, rs.goal_position)
                    if rs.position is not None and rs.goal_position is not None
                    else None
                )
                far_from_goal = (
                    pose_dist is None
                    or pose_dist > max(
                        self.tacit_goal_success_radius_m,
                        self.canceled_blacklist_min_goal_dist_m,
                    )
                )
                should_blacklist_cancel = (
                    self.blacklist_on_canceled_when_far
                    and not was_homing
                    and cancel_reason not in admin_cancel_reasons
                    and far_from_goal
                    and not (self.degraded_hold_enabled and rs.degraded_active)
                    and not self._should_skip_blacklist_due_to_lethal(rs)
                    and not link_retreat_blocks_assignment(
                        self.pause_on_link_retreat, rs.link_retreat_active)
                )
                if should_blacklist_cancel and rs.goal_position is not None:
                    rs.last_cancel_blacklist_time = (
                        self.get_clock().now().nanoseconds / 1e9)
                    rs.last_cancel_blacklist_point = rs.goal_position
                    self.get_logger().warn(
                        f'[{rs.name}] Goal canceled far from target; blacklisting '
                        f'(reason={cancel_reason}, pose_dist_m={pose_dist}, '
                        f'thresh_m={max(self.tacit_goal_success_radius_m, self.canceled_blacklist_min_goal_dist_m):.2f})'
                    )
                    self._blacklist_or_home_fail(rs, rs.goal_position)
                else:
                    self.get_logger().info(
                        f'[{rs.name}] Goal cancelled '
                        f'(reason={cancel_reason}, pose_dist_m={pose_dist}, '
                        f'blacklist={should_blacklist_cancel})'
                    )
                if rs.ignore_cooldown_on_next_cancel:
                    rs.ignore_cooldown_on_next_cancel = False
                elif was_homing:
                    self._return_home_note_failure(rs)
                    self._arm_post_failure_cooldown(rs)
                else:
                    self._arm_post_failure_cooldown(rs)
        else:
            rs.goal_status = 'failed'
            self.get_logger().info(
                f'[{rs.name}] Goal finished with status {status}')
            if was_homing:
                self._return_home_note_failure(rs)
            self._arm_post_failure_cooldown(rs)

        try:
            self.event_logger.log_goal_result(
                robot=rs.name,
                status_code=int(status),
                status_text=status_text_map.get(int(status), f'STATUS_{int(status)}'),
                result={
                    'goals_reached': int(rs.goals_reached),
                    'goals_failed': int(rs.goals_failed),
                    'distance_remaining': (
                        float(rs.last_distance_remaining)
                        if rs.last_distance_remaining is not None else None
                    ),
                    'goal_position': (
                        {'x': float(rs.goal_position[0]), 'y': float(rs.goal_position[1])}
                        if rs.goal_position is not None else None
                    ),
                },
                t_ros_ns=int(self.get_clock().now().nanoseconds),
            )
        except Exception:
            pass

        if rs.retarget_abort_blacklist_suppress_seq == goal_seq:
            rs.retarget_abort_blacklist_suppress_seq = None

        rs.goal_active = False
        rs.goal_pending = False
        rs.goal_handle = None
        rs.active_goal_seq = 0
        rs.last_distance_remaining = None
        rs.last_progress_distance = None
        rs.last_progress_time = 0.0
        rs.last_dist_to_goal = None
        rs.last_motion_time = 0.0
        rs.last_motion_sample_pose = rs.position
        rs.last_motion_sample_time = now
        rs.near_goal_since = None

    def _cancel_goal(self, rs: RobotState, *, reason: str = 'unspecified'):
        now = self.get_clock().now().nanoseconds / 1e9
        rs.last_cancel_reason = reason
        if rs.arrival_probe_in_progress:
            self._abort_arrival_probe(rs, cancel_action=True)
        if rs.path_precheck_in_progress:
            self._abort_path_precheck(
                rs,
                cancel_action=True,
                blacklist_xy=None,
                cooldown=False,
            )
        if rs.goal_handle is not None:
            try:
                rs.goal_handle.cancel_goal_async()
            except Exception:
                pass
        rs.goal_active = False
        rs.goal_pending = False
        rs.goal_status = 'canceled'
        rs.goal_handle = None
        rs.last_distance_remaining = None
        rs.last_progress_distance = None
        rs.last_progress_time = 0.0
        rs.last_dist_to_goal = None
        rs.last_motion_time = 0.0
        rs.last_motion_sample_pose = rs.position
        rs.last_motion_sample_time = now
        rs.near_goal_since = None

    # -----------------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------------

    def _publish_frontier_markers(self, frontiers: List[Frontier]):
        ma = MarkerArray()
        marker_frame = self.world_frame

        # delete old markers
        del_marker = Marker()
        del_marker.action = Marker.DELETEALL
        del_marker.header.frame_id = marker_frame
        del_marker.ns = 'frontiers'
        # RViz can warn/drop markers if nested texture headers keep empty frame_id.
        del_marker.texture.header.frame_id = marker_frame
        ma.markers.append(del_marker)

        for i, fr in enumerate(frontiers):
            m = Marker()
            m.header.frame_id = marker_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'frontiers'
            m.id = i + 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = fr.centroid_world[0]
            m.pose.position.y = fr.centroid_world[1]
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            scale = max(0.15, min(0.6, fr.size_m * 0.3))
            m.scale.x = scale
            m.scale.y = scale
            m.scale.z = scale * 0.5
            m.color.r = 0.2
            m.color.g = 0.8
            m.color.b = 0.2
            m.color.a = 0.8
            m.lifetime.sec = 5
            m.texture.header.frame_id = marker_frame
            ma.markers.append(m)

        # mark assigned goals in a different colour
        idx = len(frontiers) + 2
        for rs in self.robots.values():
            if rs.goal_active and rs.goal_position:
                gm = Marker()
                gm.header.frame_id = marker_frame
                gm.header.stamp = self.get_clock().now().to_msg()
                gm.ns = 'goals'
                gm.id = idx
                idx += 1
                gm.type = Marker.ARROW
                gm.action = Marker.ADD
                gm.pose.position.x = rs.goal_position[0]
                gm.pose.position.y = rs.goal_position[1]
                gm.pose.position.z = 0.3
                gm.pose.orientation.w = 0.707
                gm.pose.orientation.x = 0.0
                gm.pose.orientation.y = 0.707
                gm.pose.orientation.z = 0.0
                gm.scale.x = 0.4
                gm.scale.y = 0.12
                gm.scale.z = 0.12
                gm.color.r = 1.0
                gm.color.g = 0.3
                gm.color.b = 0.0
                gm.color.a = 1.0
                gm.lifetime.sec = 5
                gm.texture.header.frame_id = marker_frame
                ma.markers.append(gm)

        self.marker_pub.publish(ma)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = MultiRobotExplorer()
    # MultiThreadedExecutor avoids a single subscription or TF work starving
    # other map callbacks when DDS delivers concurrently with the planner timer.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        # Treat Ctrl+C and external shutdown events as normal exit paths.
        pass
    except Exception:
        # Context is already gone after Ctrl+C; that is not a crash.
        if rclpy.ok():
            traceback.print_exc()
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass
        # cancel all active goals
        for rs in node.robots.values():
            if rs.goal_active:
                node._cancel_goal(rs, reason='shutdown')
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()

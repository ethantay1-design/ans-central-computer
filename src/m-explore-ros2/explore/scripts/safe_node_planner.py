#!/usr/bin/env python3
"""
Safe-node planner for the central computer (phase 2 of the risk-map work).

For each robot, records Wi-Fi quality where the robot has been on the merged
map, builds a coarse risk grid and picks the nearest safe node. Uses the same
TF chain as multi_robot_explorer.py: map -> <robot>/map -> <robot>/odom ->
<robot>/base_footprint on global /tf.

Subscribes:
  /map                        nav_msgs/OccupancyGrid (merged map)
  /<robot>/wifi_quality       std_msgs/Float32, 0..1 (robot wifi_monitor.py)
  /tf, /tf_static
Publishes every publish_period_sec:
  /<robot>/connection_risk    nav_msgs/OccupancyGrid, frame map, 0.5 m cells,
                              0 safe .. 100 worst, -1 unknown
  /<robot>/safe_node          geometry_msgs/PoseStamped in <robot>/map
  /<robot>/connection_risk_display   only if publish_display_grid: same grid
                              scaled for RViz's costmap colours (1 blue ..
                              98 red; unknown and walls 0 = transparent)

safe_node doubles as the link heartbeat, so it is sent every period once a
safe node has been found: the newest one, or the last one if none is
reachable right now. Nothing is sent before the first one exists.

Not started by any script; run by hand:
  python3 scripts/safe_node_planner.py --ros-args -p robot_names:="['blinky']"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PoseStamped for Buffer.transform()

from safe_node_logic import (
    CoarseGrid, RiskParams, SignalMap, coarsen, risk_grid, select_safe_node,
)

MAP_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)


@dataclass
class RobotState:
    name: str
    # Wi-Fi samples in world (map) coordinates, kept so they can be
    # re-binned when the merged map grows or moves.
    samples: List[Tuple[float, float, float]] = field(default_factory=list)
    last_wifi_time: float = 0.0
    last_safe_node: Optional[Tuple[float, float]] = None
    risk_pub: object = None
    display_pub: object = None
    node_pub: object = None


class SafeNodePlanner(Node):
    def __init__(self):
        super().__init__('safe_node_planner')

        self.declare_parameter('robot_names', ['blinky'])
        self.declare_parameter('world_frame', 'map')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('robot_base_frame', 'base_footprint')
        self.declare_parameter('safe_node_frame_pattern', '{robot}/map')
        self.declare_parameter('publish_period_sec', 2.0)
        self.declare_parameter('coarse_resolution', 0.5)
        self.declare_parameter('clearance_m', 1.0)
        self.declare_parameter('safe_threshold', 30)
        self.declare_parameter('max_samples_per_robot', 20000)
        self.declare_parameter('publish_display_grid', False)

        self.world_frame = str(self.get_parameter('world_frame').value)
        self.base_frame = str(self.get_parameter('robot_base_frame').value)
        self.frame_pattern = str(self.get_parameter('safe_node_frame_pattern').value)
        self.max_samples = int(self.get_parameter('max_samples_per_robot').value)
        self.display_grid = bool(self.get_parameter('publish_display_grid').value)
        self.params = RiskParams(
            coarse_resolution=float(self.get_parameter('coarse_resolution').value),
            clearance_m=float(self.get_parameter('clearance_m').value),
            safe_threshold=int(self.get_parameter('safe_threshold').value),
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.map_msg: Optional[OccupancyGrid] = None
        self.grid: Optional[CoarseGrid] = None
        self.map_dirty = False
        self.create_subscription(
            OccupancyGrid, str(self.get_parameter('map_topic').value),
            self._on_map, MAP_QOS)

        self.robots: Dict[str, RobotState] = {}
        for name in self.get_parameter('robot_names').value:
            rs = RobotState(name=name)
            rs.risk_pub = self.create_publisher(
                OccupancyGrid, f'/{name}/connection_risk', MAP_QOS)
            rs.node_pub = self.create_publisher(PoseStamped, f'/{name}/safe_node', 10)
            if self.display_grid:
                rs.display_pub = self.create_publisher(
                    OccupancyGrid, f'/{name}/connection_risk_display', MAP_QOS)
            self.create_subscription(
                Float32, f'/{name}/wifi_quality',
                lambda msg, r=rs: self._on_wifi(r, msg), 10)
            self.robots[name] = rs

        period = float(self.get_parameter('publish_period_sec').value)
        self.create_timer(period, self._tick)
        self.get_logger().info(
            f'safe_node_planner: robots={list(self.robots)}, every {period:.1f} s, '
            f'{self.params.coarse_resolution} m cells, safe <= {self.params.safe_threshold}')

    # ------------------------------------------------------------------

    def _on_map(self, msg: OccupancyGrid):
        self.map_msg = msg
        self.map_dirty = True

    def _robot_xy(self, name: str) -> Optional[Tuple[float, float]]:
        try:
            t = self.tf_buffer.lookup_transform(
                self.world_frame, f'{name}/{self.base_frame}', Time(),
                timeout=Duration(seconds=0.1))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        return t.transform.translation.x, t.transform.translation.y

    def _on_wifi(self, rs: RobotState, msg: Float32):
        xy = self._robot_xy(rs.name)
        if xy is None:
            self.get_logger().warn(
                f'[{rs.name}] wifi_quality received but no robot pose in '
                f'{self.world_frame!r}; sample dropped', throttle_duration_sec=10.0)
            return
        rs.samples.append((xy[0], xy[1], float(msg.data)))
        if len(rs.samples) > self.max_samples:
            del rs.samples[: len(rs.samples) - self.max_samples]
        rs.last_wifi_time = self.get_clock().now().nanoseconds / 1e9

    def _rebuild_grid(self):
        m = self.map_msg
        self.grid = coarsen(
            m.data, m.info.width, m.info.height, m.info.resolution,
            m.info.origin.position.x, m.info.origin.position.y, self.params)
        self.map_dirty = False

    def _tick(self):
        if self.map_msg is None:
            self.get_logger().info('waiting for merged map', throttle_duration_sec=10.0)
            return
        if self.map_dirty:
            self._rebuild_grid()
        for rs in self.robots.values():
            self._plan(rs)

    def _plan(self, rs: RobotState):
        grid = self.grid
        signal = SignalMap(grid)
        for x, y, q in rs.samples:
            signal.add_sample(x, y, q)
        risk = risk_grid(grid, signal, self.params)
        self._publish_risk(rs, grid, risk)

        xy = self._robot_xy(rs.name)
        node = select_safe_node(grid, risk, xy, self.params) if xy else None
        if node is not None:
            rs.last_safe_node = node
        elif rs.last_safe_node is None:
            self.get_logger().info(
                f'[{rs.name}] no safe node yet ({len(rs.samples)} samples)',
                throttle_duration_sec=10.0)
            return
        self._publish_safe_node(rs, rs.last_safe_node, fresh=node is not None)

    def _publish_risk(self, rs: RobotState, grid: CoarseGrid, risk: List[int]):
        g = OccupancyGrid()
        g.header.frame_id = self.world_frame
        g.header.stamp = self.get_clock().now().to_msg()
        g.info.resolution = grid.resolution
        g.info.width, g.info.height = grid.width, grid.height
        g.info.origin.position.x = grid.origin_x
        g.info.origin.position.y = grid.origin_y
        g.info.origin.orientation.w = 1.0
        g.data = list(risk)
        rs.risk_pub.publish(g)
        if rs.display_pub is not None:
            g.data = [0 if r in (-1, 100) else max(1, min(98, r)) for r in risk]
            rs.display_pub.publish(g)

    def _publish_safe_node(self, rs: RobotState, node: Tuple[float, float], fresh: bool):
        ps = PoseStamped()
        ps.header.frame_id = self.world_frame
        # Time 0 = latest transform; the map -> <robot>/map offset changes slowly.
        ps.header.stamp = Time().to_msg()
        ps.pose.position.x, ps.pose.position.y = node
        ps.pose.orientation.w = 1.0
        dst = self.frame_pattern.format(robot=rs.name)
        try:
            out = self.tf_buffer.transform(ps, dst, timeout=Duration(seconds=0.2))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(
                f'[{rs.name}] safe node {self.world_frame!r} -> {dst!r} failed: {exc}',
                throttle_duration_sec=10.0)
            return
        out.header.stamp = self.get_clock().now().to_msg()
        rs.node_pub.publish(out)
        self.get_logger().info(
            f'[{rs.name}] safe node {"found" if fresh else "kept (none reachable now)"}: '
            f'map ({node[0]:.2f}, {node[1]:.2f}) = {dst} '
            f'({out.pose.position.x:.2f}, {out.pose.position.y:.2f})',
            throttle_duration_sec=4.0)


def main():
    rclpy.init()
    node = SafeNodePlanner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

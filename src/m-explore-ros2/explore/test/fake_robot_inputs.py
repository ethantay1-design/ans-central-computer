#!/usr/bin/env python3
"""
Fake Blinky + map_merge for testing safe_node_planner.py without a robot.

Publishes what central would receive in a real run, built from fake_inputs.py:
  /map                     fake merged map (latched)
  /tf_static               map -> blinky/map (offset, like map_merge would)
                           and blinky/map -> blinky/odom
  /tf                      blinky/odom -> blinky/base_footprint, moving
  /blinky/wifi_quality     Float32 0..1 at each step
  /blinky/driven_path      Path, for RViz only
  /blinky/access_point     Marker, for RViz only

The robot replays the fake drive in a loop. Not installed; run by hand:
  ROS_DOMAIN_ID=77 python3 test/fake_robot_inputs.py
--as-robot: publish like a real robot instead (map on /<robot>/map in frame
<robot>/map, TF on /<robot>/tf and /<robot>/tf_static, no map -> <robot>/map),
so start_central.sh's own TF relay and map relay are exercised.
--moving-obstacle: a 0.6 m box moves back and forth across room A and is
written into /map (as SLAM would mark it), republished at 1 Hz; also shown
as a Marker on /<robot>/moving_obstacle.
A second robot (no /map, its own offset, weaker Wi-Fi):
  ROS_DOMAIN_ID=77 python3 test/fake_robot_inputs.py --robot pinky \
      --offset 2.0 1.0 --no-map --quality-scale 0
"""

import argparse
import math
import os
import sys

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Float32
from tf2_msgs.msg import TFMessage
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_inputs  # noqa: E402

STEP_SEC = 0.3
HOLD_STEPS = 15
# Moving obstacle: 0.6 m box sweeping x = 1.0..5.0 m at y = 4.0 m in room A.
OBSTACLE_SIZE = 0.6
OBSTACLE_Y = 4.0
OBSTACLE_X_RANGE = (1.0, 5.0)
OBSTACLE_SPEED = 0.4  # m/s
MAP_PERIOD_SEC = 1.0


def transform(parent, child, x, y, stamp):
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id = parent, child
    t.header.stamp = stamp
    t.transform.translation.x, t.transform.translation.y = x, y
    t.transform.rotation.w = 1.0
    return t


class FakeRobotInputs(Node):
    def __init__(self, args):
        super().__init__(f'fake_robot_inputs_{args.robot}')
        global ROBOT, ROBOT_MAP_OFFSET
        ROBOT = args.robot
        # Where <robot>/map sits inside the merged map. Non-zero so the
        # planner's map -> <robot>/map conversion is actually exercised.
        ROBOT_MAP_OFFSET = tuple(args.offset)
        self.with_map = not args.no_map
        self.with_wifi = not args.no_wifi
        self.as_robot = args.as_robot
        self.moving = args.moving_obstacle
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        map_topic = f'/{ROBOT}/map' if self.as_robot else '/map'
        self.map_frame = f'{ROBOT}/map' if self.as_robot else 'map'
        self.map_pub = self.create_publisher(OccupancyGrid, map_topic, latched)
        if self.as_robot:
            self.robot_tf = self.create_publisher(TFMessage, f'/{ROBOT}/tf', 100)
            self.robot_tf_static = self.create_publisher(
                TFMessage, f'/{ROBOT}/tf_static', latched)
        self.wifi_pub = self.create_publisher(Float32, f'/{ROBOT}/wifi_quality', 10)
        self.path_pub = self.create_publisher(Path, f'/{ROBOT}/driven_path', 10)
        self.ap_pub = self.create_publisher(Marker, f'/{ROBOT}/access_point', latched)
        self.tf = TransformBroadcaster(self)
        self.tf_static = StaticTransformBroadcaster(self)

        self.data, self.w, self.h, self.samples = fake_inputs.make_samples()
        self.base_data = list(self.data)
        self.t_start = self.get_clock().now().nanoseconds / 1e9
        if self.moving:
            self.obstacle_pub = self.create_publisher(
                Marker, f'/{ROBOT}/moving_obstacle', 10)
        self.samples = [(x, y, q * args.quality_scale) for x, y, q in self.samples]
        now = self.get_clock().now().to_msg()
        if self.as_robot:
            self.robot_tf_static.publish(TFMessage(transforms=[
                transform(f'{ROBOT}/map', f'{ROBOT}/odom', 0.0, 0.0, now)]))
        else:
            self.tf_static.sendTransform([
                transform('map', f'{ROBOT}/map', *ROBOT_MAP_OFFSET, now),
                transform(f'{ROBOT}/map', f'{ROBOT}/odom', 0.0, 0.0, now),
            ])
        if self.with_map:
            self.publish_map()
            self.publish_access_point()
            # SLAM republishes its map; relays may subscribe after the first one.
            self.create_timer(MAP_PERIOD_SEC if self.moving else 2.0, self.publish_map)
        self.k = 0
        self.path = Path()
        self.create_timer(STEP_SEC, self.step)
        self.get_logger().info(
            f'Fake {ROBOT}: {len(self.samples)} samples, {ROBOT}/map at '
            f'{ROBOT_MAP_OFFSET} in map')

    def obstacle_xy(self):
        t = self.get_clock().now().nanoseconds / 1e9 - self.t_start
        lo, hi = OBSTACLE_X_RANGE
        span = hi - lo
        # Triangle wave: lo -> hi -> lo ...
        phase = (t * OBSTACLE_SPEED) % (2 * span)
        x = lo + (phase if phase <= span else 2 * span - phase)
        return x, OBSTACLE_Y

    def stamp_obstacle(self):
        self.data = list(self.base_data)
        ox, oy = self.obstacle_xy()
        res, half = fake_inputs.RESOLUTION, OBSTACLE_SIZE / 2.0
        for cy in range(int((oy - half) / res), int(math.ceil((oy + half) / res))):
            for cx in range(int((ox - half) / res), int(math.ceil((ox + half) / res))):
                if 0 <= cx < self.w and 0 <= cy < self.h:
                    self.data[cy * self.w + cx] = 100
        mk = Marker()
        mk.header.frame_id = 'map'
        mk.header.stamp = self.get_clock().now().to_msg()
        mk.ns, mk.id, mk.type, mk.action = 'moving_obstacle', 0, Marker.CUBE, Marker.ADD
        mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = ox, oy, 0.3
        mk.pose.orientation.w = 1.0
        mk.scale.x = mk.scale.y = OBSTACLE_SIZE
        mk.scale.z = 0.6
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = 0.9, 0.1, 0.1, 0.9
        self.obstacle_pub.publish(mk)

    def publish_map(self):
        if self.moving:
            self.stamp_obstacle()
        m = OccupancyGrid()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.info.resolution = fake_inputs.RESOLUTION
        m.info.width, m.info.height = self.w, self.h
        m.info.origin.position.x, m.info.origin.position.y = fake_inputs.ORIGIN
        m.info.origin.orientation.w = 1.0
        m.data = self.data
        self.map_pub.publish(m)

    def publish_access_point(self):
        mk = Marker()
        mk.header.frame_id = 'map'
        mk.ns, mk.id, mk.type, mk.action = 'access_point', 0, Marker.CYLINDER, Marker.ADD
        mk.pose.position.x, mk.pose.position.y = fake_inputs.ACCESS_POINT
        mk.pose.position.z = 0.25
        mk.pose.orientation.w = 1.0
        mk.scale.x = mk.scale.y = 0.3
        mk.scale.z = 0.5
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = 0.1, 0.8, 0.1, 1.0
        self.ap_pub.publish(mk)

    def step(self):
        if self.k >= len(self.samples) + HOLD_STEPS:
            self.k = 0
            self.path = Path()
        i = min(self.k, len(self.samples) - 1)
        x, y, q = self.samples[i]
        stamp = self.get_clock().now().to_msg()
        # Robot pose in its own odom frame = world pose minus the map offset.
        ox, oy = ROBOT_MAP_OFFSET
        t = transform(f'{ROBOT}/odom', f'{ROBOT}/base_footprint', x - ox, y - oy, stamp)
        if self.as_robot:
            self.robot_tf.publish(TFMessage(transforms=[t]))
        else:
            self.tf.sendTransform(t)
        if self.k < len(self.samples):
            if self.with_wifi:
                self.wifi_pub.publish(Float32(data=float(q)))
            p = PoseStamped()
            p.header.frame_id = 'map'
            p.pose.position.x, p.pose.position.y = x, y
            p.pose.orientation.w = 1.0
            self.path.poses.append(p)
        self.path.header.frame_id = 'map'
        self.path.header.stamp = stamp
        self.path_pub.publish(self.path)
        self.k += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--robot', default='blinky')
    ap.add_argument('--offset', type=float, nargs=2, default=[1.0, 0.5])
    ap.add_argument('--no-map', action='store_true', help='another instance publishes /map')
    ap.add_argument('--moving-obstacle', action='store_true',
                    help='a box moving across room A, written into /map')
    ap.add_argument('--as-robot', action='store_true',
                    help='publish map/TF under /<robot>/ like a real robot')
    ap.add_argument('--no-wifi', action='store_true',
                    help='wifi_quality comes from a real wifi_monitor.py')
    ap.add_argument('--quality-scale', type=float, default=1.0,
                    help='multiply Wi-Fi quality (0 = dead signal everywhere)')
    args = ap.parse_args()
    rclpy.init()
    node = FakeRobotInputs(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

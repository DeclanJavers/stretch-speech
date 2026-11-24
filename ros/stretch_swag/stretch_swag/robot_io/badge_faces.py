"""Perception cheat for the real robot: ArUco badges stand in for faces.

Visitors wear a printed ArUco badge (ids 100-109 by default, see badges/). Stretch's own
detector (stretch_core detect_aruco_markers) publishes /aruco/marker_array with each
marker's 3D position in the camera frame. This node turns every badge into a "face" on
/swag/faces, exactly the format the robot node expects from a real face pipeline:

    {"embedding": one-hot on the badge id, "x", "y" (odom frame), "distance", "bearing", "badge"}

The Mac face service then registers / identifies / verifies badges like faces (a badge
always matches itself and never another badge). Swapping in real face recognition later
means replacing only this node.

    ros2 run stretch_swag badge_faces
"""

import json
import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from ..core.geometry import badge_embedding
from .locate import Locator


class BadgeFaces(Node):
    def __init__(self):
        super().__init__("badge_faces")
        self.id_min = int(self.declare_parameter("badge_id_min", 100).value)
        self.id_max = int(self.declare_parameter("badge_id_max", 109).value)
        self.base_frame = self.declare_parameter("base_frame", "base_link").value
        self.max_distance = float(self.declare_parameter("max_distance", 4.5).value)
        self.rate_hz = float(self.declare_parameter("rate_hz", 5.0).value)
        self.locator = Locator(self, self.base_frame, self.declare_parameter("odom_topic", "/odom").value)
        self.last_pub = 0.0
        self.last_seen: dict[int, float] = {}
        self.create_subscription(MarkerArray, "/aruco/marker_array", self.on_markers, 10)
        self.pub = self.create_publisher(String, "/swag/faces", 10)
        self.create_timer(1.0, self.publish_empty_if_stale)
        self.get_logger().info(f"Badges {self.id_min}-{self.id_max} -> /swag/faces")

    def on_markers(self, msg: MarkerArray) -> None:
        now = time.time()
        if now - self.last_pub < 1.0 / self.rate_hz:
            return
        faces = []
        for m in msg.markers:
            if not (self.id_min <= m.id <= self.id_max):
                continue                      # the robot's own markers, or someone else's
            where = self.locator.describe(m.header.frame_id,
                                          (m.pose.position.x, m.pose.position.y, m.pose.position.z), m.header.stamp)
            if where is None or where["distance"] > self.max_distance:
                continue
            distance, bearing = where["distance"], where["bearing"]
            faces.append({"embedding": badge_embedding(m.id), **where, "badge": m.id})
            if now - self.last_seen.get(m.id, 0) > 5.0:
                self.get_logger().info(f"badge {m.id} at {distance:.2f} m, bearing {math.degrees(bearing):.0f} deg")
            self.last_seen[m.id] = now
        self.pub.publish(String(data=json.dumps(faces)))
        self.last_pub = now

    def publish_empty_if_stale(self) -> None:
        # The detector publishes nothing when it sees no markers; say "no faces" explicitly.
        if time.time() - self.last_pub > 1.0:
            self.pub.publish(String(data="[]"))
            self.last_pub = time.time()


def main():
    rclpy.init()
    node = BadgeFaces()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, RuntimeError, ExternalShutdownException):
            pass

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
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import MarkerArray

from ..core.geometry import badge_embedding, base_to_world, transform_point
from ..motion import yaw_of


class BadgeFaces(Node):
    def __init__(self):
        super().__init__("badge_faces")
        self.id_min = int(self.declare_parameter("badge_id_min", 100).value)
        self.id_max = int(self.declare_parameter("badge_id_max", 109).value)
        self.base_frame = self.declare_parameter("base_frame", "base_link").value
        self.max_distance = float(self.declare_parameter("max_distance", 4.5).value)
        self.rate_hz = float(self.declare_parameter("rate_hz", 5.0).value)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.robot = None                     # (x, y, yaw) from /odom
        self.last_pub = 0.0
        self.last_seen: dict[int, float] = {}
        self.create_subscription(Odometry, self.declare_parameter("odom_topic", "/odom").value, self.on_odom, 10)
        self.create_subscription(MarkerArray, "/aruco/marker_array", self.on_markers, 10)
        self.pub = self.create_publisher(String, "/swag/faces", 10)
        self.create_timer(1.0, self.publish_empty_if_stale)
        self.get_logger().info(f"Badges {self.id_min}-{self.id_max} -> /swag/faces")

    def on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose
        self.robot = (p.position.x, p.position.y, yaw_of(p.orientation))

    def on_markers(self, msg: MarkerArray) -> None:
        now = time.time()
        if now - self.last_pub < 1.0 / self.rate_hz:
            return
        faces = []
        for m in msg.markers:
            if not (self.id_min <= m.id <= self.id_max):
                continue                      # the robot's own markers, or someone else's
            try:
                tf = self.tf.lookup_transform(self.base_frame, m.header.frame_id, Time())
            except TransformException as e:
                self.get_logger().warn(f"no transform {m.header.frame_id} -> {self.base_frame}: {e}",
                                       throttle_duration_sec=5.0)
                continue
            t, r = tf.transform.translation, tf.transform.rotation
            px, py, _ = transform_point((t.x, t.y, t.z), (r.x, r.y, r.z, r.w),
                                        (m.pose.position.x, m.pose.position.y, m.pose.position.z))
            distance = math.hypot(px, py)
            if distance > self.max_distance:
                continue
            x, y = base_to_world(self.robot, px, py) if self.robot else (px, py)
            faces.append({"embedding": badge_embedding(m.id), "x": round(x, 3), "y": round(y, 3),
                          "distance": round(distance, 3), "bearing": round(math.atan2(py, px), 3), "badge": m.id})
            if now - self.last_seen.get(m.id, 0) > 5.0:
                self.get_logger().info(f"badge {m.id} at {distance:.2f} m, bearing {math.degrees(math.atan2(py, px)):.0f} deg")
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
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, RuntimeError):
            pass

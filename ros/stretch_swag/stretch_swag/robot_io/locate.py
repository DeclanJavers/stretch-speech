"""Shared by the perception adapters: where is a point seen by the camera, in the robot's and the
world's (odometry) frame? Uses TF for the camera mounting and /odom for the robot's pose."""

import math

from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from ..core.geometry import base_to_world, transform_point
from ..motion import yaw_of


class Locator:
    def __init__(self, node, base_frame: str = "base_link", odom_topic: str = "/odom", odom_frame: str = "odom"):
        self.node, self.base_frame, self.odom_frame = node, base_frame, odom_frame
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, node)
        self.robot = None                                    # (x, y, yaw) from /odom
        node.create_subscription(Odometry, odom_topic, self._on_odom, 10)

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose
        self.robot = (p.position.x, p.position.y, yaw_of(p.orientation))

    def _lookup(self, target: str, frame: str, stamp):
        """Transform at the image's time (the robot may be turning); latest if that's unavailable."""
        for t, timeout in ((Time.from_msg(stamp), 0.0), (Time(), 0.0)) if stamp is not None else ((Time(), 0.0),):
            try:
                return self.tf.lookup_transform(target, frame, t, timeout=Duration(seconds=timeout))
            except TransformException as e:
                err = e
        self.node.get_logger().warn(f"no transform {frame} -> {target}: {err}", throttle_duration_sec=5.0)
        return None

    @staticmethod
    def _apply(tf, point):
        t, r = tf.transform.translation, tf.transform.rotation
        return transform_point((t.x, t.y, t.z), (r.x, r.y, r.z, r.w), point)

    def to_base(self, frame: str, point, stamp=None) -> tuple[float, float, float] | None:
        tf = self._lookup(self.base_frame, frame, stamp)
        return self._apply(tf, point) if tf else None

    def describe(self, frame: str, point, stamp=None) -> dict | None:
        """A camera-frame point (seen at `stamp`) -> {"x", "y" (odom), "distance", "bearing" (robot frame)}."""
        b = self.to_base(frame, point, stamp)
        if b is None:
            return None
        px, py, _ = b
        odom = self._lookup(self.odom_frame, frame, stamp)
        if odom is not None:
            x, y, _ = self._apply(odom, point)
        else:
            x, y = base_to_world(self.robot, px, py) if self.robot else (px, py)
        return {"x": round(x, 3), "y": round(y, 3), "distance": round(math.hypot(px, py), 3),
                "bearing": round(math.atan2(py, px), 3)}

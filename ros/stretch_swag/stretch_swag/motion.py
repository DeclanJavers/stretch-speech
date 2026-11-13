"""Base motion for the demo: drive a distance, turn an angle, stop.

Moves are closed-loop on odometry (stop when the robot has actually gone far
enough), so they still come out right if the sim runs slower than real time.
If no odometry is arriving, they fall back to timing (distance / speed).

The sim driver stops the base when Twists stop arriving, so we publish
continuously while moving. Every move ends with a zero Twist, including on
errors and Ctrl+C.
"""

import math
import threading
import time

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

RATE_HZ = 20.0
ODOM_STALE_S = 1.0


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class BaseDriver:
    def __init__(self, node, cmd_vel_topic: str = "/stretch/cmd_vel", odom_topic: str = "/odom"):
        self.node = node
        self.pub = node.create_publisher(Twist, cmd_vel_topic, 10)
        self.cancel = threading.Event()
        self._pose = None          # (x, y, yaw)
        self._pose_t = 0.0
        self._lock = threading.Lock()
        node.create_subscription(Odometry, odom_topic, self._on_odom, 10)

    # ---- odometry --------------------------------------------------------------
    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose
        with self._lock:
            self._pose = (p.position.x, p.position.y, _yaw(p.orientation))
            self._pose_t = time.monotonic()

    def pose(self):
        with self._lock:
            if self._pose is None or time.monotonic() - self._pose_t > ODOM_STALE_S:
                return None
            return self._pose

    # ---- moves -----------------------------------------------------------------
    def drive_distance(self, distance_m: float, speed_mps: float) -> dict:
        """Drive straight; negative distance = backwards. Returns {success, reason}."""
        speed = math.copysign(abs(speed_mps), distance_m)
        target = abs(distance_m)
        start = self.pose()
        if start is None:
            self.node.get_logger().warn("No odometry: driving by time instead")

        def progress(p):
            return math.hypot(p[0] - start[0], p[1] - start[1])

        return self._move(linear=speed, angular=0.0, target=target, start=start, progress=progress,
                          expected_s=target / abs(speed_mps))

    def turn(self, angle_rad: float, speed_radps: float) -> dict:
        """Turn in place; positive = counter-clockwise. Returns {success, reason}."""
        w = math.copysign(abs(speed_radps), angle_rad)
        target = abs(angle_rad)
        start = self.pose()
        if start is None:
            self.node.get_logger().warn("No odometry: turning by time instead")
        state = {"last": start[2] if start else 0.0, "turned": 0.0}

        def progress(p):
            state["turned"] += abs(_wrap(p[2] - state["last"]))
            state["last"] = p[2]
            return state["turned"]

        return self._move(linear=0.0, angular=w, target=target, start=start, progress=progress,
                          expected_s=target / abs(speed_radps))

    def _move(self, linear, angular, target, start, progress, expected_s) -> dict:
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        t0 = time.monotonic()
        # Generous cap: the sim in the VM runs at ~1/3 real time, and turning in place
        # slips on top of that, so allow many times the nominal time.
        timeout_s = expected_s * 15 + 10
        try:
            while True:
                if self.cancel.is_set():
                    return {"success": False, "reason": "cancelled"}
                elapsed = time.monotonic() - t0
                if start is None:                         # time-based fallback
                    if elapsed >= expected_s:
                        return {"success": True, "reason": "timed (no odometry)"}
                else:
                    p = self.pose()
                    if p is not None and progress(p) >= target:
                        return {"success": True, "reason": ""}
                    if elapsed > timeout_s:
                        done = progress(p) if p is not None else 0.0
                        return {"success": False,
                                "reason": f"timeout after {elapsed:.0f}s ({done:.2f} of {target:.2f})"}
                self.pub.publish(msg)
                time.sleep(1.0 / RATE_HZ)
        finally:
            self.stop()

    def stop(self) -> None:
        # A few zero Twists in case one is dropped.
        zero = Twist()
        for _ in range(3):
            try:
                self.pub.publish(zero)
            except Exception:  # noqa: BLE001 - context may already be shutting down
                return
            time.sleep(0.02)

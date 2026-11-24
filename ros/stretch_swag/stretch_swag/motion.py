"""Base motion skills, closed-loop on odometry.

    go_to(x, y, stop_distance)   turn toward the goal, drive there (absolute position)
    turn_to(yaw)                 face an absolute heading
    turn(angle)                  turn by a relative angle (can exceed 180 deg, for spinning)
    drive_distance(d)            straight line (kept for the old demo node)

Every skill returns {success, reason}, publishes Twists continuously while moving
(the sim driver stops the base when they stop), and always ends with a zero Twist.

Interrupts: setting `driver.interrupt` makes the running skill stop and return
reason "interrupted" (used when perception spots someone mid-move).

Safety: `safety()` is called each step while driving forward; if it returns a reason
(someone within 0.6 m ahead), the base stops and waits; blocked for too long = failure.
"""

import math
import threading
import time
from typing import Callable

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

RATE_HZ = 20.0
ODOM_STALE_S = 1.0


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class BaseDriver:
    def __init__(self, node, cmd_vel_topic: str = "/stretch/cmd_vel", odom_topic: str = "/odom",
                 linear_speed: float = 0.3, angular_speed: float = 0.8,
                 safety: Callable[[], str | None] | None = None, blocked_timeout_s: float = 15.0,
                 dry_run: bool = False):
        self.node = node
        self.pub = node.create_publisher(Twist, cmd_vel_topic, 10)
        self.linear_speed, self.angular_speed = linear_speed, angular_speed
        self.safety = safety
        self.blocked_timeout_s = blocked_timeout_s
        self.dry_run = dry_run                # motion disabled: skills "succeed" without moving
        self.cancel = threading.Event()       # shutdown: abort everything
        self.interrupt = threading.Event()    # perception: stop the current move
        self._pose = None
        self._pose_t = 0.0
        self._lock = threading.Lock()
        node.create_subscription(Odometry, odom_topic, self._on_odom, 10)

    # ---- odometry --------------------------------------------------------------
    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose
        with self._lock:
            self._pose = (p.position.x, p.position.y, yaw_of(p.orientation))
            self._pose_t = time.monotonic()

    def pose(self):
        """(x, y, yaw), or None if odometry is missing or stale."""
        with self._lock:
            if self._pose is None or time.monotonic() - self._pose_t > ODOM_STALE_S:
                return None
            return self._pose

    def wait_for_pose(self, timeout_s: float = 10.0):
        end = time.monotonic() + timeout_s
        while time.monotonic() < end and not self.cancel.is_set():
            p = self.pose()
            if p:
                return p
            time.sleep(0.1)
        return None

    # ---- skills ----------------------------------------------------------------
    def _publish(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        self.pub.publish(msg)

    def _aborted(self):
        if self.cancel.is_set():
            return {"success": False, "reason": "cancelled"}
        if self.interrupt.is_set():
            return {"success": False, "reason": "interrupted"}
        return None

    def _dry(self) -> dict | None:
        if self.dry_run:
            time.sleep(1.0)
            return {"success": True, "reason": "motion disabled"}
        return None

    def turn_to(self, yaw: float, tolerance: float = 0.04, timeout_s: float = 30.0) -> dict:
        if (r := self._dry()):
            return r
        t0 = time.monotonic()
        try:
            while True:
                if (r := self._aborted()):
                    return r
                p = self.pose()
                if p is None:
                    return {"success": False, "reason": "no odometry"}
                err = wrap(yaw - p[2])
                if abs(err) < tolerance:
                    return {"success": True, "reason": ""}
                if time.monotonic() - t0 > timeout_s:
                    return {"success": False, "reason": "turn timeout"}
                w = clamp(2.0 * err, -self.angular_speed, self.angular_speed)
                if abs(w) < 0.15:
                    w = math.copysign(0.15, err)
                self._publish(0.0, w)
                time.sleep(1.0 / RATE_HZ)
        finally:
            self.stop()

    def turn(self, angle: float, timeout_s: float = 60.0) -> dict:
        """Relative turn, tracking accumulated rotation (works past 180 deg)."""
        if (r := self._dry()):
            return r
        p = self.pose()
        if p is None:
            return {"success": False, "reason": "no odometry"}
        last, turned, t0 = p[2], 0.0, time.monotonic()
        w = math.copysign(self.angular_speed, angle)
        try:
            while True:
                if (r := self._aborted()):
                    return r
                p = self.pose()
                if p is not None:
                    turned += abs(wrap(p[2] - last))
                    last = p[2]
                if turned >= abs(angle) - 0.02:
                    return {"success": True, "reason": ""}
                if time.monotonic() - t0 > timeout_s:
                    return {"success": False, "reason": "turn timeout"}
                self._publish(0.0, w)
                time.sleep(1.0 / RATE_HZ)
        finally:
            self.stop()

    def go_to(self, x: float, y: float, stop_distance: float = 0.0, tolerance: float = 0.08,
              timeout_s: float = 120.0) -> dict:
        if (r := self._dry()):
            return r
        p = self.pose()
        if p is None:
            return {"success": False, "reason": "no odometry"}
        if math.hypot(x - p[0], y - p[1]) <= stop_distance + tolerance:
            return {"success": True, "reason": "already there"}
        r = self.turn_to(math.atan2(y - p[1], x - p[0]), tolerance=0.1)
        if not r["success"]:
            return r
        t0 = time.monotonic()
        blocked_since = None
        try:
            while True:
                if (r := self._aborted()):
                    return r
                p = self.pose()
                if p is None:
                    return {"success": False, "reason": "no odometry"}
                dx, dy = x - p[0], y - p[1]
                dist = math.hypot(dx, dy) - stop_distance
                if dist <= tolerance:
                    return {"success": True, "reason": ""}
                if time.monotonic() - t0 > timeout_s:
                    return {"success": False, "reason": "go_to timeout"}
                reason = self.safety() if self.safety else None
                if reason:
                    self._publish(0.0, 0.0)
                    blocked_since = blocked_since or time.monotonic()
                    if time.monotonic() - blocked_since > self.blocked_timeout_s:
                        return {"success": False, "reason": f"blocked: {reason}"}
                    time.sleep(1.0 / RATE_HZ)
                    continue
                blocked_since = None
                heading_err = wrap(math.atan2(dy, dx) - p[2])
                v = clamp(0.8 * dist, 0.05, self.linear_speed) if abs(heading_err) < 0.6 else 0.0
                w = clamp(2.0 * heading_err, -self.angular_speed, self.angular_speed)
                self._publish(v, w)
                time.sleep(1.0 / RATE_HZ)
        finally:
            self.stop()

    def drive_distance(self, distance_m: float, speed_mps: float | None = None) -> dict:
        """Straight line, relative (used by the old demo node)."""
        if (r := self._dry()):
            return r
        p = self.pose()
        if p is None:
            return {"success": False, "reason": "no odometry"}
        speed = math.copysign(abs(speed_mps or self.linear_speed), distance_m)
        start = p
        try:
            while True:
                if (r := self._aborted()):
                    return r
                p = self.pose()
                if p and math.hypot(p[0] - start[0], p[1] - start[1]) >= abs(distance_m):
                    return {"success": True, "reason": ""}
                self._publish(speed, 0.0)
                time.sleep(1.0 / RATE_HZ)
        finally:
            self.stop()

    def stop(self) -> None:
        for _ in range(3):
            try:
                self._publish(0.0, 0.0)
            except Exception:  # noqa: BLE001 - context may be shutting down
                return
            time.sleep(0.02)

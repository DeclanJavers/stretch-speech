"""Base motion for the demo: drive straight for a while, then stop.

The sim driver stops the base when Twists stop arriving, so we publish
continuously while moving. Every move ends with a zero Twist, including on
errors and Ctrl+C.
"""

import threading
import time

from geometry_msgs.msg import Twist

RATE_HZ = 10.0


class BaseDriver:
    def __init__(self, node, topic: str = "/stretch/cmd_vel"):
        self.node = node
        self.pub = node.create_publisher(Twist, topic, 10)
        self.cancel = threading.Event()

    def drive(self, linear_mps: float, duration_s: float) -> dict:
        """Drive straight (negative speed = backwards). Returns {success, reason}."""
        msg = Twist()
        msg.linear.x = float(linear_mps)
        end = time.monotonic() + duration_s
        try:
            while time.monotonic() < end:
                if self.cancel.is_set():
                    return {"success": False, "reason": "cancelled"}
                self.pub.publish(msg)
                time.sleep(1.0 / RATE_HZ)
            return {"success": True, "reason": ""}
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

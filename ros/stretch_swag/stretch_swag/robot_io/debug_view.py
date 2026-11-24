"""One window to watch the robot: the camera (with detections boxed) on the left, and the
people it knows on the right: their face, name, what they asked for and how it went.

    ros2 run stretch_swag debug_view
    ros2 run rqt_image_view rqt_image_view /swag/debug

Inputs: /swag/camera_debug (from face_detect or aruco_detect) and /swag/registry (robot node).
"""

import base64
import json
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String

from .aruco_detect import bgr_to_image, image_to_bgr

H = 720                 # output height
PANEL_W = 380
ROW_H = 88
OUTCOME_COLOURS = {"served": (60, 160, 60), "waiting": (0, 140, 230), "talking": (200, 120, 0),
                   "declined": (120, 120, 120), "missed": (40, 40, 200), "no_response": (120, 120, 120),
                   "ended": (120, 120, 120)}


def decode_thumb(b64: str | None, size: int) -> np.ndarray | None:
    if not b64:
        return None
    try:
        img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        return cv2.resize(img, (size, size)) if img is not None else None
    except (ValueError, cv2.error):
        return None


def panel(registry: dict, height: int) -> np.ndarray:
    p = np.full((height, PANEL_W, 3), 245, np.uint8)
    state = registry.get("state", "?").replace("_", " ")
    held = registry.get("held_item")
    cv2.putText(p, state, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    if held:
        cv2.putText(p, f"holding: {held}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
    y = 64
    people = registry.get("people", [])
    if not people:
        cv2.putText(p, "nobody met yet", (10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)
    for q in reversed(people):                                     # newest first
        if y + ROW_H > height:
            break
        if q["id"] == registry.get("target"):
            cv2.rectangle(p, (2, y - 2), (PANEL_W - 3, y + ROW_H - 6), (0, 180, 255), 2)
        thumb = decode_thumb(q.get("thumb"), ROW_H - 12)
        if thumb is not None:
            p[y + 2:y + 2 + thumb.shape[0], 6:6 + thumb.shape[1]] = thumb
        else:
            cv2.rectangle(p, (6, y + 2), (6 + ROW_H - 12, y + ROW_H - 10), (200, 200, 200), -1)
            label = f"#{q['badge']}" if q.get("badge") is not None else "?"
            cv2.putText(p, label, (12, y + ROW_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 2)
        x = ROW_H + 4
        cv2.putText(p, f"{q['id']}  {q.get('name') or '(no name)'}", (x, y + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (20, 20, 20), 2)
        cv2.putText(p, f"wants: {q.get('item') or '-'}", (x, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        outcome = q.get("outcome") or "-"
        cv2.putText(p, outcome, (x, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    OUTCOME_COLOURS.get(outcome, (60, 60, 60)), 2)
        y += ROW_H
    return p


class DebugView(Node):
    def __init__(self):
        super().__init__("debug_view")
        self.registry: dict = {}
        self.camera: np.ndarray | None = None
        self.camera_t = 0.0
        self.header = None
        self.create_subscription(Image, "/swag/camera_debug", self.on_camera, 2)
        self.create_subscription(String, "/swag/registry", self.on_registry, 2)
        self.bags: tuple[float, dict] = (0.0, {})
        self.create_subscription(String, "/swag/bags", lambda m: setattr(self, "bags", (time.time(), json.loads(m.data))), 2)
        self.pub = self.create_publisher(Image, "/swag/debug", 2)
        self.create_timer(0.25, self.render)
        self.get_logger().info("Debug view on /swag/debug  (ros2 run rqt_image_view rqt_image_view /swag/debug)")

    def on_camera(self, msg: Image) -> None:
        img = image_to_bgr(msg)
        if img is not None:
            self.camera, self.camera_t, self.header = img, time.time(), msg.header

    def on_registry(self, msg: String) -> None:
        try:
            self.registry = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def render(self) -> None:
        if self.camera is not None and time.time() - self.camera_t < 3.0:
            c = self.camera.copy()
            t, bags = self.bags
            if time.time() - t < 1.0 and bags.get("size") == [c.shape[1], c.shape[0]]:
                for b in bags.get("bags", []):                      # bags found by colour
                    x, y, w, h = b["box"]
                    cv2.rectangle(c, (x, y), (x + w, y + h), (255, 255, 255), 3)
                    cv2.putText(c, f"{b['item']} {b['distance']:.2f}m", (x, max(14, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cam = cv2.resize(c, (int(c.shape[1] * H / c.shape[0]), H))
        else:
            cam = np.full((H, 405, 3), 40, np.uint8)
            cv2.putText(cam, "no camera image", (20, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        out = np.hstack([cam, panel(self.registry, H)])
        self.pub.publish(bgr_to_image(out, self.header or Header()))


def main():
    rclpy.init()
    node = DebugView()
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

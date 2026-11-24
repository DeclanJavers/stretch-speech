"""Finds the goodie bags by colour (pen = red, t-shirt = blue, plushie = green by default).

No depth camera needed: the bag stands on the table, so the bottom outline of its colour blob
lies on the table top. Each outline pixel's camera ray, intersected with the plane at the
table's height (TF gives the camera's pose), gives the bag's visible bottom edges in 3D (its front
edge, maybe one side). Middle of the bag: halfway along the table between the outline's ends, half
the bag's depth behind its front edge. Outlines that aren't bag-wide (a blue sweater, jeans) and
blobs cut off by the image edge are ignored.

    /swag/bags   {"bags": [{"item", "x", "y", "z" (base_link, middle of the bag at table height),
                            "wx", "wy" (odom), "distance", "box": [x, y, w, h] (upright image)}],
                  "size": [w, h] (upright image)}

The table's facing (which way is "front") comes from the robot node (/swag/table_facing, from the
lane markers). Tune for the real bags: table_height_m, bag_width_m, bag_depth_m and the colour ranges
(OpenCV HSV, hue 0-180): colours:="pen:0-8+172-180,shirt:100-128,plushie:42-85".

    ros2 run stretch_swag bag_detect
"""

import json
import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64, String

from ..core.geometry import ray_plane_z
from ..motion import yaw_of
from .aruco_detect import image_to_bgr
from .face_detect import ROTATIONS, unrotate
from .locate import Locator


def parse_colours(text: str) -> dict[str, list[tuple[int, int]]]:
    """"pen:0-8+172-180,shirt:100-128" -> {"pen": [(0, 8), (172, 180)], "shirt": [(100, 128)]}"""
    out = {}
    for part in text.split(","):
        item, _, ranges = part.strip().partition(":")
        out[item] = [tuple(int(v) for v in r.split("-")) for r in ranges.split("+") if r]
    return out


def colour_blobs(hsv: np.ndarray, hue_ranges, min_s: int, min_v: int, min_area: int, with_bottom: bool = False):
    """Connected regions of a colour: [(area, x, y, w, h)] in the image given; with_bottom adds
    the blob's bottom outline [(column, lowest row)] (every few columns)."""
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in hue_ranges:
        mask |= cv2.inRange(hsv, (lo, min_s, min_v), (hi, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for k in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[k])
        if a < min_area:
            continue
        if not with_bottom:
            out.append((a, x, y, w, h))
            continue
        sub = labels[y:y + h, x:x + w] == k
        cols = range(0, w, max(1, w // 40))
        bottom = [(x + c, y + int(np.nonzero(sub[:, c])[0].max())) for c in cols if sub[:, c].any()]
        out.append((a, x, y, w, h, bottom))
    return out


def bag_centres(points, normal, bag_w: float, bag_d: float, max_bags: int = 3):
    """Middles of the bags whose bottom outline on the table is `points` (x, y), given the table's
    facing `normal` (unit, table -> open side): [((x, y), width)]. Same-coloured bags side by side
    can merge into one blob: an outline n bags wide gives n bags. [] if it isn't bag-sized."""
    along = (-normal[1], normal[0])
    f = [p[0] * normal[0] + p[1] * normal[1] for p in points]       # toward the open side
    front = max(f)
    # only the front edge (points within 2 cm of the frontmost): past the corners the outline
    # leaves the table and its projection is meaningless
    a = [p[0] * along[0] + p[1] * along[1] for p, fi in zip(points, f) if fi > front - 0.02]
    width = max(a) - min(a)
    n = max(1, round(width / (bag_w * 1.2)))                         # bags ~20% gap apart, roughly
    if n > max_bags or not (0.6 * bag_w < width / n < 1.4 * bag_w if n == 1 else width > n * bag_w * 0.9):
        return []
    fc = front - bag_d / 2
    firsts = [min(a) + bag_w / 2 + k * (width - bag_w) / (n - 1) for k in range(n)] if n > 1 else [(max(a) + min(a)) / 2]
    return [((fc * normal[0] + ac * along[0], fc * normal[1] + ac * along[1]), width / n) for ac in firsts]


class BagDetect(Node):
    def __init__(self):
        super().__init__("bag_detect")
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.colours = parse_colours(p("colours", "pen:0-8+172-180,shirt:100-128,plushie:42-85"))
        self.min_s, self.min_v = int(p("min_saturation", 150)), int(p("min_value", 60))   # bags are vivid; jeans aren't
        self.table_z = float(p("table_height_m", 0.48))
        self.bag_w, self.bag_d = float(p("bag_width_m", 0.16)), float(p("bag_depth_m", 0.10))
        self.bag_h = float(p("bag_height_m", 0.22))
        self.max_range = float(p("max_range_m", 2.0))
        # which way the table faces (table -> open side, odom yaw): from the robot once it has seen the
        # lane markers (/swag/table_facing), until then from table_dir_deg (heading toward the table)
        self.facing = math.radians(float(p("table_dir_deg", 180.0)) + 180.0)
        self.create_subscription(Float64, "/swag/table_facing", lambda m: setattr(self, "facing", m.data), 1)
        self.rotate = p("rotate", "cw")
        self.max_hz = float(p("max_hz", 5.0))
        self.locator = Locator(self)
        self.K = None
        self.last = 0.0
        self.create_subscription(CameraInfo, p("camera_info_topic", "/camera/color/camera_info"),
                                 lambda m: setattr(self, "K", np.array(m.k, dtype=np.float64).reshape(3, 3)),
                                 qos_profile_sensor_data)
        self.create_subscription(Image, p("image_topic", "/camera/color/image_raw"), self.on_image,
                                 qos_profile_sensor_data)
        self.pub = self.create_publisher(String, "/swag/bags", 5)
        self.get_logger().info(f"Bag detection up: {', '.join(self.colours)} (table at {self.table_z:.2f} m)")

    def on_image(self, msg: Image) -> None:
        now = time.time()
        if self.K is None or now - self.last < 1.0 / self.max_hz:
            return
        self.last = now
        raw = image_to_bgr(msg)
        if raw is None:
            return
        h0, w0 = raw.shape[:2]
        img = cv2.rotate(raw, ROTATIONS[self.rotate]) if self.rotate in ROTATIONS else raw
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        cam = self.locator._lookup(self.locator.base_frame, msg.header.frame_id, msg.header.stamp)
        base = self.locator._lookup(self.locator.odom_frame, self.locator.base_frame, msg.header.stamp)
        bags = []
        if cam is not None:
            t, r = cam.transform.translation, cam.transform.rotation
            origin, rot = (t.x, t.y, t.z), (r.x, r.y, r.z, r.w)
            # the table's facing (from the table toward the open side) in the robot's frame
            yaw = yaw_of(base.transform.rotation) if base is not None else 0.0
            n = (math.cos(self.facing - yaw), math.sin(self.facing - yaw))
            for item, ranges in self.colours.items():
                for area, x, y, w, h, bottom in colour_blobs(hsv, ranges, self.min_s, self.min_v, 150, True):
                    if x <= 1 or y <= 1 or x + w >= img.shape[1] - 1 or y + h >= img.shape[0] - 1:
                        continue                                  # cut off by the image edge: can't tell its size
                    pts = []
                    for bx_px, by_px in bottom:                   # the bag's bottom outline, on the table
                        u, v = unrotate(bx_px, by_px, self.rotate, w0, h0)
                        p = ray_plane_z(origin, rot, ((u - cx) / fx, (v - cy) / fy, 1.0), self.table_z)
                        if p is not None:
                            pts.append(p)
                    if len(pts) < 5:
                        continue
                    for (bx, by), width in bag_centres(pts, n, self.bag_w, self.bag_d):
                        if not 0.2 < math.hypot(bx, by) < self.max_range:
                            continue
                        bag = {"item": item, "x": round(bx, 3), "y": round(by, 3), "z": round(self.table_z, 3),
                               "distance": round(math.hypot(bx, by), 3), "width": round(width, 3),
                               "box": [x, y, w, h], "area": area}
                        if base is not None:
                            bt, c, sn = base.transform.translation, math.cos(yaw), math.sin(yaw)
                            bag["wx"], bag["wy"] = round(bt.x + c * bx - sn * by, 3), round(bt.y + sn * bx + c * by, 3)
                        bags.append(bag)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9          # when the image was taken
        self.pub.publish(String(data=json.dumps({"bags": bags, "size": [img.shape[1], img.shape[0]],
                                                 "stamp": stamp})))

def main():
    rclpy.init()
    node = BagDetect()
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

"""Lightweight ArUco detector (colour image only), plus a debug image so you can see what it sees.

Publishes /aruco/marker_array in the same form as Stretch's own detector
(stretch_core detect_aruco_markers): one Marker per badge with its id and 3D position in the
camera's optical frame. badge_faces consumes it either way.

Why a second detector: Stretch's needs a depth image aligned to colour, which the MuJoCo sim
doesn't publish. This one needs only /camera/color/image_raw + camera_info, so it works in
the sim and is a fallback on the real robot.

    ros2 run stretch_swag aruco_detect --ros-args --params-file config/badges.yaml
    ros2 run rqt_image_view rqt_image_view /swag/camera_debug      # what the robot sees

Marker sizes come from the same badges.yaml as Stretch's detector (aruco_marker_info.<id>.length_mm).
"""

import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray

ENCODINGS = {"rgb8": (3, True), "bgr8": (3, False), "mono8": (1, False)}


def image_to_bgr(msg: Image) -> np.ndarray | None:
    if msg.encoding not in ENCODINGS:
        return None
    ch, rgb = ENCODINGS[msg.encoding]
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * ch]
    a = a.reshape(msg.height, msg.width, ch) if ch > 1 else a.reshape(msg.height, msg.width)
    if ch == 1:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    return a[:, :, ::-1].copy() if rgb else a.copy()


def bgr_to_image(img: np.ndarray, header) -> Image:
    msg = Image(header=header, height=img.shape[0], width=img.shape[1], encoding="bgr8",
                is_bigendian=0, step=img.shape[1] * 3)
    msg.data = img.tobytes()
    return msg


def make_detector(dict_id: int):
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "ArucoDetector"):                       # OpenCV >= 4.7
        det = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        return lambda gray: det.detectMarkers(gray)[:2]
    params = cv2.aruco.DetectorParameters_create()                # OpenCV 4.5 (Ubuntu 22.04 apt)
    return lambda gray: cv2.aruco.detectMarkers(gray, dictionary, parameters=params)[:2]


def rotation_to_quaternion(r: np.ndarray) -> tuple[float, float, float, float]:
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return x, y, z, w


class ArucoDetect(Node):
    def __init__(self):
        super().__init__("aruco_detect", automatically_declare_parameters_from_overrides=True)
        p = lambda name, default: (self.get_parameter(name).value if self.has_parameter(name)
                                   else self.declare_parameter(name, default).value)
        self.default_mm = float(p("default_length_mm", 100.0))
        self.max_hz = float(p("max_hz", 10.0))
        self.debug = bool(p("debug_image", True))
        self.upright = bool(p("debug_upright", True))   # Stretch's head camera is mounted sideways
        self.lengths = {}
        for key, param in self.get_parameters_by_prefix("aruco_marker_info").items():
            marker_id, _, field = key.partition(".")
            if field == "length_mm":
                self.lengths[int(marker_id)] = float(param.value)
        self.detect = make_detector(getattr(cv2.aruco, p("dictionary", "DICT_6X6_250")))
        self.K = self.D = None
        self.last = 0.0
        self.seen_log: dict[int, float] = {}
        self.create_subscription(CameraInfo, p("camera_info_topic", "/camera/color/camera_info"), self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, p("image_topic", "/camera/color/image_raw"), self.on_image, qos_profile_sensor_data)
        self.markers_pub = self.create_publisher(MarkerArray, "/aruco/marker_array", 5)
        self.debug_pub = self.create_publisher(Image, "/swag/camera_debug", 2)
        self.get_logger().info(f"ArUco detector up (sizes for {len(self.lengths)} ids, default "
                               f"{self.default_mm:.0f} mm). Debug image: /swag/camera_debug")

    def on_info(self, msg: CameraInfo) -> None:
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.array(msg.d if len(msg.d) else [0.0] * 5, dtype=np.float64)

    def on_image(self, msg: Image) -> None:
        now = time.time()
        if self.K is None or now - self.last < 1.0 / self.max_hz:
            return
        self.last = now
        img = image_to_bgr(msg)
        if img is None:
            self.get_logger().warn(f"unsupported image encoding {msg.encoding}", throttle_duration_sec=10.0)
            return
        corners, ids = self.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        out = MarkerArray()
        notes = []
        for c, marker_id in zip(corners, [] if ids is None else ids.ravel()):
            marker_id = int(marker_id)
            half = self.lengths.get(marker_id, self.default_mm) / 2000.0
            obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], np.float64)
            ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2).astype(np.float64), self.K, self.D,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            t = tvec.ravel()
            m = Marker(header=msg.header, ns="aruco", id=marker_id, type=Marker.CUBE, action=Marker.ADD)
            m.pose = Pose()
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, t)
            q = rotation_to_quaternion(cv2.Rodrigues(rvec)[0])
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
            m.scale.x = m.scale.y = 2 * half
            m.scale.z = 0.005
            m.color.g = m.color.a = 1.0
            m.text = f"badge_{marker_id}"
            out.markers.append(m)
            dist = float(np.linalg.norm(t))
            notes.append((c, marker_id, dist))
            if now - self.seen_log.get(marker_id, 0) > 5:
                self.get_logger().info(f"marker {marker_id} at {dist:.2f} m ({int(np.ptp(c[0][:, 0]))} px wide)")
            self.seen_log[marker_id] = now
        self.markers_pub.publish(out)
        if self.debug and self.debug_pub.get_subscription_count():
            h = img.shape[0]
            for c, _, _ in notes:
                cv2.polylines(img, [c.astype(np.int32)], True, (0, 255, 0), 2)
            if self.upright:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            for c, marker_id, dist in notes:
                x, y = c[0].min(axis=0)
                if self.upright:
                    x, y = h - 1 - c[0][:, 1].max(), c[0][:, 0].min()
                cv2.putText(img, f"{marker_id} {dist:.1f}m", (int(x), max(12, int(y) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 0), 2)
            cv2.putText(img, f"{len(notes)} badge(s)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            self.debug_pub.publish(bgr_to_image(img, msg.header))


def main():
    rclpy.init()
    node = ArucoDetect()
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

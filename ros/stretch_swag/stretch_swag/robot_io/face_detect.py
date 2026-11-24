"""Real face recognition on the robot: YuNet (face detector) + SFace (face embeddings), both small
CNNs that run on the CPU with plain OpenCV (no PyTorch). Same output as badge_faces:

    /swag/faces   [{"embedding": [128 floats], "x", "y" (odom), "distance", "bearing",
                    "face_px": face width in pixels, "thumb": small JPEG (base64) of the aligned face}]
    /swag/camera_debug   the camera image, upright, faces boxed (view: rqt_image_view)

The Mac face service then registers / identifies / verifies the embeddings (cosine similarity).

Distance comes from the face's size in the image (the detector's box on an adult face is ~14 cm wide: set
face_width_m if it reads consistently long or short). Stretch's head camera is mounted sideways,
so images are rotated upright before detection (rotate: cw | ccw | none).

Models: run `bash models/get_models.sh` once (needs OpenCV >= 4.8 for this YuNet).

    ros2 run stretch_swag face_detect
"""

import base64
import json
import pathlib
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from .aruco_detect import bgr_to_image, image_to_bgr
from .locate import Locator

MODELS = pathlib.Path(__file__).resolve().parents[2] / "models"
ROTATIONS = {"cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE}


def unrotate(xr: float, yr: float, rotate: str, w: int, h: int) -> tuple[float, float]:
    """A pixel in the upright (rotated) image -> the same pixel in the camera's original image (w x h)."""
    if rotate == "cw":                 # original (x, y) -> rotated (h-1-y, x)
        return yr, h - 1 - xr
    if rotate == "ccw":                # original (x, y) -> rotated (y, w-1-x)
        return w - 1 - yr, xr
    return xr, yr


class FaceDetect(Node):
    def __init__(self):
        super().__init__("face_detect")
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        models = pathlib.Path(p("models_dir", str(MODELS)))
        self.rotate = p("rotate", "cw")
        self.max_hz = float(p("max_hz", 5.0))
        self.face_width_m = float(p("face_width_m", 0.14))
        self.min_score = float(p("min_score", 0.75))
        self.min_face_px = int(p("min_face_px", 16))
        self.max_distance = float(p("max_distance", 7.0))
        self.detector = cv2.FaceDetectorYN.create(str(models / "face_detection_yunet_2023mar.onnx"), "",
                                                  (320, 320), self.min_score, 0.3, 50)
        self.recognizer = cv2.FaceRecognizerSF.create(str(models / "face_recognition_sface_2021dec.onnx"), "")
        self.locator = Locator(self)
        self.K = None
        self.last = 0.0
        self.last_pub = 0.0
        self.create_subscription(CameraInfo, p("camera_info_topic", "/camera/color/camera_info"),
                                 lambda m: setattr(self, "K", np.array(m.k, dtype=np.float64).reshape(3, 3)),
                                 qos_profile_sensor_data)
        self.create_subscription(Image, p("image_topic", "/camera/color/image_raw"), self.on_image,
                                 qos_profile_sensor_data)
        self.pub = self.create_publisher(String, "/swag/faces", 10)
        self.debug_pub = self.create_publisher(Image, "/swag/camera_debug", 2)
        self.create_timer(1.0, self.publish_empty_if_stale)
        self.get_logger().info(f"Face detection up (YuNet + SFace, OpenCV {cv2.__version__}, rotate={self.rotate})")

    def on_image(self, msg: Image) -> None:
        now = time.time()
        if self.K is None or now - self.last < 1.0 / self.max_hz:
            return
        self.last = now
        raw = image_to_bgr(msg)
        if raw is None:
            return
        h, w = raw.shape[:2]
        img = cv2.rotate(raw, ROTATIONS[self.rotate]) if self.rotate in ROTATIONS else raw
        self.detector.setInputSize((img.shape[1], img.shape[0]))
        _, found = self.detector.detect(img)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        faces, boxes = [], []
        for f in [] if found is None else found:
            x, y, bw, bh = f[:4]
            if bw < self.min_face_px:
                continue
            distance = (fx + fy) / 2 * self.face_width_m / bw
            if distance > self.max_distance:
                continue
            u, v = unrotate(x + bw / 2, y + bh / 2, self.rotate, w, h)
            ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
            point = ray / np.linalg.norm(ray) * distance             # camera optical frame
            where = self.locator.describe(msg.header.frame_id, point, msg.header.stamp)
            if where is None:
                continue
            crop = self.recognizer.alignCrop(img, f)
            emb = self.recognizer.feature(crop).ravel()
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            ok, jpg = cv2.imencode(".jpg", cv2.resize(crop, (64, 64)), [cv2.IMWRITE_JPEG_QUALITY, 70])
            faces.append({**where, "embedding": [round(float(v), 4) for v in emb], "face_px": int(bw),
                          "score": round(float(f[-1]), 2), "thumb": base64.b64encode(jpg).decode() if ok else None})
            boxes.append((int(x), int(y), int(bw), int(bh), where["distance"]))
        self.pub.publish(String(data=json.dumps(faces)))
        self.last_pub = now
        if self.debug_pub.get_subscription_count():
            for x, y, bw, bh, d in boxes:
                cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(img, f"{d:.1f}m", (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 0), 2)
            cv2.putText(img, f"{len(boxes)} face(s)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            self.debug_pub.publish(bgr_to_image(img, msg.header))

    def publish_empty_if_stale(self) -> None:
        if time.time() - self.last_pub > 1.0:
            self.pub.publish(String(data="[]"))
            self.last_pub = time.time()


def main():
    rclpy.init()
    node = FaceDetect()
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

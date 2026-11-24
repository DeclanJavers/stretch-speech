"""Speaks the robot's lines (/swag/say) out loud.

Backends (parameter `backend`, tried in this order with "auto"):
    mac     POST to the Mac's speech service (mac-service/speech_service.py, Kokoro voice);
            the sound comes out of the Mac, so keep the laptop near the robot
    espeak  `espeak-ng` (or `spd-say`) on this computer: robotic voice, no network needed
    print   just print the line

    ros2 run stretch_swag speaker_node --ros-args -p backend:=mac -p mac_url:=http://192.168.64.1:8200
"""

import json
import queue
import shutil
import subprocess
import threading
import urllib.error
import urllib.request

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SpeakerNode(Node):
    def __init__(self):
        super().__init__("speaker_node")
        self.backend = self.declare_parameter("backend", "auto").value
        self.mac_url = self.declare_parameter("mac_url", "http://192.168.64.1:8200").value.rstrip("/")
        self.espeak = shutil.which("espeak-ng") or shutil.which("espeak") or shutil.which("spd-say")
        self.lines: queue.Queue = queue.Queue()
        self.create_subscription(String, "/swag/say", self.on_say, 50)
        threading.Thread(target=self.worker, daemon=True).start()
        self.get_logger().info(f"Speaker: backend={self.backend}, espeak={self.espeak or 'not installed'}")

    def on_say(self, msg: String) -> None:
        try:
            text = json.loads(msg.data).get("text", "")
        except json.JSONDecodeError:
            text = msg.data
        if text.strip():
            self.lines.put(text.strip())

    def worker(self) -> None:
        order = {"auto": ["mac", "espeak", "print"], "mac": ["mac", "print"],
                 "espeak": ["espeak", "print"], "print": ["print"]}.get(self.backend, ["print"])
        while rclpy.ok():
            text = self.lines.get()
            for backend in order:
                if getattr(self, f"say_{backend}")(text):
                    break

    def say_mac(self, text: str) -> bool:
        req = urllib.request.Request(self.mac_url + "/say", data=json.dumps({"text": text}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:      # returns when the Mac finished speaking
                r.read()
            return True
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            self.get_logger().warn(f"Mac speech service unreachable ({e}); falling back",
                                   throttle_duration_sec=10.0)
            return False

    def say_espeak(self, text: str) -> bool:
        if not self.espeak:
            return False
        args = [self.espeak, "-w"] if self.espeak.endswith("spd-say") else [self.espeak, "-s", "165"]
        return subprocess.run(args + [text], capture_output=True).returncode == 0

    def say_print(self, text: str) -> bool:
        self.get_logger().info(f"(would say) {text}")
        return True


def main():
    rclpy.init()
    node = SpeakerNode()
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

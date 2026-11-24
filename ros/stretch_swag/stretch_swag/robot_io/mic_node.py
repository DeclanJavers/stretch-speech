"""The robot's microphone -> /swag/heard. Records on the robot, transcribes on the Mac.

`arecord` (ALSA, no Python audio packages needed) reads 16 kHz mono from the robot's mic; a simple
energy detector cuts out each utterance (it learns the room's noise level); the audio is POSTed to
the Mac's speech service (/transcribe, Whisper) and the text is published on /swag/heard.
The mic is ignored while the robot talks (/swag/speaking from speaker_node) and `echo_s` after.

    ros2 run stretch_swag mic_node --ros-args -p mac_url:=http://<mac>:8200 -p device:=default
    arecord -l                         # list the robot's mics; device:=plughw:CARD=ArrayUAC10,DEV=0
"""

import json
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String

RATE, CHUNK = 16000, 512                       # 32 ms chunks


class MicNode(Node):
    def __init__(self):
        super().__init__("mic_node")
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.mac_url = p("mac_url", "http://192.168.64.1:8200").rstrip("/")
        self.device = p("device", "default")
        self.ratio = float(p("start_ratio", 3.0))         # speech = this many times louder than the noise
        self.min_level = float(p("min_level", 300.0))     # ...and at least this loud (int16 RMS)
        self.end_silence_s = float(p("end_silence_s", 0.8))
        self.max_s = float(p("max_utterance_s", 12.0))
        self.echo_s = float(p("echo_s", 0.6))
        self.pub = self.create_publisher(String, "/swag/heard", 10)
        self.level_pub = self.create_publisher(String, "/swag/mic_level", 10)
        self.speaking, self.quiet_since = False, 0.0
        self.create_subscription(Bool, "/swag/speaking", self.on_speaking, 10)
        if not shutil.which("arecord"):
            self.get_logger().error("arecord not found: sudo apt install alsa-utils")
            return
        threading.Thread(target=self.loop, daemon=True).start()

    def on_speaking(self, msg: Bool) -> None:
        self.speaking = msg.data
        if not msg.data:
            self.quiet_since = time.time()

    def muted(self) -> bool:
        return self.speaking or time.time() - self.quiet_since < self.echo_s

    def loop(self) -> None:
        cmd = ["arecord", "-q", "-D", self.device, "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw"]
        while rclpy.ok():
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.get_logger().info(f"listening on '{self.device}' -> {self.mac_url}/transcribe")
            try:
                self.listen(proc)
            finally:
                proc.kill()
            err = proc.stderr.read().decode(errors="replace").strip()
            self.get_logger().error(f"arecord stopped ({err or 'no error'}); restarting in 2 s")
            time.sleep(2.0)

    def listen(self, proc) -> None:
        noise, recorded, speaking, last_loud, started, last_report = 200.0, [], False, 0.0, 0.0, 0.0
        preroll: list[bytes] = []
        while rclpy.ok():
            data = proc.stdout.read(CHUNK * 2)
            if len(data) < CHUNK * 2:
                return
            now = time.time()
            if self.muted():
                recorded, speaking, preroll = [], False, []
                continue
            level = float(np.sqrt(np.mean(np.frombuffer(data, np.int16).astype(np.float32) ** 2)))
            loud = level > max(self.min_level, noise * self.ratio)
            if now - last_report > 1.0:
                self.level_pub.publish(String(data=f"level {level:.0f} noise {noise:.0f}"))
                last_report = now
            if not speaking:
                noise = 0.97 * noise + 0.03 * min(level, noise * 2)      # follow the room, not the voices
                preroll = (preroll + [data])[-10:]                       # 0.3 s before the start
                if loud:
                    speaking, started, last_loud, recorded = True, now, now, preroll[:]
                continue
            recorded.append(data)
            if loud:
                last_loud = now
            if now - last_loud > self.end_silence_s or now - started > self.max_s:
                speaking = False
                if last_loud - started > 0.25:                           # not just a click
                    audio = b"".join(recorded)
                    threading.Thread(target=self.transcribe, args=(audio,), daemon=True).start()
                recorded, preroll = [], []

    def transcribe(self, audio: bytes) -> None:
        req = urllib.request.Request(self.mac_url + "/transcribe", data=audio, method="POST",
                                     headers={"Content-Type": "application/octet-stream"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                text = json.loads(r.read()).get("text", "").strip()
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as e:
            self.get_logger().warn(f"transcription failed: {e}", throttle_duration_sec=5.0)
            return
        self.get_logger().info(f"heard ({len(audio) / 2 / RATE:.1f} s): {text!r}")
        if text and not self.muted():
            self.pub.publish(String(data=json.dumps({"text": text})))


def main():
    rclpy.init()
    node = MicNode()
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

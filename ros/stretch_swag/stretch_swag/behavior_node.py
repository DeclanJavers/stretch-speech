"""The swag robot's state machine (the source of truth for what Stretch is doing).

    IDLE -> TALKING -> (asked for an item) -> FETCHING -> PICKING_UP -> RETURNING -> HANDOVER -> IDLE
                    -> (declined / unclear / no response)                                     -> IDLE

TALKING hands the whole conversation to the language server on the Mac and polls
for the outcome. For tonight's demo, fetching is "turn toward the item (pen +90,
plushie 180, t-shirt -90 deg), pause, turn back"; picking and handover are mocked.

    ros2 run stretch_swag behavior
    ros2 run stretch_swag behavior --ros-args -p fetch_distance_m:=2.0 -p language_server_url:=http://192.168.64.1:8090

Watch it:  ros2 topic echo /swag/state
"""

import json
import math
import pathlib
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

from .language_client import LanguageClient, LanguageServerError
from .motion import BaseDriver

SPOKEN_ITEM = {"pen": "pen", "shirt": "t-shirt", "plushie": "plushie"}
# Where each item "is" for the demo: turn this far (positive = counter-clockwise), then back.
ITEM_TURN_RAD = {"pen": math.pi / 2, "plushie": math.pi, "shirt": -math.pi / 2}


class SwagBehavior(Node):
    def __init__(self):
        super().__init__("swag_behavior")
        p = self.declare_parameter
        self.server_url = p("language_server_url", "http://192.168.64.1:8090").value
        self.items = list(p("items", ["pen", "shirt", "plushie"]).value)
        self.cmd_vel_topic = p("cmd_vel_topic", "/stretch/cmd_vel").value
        self.odom_topic = p("odom_topic", "/odom").value
        self.fetch_distance_m = float(p("fetch_distance_m", 1.524).value)   # 5 ft each way
        # Commanded speeds. The VM's sim runs at ~1/3 real time, so on screen these look ~3x slower.
        self.drive_speed = float(p("drive_speed", 0.3).value)               # m/s
        self.turn_speed = float(p("turn_speed", 1.0).value)                 # rad/s
        self.pickup_s = float(p("pickup_s", 3.0).value)
        self.idle_s = float(p("idle_s", 3.0).value)                   # pause before greeting the next person
        self.poll_s = float(p("poll_s", 0.5).value)
        self.conversation_timeout_s = float(p("conversation_timeout_s", 300.0).value)
        log_dir = pathlib.Path(p("log_dir", str(pathlib.Path.home() / ".ros" / "stretch_swag")).value)

        self.client = LanguageClient(self.server_url)
        self.driver = BaseDriver(self, self.cmd_vel_topic, self.odom_topic)
        self.state_pub = self.create_publisher(String, "/swag/state", 10)
        self.shutdown = threading.Event()
        self.state = "IDLE"
        self.events: list[dict] = []   # last few events, for the future LLM state block

        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"events-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        self.get_logger().info(f"Logging events to {self.log_path}")

        self.create_timer(1.0, self.publish_state)   # heartbeat so `topic echo` always shows something
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    # ---- bookkeeping -----------------------------------------------------------
    def log_event(self, kind: str, **data) -> None:
        event = {"t": time.time(), "state": self.state, "kind": kind, **data}
        self.events = (self.events + [event])[-10:]
        with self.log_path.open("a") as f:
            f.write(json.dumps(event) + "\n")

    def set_state(self, state: str, **data) -> None:
        self.get_logger().info(f"{self.state} -> {state}" + (f"  {data}" if data else ""))
        self.log_event("transition", to=state, **data)
        self.state = state
        self.publish_state()

    def publish_state(self) -> None:
        self.state_pub.publish(String(data=self.state))

    def wait(self, seconds: float) -> bool:
        """Sleep, but wake up on shutdown. Returns False if shutting down."""
        return not self.shutdown.wait(seconds)

    def say(self, text: str) -> None:
        try:
            self.client.say(text)
        except LanguageServerError as e:
            self.get_logger().warn(f"say failed: {e}")
        self.log_event("say", text=text)

    # ---- states ----------------------------------------------------------------
    def run(self) -> None:
        try:
            self.wait_for_server()
            while not self.shutdown.is_set():
                self.set_state("IDLE")
                if not self.wait(self.idle_s):
                    break
                result = self.talk()
                if result and result.get("outcome") == "item" and result.get("item"):
                    self.fetch(result["item"], result.get("name"))
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"behavior loop crashed: {e!r}")
            self.log_event("crash", error=repr(e))
        finally:
            self.driver.stop()

    def wait_for_server(self) -> None:
        while not self.shutdown.is_set():
            try:
                h = self.client.health()
                self.get_logger().info(f"Language server OK: model={h.get('model')} items={h.get('items')}")
                return
            except LanguageServerError as e:
                self.get_logger().warn(f"Waiting for language server: {e}")
                self.wait(3.0)

    def talk(self) -> dict | None:
        self.set_state("TALKING")
        try:
            cid = self.client.start_conversation(self.items)
        except LanguageServerError as e:
            self.get_logger().warn(f"Couldn't start a conversation: {e}")
            self.log_event("error", error=str(e))
            self.wait(3.0)
            return None

        seen = 0
        deadline = time.monotonic() + self.conversation_timeout_s
        while not self.shutdown.is_set() and time.monotonic() < deadline:
            try:
                record = self.client.get_conversation(cid)
            except LanguageServerError as e:
                self.get_logger().warn(f"Polling conversation {cid}: {e}")
                self.wait(1.0)
                continue
            for ev in record["events"][seen:]:
                if ev["kind"] in ("robot", "person"):
                    self.get_logger().info(f"  [{ev['kind']}] {ev['text']}")
                    self.log_event("speech", speaker=ev["kind"], text=ev["text"])
            seen = len(record["events"])
            if record["status"] == "done":
                r = record["result"]
                self.log_event("conversation_done", outcome=r["outcome"], item=r["item"], name=r["name"])
                self.get_logger().info(f"Conversation outcome: {r['outcome']} item={r['item']} name={r['name']}")
                return r
            if record["status"] == "error":
                self.get_logger().warn(f"Conversation failed on the server: {record['error']}")
                self.log_event("error", error=record["error"])
                return None
            self.wait(self.poll_s)

        self.get_logger().warn("Conversation timed out")
        self.log_event("error", error="conversation timeout")
        return None

    def run_skills(self, steps) -> bool:
        """Run (skill_name, callable) steps in order; stop at the first failure."""
        for skill, do in steps:
            r = do()
            self.log_event("skill", skill=skill, **r)
            if not r["success"]:
                self.get_logger().warn(f"{skill} failed: {r['reason']}")
                return False
        return True

    def fetch(self, item: str, name: str | None) -> None:
        # Demo: each item "lives" in a different direction, so the robot visibly reacts to the
        # answer. Turn toward it, pause (mock pickup), turn back to face the visitor.
        angle = ITEM_TURN_RAD.get(item, math.pi)
        self.set_state("FETCHING", item=item, turn_deg=round(math.degrees(angle)))
        if not self.run_skills([("turn_to_item", lambda: self.driver.turn(angle, self.turn_speed))]):
            return

        self.set_state("PICKING_UP", item=item)       # mocked: just wait
        if not self.wait(self.pickup_s):
            return

        self.set_state("RETURNING", item=item)
        if not self.run_skills([("turn_to_person", lambda: self.driver.turn(-angle, self.turn_speed))]):
            return

        self.set_state("HANDOVER", item=item)          # mocked: just say it
        who = f", {name}" if name else ""
        self.say(f"Here you go{who}! Enjoy your {SPOKEN_ITEM.get(item, item)}.")

    def stop(self) -> None:
        self.shutdown.set()
        self.driver.cancel.set()
        self.thread.join(timeout=2.0)
        self.driver.stop()


def main() -> None:
    # Handle Ctrl+C ourselves so the context is still alive to send a final zero Twist.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = SwagBehavior()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Stopping: sending zero Twist")
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

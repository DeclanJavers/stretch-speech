"""The swag robot: state machine, conversation, perception and skills in one ROS 2 node.

See MASTER-PLAN.md for the state diagram and what triggers each transition.

    ros2 run stretch_swag robot --ros-args -p llm_url:=http://192.168.64.1:8081/v1

Topics
    in   /swag/heard   std_msgs/String JSON {"text": "a few words"}         streaming STT (or a cheat)
    in   /swag/faces   std_msgs/String JSON [face, ...]                      face cutouts with position
         face = {"embedding" | "image_jpeg_b64", "x", "y", "distance", "bearing"}
    in   /swag/obstacles std_msgs/String JSON [[x, y], ...]                  people positions (lidar stand-in)
    in   /odom         nav_msgs/Odometry
    out  /swag/say     std_msgs/String JSON {"text", "person_id", "duration_s", "final"}   to TTS
    out  /swag/state   std_msgs/String                                       current state name
    out  /swag/status  std_msgs/String JSON                                  state, target, held item, stock
    out  /stretch/cmd_vel geometry_msgs/Twist
"""

from __future__ import annotations

import json
import math
import pathlib
import random
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from .core.conversation import Conversation
from .core.events import EventLog
from .core.face_client import FaceClient, FaceServiceError
from .core.llm import LLMClient, LLMError, Reply
from .core.prompt import GREETINGS, ITEM_WORDS
from .core.registry import Registry
from .core.states import InvalidTransition, RobotState, State, Trigger, next_state
from .core.geometry import scan_obstacle
from .motion import BaseDriver, wrap


def parse_pairs(text: str) -> dict[str, int]:
    return {k.strip(): int(v) for k, v in (p.split(":") for p in text.split(",") if p.strip())}


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",")]


class TopicSpeaker:
    """Publishes robot speech to /swag/say. A TTS node (or the cheat console) plays it.
    The duration is an estimate so the mic stays muted while the speech would play."""

    def __init__(self, node: Node, words_per_s: float = 2.7):
        self.node = node
        self.words_per_s = words_per_s
        self.pub = node.create_publisher(String, "/swag/say", 10)

    def say(self, text: str, person_id: str | None) -> float:
        duration = round(0.3 + len(text.split()) / self.words_per_s, 2)
        self.pub.publish(String(data=json.dumps({"text": text, "person_id": person_id,
                                                 "duration_s": duration, "final": False})))
        self.node.get_logger().info(f"\033[96mROBOT>\033[0m {text}")
        return duration

    def done(self, person_id: str | None) -> None:
        self.pub.publish(String(data=json.dumps({"text": "", "person_id": person_id,
                                                 "duration_s": 0.0, "final": True})))


class SwagRobot(Node):
    def __init__(self):
        super().__init__("swag_robot")
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.llm = LLMClient(p("llm_url", "http://192.168.64.1:8081/v1"), p("llm_model", "lfm3-1.7b"))
        self.faces = FaceClient(p("face_url", "http://192.168.64.1:8100"))
        stock = parse_pairs(p("stock", "pen:5,shirt:5,plushie:5"))
        self.table = parse_floats(p("table", "0.0,0.0"))
        self.arena = parse_floats(p("arena", "1.0,-3.0,6.0,3.0"))           # xmin, ymin, xmax, ymax
        self.approach_range = float(p("approach_range", 2.5))
        self.standoff = float(p("standoff", 1.0))                            # stop this far from a person
        self.safety_distance = float(p("safety_distance", 0.6))
        self.tick_s = float(p("tick_s", 0.5))
        self.no_response_s = float(p("no_response_s", 20.0))
        self.still_there_s = float(p("still_there_s", 8.0))
        self.max_conversation_s = float(p("max_conversation_s", 180.0))
        self.search_timeout_s = float(p("search_timeout_s", 30.0))
        self.recall_s = float(p("recall_s", 10.0))
        self.no_new_people_s = float(p("no_new_people_s", 300.0))
        self.pickup_s = float(p("pickup_s", 4.0))
        self.pickup_success = float(p("pickup_success", 1.0))
        self.pickup_attempts = int(p("pickup_attempts", 3))
        self.max_fetch_failures = int(p("max_fetch_failures", 2))   # per person, then give up politely
        self.handover_s = float(p("handover_s", 3.0))
        self.giving_chat_s = float(p("giving_chat_s", 6.0))
        self.rng = random.Random(int(p("seed", 0)) or None)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = pathlib.Path(p("log_dir", str(pathlib.Path.home() / ".ros" / "stretch_swag"))) / f"run-{stamp}"
        self.session = stamp

        self.robot = RobotState(state=State.LOOKING, since=time.time(), stock=stock)
        self.registry = Registry()
        self.log = EventLog(run_dir / "events.jsonl")
        self.speaker = TopicSpeaker(self)
        self.turn_note: str | None = None        # extra line for the LLM's system prompt, set per state
        self.turn_tools: list[str] = []           # tools the LLM may call right now
        self.conv = Conversation(self.llm, self.log, self.registry,
                                 lambda: (self.robot, self.turn_note, self.turn_tools), self.speaker,
                                 llm_log_path=run_dir / "llm.jsonl",
                                 mute_while_speaking=bool(p("mute_while_speaking", True)))
        # Real robot: safety_source "scan" (the lidar). Simulation: "obstacles" (sim_people's positions).
        self.safety_source = p("safety_source", "obstacles")
        self.scan_yaw_offset = float(p("scan_yaw_offset", 0.0))      # robot-forward angle in the lidar frame
        self.wander = bool(p("wander", True))                        # False: only spin in place while looking
        self.motion_enabled = bool(p("motion_enabled", True))        # False: skills "succeed" without moving
        self.driver = BaseDriver(self, p("cmd_vel_topic", "/stretch/cmd_vel"), p("odom_topic", "/odom"),
                                 linear_speed=float(p("linear_speed", 0.3)),
                                 angular_speed=float(p("angular_speed", 0.8)), safety=self.safety_check,
                                 dry_run=not self.motion_enabled)
        self.get_logger().info(f"Logging to {run_dir}")

        self.shutdown = threading.Event()
        self.flags_lock = threading.Lock()   # guards state changes vs. perception flags below
        self.detections: tuple[float, list[dict]] = (0.0, [])
        self.obstacles: tuple[float, list[list[float]]] = (0.0, [])
        self.scan: tuple[float, LaserScan | None] = (0.0, None)
        self.candidate: tuple[dict, str | None] | None = None   # (face, known pid) found while looking
        self.found: dict | None = None                          # target's face found while searching
        self.last_view_t = 0.0
        self.last_new_person_t = time.time()
        self._warned: dict[str, float] = {}

        self.create_subscription(String, "/swag/heard", self.on_heard, 50)
        self.create_subscription(String, "/swag/faces", self.on_faces, 10)
        self.create_subscription(String, p("obstacles_topic", "/swag/obstacles"), self.on_obstacles, 10)
        self.create_subscription(LaserScan, p("scan_topic", "/scan_filtered"),
                                 lambda msg: setattr(self, "scan", (time.time(), msg)), 10)
        self.state_pub = self.create_publisher(String, "/swag/state", 10)
        self.status_pub = self.create_publisher(String, "/swag/status", 10)
        self.create_timer(0.5, self.publish_status)

        self.threads = [threading.Thread(target=self.run, daemon=True),
                        threading.Thread(target=self.perception_loop, daemon=True)]
        for t in self.threads:
            t.start()

    # ---- helpers -----------------------------------------------------------------
    def warn_once(self, key: str, text: str, every_s: float = 10.0) -> None:
        if time.time() - self._warned.get(key, 0) > every_s:
            self._warned[key] = time.time()
            self.get_logger().warn(text)

    def log_once(self, key: str, text: str, every_s: float) -> None:
        if time.time() - self._warned.get(key, 0) > every_s:
            self._warned[key] = time.time()
            self.get_logger().info(text)

    def sleep(self, seconds: float) -> bool:
        """Returns False if shutting down."""
        return not self.shutdown.wait(seconds)

    def target(self):
        return self.registry.get(self.robot.target)

    def publish_status(self) -> None:
        person = self.target()
        self.state_pub.publish(String(data=self.robot.state.value))
        self.status_pub.publish(String(data=json.dumps({
            "state": self.robot.state.value, "target": self.robot.target,
            "target_name": person.name if person else None,
            "target_position": person.position if person else None,
            "held_item": self.robot.held_item, "stock": self.robot.stock,
            "speaking": self.conv.speaking})))

    def fire(self, trigger: Trigger) -> bool:
        if self.shutdown.is_set():
            return False
        old = self.robot.state
        try:
            new = next_state(old, trigger)
        except InvalidTransition as e:
            self.get_logger().error(str(e))
            self.log.append("note", self.robot.target, text=str(e))
            return False
        person = self.target()
        item = self.robot.held_item or (person.item if person else None)
        with self.flags_lock:
            self.robot.state, self.robot.since = new, time.time()
            self.candidate = self.found = None          # perception flags belong to one state
            self.driver.interrupt.clear()
        self.log.append("state", self.robot.target, state=new.value, prev=old.value,
                        trigger=trigger.value, item=item)
        self.get_logger().info(f"\033[93m{old.value} --{trigger.value}--> {new.value}\033[0m"
                               + (f"  [{person.id} {person.name or ''}]" if person else ""))
        self.publish_status()
        return True

    def face_of(self, det: dict) -> dict:
        return {k: det[k] for k in ("embedding", "image_jpeg_b64") if k in det}

    def turn_toward(self, x: float, y: float) -> None:
        pose = self.driver.pose()
        if pose:
            self.driver.turn_to(math.atan2(y - pose[1], x - pose[0]))

    def wait_speech_done(self, limit_s: float = 8.0) -> None:
        end = time.time() + limit_s
        while self.conv.speaking and time.time() < end and self.sleep(0.1):
            pass

    # ---- inputs ------------------------------------------------------------------
    def on_heard(self, msg: String) -> None:
        try:
            text = json.loads(msg.data).get("text", "")
        except json.JSONDecodeError:
            text = msg.data
        self.conv.heard(text)

    def on_faces(self, msg: String) -> None:
        try:
            self.detections = (time.time(), json.loads(msg.data))
        except json.JSONDecodeError:
            pass

    def on_obstacles(self, msg: String) -> None:
        try:
            self.obstacles = (time.time(), json.loads(msg.data))
        except json.JSONDecodeError:
            pass

    def safety_check(self) -> str | None:
        """Something within safety_distance in front of the base? Called every step while driving."""
        if self.safety_source == "scan":
            t, scan = self.scan
            if scan is None or time.time() - t > 0.5:
                return "no recent lidar scan"              # can't see: don't drive
            d = scan_obstacle(scan.ranges, scan.angle_min, scan.angle_increment, self.safety_distance,
                              yaw_offset=self.scan_yaw_offset, min_range=max(scan.range_min, 0.15))
            return f"obstacle {d:.2f} m ahead (lidar)" if d is not None else None
        t, points = self.obstacles                          # simulation: sim_people's positions
        pose = self.driver.pose()
        if not pose or time.time() - t > 1.0:
            return None
        for x, y in points:
            d = math.hypot(x - pose[0], y - pose[1])
            if d < self.safety_distance and abs(wrap(math.atan2(y - pose[1], x - pose[0]) - pose[2])) < 1.2:
                return f"person {d:.2f} m ahead"
        return None

    # ---- perception worker: identify / verify / register faces --------------------
    def set_flag(self, state: State, name: str, value) -> None:
        """Set candidate/found and interrupt the current move, unless the state changed meanwhile."""
        with self.flags_lock:
            if self.robot.state == state and getattr(self, name) is None:
                setattr(self, name, value)
                self.driver.interrupt.set()

    def perception_loop(self) -> None:
        while not self.shutdown.is_set():
            self.sleep(0.25)
            t, dets = self.detections
            if time.time() - t > 0.6 or not dets:
                continue
            state = self.robot.state
            try:
                if state in (State.LOOKING, State.DONE) and self.candidate is None and self.robot.items_in_stock():
                    for det in sorted(dets, key=lambda d: d["distance"]):
                        if det["distance"] > self.approach_range:
                            continue
                        pid, _ = self.faces.identify(self.face_of(det))
                        person = self.registry.get(pid)
                        if pid is None or (person and person.outcome == "missed"):
                            self.set_flag(state, "candidate", (det, pid))
                            break
                elif state == State.SEARCHING and self.found is None and self.robot.target:
                    for det in dets:
                        if det["distance"] <= self.approach_range + 1.0:
                            match, score = self.faces.verify(self.robot.target, self.face_of(det))
                            self.log_once("verify", f"verify {self.robot.target}: score {score:.2f} "
                                          f"at {det['distance']:.1f} m -> {'match' if match else 'no match'}", 2.0)
                            if match:
                                self.set_flag(state, "found", det)
                                break
                elif state in (State.TALKING, State.GIVING) and self.robot.target:
                    # Add views of the person we're talking to (the closest face, straight ahead).
                    person = self.target()
                    close = [d for d in dets if d["distance"] < 1.8 and abs(d.get("bearing", 0)) < 0.5]
                    if person and close and person.face_views < 6 and time.time() - self.last_view_t > 1.5:
                        det = min(close, key=lambda d: d["distance"])
                        person.face_views = self.faces.register(person.id, self.face_of(det))
                        self.last_view_t = time.time()
            except FaceServiceError as e:
                self.warn_once("faces", f"face service: {e}")

    # ---- the state machine ---------------------------------------------------------
    def run(self) -> None:
        try:
            if self.motion_enabled and self.driver.wait_for_pose(30.0) is None:
                self.get_logger().error("No odometry on /odom after 30 s. Is the sim (or fake_base) running?")
            for name, check in (("LLM", self.llm.health), ("face service", self.faces.health)):
                try:
                    self.get_logger().info(f"{name} OK: {check()}")
                except (LLMError, FaceServiceError) as e:
                    self.get_logger().warn(f"{name} not reachable yet: {e}")
            try:
                self.faces.reset(self.session)
            except FaceServiceError:
                pass
            self.log.append("state", None, state=self.robot.state.value, prev=None, trigger="start")
            handlers = {
                State.LOOKING: self.do_looking, State.TALKING: self.do_talking,
                State.TO_TABLE: self.do_to_table, State.PICKING_UP: self.do_picking_up,
                State.TO_PERSON: self.do_to_person, State.SEARCHING: self.do_searching,
                State.GIVING: self.do_giving, State.RETURNING_ITEM: self.do_returning_item,
                State.DONE: self.do_done,
            }
            while not self.shutdown.is_set():
                handlers[self.robot.state]()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"state machine crashed: {e!r}")
            self.log.append("note", None, text=f"crash: {e!r}")
            raise
        finally:
            self.driver.stop()

    def take_candidate(self) -> None:
        """A face worth talking to was found: register them (if new) and turn to face them."""
        det, known = self.candidate              # stays set until fire() clears it
        self.driver.interrupt.clear()
        if known:
            person = self.registry.get(known)
        else:
            person = self.registry.new(time.time())
            try:
                person.face_views = self.faces.register(person.id, self.face_of(det))
            except FaceServiceError as e:
                self.warn_once("faces", f"register failed: {e}")
            self.log.append("person", person.id, change="registered")
        person.position = (det["x"], det["y"])
        self.robot.target = person.id
        self.last_new_person_t = time.time()
        self.turn_toward(det["x"], det["y"])
        self.fire(Trigger.NEW_PERSON)

    def do_looking(self) -> None:
        if not self.robot.items_in_stock():
            self.fire(Trigger.OUT_OF_STOCK)
            return
        self.candidate = None
        self.driver.interrupt.clear()
        while not self.shutdown.is_set():
            for _ in range(8):                               # look around: 8 x 45 deg
                self.driver.turn(math.pi / 4)
                if self.candidate or not self.sleep(0.4) or self.candidate:
                    break
            if self.candidate:
                return self.take_candidate()
            if time.time() - self.last_new_person_t > self.no_new_people_s:
                self.fire(Trigger.NO_NEW_PEOPLE)
                return
            if not self.wander:                               # keep spinning in place
                continue
            x0, y0, x1, y1 = self.arena                       # wander to a random spot
            self.driver.go_to(self.rng.uniform(x0, x1), self.rng.uniform(y0, y1))
            if self.candidate:
                return self.take_candidate()

    def handle_reply(self, reply: Reply | None) -> bool:
        """Turn the LLM's tool call (if any) into a transition. Returns True if the state changed."""
        if not reply or reply.tool not in ("fetch_item", "end_conversation"):
            return False
        person = self.target()
        self.wait_speech_done()
        self.conv.stop()
        if reply.tool == "fetch_item":                   # item already checked against stock by the engine
            person.item, person.outcome = reply.args["item"], "waiting"
            self.log.append("person", person.id, change="outcome", outcome="waiting", item=person.item)
            return self.fire(Trigger.REQUEST_ITEM)
        declined = reply.args.get("reason", "declined") == "declined"
        person.outcome = "declined" if declined else "ended"
        self.log.append("person", person.id, change="outcome", outcome=person.outcome)
        self.robot.target = None
        return self.fire(Trigger.DECLINE if declined else Trigger.END)

    def do_talking(self) -> None:
        person = self.target()
        item_word = ITEM_WORDS.get(person.item, person.item)
        self.turn_tools = ["say_to_visitor", "fetch_item", "wait_for_visitor", "end_conversation"]
        self.turn_note = None
        self.conv.start(person.id)
        if person.outcome == "waiting" and self.robot.held_item is None:
            # Back from a failed fetch (the fetch_item result says what went wrong).
            if person.fetch_failures >= self.max_fetch_failures:
                self.turn_note, self.turn_tools = (f"Picking things up keeps failing, so you can't give {person.label()} "
                                                   f"anything right now. Apologize and say goodbye."), ["say_to_visitor"]
                self.conv.respond()
                self.wait_speech_done()
                self.conv.stop()
                person.outcome = "ended"
                self.log.append("person", person.id, change="outcome", outcome="ended", reason="pickup kept failing")
                self.robot.target = None
                self.fire(Trigger.END)
                return
            self.turn_note = (f"You don't have the {item_word}: picking it up failed. Apologize to {person.label()} "
                              f"and ask if they'd like something else instead.")
            person.item = None
            first = self.conv.respond()
        elif person.outcome == "missed":
            self.turn_note = f"Ask if they still want the {item_word}."
            first = self.conv.respond()
        else:
            self.conv.say(self.rng.choice(GREETINGS))
            first = None
        resumed = person.outcome in ("waiting", "missed")
        person.outcome = "talking"
        if self.handle_reply(first):
            return
        self.turn_note = None
        started, asked_still_there = time.time(), False
        while self.robot.state == State.TALKING and self.sleep(self.tick_s):
            if self.handle_reply(self.conv.tick()):
                return
            silent = self.conv.unanswered_for()
            if silent > self.still_there_s and not asked_still_there:
                asked_still_there = True
                self.conv.say("Are you still there?")
                continue
            had_reply = any(e.kind == "heard" and e.person_id == person.id for e in self.log.snapshot())
            if silent > self.no_response_s or self.conv.failures >= 5 \
                    or time.time() - started > self.max_conversation_s:
                self.conv.stop()
                person.outcome = "ended" if had_reply or resumed else "no_response"
                self.log.append("person", person.id, change="outcome", outcome=person.outcome)
                self.robot.target = None
                self.fire(Trigger.END if had_reply or resumed else Trigger.NO_RESPONSE)
                return

    def do_to_table(self) -> None:
        for attempt in range(1, 100):
            self.driver.interrupt.clear()
            r = self.driver.go_to(*self.table)
            self.log.append("skill", self.robot.target, skill="drive to table", attempt=attempt, **r)
            if r["success"] or self.shutdown.is_set():
                break
            self.sleep(2.0)
        self.fire(Trigger.ARRIVED)

    def do_picking_up(self) -> None:
        person = self.target()
        item = person.item
        for attempt in range(1, self.pickup_attempts + 1):
            if not self.sleep(self.pickup_s):               # mocked grasp
                return
            ok = self.rng.random() < self.pickup_success and self.robot.stock.get(item, 0) > 0
            self.log.append("skill", person.id, skill="pickup", attempt=attempt, success=ok,
                            reason="" if ok else "grasp failed")
            if ok:
                self.robot.held_item = item
                self.robot.stock[item] -= 1
                self.fire(Trigger.PICKED_UP)
                return
        person.fetch_failures += 1
        self.fire(Trigger.PICKUP_GAVE_UP)

    def do_to_person(self) -> None:
        person = self.target()
        x, y = person.position
        for attempt in range(1, 4):
            self.driver.interrupt.clear()
            r = self.driver.go_to(x, y, stop_distance=self.standoff)
            self.log.append("skill", person.id, skill="drive to person", attempt=attempt, **r)
            if r["success"] or self.shutdown.is_set():
                break
        self.fire(Trigger.ARRIVED)

    def do_searching(self) -> None:
        person = self.target()
        self.found = None
        self.driver.interrupt.clear()
        self.conv.start(person.id)
        held = ITEM_WORDS.get(self.robot.held_item)
        call = (f"{person.name}? " if person.name else "Hello? ") + (f"I have your {held}!" if held else "I'm back!")
        self.conv.say(call, aside=True)
        deadline, last_call = time.time() + self.search_timeout_s, time.time()
        while self.found is None and time.time() < deadline and not self.shutdown.is_set():
            self.driver.turn(math.pi / 6)
            if self.found or not self.sleep(0.5):
                break
            if time.time() - last_call > self.recall_s:
                self.conv.say(call, aside=True)
                last_call = time.time()
        self.conv.stop()
        holding = self.robot.held_item is not None
        if self.found:
            det = self.found                         # stays set until fire() clears it
            self.driver.interrupt.clear()
            person.position = (det["x"], det["y"])
            if det["distance"] > self.standoff + 0.3:          # they moved: walk up to them first
                r = self.driver.go_to(det["x"], det["y"], stop_distance=self.standoff)
                self.log.append("skill", person.id, skill="drive to person", **r)
            self.turn_toward(det["x"], det["y"])
            self.fire(Trigger.TARGET_FOUND if holding else Trigger.TARGET_FOUND_NO_ITEM)
            return
        person.outcome = "missed"
        self.log.append("person", person.id, change="outcome", outcome="missed")
        if holding:
            self.fire(Trigger.TARGET_NOT_FOUND)                # keep target: the item is theirs
        else:
            self.robot.target = None
            self.fire(Trigger.TARGET_NOT_FOUND_NO_ITEM)

    def do_giving(self) -> None:
        person = self.target()
        item = self.robot.held_item
        self.conv.start(person.id)
        self.turn_note, self.turn_tools = f"Hand them the {ITEM_WORDS.get(item, item)}.", ["say_to_visitor"]
        self.conv.respond()
        self.turn_note, self.turn_tools = None, ["say_to_visitor", "wait_for_visitor"]
        self.sleep(self.handover_s)                           # mocked: extend arm, open gripper
        self.log.append("skill", person.id, skill="handover", success=True, reason="")
        self.robot.held_item = None
        person.outcome = "served"
        self.log.append("person", person.id, change="outcome", outcome="served")
        self.turn_note = (f"You just gave {person.label()} the {ITEM_WORDS.get(item, item)}. That's their one item, "
                          f"so don't offer anything else; just be friendly.")
        end = time.time() + self.giving_chat_s                # a few seconds for "thank you!"
        while time.time() < end and self.sleep(self.tick_s):
            self.conv.tick()
        self.wait_speech_done()
        self.conv.stop()
        self.robot.target = None
        self.fire(Trigger.ITEM_GIVEN)

    def do_returning_item(self) -> None:
        item = self.robot.held_item
        r = self.driver.go_to(*self.table)
        self.log.append("skill", self.robot.target, skill="drive to table", **r)
        self.sleep(1.0)                                       # mocked: put it back
        self.robot.stock[item] = self.robot.stock.get(item, 0) + 1
        self.robot.held_item = None
        self.log.append("skill", self.robot.target, skill="put back", success=True, reason="")
        self.robot.target = None
        self.fire(Trigger.ARRIVED)

    def do_done(self) -> None:
        self.driver.go_to(*self.table)
        self.candidate = None
        while not self.shutdown.is_set():
            if self.candidate:
                return self.take_candidate()
            self.sleep(0.5)

    def stop(self) -> None:
        self.shutdown.set()
        self.driver.cancel.set()
        self.driver.stop()                                    # zero Twist first, before anything can hang
        try:
            for t in self.threads:
                t.join(timeout=1.0)
        except KeyboardInterrupt:
            pass
        self.driver.stop()


def main() -> None:
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = SwagRobot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Stopping: sending zero Twist")
        try:
            node.stop()
        except KeyboardInterrupt:                 # a second Ctrl+C: still try one last zero Twist
            node.driver.stop()
        try:
            node.destroy_node()
            rclpy.shutdown()
        except (KeyboardInterrupt, RuntimeError):
            pass


if __name__ == "__main__":
    main()

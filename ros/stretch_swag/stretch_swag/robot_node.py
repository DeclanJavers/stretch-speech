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
import signal
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor
from tf2_ros import Buffer, TransformException, TransformListener
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from .core.conversation import Conversation
from .core.events import EventLog
from .core.face_client import FaceClient, FaceServiceError
from .core.llm import LLMClient, LLMError, Reply
from .core.prompt import GREETINGS, ITEM_WORDS
from .core.registry import Registry
from .core.states import InvalidTransition, RobotState, State, Trigger, next_state
from .core.geometry import arena_reach, free_distance_points, free_distance_scan, scan_obstacle, transform_point
from .arm import Arm
from .motion import BaseDriver, wrap, yaw_of
from .pick import READY, Picker


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
        # Robot-forward angle in the lidar's frame. "tf" (default): read the lidar's mounting from
        # TF (Stretch's lidar is mounted rotated 180 deg). A number overrides it.
        yaw = self.declare_parameter("scan_yaw_offset", "tf", ParameterDescriptor(dynamic_typing=True)).value
        self.scan_yaw_fixed = None if str(yaw).strip().lower() == "tf" else float(yaw)
        self.scan_yaw_cache: dict[str, float] = {}
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.wander = bool(p("wander", True))                        # False: only spin in place while looking
        self.wander_min_m = float(p("wander_min_m", 0.8))            # each wander leg: this far ...
        self.wander_max_m = float(p("wander_max_m", 2.5))            # ... to this far
        self.look_step = math.radians(float(p("look_step_deg", 35.0)))   # head camera sees about +-21 deg
        self.look_pause_s = float(p("look_pause_s", 1.0))            # stand still and look after each move
        self.close_enough_m = float(p("close_enough_m", 0.8))        # blocked this close to the table = arrived
        self.see_range = float(p("see_range", 6.0))                  # notice new faces up to this far ...
        self.talk_range = float(p("talk_range", 1.6))                # ... walk up until this close, then talk
        self.talk_distance = float(p("talk_distance", 1.1))          # stop this far from them when walking up
        self.ignored: list[tuple[float, float, float]] = []          # (x, y, until): spots to skip while looking
        self.motion_enabled = bool(p("motion_enabled", True))        # False: skills "succeed" without moving
        self.driver = BaseDriver(self, p("cmd_vel_topic", "/stretch/cmd_vel"), p("odom_topic", "/odom"),
                                 linear_speed=float(p("linear_speed", 0.3)),
                                 angular_speed=float(p("angular_speed", 0.8)), safety=self.safety_check,
                                 dry_run=not self.motion_enabled)
        self.shutdown = threading.Event()
        self.driver.docking = False                                   # True while lining up at the table
        self.docking_distance = float(p("docking_safety_distance", 0.15))
        # Picking: "mock" (wait, configurable success rate) or "arm" (goodie bags with ArUco-tagged labels)
        self.pick_mode = p("pick_mode", "mock")
        self.arm = Arm(self, dry_run=not self.motion_enabled, stop=self.shutdown)
        self.markers: dict[int, list[tuple[float, str, tuple[float, float, float]]]] = {}
        self.create_subscription(MarkerArray, "/aruco/marker_array", self.on_markers, 10)
        self.picker = Picker(self, self.driver, self.arm, self.marker_in_base, {
            "tags": parse_pairs(p("bag_tags", "pen:200,shirt:201,plushie:202")),
            "table_dir": math.radians(float(p("table_dir_deg", 180.0))),
            "handle_above_tag_m": float(p("handle_above_tag_m", 0.145)),
            "bag_depth_m": float(p("bag_depth_m", 0.10)),
        })
        self.get_logger().info(f"Logging to {run_dir}")

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
                                 lambda msg: setattr(self, "scan", (time.time(), msg)), qos_profile_sensor_data)
        self.state_pub = self.create_publisher(String, "/swag/state", 10)
        self.status_pub = self.create_publisher(String, "/swag/status", 10)
        self.create_timer(0.5, self.publish_status)
        self.registry_pub = self.create_publisher(String, "/swag/registry", 2)
        self.create_timer(1.0, self.publish_registry)

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

    @staticmethod
    def remember_look(person, det: dict) -> None:
        """Keep the sharpest face picture (and the badge id) for the debug view."""
        if det.get("badge") is not None:
            person.badge = det["badge"]
        if det.get("thumb") and det.get("face_px", 0) >= person.thumb_px:
            person.thumb, person.thumb_px = det["thumb"], det.get("face_px", 0)

    def publish_registry(self) -> None:
        people = [{"id": q.id, "name": q.name, "item": q.item, "outcome": q.outcome, "badge": q.badge,
                   "thumb": q.thumb} for q in list(self.registry.people.values())]
        self.registry_pub.publish(String(data=json.dumps({
            "state": self.robot.state.value, "target": self.robot.target, "held_item": self.robot.held_item,
            "people": people})))

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

    def scan_yaw_offset(self, frame: str) -> float | None:
        """Robot-forward angle in the lidar frame, from TF (cached: the mounting doesn't change)."""
        if self.scan_yaw_fixed is not None:
            return self.scan_yaw_fixed
        if frame not in self.scan_yaw_cache:
            try:
                t = self.tf.lookup_transform("base_link", frame, Time())
            except TransformException as e:
                self.warn_once("scan_tf", f"no transform base_link -> {frame} yet: {e}")
                return None
            self.scan_yaw_cache[frame] = -yaw_of(t.transform.rotation)
            self.get_logger().info(f"lidar frame '{frame}' is rotated {math.degrees(-self.scan_yaw_cache[frame]):.0f} deg "
                                   "from the base (from TF)")
        return self.scan_yaw_cache[frame]

    def on_markers(self, msg: MarkerArray) -> None:
        now, seen = time.time(), {}
        for m in msg.markers:
            p = m.pose.position
            seen.setdefault(m.id, []).append((now, m.header.frame_id, (p.x, p.y, p.z)))
        self.markers.update(seen)

    def marker_in_base(self, tag: int, max_age_s: float):
        """Nearest fresh sighting of ArUco tag `tag`, in base_link (x forward, y left, z up)."""
        best = None
        for t, frame, point in self.markers.get(tag, []):
            if time.time() - t > max_age_s:
                continue
            try:
                tf = self.tf.lookup_transform("base_link", frame, Time())
            except TransformException:
                continue
            tr, r = tf.transform.translation, tf.transform.rotation
            b = transform_point((tr.x, tr.y, tr.z), (r.x, r.y, r.z, r.w), point)
            if best is None or math.hypot(b[0], b[1]) < math.hypot(best[0], best[1]):
                best = b
        return best

    def safety_check(self) -> str | None:
        """Something within safety_distance in front of the base? Called every step while driving."""
        stop_at = self.docking_distance if self.driver.docking else self.safety_distance
        if self.safety_source == "scan":
            t, scan = self.scan
            if scan is None or time.time() - t > 0.5:
                return "no recent lidar scan"              # can't see: don't drive
            offset = self.scan_yaw_offset(scan.header.frame_id)
            if offset is None:
                return "lidar mounting unknown (no TF yet)"     # can't tell front from back: don't drive
            d = scan_obstacle(scan.ranges, scan.angle_min, scan.angle_increment, stop_at,
                              yaw_offset=offset, min_range=max(scan.range_min, 0.15))
            return f"obstacle {d:.2f} m ahead (lidar)" if d is not None else None
        t, points = self.obstacles                          # simulation: sim_people's positions
        pose = self.driver.pose()
        if not pose or time.time() - t > 1.0:
            return None
        for x, y in points:
            d = math.hypot(x - pose[0], y - pose[1])
            if d < stop_at and abs(wrap(math.atan2(y - pose[1], x - pose[0]) - pose[2])) < 1.2:
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
                        if det["distance"] > self.see_range or self.is_ignored(det["x"], det["y"]):
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
                        self.remember_look(person, det)
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
            if self.pick_mode == "arm":                      # arm up and in, gripper down: clear of tables
                r = self.arm.move(**READY, head_pan=0.0, head_tilt=-0.1)
                self.get_logger().info(f"arm ready pose: {r['reason'] or 'ok'}")
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
        self.remember_look(person, det)
        self.robot.target = person.id
        self.last_new_person_t = time.time()
        self.turn_toward(det["x"], det["y"])
        self.fire(Trigger.NEW_PERSON)

    def do_looking(self) -> None:
        """Look for someone new: turn in place one camera-width at a time. The moment a new face
        shows up (even far away) walk straight toward it, then talk. After a full turn with
        nobody new, wander to an open spot (if allowed) and look again."""
        if not self.robot.items_in_stock():
            self.fire(Trigger.OUT_OF_STOCK)
            return
        self.candidate = None
        self.driver.interrupt.clear()
        looking_since, turned = time.time(), 0.0
        while not self.shutdown.is_set():
            if self.candidate:
                if self.approach_candidate():
                    return
                turned = 0.0
                continue
            if time.time() - looking_since > self.no_new_people_s:    # time spent looking, not fetching
                self.fire(Trigger.NO_NEW_PEOPLE)
                return
            pose = self.driver.pose()
            if self.wander and pose and turned >= 2 * math.pi - 0.01:
                heading, distance = self.pick_wander_target(pose)
                turned = 0.0
            else:
                heading, distance = (pose[2] if pose else 0.0) + self.look_step, 0.0
                turned += self.look_step
            if pose and distance:
                self.log_once("wander", f"nobody new around: turn {math.degrees(wrap(heading - pose[2])):+.0f} deg, "
                              f"drive {distance:.1f} m", 0.0)
            self.driver.turn_to(heading)
            if self.candidate:
                continue
            if distance and pose:
                p = self.driver.pose() or pose
                self.driver.go_to(p[0] + distance * math.cos(heading), p[1] + distance * math.sin(heading))
                if self.candidate:
                    continue
            self.sleep(self.look_pause_s)

    def is_ignored(self, x: float, y: float) -> bool:
        now = time.time()
        self.ignored = [z for z in self.ignored if z[2] > now]
        return any(math.hypot(x - zx, y - zy) < 0.8 for zx, zy, _ in self.ignored)

    def approach_candidate(self) -> bool:
        """Walk up to the new face, check up close that it really is someone new, then start
        the conversation (True). False: nobody there / someone we already served; keep looking."""
        det, known = self.candidate
        self.driver.interrupt.clear()
        target = (det["x"], det["y"])
        if det["distance"] > self.talk_range:
            self.get_logger().info(f"new face {det['distance']:.1f} m away: walking over")
            self.driver.go_to(*target, stop_distance=self.talk_distance)
            det, known = self.face_near(target)
            if det is None:
                t, dets = self.detections
                pose = self.driver.pose()
                seen = ", ".join(f"{d['distance']:.1f} m at ({d['x']:.1f}, {d['y']:.1f})" for d in dets) or "no faces"
                self.get_logger().info(f"nobody at ({target[0]:.1f}, {target[1]:.1f}) any more (robot at "
                                       f"({pose[0]:.1f}, {pose[1]:.1f}); sees {seen}): looking elsewhere")
                self.ignored.append((*target, time.time() + 30.0))
                self.candidate = None
                return False
            person = self.registry.get(known) if known else None
            if person and person.outcome != "missed":
                self.get_logger().info(f"up close it's {person.id} ({person.label()}), already met: looking elsewhere")
                self.ignored.append((det["x"], det["y"], time.time() + 120.0))
                self.candidate = None
                return False
            self.candidate = (det, known)
        self.take_candidate()
        return True

    def face_near(self, target, wait_s: float = 3.0):
        """The face closest to `target` within reach, identified up close: (det, person_id) or (None, None)."""
        self.turn_toward(*target)
        deadline = time.time() + wait_s
        while time.time() < deadline and not self.shutdown.is_set():
            t, dets = self.detections
            near = [d for d in dets if time.time() - t < 0.6 and d["distance"] <= self.talk_range + 0.7
                    and math.hypot(d["x"] - target[0], d["y"] - target[1]) < 1.2]
            if near:
                det = min(near, key=lambda d: d["distance"])
                try:
                    pid, _ = self.faces.identify(self.face_of(det))
                except FaceServiceError as e:
                    self.warn_once("faces", f"face service: {e}")
                    pid = None
                return det, pid
            self.sleep(0.2)
        return None, None

    def free_distance(self, pose, heading: float) -> float:
        """Open space ahead along `heading` (lidar in the real robot, sim_people's positions otherwise)."""
        if self.safety_source == "scan":
            t, scan = self.scan
            offset = self.scan_yaw_offset(scan.header.frame_id) if scan else None
            if scan is None or offset is None or time.time() - t > 1.0:
                return 0.0
            return free_distance_scan(scan.ranges, scan.angle_min, scan.angle_increment, wrap(heading - pose[2]),
                                      yaw_offset=offset, min_range=max(scan.range_min, 0.15))
        return free_distance_points(self.obstacles[1], pose[:2], heading)

    def pick_wander_target(self, pose) -> tuple[float, float]:
        """A random turn (40-180 deg either way) toward a spot that is open and inside the arena."""
        margin = self.safety_distance + 0.3
        for _ in range(24):
            heading = pose[2] + self.rng.choice((-1, 1)) * self.rng.uniform(math.radians(40), math.pi)
            reach = min(self.free_distance(pose, heading) - margin, arena_reach(pose[:2], heading, self.arena),
                        self.wander_max_m)
            if reach >= self.wander_min_m:
                return heading, self.rng.uniform(self.wander_min_m, reach)
        return pose[2] + self.look_step, 0.0                  # boxed in: just turn and look

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
            # A fixed opener (like GREETINGS): asking the LLM to open, with only its own earlier
            # call-outs in the chat and nothing new from the visitor, made it say goodbye.
            hello = f"{person.name}, there you are!" if person.name else "Hi again!"
            self.conv.say(f"{hello} Do you still want the {item_word}?")
            first = None
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
            pose = self.driver.pose()
            if pose and math.hypot(pose[0] - self.table[0], pose[1] - self.table[1]) < self.close_enough_m:
                break                                    # blocked or stuck, but we're at the table
            self.sleep(2.0)
        self.fire(Trigger.ARRIVED)

    def do_picking_up(self) -> None:
        person = self.target()
        item = person.item
        for attempt in range(1, self.pickup_attempts + 1):
            if self.pick_mode == "arm":
                r = self.picker.pick(item)
                ok, reason = r["success"] and self.robot.stock.get(item, 0) > 0, r["reason"]
            else:
                if not self.sleep(self.pickup_s):               # mocked grasp
                    return
                ok = self.rng.random() < self.pickup_success and self.robot.stock.get(item, 0) > 0
                reason = "" if ok else "grasp failed"
            self.log.append("skill", person.id, skill="pickup", attempt=attempt, success=ok, reason=reason)
            if self.shutdown.is_set():
                return
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
        if self.pick_mode == "arm":
            r = self.picker.hand_over(person.position or self.driver.pose()[:2], self.handover_s)
        else:
            self.sleep(self.handover_s)                       # mocked: extend arm, open gripper
            r = {"success": True, "reason": ""}
        self.log.append("skill", person.id, skill="handover", **r)
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
        if self.pick_mode == "arm":
            r = self.picker.put_back(item)
        else:
            self.sleep(1.0)                                   # mocked: put it back
            r = {"success": True, "reason": ""}
        self.robot.stock[item] = self.robot.stock.get(item, 0) + 1
        self.robot.held_item = None
        self.log.append("skill", self.robot.target, skill="put back", **r)
        self.robot.target = None
        self.fire(Trigger.ARRIVED)

    def do_done(self) -> None:
        self.driver.go_to(*self.table)
        self.candidate = None
        while not self.shutdown.is_set():
            if self.candidate and self.approach_candidate():
                return
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
    # Our own Ctrl+C handling (so we can send a zero Twist before ROS shuts down). Install it
    # explicitly: a process started in the background inherits SIGINT "ignored". SIGTERM too.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
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

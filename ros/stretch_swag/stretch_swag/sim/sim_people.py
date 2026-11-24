"""A simulated crowd: the perception and visitor cheat.

Each virtual person has a position, a facing direction, a name, what they want and a
persona. This node publishes what the robot's camera + face pipeline would:

    /swag/faces      faces in the camera's view and facing the robot, each with a
                     synthetic face embedding (same person -> similar vectors; side
                     views are noisier), its position and distance
    /swag/obstacles  everyone's position (stands in for the lidar, for the safety stop)

With auto_reply (default), people also answer the robot: they listen on /swag/say and
reply word by word on /swag/heard, like streaming STT would. Turn it off to type the
replies yourself with `ros2 run stretch_swag chat_console`.

Personas: normal, chatty (asks a question first, pauses mid-sentence), decliner,
silent (never answers), wanders_off (moves away after ordering).
"""

import json
import os
import math
import random
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from ..motion import wrap, yaw_of

SPOKEN = {"pen": "pen", "shirt": "t-shirt", "plushie": "plushie"}
DIM = 128
FOV = math.radians(35)          # half-angle of the head camera
FACE_RANGE = (0.4, 4.5)         # metres
FACING_TOLERANCE = math.radians(70)


class Person:
    def __init__(self, spec: dict, rng: random.Random):
        self.name = spec["name"]
        self.x, self.y = spec["pos"]
        self.wants = spec.get("wants")
        self.alt = spec.get("alt") or next(i for i in SPOKEN if i != self.wants) if self.wants else None
        self.persona = spec.get("persona", "normal")
        self.facing = rng.uniform(-math.pi, math.pi)
        self.next_turn = 0.0
        self.attend_until = 0.0          # facing the robot until then (called / talking)
        face_rng = random.Random(self.name)                                  # same name -> same face
        self.base = [face_rng.gauss(0, 1) for _ in range(DIM)]               # their "face"
        self.gave_name = self.asked = False
        self.walk_to = None
        self.moved = False


class SimPeople(Node):
    def __init__(self):
        super().__init__("sim_people")
        from ament_index_python.packages import get_package_share_directory
        default_world = f"{get_package_share_directory('stretch_swag')}/config/world.json"
        world_file = self.declare_parameter("world", default_world).value or default_world
        self.auto_reply = self.declare_parameter("auto_reply", True).value
        self.rng = random.Random(self.declare_parameter("seed", 1).value)
        self.word_s = self.declare_parameter("word_s", 0.25).value        # time per spoken word
        # perception:=false when the robot sees people itself (MuJoCo cutouts + camera): then we
        # only voice the visitors, and share their positions with MuJoCo through people_file.
        self.perception = self.declare_parameter("perception", True).value
        self.people_file = self.declare_parameter("people_file", "").value
        world = json.loads(open(world_file).read())
        self.people = [Person(s, self.rng) for s in world["people"]]
        self.robot = None                     # (x, y, yaw)
        self.status = {}
        self.pid_to_person: dict[str, Person] = {}
        self.turn_text: dict[str, str] = {}
        self.speech_end = 0.0
        self.outbox: list[tuple[float, str]] = []   # (time, words) to publish on /swag/heard

        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(String, "/swag/status", self.on_status, 10)
        self.create_subscription(String, "/swag/say", self.on_say, 50)
        self.faces_pub = self.create_publisher(String, "/swag/faces", 10)
        self.obstacles_pub = self.create_publisher(String, "/swag/obstacles", 10)
        self.heard_pub = self.create_publisher(String, "/swag/heard", 50)
        self.create_timer(0.2, self.publish_perception)
        self.create_timer(0.05, self.step)
        names = ", ".join(f"{p.name} ({p.persona}, wants {p.wants})" for p in self.people)
        self.get_logger().info(f"Simulated crowd: {names}. auto_reply={self.auto_reply}, "
                               f"perception={'faked' if self.perception else 'camera'}")

    # ---- inputs ------------------------------------------------------------------
    def on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose
        self.robot = (p.position.x, p.position.y, yaw_of(p.orientation))

    def on_status(self, msg: String) -> None:
        self.status = json.loads(msg.data)
        pid, pos = self.status.get("target"), self.status.get("target_position")
        if pid and pos and pid not in self.pid_to_person:
            person = min(self.people, key=lambda q: math.hypot(q.x - pos[0], q.y - pos[1]))
            if math.hypot(person.x - pos[0], person.y - pos[1]) < 1.0:
                self.pid_to_person[pid] = person
        target = self.pid_to_person.get(pid)
        state = self.status.get("state")
        if target and state == "talking_to_person":
            target.attend_until = time.time() + 2.0
        if target and target.persona == "wanders_off" and state == "picking_up_item" and not target.moved:
            target.moved = True
            target.walk_to = (target.x + 2.5, target.y + (2.5 if target.y < 0 else -2.5))
            self.get_logger().info(f"{target.name} wanders off")

    def on_say(self, msg: String) -> None:
        m = json.loads(msg.data)
        now = time.time()
        if m.get("text") and self.outbox:
            self.outbox.clear()                        # robot started talking: the visitor stops mid-sentence
        self.speech_end = max(now, self.speech_end) + m.get("duration_s", 0.0)
        text = m.get("text", "")
        for person in self.people:                    # hearing your name makes you turn around
            if self.robot and person.name.lower() in text.lower() and \
                    math.hypot(person.x - self.robot[0], person.y - self.robot[1]) < 8:
                person.attend_until = now + 10.0
        pid = m.get("person_id")
        self.turn_text[pid] = (self.turn_text.get(pid, "") + " " + text).strip()
        if m.get("final"):
            turn = self.turn_text.pop(pid, "")
            person = self.pid_to_person.get(pid)
            if self.auto_reply and person:
                reply = self.respond(person, turn)
                if reply:
                    self.schedule(reply, self.speech_end + 0.7, person.persona == "chatty")

    # ---- replies -----------------------------------------------------------------
    def respond(self, p: Person, said: str) -> str | None:
        reply = self._respond(p, said)
        if reply:
            p.last_reply = reply
        return reply

    def _respond(self, p: Person, said: str) -> str | None:
        t = said.lower()
        if p.persona == "silent":
            return None
        if any(k in t for k in ("say that again", "say again", "didn't catch", "repeat")):
            return getattr(p, "last_reply", None) or "Sorry?"
        if any(k in t for k in ("name", "call you", "who are you")) and "?" in t and not p.gave_name:
            p.gave_name = True
            return f"Oh hi! I'm {p.name}." if p.persona == "chatty" else f"I'm {p.name}."
        if any(k in t for k in ("here you go", "here's your", "here is your", "enjoy")):
            return "Thank you so much!"
        if "sorry" in t and p.wants:
            p.wants = p.alt
            return f"No worries. Could I have a {SPOKEN[p.wants]} instead?"
        if any(k in t for k in ("still want", "still there", "are you there")):
            return "Yes please!" if p.wants else "Yes, I'm here."
        if any(k in t for k in ("wait right here", "wait here", "right back", "be back")):
            return "Okay, I'll wait here."
        if any(k in t for k in ("pen", "shirt", "plushie", "would you like", "what can i", "want", "swag")):
            if not p.wants:
                return "No thanks, I'm good."
            if p.persona == "chatty" and not p.asked:
                p.asked = True
                return "Hmm, what do you have?"
            return f"I think I'd like a {SPOKEN[p.wants]}, please."
        return "Sure!"

    def schedule(self, text: str, start: float, pause_mid: bool) -> None:
        t = start
        words = text.split()
        for i, word in enumerate(words):
            if pause_mid and i == 2 and len(words) > 4:
                t += 1.3                                  # "I think …(1.3s) I'd like a plushie"
            t += self.word_s
            self.outbox.append((t, word))

    # ---- simulation --------------------------------------------------------------
    def step(self) -> None:
        now = time.time()
        due = [w for w in self.outbox if w[0] <= now]
        self.outbox = [w for w in self.outbox if w[0] > now]
        for _, word in due:
            self.heard_pub.publish(String(data=json.dumps({"text": word, "t": now})))
        for p in self.people:
            if p.walk_to:
                dx, dy = p.walk_to[0] - p.x, p.walk_to[1] - p.y
                d = math.hypot(dx, dy)
                if d < 0.05:
                    p.walk_to = None
                else:
                    s = min(d, 0.5 * 0.05)
                    p.x, p.y = p.x + dx / d * s, p.y + dy / d * s
            if self.robot and now < p.attend_until:
                p.facing = math.atan2(self.robot[1] - p.y, self.robot[0] - p.x)
            elif now > p.next_turn:
                p.next_turn = now + self.rng.uniform(6, 12)
                if self.robot and self.rng.random() < 0.5:
                    p.facing = math.atan2(self.robot[1] - p.y, self.robot[0] - p.x)
                else:
                    p.facing = self.rng.uniform(-math.pi, math.pi)

    def write_people_file(self) -> None:
        tmp = self.people_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"positions": [[round(p.x, 3), round(p.y, 3)] for p in self.people],
                       "names": [p.name for p in self.people]}, f)
        os.replace(tmp, self.people_file)

    def publish_perception(self) -> None:
        if self.people_file:
            self.write_people_file()
        if not self.robot or not self.perception:
            return
        rx, ry, ryaw = self.robot
        faces = []
        for p in self.people:
            dist = math.hypot(p.x - rx, p.y - ry)
            bearing = wrap(math.atan2(p.y - ry, p.x - rx) - ryaw)
            toward_robot = math.atan2(ry - p.y, rx - p.x)
            off_axis = abs(wrap(p.facing - toward_robot))
            if FACE_RANGE[0] <= dist <= FACE_RANGE[1] and abs(bearing) <= FOV and off_axis <= FACING_TOLERANCE:
                noise = 0.5 + 1.0 * off_axis / FACING_TOLERANCE      # side views match worse
                emb = [b + self.rng.gauss(0, noise) for b in p.base]
                faces.append({"embedding": [round(v, 3) for v in emb], "x": round(p.x, 3), "y": round(p.y, 3),
                              "distance": round(dist, 3), "bearing": round(bearing, 3)})
        self.faces_pub.publish(String(data=json.dumps(faces)))
        self.obstacles_pub.publish(String(data=json.dumps([[p.x, p.y] for p in self.people])))


def main():
    rclpy.init()
    node = SimPeople()
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

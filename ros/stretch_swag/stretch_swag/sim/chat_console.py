"""Type as the visitor: your lines go to /swag/heard word by word (like streaming STT),
and the robot's speech and state changes print here.

    ros2 run stretch_swag chat_console
    (start sim_people with auto_reply:=false so the simulated people stay quiet)

The robot talks to whoever it's facing; you play that person.
"""

import json
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


class ChatConsole(Node):
    def __init__(self):
        super().__init__("chat_console")
        self.word_s = self.declare_parameter("word_s", 0.2).value
        self.pub = self.create_publisher(String, "/swag/heard", 50)
        self.create_subscription(String, "/swag/say", self.on_say, 50)
        self.create_subscription(String, "/swag/state", self.on_state, 10)
        self.state = None
        threading.Thread(target=self.read_input, daemon=True).start()

    def on_say(self, msg: String) -> None:
        m = json.loads(msg.data)
        if m.get("text"):
            print(f"\033[96mSTRETCH ({m.get('person_id')}):\033[0m {m['text']}", flush=True)

    def on_state(self, msg: String) -> None:
        if msg.data != self.state:
            self.state = msg.data
            print(f"\033[90m[state: {msg.data}]\033[0m", flush=True)

    def read_input(self) -> None:
        while rclpy.ok():
            try:
                line = input()
            except EOFError:
                return
            line = line.strip()
            if line and line[-1] not in ".?!,":  # Enter = they finished talking (real STT punctuates too)
                line += "."
            for word in line.split():          # "..." in a line becomes a 1 s pause
                if word.strip(".") == "" and len(word) >= 3:
                    time.sleep(1.0)
                    continue
                self.pub.publish(String(data=json.dumps({"text": word, "t": time.time()})))
                time.sleep(self.word_s)


def main():
    rclpy.init()
    node = ChatConsole()
    print("Type what the visitor says and press Enter. '...' = a pause.", flush=True)
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

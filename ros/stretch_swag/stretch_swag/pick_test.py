"""Pick (and put back) goodie bags, nothing else: for setting up and tuning the table.

Stand the robot where the table spot is (in front of the table, ~0.5-1 m from its edge, facing it
or with table_dir_deg set), Stretch's driver + camera running, bag_detect + lane markers running
(robot.launch.py does that, or run them by hand). Then:

    ros2 run stretch_swag pick_test pen                       # pick it, hold it 3 s, put it back
    ros2 run stretch_swag pick_test pen shirt plushie
    ros2 run stretch_swag pick_test pen --ros-args -p handle_top_m:=0.81 -p grasp_offset_m:=0.04
    ros2 run stretch_swag pick_test pen --ros-args -p put_back:=false     # keep holding it

Same parameters as the robot node (handle_top_m, marker_to_edge_m, grip_depth_m, grasp_offset_m,
clearance_m, first_slot_m, slot_pitch_m, lane_markers, table_dir_deg). The [pick] log lines say
what it saw and did at each step. Someone on the runstop.
"""

import math
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String
from visualization_msgs.msg import MarkerArray

from .arm import Arm
from .motion import BaseDriver
from .pick import LOOK_AHEAD, READY, Picker
from .robot_io.locate import Locator
from .robot_node import SwagRobot, parse_pairs


class PickTest(Node):
    on_bags, bags_in_base = SwagRobot.on_bags, SwagRobot.bags_in_base
    on_markers, markers_seen = SwagRobot.on_markers, SwagRobot.markers_seen

    def __init__(self):
        super().__init__("pick_test")
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.bags, self.marker_obs = (0.0, []), (0.0, {})
        self.locator = Locator(self)
        self.driver = BaseDriver(self, p("cmd_vel_topic", "/stretch/cmd_vel"), p("odom_topic", "/odom"),
                                 linear_speed=float(p("linear_speed", 0.15)), angular_speed=float(p("angular_speed", 0.5)))
        self.driver.docking = False
        self.arm = Arm(self)
        self.put_back = bool(p("put_back", True))
        self.create_subscription(String, "/swag/bags", self.on_bags, 5)
        self.create_subscription(MarkerArray, "/swag/lane_markers", self.on_markers, 5)
        facing = self.create_publisher(Float64, "/swag/table_facing", 1)
        self.picker = Picker(self, self.driver, self.arm, self.bags_in_base, self.markers_seen, {
            "table_dir": math.radians(float(p("table_dir_deg", 180.0))),
            "markers": parse_pairs(p("lane_markers", "pen:20,shirt:21,plushie:22")),
            "handle_top_m": float(p("handle_top_m", 0.78)),
            "marker_to_edge_m": float(p("marker_to_edge_m", 0.45)),
            "clearance_m": float(p("clearance_m", 0.24)),
            "first_slot_m": float(p("first_slot_m", 0.08)),
            "slot_pitch_m": float(p("slot_pitch_m", 0.13)),
            "slots": int(p("slots", 3)),
            "grip_depth_m": float(p("grip_depth_m", 0.04)),
            "grasp_offset_m": float(p("grasp_offset_m", 0.0)),
        }, on_table=lambda yaw: facing.publish(Float64(data=float(yaw))))


def main():
    rclpy.init()
    items = [a for a in sys.argv[1:] if not a.startswith("-") and a in ("pen", "shirt", "plushie")]
    node = PickTest()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    log = node.get_logger().info
    try:
        if node.driver.wait_for_pose(10.0) is None:
            log("no odometry: is Stretch's driver running?")
            return
        start = node.driver.pose()
        node.arm.move(**READY, **LOOK_AHEAD)
        for item in items or ["pen"]:
            t0 = time.time()
            r = node.picker.pick(item)
            log(f"PICK {item}: {'OK' if r['success'] else 'FAILED: ' + r['reason']} ({time.time() - t0:.0f} s)")
            if not r["success"]:
                continue
            time.sleep(3.0)
            if node.put_back:
                node.driver.go_to(start[0], start[1])                 # back to the table spot, like the robot does
                t0 = time.time()
                r = node.picker.put_back(item)
                log(f"PUT BACK {item}: {'OK' if r['success'] else 'FAILED: ' + r['reason']} ({time.time() - t0:.0f} s)")
            node.driver.go_to(start[0], start[1])
    except KeyboardInterrupt:
        pass
    finally:
        node.driver.stop()
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()

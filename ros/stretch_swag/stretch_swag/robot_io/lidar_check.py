"""Pre-flight check for the lidar safety stop: prints what the robot node would see, once a second.

    ros2 run stretch_swag lidar_check
    ros2 run stretch_swag lidar_check --ros-args -p scan_yaw_offset:=3.1416     # override TF

Like the robot node, it reads how the lidar is mounted from TF (Stretch's is rotated 180 deg).
Hold a box ~0.5 m in FRONT of the robot (the direction it drives): it must say BLOCKED.
Move it beside or behind the robot: it must say clear. If front/back are still swapped, TF is
wrong: pass scan_yaw_offset:=<angle> here and to robot.launch.py.
Also prints the closest return in every direction, to spot the robot seeing itself.
"""

import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from ..core.geometry import scan_obstacle, wrap
from ..motion import yaw_of


class LidarCheck(Node):
    def __init__(self):
        super().__init__("lidar_check")
        yaw = self.declare_parameter("scan_yaw_offset", "tf", ParameterDescriptor(dynamic_typing=True)).value
        self.offset = None if str(yaw).strip().lower() == "tf" else float(yaw)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.stop = float(self.declare_parameter("safety_distance", 0.6).value)
        self.last = 0.0
        self.create_subscription(LaserScan, self.declare_parameter("scan_topic", "/scan_filtered").value,
                                 self.on_scan, qos_profile_sensor_data)
        print("Waiting for scans...", flush=True)

    def on_scan(self, s: LaserScan) -> None:
        if time.time() - self.last < 1.0:
            return
        self.last = time.time()
        if self.offset is None:
            try:
                t = self.tf.lookup_transform("base_link", s.header.frame_id, Time())
            except TransformException as e:
                print(f"waiting for TF base_link -> {s.header.frame_id}: {e}", flush=True)
                return
            self.offset = -yaw_of(t.transform.rotation)
            print(f"lidar frame '{s.header.frame_id}' is rotated {math.degrees(-self.offset):.0f} deg from the base (TF)",
                  flush=True)
        rmin = max(s.range_min, 0.15)
        d = scan_obstacle(s.ranges, s.angle_min, s.angle_increment, self.stop, yaw_offset=self.offset, min_range=rmin)
        sectors = {"front": 0, "left": 90, "back": 180, "right": -90}
        closest = {}
        for name, center in sectors.items():
            vals = [r for i, r in enumerate(s.ranges)
                    if rmin < r < s.range_max and abs(wrap(s.angle_min + i * s.angle_increment - self.offset
                                                            - math.radians(center))) < math.radians(45)]
            closest[name] = f"{min(vals):.2f}" if vals else "-"
        status = f"BLOCKED ({d:.2f} m ahead)" if d is not None else "clear"
        print(f"{status:24} closest m: " + "  ".join(f"{k} {v}" for k, v in closest.items()), flush=True)


def main():
    rclpy.init()
    node = LidarCheck()
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

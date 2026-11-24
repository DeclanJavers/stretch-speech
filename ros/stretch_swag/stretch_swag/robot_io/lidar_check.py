"""Pre-flight check for the lidar safety stop: prints what the robot node would see, once a second.

    ros2 run stretch_swag lidar_check
    ros2 run stretch_swag lidar_check --ros-args -p scan_yaw_offset:=3.1416

Hold a box ~0.5 m in FRONT of the robot (the direction it drives): it must say BLOCKED.
Move it beside or behind the robot: it must say clear. If front/back are swapped, the lidar
is mounted rotated: try scan_yaw_offset:=3.1416 and pass the same value to robot.launch.py.
Also prints the closest return in every direction, to spot the robot seeing itself.
"""

import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from ..core.geometry import scan_obstacle, wrap


class LidarCheck(Node):
    def __init__(self):
        super().__init__("lidar_check")
        self.offset = float(self.declare_parameter("scan_yaw_offset", 0.0).value)
        self.stop = float(self.declare_parameter("safety_distance", 0.6).value)
        self.last = 0.0
        self.create_subscription(LaserScan, self.declare_parameter("scan_topic", "/scan_filtered").value,
                                 self.on_scan, 10)
        print("Waiting for scans...", flush=True)

    def on_scan(self, s: LaserScan) -> None:
        if time.time() - self.last < 1.0:
            return
        self.last = time.time()
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
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, RuntimeError):
            pass

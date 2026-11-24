"""A kinematic stand-in for Stretch's base: integrates /stretch/cmd_vel into /odom.

Use it instead of the MuJoCo sim to test the whole loop quickly (no physics, no GUI,
real time). Same topics as the MuJoCo driver, so nothing else changes.
"""

import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class FakeBase(Node):
    def __init__(self):
        super().__init__("fake_base")
        self.declare_parameter("cmd_vel_topic", "/stretch/cmd_vel")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("cmd_timeout_s", 0.5)
        self.x = self.y = self.yaw = 0.0
        self.v = self.w = 0.0
        self.last_cmd = 0.0
        self.timeout = self.get_parameter("cmd_timeout_s").value
        self.create_subscription(Twist, self.get_parameter("cmd_vel_topic").value, self.on_cmd, 10)
        self.pub = self.create_publisher(Odometry, self.get_parameter("odom_topic").value, 10)
        self.dt = 1.0 / self.get_parameter("rate_hz").value
        self.last = time.monotonic()
        self.create_timer(self.dt, self.step)
        self.get_logger().info("Fake base ready: cmd_vel -> odom")

    def on_cmd(self, msg: Twist) -> None:
        self.v, self.w = msg.linear.x, msg.angular.z
        self.last_cmd = time.monotonic()

    def step(self) -> None:
        now = time.monotonic()
        dt, self.last = now - self.last, now
        if now - self.last_cmd > self.timeout:     # like the real driver: no commands -> stop
            self.v = self.w = 0.0
        self.yaw = math.atan2(math.sin(self.yaw + self.w * dt), math.cos(self.yaw + self.w * dt))
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id, msg.child_frame_id = "odom", "base_link"
        msg.pose.pose.position.x, msg.pose.pose.position.y = self.x, self.y
        msg.pose.pose.orientation.z = math.sin(self.yaw / 2)
        msg.pose.pose.orientation.w = math.cos(self.yaw / 2)
        msg.twist.twist.linear.x, msg.twist.twist.angular.z = self.v, self.w
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = FakeBase()
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

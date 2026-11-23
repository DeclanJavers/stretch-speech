"""Whole swag loop in simulation, with the perception and conversation cheats.

    ros2 launch stretch_swag sim.launch.py                      # fake base + simulated crowd + robot
    ros2 launch stretch_swag sim.launch.py fake_base:=false     # use the MuJoCo sim's /odom instead
    ros2 launch stretch_swag sim.launch.py auto_reply:=false    # then type: ros2 run stretch_swag chat_console
    ros2 launch stretch_swag sim.launch.py world:=/path/to/crowd.json
    ros2 launch stretch_swag sim.launch.py llm_url:=http://192.168.64.1:8081/v1 face_url:=http://192.168.64.1:8100
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    args = {
        "fake_base": "true",
        "auto_reply": "true",
        "llm_url": "http://192.168.64.1:8081/v1",
        "llm_model": "lfm3-1.7b",
        "face_url": "http://192.168.64.1:8100",
        "pickup_success": "1.0",
        "seed": "0",
        "world": "",                      # crowd file; empty = the package's config/world.json
    }
    c = {k: LaunchConfiguration(k) for k in args}
    return LaunchDescription([
        *[DeclareLaunchArgument(k, default_value=v) for k, v in args.items()],
        Node(package="stretch_swag", executable="fake_base", output="screen",
             condition=IfCondition(c["fake_base"])),
        Node(package="stretch_swag", executable="sim_people", output="screen",
             parameters=[{"auto_reply": c["auto_reply"]}],
             condition=IfCondition(PythonExpression(["'", c["world"], "' == ''"]))),
        Node(package="stretch_swag", executable="sim_people", output="screen",
             parameters=[{"auto_reply": c["auto_reply"], "world": c["world"]}],
             condition=IfCondition(PythonExpression(["'", c["world"], "' != ''"]))),
        Node(package="stretch_swag", executable="robot", output="screen", emulate_tty=True,
             parameters=[{"llm_url": c["llm_url"], "llm_model": c["llm_model"], "face_url": c["face_url"],
                          "pickup_success": c["pickup_success"], "seed": c["seed"]}]),
    ])

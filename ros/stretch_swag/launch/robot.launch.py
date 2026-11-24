"""The swag robot on the real Stretch (v1: badge perception, typed input, spoken output).

Start Stretch's own stack first (see STRETCH-TEST.md): the driver in navigation mode,
the lidar and the head camera. Then:

    ros2 launch stretch_swag robot.launch.py mac_ip:=<Mac's IP>                    # motion OFF (default)
    ros2 launch stretch_swag robot.launch.py mac_ip:=<Mac's IP> motion:=true       # robot moves
    ros2 run stretch_swag chat_console                                             # type what visitors say

Defaults are deliberately cautious: slow speeds, no wandering (it only spins in place
to look for people), the lidar safety stop on, and the table at the robot's start position.
"""

import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = {
        "mac_ip": "192.168.64.1",           # the Mac running llama-server, face_service, speech_service
        "motion": "false",                  # false: no movement at all, skills just "succeed"
        "wander": "false",                  # false: look for people by spinning in place only
        "speech": "auto",                   # mac (Kokoro on the Mac) | espeak (on the robot) | print | auto
        "aruco": "true",                    # start Stretch's ArUco detector with our badge sizes
        "scan_topic": "/scan_filtered",
        "scan_yaw_offset": "0.0",           # robot-forward angle in the lidar frame (check it: STRETCH-TEST.md)
        "linear_speed": "0.15",             # m/s
        "angular_speed": "0.5",             # rad/s
        "table": "0.0,0.0",                 # where the table is, in odometry (start position = 0,0)
        "arena": "-1.5,-1.5,1.5,1.5",       # wandering area (only if wander:=true)
        "pickup_s": "4.0",
    }
    c = {k: LaunchConfiguration(k) for k in args}
    here = get_package_share_directory("stretch_swag")
    actions = [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]

    try:
        stretch_dict = os.path.join(get_package_share_directory("stretch_core"), "config", "stretch_marker_dict.yaml")
        actions.append(Node(package="stretch_core", executable="detect_aruco_markers", output="screen",
                            parameters=[stretch_dict, os.path.join(here, "config", "badges.yaml")],
                            condition=IfCondition(c["aruco"])))
    except PackageNotFoundError:
        actions.append(LogInfo(msg="stretch_core not found: start an ArUco detector yourself"))

    actions += [
        Node(package="stretch_swag", executable="badge_faces", output="screen"),
        Node(package="stretch_swag", executable="speaker_node", output="screen",
             parameters=[{"backend": c["speech"], "mac_url": ["http://", c["mac_ip"], ":8200"]}]),
        Node(package="stretch_swag", executable="robot", output="screen", emulate_tty=True, parameters=[{
            "llm_url": ["http://", c["mac_ip"], ":8081/v1"],
            "face_url": ["http://", c["mac_ip"], ":8100"],
            "motion_enabled": c["motion"],
            "wander": c["wander"],
            "safety_source": "scan",
            "scan_topic": c["scan_topic"],
            "scan_yaw_offset": c["scan_yaw_offset"],
            "linear_speed": c["linear_speed"],
            "angular_speed": c["angular_speed"],
            "table": c["table"],
            "arena": c["arena"],
            "pickup_s": c["pickup_s"],
            "approach_range": 2.0,
            "mute_while_speaking": False,      # v1 input is typed (chat_console): never drop typed lines
            "standoff": 1.0,
        }]),
    ]
    return LaunchDescription(actions)

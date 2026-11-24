"""The swag robot on the real Stretch: faces, the robot's mic (Whisper on the Mac), espeak voice.

Start Stretch's own stack first (see STRETCH-TEST.md): the driver in navigation mode,
the lidar and the head camera. Then:

    ros2 launch stretch_swag robot.launch.py mac_ip:=<Mac's IP>                    # motion OFF (default)
    ros2 launch stretch_swag robot.launch.py mac_ip:=<Mac's IP> motion:=true       # robot moves
    ros2 launch stretch_swag robot.launch.py mac_ip:=<Mac's IP> perception:=badges # ArUco badges instead of faces
    ros2 launch stretch_swag robot.launch.py mac_ip:=<Mac's IP> pick_mode:=arm     # pick bags from the lanes
    ros2 run rqt_image_view rqt_image_view /swag/debug                             # camera + who it knows
    ros2 launch stretch_swag robot.launch.py ... mic:=false speech:=print          # local testing: no audio
    ros2 run stretch_swag chat_console                                             # type what visitors say

Defaults are deliberately cautious: slow speeds, no wandering (it only spins in place
to look for people), the lidar safety stop on, and the table at the robot's start position.
"""

import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    args = {
        "mac_ip": "192.168.64.1",           # the Mac running llama-server, face_service, speech_service
        "motion": "false",                  # false: no movement at all, skills just "succeed"
        "wander": "false",                  # false: look for people by spinning in place only
        "speech": "espeak",                 # espeak (robot's speaker) | mac (Kokoro, Mac's speaker) | print (no sound)
        "mic": "true",                      # true: robot's microphone -> Whisper on the Mac | false: type (chat_console)
        "mic_device": "default",            # ALSA device for the robot's mic (arecord -l)
        "perception": "faces",              # faces (YuNet + SFace on the robot's CPU) | badges (ArUco fallback)
        "aruco": "stretch",                 # badges only: stretch (Stretch's detector) | swag (ours)
        "pick_mode": "mock",                # mock (pause) | arm (goodie bags in lanes, see pick.py)
        "table_dir_deg": "180.0",           # roughly: heading from the start spot toward the table (180 = behind)
        # the table (see STRETCH-TEST.md, "Calibrating the table"):
        "handle_top_m": "0.78",             # MEASURE: floor -> top of a bag's rope loops (standing on the table)
        "marker_to_edge_m": "0.45",         # MEASURE: lane marker's face -> table's front edge
        "marker_mm": "100.0",               # MEASURE: the printed marker's black square
        "grasp_offset_m": "0.0",            # grasp this far from the bag's middle toward the robot (tune: pick_test)
        "grip_depth_m": "0.04",             # fingers close this far below the loops' top (tune: pick_test)
        "clearance_m": "0.24",              # base_link -> table edge when it turns beside the table (needs >= 0.18)
        "first_slot_m": "0.08",             # table edge -> middle of a lane's front bag (MEASURE)
        "slot_pitch_m": "0.13",             # bag to bag in a lane (MEASURE, middle to middle)
        "scan_topic": "/scan_filtered",
        "scan_yaw_offset": "tf",            # lidar mounting: from TF, or an angle to override (STRETCH-TEST.md)
        "linear_speed": "0.15",             # m/s
        "angular_speed": "0.5",             # rad/s
        "table": "0.0,0.0",                 # where the table is, in odometry (start position = 0,0)
        "arena": "-1.5,-1.5,1.5,1.5",       # wandering area (only if wander:=true)
        "pickup_s": "4.0",
    }
    c = {k: LaunchConfiguration(k) for k in args}
    here = get_package_share_directory("stretch_swag")
    actions = [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]

    badges_yaml = os.path.join(here, "config", "badges.yaml")
    is_ = lambda arg, value: PythonExpression(["'", c[arg], "' == '", value, "'"])  # noqa: E731
    badges = IfCondition(is_("perception", "badges"))
    faces = IfCondition(is_("perception", "faces"))
    try:
        stretch_dict = os.path.join(get_package_share_directory("stretch_core"), "config", "stretch_marker_dict.yaml")
        actions.append(Node(package="stretch_core", executable="detect_aruco_markers", output="screen",
                            parameters=[stretch_dict, badges_yaml],
                            condition=IfCondition(PythonExpression(["'", c["perception"], "' == 'badges' and '",
                                                                     c["aruco"], "' == 'stretch'"]))))
    except PackageNotFoundError:
        actions.append(LogInfo(msg="stretch_core not found: use aruco:=swag"))
    actions.append(Node(package="stretch_swag", executable="aruco_detect", output="screen",
                        parameters=[badges_yaml],
                        condition=IfCondition(PythonExpression(["'", c["perception"], "' == 'badges' and '",
                                                                c["aruco"], "' == 'swag'"]))))

    actions += [
        Node(package="stretch_swag", executable="badge_faces", output="screen", condition=badges),
        Node(package="stretch_swag", executable="face_detect", output="screen", condition=faces),
        Node(package="stretch_swag", executable="mic_node", output="screen", condition=IfCondition(c["mic"]),
             parameters=[{"mac_url": ["http://", c["mac_ip"], ":8200"], "device": c["mic_device"]}]),
        Node(package="stretch_swag", executable="aruco_detect", name="lane_markers", output="screen",
             parameters=[{"default_length_mm": c["marker_mm"], "debug_image": False, "max_hz": 10.0}],
             remappings=[("/aruco/marker_array", "/swag/lane_markers")]),
        Node(package="stretch_swag", executable="debug_view", output="screen"),
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
            "mute_while_speaking": c["mic"],   # real mic: don't hear itself; typed input: never drop lines
            "standoff": 1.0,
            "pick_mode": c["pick_mode"],
            "table_dir_deg": c["table_dir_deg"],
            "handle_top_m": c["handle_top_m"], "marker_to_edge_m": c["marker_to_edge_m"],
            "first_slot_m": c["first_slot_m"], "slot_pitch_m": c["slot_pitch_m"],
            "grasp_offset_m": c["grasp_offset_m"], "grip_depth_m": c["grip_depth_m"], "clearance_m": c["clearance_m"],
        }]),
    ]
    return LaunchDescription(actions)

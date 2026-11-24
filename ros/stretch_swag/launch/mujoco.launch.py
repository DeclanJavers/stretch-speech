"""Whole swag loop in MuJoCo with real perception: cutout people wearing ArUco badges, seen
through the simulated head camera, and the real lidar safety stop. Everything downstream of
the camera is what runs on the robot (aruco_detect -> badge_faces -> robot).

    ros2 launch stretch_swag mujoco.launch.py                       # 2x speed; you type the visitors' lines:
    ros2 run stretch_swag chat_console                              #   (second terminal)
    ros2 launch stretch_swag mujoco.launch.py speed:=3
    ros2 launch stretch_swag mujoco.launch.py auto_reply:=true      # visitors answer by themselves
    ros2 launch stretch_swag mujoco.launch.py viewer:=false         # no MuJoCo window (works over SSH)

Dress rehearsal for the real robot: MuJoCo stands in for the hardware, robot.launch.py runs on top:
    ros2 launch stretch_swag mujoco.launch.py stack:=false
    ros2 launch stretch_swag robot.launch.py mac_ip:=192.168.64.1 aruco:=swag motion:=true

See what the robot sees, and who it knows:   ros2 run rqt_image_view rqt_image_view /swag/debug
Real face recognition instead of badges:     perception:=faces  (put photos in sim_faces/<Name>.jpg)
Run it in the VM's desktop (the MuJoCo window needs a display). Mac services as usual.
"""

import os

from ament_index_python.packages import get_package_share_directory, get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = {
        "speed": "2.0",                    # physics speed vs real time (conversation timing stays real)
        "viewer": "true",                  # the MuJoCo window
        "auto_reply": "false",              # false: you type the visitors' lines in chat_console; true: they answer themselves
        "llm_url": "http://192.168.64.1:8081/v1",
        "llm_model": "lfm3-1.7b",
        "face_url": "http://192.168.64.1:8100",
        "world": "",                       # crowd file; empty = the package's config/world.json
        "camera_scale": "1.5",             # head camera 424x240 x this (1.5 = 640x360)
        "camera_hz": "10",
        "pickup_success": "1.0",
        "seed": "0",
        "perception": "badges",            # badges (ArUco) | faces (YuNet + SFace; cutouts need photos in sim_faces/)
        "pick_mode": "arm",                # arm: pick the goodie bags for real | mock: just wait
        "stack": "true",                   # false: only the simulated robot + visitors (run robot.launch.py on top)
    }
    c = {k: LaunchConfiguration(k) for k in args}
    here = get_package_share_directory("stretch_swag")
    people_file = "/tmp/stretch_swag_people.json"
    on = lambda value: IfCondition(PythonExpression(["'", c["stack"], "' == 'true' and '", c["perception"],  # noqa: E731
                                                     "' == '", value, "'"]))
    badges, faces = on("badges"), on("faces")

    urdf = get_package_share_path("stretch_description") / "urdf" / "stretch.urdf"
    if not urdf.is_file():
        urdf = get_package_share_path("stretch_description") / "urdf" / "stretch_description_SE3_eoa_wrist_dw3_tool_sg3.xacro"
    robot_description = ParameterValue(Command(["xacro ", str(urdf)]), value_type=str)

    return LaunchDescription([
        *[DeclareLaunchArgument(k, default_value=v) for k, v in args.items()],

        # --- the simulated robot (Stretch's MuJoCo driver with our physics loop and people) ---
        Node(package="joint_state_publisher", executable="joint_state_publisher", output="log",
             parameters=[{"source_list": ["/stretch/joint_states"], "rate": 30.0,
                          "robot_description": robot_description}],
             arguments=["--ros-args", "--log-level", "error"]),
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="log",
             parameters=[{"robot_description": robot_description, "publish_frequency": 30.0}],
             arguments=["--ros-args", "--log-level", "error"]),
        Node(package="stretch_swag", executable="mujoco_sim", output="screen", emulate_tty=True,
             remappings=[("cmd_vel", "/stretch/cmd_vel"), ("joint_states", "/stretch/joint_states")],
             parameters=[{"rate": 30.0, "timeout": 0.5, "broadcast_odom_tf": True,
                          "fail_out_of_range_goal": False, "mode": "navigation",
                          "use_mujoco_viewer": c["viewer"], "use_cameras": True, "use_robocasa": False}],
             additional_env={"SWAG_SIM_SPEED": c["speed"], "SWAG_WORLD": c["world"],
                             "SWAG_PEOPLE_FILE": people_file, "SWAG_CAMERA_SCALE": c["camera_scale"],
                             "SWAG_CAMERA_HZ": c["camera_hz"], "MUJOCO_GL": "egl"}),

        # --- perception: exactly the real robot's chain from the camera on ---
        Node(package="stretch_swag", executable="aruco_detect", output="screen", condition=IfCondition(c["stack"]),
             parameters=[os.path.join(here, "config", "badges.yaml"),          # badges and the bags' tags
                         {"debug_image": PythonExpression(["'", c["perception"], "' == 'badges'"])}]),
        Node(package="stretch_swag", executable="badge_faces", output="screen", condition=badges),
        Node(package="stretch_swag", executable="face_detect", output="screen", condition=faces),
        Node(package="stretch_swag", executable="debug_view", output="screen", condition=IfCondition(c["stack"])),

        # --- the visitors' voices (and where they walk, shared with MuJoCo) ---
        Node(package="stretch_swag", executable="sim_people", output="screen",
             parameters=[{"auto_reply": c["auto_reply"], "perception": False, "people_file": people_file,
                          "world": c["world"]}]),

        # --- the robot ---
        Node(package="stretch_swag", executable="robot", output="screen", emulate_tty=True,
             condition=IfCondition(c["stack"]), parameters=[{
            "llm_url": c["llm_url"], "llm_model": c["llm_model"], "face_url": c["face_url"],
            "pickup_success": c["pickup_success"], "seed": c["seed"],
            "safety_source": "scan",                      # MuJoCo's simulated lidar on /scan_filtered
            "pick_mode": c["pick_mode"], "stock": "pen:2,shirt:2,plushie:2", "table_dir_deg": 180.0,
            "mute_while_speaking": c["auto_reply"],       # typing in chat_console: never drop lines
        }]),
    ])

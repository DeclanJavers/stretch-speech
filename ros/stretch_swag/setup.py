from glob import glob

from setuptools import setup

package_name = "stretch_swag"

setup(
    name=package_name,
    version="0.2.0",
    packages=[package_name, f"{package_name}.core", f"{package_name}.sim", f"{package_name}.robot_io"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.json") + glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Swag-handout robot for Stretch: state machine, LLM harness, skills and sim cheats",
    license="MIT",
    entry_points={
        "console_scripts": [
            "robot = stretch_swag.robot_node:main",
            "fake_base = stretch_swag.sim.fake_base:main",
            "sim_people = stretch_swag.sim.sim_people:main",
            "chat_console = stretch_swag.sim.chat_console:main",
            "badge_faces = stretch_swag.robot_io.badge_faces:main",
            "speaker_node = stretch_swag.robot_io.speaker_node:main",
            "mic_node = stretch_swag.robot_io.mic_node:main",
            "lidar_check = stretch_swag.robot_io.lidar_check:main",
            "aruco_detect = stretch_swag.robot_io.aruco_detect:main",
            "face_detect = stretch_swag.robot_io.face_detect:main",
            "bag_detect = stretch_swag.robot_io.bag_detect:main",
            "debug_view = stretch_swag.robot_io.debug_view:main",
            "mujoco_sim = stretch_swag.sim.mujoco_sim:main",
            "pick_test = stretch_swag.pick_test:main",
            "behavior = stretch_swag.behavior_node:main",   # the progress-report demo
        ],
    },
)

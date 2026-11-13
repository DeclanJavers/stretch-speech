from setuptools import setup

package_name = "stretch_swag"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Swag-handout behavior for Stretch",
    license="MIT",
    entry_points={
        "console_scripts": [
            "behavior = stretch_swag.behavior_node:main",
        ],
    },
)

from setuptools import setup
from glob import glob

package_name = "rover_recorder"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hyunseo",
    maintainer_email="inhsroy@hanyang.ac.kr",
    description="Motor bridge + rosbag recorder.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "motor_bridge_node = rover_recorder.motor_bridge_node:main",
            "bag_recorder_node = rover_recorder.bag_recorder_node:main",
        ],
    },
)

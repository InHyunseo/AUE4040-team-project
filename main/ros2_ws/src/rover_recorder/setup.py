from setuptools import setup

package_name = "rover_recorder"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="team",
    maintainer_email="inhsroy@hanyang.ac.kr",
    description="Data recorder + teleop.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "recorder_node = rover_recorder.recorder_node:main",
        ],
    },
)

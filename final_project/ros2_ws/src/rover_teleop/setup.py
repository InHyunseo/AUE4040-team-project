from setuptools import setup

package_name = "rover_teleop"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hyunseo",
    maintainer_email="inhsroy@hanyang.ac.kr",
    description="1D steering-level keyboard teleop.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "teleop_node = rover_teleop.teleop_node:main",
        ],
    },
)

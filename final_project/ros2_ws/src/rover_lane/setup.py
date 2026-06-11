from setuptools import setup
from glob import glob

package_name = "rover_lane"

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
    description="E2E autonomous-driving inference (SegFormer+YOLO+E2ENet TensorRT) -> /cmd_vel.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "e2e_infer_node = rover_lane.e2e_infer_node:main",
        ],
    },
)

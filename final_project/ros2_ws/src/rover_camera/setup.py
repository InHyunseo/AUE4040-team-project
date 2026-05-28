from setuptools import setup
from glob import glob

package_name = "rover_camera"

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
    description="Dual CSI camera publisher (BEV + Front).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "camera_node = rover_camera.camera_node:main",
        ],
    },
)

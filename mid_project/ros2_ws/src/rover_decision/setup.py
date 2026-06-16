from setuptools import setup

package_name = "rover_decision"

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
    description="Decision FSM node.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "decision_node = rover_decision.decision_node:main",
        ],
    },
)

from setuptools import find_packages, setup

package_name = "anomaly_inspection_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/inspection.launch.py", "launch/inspection_only.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Adrian",
    maintainer_email="adrian@example.com",
    description="Lifecycle ROS 2 node for ONNX-based visual anomaly inspection.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "inspection_node = anomaly_inspection_ros.inspection_node:main",
            "mvtec_image_publisher = anomaly_inspection_ros.mvtec_image_publisher:main",
        ],
    },
)

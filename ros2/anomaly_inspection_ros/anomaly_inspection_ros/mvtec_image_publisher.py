"""Publishes MVTec AD test images in a loop, simulating a camera feed for the inspection node."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterator

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp"})


def _sensor_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class MvtecImagePublisher(Node):
    def __init__(self) -> None:
        super().__init__("mvtec_image_publisher")
        self.declare_parameter("dataset_root", str(Path.home() / "datasets" / "MVTecAD"))
        self.declare_parameter("category", "metal_nut")
        self.declare_parameter("split", "test")
        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter("output_topic", "/camera/image_raw")

        root = Path(self.get_parameter("dataset_root").value).expanduser()
        category = self.get_parameter("category").value
        split = self.get_parameter("split").value
        image_dir = root / category / split

        self._paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self._paths:
            raise FileNotFoundError(f"No images found under {image_dir}")
        self.get_logger().info(f"Looping over {len(self._paths)} images from {image_dir}")

        self._cycle: Iterator[Path] = itertools.cycle(self._paths)
        self._bridge = CvBridge()
        self._pub = self.create_publisher(Image, self.get_parameter("output_topic").value, _sensor_qos())

        rate_hz = self.get_parameter("rate_hz").value
        self._timer = self.create_timer(1.0 / rate_hz, self._publish_next)

    def _publish_next(self) -> None:
        path = next(self._cycle)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warning(f"Could not read {path}, skipping")
            return
        msg = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self._pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MvtecImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

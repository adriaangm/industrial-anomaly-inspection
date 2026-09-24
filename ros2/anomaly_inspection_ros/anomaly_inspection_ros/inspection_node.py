"""Lifecycle ROS 2 node wrapping the torch-free OnnxInspector."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.lifecycle import LifecycleNode, LifecycleState, Publisher as LifecyclePublisher, TransitionCallbackReturn
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.subscription import Subscription
from sensor_msgs.msg import Image

from anomaly_inspection_msgs.msg import InspectionResult


def _sensor_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class InspectionNode(LifecycleNode):
    """Subscribes to Image, runs ONNX anomaly inspection, publishes InspectionResult (+ optional heatmap)."""

    def __init__(self) -> None:
        super().__init__("anomaly_inspection_node")

        self.declare_parameter("model_path", "")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("input_topic", "/camera/image_raw")
        self.declare_parameter("output_topic", "/inspection/result")
        self.declare_parameter("publish_heatmap", True)
        self.declare_parameter("heatmap_topic", "/inspection/heatmap")
        self.declare_parameter("warmup_iterations", 10)

        self._bridge = CvBridge()
        self._inspector = None
        self._sub: Optional[Subscription] = None
        self._result_pub: Optional[LifecyclePublisher] = None
        self._heatmap_pub: Optional[LifecyclePublisher] = None

    # -- Lifecycle callbacks --------------------------------------------------

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        model_path = Path(self.get_parameter("model_path").value)
        device = self.get_parameter("device").value
        threshold = self.get_parameter("threshold").value
        warmup = self.get_parameter("warmup_iterations").value

        if not model_path.name:
            self.get_logger().error("Parameter 'model_path' is empty.")
            return TransitionCallbackReturn.FAILURE

        try:
            # Lazy import inside the try: import failures must be logged, not swallowed by rclpy.
            from anomaly_inspection.runtime import OnnxInspector

            self._inspector = OnnxInspector(model_path, device=device, threshold=threshold)
            self._inspector.warmup(warmup)
        except Exception:
            self.get_logger().error(f"Failed to initialize OnnxInspector:\n{traceback.format_exc()}")
            return TransitionCallbackReturn.FAILURE

        self.get_logger().info(
            f"Loaded '{model_path.name}' on {self._inspector.providers[0]} "
            f"(input {self._inspector.input_hw})"
        )

        self._result_pub = self.create_publisher(InspectionResult, self.get_parameter("output_topic").value, 10)
        if self.get_parameter("publish_heatmap").value:
            self._heatmap_pub = self.create_publisher(Image, self.get_parameter("heatmap_topic").value, 10)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        input_topic = self.get_parameter("input_topic").value
        self._sub = self.create_subscription(Image, input_topic, self._on_image, _sensor_qos())
        self.get_logger().info(f"Activated. Listening on '{input_topic}'.")
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        self.get_logger().info("Deactivated.")
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._release_resources()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._release_resources()
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._release_resources()
        return TransitionCallbackReturn.SUCCESS

    def _release_resources(self) -> None:
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        if self._result_pub is not None:
            self.destroy_publisher(self._result_pub)
            self._result_pub = None
        if self._heatmap_pub is not None:
            self.destroy_publisher(self._heatmap_pub)
            self._heatmap_pub = None
        self._inspector = None

    # -- Callback principal -----------------------------------------------------

    def _on_image(self, msg: Image) -> None:
        if self._inspector is None or self._result_pub is None:
            return

        try:
            image_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError:
            self.get_logger().warning("Failed to convert incoming Image message", throttle_duration_sec=5.0)
            return

        result = self._inspector.infer(image_bgr)

        out = InspectionResult()
        out.header = msg.header
        out.model_name = Path(self.get_parameter("model_path").value).stem
        out.score = result.score
        out.is_anomalous = result.is_anomalous
        out.threshold = self._inspector.threshold
        out.inference_ms = result.inference_ms
        out.total_ms = result.total_ms
        self._result_pub.publish(out)

        if self._heatmap_pub is not None and result.anomaly_map is not None:
            self._publish_heatmap(msg.header, image_bgr, result.anomaly_map)

    def _publish_heatmap(self, header, image_bgr: np.ndarray, anomaly_map: np.ndarray) -> None:
        normalized = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        colored = cv2.resize(colored, (image_bgr.shape[1], image_bgr.shape[0]))
        overlay = cv2.addWeighted(image_bgr, 0.6, colored, 0.4, 0)
        out_msg = self._bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        out_msg.header = header
        self._heatmap_pub.publish(out_msg)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = InspectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

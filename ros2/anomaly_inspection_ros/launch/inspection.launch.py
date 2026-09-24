"""Launch the MVTec test publisher and the anomaly inspection lifecycle node,
auto-configuring and activating it (same pattern Nav2 uses internally)."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition

DEFAULT_DATASET_ROOT = os.path.expanduser("~/datasets/MVTecAD")


def generate_launch_description() -> LaunchDescription:
    model_path_arg = DeclareLaunchArgument("model_path", description="Absolute path to the exported .onnx model")
    device_arg = DeclareLaunchArgument("device", default_value="cuda")
    category_arg = DeclareLaunchArgument("category", default_value="metal_nut")
    dataset_root_arg = DeclareLaunchArgument("dataset_root", default_value=DEFAULT_DATASET_ROOT)
    rate_arg = DeclareLaunchArgument("rate_hz", default_value="5.0")

    inspection_node = LifecycleNode(
        package="anomaly_inspection_ros",
        executable="inspection_node",
        name="anomaly_inspection_node",
        namespace="",
        parameters=[{
            "model_path": LaunchConfiguration("model_path"),
            "device": LaunchConfiguration("device"),
            "input_topic": "/camera/image_raw",
            "output_topic": "/inspection/result",
            "publish_heatmap": True,
            "heatmap_topic": "/inspection/heatmap",
        }],
    )

    image_publisher = Node(
        package="anomaly_inspection_ros",
        executable="mvtec_image_publisher",
        name="mvtec_image_publisher",
        parameters=[{
            "dataset_root": LaunchConfiguration("dataset_root"),
            "category": LaunchConfiguration("category"),
            "rate_hz": LaunchConfiguration("rate_hz"),
            "output_topic": "/camera/image_raw",
        }],
    )

    configure_on_start = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(inspection_node),
            transition_id=Transition.TRANSITION_CONFIGURE,
        )
    )
    activate_on_configured = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=inspection_node,
            goal_state="inactive",
            entities=[EmitEvent(
                event=ChangeState(
                    lifecycle_node_matcher=matches_action(inspection_node),
                    transition_id=Transition.TRANSITION_ACTIVATE,
                )
            )],
        )
    )

    return LaunchDescription([
        model_path_arg, device_arg, category_arg, dataset_root_arg, rate_arg,
        inspection_node,
        image_publisher,
        configure_on_start,
        activate_on_configured,
    ])

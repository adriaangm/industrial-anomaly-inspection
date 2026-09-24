"""Production launch: only the anomaly inspection lifecycle node (no test image publisher).
Auto-configures and activates it, same pattern as inspection.launch.py."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def generate_launch_description() -> LaunchDescription:
    model_path_arg = DeclareLaunchArgument("model_path", description="Absolute path to the exported .onnx model")
    device_arg = DeclareLaunchArgument("device", default_value="cuda")
    input_topic_arg = DeclareLaunchArgument("input_topic", default_value="/camera/image_raw")

    inspection_node = LifecycleNode(
        package="anomaly_inspection_ros",
        executable="inspection_node",
        name="anomaly_inspection_node",
        namespace="",
        parameters=[{
            "model_path": LaunchConfiguration("model_path"),
            "device": LaunchConfiguration("device"),
            "input_topic": LaunchConfiguration("input_topic"),
            "output_topic": "/inspection/result",
            "publish_heatmap": True,
            "heatmap_topic": "/inspection/heatmap",
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
        model_path_arg, device_arg, input_topic_arg,
        inspection_node,
        configure_on_start,
        activate_on_configured,
    ])

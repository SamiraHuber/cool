import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def generate_launch_description():
    pkg_share = get_package_share_directory('bordsupr')

    config_yaml = os.path.join(pkg_share, 'config', 'config.yaml')
    embed_model = os.getenv("BORDSUPR_EMBED_MODEL", "convnext_tiny.fb_in1k")
    embed_fallback_model = os.getenv(
        "BORDSUPR_EMBED_FALLBACK_MODEL",
        "convnext_tiny.fb_in1k",
    )
    osnet_checkpoint = os.getenv(
        "BORDSUPR_OSNET_CHECKPOINT",
        "/shared/cluster_testsets/osnet_finetuned_persons_big_2_improved.pth",
    )
    vlm_overrides = {
        "use_kimi": _env_flag("USE_KIMI", False),
        "kimi_api_url": os.getenv("KIMI_API_URL", "https://api.moonshot.ai/v1"),
        "kimi_api_key": os.getenv("KIMI_API_KEY", ""),
        "kimi_model": os.getenv("KIMI_MODEL", "kimi-k2.6"),
        "use_gemini": _env_flag("USE_GEMINI", False),
        "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
    }

    database_overrides = {
        "scene_dedup_enabled": _env_flag("BORDSUPR_SCENE_DEDUP_ENABLED", True),
    }

    enable_optical_flow = _env_flag("BORDSUPR_ENABLE_OPTICAL_FLOW", False)
    enable_lidar_occupancy = _env_flag("BORDSUPR_ENABLE_LIDAR_OCCUPANCY", False)

    nodes = [
        # NOTE: DynoSAM (the C++ dynamic-SLAM backend) is no longer used. The nodes
        # below named "dynosam_*" and the "/dynosam/*" topics are LOCAL nodes/topics
        # with legacy names kept for compatibility; they do not require DynoSAM.
        # Object 3D positions come from the depth camera (position_source='depth').
        # yolo_segmentation_node disabled in favor of dynosam_segmentation_node
        # which uses the switchable detection_engine (YOLO or OWLv2).
        # Node(
        #     package="bordsupr",
        #     executable="yolo_segmentation_node",
        #     name="yolo_segmentation_node",
        #     output="screen",
        #     parameters=[config_yaml],
        # ),
        Node(
            package="bordsupr_services",
            executable="convnext_embedding_service",
            name="get_convnext_embedding",
            output="screen",
            parameters=[{
                "service_name": "/get_convnext_embedding",
                "model_name": embed_model,
                "use_multiview": False,
                "center_crop_ratio": 0.82,
                "border_suppression_ratio": 0.12,
                "min_side_for_extra_views_px": 72,
                "white_balance": True,
                "auto_brightness": True,
            }],
        ),
        Node(
            package="bordsupr_services",
            executable="osnet_embedding_service",
            name="get_osnet_embedding",
            output="screen",
            parameters=[{
                "service_name": "/get_osnet_embedding",
                "checkpoint_path": osnet_checkpoint,
                "white_balance": True,
                "auto_brightness": True,
            }],
        ),
        Node(
            package="bordsupr",
            executable="dynosam_optical_flow_node",
            name="dynosam_optical_flow_node",
            output="screen",
            parameters=[config_yaml],
        ) if enable_optical_flow else None,
        Node(
            package="bordsupr",
            executable="dynosam_segmentation_node",
            name="dynosam_segmentation_node",
            output="screen",
            parameters=[
                config_yaml,
                {
                    "yolo_output_topic": "/dynosam/yolo_output",
                },
            ],
        ),
        Node(
            package="bordsupr",
            executable="face_detector_node",
            name="face_detector_node",
            output="screen",
            parameters=[config_yaml],
        ),
        Node(
            package="bordsupr",
            executable="scene_description_node",
            name="scene_description_node",
            output="screen",
            parameters=[config_yaml, vlm_overrides],
        ),
        Node(
            package="bordsupr",
            executable="stitched_scene_description_node",
            name="stitched_scene_description_node",
            output="screen",
            parameters=[config_yaml, vlm_overrides, {"rgb_topic": "/spot/camera/frontmiddle_virtual/image"}],
        ),
        Node(
            package="bordsupr",
            executable="video_recorder_node",
            name="video_recorder_node",
            output="screen",
            parameters=[config_yaml],
        ),
        Node(
            package="bordsupr",
            executable="interaction_description_node",
            name="interaction_description_node",
            output="screen",
            parameters=[config_yaml, vlm_overrides],
        ),
        Node(
            package="bordsupr",
            executable="database_node",
            name="database_node",
            output="screen",
            parameters=[config_yaml, database_overrides],
        ),
        Node(
            package="bordsupr",
            executable="rotate_topics",
            name="rotate_topics",
            output="screen",
            parameters=[config_yaml],
        ),
        Node(
            package="bordsupr",
            executable="slam_listener",
            name="slam_listener",
            output="screen",
        ),
        Node(
            package="bordsupr",
            executable="lidar_occupancy_map_node",
            name="lidar_occupancy_map_node",
            output="screen",
            parameters=[config_yaml],
        ) if enable_lidar_occupancy else None,
    ]

    return LaunchDescription([n for n in nodes if n is not None])

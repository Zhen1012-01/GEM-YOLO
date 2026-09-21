"""Launch the configurable GEM lane follower."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("gem_lane_yolo")
    default_config = os.path.join(package_share, "config", "lane_follow.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            Node(
                package="gem_lane_yolo",
                executable="lane_follow_node",
                name="lane_follow_node",
                output="screen",
                parameters=[LaunchConfiguration("config")],
            ),
        ]
    )

"""
Full-stack ergonomic monitoring launcher.

Usage:
    ros2 launch rula_gui ergonomic_stack.launch.py \
        front_id:=<camera-serial> \
        right_id:=<camera-serial> \
        left_id:=<camera-serial> \
        robot_ip:=192.168.0.100
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Launch arguments ────────────────────────────────────────────────────
    front_id_arg = DeclareLaunchArgument(
        'front_id',
        description='RealSense serial number for the front-facing camera')

    right_id_arg = DeclareLaunchArgument(
        'right_id',
        description='RealSense serial number for the right-side camera')

    left_id_arg = DeclareLaunchArgument(
        'left_id',
        description='RealSense serial number for the left-side camera')

    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.0.100',
        description='IP address of the UR5e robot')

    # ── point_2D_extractor — three camera nodes ─────────────────────────────
    front_camera = Node(
        package='point_2D_extractor',
        executable='point_2D',
        name='front_camera',
        output='screen',
        arguments=[
            '--active_sides', '0',
            '--device_name', LaunchConfiguration('front_id'),
        ],
    )

    right_camera = Node(
        package='point_2D_extractor',
        executable='point_2D',
        name='right_camera',
        output='screen',
        arguments=[
            '--active_sides', '1',
            '--device_name', LaunchConfiguration('right_id'),
        ],
    )

    left_camera = Node(
        package='point_2D_extractor',
        executable='point_2D',
        name='left_camera',
        output='screen',
        arguments=[
            '--active_sides', '2',
            '--device_name', LaunchConfiguration('left_id'),
        ],
    )

    # ── RULA calculator ──────────────────────────────────────────────────────
    rula_calculator = Node(
        package='rula_calculator',
        executable='rula_calculator',
        name='rula_calculator',
        output='screen',
    )

    # ── Ergonomic assistant (robot controller) ───────────────────────────────
    config_file = os.path.join(
        get_package_share_directory('rula_calculator'),
        'config', 'ergonomic_assistant.yaml')

    ergonomic_assistant = Node(
        package='rula_calculator',
        executable='pcb_ergonomic_assistant',
        name='pcb_ergonomic_assistant',
        output='screen',
        parameters=[
            config_file,
            {'robot_ip': LaunchConfiguration('robot_ip')},
        ],
    )

    # ── GUI ──────────────────────────────────────────────────────────────────
    gui = Node(
        package='rula_gui',
        executable='rulaGui',
        name='rula_gui',
        output='screen',
    )

    return LaunchDescription([
        front_id_arg,
        right_id_arg,
        left_id_arg,
        robot_ip_arg,
        front_camera,
        right_camera,
        left_camera,
        rula_calculator,
        ergonomic_assistant,
        gui,
    ])

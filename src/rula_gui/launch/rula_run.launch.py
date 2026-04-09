from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Node 1: From package1
        Node(
            package='rula_gui',
            executable='rulaGui',
            name='GUI',
            output='screen',
        ),

        # Node 2: From package2
        Node(
            package='rula_calculator',
            executable='rula_calculator',
            name='RULA_CALCULATOR',
            output='screen'
        )
    ])

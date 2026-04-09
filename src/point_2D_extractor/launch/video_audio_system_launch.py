from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare multiple mandatory launch arguments
    front_id_arg = DeclareLaunchArgument(
        'front_id',
        description='front camera number'
    )

    left_id_arg = DeclareLaunchArgument(
        'left_id',
        description='left camera number'
    )

    right_id_arg = DeclareLaunchArgument(
        'right_id',
        description='right camera number'
    )
    front_camera = Node(
        package='point_2D_extractor', 
        executable='point_2D',        
        name='front_camera',
        output='screen',
        arguments=['--camera_view' ,0 ,'--device_name', LaunchConfiguration('front_id')]
    )

    right_camera = Node(
        package='point_2D_extractor',   
        executable='point_2D',        
        name='right_camera',
        output='screen',
        arguments=['--camera_view' ,1 ,'--device_name', LaunchConfiguration('right_id')]

    )
    
    left_camera = Node(
        package='point_2D_extractor', 
        executable='point_2D',        
        name='left_camera',
        output='screen',
        arguments=['--camera_view' ,2 ,'--device_name', LaunchConfiguration('left_id')]

    )
    voice_transcriber = Node(
        package='voice_transcriber',   
        executable='voice_transcriber',        
        name='voice_transcriber',
        output='screen'
    )
    
    sound_engine = Node(
        package='tts_system', 
        executable='tts_engine',        
        name='sound_engine',
        output='screen'
    )

    return LaunchDescription([
        front_id_arg,
        left_id_arg,
        right_id_arg,
        front_camera,
        right_camera,
        left_camera,
        voice_transcriber,
        sound_engine
    ])

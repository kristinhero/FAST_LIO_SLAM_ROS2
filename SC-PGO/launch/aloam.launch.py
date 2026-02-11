from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, EnvironmentVariable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # --- Launch arguments ---
    rviz_arg = DeclareLaunchArgument(
        'rvizscpgo',
        default_value='true',
        description='Whether to launch RViz2 for Scan-Context PGO'
    )

    # --- Package paths ---
    aloam_pkg_dir = get_package_share_directory('aloam_velodyne')
    rviz_config_file = PathJoinSubstitution([aloam_pkg_dir, 'rviz_cfg', 'aloam_velodyne.rviz'])

    # --- Parameters for Scan-Context PGO ---
    params = {
    'scan_line': 4,
    'minimum_range': 0.5,
    'mapping_line_resolution': 0.8,
    'mapping_plane_resolution': 1.0,
    'mapviz_filter_size': 0.05,
    'keyframe_meter_gap': 0.5,
    'sc_dist_thres': 0.3,
    'sc_max_radius': 80.0,
    'lidar_type': 'MID-360',
    'save_directory': '/home/aidit/ws_slam/src/FAST_LIO_SLAM_ROS2/SC-PGO/PCD/'
}

    # --- Remappings from FAST-LIO outputs ---
    remappings = [
        ('/aft_mapped_to_init', '/Odometry'),
        ('/velodyne_cloud_registered_local', '/cloud_registered_body'),
        ('/cloud_for_scancontext', '/cloud_registered_lidar')
    ]

    # --- Scan-Context PGO node ---
    alaser_pgo_node = Node(
        package='aloam_velodyne',
        executable='alaserPGO',
        name='alaserPGO',
        output='screen',
        parameters=[params],
        remappings=remappings
    )

    # --- RViz node ---
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rvizscpgo',
        output='screen',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(LaunchConfiguration('rvizscpgo'))
    )

    return LaunchDescription([
        rviz_arg,
        alaser_pgo_node,
        rviz_node
    ])

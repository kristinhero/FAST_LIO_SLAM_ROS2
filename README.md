# FAST_LIO_SLAM

- Original FAST_LIO_SLAM at [FAST_LIO_SLAM](https://github.com/gisbi-kim/FAST_LIO_SLAM)
- This repository is forked from [FAST_LIO_SLAM_ros2](https://github.com/rohrschacht/FAST_LIO_SLAM_ros2) for porting to ROS2
- FAST_LIO_ROS2 code is copied from [Ericsii/FAST_LIO_ROS2](https://github.com/Ericsii/FAST_LIO_ROS2) for porting to ROS2 and changing from livox_ros_driver to livox_ros_driver2

## What is FAST_LIO_SLAM?
Integration of 
1. [FAST-LIO2](https://github.com/hku-mars/FAST_LIO) (Odometry): A computationally efficient and robust LiDAR-inertial odometry (LIO) package
2. [SC-PGO](https://github.com/gisbi-kim/SC-A-LOAM) (Loop detection and Pose-graph Optimization): [Scan Context](https://github.com/irapkaist/scancontext)-based Loop detection and GTSAM-based Pose-graph optimization

## Features
- An easy-to-use plug-and-play LiDAR SLAM 
    - FAST-LIO2 and SC-PGO run separately (see below How to use? tab).
    - SC-PGO takes odometry and lidar point cloud topics from the FAST-LIO2 node. 
    - Finally, an optimized map is made within the SC-PGO node. 

## Dependencies
### **Ubuntu** and **ROS**
**Ubuntu >= 20.04**
This repository is tested on Ubuntu 22.04 (Jammy)

ROS >= Foxy (Recommend to use ROS-Humble). [ROS Installation](https://docs.ros.org/en/humble/Installation.html)

### **PCL && Eigen**
PCL    >= 1.8,   Follow [PCL Installation](https://pointclouds.org/downloads/#linux).

Eigen  >= 3.3.4, Follow [Eigen Installation](http://eigen.tuxfamily.org/index.php?title=Main_Page).

### **livox_ros_driver2**
Follow [livox_ros_driver2 Installation](https://github.com/Livox-SDK/livox_ros_driver2).

*Remarks:*
- Since the FAST-LIO must support Livox serials LiDAR firstly, so the **livox_ros_driver2** must be installed and **sourced** before running any FAST-LIO launch file.
### Ceres Solver and GTSAM for Scan-Context Pose Graph Optimization
Important for SC-PGO

## How to use?
### Clone the repository and colcon build:

```bash
    cd <ros2_ws>/src # cd into a ros2 workspace folder
    git clone https://github.com/kristinhero/FAST_LIO_SLAM_ROS2.git
    cd ..
    source ~/ws_livox/install/setup.bash
    rosdep install --from-paths src --ignore-src -y
    colcon build --symlink-install
```
- **Remember to source the livox_ros_driver before build**
### Run with ROS bag
#### Run IMU-converter for Livox MID-360
```bash
    cd <ros2_ws>
    source install/setup.bash
    ros2 run imu_unit_converter imu_unit_converter
```
#### Run FAST-LIO in a separate terminal
```bash
    cd <ros2_ws>
    source ~/ws_livox/install/setup.bash
    . ./install/setup.bash
    ros2 launch fast_lio mapping.launch.py config_file:=mid360.yaml
```
#### Play ROS bag with Livox Data
```bash
    cd <bag_folder>
    ros2 bag play <your_bag>.mcap --read-aead-queue-size 5000
```
Note: FAST-LIO is sensitive to jittery messages, so it is important that the message queue is not starved

Important: The current configuration file mid360.yaml is compatible with Livox ROS Driver 2 message type 0, Pointcloud2 messages of type PointXYZRTLT. If the custom Livox message is used, it should be compatible with lidar type 1 but this has not been tested in this repo. 

## Utility
- We support keyframe scan saver (as in .pcd) and provide a script reconstructs a point cloud map by merging the saved scans using the optimized poses. See [here](https://github.com/gisbi-kim/FAST_LIO_SLAM/blob/bf975560741c425f71811c864af5d35aa880c797/SC-PGO/utils/python/makeMergedMap.py#L7).

## Example results 
- [Tutorial video 1](https://youtu.be/nu8j4yaBMnw) (using KAIST 03 sequence of [MulRan dataset](https://sites.google.com/view/mulran-pr/dataset))
    - Example result captures 
        <p align="center"><img src="docs/kaist03.png" width=700></p>
    - [download the KAIST03 pcd map](https://www.dropbox.com/s/w599ozdg7h6215q/KAIST03.pcd?dl=0) made by FAST-LIO-SLAM, 500MB
    
- [Example Video 2](https://youtu.be/94mC05PesvQ) (Riverside 02 sequence of [MulRan dataset](https://sites.google.com/view/mulran-pr/dataset))
    - Example result captures
        <p align="center"><img src="docs/riverside02.png" width=700></p>
    -  [download the Riverisde02 pcd map](https://www.dropbox.com/s/1aolth7ry4odxo4/Riverside02.pcd?dl=0) made by FAST-LIO-SLAM, 400MB

## Acknowledgements 
- Thanks for [FAST_LIO](https://github.com/hku-mars/FAST_LIO) authors.
- You may have an interest in [this version of FAST-LIO + Loop closure](https://github.com/yanliang-wang/FAST_LIO_LC), implemented by [yanliang-wang](https://github.com/yanliang-wang)



#!/bin/bash

set -e
mkdir -p /workspaces/CoVAPSy-SETI-2025/ros2_ws/src
rm -rf /workspaces/CoVAPSy-SETI-2025/ros2_ws/src/tt02_driver
ln -s /workspaces/CoVAPSy-SETI-2025/Software/node_ros_2/tt02_driver /workspaces/CoVAPSy-SETI-2025/ros2_ws/src/tt02_driver
ls -l /workspaces/CoVAPSy-SETI-2025/ros2_ws/src
cd /workspaces/CoVAPSy-SETI-2025/ros2_ws
source /opt/ros/humble/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
echo "source /workspaces/CoVAPSy-SETI-2025/ros2_ws/install/setup.bash" >> /home/ros/.bashrc

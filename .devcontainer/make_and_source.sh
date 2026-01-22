#!/bin/bash

# Test if being sourced or executed
(return 0 2>/dev/null) && sourced=1 || sourced=0
if [ $sourced -eq 0 ]; then
    echo "This script must be sourced. Use 'source make_and_source.sh' or '. make_and_source.sh'"
    exit 1
fi

cd /workspaces/CoVAPSy-SETI-2025/ros2_ws
colcon build --symlink-install
source install/setup.bash
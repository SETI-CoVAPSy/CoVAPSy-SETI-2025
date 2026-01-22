FROM osrf/ros:humble-desktop-full

# Layers
# =========================================================================
#  Ubuntu configuration
# =========================================================================
# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
# Timezone
ENV TZ=Europe/Paris
# Keyboard layout (French
ENV DEBCONF_LOCALE=fr_FR.UTF-8
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# =========================================================================
#  Basic System Utilities
# =========================================================================
# We install software-properties-common to manage repositories
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    python3-pip \
    vim \
    htop \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# =========================================================================
#  ROS Configuration
# =========================================================================
# Update rosdep to speed up container creation
RUN rosdep update

# =========================================================================
#  Webots
# =========================================================================
# We add the Cyberbotics repository and install the latest Webots
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://cyberbotics.com/Cyberbotics.asc | tee /etc/apt/keyrings/Cyberbotics.asc && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/Cyberbotics.asc] https://cyberbotics.com/debian binary-amd64/" | tee /etc/apt/sources.list.d/Cyberbotics.list && \
    apt-get update && apt-get install -y \
    webots \
    ros-humble-webots-ros2 \
    && rm -rf /var/lib/apt/lists/*

# =========================================================================
#  Python Large Libraries
# =========================================================================
RUN pip3 install \
    stable-baselines3 \
    gymnasium
    # tensorboard \
    # pytest

# =========================================================================
#  Python Other Libraries / Packages
# =========================================================================
# RUN pip3 install
RUN sudo apt update && sudo apt install ros-humble-slam-toolbox -y

# =========================================================================
# Final Setup
# =========================================================================
# Add ROS 2 to the bashrc so it sources automatically every time you open a terminal
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
# Set the default shell to bash
SHELL ["/bin/bash", "-c"]
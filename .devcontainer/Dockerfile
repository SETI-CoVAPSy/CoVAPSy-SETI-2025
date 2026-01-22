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
    sudo \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# =========================================================================
#  ROS Configuration
# =========================================================================

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
    "numpy<2.0" \
    stable-baselines3 \
    gymnasium
    # tensorboard \
    # pytest

# =========================================================================
#  Python Other Libraries / Packages
# =========================================================================
# RUN pip3 install
RUN apt update \
    && apt install -y ros-humble-slam-toolbox

# =========================================================================
#  Create Non-Root User
# =========================================================================
# Create a user 'ros' with sudo privileges
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# =========================================================================
# Final Setup
# =========================================================================
# Add ROS 2 to the bashrc for both root and ros user
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
    && echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "export WEBOTS_HOME=/usr/local/webots" >> /home/$USERNAME/.bashrc

# Switch to non-root user
USER $USERNAME
RUN echo "export USER=$USERNAME" >> /home/$USERNAME/.bashrc

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]
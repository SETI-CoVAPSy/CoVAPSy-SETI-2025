# Docker setup instructions

This project makes use of a development container (devcontainer) for consistent development environments. The devcontainer is defined in the `.devcontainer` folder.

## Prerequisites

- Operating System

    Ideally, a Linux-based operating system is recommended for plug-and-play compatibility with GUI passthrough.

- Install `Docker` 
    ```bash
    sudo apt install docker.io
    ```

- Add the user to the `docker` group
    ```bash
    sudo usermod -aG docker $USER
    ```
- Log out and log back in to apply the group changes.
- Install `Docker buildx` plugin
    ```bash
    sudo apt install docker-buildx
    docker buildx install
    ```

## Use the container

### Using VS Code (recommended)
Ideally, use VS Code with the [`Dev Containers`]([https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)) extension to open the project folder in the devcontainer environment directly.

- Install the `Dev Containers` extension in VS Code.
- Open the project folder in VS Code on host.
- A prompt should appear to reopen the folder in the container. Please accept it.
<img width="684" height="165" alt="Screenshot_20260122_233349" src="https://github.com/user-attachments/assets/f9a61406-0582-4010-8845-b680db67e710" />

When the container is being opened for the first time, it will build the Docker image as per the `Dockerfile` defined in the `.devcontainer` folder. One may show the logs during the build process.

### Without VS Code (not tested)

- Navigate to the project root directory
    ```bash
    cd /path/to/project/
    ```
- Build the Docker image
    ```bash
    docker build -f .devcontainer/Dockerfile -t CoVAPSy-SETI-2025 .
    ```


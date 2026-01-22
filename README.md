# CoVAPSy SETI 2025

Main repository for the SETI 2025 [CoVAPSy](https://github.com/ajuton-ens/CourseVoituresAutonomesSaclay) team.

## Clone this repository

To clone this repository with all its submodules, use the following command:

```bash
git clone --recurse-submodules https://github.com/SETI-CoVAPSy/CoVAPSy-SETI-2025.git
```

## Repository content

This repository contains several main folders:
- [`Software/`](Software/): Code for the car
- [`Webots/`](Webots/): Webots simulation files (worlds, based on CoVAPSy's webot worlds)

Some additional bash scripts are provided at the root of this repository:
- `webots_start.sh`: Script to launch Webots with the CoVAPSy SETI 2025 world.
- `make_and_source.sh`: Script to build the ros2 workspace and source the setup file.


This main repository links to sub repositories:
- [`CoVAPSy SETI before 2025/`](https://github.com/SETI-CoVAPSy/CoVAPSy-SETI-before-2025): Unpublished code and references by previous SETI teams for CoVAPSy.


## Install the docker environment

This project is designed to be used with Docker. Please refer to the [Docker installation instructions](./devcontainer/README.md) to set up the development container.
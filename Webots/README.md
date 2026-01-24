# CoVAPSy-SETI-2025 Webots Files

This folder contains Webots-related files for the CoVAPSy SETI 2025 project, including the track generator and simulation worlds.

## Folder content

- [`Webots_SETI/`](./Webots_SETI/): Main Webots world files for the CoVAPSy SETI 2025 project.
- [`Webots_SETI_gen/`](./Webots_SETI_gen/): Generated worlds
- Files related to track generation in current folder.

Please note that the worlds are all based on the 2025b [CoVAPSy](https://github.com/ajuton-ens/CourseVoituresAutonomesSaclay) worlds provided  [here](https://github.com/ajuton-ens/CourseVoituresAutonomesSaclay/blob/main/Simulator/Simulateur_CoVAPSy_Webots2025a_Base_v2.zip).

## Track Generation

A procedural track generator is defined in the current folder. It can be used to generate race tracks to populate a Webots world.

### Content
- [`main_gen_world.py`](./main_gen_world.py): Main script to run for generating a world with a procedural track.
- [`lib_track_generator.py`](./lib_track_generator.py): Module defining the track generation logic and components
- [`lib_webots_world.py`](./lib_webots_world.py): Module to write Webots world files.
- [`track_template_build_test.png`](./track_template_build_test.png): Image defining a track used for testing how 3D components are placed.

### Usage
To generate a new world, run the [`main_gen_world.py`](./main_gen_world.py) script:
```bash
python3 main_gen_world.py
```

This will create a new Webots world file with a generated track in the `Webots_SETI_gen/worlds/` folder. You can customize the generation parameters by modifying the script.

### Configuration
In `main_gen_world.py`, one may change the `User parameters` section to adjust the track generation settings.

### About the generator

The track generation procedure's core principle was inspired by the maze generating algorithm described in [A new maze algorithm optimized for Redstone](https://www.youtube.com/watch?v=o7OhjEqCvSo). The key idea is to start from a valid track and iteratively modify it while ensuring it remains valid.

The proposed method allows generating tracks aligned to a grid.

Here are the main steps of the algorithm:
1. **Initial Track Creation**: Start with a simple valid track layout. (rectangular loop)
2. **Iterative Modification**: Randomly rotate/flip the grid, and randomly select a cell in the grid and attempt to apply one defined _operation_ (that keeps local connectivity) to modify the track.
3. **Create Webots world**: Convert encoded grid to Webots world format and save it.

Why randomly rotate/flip the grid? This allows defining operations for one specific orientation only, here assuming begin facing east.

### Main limitations
- The generated tracks are aligned to a grid, which limits the class of possible tracks.
- The road width is constant.
- Only 2 operations are defined, which gives satisfactory variety for our needs but limits the class of possible tracks.
- High number of objects: there is currently no optimization to reduce the number of 3D objects used to build the track (e.g., merging consecutive straight segments into a single object).


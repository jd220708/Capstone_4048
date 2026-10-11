# UAV Target Tracking — Capstone Simulation

Autonomous UAV tracking of a moving target in Gazebo, using a YOLO detector,
gimbal control, geometric position estimation and a standoff follower.

## Layout
- `scripts/` — ROS 2 nodes and training tools
- `sim/` — server.config, world, gimbal model, plugin sources, city mesh
- `model/` — trained YOLO11n weights and training results
- `docs/` — run sheet with all launch commands and tuned settings
- `logs/` — sample flight log

## Before running
1. **Fix the mesh path.** Edit the `<uri>` in `sim/worlds/city_test.sdf` to the
   absolute path of `sim/city_mesh/scene.gltf` on your machine.
2. **Build the plugins** — build folders are not included. Run `cd sim/plugins/sphere_system && mkdir build && cd build && cmake .. && make`.
   Same for `camera_zoom_system`.
3. Follow `docs/capstone_final_run_sheet.pdf` for setup and launch order.

## Requirements
Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic, PX4-Autopilot, NVIDIA GPU recommended.

## Not included
The 647-frame training dataset (~600 MB) is too large for the repo.
Regenerate it with `scripts/capture_frames.py` and `scripts/autolabel_spheres.py`.

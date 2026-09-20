Simulation_City(Copy) contains all code relevant to the simulation world and plugins.

Gimbal contains the modified PX4 gimballed camera used in the simulation, the "gimbal" folder in ~/PX4-Autopilot/Tools/simulation/gz/models should be replaced with this.

server.config contains the configuration of plugins used in the simulation the original found in ~/PX4-Autopilot/src/modules/simulation/gz_bridge must be replaced for the plugins to work properly. In this config file, the paths for the spheres and their speeds + spawn intervals are defined and can be created. 

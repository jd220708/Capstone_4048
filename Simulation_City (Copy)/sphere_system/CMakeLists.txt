cmake_minimum_required(VERSION 3.16)

project(SphereNetworkSystem)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# ============================================================
# Gazebo Harmonic dependencies
# ============================================================

find_package(gz-sim8 REQUIRED)

find_package(
  gz-plugin2
  REQUIRED
  COMPONENTS register
)

find_package(gz-math7 REQUIRED)

find_package(sdformat14 REQUIRED)


# ============================================================
# Build Gazebo System plugin
# ============================================================

add_library(
  SphereNetworkSystem
  SHARED
  SphereNetworkSystem.cc
)


target_link_libraries(
  SphereNetworkSystem

  PRIVATE
  gz-sim8::gz-sim8
  gz-plugin2::gz-plugin2
  gz-math7::gz-math7
  sdformat14::sdformat14
)


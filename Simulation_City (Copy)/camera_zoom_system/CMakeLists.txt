cmake_minimum_required(VERSION 3.16)

project(CameraZoomSystem)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-rendering8 REQUIRED)
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)

add_library(
  CameraZoomSystem
  SHARED
  CameraZoomSystem.cc
)

target_link_libraries(
  CameraZoomSystem
  PRIVATE
  gz-sim8::gz-sim8
  gz-plugin2::gz-plugin2
  gz-rendering8::gz-rendering8
  gz-transport13::gz-transport13
  gz-msgs10::gz-msgs10
)


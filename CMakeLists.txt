cmake_minimum_required(VERSION 3.16)
project(starlab_project LANGUAGES CXX CUDA)

############################## Project Settings ##############################

# CMake
set(CMAKE_CXX_STANDARD 14)

# Name
set(KF_NAME kf)

# Package
find_package(Eigen3 REQUIRED) # Eigen (EIGEN3_INCLUDE_DIRS)
find_package(OpenCV REQUIRED) # OpenCV (OpenCV_INCLUDE_DIRS, OpenCV_LIBRARIES)

############################## Kinect Fusion Library ##############################

# Eigen include
list(APPEND KF_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})

# OpenCV link
list(APPEND KF_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
list(APPEND KF_LIBRARIES ${OpenCV_LIBRARIES})

# Source files
file(GLOB KF_SOURCES ${CMAKE_SOURCE_DIR}/slam/kf/*.cu)

# Library
include_directories(${KF_INCLUDE_DIRS})
add_library(${KF_NAME} SHARED ${KF_SOURCES})
target_link_libraries(${KF_NAME} ${KF_LIBRARIES})

# Directory
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/slam/build/)
set_target_properties(${KF_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/slam/build/)

# Test program
add_executable(${KF_NAME}_test ${CMAKE_SOURCE_DIR}/slam/kf/kf_test.cu)
target_link_libraries(${KF_NAME}_test ${KF_NAME})

# CMake Version 
CMAKE_MINIMUM_REQUIRED (VERSION 3.8)

# Project Name
PROJECT(Deep-Vision-X)

#
# Configuration
#
# SET(CMAKE_VERBOSE_MAKEFILE ON)
#

option(USE_OPENCV       "Use OpenCV"  ON)
option(USE_CUDA         "Use CUDA"    OFF)
option(USE_RELEASE_MODE "Use CUDA"    ON)

#
# Add Compiler Option and Macro
#
if (USE_RELEASE_MODE)
  set (CMAKE_CXX_FLAGS "-fPIC -Ofast")
endif()

# Set Build Directories
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

SET(DEEP_VISION_X_SW_PLATFORM ${CMAKE_CURRENT_SOURCE_DIR}/dependency)

#
# Make Libraries
#
#
# Make Libraries
#
ADD_SUBDIRECTORY(dependency/darknet-cpp)
ADD_SUBDIRECTORY(app/player)

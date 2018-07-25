cmake_minimum_required(VERSION 3.8)

# options below defines at deep-vision-x-player/CMakeLists.txt

# option(USE_OPENCV "Use OpenCV"  ON)
# option(USE_CUDA   "Use CUDA"    ON)

if (NOT USE_CUDA)
  project(DeepVisionXPlayer)
else()
  project(DeepVisionXPlayer LANGUAGES CXX CUDA)
endif()

include_directories( 
  ${DEEP_VISION_X_SW_PLATFORM} 
)

##################
# Source
##################
file(GLOB SRC_FILES 
  src/*.cpp
)

add_executable (${PROJECT_NAME}
  ${SRC_FILES}
)

###########
# for CUDA
###########
if (USE_CUDA)
  find_package(CUDA REQUIRED)
  
  if (NOT CUDA_FOUND)
    message(STATUS "CUDA not found")
    set(USE_CUDA OFF)
  else()
    add_definitions(-DGPU)
  
    include_directories( ${CUDA_INCLUDE_DIRS} )
  endif()
endif()

###########
# OpenCV
###########
if (USE_OPENCV)
  find_package( OpenCV REQUIRED )

  if (NOT OpenCV_FOUND)
    message(STATUS "OpenCV not found")
    set(USE_OPENCV OFF)
  else()
    message(STATUS "OpenCV found")

    add_definitions(-DOPENCV)

    include_directories( ${OpenCV_INCLUDE_DIRS} )
  endif()
endif()

TARGET_LINK_LIBRARIES(${PROJECT_NAME}
  pthread
  darknet-cpp
)

if (USE_OPENCV)
  target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBS} 
  )
endif()

if (USE_CUDA)
  target_link_libraries(${PROJECT_NAME} 
    ${CUDA_curand_LIBRARY}
    ${CUDA_CUBLAS_LIBRARIES}
  )
endif()

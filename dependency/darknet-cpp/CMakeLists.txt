cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# options below defines at deep-vision-x-player/CMakeLists.txt

# option(USE_OPENCV "Use OpenCV"  ON)
# option(USE_CUDA   "Use CUDA"    ON)

if (NOT USE_CUDA)
  project(darknet-cpp)
else()
  project(darknet-cpp LANGUAGES CXX CUDA)
endif()

file(GLOB YOLO_SOURCE_FILES src/*.cpp)
file(GLOB CUDA_SOURCE_FILES src/*.cu)
file(GLOB EXAMPLE_SOURCE_FILES example/*.cpp)

include_directories( ./include )

## for CUDA
if (USE_CUDA)
  find_package(CUDA QUIET REQUIRED)
  
  if (NOT CUDA_FOUND)
    message(STATUS "CUDA not found")
    set(USE_CUDA OFF)
  else() 
    add_definitions(-DGPU)
    include_directories( ${CUDA_INCLUDE_DIRS} )

    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_WARNING "cross-execution-space-call")
  endif()
endif()

## for OpenCV
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

add_executable (darknet
  ${EXAMPLE_SOURCE_FILES}
)

add_library(${PROJECT_NAME} STATIC
  ${YOLO_SOURCE_FILES}
)

target_link_libraries(darknet
  pthread
  ${PROJECT_NAME}
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

if (USE_OPENCV)
  target_link_libraries(darknet
    ${OpenCV_LIBS} 
  )
endif()

if (USE_CUDA)
  target_link_libraries(darknet 
    ${CUDA_curand_LIBRARY}
    ${CUDA_CUBLAS_LIBRARIES}
  )

  add_library(${PROJECT_NAME}_cuda STATIC
    ${CUDA_SOURCE_FILES}
  )

  target_compile_features(${PROJECT_NAME}_cuda PUBLIC cxx_std_11)
  target_compile_definitions(${PROJECT_NAME}_cuda PUBLIC DKGPU)  

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror ${CUDA_WARNING} -restrict")
  
  #
  # You have to configure following setting to meet your GPU environment!
  #

  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61")
  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_52,code=sm_52")
  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50")
  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_35,code=sm_35")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_30,code=sm_30")

  target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_cuda)
endif()

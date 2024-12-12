cmake_minimum_required(VERSION 3.29)
project(build_basic LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

include_directories(.)


## 输出 TensorRT 配置路径
#message(STATUS "TensorRT root directory: ${TensorRT_ROOT}")
#message(STATUS "TensorRT include directory: ${TensorRT_INCLUDE_DIR}")
#message(STATUS "TensorRT library directory: ${TensorRT_LIBRARIES}")
#
## 设置包含目录
#include_directories(/usr/local/cuda/include)
#include_directories(${CMAKE_SOURCE_DIR}/utils)
#include_directories(${TensorRT_INCLUDE_DIR})
#include_directories(${TensorRT_SAMPLE_DIR}/common)
set(TensorRT_INCLUDE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include")
set(TensorRT_SAMPLE_COMMON "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/samples/common")
include_directories(${TensorRT_SAMPLE_COMMON})
include_directories(${TensorRT_INCLUDE})


# 添加可执行文件
add_executable(build
    src/cpp/build.cpp
    ${TensorRT_SAMPLE_COMMON}/logger.cpp
    ${TensorRT_SAMPLE_COMMON}/sampleUtils.cpp
)

file(GLOB TensorRT_LIBRARIES "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64/*.lib")
target_link_libraries(build PRIVATE
    ${TensorRT_LIBRARIES}
    stdc++fs
)

set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "61;70;75")
















# 项目基本信息
cmake_minimum_required(VERSION 3.29)
project(builde_infer LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

include_directories(.)

# 必要的函数库
#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
#find_package(OpenCV REQUIRED)
#find_package(TensorRT REQUIRED)

# 输出配置路径
#message(STATUS "TensorRT root directory:    ${TensorRT_ROOT}")
#message(STATUS "TensorRT library directory: ${TensorRT_LIBRARIES}")
#message(STATUS "TensorRT include directory: ${TensorRT_INCLUDE_DIR}")

# 头文件
set(TensorRT_INCLUDE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include")
set(TensorRT_SAMPLE_COMMON "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/samples/common")
include_directories(${TensorRT_SAMPLE_COMMON})
include_directories(${TensorRT_INCLUDE})
include_directories(include)
#set(OpenCV_INCLUDE_DIRS "D:/App/opencv/build/include")
#include_directories(${OpenCV_INCLUDE_DIRS})
set(OpenCV_DIR D:/App/opencv490/opencv/build)
find_package(OpenCV REQUIRED)

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# 添加可执行文件
add_executable(main
    src/cpp/main.cpp
    src/cpp/model.cpp
    src/cpp/utils.cpp
    src/cpp/timer.cpp
)


# 链接库
file(GLOB TensorRT_LIBRARIES "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64/*.lib")
#message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
target_link_libraries(main
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
#    /usr/local/cuda/lib64/libcudart.so  # 添加 CUDA 运行时库
#    stdc++fs
)


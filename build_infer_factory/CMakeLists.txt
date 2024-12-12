# 项目基本信息
cmake_minimum_required(VERSION 3.29)
project(build_infer_factory LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

include_directories(.)

# 必要的函数库
#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
#find_package(OpenCV REQUIRED)
#find_package(TensorRT REQUIRED)
#
## 输出配置路径
#message(STATUS "TensorRT root directory:    ${TensorRT_ROOT}")
#message(STATUS "TensorRT library directory: ${TensorRT_LIBRARIES}")
#message(STATUS "TensorRT include directory: ${TensorRT_INCLUDE_DIR}")

# 头文件
set(TensorRT_INCLUDE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include")
include_directories(${TensorRT_INCLUDE})
#message(STATUS "TensorRT_INCLUDE:  ${TensorRT_INCLUDE}")
#message(STATUS "TensorRT_LIBRARIES:  ${TensorRT_LIBRARIES}")
include_directories(include)
#include_directories(${OpenCV_INCLUDE_DIRS})
set(OpenCV_DIR D:/App/opencv490/opencv/build)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_LIBS:  ${OpenCV_LIBS}")

#add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
#add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# 添加可执行文件
add_executable(main
    src/cpp/main.cpp
    src/cpp/utils.cpp
    src/cpp/trt_model.cpp
    src/cpp/trt_timer.cpp
    src/cpp/trt_logger.cpp
    src/cpp/trt_worker.cpp
    src/cpp/trt_preprocess.cu
    src/cpp/trt_preprocess.cpp
    src/cpp/trt_calibrator.cpp
    src/cpp/trt_classifier.cpp
)

# 链接库
file(GLOB TensorRT_LIBRARIES "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64/*.lib")
target_link_libraries(main
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
#    /usr/local/cuda/lib64/libcudart.so  # 添加 CUDA 运行时库
#    stdc++fs
)
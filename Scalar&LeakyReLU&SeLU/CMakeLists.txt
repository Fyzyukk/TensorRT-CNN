cmake_minimum_required(VERSION 3.29)
project(builde_infer LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

include_directories(.)
set(TensorRT_INCLUDE_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include")
set(TensorRT_SAMPLE_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/samples")
include_directories(${TensorRT_SAMPLE_DIR})
include_directories(${TensorRT_INCLUDE_DIR})
set(OpenCV_DIR D:/App/opencv490/opencv/build)
find_package(OpenCV REQUIRED)

# 添加自定义插件库
add_library(customPlugin SHARED
    src/cpp/model.cpp
    src/cpp/utils.cpp
    src/cpp/customSELU_kernel.cu
    src/cpp/customSELU_plugin.cpp
    src/cpp/customScalar_kernel.cu
    src/cpp/customScalar_plugin.cpp
    src/cpp/customLeakyReLU_kernel.cu
    src/cpp/customLeakyReLU_plugin.cpp
)

# 设置头文件目录
target_include_directories(customPlugin PUBLIC "src/cpp")
target_include_directories(customPlugin PUBLIC ${OpenCV_INCLUDE_DIR})
target_include_directories(customPlugin PUBLIC ${TensorRT_INCLUDE_DIR})

# 链接 TensorRT 和 OpenCV 库
file(GLOB TensorRT_LIBRARIES "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64/*.lib")
target_link_libraries(customPlugin PUBLIC
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
)

# 链接 CUDA 库
target_link_libraries(customPlugin PRIVATE
    ${CUDA_LIBRARIES}
)

set_target_properties(customPlugin PROPERTIES CUDA_ARCHITECTURES "61;70;75")

# 添加可执行文件
add_executable(main
    src/cpp/main.cpp
    ${TensorRT_SAMPLE_DIR}/common/logger.cpp
    ${TensorRT_SAMPLE_DIR}/common/sampleUtils.cpp
)
target_include_directories(main PRIVATE ${TensorRT_INCLUDE_DIR} ${TensorRT_SAMPLE_DIR}/common)

# 链接自定义插件库
target_link_libraries(main PRIVATE
    customPlugin
    ${CUDA_LIBRARIES}
)

set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "61;70;75")
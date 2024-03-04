cmake_minimum_required(VERSION 2.8.12)

PROJECT(faceswap)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加你的可执行文件
add_executable(faceswap faceswap.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(faceswap ${OpenCV_LIBS})
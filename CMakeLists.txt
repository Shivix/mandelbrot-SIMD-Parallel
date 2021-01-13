cmake_minimum_required(VERSION 3.17)
project(mandelbrot_cv)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-march=native -pthread")

add_compile_options(-Wall -Wpedantic -Wextra)

find_package(OpenCV REQUIRED)
include_directories(version2)

add_executable(mandelbrot_cv main.cpp)

target_link_libraries(mandelbrot_cv ${OpenCV_LIBS})
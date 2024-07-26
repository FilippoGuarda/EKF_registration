cmake_minimum_required(VERSION 3.10)
project(ukf_cpp)


add_subdirectory(thirdparty/pybind11)
add_subdirectory(thirdparty/eigen)
include_directories(thirdparty/eigen)

pybind11_add_module(ukf_cpp wrappers.cpp ukf.h)



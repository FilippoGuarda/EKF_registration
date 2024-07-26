cmake_minimum_required(VERSION 3.10)
project(ekf_code)


add_subdirectory(thirdparty/pybind11)
include_directories(thirdparty/eigen)

pybind11_add_module(ukf_cpp wrappers.cpp)



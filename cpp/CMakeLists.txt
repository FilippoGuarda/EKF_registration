cmake_minimum_required(VERSION 3.10)
project(ukf_cpp)


add_subdirectory(thirdparty/pybind11)
include_directories(thirdparty/eigen)

pybind11_add_module(ukf_cpy wrappers.cpp ukf.cpp ukf.h)
target_link_libraries(ukf_cpy PRIVATE ${MY_LIBRARIES})
add_executable(ukf main.cpp ukf.cpp ukf.h)



#include <pybind11/pybind11.h>
#include <cstdio>

#include "ukf.h"

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(ekf_cpp, m) {
    m.doc() = "C++ implementation of an unscented kalman filter"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}
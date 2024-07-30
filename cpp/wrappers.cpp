#include <pybind11/pybind11.h>
#include <cstdio>
#include "ukf.h"



int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

// Your UKF class definition and other necessary includes go here

PYBIND11_MODULE(ukf_cpp, m) {
    m.doc() = "C++ implementation of an unscented kalman filter";

    m.def("add", &add, "A function that adds two numbers");

    py::class_<UKF>(m, "UKF")
        .def(py::init<int, int>())
        .def("predict", &UKF::predict)
        .def("update", &UKF::update);
}


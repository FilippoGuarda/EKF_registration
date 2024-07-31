#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdio>
#include "ukf.h"



int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

// Your UKF class definition and other necessary includes go here

PYBIND11_MODULE(ukf_cpy, m) {
    m.doc() = "C++ implementation of an unscented kalman filter";

    m.def("add", &add, "A function that adds two numbers");

    py::class_<UKF>(m, "UKF")
        .def(py::init<int, int, Eigen::VectorXd, Eigen::MatrixXd>())
        .def("predict", &UKF::predict)
        .def("update", &UKF::update)
        .def_property_readonly("mean", &UKF::mean)
        .def_property_readonly("cov", &UKF::cov)
        .def("print_state", &UKF::printState);
}


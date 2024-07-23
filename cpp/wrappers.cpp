#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(ekf_cpp, m) {
    m.doc() = "C++ implementation of an extended kalman filter"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}
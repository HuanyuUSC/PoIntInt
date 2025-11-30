#include <pybind11/pybind11.h>

namespace py = pybind11;

const char* say_hello() {
    return "Hello, World!";
}

PYBIND11_MODULE(minimal, m) {
    m.doc() = "A minimal pybind11 example";
    m.def("say_hello", &say_hello, "A function that says hello");
}

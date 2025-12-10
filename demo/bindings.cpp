#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "geometry/packing.hpp"
#include "geometry/geometry_helpers.hpp"
#include "quadrature/lebedev_io.hpp"
#include "quadrature/kgrid.hpp"
#include "compute_intersection_volume.hpp"
#include "compute_volume.hpp"
#include "analytical_intersection.hpp"

namespace py = pybind11;
using namespace PoIntInt;

PYBIND11_MODULE(pointint_cpp, m) {
    m.doc() = "PoIntInt intersection volume demo bindings";

    // Geometry class (opaque wrapper)
    py::class_<Geometry>(m, "Geometry")
        .def(py::init<>())
        .def("num_elements", &Geometry::num_elements);

    // KGrid class (opaque wrapper)
    py::class_<KGrid>(m, "KGrid")
        .def(py::init<>());

    // Geometry creation functions
    m.def("make_triangle_mesh", &make_triangle_mesh,
        py::arg("V"), py::arg("F"),
        "Create triangle mesh geometry from vertices and faces");

    m.def("make_point_cloud", &make_point_cloud,
        py::arg("P"), py::arg("N"), py::arg("radii"), py::arg("is_radius") = true,
        "Create point cloud geometry from positions, normals, and radii");

    m.def("create_unit_cube_mesh", []() {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        create_unit_cube_mesh(V, F);
        return std::make_tuple(V, F);
    }, "Create unit cube mesh centered at origin [-0.5, 0.5]^3");

    m.def("create_sphere_pointcloud", [](int n) {
        Eigen::MatrixXd P, N;
        Eigen::VectorXd radii;
        create_sphere_pointcloud(P, N, radii, n);
        return std::make_tuple(P, N, radii);
    }, py::arg("n") = 1000,
    "Create unit sphere as oriented point cloud");

    // KGrid creation
    m.def("load_lebedev", [](const std::string& path) {
        LebedevGrid L = load_lebedev_txt(path);
        return std::make_tuple(L.dirs, L.weights);
    }, py::arg("path"),
    "Load Lebedev quadrature grid from file");

    m.def("build_kgrid", &build_kgrid,
        py::arg("leb_dirs"), py::arg("leb_weights"), py::arg("n_radial"),
        "Build k-grid from Lebedev directions and radial points");

    // Intersection volume computation - CPU version
    m.def("compute_intersection_volume_cpu",
        [](const Geometry& g1, const Geometry& g2, const KGrid& kg) {
            return compute_intersection_volume_cpu(g1, g2, kg, false);
        },
        py::arg("geom1"), py::arg("geom2"), py::arg("kgrid"),
        "Compute intersection volume using CPU/TBB");

    // Intersection volume computation - CUDA version
    m.def("compute_intersection_volume_cuda",
        [](const Geometry& g1, const Geometry& g2, const KGrid& kg, int block_size) {
            return compute_intersection_volume_cuda(g1, g2, kg, block_size, false);
        },
        py::arg("geom1"), py::arg("geom2"), py::arg("kgrid"), py::arg("block_size") = 256,
        "Compute intersection volume using CUDA");

    // Single geometry volume computation
    m.def("compute_volume_cpu",
        [](const Geometry& g) {
            return compute_volume_cpu(g, false);
        },
        py::arg("geom"),
        "Compute volume of a single geometry using CPU");

    // Analytical ground truth functions
    m.def("box_box_intersection_volume", &box_box_intersection_volume,
        py::arg("c1"), py::arg("h1"), py::arg("c2"), py::arg("h2"),
        "Compute analytical intersection volume of two axis-aligned boxes");

    m.def("sphere_sphere_intersection_volume", &sphere_sphere_intersection_volume,
        py::arg("c1"), py::arg("r1"), py::arg("c2"), py::arg("r2"),
        "Compute analytical intersection volume of two spheres");

    // Utility: translate points
    m.def("translate_points",
        [](const Eigen::MatrixXd& V, const Eigen::Vector3d& t) {
            return (V.rowwise() + t.transpose()).eval();
        },
        py::arg("points"), py::arg("translation"),
        "Translate points by a vector");
}

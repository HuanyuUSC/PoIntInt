#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include "geometry/geometry.hpp"
#include "geometry/packing.hpp"
#include "dof/affine_dof.hpp"
#include "dof/triangle_mesh_dof.hpp"
#include "quadrature/kgrid.hpp"
#include "quadrature/lebedev_io.hpp"
#include "compute_intersection_volume.hpp"
#include "compute_intersection_volume_multi_object.hpp"

namespace py = pybind11;
using namespace PoIntInt;

PYBIND11_MODULE(pointint_core_python, m) {
    m.doc() = "Python bindings for PoIntInt";

    // ========================================================================
    // Geometry
    // ========================================================================
    py::class_<Geometry, std::shared_ptr<Geometry>>(m, "Geometry")
        .def_readonly("type", &Geometry::type);

    m.def("make_triangle_mesh", &make_triangle_mesh, "Create a triangle mesh geometry");
    m.def("make_point_cloud", &make_point_cloud, "Create a point cloud geometry");
    m.def("make_gaussian_splat", &make_gaussian_splat, "Create a Gaussian splat geometry");

    // ========================================================================
    // Quadrature
    // ========================================================================
    py::class_<KGrid>(m, "KGrid")
        .def(py::init<>())
        .def_readwrite("dirs", &KGrid::dirs)
        .def_readwrite("kmag", &KGrid::kmag)
        .def_readwrite("w", &KGrid::w);

    py::class_<LebedevGrid>(m, "LebedevGrid")
        .def(py::init<>())
        .def_readwrite("dirs", &LebedevGrid::dirs)
        .def_readwrite("weights", &LebedevGrid::weights);

    m.def("load_lebedev_txt", &load_lebedev_txt, "Load a Lebedev grid from a text file");
    m.def("build_kgrid", &build_kgrid, "Build a K-Grid from a Lebedev grid");

    // ========================================================================
    // Degrees of Freedom (DoF)
    // ========================================================================
    py::class_<DoFParameterization, std::shared_ptr<DoFParameterization>>(m, "DoFParameterization")
        .def("num_dofs", &DoFParameterization::num_dofs);

    py::class_<AffineDoF, DoFParameterization, std::shared_ptr<AffineDoF>>(m, "AffineDoF")
        .def(py::init<>());

    py::class_<TriangleMeshDoF, DoFParameterization, std::shared_ptr<TriangleMeshDoF>>(m, "TriangleMeshDoF")
        .def(py::init<int>());

    // ========================================================================
    // Computation Results
    // ========================================================================
    py::class_<IntersectionVolumeResult>(m, "IntersectionVolumeResult")
        .def(py::init<>())
        .def_readwrite("volume", &IntersectionVolumeResult::volume)
        .def_readwrite("grad_geom1", &IntersectionVolumeResult::grad_geom1)
        .def_readwrite("grad_geom2", &IntersectionVolumeResult::grad_geom2)
        .def_readwrite("hessian_geom1", &IntersectionVolumeResult::hessian_geom1)
        .def_readwrite("hessian_geom2", &IntersectionVolumeResult::hessian_geom2)
        .def_readwrite("hessian_cross", &IntersectionVolumeResult::hessian_cross);

    py::class_<IntersectionVolumeMatrixResult>(m, "IntersectionVolumeMatrixResult")
        .def(py::init<>())
        .def_readwrite("volume_matrix", &IntersectionVolumeMatrixResult::volume_matrix);
        // grad_matrix and hessian matrices can be added if needed

    // ========================================================================
    // Computation Functions
    // ========================================================================
    m.def("compute_intersection_volume_cuda", 
          py::overload_cast<const Geometry&, const Geometry&, const KGrid&, int, bool>(&compute_intersection_volume_cuda),
          "Compute intersection volume on GPU for two geometries without DoFs",
          py::arg("geom1"), py::arg("geom2"), py::arg("kgrid"), py::arg("block_size") = 256, py::arg("profile") = false);

    m.def("compute_intersection_volume_cuda", 
          py::overload_cast<const Geometry&, const Geometry&, const std::shared_ptr<DoFParameterization>&, const std::shared_ptr<DoFParameterization>&, const Eigen::VectorXd&, const Eigen::VectorXd&, const KGrid&, ComputationFlags, int, bool>(&compute_intersection_volume_cuda),
          "Compute intersection volume and derivatives on GPU for two geometries with DoFs",
          py::arg("geom1"), py::arg("geom2"), py::arg("dof1"), py::arg("dof2"), py::arg("dofs1"), py::arg("dofs2"), py::arg("kgrid"), py::arg("flags"), py::arg("block_size") = 256, py::arg("profile") = false);
    
    m.def("compute_intersection_volume_matrix_cuda",
        py::overload_cast<const std::vector<Geometry>&, const KGrid&, int, bool>(&compute_intersection_volume_matrix_cuda),
        "Compute intersection volume matrix on GPU for multiple geometries without DoFs",
        py::arg("geometries"), py::arg("kgrid"), py::arg("block_size") = 256, py::arg("profile") = false);

    // Add CPU versions if needed
    m.def("compute_intersection_volume_cpu", 
          py::overload_cast<const Geometry&, const Geometry&, const KGrid&, bool>(&compute_intersection_volume_cpu),
          "Compute intersection volume on CPU for two geometries without DoFs",
          py::arg("geom1"), py::arg("geom2"), py::arg("kgrid"), py::arg("profile") = false);

    m.def("compute_intersection_volume_cpu", 
          py::overload_cast<const Geometry&, const Geometry&, const std::shared_ptr<DoFParameterization>&, const std::shared_ptr<DoFParameterization>&, const Eigen::VectorXd&, const Eigen::VectorXd&, const KGrid&, ComputationFlags, bool>(&compute_intersection_volume_cpu),
          "Compute intersection volume and derivatives on CPU for two geometries with DoFs",
          py::arg("geom1"), py::arg("geom2"), py::arg("dof1"), py::arg("dof2"), py::arg("dofs1"), py::arg("dofs2"), py::arg("kgrid"), py::arg("flags"), py::arg("profile") = false);

    // ========================================================================
    // Computation Flags
    // ========================================================================
    py::enum_<ComputationFlags>(m, "ComputationFlags")
        .value("VOLUME_ONLY", ComputationFlags::VOLUME_ONLY)
        .value("GRADIENT", ComputationFlags::GRADIENT)
        .value("HESSIAN", ComputationFlags::HESSIAN)
        .value("ALL", ComputationFlags::ALL)
        .export_values();
}

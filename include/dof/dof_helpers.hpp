#pragma once
#include "../geometry.hpp"
#include "../geometry_types.hpp"
#include <Eigen/Dense>
#include <complex>

namespace PoIntInt {

// Helper functions for computing A(k) on CPU (used for gradient computation)

// Compute A_parallel(k) = (k·A(k))/|k| for triangles
std::complex<double> compute_A_triangle_cpu(
  const TriPacked& tri,
  const Eigen::Vector3d& k);

// Compute A_parallel(k) for disks
std::complex<double> compute_A_disk_cpu(
  const DiskPacked& disk,
  const Eigen::Vector3d& k);

// Compute A_parallel(k) for Gaussian splats
std::complex<double> compute_A_gaussian_cpu(
  const GaussianPacked& gauss,
  const Eigen::Vector3d& k);

// Compute A_parallel(k) for entire geometry
std::complex<double> compute_A_geometry_cpu(
  const Geometry& geom,
  const Eigen::Vector3d& k);

} // namespace PoIntInt


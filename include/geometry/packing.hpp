#pragma once
#include <vector>
#include <array>
#include <Eigen/Dense>
#include "geometry/types.hpp"
#include "geometry/geometry.hpp"

namespace PoIntInt {

// Pack triangle mesh into CUDA-friendly format
std::vector<TriPacked> pack_tris(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F);

// Pack oriented point cloud: positions P, normals N, radii (or areas)
// If radii provided, area = π * rho^2. If areas provided, rho = sqrt(area/π)
std::vector<DiskPacked> pack_disks(
  const Eigen::MatrixXd& P,  // positions
  const Eigen::MatrixXd& N,  // normals (will be normalized)
  const Eigen::VectorXd& radii_or_areas,  // either radii or areas
  bool is_radius = true);    // true if radii, false if areas

// Pack Gaussian splats: positions P, normals N, sigmas (std dev in tangent plane), weights
std::vector<GaussianPacked> pack_gaussians(
  const Eigen::MatrixXd& P,  // positions
  const Eigen::MatrixXd& N,  // normals (will be normalized)
  const Eigen::VectorXd& sigmas,  // standard deviations in tangent plane
  const Eigen::VectorXd& weights); // area weights

// Factory functions to create Geometry objects
Geometry make_triangle_mesh(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F);

Geometry make_point_cloud(
  const Eigen::MatrixXd& P,  // positions
  const Eigen::MatrixXd& N,  // normals (will be normalized)
  const Eigen::VectorXd& radii_or_areas,  // either radii or areas
  bool is_radius = true);    // true if radii, false if areas

Geometry make_gaussian_splat(
  const Eigen::MatrixXd& P,  // positions
  const Eigen::MatrixXd& N,  // normals (will be normalized)
  const Eigen::VectorXd& sigmas,  // standard deviations in tangent plane
  const Eigen::VectorXd& weights); // area weights

} // namespace PoIntInt


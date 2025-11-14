#pragma once
#include <Eigen/Dense>

namespace PoIntInt {

// ============================================================================
// Geometry Creation Helpers
// ============================================================================

// Create unit cube mesh centered at origin [-1/2, 1/2]^3
void create_unit_cube_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F);

// Create unit sphere as oriented point cloud
// Uses Fibonacci sphere algorithm for uniform distribution
void create_sphere_pointcloud(
  Eigen::MatrixXd& P,  // positions
  Eigen::MatrixXd& N,  // normals (outward pointing)
  Eigen::VectorXd& radii,
  int n_points = 1000);

// Create unit sphere as Gaussian splats
// Uses Fibonacci sphere algorithm for uniform distribution
void create_sphere_gaussians(
  Eigen::MatrixXd& P,  // positions
  Eigen::MatrixXd& N,  // normals (outward pointing)
  Eigen::VectorXd& sigmas,  // standard deviations in tangent plane
  Eigen::VectorXd& weights,  // area weights
  int n_points = 1000);

} // namespace PoIntInt


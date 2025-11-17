#pragma once
#include "geometry/geometry.hpp"
#include "dof/affine_dof.hpp"
#include <Eigen/Dense>
#include <memory>

namespace PoIntInt {

// Structure to hold geometric moments
struct Moments {
  double volume;           // Zeroth moment (volume)
  Eigen::Vector3d centroid;  // First moment (centroid) = (1/V) * ∫ x dV
  Eigen::Matrix3d covariance; // Second moment (covariance) = (1/V) * ∫ (x-c)(x-c)^T dV
};

// Compute geometric moments (volume, centroid, covariance) for a geometry
// For triangle meshes, uses direct computation from vertices
// For other types, uses divergence theorem via DoF interface
Moments compute_moments(
  const Geometry& geom,
  const std::shared_ptr<AffineDoF>& affine_dof,
  const Eigen::VectorXd& dofs
);

// Compute warm start affine transform that matches moments of two geometries
// Returns: AffineDoF parameters (12-vector) that best aligns geom2 to geom1
// The transform A, t satisfies:
//   - Translation: t = c1 - A*c2  (aligns centroids)
//   - Rotation/Scaling: A*C2*A^T = C1  (matches covariance)
Eigen::VectorXd compute_warm_start(
  const Geometry& geom1,
  const Geometry& geom2,
  const std::shared_ptr<AffineDoF>& affine_dof
);

} // namespace PoIntInt


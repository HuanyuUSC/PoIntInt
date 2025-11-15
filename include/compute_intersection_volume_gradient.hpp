#pragma once
#include "geometry/geometry.hpp"
#include "dof/dof_parameterization.hpp"
#include "quadrature/kgrid.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace PoIntInt {

// Result structure for intersection volume gradient computation
struct IntersectionVolumeGradientResult {
  Eigen::VectorXd grad_geom1;  // Gradient w.r.t. DoFs of geometry 1
  Eigen::VectorXd grad_geom2;  // Gradient w.r.t. DoFs of geometry 2
  double volume;                // Intersection volume (for convenience)
};

// Compute gradient of intersection volume w.r.t. DoFs of both geometries
// Mathematical formulation:
//   ∂V/∂θ₁ = (1/(8π³)) · Σ_q w_q · Re( (∂A₁(k_q)/∂θ₁) · conj(A₂(k_q)) )
//   ∂V/∂θ₂ = (1/(8π³)) · Σ_q w_q · Re( A₁(k_q) · conj(∂A₂(k_q)/∂θ₂) )

// CUDA version
IntersectionVolumeGradientResult compute_intersection_volume_gradient_cuda(
  const Geometry& geom1,
  const Geometry& geom2,
  const std::shared_ptr<DoFParameterization>& dof1,
  const std::shared_ptr<DoFParameterization>& dof2,
  const Eigen::VectorXd& dofs1,
  const Eigen::VectorXd& dofs2,
  const KGrid& kgrid,
  int blockSize = 256,
  bool enable_profiling = false
);

// CPU version (parallelized with TBB)
IntersectionVolumeGradientResult compute_intersection_volume_gradient_cpu(
  const Geometry& geom1,
  const Geometry& geom2,
  const std::shared_ptr<DoFParameterization>& dof1,
  const std::shared_ptr<DoFParameterization>& dof2,
  const Eigen::VectorXd& dofs1,
  const Eigen::VectorXd& dofs2,
  const KGrid& kgrid,
  bool enable_profiling = false
);

// ============================================================================
// Phase 1: Test A(k) computation
// ============================================================================

// Compute A(k) for a single k-vector using CUDA (for testing)
// Only supports AffineDoF with triangle meshes
std::complex<double> compute_Ak_cuda(
  const Geometry& geom,
  const Eigen::Vector3d& k,
  const std::shared_ptr<DoFParameterization>& dof,
  const Eigen::VectorXd& dofs,
  int blockSize = 256
);

// ============================================================================
// Phase 2: Test ∂A(k)/∂θ computation
// ============================================================================

// Compute gradient of A(k) w.r.t. DoFs for a single k-vector using CUDA (for testing)
// Only supports AffineDoF with triangle meshes
Eigen::VectorXcd compute_Ak_gradient_cuda(
  const Geometry& geom,
  const Eigen::Vector3d& k,
  const std::shared_ptr<DoFParameterization>& dof,
  const Eigen::VectorXd& dofs,
  int blockSize = 256
);

} // namespace PoIntInt


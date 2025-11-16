#pragma once
#include "geometry/geometry.hpp"
#include "dof/dof_parameterization.hpp"
#include "dof/affine_dof.hpp"
#include "quadrature/kgrid.hpp"
#include "computation_flags.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace PoIntInt {

// ============================================================================
// Result Structures
// ============================================================================

// Result structure for intersection volume computation (two geometries)
struct IntersectionVolumeResult {
  double volume;  // Intersection volume
  
  // Gradients (only computed if requested)
  Eigen::VectorXd grad_geom1;  // Gradient w.r.t. DoFs of geometry 1
  Eigen::VectorXd grad_geom2;  // Gradient w.r.t. DoFs of geometry 2
  
  // Hessians (only computed if requested, Gauss-Newton approximation)
  // Note: Gauss-Newton Hessian is approximated as sum of outer products of gradients:
  //   H₁₁ ≈ Σ_q w_q · (∂A₁(k_q)/∂θ₁) · (∂A₁(k_q)/∂θ₁)^T
  //   H₂₂ ≈ Σ_q w_q · (∂A₂(k_q)/∂θ₂) · (∂A₂(k_q)/∂θ₂)^T
  //   H₁₂ ≈ Σ_q w_q · (∂A₁(k_q)/∂θ₁) · (∂A₂(k_q)/∂θ₂)^T
  // This does NOT require second derivatives d²A/dθ²
  Eigen::MatrixXd hessian_geom1;  // Hessian w.r.t. DoFs of geometry 1
  Eigen::MatrixXd hessian_geom2;  // Hessian w.r.t. DoFs of geometry 2
  Eigen::MatrixXd hessian_cross;  // Cross-term Hessian (∂²V/∂θ₁∂θ₂)
};

// ============================================================================
// Intersection Volume Interface
// ============================================================================
// Note: Self-volume computation is NOT implemented here. Self-volume should
// be computed using the divergence theorem: V = (1/3) ∫_S (x, y, z) · n dS,
// where S is the boundary surface and n is the outward normal. This requires
// DoF-specific implementations of local contributions (area weight × n · position)
// for each element. See DoFParameterization::compute_divergence_contribution()
// and related methods. This will be implemented in a future phase.
// ============================================================================

// CUDA version
IntersectionVolumeResult compute_intersection_volume_cuda(
  const Geometry& ref_geom1,
  const Geometry& ref_geom2,
  const std::shared_ptr<DoFParameterization>& dof1,
  const std::shared_ptr<DoFParameterization>& dof2,
  const Eigen::VectorXd& dofs1,
  const Eigen::VectorXd& dofs2,
  const KGrid& kgrid,
  ComputationFlags flags = ComputationFlags::VOLUME_ONLY,
  int blockSize = 256,
  bool enable_profiling = false
);

// CPU/TBB version
IntersectionVolumeResult compute_intersection_volume_cpu(
  const Geometry& ref_geom1,
  const Geometry& ref_geom2,
  const std::shared_ptr<DoFParameterization>& dof1,
  const std::shared_ptr<DoFParameterization>& dof2,
  const Eigen::VectorXd& dofs1,
  const Eigen::VectorXd& dofs2,
  const KGrid& kgrid,
  ComputationFlags flags = ComputationFlags::VOLUME_ONLY,
  bool enable_profiling = false
);

// ============================================================================
// Convenience Overloads (for backward compatibility)
// ============================================================================
// These overloads provide the old interface signature, using identity AffineDoF
// internally for cases where DoF parameterizations are not needed.

// CUDA version (old interface compatibility - returns double)
inline double compute_intersection_volume_cuda(
  const Geometry& geom1,
  const Geometry& geom2,
  const KGrid& kgrid,
  int blockSize = 256,
  bool enable_profiling = false)
{
  auto affine_dof = std::make_shared<AffineDoF>();
  Eigen::VectorXd identity_dofs(12);
  identity_dofs << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
  
  auto result = compute_intersection_volume_cuda(
    geom1, geom2, affine_dof, affine_dof, identity_dofs, identity_dofs,
    kgrid, ComputationFlags::VOLUME_ONLY, blockSize, enable_profiling);
  return result.volume;
}

// CPU version (old interface compatibility - returns double)
inline double compute_intersection_volume_cpu(
  const Geometry& geom1,
  const Geometry& geom2,
  const KGrid& kgrid,
  bool enable_profiling = false)
{
  auto affine_dof = std::make_shared<AffineDoF>();
  Eigen::VectorXd identity_dofs(12);
  identity_dofs << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
  
  auto result = compute_intersection_volume_cpu(
    geom1, geom2, affine_dof, affine_dof, identity_dofs, identity_dofs,
    kgrid, ComputationFlags::VOLUME_ONLY, enable_profiling);
  return result.volume;
}

// ============================================================================
// Helper Functions for Testing (Single k-vector A(k) and ∂A(k)/∂θ)
// ============================================================================

// Compute A(k) for a single k-vector using CUDA (for testing)
std::complex<double> compute_Ak_cuda(
  const Geometry& geom,
  const Eigen::Vector3d& k,
  const std::shared_ptr<DoFParameterization>& dof,
  const Eigen::VectorXd& dofs,
  int blockSize = 256
);

// Compute gradient of A(k) w.r.t. DoFs for a single k-vector using CUDA (for testing)
Eigen::VectorXcd compute_Ak_gradient_cuda(
  const Geometry& geom,
  const Eigen::Vector3d& k,
  const std::shared_ptr<DoFParameterization>& dof,
  const Eigen::VectorXd& dofs,
  int blockSize = 256
);

} // namespace PoIntInt


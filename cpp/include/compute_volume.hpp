#pragma once
#include "geometry/geometry.hpp"
#include "dof/dof_parameterization.hpp"
#include "computation_flags.hpp"
#include <Eigen/Dense>
#include <memory>

namespace PoIntInt {

// ============================================================================
// Result Structures
// ============================================================================

// Result structure for self-volume computation (single geometry)
struct VolumeResult {
  double volume;  // Self-volume
  
  // Gradient (only computed if requested)
  Eigen::VectorXd grad;  // Gradient w.r.t. DoFs
};

// ============================================================================
// Self-Volume Interface
// ============================================================================
// Self-volume is computed using the divergence theorem: V = (1/3) ∫_S (x, y, z) · n dS,
// where S is the boundary surface and n is the outward normal. This is implemented
// via DoFParameterization::compute_volume() and compute_volume_gradient() methods.
// ============================================================================

// CPU/TBB version (unified interface)
VolumeResult compute_volume_cpu(
  const Geometry& ref_geom,
  const std::shared_ptr<DoFParameterization>& dof,
  const Eigen::VectorXd& dofs,
  ComputationFlags flags = ComputationFlags::VOLUME_ONLY,
  bool enable_profiling = false
);

// ============================================================================
// Convenience Overloads (for backward compatibility)
// ============================================================================
// These overloads provide the old interface signature, using identity AffineDoF
// internally for cases where DoF parameterizations are not needed.

// CPU version (old interface compatibility - returns double)
double compute_volume_cpu(const Geometry& geom, bool enable_profiling = false);

} // namespace PoIntInt

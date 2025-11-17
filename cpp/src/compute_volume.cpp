#include "compute_volume.hpp"
#include "dof/affine_dof.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

namespace PoIntInt {

// ============================================================================
// Unified Interface Implementation
// ============================================================================

VolumeResult compute_volume_cpu(
  const Geometry& ref_geom,
  const std::shared_ptr<DoFParameterization>& dof,
  const Eigen::VectorXd& dofs,
  ComputationFlags flags,
  bool enable_profiling)
{
  auto t_start = high_resolution_clock::now();
  
  VolumeResult result;
  
  // Compute volume
  if (needs_volume(flags)) {
    result.volume = dof->compute_volume(ref_geom, dofs);
  } else {
    result.volume = 0.0;
  }
  
  // Compute gradient if requested
  if (needs_gradient(flags)) {
    result.grad = dof->compute_volume_gradient(ref_geom, dofs);
  }
  
  auto t_end = high_resolution_clock::now();
  
  if (enable_profiling) {
    auto total_time = duration_cast<microseconds>(t_end - t_start).count() / 1000.0;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== CPU Volume Computation (Unified Interface) ===" << std::endl;
    std::cout << "Geometry: " << ref_geom.num_elements() << " elements" << std::endl;
    std::cout << "DoF type: " << dof->num_dofs() << " DoFs" << std::endl;
    std::cout << "Flags: " << static_cast<int>(flags) << std::endl;
    std::cout << "--- Timing (ms) ---" << std::endl;
    std::cout << "  Total time: " << std::setw(8) << total_time << " ms" << std::endl;
    std::cout << "==========================================\n" << std::endl;
  }
  
  return result;
}

// ============================================================================
// Convenience Overload (Backward Compatibility)
// ============================================================================

double compute_volume_cpu(const Geometry& geom, bool enable_profiling) {
  // Create identity AffineDoF for backward compatibility
  auto identity_dof = std::make_shared<AffineDoF>();
  Eigen::VectorXd identity_dofs(12);
  identity_dofs << 0.0, 0.0, 0.0,  // translation
                   1.0, 0.0, 0.0,  // first row of identity matrix
                   0.0, 1.0, 0.0,  // second row of identity matrix
                   0.0, 0.0, 1.0;  // third row of identity matrix
  
  // Call unified interface
  VolumeResult result = compute_volume_cpu(geom, identity_dof, identity_dofs, ComputationFlags::VOLUME_ONLY, enable_profiling);
  return result.volume;
}

} // namespace PoIntInt


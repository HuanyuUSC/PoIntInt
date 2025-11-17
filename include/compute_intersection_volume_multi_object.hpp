#pragma once
#include "geometry/geometry.hpp"
#include "dof/dof_parameterization.hpp"
#include "dof/affine_dof.hpp"
#include "quadrature/kgrid.hpp"
#include "computation_flags.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace PoIntInt {

// ============================================================================
// Result Structure
// ============================================================================

// Result structure for multi-object intersection volume matrix computation
struct IntersectionVolumeMatrixResult {
  Eigen::MatrixXd volume_matrix;  // nObj × nObj symmetric matrix
  
  // Gradients (only computed if requested)
  // grad_matrix[i][j] is the gradient of V[i,j] w.r.t. DoFs of object i
  // NOTE: For diagonal entries (i == j), grad_matrix[i][i] stores the contribution
  // from a single copy of the geometry. The full gradient of V[i,i] w.r.t. object i
  // is 2 * grad_matrix[i][i]. The same applies to the Hessian blocks below.
  std::vector<std::vector<Eigen::VectorXd>> grad_matrix;  // Empty if not computed
  
  // Hessians (only computed if requested, Gauss-Newton approximation)
  // For each pair (i,j), we store three blocks:
  //   - hessian_ii[i][j]: ∂²V[i,j]/∂θ_i² (Hessian w.r.t. DoFs of object i)
  //   - hessian_jj[i][j]: ∂²V[i,j]/∂θ_j² (Hessian w.r.t. DoFs of object j)
  //   - hessian_ij[i][j]: ∂²V[i,j]/∂θ_i∂θ_j (Cross Hessian)
  // NOTE: For Gauss-Newton approximation, hessian_ii[i][j] and hessian_jj[i][j] are
  // always zero (we ignore second derivatives). Only hessian_ij[i][j] is non-zero.
  // The structure is kept for future support of full Hessian computation.
  // For diagonal entries (i == j), the full Hessian of V[i,i] w.r.t. object i 
  // is hessian_ii[i][i] + hessian_jj[i][i] + hessian_ij[i][i] + hessian_ij[i][i]^T.
  std::vector<std::vector<Eigen::MatrixXd>> hessian_ii;  // Empty if not computed, always zero for Gauss-Newton
  std::vector<std::vector<Eigen::MatrixXd>> hessian_jj;  // Empty if not computed, always zero for Gauss-Newton
  std::vector<std::vector<Eigen::MatrixXd>> hessian_ij;  // Empty if not computed, only non-zero block for Gauss-Newton
  
  // Metadata
  int num_objects() const { return (int)volume_matrix.rows(); }
};

// ============================================================================
// Unified Multi-Object Interface
// ============================================================================

// CUDA version (unified interface with DoF support)
IntersectionVolumeMatrixResult compute_intersection_volume_matrix_cuda(
  const std::vector<Geometry>& ref_geometries,
  const std::vector<std::shared_ptr<DoFParameterization>>& dofs,
  const std::vector<Eigen::VectorXd>& dof_vectors,
  const KGrid& kgrid,
  ComputationFlags flags = ComputationFlags::VOLUME_ONLY,
  int blockSize = 256,
  bool enable_profiling = false
);

// CPU/TBB version (unified interface with DoF support)
IntersectionVolumeMatrixResult compute_intersection_volume_matrix_cpu(
  const std::vector<Geometry>& ref_geometries,
  const std::vector<std::shared_ptr<DoFParameterization>>& dofs,
  const std::vector<Eigen::VectorXd>& dof_vectors,
  const KGrid& kgrid,
  ComputationFlags flags = ComputationFlags::VOLUME_ONLY,
  bool enable_profiling = false
);

// ============================================================================
// Convenience Overloads (for backward compatibility)
// ============================================================================
// These overloads provide the old interface signature, using identity AffineDoF
// internally for cases where DoF parameterizations are not needed.

// CUDA version (old interface compatibility)
inline IntersectionVolumeMatrixResult compute_intersection_volume_matrix_cuda(
  const std::vector<Geometry>& geometries,
  const KGrid& kgrid,
  int blockSize = 256,
  bool enable_profiling = false)
{
  int num_objects = (int)geometries.size();
  std::vector<std::shared_ptr<DoFParameterization>> dofs(num_objects);
  std::vector<Eigen::VectorXd> dof_vectors(num_objects);
  
  Eigen::VectorXd identity_dofs(12);
  identity_dofs << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
  
  for (int i = 0; i < num_objects; ++i) {
    dofs[i] = std::make_shared<AffineDoF>();
    dof_vectors[i] = identity_dofs;
  }
  
  return compute_intersection_volume_matrix_cuda(
    geometries, dofs, dof_vectors, kgrid, ComputationFlags::VOLUME_ONLY, blockSize, enable_profiling);
}

// CPU version (old interface compatibility)
inline IntersectionVolumeMatrixResult compute_intersection_volume_matrix_cpu(
  const std::vector<Geometry>& geometries,
  const KGrid& kgrid,
  bool enable_profiling = false)
{
  int num_objects = (int)geometries.size();
  std::vector<std::shared_ptr<DoFParameterization>> dofs(num_objects);
  std::vector<Eigen::VectorXd> dof_vectors(num_objects);
  
  Eigen::VectorXd identity_dofs(12);
  identity_dofs << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
  
  for (int i = 0; i < num_objects; ++i) {
    dofs[i] = std::make_shared<AffineDoF>();
    dof_vectors[i] = identity_dofs;
  }
  
  return compute_intersection_volume_matrix_cpu(
    geometries, dofs, dof_vectors, kgrid, ComputationFlags::VOLUME_ONLY, enable_profiling);
}

} // namespace PoIntInt


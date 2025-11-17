#include "similarity_measure.hpp"
#include "computation_flags.hpp"
#include <cmath>
#include <iostream>

namespace PoIntInt {

SimilarityResult compute_similarity(
  const Geometry& ref_geom1,
  const Geometry& ref_geom2,
  const std::shared_ptr<AffineDoF>& affine_dof,
  const Eigen::VectorXd& dofs2,
  const KGrid& kgrid,
  SimilarityMeasure measure,
  double V1,
  int flags
) {
  SimilarityResult result;
  
  // Determine what to compute for intersection volume
  bool need_gradient = (flags & SIM_GRADIENT) != 0;
  bool need_hessian = (flags & SIM_HESSIAN) != 0;
  
  ComputationFlags vol_flags = ComputationFlags::VOLUME_ONLY;
  if (need_gradient) {
    vol_flags = vol_flags | ComputationFlags::GRADIENT;
  }
  if (need_hessian) {
    vol_flags = vol_flags | ComputationFlags::HESSIAN;
  }
  
  // Create identity DoF vector for mesh 1
  static Eigen::VectorXd identity_dofs(12);
  identity_dofs << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
  
  auto vol_result = compute_intersection_volume_cuda(
    ref_geom1, ref_geom2,
    std::make_shared<AffineDoF>(), affine_dof,  // Mesh 1: identity, Mesh 2: affine
    identity_dofs, dofs2,  // Identity DoFs for mesh 1, affine DoFs for mesh 2
    kgrid, vol_flags, 256, false
  );
  
  double V_int = vol_result.volume;
  Eigen::VectorXd grad_V_int = vol_result.grad_geom2;  // Gradient w.r.t. mesh 2 DoFs
  
  
  // Compute volume of mesh 2 and its gradient
  // For self-intersection volume V(geom2, geom2), the gradient w.r.t. dofs2 is:
  // grad_V2 = grad_geom1 + grad_geom2 (since both geometries use the same DoFs)
  auto V2_result = compute_intersection_volume_cuda(
    ref_geom2, ref_geom2,
    affine_dof, affine_dof,  // Both use affine DoF
    dofs2, dofs2,  // Both use the same DoFs
    kgrid, vol_flags, 256, false
  );
  double V2 = V2_result.volume;
  Eigen::VectorXd grad_V2 = V2_result.grad_geom1 + V2_result.grad_geom2;
  /*double V2 = affine_dof->compute_volume(ref_geom2, dofs2);
  Eigen::VectorXd grad_V2 = affine_dof->compute_volume_gradient(ref_geom2, dofs2);*/
  
  // Compute similarity measure based on type
  double similarity = 0.0;
  Eigen::VectorXd grad_sim;
  Eigen::MatrixXd hessian_sim;
  
  // Initialize gradient and hessian only if needed
  if (need_gradient) {
    grad_sim = Eigen::VectorXd(12);
  }
  if (need_hessian) {
    hessian_sim = Eigen::MatrixXd(12, 12);
  }
  
  if (measure == SIM_OCHIAI) {
    // Ochiai: V_int / sqrt(V1 * V2)
    double denom = std::sqrt(V1 * V2); 
    double inv_denom = 1.0 / denom;
    similarity = inv_denom * V_int;
    
    // Gradient: ∂(V_int / sqrt(V1*V2)) / ∂θ
    // = (1/sqrt(V1*V2)) * ∂V_int/∂θ - (V_int/(2*sqrt(V1*V2))) * (V1/V2) * ∂V2/∂θ
    if (need_gradient) {
      double coeff = -0.5 * inv_denom * inv_denom * inv_denom * V_int * V1;
      grad_sim = inv_denom * grad_V_int + coeff * grad_V2;
    }
    
    if (need_hessian) {
      // Hessian computation for Ochiai (simplified - using Gauss-Newton for V_int)
      Eigen::MatrixXd H_V_int = vol_result.hessian_geom2;
      Eigen::MatrixXd H_V2 = V2_result.hessian_geom1 + V2_result.hessian_cross 
        + V2_result.hessian_cross.transpose() + V2_result.hessian_geom2;
      hessian_sim = inv_denom * H_V_int;
      hessian_sim -= 0.5 * inv_denom * V_int / V2 * H_V2;
      hessian_sim += 0.75 * inv_denom * V_int / (V2 * V2) * grad_V2 * grad_V2.transpose();
      hessian_sim -= 0.5 * inv_denom / V2 * (grad_V_int * grad_V2.transpose() + grad_V2 * grad_V_int.transpose());
    }
  } else if (measure == SIM_JACCARD) {
    // Jaccard: V_int / (V1 + V2 - V_int)
    double denom = V1 + V2 - V_int;
    similarity = V_int / denom;
    double inv_denom = 1.0 / denom;
    double inv_denom_sq = 1.0 / (denom * denom);
    
    if (need_gradient) {
      // Gradient: ∂(V_int / (V1 + V2 - V_int)) / ∂θ
      // = (1/denom) * ∂V_int/∂θ - (V_int/denom²) * (∂V2/∂θ - ∂V_int/∂θ)
      grad_sim = inv_denom * grad_V_int - inv_denom_sq * V_int * (grad_V2 - grad_V_int);
    }
    
    if (need_hessian) {
      Eigen::MatrixXd H_V_int = vol_result.hessian_geom2;
      Eigen::MatrixXd H_V2 = V2_result.hessian_geom1 + V2_result.hessian_cross
        + V2_result.hessian_cross.transpose() + V2_result.hessian_geom2;
      // Hessian for Jaccard

      const Eigen::VectorXd dD = grad_V2 - grad_V_int;

      hessian_sim = inv_denom * H_V_int - inv_denom_sq * V_int * (H_V2 - H_V_int);
      hessian_sim -= inv_denom_sq * (grad_V_int * dD.transpose()
        + dD * grad_V_int.transpose());
      hessian_sim += 2.0 * inv_denom_sq * inv_denom * V_int * (dD * dD.transpose());
    }
  } else if (measure == SIM_DICE) {
    // Dice: 2 * V_int / (V1 + V2)
    double denom = V1 + V2;
    similarity = 2.0 * V_int / denom;
    double inv_denom = 2.0 / denom;
    double coeff = -2.0 * V_int / (denom * denom);
    
    if (need_gradient) {
      // Gradient: ∂(2*V_int / (V1 + V2)) / ∂θ
      // = (2/denom) * ∂V_int/∂θ - (2*V_int/denom²) * ∂V2/∂θ
      grad_sim = inv_denom * grad_V_int + coeff * grad_V2;
    }
    
    if (need_hessian) {
      Eigen::MatrixXd H_V_int = vol_result.hessian_geom2;
      Eigen::MatrixXd H_V2 = V2_result.hessian_geom1 + V2_result.hessian_cross
        + V2_result.hessian_cross.transpose() + V2_result.hessian_geom2;
      hessian_sim = inv_denom * H_V_int + coeff * H_V2;
      hessian_sim -= inv_denom / denom * (grad_V_int * grad_V2.transpose()
        + grad_V2 * grad_V_int.transpose());
      hessian_sim -= 2.0 * coeff / denom * grad_V2 * grad_V2.transpose();
    }
  }
  
  // Convert to -log(similarity) for minimization
  // Objective: minimize -log(similarity) (equivalent to maximizing similarity)
  // Gradient: d(-log(s))/dθ = -(1/s) * ds/dθ
  // Hessian: d²(-log(s))/dθ² = (1/s²) * (ds/dθ)(ds/dθ)^T - (1/s) * d²s/dθ²
  
  // Store original similarity for display
  result.similarity = similarity;
  
  // Avoid log(0) or log(negative) - clamp to small positive value
  double s_safe = std::max(similarity, 1e-10);
  
  // Compute -log(similarity) and its derivatives
  result.value = -std::log(s_safe);
  
  if (need_gradient) {
    result.gradient = -grad_sim / s_safe;
  } else {
    result.gradient = Eigen::VectorXd();  // Empty if not computed
  }
  
  if (need_hessian) {
    // Hessian of -log(s): (1/s²) * grad_s * grad_s^T - (1/s) * hessian_s
    Eigen::MatrixXd grad_outer = grad_sim * grad_sim.transpose();
    result.hessian = (1.0 / (s_safe * s_safe)) * grad_outer - (1.0 / s_safe) * hessian_sim;
  } else {
    result.hessian = Eigen::MatrixXd();  // Empty if not computed
  }
  
  return result;
}

} // namespace PoIntInt


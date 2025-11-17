#pragma once
#include "compute_intersection_volume.hpp"
#include "dof/affine_dof.hpp"
#include "quadrature/kgrid.hpp"
#include <Eigen/Dense>
#include <memory>

namespace PoIntInt {

// Similarity measure types
enum SimilarityMeasure {
  SIM_OCHIAI,
  SIM_JACCARD,
  SIM_DICE
};

// Flags for similarity computation (what to compute)
enum SimilarityComputationFlags {
  SIM_FUNC_ONLY = 0x01,      // Function value only
  SIM_GRADIENT = 0x02,        // Gradient
  SIM_HESSIAN = 0x04,         // Hessian
  SIM_FUNC_GRAD = SIM_FUNC_ONLY | SIM_GRADIENT,  // Function + Gradient
  SIM_ALL = SIM_FUNC_ONLY | SIM_GRADIENT | SIM_HESSIAN  // All
};

// Result structure for similarity computation
// Note: We optimize -log(similarity) to convert maximization to minimization
struct SimilarityResult {
  double value;  // -log(similarity)
  Eigen::VectorXd gradient;  // Gradient of -log(similarity) w.r.t. DoFs
  Eigen::MatrixXd hessian;  // Hessian of -log(similarity) w.r.t. DoFs
  double similarity;  // Original similarity value (for display)
};

// Compute similarity measure and its gradient/Hessian w.r.t. affine DoFs of mesh 2
// Note: Mesh 1 is fixed, only mesh 2 is transformed
// flags: Combination of SimilarityComputationFlags to specify what to compute
SimilarityResult compute_similarity(
  const Geometry& ref_geom1,
  const Geometry& ref_geom2,
  const std::shared_ptr<AffineDoF>& affine_dof,
  const Eigen::VectorXd& dofs2,
  const KGrid& kgrid,
  SimilarityMeasure measure,
  double V1,  // Volume of mesh 1 (constant)
  int flags = SIM_FUNC_GRAD  // What to compute (default: function + gradient)
);

} // namespace PoIntInt


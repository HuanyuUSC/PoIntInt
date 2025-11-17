#pragma once
#include "similarity_measure.hpp"
#include "geometry/geometry.hpp"
#include "dof/affine_dof.hpp"
#include "quadrature/kgrid.hpp"
#include <Eigen/Dense>
#include <memory>
#include <string>

namespace PoIntInt {

// Optimization state structure
struct OptimizationState {
  Eigen::VectorXd dofs;
  double similarity;  // Similarity value (0-1)
  double similarity_loss;  // -log(similarity), the objective value
  int iteration;
  bool converged;
  std::string status;
};

// Newton-Raphson optimizer for mesh alignment
OptimizationState optimize_alignment(
  const Geometry& ref_geom1,
  const Geometry& ref_geom2,
  const std::shared_ptr<AffineDoF>& affine_dof,
  const Eigen::VectorXd& initial_dofs,
  const KGrid& kgrid,
  SimilarityMeasure measure,
  double V1,  // Volume of mesh 1 (constant)
  double tolerance = 1e-6,
  int max_iterations = 50,
  double damping = 0.1  // Damping factor for Newton step
);

} // namespace PoIntInt


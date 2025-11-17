#include "optimizer.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <algorithm>

namespace PoIntInt {

OptimizationState optimize_alignment(
  const Geometry& ref_geom1,
  const Geometry& ref_geom2,
  const std::shared_ptr<AffineDoF>& affine_dof,
  const Eigen::VectorXd& initial_dofs,
  const KGrid& kgrid,
  SimilarityMeasure measure,
  double V1,
  double tolerance,
  int max_iterations,
  double damping
) {
  OptimizationState state;
  state.dofs = initial_dofs;
  state.iteration = 0;
  state.converged = false;
  
  // Line search parameters
  double alpha_init = 1.0;  // Initial step size
  double alpha_min = 1e-10;  // Minimum step size
  double c = 0.1;  // Armijo constant (sufficient decrease parameter)
  double rho = 0.5;  // Backtracking factor
  
  for (int iter = 0; iter < max_iterations; ++iter) {
    state.iteration = iter;
    
    // Compute similarity, gradient, and Hessian (for Newton-Raphson)
    auto sim_result = compute_similarity(
      ref_geom1, ref_geom2, affine_dof, state.dofs, kgrid,
      measure, V1, SIM_ALL  // Request function, gradient, and Hessian
    );
    
    state.similarity = sim_result.similarity;  // Store original similarity (0-1)
    state.similarity_loss = sim_result.value;  // Store -log(similarity), the objective
    
    // Check convergence (gradient of -log(similarity))
    double grad_norm = sim_result.gradient.norm();
    if (grad_norm < tolerance) {
      state.converged = true;
      state.status = "Converged (gradient norm < tolerance)";
      break;
    }
    
    // Project Hessian to positive semi-definite (PSD)
    // Compute eigendecomposition: H = Q * Λ * Q^T
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(sim_result.hessian);
    if (eigen_solver.info() != Eigen::Success) {
      state.status = "Hessian eigendecomposition failed";
      break;
    }
    
    // Get eigenvalues and eigenvectors
    Eigen::VectorXd eigenvalues = eigen_solver.eigenvalues();
    Eigen::MatrixXd eigenvectors = eigen_solver.eigenvectors();
    
    // Project to PSD: set negative eigenvalues to zero
    Eigen::VectorXd eigenvalues_psd = eigenvalues.cwiseMax(0.0);
    
    // Reconstruct PSD Hessian: H_psd = Q * Λ_psd * Q^T
    Eigen::MatrixXd H_psd = eigenvectors * eigenvalues_psd.asDiagonal() * eigenvectors.transpose();
    
    // Add damping after projection: H_final = H_psd + damping * I
    Eigen::MatrixXd H_final = H_psd + damping * Eigen::MatrixXd::Identity(12, 12);
    
    // Newton step: H_final * delta = -g
    Eigen::LDLT<Eigen::MatrixXd> solver(H_final);
    if (solver.info() != Eigen::Success) {
      state.status = "Hessian solve failed";
      break;
    }
    
    Eigen::VectorXd delta = solver.solve(-sim_result.gradient);
    
    // Check descent direction: -d·g should be positive
    double descent_check = -delta.dot(sim_result.gradient);
    std::cout << "  Iteration " << iter << ": -d·g = " << descent_check << std::endl;
    
    // Backtracking line search
    // Armijo condition: f(x + α*d) <= f(x) + c*α*d^T*g
    double f0 = sim_result.value;  // Current function value
    double alpha = alpha_init;
    Eigen::VectorXd dofs_new = state.dofs + alpha * delta;
    
    // Evaluate function at new point (function only)
    auto sim_result_new = compute_similarity(
      ref_geom1, ref_geom2, affine_dof, dofs_new, kgrid,
      measure, V1, SIM_FUNC_ONLY
    );
    double f_new = sim_result_new.value;
    
    // Backtrack until Armijo condition is satisfied
    int ls_iter = 0;
    int max_ls_iter = 20;
    printf("Backtracking line search.\n");
    printf("Trying alpha = %g, f = %g.\n", alpha, f_new);
    while (f_new > f0 - c * alpha * descent_check && alpha > alpha_min && ls_iter < max_ls_iter) {
      alpha *= rho;
      dofs_new = state.dofs + alpha * delta;
      sim_result_new = compute_similarity(
        ref_geom1, ref_geom2, affine_dof, dofs_new, kgrid,
        measure, V1, SIM_FUNC_ONLY
      );
      f_new = sim_result_new.value;
      printf("Trying alpha = %g, f = %g.\n", alpha, f_new);
      ls_iter++;
    }
    
    if (alpha <= alpha_min) {
      state.status = "Line search failed (step size too small)";
      break;
    }
    
    std::cout << "    Line search: α = " << alpha << ", f_new = " << f_new << ", f0 = " << f0 << std::endl;
    
    // Update DoFs with the accepted step
    state.dofs = dofs_new;
  }
  
  if (!state.converged) {
    state.status = "Reached max iterations";
  }
  
  return state;
}

} // namespace PoIntInt


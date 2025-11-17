#include <vector>
#include <cmath>
#include <complex>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include "compute_intersection_volume.hpp"
#include "geometry/geometry.hpp"
#include "dof/dof_parameterization.hpp"
#include "quadrature/kgrid.hpp"

namespace PoIntInt {

// ============================================================================
// CPU Implementation: Intersection Volume
// ============================================================================

IntersectionVolumeResult compute_intersection_volume_cpu(
  const Geometry& ref_geom1,
  const Geometry& ref_geom2,
  const std::shared_ptr<DoFParameterization>& dof1,
  const std::shared_ptr<DoFParameterization>& dof2,
  const Eigen::VectorXd& dofs1,
  const Eigen::VectorXd& dofs2,
  const KGrid& kgrid,
  ComputationFlags flags,
  bool enable_profiling)
{
  auto t_start = std::chrono::high_resolution_clock::now();
  
  int num_dofs1 = dof1->num_dofs();
  int num_dofs2 = dof2->num_dofs();
  int Q = kgrid.dirs.size();
  
  IntersectionVolumeResult result;
  result.volume = 0.0;
  
  // Initialize result vectors based on flags
  if (needs_gradient(flags)) {
    result.grad_geom1 = Eigen::VectorXd::Zero(num_dofs1);
    result.grad_geom2 = Eigen::VectorXd::Zero(num_dofs2);
  }
  if (needs_hessian(flags)) {
    result.hessian_geom1 = Eigen::MatrixXd::Zero(num_dofs1, num_dofs1);
    result.hessian_geom2 = Eigen::MatrixXd::Zero(num_dofs2, num_dofs2);
    result.hessian_cross = Eigen::MatrixXd::Zero(num_dofs1, num_dofs2);
  }
  
  // Use a struct to hold local accumulators for parallel reduction
  struct LocalAccumulator {
    Eigen::VectorXd grad_geom1;
    Eigen::VectorXd grad_geom2;
    Eigen::MatrixXd hessian_geom1;
    Eigen::MatrixXd hessian_geom2;
    Eigen::MatrixXd hessian_cross;
    double volume;
    bool needs_grad;
    bool needs_hess;
    
    LocalAccumulator(int n1, int n2, bool grad, bool hess) 
      : grad_geom1(grad ? Eigen::VectorXd::Zero(n1) : Eigen::VectorXd()),
        grad_geom2(grad ? Eigen::VectorXd::Zero(n2) : Eigen::VectorXd()),
        hessian_geom1(hess ? Eigen::MatrixXd::Zero(n1, n1) : Eigen::MatrixXd()),
        hessian_geom2(hess ? Eigen::MatrixXd::Zero(n2, n2) : Eigen::MatrixXd()),
        hessian_cross(hess ? Eigen::MatrixXd::Zero(n1, n2) : Eigen::MatrixXd()),
        volume(0.0),
        needs_grad(grad),
        needs_hess(hess) {}
  };
  
  auto t_compute_start = std::chrono::high_resolution_clock::now();
  
  LocalAccumulator local_result = tbb::parallel_reduce(
    tbb::blocked_range<int>(0, Q),
    LocalAccumulator(num_dofs1, num_dofs2, needs_gradient(flags), needs_hessian(flags)),
    [&](const tbb::blocked_range<int>& range, LocalAccumulator local_acc) -> LocalAccumulator {
      for (int q = range.begin(); q < range.end(); ++q) {
        // Construct k-vector from KGrid
        Eigen::Vector3d k(
          kgrid.dirs[q][0] * kgrid.kmag[q],
          kgrid.dirs[q][1] * kgrid.kmag[q],
          kgrid.dirs[q][2] * kgrid.kmag[q]
        );
        
        double w = kgrid.w[q];
        
        // Phase 1: Compute A(k) for both geometries (always needed)
        std::complex<double> A1 = dof1->compute_A(ref_geom1, k, dofs1);
        std::complex<double> A2 = dof2->compute_A(ref_geom2, k, dofs2);
        
        // Phase 2: Compute ∂A(k)/∂θ for both geometries (if needed)
        Eigen::VectorXcd grad_A1, grad_A2;
        if (local_acc.needs_grad || local_acc.needs_hess) {
          grad_A1 = dof1->compute_A_gradient(ref_geom1, k, dofs1);
          grad_A2 = dof2->compute_A_gradient(ref_geom2, k, dofs2);
        }
        
        // Phase 3: Accumulate contributions
        // Volume: Re(A1 * conj(A2)) = Re(A1) * Re(A2) + Im(A1) * Im(A2)
        double vol_contrib = A1.real() * A2.real() + A1.imag() * A2.imag();
        local_acc.volume += w * vol_contrib;
        
        // Gradient for geometry 1: Re(dA1 * conj(A2))
        if (local_acc.needs_grad || local_acc.needs_hess) {
          for (int i = 0; i < num_dofs1; ++i) {
            std::complex<double> dA1 = grad_A1(i);
            double grad_contrib = dA1.real() * A2.real() + dA1.imag() * A2.imag();
            if (local_acc.needs_grad) {
              local_acc.grad_geom1(i) += w * grad_contrib;
            }
            
            // Hessian contributions for geometry 1
            if (local_acc.needs_hess) {
              for (int j = 0; j < num_dofs1; ++j) {
                std::complex<double> dA1_j = grad_A1(j);
                // Real part of (dA1_i * conj(dA1_j))
                double hess_contrib = dA1.real() * dA1_j.real() + dA1.imag() * dA1_j.imag();
                local_acc.hessian_geom1(i, j) += w * hess_contrib;
              }
              
              // Cross-term Hessian: H₁₂
              for (int j = 0; j < num_dofs2; ++j) {
                std::complex<double> dA2_j = grad_A2(j);
                // Real part of (dA1_i * conj(dA2_j))
                double hess_contrib = dA1.real() * dA2_j.real() + dA1.imag() * dA2_j.imag();
                local_acc.hessian_cross(i, j) += w * hess_contrib;
              }
            }
          }
          
          // Gradient for geometry 2: Re(A1 * conj(dA2))
          for (int i = 0; i < num_dofs2; ++i) {
            std::complex<double> dA2 = grad_A2(i);
            double grad_contrib = A1.real() * dA2.real() + A1.imag() * dA2.imag();
            if (local_acc.needs_grad) {
              local_acc.grad_geom2(i) += w * grad_contrib;
            }
            
            // Hessian contributions for geometry 2
            if (local_acc.needs_hess) {
              for (int j = 0; j < num_dofs2; ++j) {
                std::complex<double> dA2_j = grad_A2(j);
                // Real part of (dA2_i * conj(dA2_j))
                double hess_contrib = dA2.real() * dA2_j.real() + dA2.imag() * dA2_j.imag();
                local_acc.hessian_geom2(i, j) += w * hess_contrib;
              }
            }
          }
        }
      }
      return local_acc;
    },
    [](const LocalAccumulator& a, const LocalAccumulator& b) -> LocalAccumulator {
      LocalAccumulator result(a.grad_geom1.size(), a.grad_geom2.size(), a.needs_grad, a.needs_hess);
      result.volume = a.volume + b.volume;
      if (a.needs_grad) {
        result.grad_geom1 = a.grad_geom1 + b.grad_geom1;
        result.grad_geom2 = a.grad_geom2 + b.grad_geom2;
      }
      if (a.needs_hess) {
        result.hessian_geom1 = a.hessian_geom1 + b.hessian_geom1;
        result.hessian_geom2 = a.hessian_geom2 + b.hessian_geom2;
        result.hessian_cross = a.hessian_cross + b.hessian_cross;
      }
      return result;
    }
  );
  
  auto t_compute_end = std::chrono::high_resolution_clock::now();
  auto t_end = std::chrono::high_resolution_clock::now();
  
  // Apply scaling factor: 1/(8π³)
  const double scale = 1.0 / (8.0 * M_PI * M_PI * M_PI);
  result.volume = local_result.volume * scale;
  if (needs_gradient(flags)) {
    result.grad_geom1 = local_result.grad_geom1 * scale;
    result.grad_geom2 = local_result.grad_geom2 * scale;
  }
  if (needs_hessian(flags)) {
    result.hessian_geom1 = local_result.hessian_geom1 * scale;
    result.hessian_geom2 = local_result.hessian_geom2 * scale;
    result.hessian_cross = local_result.hessian_cross * scale;
  }
  
  if (enable_profiling) {
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(
      t_compute_end - t_compute_start).count() / 1000.0;
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
      t_end - t_start).count() / 1000.0;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== CPU Intersection Volume Profiler ===" << std::endl;
    std::cout << "K-grid nodes: " << Q << std::endl;
    std::cout << "DoFs per geometry: " << num_dofs1 << ", " << num_dofs2 << std::endl;
    std::cout << "Flags: Volume=" << needs_volume(flags) 
              << ", Gradient=" << needs_gradient(flags) 
              << ", Hessian=" << needs_hessian(flags) << std::endl;
    std::cout << "--- Timing (ms) ---" << std::endl;
    std::cout << "  Computation:      " << std::setw(8) << compute_time << " ms" << std::endl;
    std::cout << "  Total time:       " << std::setw(8) << total_time << " ms" << std::endl;
    std::cout << "==========================================\n" << std::endl;
  }
  
  return result;
}

} // namespace PoIntInt


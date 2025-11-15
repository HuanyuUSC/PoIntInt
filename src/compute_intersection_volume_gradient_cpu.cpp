#include <vector>
#include <cmath>
#include <complex>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include "compute_intersection_volume_gradient.hpp"
#include "geometry/geometry.hpp"
#include "dof/dof_parameterization.hpp"
#include "quadrature/kgrid.hpp"

namespace PoIntInt {


IntersectionVolumeGradientResult compute_intersection_volume_gradient_cpu(
  const Geometry& geom1,
  const Geometry& geom2,
  const std::shared_ptr<DoFParameterization>& dof1,
  const std::shared_ptr<DoFParameterization>& dof2,
  const Eigen::VectorXd& dofs1,
  const Eigen::VectorXd& dofs2,
  const KGrid& kgrid,
  bool enable_profiling)
{
  auto t_start = std::chrono::high_resolution_clock::now();
  
  int num_dofs1 = dof1->num_dofs();
  int num_dofs2 = dof2->num_dofs();
  int Q = kgrid.dirs.size();
  
  // Initialize result vectors
  Eigen::VectorXd grad_geom1 = Eigen::VectorXd::Zero(num_dofs1);
  Eigen::VectorXd grad_geom2 = Eigen::VectorXd::Zero(num_dofs2);
  double volume = 0.0;
  
  // Parallel reduction over k-nodes
  auto t_compute_start = std::chrono::high_resolution_clock::now();
  
  // Use a struct to hold local accumulators for parallel reduction
  struct LocalAccumulator {
    Eigen::VectorXd grad_geom1;
    Eigen::VectorXd grad_geom2;
    double volume;
    
    LocalAccumulator(int n1, int n2) 
      : grad_geom1(Eigen::VectorXd::Zero(n1)),
        grad_geom2(Eigen::VectorXd::Zero(n2)),
        volume(0.0) {}
  };
  
  LocalAccumulator result = tbb::parallel_reduce(
    tbb::blocked_range<int>(0, Q),
    LocalAccumulator(num_dofs1, num_dofs2),
    [&](const tbb::blocked_range<int>& range, LocalAccumulator local_acc) -> LocalAccumulator {
      for (int q = range.begin(); q < range.end(); ++q) {
        // Construct k-vector from KGrid
        Eigen::Vector3d k(
          kgrid.dirs[q][0] * kgrid.kmag[q],
          kgrid.dirs[q][1] * kgrid.kmag[q],
          kgrid.dirs[q][2] * kgrid.kmag[q]
        );
        
        double w = kgrid.w[q];
        
        // Phase 1: Compute A(k) for both geometries
        std::complex<double> A1 = dof1->compute_A(geom1, k, dofs1);
        std::complex<double> A2 = dof2->compute_A(geom2, k, dofs2);
        
        // Phase 2: Compute ∂A(k)/∂θ for both geometries
        Eigen::VectorXcd grad_A1 = dof1->compute_A_gradient(geom1, k, dofs1);
        Eigen::VectorXcd grad_A2 = dof2->compute_A_gradient(geom2, k, dofs2);
        
        // Phase 3: Accumulate contributions
        // Volume: Re(A1 * conj(A2)) = Re(A1) * Re(A2) + Im(A1) * Im(A2)
        double vol_contrib = A1.real() * A2.real() + A1.imag() * A2.imag();
        local_acc.volume += w * vol_contrib;
        
        // Gradient for geometry 1: Re(dA1 * conj(A2))
        // Re(dA1 * conj(A2)) = Re(dA1) * Re(A2) + Im(dA1) * Im(A2)
        for (int i = 0; i < num_dofs1; ++i) {
          std::complex<double> dA1 = grad_A1(i);
          double grad_contrib = dA1.real() * A2.real() + dA1.imag() * A2.imag();
          local_acc.grad_geom1(i) += w * grad_contrib;
        }
        
        // Gradient for geometry 2: Re(A1 * conj(dA2))
        // Re(A1 * conj(dA2)) = Re(A1) * Re(dA2) + Im(A1) * Im(dA2)
        for (int i = 0; i < num_dofs2; ++i) {
          std::complex<double> dA2 = grad_A2(i);
          double grad_contrib = A1.real() * dA2.real() + A1.imag() * dA2.imag();
          local_acc.grad_geom2(i) += w * grad_contrib;
        }
      }
      return local_acc;
    },
    [](const LocalAccumulator& a, const LocalAccumulator& b) -> LocalAccumulator {
      LocalAccumulator result(a.grad_geom1.size(), a.grad_geom2.size());
      result.grad_geom1 = a.grad_geom1 + b.grad_geom1;
      result.grad_geom2 = a.grad_geom2 + b.grad_geom2;
      result.volume = a.volume + b.volume;
      return result;
    }
  );
  
  auto t_compute_end = std::chrono::high_resolution_clock::now();
  auto t_end = std::chrono::high_resolution_clock::now();
  
  // Apply scaling factor: 1/(8π³)
  const double scale = 1.0 / (8.0 * M_PI * M_PI * M_PI);
  grad_geom1 = result.grad_geom1 * scale;
  grad_geom2 = result.grad_geom2 * scale;
  volume = result.volume * scale;
  
  if (enable_profiling) {
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(
      t_compute_end - t_compute_start).count() / 1000.0;
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
      t_end - t_start).count() / 1000.0;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== CPU Intersection Volume Gradient Profiler ===" << std::endl;
    std::cout << "K-grid nodes: " << Q << std::endl;
    std::cout << "DoFs per geometry: " << num_dofs1 << ", " << num_dofs2 << std::endl;
    std::cout << "--- Timing (ms) ---" << std::endl;
    std::cout << "  Computation:      " << std::setw(8) << compute_time << " ms" << std::endl;
    std::cout << "  Total time:       " << std::setw(8) << total_time << " ms" << std::endl;
    std::cout << "==========================================\n" << std::endl;
  }
  
  IntersectionVolumeGradientResult final_result;
  final_result.grad_geom1 = grad_geom1;
  final_result.grad_geom2 = grad_geom2;
  final_result.volume = volume;
  
  return final_result;
}

} // namespace PoIntInt


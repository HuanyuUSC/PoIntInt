#include "compute_intersection_volume_multi_object.hpp"
#include "dof/dof_parameterization.hpp"
#include "computation_flags.hpp"
#include "geometry/types.hpp"
#include <cmath>
#include <complex>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <Eigen/Dense>

using PoIntInt::get_geometry_type_name;
using PoIntInt::get_geometry_element_name;
using PoIntInt::GEOM_TRIANGLE;
using PoIntInt::GEOM_DISK;
using PoIntInt::GEOM_GAUSSIAN;
using PoIntInt::needs_volume;
using PoIntInt::needs_gradient;
using PoIntInt::needs_hessian;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace PoIntInt {

IntersectionVolumeMatrixResult compute_intersection_volume_matrix_cpu(
  const std::vector<Geometry>& ref_geometries,
  const std::vector<std::shared_ptr<DoFParameterization>>& dofs,
  const std::vector<Eigen::VectorXd>& dof_vectors,
  const KGrid& kgrid,
  ComputationFlags flags,
  bool enable_profiling)
{
  auto t_start = std::chrono::high_resolution_clock::now();
  
  IntersectionVolumeMatrixResult result;
  int num_objects = (int)ref_geometries.size();
  
  // Validate inputs
  if (num_objects == 0) {
    result.volume_matrix = Eigen::MatrixXd::Zero(0, 0);
    return result;
  }
  
  if ((int)dofs.size() != num_objects || (int)dof_vectors.size() != num_objects) {
    std::cerr << "Error: Mismatch between number of geometries (" << num_objects 
              << ") and number of DoF parameterizations (" << dofs.size() 
              << ") or DoF vectors (" << dof_vectors.size() << ")" << std::endl;
    result.volume_matrix = Eigen::MatrixXd::Zero(num_objects, num_objects);
    return result;
  }
  
  int Q = (int)kgrid.kmag.size();
  
  // Get number of DoFs for each object
  std::vector<int> num_dofs_per_obj(num_objects);
  for (int obj = 0; obj < num_objects; ++obj) {
    num_dofs_per_obj[obj] = (int)dof_vectors[obj].size();
  }
  
  auto t_phase1_start = std::chrono::high_resolution_clock::now();
  
  // Phase 1: Compute form factor matrix J (Q × num_objects)
  // J[q, obj] = A_obj(k_q) for each k-node q and object obj
  std::vector<std::complex<double>> J(Q * num_objects);
  
  // Parallelize over k-nodes (outer loop)
  // For each k-node, compute A(k) for all objects using their DoF parameterizations
  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, Q),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t q = r.begin(); q < r.end(); ++q) {
        // Get k-vector: k = kmag[q] * kdir[q]
        double kmag = (double)kgrid.kmag[q];
        const auto& kdir_arr = kgrid.dirs[q];
        Eigen::Vector3d kdir((double)kdir_arr[0], (double)kdir_arr[1], (double)kdir_arr[2]);
        Eigen::Vector3d k = kmag * kdir;
        
        // Compute A(k) for all objects at this k-node using their DoF parameterizations
        for (size_t obj = 0; obj < num_objects; ++obj) {
          // Compute A(k) for this geometry using its DoF parameterization
          std::complex<double> A = dofs[obj]->compute_A(ref_geometries[obj], k, dof_vectors[obj]);
          
          // Store in row-major format: J[q * num_objects + obj]
          J[q * num_objects + obj] = A;
        }
      }
    });
  
  auto t_phase1_end = std::chrono::high_resolution_clock::now();
  auto t_phase2_start = std::chrono::high_resolution_clock::now();
  
  // Phase 2: Compute volume matrix V = J^T D J
  // V[i,j] = (1/(8π³)) · Σ_q w_q · Re(J[q,i] · conj(J[q,j]))
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(num_objects, num_objects);
  
  // Parallelize over upper triangle (including diagonal)
  // Use a 1D range and compute (i,j) from the index
  int num_upper_triangle = num_objects * (num_objects + 1) / 2;
  tbb::parallel_for(
    tbb::blocked_range<int>(0, num_upper_triangle),
    [&](const tbb::blocked_range<int>& r) {
      for (int idx = r.begin(); idx < r.end(); ++idx) {
        // Convert linear index to (i, j) in upper triangle
        // For upper triangle: idx = i * num_objects - i*(i-1)/2 + j - i
        // Solving: i = floor((2*num_objects + 1 - sqrt((2*num_objects+1)^2 - 8*idx))/2)
        // Simpler: iterate i from 0, find j such that idx is in range
        int i = 0, j = 0;
        int remaining = idx;
        for (i = 0; i < num_objects; ++i) {
          int row_size = num_objects - i;  // Number of elements in row i (from i to num_objects-1)
          if (remaining < row_size) {
            j = i + remaining;
            break;
          }
          remaining -= row_size;
        }
        
        double sum = 0.0;
        
        // Compute V[i,j] = Σ_q w_q · Re(J[q,i] · conj(J[q,j]))
        for (int q = 0; q < Q; ++q) {
          int idx_i = q * num_objects + i;
          int idx_j = q * num_objects + j;
          std::complex<double> Ji = J[idx_i];
          std::complex<double> Jj = J[idx_j];
          
          // J[q,i] · conj(J[q,j]) = (Ji.re + i*Ji.im) · (Jj.re - i*Jj.im)
          //                        = Ji.re*Jj.re + Ji.im*Jj.im + i*(Ji.im*Jj.re - Ji.re*Jj.im)
          // Real part: Ji.re*Jj.re + Ji.im*Jj.im
          double real_part = Ji.real() * Jj.real() + Ji.imag() * Jj.imag();
          sum += kgrid.w[q] * real_part;
        }
        
        // V = (1/(8π³)) · sum
        double V_ij = sum / (8.0 * M_PI * M_PI * M_PI);
        
        // Write to both (i,j) and (j,i) for symmetry
        V(i, j) = V_ij;
        if (i != j) {
          V(j, i) = V_ij;
        }
      }
    });
  
  auto t_phase2_end = std::chrono::high_resolution_clock::now();
  auto t_phase3_start = std::chrono::high_resolution_clock::now();
  
  // Phase 3: Compute gradients (if needed)
  std::vector<std::vector<Eigen::VectorXcd>> grad_J;  // Q × num_objects × num_dofs[obj]
  if (needs_gradient(flags) || needs_hessian(flags)) {
    grad_J.resize(Q);
    for (int q = 0; q < Q; ++q) {
      grad_J[q].resize(num_objects);
      for (int obj = 0; obj < num_objects; ++obj) {
        grad_J[q][obj] = Eigen::VectorXcd::Zero(num_dofs_per_obj[obj]);
      }
    }
    
    // Compute ∂A(k)/∂θ for each object at each k-node
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, Q),
      [&](const tbb::blocked_range<size_t>& r) {
        for (size_t q = r.begin(); q < r.end(); ++q) {
          double kmag = (double)kgrid.kmag[q];
          const auto& kdir_arr = kgrid.dirs[q];
          Eigen::Vector3d kdir((double)kdir_arr[0], (double)kdir_arr[1], (double)kdir_arr[2]);
          Eigen::Vector3d k = kmag * kdir;
          
          for (size_t obj = 0; obj < num_objects; ++obj) {
            grad_J[q][obj] = dofs[obj]->compute_A_gradient(ref_geometries[obj], k, dof_vectors[obj]);
          }
        }
      });
  }
  
  auto t_phase3_end = std::chrono::high_resolution_clock::now();
  auto t_phase4_start = std::chrono::high_resolution_clock::now();
  
  // Phase 4: Compute gradients and Hessians for each pair (i,j)
  if (needs_gradient(flags)) {
    result.grad_matrix.resize(num_objects);
    for (int i = 0; i < num_objects; ++i) {
      result.grad_matrix[i].resize(num_objects);
      for (int j = 0; j < num_objects; ++j) {
        result.grad_matrix[i][j] = Eigen::VectorXd::Zero(num_dofs_per_obj[i]);
      }
    }
    
    // Compute gradients for each pair
    int num_pairs = num_objects * num_objects;
    tbb::parallel_for(
      tbb::blocked_range<int>(0, num_pairs),
      [&](const tbb::blocked_range<int>& r) {
        for (int idx = r.begin(); idx < r.end(); ++idx) {
          int i = idx / num_objects;
          int j = idx % num_objects;
          
          Eigen::VectorXd grad_i = Eigen::VectorXd::Zero(num_dofs_per_obj[i]);
          Eigen::VectorXd grad_j = Eigen::VectorXd::Zero(num_dofs_per_obj[j]);
          
          // Compute gradients: ∂V[i,j]/∂θ_i and ∂V[i,j]/∂θ_j
          for (int q = 0; q < Q; ++q) {
            double w = kgrid.w[q];
            std::complex<double> Ai = J[q * num_objects + i];
            std::complex<double> Aj = J[q * num_objects + j];
            
            // Gradient w.r.t. DoFs of object i: Re((∂A_i/∂θ_i) · conj(A_j))
            for (int dof = 0; dof < num_dofs_per_obj[i]; ++dof) {
              std::complex<double> dAi = grad_J[q][i](dof);
              double grad_contrib = dAi.real() * Aj.real() + dAi.imag() * Aj.imag();
              grad_i(dof) += w * grad_contrib;
            }
            
            // Gradient w.r.t. DoFs of object j: Re(A_i · conj(∂A_j/∂θ_j))
            for (int dof = 0; dof < num_dofs_per_obj[j]; ++dof) {
              std::complex<double> dAj = grad_J[q][j](dof);
              double grad_contrib = Ai.real() * dAj.real() + Ai.imag() * dAj.imag();
              grad_j(dof) += w * grad_contrib;
            }
          }
          
          // Apply scaling factor: 1/(8π³)
          const double scale = 1.0 / (8.0 * M_PI * M_PI * M_PI);
          result.grad_matrix[i][j] = grad_i * scale;
          result.grad_matrix[j][i] = grad_j * scale;
        }
      });
  }
  
  if (needs_hessian(flags)) {
    // Initialize Hessian storage
    result.hessian_ii.resize(num_objects);
    result.hessian_jj.resize(num_objects);
    result.hessian_ij.resize(num_objects);
    for (int i = 0; i < num_objects; ++i) {
      result.hessian_ii[i].resize(num_objects);
      result.hessian_jj[i].resize(num_objects);
      result.hessian_ij[i].resize(num_objects);
      for (int j = 0; j < num_objects; ++j) {
        result.hessian_ii[i][j] = Eigen::MatrixXd::Zero(num_dofs_per_obj[i], num_dofs_per_obj[i]);
        result.hessian_jj[i][j] = Eigen::MatrixXd::Zero(num_dofs_per_obj[j], num_dofs_per_obj[j]);
        result.hessian_ij[i][j] = Eigen::MatrixXd::Zero(num_dofs_per_obj[i], num_dofs_per_obj[j]);
      }
    }
    
    // Compute Hessians for each pair (Gauss-Newton approximation)
    int num_pairs = num_objects * num_objects;
    tbb::parallel_for(
      tbb::blocked_range<int>(0, num_pairs),
      [&](const tbb::blocked_range<int>& r) {
        for (int idx = r.begin(); idx < r.end(); ++idx) {
          int i = idx / num_objects;
          int j = idx % num_objects;
          
          // For Gauss-Newton, H_ii and H_jj are zero (we ignore second derivatives)
          // Only compute H_ij: cross-term Hessian
          Eigen::MatrixXd H_ij = Eigen::MatrixXd::Zero(num_dofs_per_obj[i], num_dofs_per_obj[j]);
          
          // Compute Hessian using Gauss-Newton approximation
          // V[i,j] = (1/(8π³)) · Σ_q w_q · Re(A_i(k_q) · conj(A_j(k_q)))
          // = (1/(8π³)) · Σ_q w_q · g_q(A_i) · h_q(A_j)
          // Gauss-Newton ignores second derivatives, so only the cross-term is non-zero:
          // H_ij[i,j] = (1/(8π³)) · Σ_q w_q · (∂A_i/∂θ_i) · (∂A_j/∂θ_j)^T
          for (int q = 0; q < Q; ++q) {
            double w = kgrid.w[q];
            
            // H_ij: outer product of grad_A_i with grad_A_j
            for (int di = 0; di < num_dofs_per_obj[i]; ++di) {
              std::complex<double> dAi_di = grad_J[q][i](di);
              for (int dj = 0; dj < num_dofs_per_obj[j]; ++dj) {
                std::complex<double> dAj_dj = grad_J[q][j](dj);
                double hess_contrib = dAi_di.real() * dAj_dj.real() + dAi_di.imag() * dAj_dj.imag();
                H_ij(di, dj) += w * hess_contrib;
              }
            }
          }
          
          // Apply scaling factor: 1/(8π³)
          const double scale = 1.0 / (8.0 * M_PI * M_PI * M_PI);
          // H_ii and H_jj remain zero for Gauss-Newton
          result.hessian_ii[i][j] = Eigen::MatrixXd::Zero(num_dofs_per_obj[i], num_dofs_per_obj[i]);
          result.hessian_jj[i][j] = Eigen::MatrixXd::Zero(num_dofs_per_obj[j], num_dofs_per_obj[j]);
          result.hessian_ij[i][j] = H_ij * scale;
        }
      });
  }
  
  auto t_phase4_end = std::chrono::high_resolution_clock::now();
  auto t_end = std::chrono::high_resolution_clock::now();
  
  result.volume_matrix = V;
  
  if (enable_profiling) {
    auto phase1_time = std::chrono::duration_cast<std::chrono::microseconds>(t_phase1_end - t_phase1_start).count() / 1000.0;
    auto phase2_time = std::chrono::duration_cast<std::chrono::microseconds>(t_phase2_end - t_phase2_start).count() / 1000.0;
    auto phase3_time = (needs_gradient(flags) || needs_hessian(flags)) ? 
                       std::chrono::duration_cast<std::chrono::microseconds>(t_phase3_end - t_phase3_start).count() / 1000.0 : 0.0;
    auto phase4_time = (needs_gradient(flags) || needs_hessian(flags)) ? 
                       std::chrono::duration_cast<std::chrono::microseconds>(t_phase4_end - t_phase4_start).count() / 1000.0 : 0.0;
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== CPU Multi-Object Volume Computation Profiler ===" << std::endl;
    std::cout << "Number of objects: " << num_objects << std::endl;
    std::cout << "K-grid nodes: " << Q << std::endl;
    
    // Count geometry types
    int num_meshes = 0, num_pointclouds = 0, num_gaussians = 0;
    int total_tris = 0, total_disks = 0, total_gaussians = 0;
    for (int obj = 0; obj < num_objects; ++obj) {
      if (ref_geometries[obj].type == GEOM_TRIANGLE) {
        num_meshes++;
        total_tris += (int)ref_geometries[obj].tris.size();
      } else if (ref_geometries[obj].type == GEOM_DISK) {
        num_pointclouds++;
        total_disks += (int)ref_geometries[obj].disks.size();
      } else if (ref_geometries[obj].type == GEOM_GAUSSIAN) {
        num_gaussians++;
        total_gaussians += (int)ref_geometries[obj].gaussians.size();
      }
    }
    // Build geometry types string
    std::string geom_types_str;
    if (num_meshes > 0) {
      geom_types_str += std::to_string(num_meshes) + " " + get_geometry_type_name(GEOM_TRIANGLE);
      if (num_meshes > 1) geom_types_str += "s";  // Pluralize
    }
    if (num_pointclouds > 0) {
      if (!geom_types_str.empty()) geom_types_str += ", ";
      geom_types_str += std::to_string(num_pointclouds) + " " + get_geometry_type_name(GEOM_DISK);
      if (num_pointclouds > 1) geom_types_str += "s";  // Pluralize
    }
    if (num_gaussians > 0) {
      if (!geom_types_str.empty()) geom_types_str += ", ";
      geom_types_str += std::to_string(num_gaussians) + " " + get_geometry_type_name(GEOM_GAUSSIAN);
      if (num_gaussians > 1) geom_types_str += "s";  // Pluralize
    }
    std::cout << "Geometry types: " << geom_types_str << std::endl;
    if (total_tris > 0) std::cout << "Total " << get_geometry_element_name(GEOM_TRIANGLE) << ": " << total_tris << std::endl;
    if (total_disks > 0) std::cout << "Total " << get_geometry_element_name(GEOM_DISK) << ": " << total_disks << std::endl;
    if (total_gaussians > 0) std::cout << "Total " << get_geometry_element_name(GEOM_GAUSSIAN) << ": " << total_gaussians << std::endl;
    
    std::cout << "--- Timing (ms) ---" << std::endl;
    std::cout << "  Phase 1 (Form Factor J): " << std::setw(8) << phase1_time << " ms" << std::endl;
    std::cout << "  Phase 2 (Volume Matrix): " << std::setw(8) << phase2_time << " ms" << std::endl;
    if (needs_gradient(flags) || needs_hessian(flags)) {
      std::cout << "  Phase 3 (Gradients ∂A/∂θ): " << std::setw(8) << phase3_time << " ms" << std::endl;
      std::cout << "  Phase 4 (Gradients & Hessians): " << std::setw(8) << phase4_time << " ms" << std::endl;
    }
    std::cout << "  Total time:              " << std::setw(8) << total_time << " ms" << std::endl;
    std::cout << "==========================================\n" << std::endl;
  }
  
  return result;
}

} // namespace PoIntInt


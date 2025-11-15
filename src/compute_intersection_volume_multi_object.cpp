#include "compute_intersection_volume_multi_object.hpp"
#include "form_factor_helpers.hpp"
#include <cmath>
#include <complex>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <Eigen/Dense>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace PoIntInt {

IntersectionVolumeMatrixResult compute_intersection_volume_matrix_cpu(
  const std::vector<Geometry>& geometries,
  const KGrid& kgrid,
  bool enable_profiling)
{
  auto t_start = std::chrono::high_resolution_clock::now();
  
  IntersectionVolumeMatrixResult result;
  int num_objects = (int)geometries.size();
  
  if (num_objects == 0) {
    result.volume_matrix = Eigen::MatrixXd::Zero(0, 0);
    return result;
  }
  
  int Q = (int)kgrid.kmag.size();
  
  auto t_phase1_start = std::chrono::high_resolution_clock::now();
  
  // Phase 1: Compute form factor matrix J (Q × num_objects)
  // J[q, obj] = A_obj(k_q) for each k-node q and object obj
  std::vector<std::complex<double>> J(Q * num_objects);
  
  // Parallelize over k-nodes (outer loop)
  // For each k-node, compute A(k) for all objects
  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, Q),
    [&](const tbb::blocked_range<size_t>& r) {
      for (size_t q = r.begin(); q < r.end(); ++q) {
        // Get k-vector: k = kmag[q] * kdir[q]
        double kmag = (double)kgrid.kmag[q];
        const auto& kdir_arr = kgrid.dirs[q];
        Eigen::Vector3d kdir((double)kdir_arr[0], (double)kdir_arr[1], (double)kdir_arr[2]);
        Eigen::Vector3d k = kmag * kdir;
        
        // Compute A(k) for all objects at this k-node
        for (size_t obj = 0; obj < num_objects; ++obj) {
          // Compute A(k) for this geometry
          std::complex<double> A = compute_A_geometry(geometries[obj], k);
          
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
  auto t_end = std::chrono::high_resolution_clock::now();
  
  result.volume_matrix = V;
  
  if (enable_profiling) {
    auto phase1_time = std::chrono::duration_cast<std::chrono::microseconds>(t_phase1_end - t_phase1_start).count() / 1000.0;
    auto phase2_time = std::chrono::duration_cast<std::chrono::microseconds>(t_phase2_end - t_phase2_start).count() / 1000.0;
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== CPU Multi-Object Volume Computation Profiler ===" << std::endl;
    std::cout << "Number of objects: " << num_objects << std::endl;
    std::cout << "K-grid nodes: " << Q << std::endl;
    
    // Count geometry types
    int num_meshes = 0, num_pointclouds = 0, num_gaussians = 0;
    int total_tris = 0, total_disks = 0, total_gaussians = 0;
    for (const auto& geom : geometries) {
      if (geom.type == GEOM_TRIANGLE) {
        num_meshes++;
        total_tris += (int)geom.tris.size();
      } else if (geom.type == GEOM_DISK) {
        num_pointclouds++;
        total_disks += (int)geom.disks.size();
      } else if (geom.type == GEOM_GAUSSIAN) {
        num_gaussians++;
        total_gaussians += (int)geom.gaussians.size();
      }
    }
    std::cout << "Geometry types: " << num_meshes << " meshes, " << num_pointclouds << " point clouds, " << num_gaussians << " Gaussian splats" << std::endl;
    if (total_tris > 0) std::cout << "Total triangles: " << total_tris << std::endl;
    if (total_disks > 0) std::cout << "Total disks: " << total_disks << std::endl;
    if (total_gaussians > 0) std::cout << "Total Gaussians: " << total_gaussians << std::endl;
    
    std::cout << "--- Timing (ms) ---" << std::endl;
    std::cout << "  Phase 1 (Form Factor J): " << std::setw(8) << phase1_time << " ms" << std::endl;
    std::cout << "  Phase 2 (Volume Matrix): " << std::setw(8) << phase2_time << " ms" << std::endl;
    std::cout << "  Total time:              " << std::setw(8) << total_time << " ms" << std::endl;
    std::cout << "==========================================\n" << std::endl;
  }
  
  return result;
}

} // namespace PoIntInt


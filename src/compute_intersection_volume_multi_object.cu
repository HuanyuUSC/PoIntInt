#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cmath>
#include <cassert>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <math_constants.h>
#include "compute_intersection_volume_multi_object.hpp"
#include "geometry/geometry.hpp"
#include "geometry/types.hpp"
#include "cuda/cuda_helpers.hpp"
#include "dof/dof_parameterization.hpp"
#include "dof/cuda/dof_cuda_interface.hpp"
#include "quadrature/kgrid.hpp"
#include "computation_flags.hpp"

using PoIntInt::CudaKernelRegistry;
using PoIntInt::CudaComputeAkFunc;
using PoIntInt::CudaComputeAkGradientFunc;
using PoIntInt::get_dof_type_name;
using PoIntInt::get_geometry_type_name;
using PoIntInt::get_geometry_element_name;
using PoIntInt::needs_volume;
using PoIntInt::needs_gradient;
using PoIntInt::needs_hessian;
using PoIntInt::ComputationFlags;

// ===================== Helper Kernel: Copy A(k) to J Matrix =====================
// Kernel to copy per-object A(k) buffer to J matrix (row-major: J[q * num_objects + obj])
extern "C" __global__
void copy_A_to_J_kernel(
  const double2* __restrict__ A_obj,  // Input: Q elements (A(k) for one object)
  int obj,                             // Object index (column in J matrix)
  int num_objects,
  int Q,
  double2* J                           // Output: Q × num_objects (row-major)
) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= Q) return;
  
  // Copy A_obj[q] to J[q * num_objects + obj]
  J[q * num_objects + obj] = A_obj[q];
}

// ===================== Phase 2: Compute Volume Matrix V = J^T D J =====================
// Kernel to compute V[i,j] = (1/(8π³)) · Σ_q w_q · Re(J[q,i] · conj(J[q,j]))
// Each thread computes one entry (i,j) of the symmetric matrix
extern "C" __global__
void compute_volume_matrix_kernel(
  const double2* __restrict__ J,                  // Input: Q × num_objects (row-major)
  const double* __restrict__ weights,             // Q weights
  int Q,
  int num_objects,
  double* V                                       // Output: num_objects × num_objects (row-major, symmetric)
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (i >= num_objects || j >= num_objects) return;
  
  // Only compute upper triangle (including diagonal)
  if (i > j) return;
  
  double sum = 0.0;
  
  // Compute V[i,j] = Σ_q w_q · Re(J[q,i] · conj(J[q,j]))
  for (int q = 0; q < Q; ++q) {
    int idx_i = q * num_objects + i;
    int idx_j = q * num_objects + j;
    double2 Ji = J[idx_i];
    double2 Jj = J[idx_j];
    
    // J[q,i] · conj(J[q,j]) = (Ji.x + i*Ji.y) · (Jj.x - i*Jj.y)
    //                        = Ji.x*Jj.x + Ji.y*Jj.y + i*(Ji.y*Jj.x - Ji.x*Jj.y)
    // Real part: Ji.x*Jj.x + Ji.y*Jj.y
    double real_part = Ji.x * Jj.x + Ji.y * Jj.y;
    sum += weights[q] * real_part;
  }
  
  // V = (1/(8π³)) · sum
  double V_ij = sum / (8.0 * CUDART_PI * CUDART_PI * CUDART_PI);
  
  // Write to both (i,j) and (j,i) for symmetry
  int idx_ij = i * num_objects + j;
  int idx_ji = j * num_objects + i;
  V[idx_ij] = V_ij;
  if (i != j) {
    V[idx_ji] = V_ij;
  }
}

// ===================== Host Implementation =====================
namespace PoIntInt {

IntersectionVolumeMatrixResult compute_intersection_volume_matrix_cuda(
  const std::vector<Geometry>& ref_geometries,
  const std::vector<std::shared_ptr<DoFParameterization>>& dofs,
  const std::vector<Eigen::VectorXd>& dof_vectors,
  const KGrid& kgrid,
  ComputationFlags flags,
  int blockSize,
  bool enable_profiling)
{
  // Start total timing
  auto t_start_total = std::chrono::high_resolution_clock::now();
  
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
  
  // For now, only VOLUME_ONLY is supported
  if (needs_gradient(flags) || needs_hessian(flags)) {
    std::cerr << "Warning: Gradients and Hessians for multi-object intersection volume matrix are not yet implemented. Computing volume only." << std::endl;
    flags = ComputationFlags::VOLUME_ONLY;
  }
  
  int Q = (int)kgrid.kmag.size();
  
  // Get DoF type names and check for CUDA kernels
  std::vector<std::string> dof_types(num_objects);
  std::vector<std::pair<CudaComputeAkFunc, CudaComputeAkGradientFunc>> kernels(num_objects);
  bool all_kernels_available = true;
  
  for (int obj = 0; obj < num_objects; ++obj) {
    dof_types[obj] = get_dof_type_name(dofs[obj]);
    bool has_cuda = CudaKernelRegistry::has_kernels(dof_types[obj], ref_geometries[obj].type);
    
    if (!has_cuda) {
      std::cerr << "Error: CUDA kernels not available for DoF type '" << dof_types[obj] 
                << "' with geometry type " << (int)ref_geometries[obj].type 
                << " for object " << obj << std::endl;
      all_kernels_available = false;
      break;
    }
    
    kernels[obj] = CudaKernelRegistry::get_kernels(dof_types[obj], ref_geometries[obj].type);
    
    if (!kernels[obj].first) {
      std::cerr << "Error: Failed to retrieve CUDA kernel function from registry for object " << obj << std::endl;
      all_kernels_available = false;
      break;
    }
  }
  
  if (!all_kernels_available) {
    result.volume_matrix = Eigen::MatrixXd::Zero(num_objects, num_objects);
    return result;
  }
  
  // Allocate device memory
  double2* d_J = nullptr;
  double* d_weights = nullptr;
  double* d_V = nullptr;
  
  // Per-object A(k) buffers (temporary, for calling kernels)
  std::vector<double2*> d_A_objects(num_objects, nullptr);
  
  // Host result buffer (declare early to avoid goto issues)
  std::vector<double> h_V(num_objects * num_objects);
  
  #define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
      goto cleanup; \
    } \
  } while(0)
  
  auto t_malloc_start = std::chrono::high_resolution_clock::now();
  
  // Allocate J matrix (Q × num_objects complex numbers, row-major)
  CUDA_CHECK(cudaMalloc(&d_J, Q * num_objects * sizeof(double2)));
  
  // Allocate per-object A(k) buffers
  for (int obj = 0; obj < num_objects; ++obj) {
    CUDA_CHECK(cudaMalloc(&d_A_objects[obj], Q * sizeof(double2)));
  }
  
  // Allocate k-grid weights
  CUDA_CHECK(cudaMalloc(&d_weights, Q * sizeof(double)));
  
  // Allocate volume matrix
  CUDA_CHECK(cudaMalloc(&d_V, num_objects * num_objects * sizeof(double)));
  
  auto t_malloc_end = std::chrono::high_resolution_clock::now();
  auto t_memcpy_start = std::chrono::high_resolution_clock::now();
  
  // Copy k-grid weights
  CUDA_CHECK(cudaMemcpy(d_weights, kgrid.w.data(), Q * sizeof(double), cudaMemcpyHostToDevice));
  
  auto t_memcpy_end = std::chrono::high_resolution_clock::now();
  auto t_kernel1_start = std::chrono::high_resolution_clock::now();
  
  // Phase 1: Compute form factor matrix J using DoF-specific kernels
  // For each object, compute A(k) for all k-nodes using its DoF parameterization
  for (int obj = 0; obj < num_objects; ++obj) {
    // Call the registered kernel to compute A(k) for this object
    kernels[obj].first(ref_geometries[obj], kgrid, dof_vectors[obj], d_A_objects[obj], blockSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy from per-object buffer to J matrix (row-major: J[q * num_objects + obj])
    // Use a simple kernel to copy A_obj[q] to J[q * num_objects + obj]
    dim3 grid_copy((Q + blockSize - 1) / blockSize);
    dim3 block_copy(blockSize);
    copy_A_to_J_kernel<<<grid_copy, block_copy>>>(
      d_A_objects[obj], obj, num_objects, Q, d_J
    );
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  
  auto t_kernel1_end = std::chrono::high_resolution_clock::now();
  auto t_kernel2_start = std::chrono::high_resolution_clock::now();
  
  // Phase 2: Compute volume matrix V = J^T D J
  // Use 16x16 thread blocks for matrix computation
  dim3 block_V(16, 16);
  dim3 grid_V((num_objects + block_V.x - 1) / block_V.x,
              (num_objects + block_V.y - 1) / block_V.y);
  compute_volume_matrix_kernel<<<grid_V, block_V>>>(
    d_J, d_weights, Q, num_objects, d_V
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
  auto t_kernel2_end = std::chrono::high_resolution_clock::now();
  auto t_result_start = std::chrono::high_resolution_clock::now();
  
  // Copy result back
  CUDA_CHECK(cudaMemcpy(h_V.data(), d_V, num_objects * num_objects * sizeof(double), cudaMemcpyDeviceToHost));
  
  auto t_result_end = std::chrono::high_resolution_clock::now();
  
  // Convert to Eigen matrix
  result.volume_matrix = Eigen::Map<Eigen::MatrixXd>(h_V.data(), num_objects, num_objects);
  
  // End total timing
  auto t_end_total = std::chrono::high_resolution_clock::now();
  
  // Calculate timing statistics
  auto malloc_time = std::chrono::duration_cast<std::chrono::microseconds>(t_malloc_end - t_malloc_start).count() / 1000.0;
  auto memcpy_time = std::chrono::duration_cast<std::chrono::microseconds>(t_memcpy_end - t_memcpy_start).count() / 1000.0;
  auto kernel1_time = std::chrono::duration_cast<std::chrono::microseconds>(t_kernel1_end - t_kernel1_start).count() / 1000.0;
  auto kernel2_time = std::chrono::duration_cast<std::chrono::microseconds>(t_kernel2_end - t_kernel2_start).count() / 1000.0;
  auto result_time = std::chrono::duration_cast<std::chrono::microseconds>(t_result_end - t_result_start).count() / 1000.0;
  auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end_total - t_start_total).count() / 1000.0;
  
  // Print timing information if profiling is enabled
  if (enable_profiling) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== CUDA Multi-Object Volume Computation Profiler ===" << std::endl;
    std::cout << "Number of objects: " << num_objects << std::endl;
    std::cout << "K-grid nodes: " << Q << std::endl;
    std::cout << "Block size: " << blockSize << std::endl;
    
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
    std::cout << "  Memory allocation: " << std::setw(8) << malloc_time << " ms" << std::endl;
    std::cout << "  Memory copy (H->D): " << std::setw(8) << memcpy_time << " ms" << std::endl;
    std::cout << "  Kernel 1 (Form Factor J): " << std::setw(8) << kernel1_time << " ms" << std::endl;
    std::cout << "  Kernel 2 (Volume Matrix): " << std::setw(8) << kernel2_time << " ms" << std::endl;
    std::cout << "  Memory copy (D->H): " << std::setw(8) << result_time << " ms" << std::endl;
    std::cout << "  Total time:         " << std::setw(8) << total_time << " ms" << std::endl;
    std::cout << "==========================================\n" << std::endl;
  }
  
  cleanup:
  // Cleanup
  if (d_J) cudaFree(d_J);
  if (d_weights) cudaFree(d_weights);
  if (d_V) cudaFree(d_V);
  for (int obj = 0; obj < num_objects; ++obj) {
    if (d_A_objects[obj]) cudaFree(d_A_objects[obj]);
  }
  
  #undef CUDA_CHECK
  
  return result;
}

} // namespace PoIntInt

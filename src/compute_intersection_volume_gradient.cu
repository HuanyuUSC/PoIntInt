#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cmath>
#include <cassert>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <math_constants.h>
#include <Eigen/Dense>
#include <complex>
#include "compute_intersection_volume_gradient.hpp"
#include "compute_intersection_volume.hpp"
#include "geometry/geometry.hpp"
#include "geometry/types.hpp"
#include "dof/dof_parameterization.hpp"
#include "dof/cuda/dof_cuda_interface.hpp"
#include "quadrature/kgrid.hpp"

using PoIntInt::TriPacked;
using PoIntInt::DiskPacked;
using PoIntInt::GaussianPacked;
using PoIntInt::CudaKernelRegistry;
using PoIntInt::get_dof_type_name;

// ===================== Phase 2: Compute Intersection Volume Gradient =====================

// Kernel to compute gradient of intersection volume and volume itself
// Parallelizes over k-nodes (Q threads)
// Each thread processes one k-node and accumulates contributions to:
//   - Volume: Re(A1 * conj(A2)) * weight
//   - Gradients: Re(dA1 * conj(A2)) * weight and Re(A1 * conj(dA2)) * weight
// ∂V/∂θ₁ = (1/(8π³)) · Σ_q w_q · Re( (∂A₁(k_q)/∂θ₁) · conj(A₂(k_q)) )
// ∂V/∂θ₂ = (1/(8π³)) · Σ_q w_q · Re( A₁(k_q) · conj(∂A₂(k_q)/∂θ₂) )
// V = (1/(8π³)) · Σ_q w_q · Re( A₁(k_q) · conj(A₂(k_q)) )
__global__ void compute_intersection_volume_gradient_kernel(
  const double2* __restrict__ grad_A1,  // Q × num_dofs1: ∂A₁(k_q)/∂θ₁
  const double2* __restrict__ grad_A2,  // Q × num_dofs2: ∂A₂(k_q)/∂θ₂
  const double2* __restrict__ A1,       // Q: A₁(k_q)
  const double2* __restrict__ A2,       // Q: A₂(k_q)
  const double* __restrict__ weights,  // Q weights
  int Q,
  int num_dofs1,
  int num_dofs2,
  double* grad_V1,  // Output: num_dofs1 (accumulated with atomics)
  double* grad_V2,  // Output: num_dofs2 (accumulated with atomics)
  double* volume    // Output: volume (accumulated with atomics)
) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= Q) return;
  
  double w = weights[q];
  double2 a1 = A1[q];
  double2 a2 = A2[q];
  
  // Compute volume contribution: Re(A1 * conj(A2)) = Re(A1) * Re(A2) + Im(A1) * Im(A2)
  double vol_contrib = a1.x * a2.x + a1.y * a2.y;
  atomicAdd(volume, w * vol_contrib);
  
  // Compute gradient contributions for geometry 1
  // For each DoF, accumulate: Re(dA1 * conj(A2)) * weight
  for (int dof = 0; dof < num_dofs1; ++dof) {
    double2 dA1 = grad_A1[q * num_dofs1 + dof];
    // Re(dA1 * conj(A2)) = Re(dA1) * Re(A2) + Im(dA1) * Im(A2)
    double grad_contrib = dA1.x * a2.x + dA1.y * a2.y;
    atomicAdd(&grad_V1[dof], w * grad_contrib);
  }
  
  // Compute gradient contributions for geometry 2
  // For each DoF, accumulate: Re(A1 * conj(dA2)) * weight
  for (int dof = 0; dof < num_dofs2; ++dof) {
    double2 dA2 = grad_A2[q * num_dofs2 + dof];
    // Re(A1 * conj(dA2)) = Re(A1) * Re(dA2) + Im(A1) * Im(dA2)
    double grad_contrib = a1.x * dA2.x + a1.y * dA2.y;
    atomicAdd(&grad_V2[dof], w * grad_contrib);
  }
}

// ===================== Host Implementation =====================

namespace PoIntInt {

IntersectionVolumeGradientResult compute_intersection_volume_gradient_cuda(
  const Geometry& geom1,
  const Geometry& geom2,
  const std::shared_ptr<DoFParameterization>& dof1,
  const std::shared_ptr<DoFParameterization>& dof2,
  const Eigen::VectorXd& dofs1,
  const Eigen::VectorXd& dofs2,
  const KGrid& kgrid,
  int blockSize,
  bool enable_profiling)
{
  // Declare all variables at the top to avoid goto issues
  auto t_start = std::chrono::high_resolution_clock::now();
  IntersectionVolumeGradientResult result;
  
  // Device memory pointers (initialized to nullptr)
  // Note: Wrapper functions handle their own temporary allocations for geometry data, k-grid, and DoFs
  double* d_weights = nullptr;
  double2* d_A1 = nullptr;
  double2* d_A2 = nullptr;
  double2* d_grad_A1 = nullptr;
  double2* d_grad_A2 = nullptr;
  double* d_grad_V1 = nullptr;
  double* d_grad_V2 = nullptr;
  double* d_volume = nullptr;
  
  // Host vectors
  std::vector<double> h_grad_V1, h_grad_V2;
  
  // Timing variables
  auto t_alloc_start = std::chrono::high_resolution_clock::now();
  auto t_alloc_end = std::chrono::high_resolution_clock::now();
  auto t_memcpy_start = std::chrono::high_resolution_clock::now();
  auto t_memcpy_end = std::chrono::high_resolution_clock::now();
  auto t_kernel1_start = std::chrono::high_resolution_clock::now();
  auto t_kernel1_end = std::chrono::high_resolution_clock::now();
  auto t_kernel2_start = std::chrono::high_resolution_clock::now();
  auto t_kernel2_end = std::chrono::high_resolution_clock::now();
  auto t_kernel3_start = std::chrono::high_resolution_clock::now();
  auto t_kernel3_end = std::chrono::high_resolution_clock::now();
  auto t_result_start = std::chrono::high_resolution_clock::now();
  auto t_result_end = std::chrono::high_resolution_clock::now();
  auto t_end = std::chrono::high_resolution_clock::now();
  
  // Grid/block dimensions
  dim3 grid_vol_grad, block_vol_grad;
  
  // Other variables
  int NF1 = 0, NF2 = 0;
  int Q = 0;
  int num_dofs1 = 0, num_dofs2 = 0;
  std::string dof_type1, dof_type2;  // DoF type names (for profiling)
  
  #define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
      goto cleanup; \
    } \
  } while(0)
  
  // Get DoF type names and check if CUDA kernels are available
  dof_type1 = get_dof_type_name(dof1);
  dof_type2 = get_dof_type_name(dof2);
  
  if (!CudaKernelRegistry::has_kernels(dof_type1, geom1.type)) {
    std::cerr << "Error: No CUDA kernels registered for DoF type '" << dof_type1 
              << "' with geometry type " << (int)geom1.type << std::endl;
    result.grad_geom1 = Eigen::VectorXd::Zero(dofs1.size());
    result.grad_geom2 = Eigen::VectorXd::Zero(dofs2.size());
    result.volume = 0.0;
    return result;
  }
  
  if (!CudaKernelRegistry::has_kernels(dof_type2, geom2.type)) {
    std::cerr << "Error: No CUDA kernels registered for DoF type '" << dof_type2 
              << "' with geometry type " << (int)geom2.type << std::endl;
    result.grad_geom1 = Eigen::VectorXd::Zero(dofs1.size());
    result.grad_geom2 = Eigen::VectorXd::Zero(dofs2.size());
    result.volume = 0.0;
    return result;
  }
  
  // Get registered kernel functions
  auto kernels1 = CudaKernelRegistry::get_kernels(dof_type1, geom1.type);
  auto kernels2 = CudaKernelRegistry::get_kernels(dof_type2, geom2.type);
  
  if (!kernels1.first || !kernels1.second || !kernels2.first || !kernels2.second) {
    std::cerr << "Error: Failed to retrieve CUDA kernel functions from registry" << std::endl;
    result.grad_geom1 = Eigen::VectorXd::Zero(dofs1.size());
    result.grad_geom2 = Eigen::VectorXd::Zero(dofs2.size());
    result.volume = 0.0;
    return result;
  }
  
  Q = (int)kgrid.kmag.size();
  num_dofs1 = (int)dofs1.size();
  num_dofs2 = (int)dofs2.size();
  
  // Get geometry sizes for profiling
  if (geom1.type == GEOM_TRIANGLE) {
    NF1 = (int)geom1.tris.size();
  } else if (geom1.type == GEOM_DISK) {
    NF1 = (int)geom1.disks.size();
  } else if (geom1.type == GEOM_GAUSSIAN) {
    NF1 = (int)geom1.gaussians.size();
  } else {
    NF1 = 0;
  }
  
  if (geom2.type == GEOM_TRIANGLE) {
    NF2 = (int)geom2.tris.size();
  } else if (geom2.type == GEOM_DISK) {
    NF2 = (int)geom2.disks.size();
  } else if (geom2.type == GEOM_GAUSSIAN) {
    NF2 = (int)geom2.gaussians.size();
  } else {
    NF2 = 0;
  }
  
  t_alloc_start = std::chrono::high_resolution_clock::now();
  
  // Allocate output buffers (wrapper functions handle their own temporary allocations)
  // Allocate form factors and gradients
  CUDA_CHECK(cudaMalloc(&d_A1, Q * sizeof(double2)));
  CUDA_CHECK(cudaMalloc(&d_A2, Q * sizeof(double2)));
  CUDA_CHECK(cudaMalloc(&d_grad_A1, Q * num_dofs1 * sizeof(double2)));
  CUDA_CHECK(cudaMalloc(&d_grad_A2, Q * num_dofs2 * sizeof(double2)));
  
  // Allocate k-grid weights (needed for Phase 3)
  CUDA_CHECK(cudaMalloc(&d_weights, Q * sizeof(double)));
  
  // Allocate output gradients and volume
  CUDA_CHECK(cudaMalloc(&d_grad_V1, num_dofs1 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_grad_V2, num_dofs2 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_volume, sizeof(double)));
  
  t_alloc_end = std::chrono::high_resolution_clock::now();
  t_memcpy_start = std::chrono::high_resolution_clock::now();
  
  // Copy k-grid weights to device (wrapper functions handle kdirs and kmags)
  CUDA_CHECK(cudaMemcpy(d_weights, kgrid.w.data(), Q * sizeof(double), cudaMemcpyHostToDevice));
  
  // Initialize gradients and volume to zero
  CUDA_CHECK(cudaMemset(d_grad_V1, 0, num_dofs1 * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_grad_V2, 0, num_dofs2 * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_volume, 0, sizeof(double)));
  
  t_memcpy_end = std::chrono::high_resolution_clock::now();
  t_kernel1_start = std::chrono::high_resolution_clock::now();
  
  // Phase 1: Compute A(k) for both geometries using registered CUDA kernels
  // Create temporary KGrid structures for the wrapper functions
  // Note: The wrapper functions handle all memory allocation and kernel launches internally
  kernels1.first(geom1, kgrid, dofs1, d_A1, blockSize);
  CUDA_CHECK(cudaDeviceSynchronize());
  
  kernels2.first(geom2, kgrid, dofs2, d_A2, blockSize);
  CUDA_CHECK(cudaDeviceSynchronize());
  
  t_kernel1_end = std::chrono::high_resolution_clock::now();
  t_kernel2_start = std::chrono::high_resolution_clock::now();
  
  // Phase 2: Compute ∂A(k)/∂θ for both geometries using registered CUDA kernels
  kernels1.second(geom1, kgrid, dofs1, d_grad_A1, blockSize);
  CUDA_CHECK(cudaDeviceSynchronize());
  
  kernels2.second(geom2, kgrid, dofs2, d_grad_A2, blockSize);
  CUDA_CHECK(cudaDeviceSynchronize());
  
  t_kernel2_end = std::chrono::high_resolution_clock::now();
  t_kernel3_start = std::chrono::high_resolution_clock::now();
  
  // Phase 3: Compute intersection volume gradient and volume
  // Parallelize over k-nodes (Q threads)
  grid_vol_grad = dim3((Q + blockSize - 1) / blockSize);
  block_vol_grad = dim3(blockSize);
  
  compute_intersection_volume_gradient_kernel<<<grid_vol_grad, block_vol_grad>>>(
    d_grad_A1, d_grad_A2, d_A1, d_A2, d_weights, Q, num_dofs1, num_dofs2, d_grad_V1, d_grad_V2, d_volume
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
  t_kernel3_end = std::chrono::high_resolution_clock::now();
  t_result_start = std::chrono::high_resolution_clock::now();
  
  // Copy results back
  h_grad_V1.resize(num_dofs1);
  h_grad_V2.resize(num_dofs2);
  CUDA_CHECK(cudaMemcpy(h_grad_V1.data(), d_grad_V1, num_dofs1 * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_grad_V2.data(), d_grad_V2, num_dofs2 * sizeof(double), cudaMemcpyDeviceToHost));
  
  double h_volume = 0.0;
  CUDA_CHECK(cudaMemcpy(&h_volume, d_volume, sizeof(double), cudaMemcpyDeviceToHost));
  
  // Apply scaling factor: 1/(8π³)
  const double scale = 1.0 / (8.0 * CUDART_PI * CUDART_PI * CUDART_PI);
  for (int i = 0; i < num_dofs1; ++i) {
    h_grad_V1[i] *= scale;
  }
  for (int i = 0; i < num_dofs2; ++i) {
    h_grad_V2[i] *= scale;
  }
  result.volume = h_volume * scale;
  
  result.grad_geom1 = Eigen::Map<Eigen::VectorXd>(h_grad_V1.data(), num_dofs1);
  result.grad_geom2 = Eigen::Map<Eigen::VectorXd>(h_grad_V2.data(), num_dofs2);
  
  t_result_end = std::chrono::high_resolution_clock::now();
  t_end = std::chrono::high_resolution_clock::now();
  
  if (enable_profiling) {
    auto alloc_time = std::chrono::duration_cast<std::chrono::microseconds>(t_alloc_end - t_alloc_start).count() / 1000.0;
    auto memcpy_time = std::chrono::duration_cast<std::chrono::microseconds>(t_memcpy_end - t_memcpy_start).count() / 1000.0;
    auto kernel1_time = std::chrono::duration_cast<std::chrono::microseconds>(t_kernel1_end - t_kernel1_start).count() / 1000.0;
    auto kernel2_time = std::chrono::duration_cast<std::chrono::microseconds>(t_kernel2_end - t_kernel2_start).count() / 1000.0;
    auto kernel3_time = std::chrono::duration_cast<std::chrono::microseconds>(t_kernel3_end - t_kernel3_start).count() / 1000.0;
    auto result_time = std::chrono::duration_cast<std::chrono::microseconds>(t_result_end - t_result_start).count() / 1000.0;
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== CUDA Intersection Volume Gradient Profiler ===" << std::endl;
    std::cout << "Geometry 1: " << get_geometry_type_name(geom1.type) << " (" << NF1 << " " << get_geometry_element_name(geom1.type) << "), DoF: " << dof_type1 << std::endl;
    std::cout << "Geometry 2: " << get_geometry_type_name(geom2.type) << " (" << NF2 << " " << get_geometry_element_name(geom2.type) << "), DoF: " << dof_type2 << std::endl;
    std::cout << "K-grid nodes: " << Q << std::endl;
    std::cout << "DoFs per geometry: " << num_dofs1 << ", " << num_dofs2 << std::endl;
    std::cout << "--- Timing (ms) ---" << std::endl;
    std::cout << "  Memory allocation: " << std::setw(8) << alloc_time << " ms" << std::endl;
    std::cout << "  Memory copy (H->D): " << std::setw(8) << memcpy_time << " ms" << std::endl;
    std::cout << "  Phase 1 (A(k)):     " << std::setw(8) << kernel1_time << " ms" << std::endl;
    std::cout << "  Phase 2 (∂A/∂θ):    " << std::setw(8) << kernel2_time << " ms" << std::endl;
    std::cout << "  Phase 3 (∇V):        " << std::setw(8) << kernel3_time << " ms" << std::endl;
    std::cout << "  Memory copy (D->H): " << std::setw(8) << result_time << " ms" << std::endl;
    std::cout << "  Total time:          " << std::setw(8) << total_time << " ms" << std::endl;
    std::cout << "==========================================\n" << std::endl;
  }
  
cleanup:
  if (d_weights) cudaFree(d_weights);
  if (d_A1) cudaFree(d_A1);
  if (d_A2) cudaFree(d_A2);
  if (d_grad_A1) cudaFree(d_grad_A1);
  if (d_grad_A2) cudaFree(d_grad_A2);
  if (d_grad_V1) cudaFree(d_grad_V1);
  if (d_grad_V2) cudaFree(d_grad_V2);
  if (d_volume) cudaFree(d_volume);
  
  #undef CUDA_CHECK
  
  return result;
}

// ============================================================================
// Phase 1: Compute A(k) for a single k-vector (for testing)
// ============================================================================

std::complex<double> compute_Ak_cuda(
  const Geometry& geom,
  const Eigen::Vector3d& k,
  const std::shared_ptr<DoFParameterization>& dof,
  const Eigen::VectorXd& dofs,
  int blockSize)
{
  // Get DoF type name and check if CUDA kernels are available
  std::string dof_type = get_dof_type_name(dof);
  
  if (!CudaKernelRegistry::has_kernels(dof_type, geom.type)) {
    std::cerr << "Error: No CUDA kernels registered for DoF type '" << dof_type 
              << "' with geometry type " << (int)geom.type << std::endl;
    return std::complex<double>(0.0, 0.0);
  }
  
  // Get registered kernel function
  auto kernels = CudaKernelRegistry::get_kernels(dof_type, geom.type);
  
  if (!kernels.first) {
    std::cerr << "Error: Failed to retrieve CUDA kernel function from registry" << std::endl;
    return std::complex<double>(0.0, 0.0);
  }
  
  // Prepare k-vector
  double kmag = k.norm();
  if (kmag < 1e-10) return std::complex<double>(0.0, 0.0);
  
  // Create a single-point KGrid
  KGrid single_kgrid;
  single_kgrid.dirs.push_back({k.x() / kmag, k.y() / kmag, k.z() / kmag});
  single_kgrid.kmag.push_back(kmag);
  single_kgrid.w.push_back(1.0);
  
  // Allocate output buffer
  double2* d_A = nullptr;
  double2 h_A = make_double2(0.0, 0.0);
  
  #define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
      goto cleanup; \
    } \
  } while(0)
  
  CUDA_CHECK(cudaMalloc(&d_A, sizeof(double2)));
  
  // Call registered kernel wrapper
  kernels.first(geom, single_kgrid, dofs, d_A, blockSize);
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Copy result back
  CUDA_CHECK(cudaMemcpy(&h_A, d_A, sizeof(double2), cudaMemcpyDeviceToHost));
  
  cleanup:
  if (d_A) cudaFree(d_A);
  
  #undef CUDA_CHECK
  
  return std::complex<double>(h_A.x, h_A.y);
}

// ============================================================================
// Phase 2: Compute ∂A(k)/∂θ for a single k-vector (for testing)
// ============================================================================

Eigen::VectorXcd compute_Ak_gradient_cuda(
  const Geometry& geom,
  const Eigen::Vector3d& k,
  const std::shared_ptr<DoFParameterization>& dof,
  const Eigen::VectorXd& dofs,
  int blockSize)
{
  // Get DoF type name and check if CUDA kernels are available
  std::string dof_type = get_dof_type_name(dof);
  
  if (!CudaKernelRegistry::has_kernels(dof_type, geom.type)) {
    std::cerr << "Error: No CUDA kernels registered for DoF type '" << dof_type 
              << "' with geometry type " << (int)geom.type << std::endl;
    return Eigen::VectorXcd::Zero(dof->num_dofs());
  }
  
  // Get registered kernel function
  auto kernels = CudaKernelRegistry::get_kernels(dof_type, geom.type);
  
  if (!kernels.second) {
    std::cerr << "Error: Failed to retrieve CUDA gradient kernel function from registry" << std::endl;
    return Eigen::VectorXcd::Zero(dof->num_dofs());
  }
  
  int num_dofs = (int)dof->num_dofs();
  
  // Prepare k-vector
  double kmag = k.norm();
  if (kmag < 1e-10) return Eigen::VectorXcd::Zero(num_dofs);
  
  // Create a single-point KGrid
  KGrid single_kgrid;
  single_kgrid.dirs.push_back({k.x() / kmag, k.y() / kmag, k.z() / kmag});
  single_kgrid.kmag.push_back(kmag);
  single_kgrid.w.push_back(1.0);
  
  // Allocate output buffer
  double2* d_grad_A = nullptr;
  std::vector<double2> h_grad_A(num_dofs);
  Eigen::VectorXcd grad(num_dofs);
  
  #define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
      goto cleanup; \
    } \
  } while(0)
  
  CUDA_CHECK(cudaMalloc(&d_grad_A, num_dofs * sizeof(double2)));
  
  // Call registered kernel wrapper
  kernels.second(geom, single_kgrid, dofs, d_grad_A, blockSize);
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Copy result back
  CUDA_CHECK(cudaMemcpy(h_grad_A.data(), d_grad_A, num_dofs * sizeof(double2), cudaMemcpyDeviceToHost));
  
  cleanup:
  if (d_grad_A) cudaFree(d_grad_A);
  
  #undef CUDA_CHECK
  
  // Convert result to Eigen::VectorXcd
  for (int i = 0; i < num_dofs; ++i) {
    grad(i) = std::complex<double>(h_grad_A[i].x, h_grad_A[i].y);
  }
  
  return grad;
}

} // namespace PoIntInt


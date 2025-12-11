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
#include "compute_intersection_volume.hpp"
#include "geometry/geometry.hpp"
#include "geometry/types.hpp"
#include "cuda/cuda_helpers.hpp"
#include "dof/dof_parameterization.hpp"
#include "dof/cuda/dof_cuda_interface.hpp"
#include "quadrature/kgrid.hpp"

using PoIntInt::CudaKernelRegistry;
using PoIntInt::CudaComputeAkFunc;
using PoIntInt::CudaComputeAkGradientFunc;
using PoIntInt::get_dof_type_name;
using PoIntInt::get_geometry_type_name;
using PoIntInt::get_geometry_element_name;

// ============================================================================
// CUDA Kernels for Volume, Gradient, and Hessian Computation
// ============================================================================

// Kernel to compute intersection volume only (no gradients)
// Parallelizes over k-nodes (Q threads)
__global__ void compute_intersection_volume_kernel(
  const double2* __restrict__ A1,       // Q: A₁(k_q)
  const double2* __restrict__ A2,       // Q: A₂(k_q)
  const double* __restrict__ weights,  // Q weights
  int Q,
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
}

// Kernel to compute gradient of intersection volume and volume itself
// Parallelizes over k-nodes (Q threads)
// Each thread processes one k-node and accumulates contributions to:
//   - Volume: Re(A1 * conj(A2)) * weight
//   - Gradients: Re(dA1 * conj(A2)) * weight and Re(A1 * conj(dA2)) * weight
// ∂V/∂θ₁ = (1/(8π³)) · Σ_q w_q · Re( (∂A₁(k_q)/∂θ₁) · conj(A₂(k_q)) )
// ∂V/∂θ₂ = (1/(8π³)) · Σ_q w_q · Re( A₁(k_q) · conj(∂A₂(k_q)/∂θ₂) )
// V = (1/(8π³)) · Σ_q w_q · Re( A₁(k_q) · conj(A₂(k_q)) )
extern "C" __global__ void compute_intersection_volume_gradient_kernel(
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

// Kernel to compute intersection volume Hessian (Gauss-Newton approximation)
// Parallelizes over k-nodes (Q threads)
// Each thread processes one k-node and accumulates outer products
// H₁₁ ≈ Σ_q w_q · (∂A₁/∂θ₁) · (∂A₁/∂θ₁)^T
// H₂₂ ≈ Σ_q w_q · (∂A₂/∂θ₂) · (∂A₂/∂θ₂)^T
// H₁₂ ≈ Σ_q w_q · (∂A₁/∂θ₁) · (∂A₂/∂θ₂)^T
__global__ void compute_intersection_volume_hessian_kernel(
  const double2* __restrict__ grad_A1,  // Q × num_dofs1: ∂A₁(k_q)/∂θ₁
  const double2* __restrict__ grad_A2,  // Q × num_dofs2: ∂A₂(k_q)/∂θ₂
  const double* __restrict__ weights,   // Q weights
  int Q,
  int num_dofs1,
  int num_dofs2,
  double* hessian_11,  // Output: num_dofs1 × num_dofs1 (row-major, accumulated with atomics)
  double* hessian_22,  // Output: num_dofs2 × num_dofs2 (row-major, accumulated with atomics)
  double* hessian_12   // Output: num_dofs1 × num_dofs2 (row-major, accumulated with atomics)
) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= Q) return;
  
  double w = weights[q];
  
  // Compute H₁₁: outer product of grad_A1 with itself
  for (int i = 0; i < num_dofs1; ++i) {
    double2 dA1_i = grad_A1[q * num_dofs1 + i];
    for (int j = 0; j < num_dofs1; ++j) {
      double2 dA1_j = grad_A1[q * num_dofs1 + j];
      // Real part of (dA1_i * conj(dA1_j)) = Re(dA1_i) * Re(dA1_j) + Im(dA1_i) * Im(dA1_j)
      double hess_contrib = dA1_i.x * dA1_j.x + dA1_i.y * dA1_j.y;
      atomicAdd(&hessian_11[i * num_dofs1 + j], w * hess_contrib);
    }
  }
  
  // Compute H₂₂: outer product of grad_A2 with itself
  for (int i = 0; i < num_dofs2; ++i) {
    double2 dA2_i = grad_A2[q * num_dofs2 + i];
    for (int j = 0; j < num_dofs2; ++j) {
      double2 dA2_j = grad_A2[q * num_dofs2 + j];
      // Real part of (dA2_i * conj(dA2_j)) = Re(dA2_i) * Re(dA2_j) + Im(dA2_i) * Im(dA2_j)
      double hess_contrib = dA2_i.x * dA2_j.x + dA2_i.y * dA2_j.y;
      atomicAdd(&hessian_22[i * num_dofs2 + j], w * hess_contrib);
    }
  }
  
  // Compute H₁₂: outer product of grad_A1 with grad_A2
  for (int i = 0; i < num_dofs1; ++i) {
    double2 dA1_i = grad_A1[q * num_dofs1 + i];
    for (int j = 0; j < num_dofs2; ++j) {
      double2 dA2_j = grad_A2[q * num_dofs2 + j];
      // Real part of (dA1_i * conj(dA2_j)) = Re(dA1_i) * Re(dA2_j) + Im(dA1_i) * Im(dA2_j)
      double hess_contrib = dA1_i.x * dA2_j.x + dA1_i.y * dA2_j.y;
      atomicAdd(&hessian_12[i * num_dofs2 + j], w * hess_contrib);
    }
  }
}

// ============================================================================
// Host Implementation: Intersection Volume
// ============================================================================
// Note: Self-volume computation is NOT implemented here. Self-volume should
// be computed using the divergence theorem: V = (1/3) ∫_S (x, y, z) · n dS,
// where S is the boundary surface and n is the outward normal. This requires
// DoF-specific implementations of local contributions (area weight × n · position)
// for each element. See DoFParameterization::compute_divergence_contribution()
// and related methods. This will be implemented in a future phase.
// ============================================================================

namespace PoIntInt {

IntersectionVolumeResult compute_intersection_volume_cuda(
  const Geometry& ref_geom1,
  const Geometry& ref_geom2,
  const std::shared_ptr<DoFParameterization>& dof1,
  const std::shared_ptr<DoFParameterization>& dof2,
  const Eigen::VectorXd& dofs1,
  const Eigen::VectorXd& dofs2,
  const KGrid& kgrid,
  ComputationFlags flags,
  int blockSize,
  bool enable_profiling)
{
  // Declare all variables at the top to avoid goto issues
  auto t_start = std::chrono::high_resolution_clock::now();
  IntersectionVolumeResult result;
  result.volume = 0.0;
  
  int num_dofs1 = dof1->num_dofs();
  int num_dofs2 = dof2->num_dofs();
  int Q = kgrid.dirs.size();
  
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
  
  // Device memory pointers
  double* d_weights = nullptr;
  double2* d_A1 = nullptr;
  double2* d_A2 = nullptr;
  double2* d_grad_A1 = nullptr;
  double2* d_grad_A2 = nullptr;
  double* d_grad_V1 = nullptr;
  double* d_grad_V2 = nullptr;
  double* d_volume = nullptr;
  double* d_hessian_11 = nullptr;
  double* d_hessian_22 = nullptr;
  double* d_hessian_12 = nullptr;
  
  // Timing variables
  auto t_alloc_start = std::chrono::high_resolution_clock::now();
  auto t_alloc_end = t_alloc_start;
  auto t_memcpy_start = t_alloc_start;
  auto t_memcpy_end = t_alloc_start;
  auto t_kernel1_start = t_alloc_start;
  auto t_kernel1_end = t_alloc_start;
  auto t_kernel2_start = t_alloc_start;
  auto t_kernel2_end = t_alloc_start;
  auto t_kernel3_start = t_alloc_start;
  auto t_kernel3_end = t_alloc_start;
  auto t_kernel4_start = t_alloc_start;
  auto t_kernel4_end = t_alloc_start;
  auto t_result_start = t_alloc_start;
  auto t_result_end = t_alloc_start;
  
  dim3 grid_vol_grad, block_vol_grad;
  dim3 grid_hessian, block_hessian;
  
  // Declare variables that need to be initialized before goto cleanup
  std::string dof1_type, dof2_type;
  std::pair<CudaComputeAkFunc, CudaComputeAkGradientFunc> kernels1, kernels2;
  std::vector<double> h_weights;
  double h_volume = 0.0;
  const double scale = 1.0 / (8.0 * CUDART_PI * CUDART_PI * CUDART_PI);
  
  #define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
      goto cleanup; \
    } \
  } while(0)
  
  // Get DoF type names and check for CUDA kernels
  dof1_type = get_dof_type_name(dof1);
  dof2_type = get_dof_type_name(dof2);
  
  bool has_cuda1 = CudaKernelRegistry::has_kernels(dof1_type, ref_geom1.type);
  bool has_cuda2 = CudaKernelRegistry::has_kernels(dof2_type, ref_geom2.type);
  
  if (!has_cuda1 || !has_cuda2) {
    std::cerr << "Error: CUDA kernels not available for DoF types '" << dof1_type 
              << "' or '" << dof2_type << "' with geometry types" << std::endl;
    goto cleanup;
  }
  
  kernels1 = CudaKernelRegistry::get_kernels(dof1_type, ref_geom1.type);
  kernels2 = CudaKernelRegistry::get_kernels(dof2_type, ref_geom2.type);
  
  if (!kernels1.first || !kernels1.second || !kernels2.first || !kernels2.second) {
    std::cerr << "Error: Failed to retrieve CUDA kernel functions from registry" << std::endl;
    goto cleanup;
  }
  
  t_alloc_start = std::chrono::high_resolution_clock::now();
  
  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&d_weights, Q * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_A1, Q * sizeof(double2)));
  CUDA_CHECK(cudaMalloc(&d_A2, Q * sizeof(double2)));
  
  if (needs_gradient(flags) || needs_hessian(flags)) {
    CUDA_CHECK(cudaMalloc(&d_grad_A1, Q * num_dofs1 * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_grad_A2, Q * num_dofs2 * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_grad_V1, num_dofs1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad_V2, num_dofs2 * sizeof(double)));
  }
  
  CUDA_CHECK(cudaMalloc(&d_volume, sizeof(double)));
  
  if (needs_hessian(flags)) {
    CUDA_CHECK(cudaMalloc(&d_hessian_11, num_dofs1 * num_dofs1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_hessian_22, num_dofs2 * num_dofs2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_hessian_12, num_dofs1 * num_dofs2 * sizeof(double)));
  }
  
  t_alloc_end = std::chrono::high_resolution_clock::now();
  t_memcpy_start = t_alloc_end;
  
  // Copy weights to device
  h_weights.resize(Q);
  for (int q = 0; q < Q; ++q) {
    h_weights[q] = kgrid.w[q];
  }
  CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), Q * sizeof(double), cudaMemcpyHostToDevice));
  
  // Initialize output buffers
  CUDA_CHECK(cudaMemset(d_volume, 0, sizeof(double)));
  if (needs_gradient(flags) || needs_hessian(flags)) {
    CUDA_CHECK(cudaMemset(d_grad_V1, 0, num_dofs1 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_grad_V2, 0, num_dofs2 * sizeof(double)));
  }
  if (needs_hessian(flags)) {
    CUDA_CHECK(cudaMemset(d_hessian_11, 0, num_dofs1 * num_dofs1 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_hessian_22, 0, num_dofs2 * num_dofs2 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_hessian_12, 0, num_dofs1 * num_dofs2 * sizeof(double)));
  }
  
  t_memcpy_end = std::chrono::high_resolution_clock::now();
  t_kernel1_start = t_memcpy_end;
  
  // Phase 1: Compute A(k) for both geometries using registered CUDA kernels
  kernels1.first(ref_geom1, kgrid, dofs1, d_A1, blockSize);
  CUDA_CHECK(cudaDeviceSynchronize());
  
  kernels2.first(ref_geom2, kgrid, dofs2, d_A2, blockSize);
  CUDA_CHECK(cudaDeviceSynchronize());
  
  t_kernel1_end = std::chrono::high_resolution_clock::now();
  t_kernel2_start = t_kernel1_end;
  
  // Phase 2: Compute ∂A(k)/∂θ for both geometries (if needed)
  if (needs_gradient(flags) || needs_hessian(flags)) {
    kernels1.second(ref_geom1, kgrid, dofs1, d_grad_A1, blockSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    kernels2.second(ref_geom2, kgrid, dofs2, d_grad_A2, blockSize);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  
  t_kernel2_end = std::chrono::high_resolution_clock::now();
  t_kernel3_start = t_kernel2_end;
  
  // Phase 3: Compute volume and gradients
  if (needs_gradient(flags) || needs_hessian(flags)) {
    // Need gradients, so use the gradient kernel
    grid_vol_grad = dim3((Q + blockSize - 1) / blockSize);
    block_vol_grad = dim3(blockSize);
    
    compute_intersection_volume_gradient_kernel<<<grid_vol_grad, block_vol_grad>>>(
      d_grad_A1, d_grad_A2, d_A1, d_A2, d_weights, Q, num_dofs1, num_dofs2, 
      d_grad_V1, d_grad_V2, d_volume
    );
    CUDA_CHECK(cudaDeviceSynchronize());
  } else if (needs_volume(flags)) {
    // Only compute volume (no gradients needed) - use simpler kernel
    grid_vol_grad = dim3((Q + blockSize - 1) / blockSize);
    block_vol_grad = dim3(blockSize);
    
    compute_intersection_volume_kernel<<<grid_vol_grad, block_vol_grad>>>(
      d_A1, d_A2, d_weights, Q, d_volume
    );
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  
  t_kernel3_end = std::chrono::high_resolution_clock::now();
  t_kernel4_start = t_kernel3_end;
  
  // Phase 4: Compute Hessians (if needed)
  if (needs_hessian(flags)) {
    grid_hessian = dim3((Q + blockSize - 1) / blockSize);
    block_hessian = dim3(blockSize);
    
    compute_intersection_volume_hessian_kernel<<<grid_hessian, block_hessian>>>(
      d_grad_A1, d_grad_A2, d_weights, Q, num_dofs1, num_dofs2,
      d_hessian_11, d_hessian_22, d_hessian_12
    );
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  
  t_kernel4_end = std::chrono::high_resolution_clock::now();
  t_result_start = t_kernel4_end;
  
  // Copy results back
  CUDA_CHECK(cudaMemcpy(&h_volume, d_volume, sizeof(double), cudaMemcpyDeviceToHost));
  
  if (needs_gradient(flags) || needs_hessian(flags)) {
    std::vector<double> h_grad_V1(num_dofs1);
    std::vector<double> h_grad_V2(num_dofs2);
    CUDA_CHECK(cudaMemcpy(h_grad_V1.data(), d_grad_V1, num_dofs1 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_V2.data(), d_grad_V2, num_dofs2 * sizeof(double), cudaMemcpyDeviceToHost));
    
    if (needs_gradient(flags)) {
      result.grad_geom1 = Eigen::Map<Eigen::VectorXd>(h_grad_V1.data(), num_dofs1);
      result.grad_geom2 = Eigen::Map<Eigen::VectorXd>(h_grad_V2.data(), num_dofs2);
    }
  }
  
  if (needs_hessian(flags)) {
    std::vector<double> h_hessian_11(num_dofs1 * num_dofs1);
    std::vector<double> h_hessian_22(num_dofs2 * num_dofs2);
    std::vector<double> h_hessian_12(num_dofs1 * num_dofs2);
    
    CUDA_CHECK(cudaMemcpy(h_hessian_11.data(), d_hessian_11, num_dofs1 * num_dofs1 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hessian_22.data(), d_hessian_22, num_dofs2 * num_dofs2 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hessian_12.data(), d_hessian_12, num_dofs1 * num_dofs2 * sizeof(double), cudaMemcpyDeviceToHost));
    
    result.hessian_geom1 = Eigen::Map<Eigen::MatrixXd>(h_hessian_11.data(), num_dofs1, num_dofs1);
    result.hessian_geom2 = Eigen::Map<Eigen::MatrixXd>(h_hessian_22.data(), num_dofs2, num_dofs2);
    result.hessian_cross = Eigen::Map<Eigen::MatrixXd>(h_hessian_12.data(), num_dofs1, num_dofs2);
  }
  
  t_result_end = std::chrono::high_resolution_clock::now();
  
  // Apply scaling factor: 1/(8π³)
  result.volume = h_volume * scale;
  if (needs_gradient(flags)) {
    result.grad_geom1 *= scale;
    result.grad_geom2 *= scale;
  }
  if (needs_hessian(flags)) {
    result.hessian_geom1 *= scale;
    result.hessian_geom2 *= scale;
    result.hessian_cross *= scale;
  }
  
  if (enable_profiling) {
    auto alloc_time = std::chrono::duration_cast<std::chrono::microseconds>(
      t_alloc_end - t_alloc_start).count() / 1000.0;
    auto memcpy_time = std::chrono::duration_cast<std::chrono::microseconds>(
      t_memcpy_end - t_memcpy_start).count() / 1000.0;
    auto kernel1_time = std::chrono::duration_cast<std::chrono::microseconds>(
      t_kernel1_end - t_kernel1_start).count() / 1000.0;
    auto kernel2_time = std::chrono::duration_cast<std::chrono::microseconds>(
      t_kernel2_end - t_kernel2_start).count() / 1000.0;
    auto kernel3_time = std::chrono::duration_cast<std::chrono::microseconds>(
      t_kernel3_end - t_kernel3_start).count() / 1000.0;
    auto kernel4_time = std::chrono::duration_cast<std::chrono::microseconds>(
      t_kernel4_end - t_kernel4_start).count() / 1000.0;
    auto result_time = std::chrono::duration_cast<std::chrono::microseconds>(
      t_result_end - t_result_start).count() / 1000.0;
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
      t_result_end - t_start).count() / 1000.0;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== CUDA Intersection Volume Profiler ===" << std::endl;
    std::cout << "Geometry 1: " << get_geometry_type_name(ref_geom1.type) 
              << " (" << (ref_geom1.type == GEOM_TRIANGLE ? ref_geom1.tris.size() : 
                          (ref_geom1.type == GEOM_DISK ? ref_geom1.disks.size() : ref_geom1.gaussians.size()))
              << " " << get_geometry_element_name(ref_geom1.type) << "), DoF: " << dof1_type << std::endl;
    std::cout << "Geometry 2: " << get_geometry_type_name(ref_geom2.type) 
              << " (" << (ref_geom2.type == GEOM_TRIANGLE ? ref_geom2.tris.size() : 
                          (ref_geom2.type == GEOM_DISK ? ref_geom2.disks.size() : ref_geom2.gaussians.size()))
              << " " << get_geometry_element_name(ref_geom2.type) << "), DoF: " << dof2_type << std::endl;
    std::cout << "K-grid nodes: " << Q << std::endl;
    std::cout << "DoFs per geometry: " << num_dofs1 << ", " << num_dofs2 << std::endl;
    std::cout << "Flags: Volume=" << needs_volume(flags) 
              << ", Gradient=" << needs_gradient(flags) 
              << ", Hessian=" << needs_hessian(flags) << std::endl;
    std::cout << "--- Timing (ms) ---" << std::endl;
    std::cout << "  Memory allocation:   " << std::setw(8) << alloc_time << " ms" << std::endl;
    std::cout << "  Memory copy (H->D):  " << std::setw(8) << memcpy_time << " ms" << std::endl;
    std::cout << "  Phase 1 (A(k)):      " << std::setw(8) << kernel1_time << " ms" << std::endl;
    if (needs_gradient(flags) || needs_hessian(flags)) {
      std::cout << "  Phase 2 (∂A/∂θ):     " << std::setw(8) << kernel2_time << " ms" << std::endl;
    }
    std::cout << "  Phase 3 (V, ∇V):     " << std::setw(8) << kernel3_time << " ms" << std::endl;
    if (needs_hessian(flags)) {
      std::cout << "  Phase 4 (H):         " << std::setw(8) << kernel4_time << " ms" << std::endl;
    }
    std::cout << "  Memory copy (D->H):  " << std::setw(8) << result_time << " ms" << std::endl;
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
  if (d_hessian_11) cudaFree(d_hessian_11);
  if (d_hessian_22) cudaFree(d_hessian_22);
  if (d_hessian_12) cudaFree(d_hessian_12);
  
  #undef CUDA_CHECK
  
  return result;
}

// ============================================================================
// Helper Functions for Testing (Single k-vector A(k) and ∂A(k)/∂θ)
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


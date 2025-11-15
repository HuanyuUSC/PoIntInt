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
#include "dof/affine_dof.hpp"
#include "dof/triangle_mesh_dof.hpp"
#include "dof/cuda/affine_dof_cuda.hpp"

using PoIntInt::TriPacked;
using PoIntInt::DiskPacked;
using PoIntInt::GaussianPacked;

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
  TriPacked* d_tris1 = nullptr;
  TriPacked* d_tris2 = nullptr;
  double3* d_kdirs = nullptr;
  double* d_kmags = nullptr;
  double* d_weights = nullptr;
  double* d_dofs1 = nullptr;
  double* d_dofs2 = nullptr;
  double2* d_A1 = nullptr;
  double2* d_A2 = nullptr;
  double2* d_grad_A1 = nullptr;
  double2* d_grad_A2 = nullptr;
  double* d_grad_V1 = nullptr;
  double* d_grad_V2 = nullptr;
  double* d_volume = nullptr;
  
  // Host vectors
  std::vector<double3> h_kdirs;
  std::vector<double> h_dofs1, h_dofs2;
  std::vector<double2> h_A1, h_A2;
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
  dim3 grid_grad, block_grad, grid_vol_grad, block_vol_grad;
  dim3 grid_A, block_A;  // For Phase 1 kernel launches
  
  // Transformed geometries (for volume computation)
  Geometry geom1_transformed, geom2_transformed;
  
  // Other variables
  int NF1 = 0, NF2 = 0;
  int Q = 0;
  int num_dofs1 = 0, num_dofs2 = 0;
  
  #define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
      goto cleanup; \
    } \
  } while(0)
  
  // Check DoF types - for now, only support AffineDoF
  // TODO: Add support for TriangleMeshDoF and other DoF types
  auto affine_dof1 = std::dynamic_pointer_cast<AffineDoF>(dof1);
  auto affine_dof2 = std::dynamic_pointer_cast<AffineDoF>(dof2);
  
  if (!affine_dof1 || !affine_dof2) {
    std::cerr << "Error: Only AffineDoF is currently supported for CUDA gradient computation" << std::endl;
    result.grad_geom1 = Eigen::VectorXd::Zero(dofs1.size());
    result.grad_geom2 = Eigen::VectorXd::Zero(dofs2.size());
    result.volume = 0.0;
    return result;
  }
  
  if (dofs1.size() != 12 || dofs2.size() != 12) {
    std::cerr << "Error: AffineDoF requires 12 DoFs" << std::endl;
    result.grad_geom1 = Eigen::VectorXd::Zero(dofs1.size());
    result.grad_geom2 = Eigen::VectorXd::Zero(dofs2.size());
    result.volume = 0.0;
    return result;
  }
  
  Q = (int)kgrid.kmag.size();
  num_dofs1 = 12;
  num_dofs2 = 12;
  
  // For now, only support triangle meshes
  // TODO: Add support for disks and Gaussian splats
  if (geom1.type != GEOM_TRIANGLE || geom2.type != GEOM_TRIANGLE) {
    std::cerr << "Error: Only triangle meshes are currently supported for CUDA gradient computation" << std::endl;
    result.grad_geom1 = Eigen::VectorXd::Zero(num_dofs1);
    result.grad_geom2 = Eigen::VectorXd::Zero(num_dofs2);
    result.volume = 0.0;
    return result;
  }
  
  t_alloc_start = std::chrono::high_resolution_clock::now();
  
  // Allocate geometry data
  NF1 = (int)geom1.tris.size();
  NF2 = (int)geom2.tris.size();
  
  if (NF1 > 0) {
    CUDA_CHECK(cudaMalloc(&d_tris1, NF1 * sizeof(TriPacked)));
  } else {
    CUDA_CHECK(cudaMalloc(&d_tris1, sizeof(TriPacked)));
  }
  
  if (NF2 > 0) {
    CUDA_CHECK(cudaMalloc(&d_tris2, NF2 * sizeof(TriPacked)));
  } else {
    CUDA_CHECK(cudaMalloc(&d_tris2, sizeof(TriPacked)));
  }
  
  // Allocate k-grid data
  h_kdirs.resize(Q);
  for (int q = 0; q < Q; ++q) {
    h_kdirs[q] = make_double3(kgrid.dirs[q][0], kgrid.dirs[q][1], kgrid.dirs[q][2]);
  }
  CUDA_CHECK(cudaMalloc(&d_kdirs, Q * sizeof(double3)));
  CUDA_CHECK(cudaMalloc(&d_kmags, Q * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_weights, Q * sizeof(double)));
  
  // Allocate DoF parameters
  CUDA_CHECK(cudaMalloc(&d_dofs1, num_dofs1 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_dofs2, num_dofs2 * sizeof(double)));
  
  // Allocate form factors and gradients
  CUDA_CHECK(cudaMalloc(&d_A1, Q * sizeof(double2)));
  CUDA_CHECK(cudaMalloc(&d_A2, Q * sizeof(double2)));
  CUDA_CHECK(cudaMalloc(&d_grad_A1, Q * num_dofs1 * sizeof(double2)));
  CUDA_CHECK(cudaMalloc(&d_grad_A2, Q * num_dofs2 * sizeof(double2)));
  
  // Allocate output gradients and volume
  CUDA_CHECK(cudaMalloc(&d_grad_V1, num_dofs1 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_grad_V2, num_dofs2 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_volume, sizeof(double)));
  
  t_alloc_end = std::chrono::high_resolution_clock::now();
  t_memcpy_start = std::chrono::high_resolution_clock::now();
  
  // Copy data to device
  if (NF1 > 0) {
    CUDA_CHECK(cudaMemcpy(d_tris1, geom1.tris.data(), NF1 * sizeof(TriPacked), cudaMemcpyHostToDevice));
  }
  if (NF2 > 0) {
    CUDA_CHECK(cudaMemcpy(d_tris2, geom2.tris.data(), NF2 * sizeof(TriPacked), cudaMemcpyHostToDevice));
  }
  
  CUDA_CHECK(cudaMemcpy(d_kdirs, h_kdirs.data(), Q * sizeof(double3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kmags, kgrid.kmag.data(), Q * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weights, kgrid.w.data(), Q * sizeof(double), cudaMemcpyHostToDevice));
  
  // Copy DoFs
  h_dofs1.resize(num_dofs1);
  h_dofs2.resize(num_dofs2);
  for (int i = 0; i < num_dofs1; ++i) {
    h_dofs1[i] = dofs1(i);
  }
  for (int i = 0; i < num_dofs2; ++i) {
    h_dofs2[i] = dofs2(i);
  }
  CUDA_CHECK(cudaMemcpy(d_dofs1, h_dofs1.data(), num_dofs1 * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dofs2, h_dofs2.data(), num_dofs2 * sizeof(double), cudaMemcpyHostToDevice));
  
  // Initialize gradients and volume to zero
  CUDA_CHECK(cudaMemset(d_grad_V1, 0, num_dofs1 * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_grad_V2, 0, num_dofs2 * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_volume, 0, sizeof(double)));
  
  t_memcpy_end = std::chrono::high_resolution_clock::now();
  t_kernel1_start = std::chrono::high_resolution_clock::now();
  
  // Phase 1: Compute A(k) for both geometries using CUDA
  grid_A = dim3(Q, 1);
  block_A = dim3(blockSize);
  
  compute_A_affine_triangle_kernel<<<grid_A, block_A>>>(
    d_tris1, NF1, d_kdirs, d_kmags, d_dofs1, Q, d_A1
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
  compute_A_affine_triangle_kernel<<<grid_A, block_A>>>(
    d_tris2, NF2, d_kdirs, d_kmags, d_dofs2, Q, d_A2
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
  t_kernel1_end = std::chrono::high_resolution_clock::now();
  t_kernel2_start = std::chrono::high_resolution_clock::now();
  
  // Phase 2: Compute ∂A(k)/∂θ for both geometries
  grid_grad = dim3((Q + blockSize - 1) / blockSize);
  block_grad = dim3(blockSize);
  
  compute_A_gradient_affine_triangle_kernel<<<grid_grad, block_grad>>>(
    d_tris1, NF1, d_kdirs, d_kmags, d_dofs1, Q, d_grad_A1
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
  compute_A_gradient_affine_triangle_kernel<<<grid_grad, block_grad>>>(
    d_tris2, NF2, d_kdirs, d_kmags, d_dofs2, Q, d_grad_A2
  );
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
    std::cout << "Geometry 1: " << NF1 << " triangles" << std::endl;
    std::cout << "Geometry 2: " << NF2 << " triangles" << std::endl;
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
  if (d_tris1) cudaFree(d_tris1);
  if (d_tris2) cudaFree(d_tris2);
  if (d_kdirs) cudaFree(d_kdirs);
  if (d_kmags) cudaFree(d_kmags);
  if (d_weights) cudaFree(d_weights);
  if (d_dofs1) cudaFree(d_dofs1);
  if (d_dofs2) cudaFree(d_dofs2);
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
  // Only support AffineDoF with triangle meshes for now
  if (geom.type != GEOM_TRIANGLE) {
    std::cerr << "Error: Only triangle meshes are currently supported" << std::endl;
    return std::complex<double>(0.0, 0.0);
  }
  
  auto affine_dof = std::dynamic_pointer_cast<AffineDoF>(dof);
  if (!affine_dof) {
    std::cerr << "Error: Only AffineDoF is currently supported" << std::endl;
    return std::complex<double>(0.0, 0.0);
  }
  
  if (dofs.size() != 12) {
    std::cerr << "Error: AffineDoF requires 12 DoFs" << std::endl;
    return std::complex<double>(0.0, 0.0);
  }
  
  int NF = (int)geom.tris.size();
  if (NF == 0) return std::complex<double>(0.0, 0.0);
  
  // Prepare k-vector
  double kmag = k.norm();
  if (kmag < 1e-10) return std::complex<double>(0.0, 0.0);
  
  Eigen::Vector3d khat = k / kmag;
  double3 kdir = make_double3(khat.x(), khat.y(), khat.z());
  double kmag_d = kmag;
  
  // Declare all variables at the top (before any goto cleanup)
  TriPacked* d_tris = nullptr;
  double* d_dofs = nullptr;
  double2* d_A = nullptr;
  double3* d_kdirs = nullptr;
  double* d_kmags = nullptr;
  std::vector<double> h_dofs(12);
  dim3 grid(1);
  dim3 block(blockSize);
  double2 h_A = make_double2(0.0, 0.0);
  
  #define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
      goto cleanup; \
    } \
  } while(0)
  
  CUDA_CHECK(cudaMalloc(&d_tris, NF * sizeof(TriPacked)));
  CUDA_CHECK(cudaMalloc(&d_dofs, 12 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_A, sizeof(double2)));
  CUDA_CHECK(cudaMalloc(&d_kdirs, sizeof(double3)));
  CUDA_CHECK(cudaMalloc(&d_kmags, sizeof(double)));
  
  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_tris, geom.tris.data(), NF * sizeof(TriPacked), cudaMemcpyHostToDevice));
  
  for (int i = 0; i < 12; ++i) {
    h_dofs[i] = dofs(i);
  }
  CUDA_CHECK(cudaMemcpy(d_dofs, h_dofs.data(), 12 * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kdirs, &kdir, sizeof(double3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kmags, &kmag_d, sizeof(double), cudaMemcpyHostToDevice));
  
  // Launch kernel (single k-point, so Q=1)
  compute_A_affine_triangle_kernel<<<grid, block>>>(
    d_tris, NF, d_kdirs, d_kmags, d_dofs, 1, d_A
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Copy result back
  CUDA_CHECK(cudaMemcpy(&h_A, d_A, sizeof(double2), cudaMemcpyDeviceToHost));
  
  cleanup:
  if (d_tris) cudaFree(d_tris);
  if (d_dofs) cudaFree(d_dofs);
  if (d_A) cudaFree(d_A);
  if (d_kdirs) cudaFree(d_kdirs);
  if (d_kmags) cudaFree(d_kmags);
  
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
  // Only support AffineDoF with triangle meshes for now
  if (geom.type != GEOM_TRIANGLE) {
    std::cerr << "Error: Only triangle meshes are currently supported" << std::endl;
    return Eigen::VectorXcd::Zero(dof->num_dofs());
  }
  
  auto affine_dof = std::dynamic_pointer_cast<AffineDoF>(dof);
  if (!affine_dof) {
    std::cerr << "Error: Only AffineDoF is currently supported" << std::endl;
    return Eigen::VectorXcd::Zero(dof->num_dofs());
  }
  
  if (dofs.size() != 12) {
    std::cerr << "Error: AffineDoF requires 12 DoFs" << std::endl;
    return Eigen::VectorXcd::Zero(12);
  }
  
  int NF = (int)geom.tris.size();
  if (NF == 0) return Eigen::VectorXcd::Zero(12);
  
  // Prepare k-vector
  double kmag = k.norm();
  if (kmag < 1e-10) return Eigen::VectorXcd::Zero(12);
  
  Eigen::Vector3d khat = k / kmag;
  double3 kdir = make_double3(khat.x(), khat.y(), khat.z());
  double kmag_d = kmag;
  
  // Declare all variables at the top (before any goto cleanup)
  TriPacked* d_tris = nullptr;
  double* d_dofs = nullptr;
  double2* d_grad_A = nullptr;
  double3* d_kdirs = nullptr;
  double* d_kmags = nullptr;
  std::vector<double> h_dofs(12);
  dim3 grid(1);
  dim3 block(blockSize);
  std::vector<double2> h_grad_A(12);
  Eigen::VectorXcd grad(12);
  
  #define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
      goto cleanup; \
    } \
  } while(0)
  
  CUDA_CHECK(cudaMalloc(&d_tris, NF * sizeof(TriPacked)));
  CUDA_CHECK(cudaMalloc(&d_dofs, 12 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_grad_A, 12 * sizeof(double2)));
  CUDA_CHECK(cudaMalloc(&d_kdirs, sizeof(double3)));
  CUDA_CHECK(cudaMalloc(&d_kmags, sizeof(double)));
  
  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_tris, geom.tris.data(), NF * sizeof(TriPacked), cudaMemcpyHostToDevice));
  
  for (int i = 0; i < 12; ++i) {
    h_dofs[i] = dofs(i);
  }
  CUDA_CHECK(cudaMemcpy(d_dofs, h_dofs.data(), 12 * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kdirs, &kdir, sizeof(double3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kmags, &kmag_d, sizeof(double), cudaMemcpyHostToDevice));
  
  // Launch kernel (single k-point, so Q=1)
  compute_A_gradient_affine_triangle_kernel<<<grid, block>>>(
    d_tris, NF, d_kdirs, d_kmags, d_dofs, 1, d_grad_A
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Copy result back
  CUDA_CHECK(cudaMemcpy(h_grad_A.data(), d_grad_A, 12 * sizeof(double2), cudaMemcpyDeviceToHost));
  
  cleanup:
  if (d_tris) cudaFree(d_tris);
  if (d_dofs) cudaFree(d_dofs);
  if (d_grad_A) cudaFree(d_grad_A);
  if (d_kdirs) cudaFree(d_kdirs);
  if (d_kmags) cudaFree(d_kmags);
  
  #undef CUDA_CHECK
  
  // Convert result to Eigen::VectorXcd
  for (int i = 0; i < 12; ++i) {
    grad(i) = std::complex<double>(h_grad_A[i].x, h_grad_A[i].y);
  }
  
  return grad;
}

} // namespace PoIntInt


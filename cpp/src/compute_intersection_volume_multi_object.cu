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

// ===================== Helper Kernel: Copy Grad A(k) to Grad J Matrix =====================
// Kernel to copy per-object gradient buffer to packed grad_J matrix
extern "C" __global__
void copy_grad_A_to_grad_J_kernel(
  const double2* __restrict__ grad_A_obj,  // Input: Q × num_dofs (gradient for one object)
  int obj,                                  // Object index
  int num_dofs,                             // Number of DoFs for this object
  int dof_offset,                           // Offset in grad_J for this object
  int total_grad_size,                      // Total gradient size per k-node
  int Q,
  double2* grad_J                           // Output: Q × total_grad_size (row-major)
) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= Q) return;
  
  // Copy grad_A_obj[q * num_dofs + dof] to grad_J[q * total_grad_size + dof_offset + dof]
  for (int dof = 0; dof < num_dofs; ++dof) {
    int src_idx = q * num_dofs + dof;
    int dst_idx = q * total_grad_size + dof_offset + dof;
    grad_J[dst_idx] = grad_A_obj[src_idx];
  }
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

// ===================== Phase 3: Compute Gradients =====================
// Kernel to compute gradients of volume matrix entries
// For each pair (i,j), compute:
//   ∂V[i,j]/∂θ_i = (1/(8π³)) · Σ_q w_q · Re((∂A_i(k_q)/∂θ_i) · conj(A_j(k_q)))
//   ∂V[i,j]/∂θ_j = (1/(8π³)) · Σ_q w_q · Re(A_i(k_q) · conj(∂A_j(k_q)/∂θ_j))
// Parallelizes over k-nodes (Q threads)
extern "C" __global__
void compute_volume_matrix_gradient_kernel(
  const double2* __restrict__ grad_J,
  const double2* __restrict__ J,
  const double* __restrict__ weights,
  int Q,
  int num_objects,
  const int* __restrict__ num_dofs,
  const int* __restrict__ dof_offsets,
  int i,
  int j,
  double* grad_V_i,
  double* grad_V_j
) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= Q) return;
  
  double w = weights[q];
  
  // Get A_i(k_q) and A_j(k_q)
  int idx_i = q * num_objects + i;
  int idx_j = q * num_objects + j;
  double2 Ai = J[idx_i];
  double2 Aj = J[idx_j];
  
  // Compute gradient w.r.t. DoFs of object i: Re((∂A_i/∂θ_i) · conj(A_j))
  int offset_i = dof_offsets[i];
  int num_dofs_i = num_dofs[i];
  for (int dof = 0; dof < num_dofs_i; ++dof) {
    // grad_J is stored as: [q * total_grad_size + offset_i + dof]
    // We need to compute the total size per k-node
    int total_grad_size = dof_offsets[num_objects - 1] + num_dofs[num_objects - 1];
    int grad_idx = q * total_grad_size + offset_i + dof;
    double2 dAi = grad_J[grad_idx];
    
    // Re(dAi * conj(Aj)) = Re(dAi) * Re(Aj) + Im(dAi) * Im(Aj)
    double grad_contrib = dAi.x * Aj.x + dAi.y * Aj.y;
    atomicAdd(&grad_V_i[dof], w * grad_contrib);
  }
  
  // Compute gradient w.r.t. DoFs of object j: Re(A_i · conj(∂A_j/∂θ_j))
  int offset_j = dof_offsets[j];
  int num_dofs_j = num_dofs[j];
  for (int dof = 0; dof < num_dofs_j; ++dof) {
    int total_grad_size = dof_offsets[num_objects - 1] + num_dofs[num_objects - 1];
    int grad_idx = q * total_grad_size + offset_j + dof;
    double2 dAj = grad_J[grad_idx];
    
    // Re(Ai * conj(dAj)) = Re(Ai) * Re(dAj) + Im(Ai) * Im(dAj)
    double grad_contrib = Ai.x * dAj.x + Ai.y * dAj.y;
    atomicAdd(&grad_V_j[dof], w * grad_contrib);
  }
}

// ===================== Phase 4: Compute Hessians (Gauss-Newton) =====================
// Kernel to compute Hessians of volume matrix entries (Gauss-Newton approximation)
// For Gauss-Newton, we only compute the cross-term H_ij, since:
//   V[i,j] = (1/(8π³)) · Σ_q w_q · Re(A_i(k_q) · conj(A_j(k_q)))
//   = (1/(8π³)) · Σ_q w_q · g_q(A_i) · h_q(A_j)
// where g_q and h_q are functions of the form factors.
// Gauss-Newton ignores second derivatives, so:
//   H_ii[i,j] = 0 (ignoring ∂²A_i/∂θ_i² terms)
//   H_jj[i,j] = 0 (ignoring ∂²A_j/∂θ_j² terms)
//   H_ij[i,j] = (1/(8π³)) · Σ_q w_q · (∂A_i/∂θ_i) · (∂A_j/∂θ_j)^T
// Parallelizes over k-nodes (Q threads)
extern "C" __global__
void compute_volume_matrix_hessian_kernel(
  const double2* __restrict__ grad_J,
  const double* __restrict__ weights,
  int Q,
  int num_objects,
  const int* __restrict__ num_dofs,
  const int* __restrict__ dof_offsets,
  int i,
  int j,
  double* hessian_ii,  // Output: set to zero (not computed for Gauss-Newton)
  double* hessian_jj,  // Output: set to zero (not computed for Gauss-Newton)
  double* hessian_ij   // Output: cross-term Hessian
) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= Q) return;
  
  double w = weights[q];
  
  int offset_i = dof_offsets[i];
  int offset_j = dof_offsets[j];
  int num_dofs_i = num_dofs[i];
  int num_dofs_j = num_dofs[j];
  int total_grad_size = dof_offsets[num_objects - 1] + num_dofs[num_objects - 1];
  
  // For Gauss-Newton, H_ii and H_jj are zero (we ignore second derivatives)
  // Only compute H_ij: outer product of grad_A_i with grad_A_j
  for (int di = 0; di < num_dofs_i; ++di) {
    int grad_idx_i = q * total_grad_size + offset_i + di;
    double2 dAi_di = grad_J[grad_idx_i];
    
    for (int dj = 0; dj < num_dofs_j; ++dj) {
      int grad_idx_j = q * total_grad_size + offset_j + dj;
      double2 dAj_dj = grad_J[grad_idx_j];
      
      // Real part of (dAi_di * conj(dAj_dj)) = Re(dAi_di) * Re(dAj_dj) + Im(dAi_di) * Im(dAj_dj)
      double hess_contrib = dAi_di.x * dAj_dj.x + dAi_di.y * dAj_dj.y;
      atomicAdd(&hessian_ij[di * num_dofs_j + dj], w * hess_contrib);
    }
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
  
  int Q = (int)kgrid.kmag.size();
  
  // Get DoF type names, number of DoFs, and check for CUDA kernels
  std::vector<std::string> dof_types(num_objects);
  std::vector<int> num_dofs_per_obj(num_objects);
  std::vector<int> dof_offsets(num_objects);
  std::vector<std::pair<CudaComputeAkFunc, CudaComputeAkGradientFunc>> kernels(num_objects);
  bool all_kernels_available = true;
  bool need_gradients = needs_gradient(flags) || needs_hessian(flags);
  
  // Compute number of DoFs and offsets for each object
  int total_grad_size = 0;
  for (int obj = 0; obj < num_objects; ++obj) {
    num_dofs_per_obj[obj] = (int)dof_vectors[obj].size();
    dof_offsets[obj] = total_grad_size;
    total_grad_size += num_dofs_per_obj[obj];
  }
  
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
    
    // Check if gradient kernel is available if gradients are needed
    if (need_gradients && !kernels[obj].second) {
      std::cerr << "Error: Gradient kernel not available for DoF type '" << dof_types[obj] 
                << "' with geometry type " << (int)ref_geometries[obj].type 
                << " for object " << obj << std::endl;
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
  
  // Gradient buffers (if needed)
  std::vector<double2*> d_grad_A_objects(num_objects, nullptr);
  double2* d_grad_J = nullptr;  // Packed gradient buffer: Q × total_grad_size
  int* d_num_dofs = nullptr;
  int* d_dof_offsets = nullptr;
  
  // Host result buffer (declare early to avoid goto issues)
  std::vector<double> h_V(num_objects * num_objects);

  // Timing variables (declared early to avoid goto issues)
  auto t_malloc_start = t_start_total;
  auto t_malloc_end = t_start_total;
  auto t_memcpy_start = t_start_total;
  auto t_memcpy_end = t_start_total;
  auto t_kernel1_start = t_start_total;
  auto t_kernel1_end = t_start_total;
  auto t_kernel2_start = t_start_total;
  auto t_kernel2_end = t_start_total;
  auto t_kernel3_start = t_start_total;
  auto t_kernel3_end = t_start_total;
  auto t_kernel4_start = t_start_total;
  auto t_kernel4_end = t_start_total;
  auto t_result_start = t_start_total;
  auto t_result_end = t_start_total;
  auto t_end_total = t_start_total;
  double malloc_time = 0.0, memcpy_time = 0.0, kernel1_time = 0.0, kernel2_time = 0.0;
  double kernel3_time = 0.0, kernel4_time = 0.0, result_time = 0.0, total_time = 0.0;
  dim3 block_V, grid_V;

  #define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
      goto cleanup; \
    } \
  } while(0)

  t_malloc_start = std::chrono::high_resolution_clock::now();
  
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
  
  // Allocate gradient buffers if needed
  if (need_gradients) {
    // Allocate per-object gradient buffers
    for (int obj = 0; obj < num_objects; ++obj) {
      CUDA_CHECK(cudaMalloc(&d_grad_A_objects[obj], Q * num_dofs_per_obj[obj] * sizeof(double2)));
    }
    
    // Allocate packed gradient buffer
    CUDA_CHECK(cudaMalloc(&d_grad_J, Q * total_grad_size * sizeof(double2)));
    
    // Allocate and copy num_dofs and dof_offsets to device
    CUDA_CHECK(cudaMalloc(&d_num_dofs, num_objects * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dof_offsets, num_objects * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_num_dofs, num_dofs_per_obj.data(), num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dof_offsets, dof_offsets.data(), num_objects * sizeof(int), cudaMemcpyHostToDevice));
  }
  
  t_malloc_end = std::chrono::high_resolution_clock::now();
  t_memcpy_start = std::chrono::high_resolution_clock::now();
  
  // Copy k-grid weights
  CUDA_CHECK(cudaMemcpy(d_weights, kgrid.w.data(), Q * sizeof(double), cudaMemcpyHostToDevice));
  
  t_memcpy_end = std::chrono::high_resolution_clock::now();
  t_kernel1_start = std::chrono::high_resolution_clock::now();
  
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
  
  t_kernel1_end = std::chrono::high_resolution_clock::now();
  t_kernel2_start = std::chrono::high_resolution_clock::now();
  
  // Phase 2: Compute volume matrix V = J^T D J
  // Use 16x16 thread blocks for matrix computation
  block_V = dim3(16, 16);
  grid_V = dim3((num_objects + block_V.x - 1) / block_V.x,
                (num_objects + block_V.y - 1) / block_V.y);
  compute_volume_matrix_kernel<<<grid_V, block_V>>>(
    d_J, d_weights, Q, num_objects, d_V
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
  t_kernel2_end = std::chrono::high_resolution_clock::now();
  t_kernel3_start = std::chrono::high_resolution_clock::now();
  
  // Phase 3: Compute gradients (if needed)
  if (need_gradients) {
    // Compute ∂A(k)/∂θ for each object
    for (int obj = 0; obj < num_objects; ++obj) {
      kernels[obj].second(ref_geometries[obj], kgrid, dof_vectors[obj], d_grad_A_objects[obj], blockSize);
      CUDA_CHECK(cudaDeviceSynchronize());
      
      // Copy from per-object gradient buffer to packed grad_J buffer
      // grad_J[q * total_grad_size + dof_offsets[obj] + dof] = grad_A_objects[obj][q * num_dofs_per_obj[obj] + dof]
      dim3 grid_grad((Q + blockSize - 1) / blockSize);
      dim3 block_grad(blockSize);
      copy_grad_A_to_grad_J_kernel<<<grid_grad, block_grad>>>(
        d_grad_A_objects[obj], obj, num_dofs_per_obj[obj], dof_offsets[obj], 
        total_grad_size, Q, d_grad_J
      );
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
  
  t_kernel3_end = std::chrono::high_resolution_clock::now();
  t_kernel4_start = std::chrono::high_resolution_clock::now();
  
  // Phase 4: Compute gradients and Hessians for each pair (i,j)
  if (needs_gradient(flags)) {
    // Initialize gradient storage
    result.grad_matrix.resize(num_objects);
    for (int i = 0; i < num_objects; ++i) {
      result.grad_matrix[i].resize(num_objects);
      for (int j = 0; j < num_objects; ++j) {
        result.grad_matrix[i][j] = Eigen::VectorXd::Zero(num_dofs_per_obj[i]);
      }
    }
    
    // Allocate device buffers for gradients
    std::vector<double*> d_grad_V_i_buffers(num_objects * num_objects, nullptr);
    std::vector<double*> d_grad_V_j_buffers(num_objects * num_objects, nullptr);
    
    for (int i = 0; i < num_objects; ++i) {
      for (int j = 0; j < num_objects; ++j) {
        CUDA_CHECK(cudaMalloc(&d_grad_V_i_buffers[i * num_objects + j], num_dofs_per_obj[i] * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_grad_V_j_buffers[i * num_objects + j], num_dofs_per_obj[j] * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_grad_V_i_buffers[i * num_objects + j], 0, num_dofs_per_obj[i] * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_grad_V_j_buffers[i * num_objects + j], 0, num_dofs_per_obj[j] * sizeof(double)));
      }
    }
    
    // Compute gradients for each pair
    for (int i = 0; i < num_objects; ++i) {
      for (int j = 0; j < num_objects; ++j) {
        dim3 grid_grad((Q + blockSize - 1) / blockSize);
        dim3 block_grad(blockSize);
        compute_volume_matrix_gradient_kernel<<<grid_grad, block_grad>>>(
          d_grad_J, d_J, d_weights, Q, num_objects,
          d_num_dofs, d_dof_offsets, i, j,
          d_grad_V_i_buffers[i * num_objects + j],
          d_grad_V_j_buffers[i * num_objects + j]
        );
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }
    
    // Copy gradients back
    for (int i = 0; i < num_objects; ++i) {
      for (int j = 0; j < num_objects; ++j) {
        std::vector<double> h_grad_i(num_dofs_per_obj[i]);
        std::vector<double> h_grad_j(num_dofs_per_obj[j]);
        CUDA_CHECK(cudaMemcpy(h_grad_i.data(), d_grad_V_i_buffers[i * num_objects + j], 
                             num_dofs_per_obj[i] * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_grad_j.data(), d_grad_V_j_buffers[i * num_objects + j], 
                             num_dofs_per_obj[j] * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Apply scaling factor: 1/(8π³)
        const double scale = 1.0 / (8.0 * CUDART_PI * CUDART_PI * CUDART_PI);
        result.grad_matrix[i][j] = Eigen::Map<Eigen::VectorXd>(h_grad_i.data(), num_dofs_per_obj[i]) * scale;
        result.grad_matrix[j][i] = Eigen::Map<Eigen::VectorXd>(h_grad_j.data(), num_dofs_per_obj[j]) * scale;
      }
    }
    
    // Free gradient buffers
    for (int i = 0; i < num_objects; ++i) {
      for (int j = 0; j < num_objects; ++j) {
        if (d_grad_V_i_buffers[i * num_objects + j]) cudaFree(d_grad_V_i_buffers[i * num_objects + j]);
        if (d_grad_V_j_buffers[i * num_objects + j]) cudaFree(d_grad_V_j_buffers[i * num_objects + j]);
      }
    }
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
    
    // Allocate device buffers for Hessians
    std::vector<double*> d_hessian_ii_buffers(num_objects * num_objects, nullptr);
    std::vector<double*> d_hessian_jj_buffers(num_objects * num_objects, nullptr);
    std::vector<double*> d_hessian_ij_buffers(num_objects * num_objects, nullptr);
    
    for (int i = 0; i < num_objects; ++i) {
      for (int j = 0; j < num_objects; ++j) {
        CUDA_CHECK(cudaMalloc(&d_hessian_ii_buffers[i * num_objects + j], 
                             num_dofs_per_obj[i] * num_dofs_per_obj[i] * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_hessian_jj_buffers[i * num_objects + j], 
                             num_dofs_per_obj[j] * num_dofs_per_obj[j] * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_hessian_ij_buffers[i * num_objects + j], 
                             num_dofs_per_obj[i] * num_dofs_per_obj[j] * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_hessian_ii_buffers[i * num_objects + j], 0, 
                             num_dofs_per_obj[i] * num_dofs_per_obj[i] * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_hessian_jj_buffers[i * num_objects + j], 0, 
                             num_dofs_per_obj[j] * num_dofs_per_obj[j] * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_hessian_ij_buffers[i * num_objects + j], 0, 
                             num_dofs_per_obj[i] * num_dofs_per_obj[j] * sizeof(double)));
      }
    }
    
    // Compute Hessians for each pair
    for (int i = 0; i < num_objects; ++i) {
      for (int j = 0; j < num_objects; ++j) {
        dim3 grid_hessian((Q + blockSize - 1) / blockSize);
        dim3 block_hessian(blockSize);
        compute_volume_matrix_hessian_kernel<<<grid_hessian, block_hessian>>>(
          d_grad_J, d_weights, Q, num_objects,
          d_num_dofs, d_dof_offsets, i, j,
          d_hessian_ii_buffers[i * num_objects + j],
          d_hessian_jj_buffers[i * num_objects + j],
          d_hessian_ij_buffers[i * num_objects + j]
        );
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }
    
    // Copy Hessians back
    for (int i = 0; i < num_objects; ++i) {
      for (int j = 0; j < num_objects; ++j) {
        std::vector<double> h_hessian_ii(num_dofs_per_obj[i] * num_dofs_per_obj[i]);
        std::vector<double> h_hessian_jj(num_dofs_per_obj[j] * num_dofs_per_obj[j]);
        std::vector<double> h_hessian_ij(num_dofs_per_obj[i] * num_dofs_per_obj[j]);
        
        CUDA_CHECK(cudaMemcpy(h_hessian_ii.data(), d_hessian_ii_buffers[i * num_objects + j], 
                             num_dofs_per_obj[i] * num_dofs_per_obj[i] * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_hessian_jj.data(), d_hessian_jj_buffers[i * num_objects + j], 
                             num_dofs_per_obj[j] * num_dofs_per_obj[j] * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_hessian_ij.data(), d_hessian_ij_buffers[i * num_objects + j], 
                             num_dofs_per_obj[i] * num_dofs_per_obj[j] * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Apply scaling factor: 1/(8π³)
        const double scale = 1.0 / (8.0 * CUDART_PI * CUDART_PI * CUDART_PI);
        // For Gauss-Newton, H_ii and H_jj are zero (we ignore second derivatives)
        result.hessian_ii[i][j] = Eigen::MatrixXd::Zero(num_dofs_per_obj[i], num_dofs_per_obj[i]);
        result.hessian_jj[i][j] = Eigen::MatrixXd::Zero(num_dofs_per_obj[j], num_dofs_per_obj[j]);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> H_ij_row(
          h_hessian_ij.data(), num_dofs_per_obj[i], num_dofs_per_obj[j]);
        result.hessian_ij[i][j] = Eigen::MatrixXd(H_ij_row) * scale;
      }
    }
    
    // Free Hessian buffers
    for (int i = 0; i < num_objects; ++i) {
      for (int j = 0; j < num_objects; ++j) {
        if (d_hessian_ii_buffers[i * num_objects + j]) cudaFree(d_hessian_ii_buffers[i * num_objects + j]);
        if (d_hessian_jj_buffers[i * num_objects + j]) cudaFree(d_hessian_jj_buffers[i * num_objects + j]);
        if (d_hessian_ij_buffers[i * num_objects + j]) cudaFree(d_hessian_ij_buffers[i * num_objects + j]);
      }
    }
  }
  
  t_kernel4_end = std::chrono::high_resolution_clock::now();
  t_result_start = std::chrono::high_resolution_clock::now();
  
  // Copy volume matrix result back
  CUDA_CHECK(cudaMemcpy(h_V.data(), d_V, num_objects * num_objects * sizeof(double), cudaMemcpyDeviceToHost));
  
  t_result_end = std::chrono::high_resolution_clock::now();
  
  // Convert to Eigen matrix
  result.volume_matrix = Eigen::Map<Eigen::MatrixXd>(h_V.data(), num_objects, num_objects);
  
  // End total timing
  t_end_total = std::chrono::high_resolution_clock::now();
  
  // Calculate timing statistics
  malloc_time = std::chrono::duration_cast<std::chrono::microseconds>(t_malloc_end - t_malloc_start).count() / 1000.0;
  memcpy_time = std::chrono::duration_cast<std::chrono::microseconds>(t_memcpy_end - t_memcpy_start).count() / 1000.0;
  kernel1_time = std::chrono::duration_cast<std::chrono::microseconds>(t_kernel1_end - t_kernel1_start).count() / 1000.0;
  kernel2_time = std::chrono::duration_cast<std::chrono::microseconds>(t_kernel2_end - t_kernel2_start).count() / 1000.0;
  kernel3_time = need_gradients ? std::chrono::duration_cast<std::chrono::microseconds>(t_kernel3_end - t_kernel3_start).count() / 1000.0 : 0.0;
  kernel4_time = (needs_gradient(flags) || needs_hessian(flags)) ?
                      std::chrono::duration_cast<std::chrono::microseconds>(t_kernel4_end - t_kernel4_start).count() / 1000.0 : 0.0;
  result_time = std::chrono::duration_cast<std::chrono::microseconds>(t_result_end - t_result_start).count() / 1000.0;
  total_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end_total - t_start_total).count() / 1000.0;
  
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
    if (need_gradients) {
      std::cout << "  Kernel 3 (Gradients ∂A/∂θ): " << std::setw(8) << kernel3_time << " ms" << std::endl;
    }
    if (needs_gradient(flags) || needs_hessian(flags)) {
      std::cout << "  Kernel 4 (Gradients & Hessians): " << std::setw(8) << kernel4_time << " ms" << std::endl;
    }
    std::cout << "  Memory copy (D->H): " << std::setw(8) << result_time << " ms" << std::endl;
    std::cout << "  Total time:         " << std::setw(8) << total_time << " ms" << std::endl;
    std::cout << "==========================================\n" << std::endl;
  }
  
  cleanup:
  // Cleanup
  if (d_J) cudaFree(d_J);
  if (d_weights) cudaFree(d_weights);
  if (d_V) cudaFree(d_V);
  if (d_grad_J) cudaFree(d_grad_J);
  if (d_num_dofs) cudaFree(d_num_dofs);
  if (d_dof_offsets) cudaFree(d_dof_offsets);
  for (int obj = 0; obj < num_objects; ++obj) {
    if (d_A_objects[obj]) cudaFree(d_A_objects[obj]);
    if (d_grad_A_objects[obj]) cudaFree(d_grad_A_objects[obj]);
  }
  
  #undef CUDA_CHECK
  
  return result;
}

} // namespace PoIntInt

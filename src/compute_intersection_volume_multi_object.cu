#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <array>
#include <cmath>
#include <cassert>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <math_constants.h>
#include "compute_intersection_volume_multi_object.hpp"
#include "geometry/geometry.hpp"
#include "geometry/types.hpp"

// Reuse utility functions from compute_volume.cu
// (In a real implementation, these would be in a shared header)
__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
  return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

__device__ __forceinline__ float2 cexp_i(float phase) {
  float s, c;
  __sincosf(phase, &s, &c);
  return make_float2(c, s);
}

__device__ __forceinline__ float J1_over_x(float x) {
  float ax = fabsf(x);
  if (ax < 1e-3f) {
    float x2 = x * x;
    float t = 0.5f;
    t += (-1.0f / 16.0f) * x2;
    float x4 = x2 * x2;
    t += (1.0f / 384.0f) * x4;
    float x6 = x4 * x2;
    t += (-1.0f / 18432.0f) * x6;
    return t;
  }
  if (ax <= 12.0f) {
    float q = 0.25f * x * x;
    float term = 0.5f;
    float sum = term;
#pragma unroll
    for (int m = 0; m < 20; ++m) {
      float denom = (float)(m + 1) * (float)(m + 2);
      term *= -q / denom;
      sum += term;
      if (fabsf(term) < 1e-7f * fabsf(sum)) break;
    }
    return sum;
  }
  float invx = 1.0f / ax;
  float invx2 = invx * invx;
  float invx3 = invx2 * invx;
  float chi = ax - 0.75f * CUDART_PI;
  float s, c;
  __sincosf(chi, &s, &c);
  float amp = sqrtf(2.0f / (CUDART_PI * ax));
  float cosp = (1.0f - 15.0f / 128.0f * invx2) * c;
  float sinp = (3.0f / 8.0f * invx - 315.0f / 3072.0f * invx3) * s;
  float J1 = amp * (cosp - sinp);
  return J1 * invx;
}

__device__ __forceinline__ float2 E_func(float z) {
  float az = fabsf(z);
  float threshold = 1e-4f;
  if (az < threshold) {
    float z2 = z*z, z4 = z2*z2;
    float real = 1.0f - z2*(1.0f/6.0f) + z4*(1.0f/120.0f);
    float imag = z*(0.5f) - z*z2*(1.0f/24.0f) + z4*z*(1.0f/720.0f);
    return make_float2(real, imag);
  } else {
    float s, c;
    __sincosf(z, &s, &c);
    return make_float2(s/z, (1.0f - c)/z);
  }
}

__device__ __forceinline__ float2 E_prime(float z) {
  float az = fabsf(z);
  float threshold = 1e-4f;
  if (az < threshold) {
    float z2 = z*z, z3 = z2*z, z4 = z2*z2;
    float real = -(1.0f/3.0f)*z + (1.0f/30.0f)*z3;
    float imag = 0.5f - (1.0f/8.0f)*z2 + (1.0f/144.0f)*z4;
    return make_float2(real, imag);
  } else {
    float s, c;
    __sincosf(z, &s, &c);
    float z2 = z*z;
    float re = (z*c - s)/z2;
    float im = (z*s - (1.0f - c))/z2;
    return make_float2(re, im);
  }
}

__device__ __forceinline__ float2 Phi_ab(float alpha, float beta) {
  float d = beta - alpha;
  float threshold = 1e-5f;
  if (fabsf(d) < threshold) {
    float2 Ep = E_prime(0.5f*(alpha+beta));
    return make_float2(2.0f*Ep.y, -2.0f*Ep.x);
  } else {
    float2 Ea = E_func(alpha);
    float2 Eb = E_func(beta);
    float2 num = make_float2(Eb.x - Ea.x, Eb.y - Ea.y);
    float invd = 1.0f/d;
    float2 q = make_float2(num.x*invd, num.y*invd);
    return make_float2(2.0f*q.y, -2.0f*q.x);
  }
}

using PoIntInt::TriPacked;
using PoIntInt::DiskPacked;
using PoIntInt::GaussianPacked;

// ===================== Phase 1: Compute Form Factor Matrix J =====================
// Kernel to compute J[q, obj] = A_obj(k_q) for all k-nodes and objects
// Each thread handles one (k-node, object) pair
extern "C" __global__
void compute_form_factor_matrix_kernel(
  const TriPacked* __restrict__ tris_array,      // Flattened: all triangles from all objects
  const int* __restrict__ tri_counts,             // Number of triangles per object
  const DiskPacked* __restrict__ disks_array,     // Flattened: all disks from all objects
  const int* __restrict__ disk_counts,            // Number of disks per object
  const GaussianPacked* __restrict__ gaussians_array, // Flattened: all Gaussians from all objects
  const int* __restrict__ gaussian_counts,        // Number of Gaussians per object
  const int* __restrict__ geom_types,             // Geometry type per object (0=triangle, 1=disk, 2=Gaussian)
  const int* __restrict__ tri_offsets,            // Offset into tris_array for each object
  const int* __restrict__ disk_offsets,           // Offset into disks_array for each object
  const int* __restrict__ gaussian_offsets,       // Offset into gaussians_array for each object
  int num_objects,
  const float3* __restrict__ kdirs,               // Q k-directions
  const float* __restrict__ kmags,                // Q k-magnitudes
  int Q,
  float2* J                                        // Output: Q × num_objects (row-major)
) {
  int q = blockIdx.x;  // k-node index
  int obj = blockIdx.y; // object index
  
  if (q >= Q || obj >= num_objects) return;
  
  float3 u = kdirs[q];
  float k = kmags[q];
  float kx = k * u.x, ky = k * u.y, kz = k * u.z;
  
  float2 A = make_float2(0.0f, 0.0f);
  int geom_type = geom_types[obj];
  
  if (geom_type == 0) {  // GEOM_TRIANGLE
    int tri_offset = tri_offsets[obj];
    int NF = tri_counts[obj];
    
    // Each thread processes multiple triangles
    for (int i = threadIdx.x; i < NF; i += blockDim.x) {
      TriPacked t = tris_array[tri_offset + i];
      float alpha = kx * t.e1.x + ky * t.e1.y + kz * t.e1.z;
      float beta = kx * t.e2.x + ky * t.e2.y + kz * t.e2.z;
      float gamma = u.x * t.S.x + u.y * t.S.y + u.z * t.S.z;
      float2 phi = Phi_ab(alpha, beta);
      float phase = kx * t.a.x + ky * t.a.y + kz * t.a.z;
      float2 expp = cexp_i(phase);
      float2 scal = cmul(expp, phi);
      A.x += scal.x * gamma;
      A.y += scal.y * gamma;
    }
  } else if (geom_type == 1) {  // GEOM_DISK
    int disk_offset = disk_offsets[obj];
    int ND = disk_counts[obj];
    
    // Each thread processes multiple disks
    for (int i = threadIdx.x; i < ND; i += blockDim.x) {
      DiskPacked disk = disks_array[disk_offset + i];
      if (disk.rho > 0.0f && disk.area >= 0.0f) {
        float kpar_coeff = fmaf(u.x, disk.n.x, fmaf(u.y, disk.n.y, u.z * disk.n.z));
        float kdotn = k * kpar_coeff;
        float r2 = fmaxf(k * k - kdotn * kdotn, 0.0f);
        float r = sqrtf(r2);
        
        float phase = fmaf(kx, disk.c.x, fmaf(ky, disk.c.y, kz * disk.c.z));
        float2 eikc = cexp_i(phase);
        
        float Smag;
        if (r < 1e-6f) {
          Smag = disk.area;
        } else {
          float x = fminf(disk.rho * r, 100.0f);
          Smag = disk.area * 2.0f * J1_over_x(x);
        }
        
        float Apar = kpar_coeff * Smag;
        if (!isfinite(Apar)) Apar = 0.0f;
        
        A.x += eikc.x * Apar;
        A.y += eikc.y * Apar;
      }
    }
  } else if (geom_type == 2) {  // GEOM_GAUSSIAN
    int gaussian_offset = gaussian_offsets[obj];
    int NG = gaussian_counts[obj];
    
    // Each thread processes multiple Gaussians
    for (int i = threadIdx.x; i < NG; i += blockDim.x) {
      GaussianPacked g = gaussians_array[gaussian_offset + i];
      if (g.sigma > 0.0f && g.w >= 0.0f) {
        float kpar_coeff = fmaf(u.x, g.n.x, fmaf(u.y, g.n.y, u.z * g.n.z));
        float kdotn = k * kpar_coeff;
        float r2 = fmaxf(k * k - kdotn * kdotn, 0.0f);  // ||k_perp||^2
        
        float phase = fmaf(kx, g.c.x, fmaf(ky, g.c.y, kz * g.c.z));
        float2 eikc = cexp_i(phase);
        
        // S_gauss(k) = w * exp(-0.5 * sigma^2 * ||k_perp||^2)
        float exp_arg = -0.5f * g.sigma * g.sigma * r2;
        float Smag = g.w * expf(exp_arg);
        
        float Apar = kpar_coeff * Smag;
        if (!isfinite(Apar)) Apar = 0.0f;
        
        A.x += eikc.x * Apar;
        A.y += eikc.y * Apar;
      }
    }
  }
  
  // Tree reduction in shared memory
  __shared__ float s_A_x[256];
  __shared__ float s_A_y[256];
  
  int tid = threadIdx.x;
  if (tid < blockDim.x) {
    s_A_x[tid] = A.x;
    s_A_y[tid] = A.y;
  }
  __syncthreads();
  
  // Tree reduction
  int n = blockDim.x;
  for (int s = n / 2; s > 0; s >>= 1) {
    if (tid < s && tid + s < n) {
      s_A_x[tid] += s_A_x[tid + s];
      s_A_y[tid] += s_A_y[tid + s];
    }
    __syncthreads();
  }
  
  // Write result to J matrix (row-major: J[q * num_objects + obj])
  if (tid == 0) {
    int idx = q * num_objects + obj;
    J[idx] = make_float2(s_A_x[0], s_A_y[0]);
  }
}

// ===================== Phase 2: Compute Volume Matrix V = J^T D J =====================
// Kernel to compute V[i,j] = (1/(8π³)) · Σ_q w_q · Re(J[q,i] · conj(J[q,j]))
// Each thread computes one entry (i,j) of the symmetric matrix
extern "C" __global__
void compute_volume_matrix_kernel(
  const float2* __restrict__ J,                  // Input: Q × num_objects (row-major)
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
    float2 Ji = J[idx_i];
    float2 Jj = J[idx_j];
    
    // J[q,i] · conj(J[q,j]) = (Ji.x + i*Ji.y) · (Jj.x - i*Jj.y)
    //                        = Ji.x*Jj.x + Ji.y*Jj.y + i*(Ji.y*Jj.x - Ji.x*Jj.y)
    // Real part: Ji.x*Jj.x + Ji.y*Jj.y
    double real_part = (double)Ji.x * (double)Jj.x + (double)Ji.y * (double)Jj.y;
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
  const std::vector<Geometry>& geometries,
  const KGrid& kgrid,
  int blockSize,
  bool enable_profiling)
{
  // Start total timing
  auto t_start_total = std::chrono::high_resolution_clock::now();
  
  IntersectionVolumeMatrixResult result;
  int num_objects = (int)geometries.size();
  
  if (num_objects == 0) {
    result.volume_matrix = Eigen::MatrixXd::Zero(0, 0);
    return result;
  }
  
  int Q = (int)kgrid.kmag.size();
  
  // Flatten geometry data for GPU
  std::vector<TriPacked> all_tris;
  std::vector<DiskPacked> all_disks;
  std::vector<GaussianPacked> all_gaussians;
  std::vector<int> tri_counts(num_objects);
  std::vector<int> disk_counts(num_objects);
  std::vector<int> gaussian_counts(num_objects);
  std::vector<int> geom_types(num_objects);
  std::vector<int> tri_offsets(num_objects);
  std::vector<int> disk_offsets(num_objects);
  std::vector<int> gaussian_offsets(num_objects);
  
  int tri_offset = 0;
  int disk_offset = 0;
  int gaussian_offset = 0;
  
  for (int obj = 0; obj < num_objects; ++obj) {
    geom_types[obj] = (int)geometries[obj].type;
    tri_offsets[obj] = tri_offset;
    disk_offsets[obj] = disk_offset;
    gaussian_offsets[obj] = gaussian_offset;
    
    if (geometries[obj].type == GEOM_TRIANGLE) {
      tri_counts[obj] = (int)geometries[obj].tris.size();
      disk_counts[obj] = 0;
      gaussian_counts[obj] = 0;
      for (const auto& t : geometries[obj].tris) {
        all_tris.push_back(t);
      }
      tri_offset += tri_counts[obj];
    } else if (geometries[obj].type == GEOM_DISK) {
      tri_counts[obj] = 0;
      disk_counts[obj] = (int)geometries[obj].disks.size();
      gaussian_counts[obj] = 0;
      for (const auto& d : geometries[obj].disks) {
        all_disks.push_back(d);
      }
      disk_offset += disk_counts[obj];
    } else if (geometries[obj].type == GEOM_GAUSSIAN) {
      tri_counts[obj] = 0;
      disk_counts[obj] = 0;
      gaussian_counts[obj] = (int)geometries[obj].gaussians.size();
      for (const auto& g : geometries[obj].gaussians) {
        all_gaussians.push_back(g);
      }
      gaussian_offset += gaussian_counts[obj];
    }
  }
  
  // Allocate device memory
  TriPacked* d_tris = nullptr;
  DiskPacked* d_disks = nullptr;
  GaussianPacked* d_gaussians = nullptr;
  int* d_tri_counts = nullptr;
  int* d_disk_counts = nullptr;
  int* d_gaussian_counts = nullptr;
  int* d_geom_types = nullptr;
  int* d_tri_offsets = nullptr;
  int* d_disk_offsets = nullptr;
  int* d_gaussian_offsets = nullptr;
  float3* d_kdirs = nullptr;
  float* d_kmags = nullptr;
  float2* d_J = nullptr;
  double* d_weights = nullptr;
  double* d_V = nullptr;
  
  #define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
      if (d_tris) cudaFree(d_tris); \
      if (d_disks) cudaFree(d_disks); \
      if (d_gaussians) cudaFree(d_gaussians); \
      if (d_tri_counts) cudaFree(d_tri_counts); \
      if (d_disk_counts) cudaFree(d_disk_counts); \
      if (d_gaussian_counts) cudaFree(d_gaussian_counts); \
      if (d_geom_types) cudaFree(d_geom_types); \
      if (d_tri_offsets) cudaFree(d_tri_offsets); \
      if (d_disk_offsets) cudaFree(d_disk_offsets); \
      if (d_gaussian_offsets) cudaFree(d_gaussian_offsets); \
      if (d_kdirs) cudaFree(d_kdirs); \
      if (d_kmags) cudaFree(d_kmags); \
      if (d_J) cudaFree(d_J); \
      if (d_weights) cudaFree(d_weights); \
      if (d_V) cudaFree(d_V); \
      result.volume_matrix = Eigen::MatrixXd::Zero(num_objects, num_objects); \
      return result; \
    } \
  } while(0)
  
  auto t_malloc_start = std::chrono::high_resolution_clock::now();
  
  // Allocate device memory
  if (!all_tris.empty()) {
    CUDA_CHECK(cudaMalloc(&d_tris, all_tris.size() * sizeof(TriPacked)));
  } else {
    CUDA_CHECK(cudaMalloc(&d_tris, sizeof(TriPacked)));  // At least 1 element
  }
  
  if (!all_disks.empty()) {
    CUDA_CHECK(cudaMalloc(&d_disks, all_disks.size() * sizeof(DiskPacked)));
  } else {
    CUDA_CHECK(cudaMalloc(&d_disks, sizeof(DiskPacked)));  // At least 1 element
  }
  
  if (!all_gaussians.empty()) {
    CUDA_CHECK(cudaMalloc(&d_gaussians, all_gaussians.size() * sizeof(GaussianPacked)));
  } else {
    CUDA_CHECK(cudaMalloc(&d_gaussians, sizeof(GaussianPacked)));  // At least 1 element
  }
  
  CUDA_CHECK(cudaMalloc(&d_tri_counts, num_objects * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_disk_counts, num_objects * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_gaussian_counts, num_objects * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_geom_types, num_objects * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_tri_offsets, num_objects * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_disk_offsets, num_objects * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_gaussian_offsets, num_objects * sizeof(int)));
  
  // Allocate k-grid data
  std::vector<float3> h_kdirs(Q);
  for (int q = 0; q < Q; ++q) {
    h_kdirs[q] = make_float3(kgrid.dirs[q][0], kgrid.dirs[q][1], kgrid.dirs[q][2]);
  }
  CUDA_CHECK(cudaMalloc(&d_kdirs, Q * sizeof(float3)));
  CUDA_CHECK(cudaMalloc(&d_kmags, Q * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weights, Q * sizeof(double)));
  
  // Allocate J matrix (Q × num_objects complex numbers)
  CUDA_CHECK(cudaMalloc(&d_J, Q * num_objects * sizeof(float2)));
  
  auto t_malloc_end = std::chrono::high_resolution_clock::now();
  auto t_memcpy_start = std::chrono::high_resolution_clock::now();
  
  // Copy geometry data
  if (!all_tris.empty()) {
    CUDA_CHECK(cudaMemcpy(d_tris, all_tris.data(), all_tris.size() * sizeof(TriPacked), cudaMemcpyHostToDevice));
  }
  if (!all_disks.empty()) {
    CUDA_CHECK(cudaMemcpy(d_disks, all_disks.data(), all_disks.size() * sizeof(DiskPacked), cudaMemcpyHostToDevice));
  }
  if (!all_gaussians.empty()) {
    CUDA_CHECK(cudaMemcpy(d_gaussians, all_gaussians.data(), all_gaussians.size() * sizeof(GaussianPacked), cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(d_tri_counts, tri_counts.data(), num_objects * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_disk_counts, disk_counts.data(), num_objects * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gaussian_counts, gaussian_counts.data(), num_objects * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_geom_types, geom_types.data(), num_objects * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_tri_offsets, tri_offsets.data(), num_objects * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_disk_offsets, disk_offsets.data(), num_objects * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gaussian_offsets, gaussian_offsets.data(), num_objects * sizeof(int), cudaMemcpyHostToDevice));
  
  // Copy k-grid data
  CUDA_CHECK(cudaMemcpy(d_kdirs, h_kdirs.data(), Q * sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kmags, kgrid.kmag.data(), Q * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weights, kgrid.w.data(), Q * sizeof(double), cudaMemcpyHostToDevice));
  
  auto t_memcpy_end = std::chrono::high_resolution_clock::now();
  auto t_kernel1_start = std::chrono::high_resolution_clock::now();
  
  // Phase 1: Compute form factor matrix J
  dim3 grid_J(Q, num_objects);
  dim3 block_J(blockSize);
  compute_form_factor_matrix_kernel<<<grid_J, block_J>>>(
    d_tris, d_tri_counts, d_disks, d_disk_counts, d_gaussians, d_gaussian_counts,
    d_geom_types, d_tri_offsets, d_disk_offsets, d_gaussian_offsets,
    num_objects, d_kdirs, d_kmags, Q, d_J
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
  auto t_kernel1_end = std::chrono::high_resolution_clock::now();
  auto t_kernel2_start = std::chrono::high_resolution_clock::now();
  
  // Phase 2: Compute volume matrix V = J^T D J
  CUDA_CHECK(cudaMalloc(&d_V, num_objects * num_objects * sizeof(double)));
  
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
  std::vector<double> h_V(num_objects * num_objects);
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
    std::cout << "  Memory allocation: " << std::setw(8) << malloc_time << " ms" << std::endl;
    std::cout << "  Memory copy (H->D): " << std::setw(8) << memcpy_time << " ms" << std::endl;
    std::cout << "  Kernel 1 (Form Factor J): " << std::setw(8) << kernel1_time << " ms" << std::endl;
    std::cout << "  Kernel 2 (Volume Matrix): " << std::setw(8) << kernel2_time << " ms" << std::endl;
    std::cout << "  Memory copy (D->H): " << std::setw(8) << result_time << " ms" << std::endl;
    std::cout << "  Total time:         " << std::setw(8) << total_time << " ms" << std::endl;
    std::cout << "==========================================\n" << std::endl;
  }
  
  // Cleanup
  cudaFree(d_tris);
  cudaFree(d_disks);
  cudaFree(d_gaussians);
  cudaFree(d_tri_counts);
  cudaFree(d_disk_counts);
  cudaFree(d_gaussian_counts);
  cudaFree(d_geom_types);
  cudaFree(d_tri_offsets);
  cudaFree(d_disk_offsets);
  cudaFree(d_gaussian_offsets);
  cudaFree(d_kdirs);
  cudaFree(d_kmags);
  cudaFree(d_J);
  cudaFree(d_weights);
  cudaFree(d_V);
  
  #undef CUDA_CHECK
  
  return result;
}

} // namespace PoIntInt


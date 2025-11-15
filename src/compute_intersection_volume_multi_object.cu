#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <array>
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

using PoIntInt::cmul;
using PoIntInt::cexp_i;
using PoIntInt::J1_over_x;
using PoIntInt::Phi_ab;

using PoIntInt::TriPacked;
using PoIntInt::DiskPacked;
using PoIntInt::GaussianPacked;
using PoIntInt::get_geometry_type_name;
using PoIntInt::get_geometry_element_name;
using PoIntInt::GEOM_TRIANGLE;
using PoIntInt::GEOM_DISK;
using PoIntInt::GEOM_GAUSSIAN;

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
  const double3* __restrict__ kdirs,               // Q k-directions
  const double* __restrict__ kmags,                // Q k-magnitudes
  int Q,
  double2* J                                        // Output: Q × num_objects (row-major)
) {
  int q = blockIdx.x;  // k-node index
  int obj = blockIdx.y; // object index
  
  if (q >= Q || obj >= num_objects) return;
  
  double3 u = kdirs[q];
  double k = kmags[q];
  double kx = k * u.x, ky = k * u.y, kz = k * u.z;
  
  double2 A = make_double2(0.0, 0.0);
  int geom_type = geom_types[obj];
  
  if (geom_type == 0) {  // GEOM_TRIANGLE
    int tri_offset = tri_offsets[obj];
    int NF = tri_counts[obj];
    
    // Each thread processes multiple triangles
    for (int i = threadIdx.x; i < NF; i += blockDim.x) {
      TriPacked t = tris_array[tri_offset + i];
      double alpha = kx * t.e1.x + ky * t.e1.y + kz * t.e1.z;
      double beta = kx * t.e2.x + ky * t.e2.y + kz * t.e2.z;
      double gamma = u.x * t.S.x + u.y * t.S.y + u.z * t.S.z;
      double2 phi = Phi_ab(alpha, beta);
      double phase = kx * t.a.x + ky * t.a.y + kz * t.a.z;
      double2 expp = cexp_i(phase);
      double2 scal = cmul(expp, phi);
      A.x += scal.x * gamma;
      A.y += scal.y * gamma;
    }
  } else if (geom_type == 1) {  // GEOM_DISK
    int disk_offset = disk_offsets[obj];
    int ND = disk_counts[obj];
    
    // Each thread processes multiple disks
    for (int i = threadIdx.x; i < ND; i += blockDim.x) {
      DiskPacked disk = disks_array[disk_offset + i];
      if (disk.rho > 0.0 && disk.area >= 0.0) {
        double kpar_coeff = fma(u.x, disk.n.x, fma(u.y, disk.n.y, u.z * disk.n.z));
        double kdotn = k * kpar_coeff;
        double r2 = fmax(k * k - kdotn * kdotn, 0.0);
        double r = sqrt(r2);
        
        double phase = fma(kx, disk.c.x, fma(ky, disk.c.y, kz * disk.c.z));
        double2 eikc = cexp_i(phase);
        
        double Smag;
        if (r < 1e-6) {
          Smag = disk.area;
        } else {
          double x = fmin(disk.rho * r, 100.0);
          Smag = disk.area * 2.0 * J1_over_x(x);
        }
        
        double Apar = kpar_coeff * Smag;
        if (!isfinite(Apar)) Apar = 0.0;
        
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
      if (g.sigma > 0.0 && g.w >= 0.0) {
        double kpar_coeff = fma(u.x, g.n.x, fma(u.y, g.n.y, u.z * g.n.z));
        double kdotn = k * kpar_coeff;
        double r2 = fmax(k * k - kdotn * kdotn, 0.0);  // ||k_perp||^2
        
        double phase = fma(kx, g.c.x, fma(ky, g.c.y, kz * g.c.z));
        double2 eikc = cexp_i(phase);
        
        // S_gauss(k) = w * exp(-0.5 * sigma^2 * ||k_perp||^2)
        double exp_arg = -0.5 * g.sigma * g.sigma * r2;
        double Smag = g.w * exp(exp_arg);
        
        double Apar = kpar_coeff * Smag;
        if (!isfinite(Apar)) Apar = 0.0;
        
        A.x += eikc.x * Apar;
        A.y += eikc.y * Apar;
      }
    }
  }
  
  // Tree reduction in shared memory
  __shared__ double s_A_x[256];
  __shared__ double s_A_y[256];
  
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
    J[idx] = make_double2(s_A_x[0], s_A_y[0]);
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
  double3* d_kdirs = nullptr;
  double* d_kmags = nullptr;
  double2* d_J = nullptr;
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
  std::vector<double3> h_kdirs(Q);
  for (int q = 0; q < Q; ++q) {
    h_kdirs[q] = make_double3(kgrid.dirs[q][0], kgrid.dirs[q][1], kgrid.dirs[q][2]);
  }
  CUDA_CHECK(cudaMalloc(&d_kdirs, Q * sizeof(double3)));
  CUDA_CHECK(cudaMalloc(&d_kmags, Q * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_weights, Q * sizeof(double)));
  
  // Allocate J matrix (Q × num_objects complex numbers)
  CUDA_CHECK(cudaMalloc(&d_J, Q * num_objects * sizeof(double2)));
  
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
  CUDA_CHECK(cudaMemcpy(d_kdirs, h_kdirs.data(), Q * sizeof(double3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kmags, kgrid.kmag.data(), Q * sizeof(double), cudaMemcpyHostToDevice));
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


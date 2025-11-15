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
#include <Eigen/Dense>
#include <complex>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <functional>
#include "compute_intersection_volume.hpp"
#include "geometry/geometry.hpp"
#include "geometry/types.hpp"
#include "form_factor_helpers.hpp"
#include "cuda/cuda_helpers.hpp"

using PoIntInt::cmul;
using PoIntInt::cexp_i;
using PoIntInt::J1_over_x;
using PoIntInt::Phi_ab;

// ===================== kernel =====================

// Unified kernel supporting triangle meshes, point clouds (disks), and Gaussian splats
// type1/type2: 0 = triangle, 1 = disk, 2 = Gaussian
using PoIntInt::TriPacked;
using PoIntInt::DiskPacked;
using PoIntInt::GaussianPacked;
using PoIntInt::get_geometry_type_name;
using PoIntInt::get_geometry_element_name;

extern "C" __global__
void accumulate_intersection_volume_kernel(
  const TriPacked* __restrict__ tris1, int NF1,
  const DiskPacked* __restrict__ disks1, int ND1,
  const GaussianPacked* __restrict__ gaussians1, int NG1,
  int type1,
  const TriPacked* __restrict__ tris2, int NF2,
  const DiskPacked* __restrict__ disks2, int ND2,
  const GaussianPacked* __restrict__ gaussians2, int NG2,
  int type2,
  const double3*  __restrict__ kdirs,      // Q
  const double*  __restrict__ weights_k,  // Q
  const double*   __restrict__ kmags,      // Q: k=tan(t)
  int Q,
  double* out_scalar
){
  int q = blockIdx.x;
  if (q >= Q) return;

  double3 u = kdirs[q];
  double  k = kmags[q];
  double  kx = k*u.x, ky = k*u.y, kz = k*u.z;

  // Accumulate A1_parallel(k) for geometry 1
  // type1: 0 = GEOM_TRIANGLE, 1 = GEOM_DISK, 2 = GEOM_GAUSSIAN
  double2 A1; A1.x = A1.y = 0.0;

  if (type1 == 0) {  // GEOM_TRIANGLE
    // Triangle mesh
    if (NF1 > 0) {
      for (int i = threadIdx.x; i < NF1; i += blockDim.x) {
        TriPacked t = tris1[i];
        double alpha = kx * t.e1.x + ky * t.e1.y + kz * t.e1.z;
        double beta = kx * t.e2.x + ky * t.e2.y + kz * t.e2.z;
        double gamma = u.x * t.S.x + u.y * t.S.y + u.z * t.S.z;
        
        double2 phi = Phi_ab(alpha, beta);
        double phase = kx * t.a.x + ky * t.a.y + kz * t.a.z;
        double2 expp = cexp_i(phase);
        double2 scal = cmul(expp, phi);
        A1.x += scal.x * gamma;
        A1.y += scal.y * gamma;
      }
    }
  } else if (type1 == 1) {  // GEOM_DISK
    if (ND1 > 0) {
      for (int i = threadIdx.x; i < ND1; i += blockDim.x) {
        DiskPacked disk = disks1[i];
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

          A1.x += eikc.x * Apar;
          A1.y += eikc.y * Apar;
        }
      }
    }
  } else if (type1 == 2) {  // GEOM_GAUSSIAN
    if (NG1 > 0) {
      for (int i = threadIdx.x; i < NG1; i += blockDim.x) {
        GaussianPacked g = gaussians1[i];
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
          
          A1.x += eikc.x * Apar;
          A1.y += eikc.y * Apar;
        }
      }
    }
  }

  // Accumulate A2_parallel(k) for geometry 2
  // type2: 0 = GEOM_TRIANGLE, 1 = GEOM_DISK, 2 = GEOM_GAUSSIAN
  double2 A2; A2.x = A2.y = 0.0;

  if (type2 == 0) {  // GEOM_TRIANGLE
    // Triangle mesh
    if (NF2 > 0) {
      for (int i = threadIdx.x; i < NF2; i += blockDim.x) {
        TriPacked t = tris2[i];
        double alpha = kx * t.e1.x + ky * t.e1.y + kz * t.e1.z;
        double beta = kx * t.e2.x + ky * t.e2.y + kz * t.e2.z;
        double gamma = u.x * t.S.x + u.y * t.S.y + u.z * t.S.z;
        
        double2 phi = Phi_ab(alpha, beta);
        double phase = kx * t.a.x + ky * t.a.y + kz * t.a.z;
        double2 expp = cexp_i(phase);
        double2 scal = cmul(expp, phi);
        A2.x += scal.x * gamma;
        A2.y += scal.y * gamma;
      }
    }
  } else if (type2 == 1) {  // GEOM_DISK
    if (ND2 > 0) {
      for (int i = threadIdx.x; i < ND2; i += blockDim.x) {
        DiskPacked disk = disks2[i];
        if (disk.rho > 0.0 && disk.area > 0.0) {
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

          A2.x += eikc.x * Apar;
          A2.y += eikc.y * Apar;
        }
      }
    }
  } else if (type2 == 2) {  // GEOM_GAUSSIAN
    if (NG2 > 0) {
      for (int i = threadIdx.x; i < NG2; i += blockDim.x) {
        GaussianPacked g = gaussians2[i];
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
          
          A2.x += eikc.x * Apar;
          A2.y += eikc.y * Apar;
        }
      }
    }
  }

  // Tree reduction using shared memory
  __shared__ double s_A1_x[256];
  __shared__ double s_A1_y[256];
  __shared__ double s_A2_x[256];
  __shared__ double s_A2_y[256];
  
  int tid = threadIdx.x;
  if (tid < blockDim.x) {
    s_A1_x[tid] = A1.x;
    s_A1_y[tid] = A1.y;
    s_A2_x[tid] = A2.x;
    s_A2_y[tid] = A2.y;
  }
  __syncthreads();
  
  // Tree reduction
  int n = blockDim.x;
  for (int s = n / 2; s > 0; s >>= 1) {
    if (tid < s && tid + s < n) {
      s_A1_x[tid] += s_A1_x[tid + s];
      s_A1_y[tid] += s_A1_y[tid + s];
      s_A2_x[tid] += s_A2_x[tid + s];
      s_A2_y[tid] += s_A2_y[tid + s];
    }
    __syncthreads();
  }
  
  if (tid == 0) {
    double dot_real = s_A1_x[0] * s_A2_x[0] + s_A1_y[0] * s_A2_y[0];
    atomicAdd(out_scalar, weights_k[q] * dot_real);
  }
}

// Unified intersection volume computation supporting both meshes and point clouds
namespace PoIntInt {

// Internal implementation (not exposed in header)
static double compute_intersection_volume_cuda_impl(
  const std::vector<TriPacked>& tris1,
  const std::vector<DiskPacked>& disks1,
  const std::vector<GaussianPacked>& gaussians1,
  GeometryType type1,
  const std::vector<TriPacked>& tris2,
  const std::vector<DiskPacked>& disks2,
  const std::vector<GaussianPacked>& gaussians2,
  GeometryType type2,
  const KGrid& KG,
  int blockSize,
  bool enable_profiling)
{
  // Start total timing
  auto t_start_total = std::chrono::high_resolution_clock::now();
  
  int NF1 = (type1 == GEOM_TRIANGLE) ? (int)tris1.size() : 0;
  int ND1 = (type1 == GEOM_DISK) ? (int)disks1.size() : 0;
  int NG1 = (type1 == GEOM_GAUSSIAN) ? (int)gaussians1.size() : 0;
  int NF2 = (type2 == GEOM_TRIANGLE) ? (int)tris2.size() : 0;
  int ND2 = (type2 == GEOM_DISK) ? (int)disks2.size() : 0;
  int NG2 = (type2 == GEOM_GAUSSIAN) ? (int)gaussians2.size() : 0;
  int Q = (int)KG.kmag.size();

  DiskPacked* d_disks1=nullptr;
  DiskPacked* d_disks2=nullptr;
  TriPacked* d_tris1=nullptr;
  TriPacked* d_tris2=nullptr;
  GaussianPacked* d_gaussians1=nullptr;
  GaussianPacked* d_gaussians2=nullptr;
  double3*     d_dirs=nullptr;
  double*      d_kmag=nullptr;
  double*    d_w=nullptr;
  double*    d_out=nullptr;

  // Helper macro for CUDA error checking with cleanup
  #define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
      if (d_tris1) cudaFree(d_tris1); \
      if (d_tris2) cudaFree(d_tris2); \
      if (d_disks1) cudaFree(d_disks1); \
      if (d_disks2) cudaFree(d_disks2); \
      if (d_gaussians1) cudaFree(d_gaussians1); \
      if (d_gaussians2) cudaFree(d_gaussians2); \
      if (d_dirs) cudaFree(d_dirs); \
      if (d_kmag) cudaFree(d_kmag); \
      if (d_w) cudaFree(d_w); \
      if (d_out) cudaFree(d_out); \
      return 0.0; \
    } \
  } while(0)

  auto t_malloc_start = std::chrono::high_resolution_clock::now();
  // Allocate memory (always allocate at least 1 element to avoid null pointers)
  CUDA_CHECK(cudaMalloc(&d_tris1, (NF1 > 0 ? NF1 : 1) * sizeof(TriPacked)));
  CUDA_CHECK(cudaMalloc(&d_disks1, (ND1 > 0 ? ND1 : 1) * sizeof(DiskPacked)));
  CUDA_CHECK(cudaMalloc(&d_gaussians1, (NG1 > 0 ? NG1 : 1) * sizeof(GaussianPacked)));
  CUDA_CHECK(cudaMalloc(&d_tris2, (NF2 > 0 ? NF2 : 1) * sizeof(TriPacked)));
  CUDA_CHECK(cudaMalloc(&d_disks2, (ND2 > 0 ? ND2 : 1) * sizeof(DiskPacked)));
  CUDA_CHECK(cudaMalloc(&d_gaussians2, (NG2 > 0 ? NG2 : 1) * sizeof(GaussianPacked)));
  CUDA_CHECK(cudaMalloc(&d_dirs, Q *sizeof(double3)));
  CUDA_CHECK(cudaMalloc(&d_kmag, Q *sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_w,    Q *sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_out,  sizeof(double)));
  auto t_malloc_end = std::chrono::high_resolution_clock::now();
  
  auto t_memcpy_start = std::chrono::high_resolution_clock::now();
  if (type1 == GEOM_TRIANGLE && NF1 > 0) {
    CUDA_CHECK(cudaMemcpy(d_tris1, tris1.data(), NF1 * sizeof(TriPacked), cudaMemcpyHostToDevice));
  }
  if (type1 == GEOM_DISK && ND1 > 0) {
    CUDA_CHECK(cudaMemcpy(d_disks1, disks1.data(), ND1 * sizeof(DiskPacked), cudaMemcpyHostToDevice));
  }
  if (type1 == GEOM_GAUSSIAN && NG1 > 0) {
    CUDA_CHECK(cudaMemcpy(d_gaussians1, gaussians1.data(), NG1 * sizeof(GaussianPacked), cudaMemcpyHostToDevice));
  }
  if (type2 == GEOM_TRIANGLE && NF2 > 0) {
    CUDA_CHECK(cudaMemcpy(d_tris2, tris2.data(), NF2 * sizeof(TriPacked), cudaMemcpyHostToDevice));
  }
  if (type2 == GEOM_DISK && ND2 > 0) {
    CUDA_CHECK(cudaMemcpy(d_disks2, disks2.data(), ND2 * sizeof(DiskPacked), cudaMemcpyHostToDevice));
  }
  if (type2 == GEOM_GAUSSIAN && NG2 > 0) {
    CUDA_CHECK(cudaMemcpy(d_gaussians2, gaussians2.data(), NG2 * sizeof(GaussianPacked), cudaMemcpyHostToDevice));
  }
  
  std::vector<double3> hdirs(Q);
  for (int q = 0; q < Q; ++q) {
    hdirs[q] = make_double3((double)KG.dirs[q][0], (double)KG.dirs[q][1], (double)KG.dirs[q][2]);
  }
  CUDA_CHECK(cudaMemcpy(d_dirs, hdirs.data(), Q * sizeof(double3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kmag, KG.kmag.data(), Q * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, KG.w.data(), Q * sizeof(double), cudaMemcpyHostToDevice));
  
  double zero = 0.0;
  CUDA_CHECK(cudaMemcpy(d_out, &zero, sizeof(double), cudaMemcpyHostToDevice));
  auto t_memcpy_end = std::chrono::high_resolution_clock::now();

  dim3 grid(Q);
  dim3 block(blockSize);
  
  auto t_kernel_start = std::chrono::high_resolution_clock::now();
  accumulate_intersection_volume_kernel<<<grid, block, 0>>>(
    d_tris1, NF1, d_disks1, ND1, d_gaussians1, NG1, (int)type1,
    d_tris2, NF2, d_disks2, ND2, d_gaussians2, NG2, (int)type2,
    d_dirs, d_w, d_kmag, Q, d_out);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t_kernel_end = std::chrono::high_resolution_clock::now();

  auto t_result_start = std::chrono::high_resolution_clock::now();
  double I=0.0; // integral I
  CUDA_CHECK(cudaMemcpy(&I, d_out, sizeof(double), cudaMemcpyDeviceToHost));
  auto t_result_end = std::chrono::high_resolution_clock::now();

  cudaFree(d_tris1);
  cudaFree(d_tris2);
  cudaFree(d_disks1);
  cudaFree(d_disks2);
  cudaFree(d_gaussians1);
  cudaFree(d_gaussians2);
  cudaFree(d_dirs);
  cudaFree(d_kmag);
  cudaFree(d_w);
  cudaFree(d_out);
  
  #undef CUDA_CHECK

  // End total timing
  auto t_end_total = std::chrono::high_resolution_clock::now();
  
  // Calculate timing statistics
  auto malloc_time = std::chrono::duration_cast<std::chrono::microseconds>(t_malloc_end - t_malloc_start).count() / 1000.0;
  auto memcpy_time = std::chrono::duration_cast<std::chrono::microseconds>(t_memcpy_end - t_memcpy_start).count() / 1000.0;
  auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(t_kernel_end - t_kernel_start).count() / 1000.0;
  auto result_time = std::chrono::duration_cast<std::chrono::microseconds>(t_result_end - t_result_start).count() / 1000.0;
  auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end_total - t_start_total).count() / 1000.0;
  
  // Print timing information if profiling is enabled
  if (enable_profiling) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== CUDA Volume Computation Profiler ===" << std::endl;
    int count1 = (type1 == GEOM_TRIANGLE) ? NF1 : ((type1 == GEOM_DISK) ? ND1 : NG1);
    int count2 = (type2 == GEOM_TRIANGLE) ? NF2 : ((type2 == GEOM_DISK) ? ND2 : NG2);
    std::cout << "Geometry 1: " << get_geometry_type_name(type1) << " (" << count1 << " " << get_geometry_element_name(type1) << ")" << std::endl;
    std::cout << "Geometry 2: " << get_geometry_type_name(type2) << " (" << count2 << " " << get_geometry_element_name(type2) << ")" << std::endl;
    std::cout << "K-grid nodes: " << Q << std::endl;
    std::cout << "Block size: " << blockSize << std::endl;
    std::cout << "--- Timing (ms) ---" << std::endl;
    std::cout << "  Memory allocation: " << std::setw(8) << malloc_time << " ms" << std::endl;
    std::cout << "  Memory copy (H->D): " << std::setw(8) << memcpy_time << " ms" << std::endl;
    std::cout << "  Kernel execution:   " << std::setw(8) << kernel_time << " ms" << std::endl;
    std::cout << "  Memory copy (D->H): " << std::setw(8) << result_time << " ms" << std::endl;
    std::cout << "  Total time:         " << std::setw(8) << total_time << " ms" << std::endl;
    std::cout << "==========================================\n" << std::endl;
  }

  // V = I/(8π^3)
  return I / (8.0 * CUDART_PI * CUDART_PI * CUDART_PI);
}

// Main interface using Geometry struct
double compute_intersection_volume_cuda(
  const Geometry& geom1,
  const Geometry& geom2,
  const KGrid& KG,
  int blockSize,
  bool enable_profiling)
{
  return compute_intersection_volume_cuda_impl(
    geom1.tris, geom1.disks, geom1.gaussians, geom1.type,
    geom2.tris, geom2.disks, geom2.gaussians, geom2.type,
    KG, blockSize, enable_profiling);
}

// CPU version of intersection volume computation
double compute_intersection_volume_cpu(
  const Geometry& geom1,
  const Geometry& geom2,
  const KGrid& KG,
  bool enable_profiling)
{
  auto t_start = std::chrono::high_resolution_clock::now();
  
  #ifndef M_PI
  #define M_PI 3.14159265358979323846
  #endif
  
  int Q = (int)KG.kmag.size();
  if (Q == 0) return 0.0;
  
  auto t_compute_start = std::chrono::high_resolution_clock::now();
  
  // Parallel reduction over k-points
  // For each k-point: compute A1(k) and A2(k), then Re(A1 * conj(A2)) * weight
  double I = tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, Q),
    0.0,
    [&](const tbb::blocked_range<size_t>& r, double local_sum) -> double {
      for (size_t q = r.begin(); q < r.end(); ++q) {
        // Get k-vector: k = kmag[q] * kdir[q]
        double kmag = (double)KG.kmag[q];
        const auto& kdir_arr = KG.dirs[q];
        Eigen::Vector3d kdir((double)kdir_arr[0], (double)kdir_arr[1], (double)kdir_arr[2]);
        Eigen::Vector3d k = kmag * kdir;
        
        // Compute A1(k) and A2(k) for the two geometries
        std::complex<double> A1 = compute_A_geometry(geom1, k);
        std::complex<double> A2 = compute_A_geometry(geom2, k);
        
        // Compute Re(A1 * conj(A2)) = Re(A1) * Re(A2) + Im(A1) * Im(A2)
        // A1 * conj(A2) = (a1 + i*b1) * (a2 - i*b2) = a1*a2 + b1*b2 + i*(b1*a2 - a1*b2)
        // Real part: a1*a2 + b1*b2
        double real_part = A1.real() * A2.real() + A1.imag() * A2.imag();
        
        // Accumulate: weight * Re(A1 * conj(A2))
        local_sum += KG.w[q] * real_part;
      }
      return local_sum;
    },
    std::plus<double>());
  
  auto t_compute_end = std::chrono::high_resolution_clock::now();
  
  // Final result: V = I / (8π³)
  double volume = I / (8.0 * M_PI * M_PI * M_PI);
  
  auto t_end = std::chrono::high_resolution_clock::now();
  
  if (enable_profiling) {
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t_compute_end - t_compute_start).count() / 1000.0;
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== CPU Volume Computation Profiler ===" << std::endl;
    int count1 = (geom1.type == GEOM_TRIANGLE) ? (int)geom1.tris.size() : ((geom1.type == GEOM_DISK) ? (int)geom1.disks.size() : (int)geom1.gaussians.size());
    int count2 = (geom2.type == GEOM_TRIANGLE) ? (int)geom2.tris.size() : ((geom2.type == GEOM_DISK) ? (int)geom2.disks.size() : (int)geom2.gaussians.size());
    std::cout << "Geometry 1: " << get_geometry_type_name(geom1.type) << " (" << count1 << " " << get_geometry_element_name(geom1.type) << ")" << std::endl;
    std::cout << "Geometry 2: " << get_geometry_type_name(geom2.type) << " (" << count2 << " " << get_geometry_element_name(geom2.type) << ")" << std::endl;
    std::cout << "K-grid nodes: " << Q << std::endl;
    std::cout << "--- Timing (ms) ---" << std::endl;
    std::cout << "  Computation:      " << std::setw(8) << compute_time << " ms" << std::endl;
    std::cout << "  Total time:       " << std::setw(8) << total_time << " ms" << std::endl;
    std::cout << "==========================================\n" << std::endl;
  }
  
  return volume;
}

} // namespace PoIntInt

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

// ===================== utilities =====================

__device__ __forceinline__ double2 cmul(double2 a, double2 b) {
  return make_double2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

__device__ __forceinline__ double2 cexp_i(double phase) {
  double s, c;
  sincos(phase, &s, &c);
  return make_double2(c, s);
}

// J1_over_x(x) := J1(x)/x  (accurate ~1e-6 in double across full range)
__device__ __forceinline__ double J1_over_x(double x)
{
  double ax = fabs(x);

  // Tiny x: even Taylor up to x^8
  if (ax < 1e-3) {
    double x2 = x * x;
    double t = 0.5;                      // 1/2
    t += (-1.0 / 16.0) * x2;              // - x^2/16
    double x4 = x2 * x2;
    t += (1.0 / 384.0) * x4;              // + x^4/384
    double x6 = x4 * x2;
    t += (-1.0 / 18432.0) * x6;           // - x^6/18432
    return t;
  }

  // Small–moderate x: stable series with term recurrence
  if (ax <= 12.0) {
    double q = 0.25 * x * x;            // x^2/4
    double term = 0.5;                    // m=0
    double sum = term;
#pragma unroll
    for (int m = 0; m < 20; ++m) {
      double denom = (double)(m + 1) * (double)(m + 2);
      term *= -q / denom;              // term_{m+1}
      sum += term;
      if (fabs(term) < 1e-7 * fabs(sum)) break;
    }
    return sum;
  }

  // Large x: Hankel asymptotics (3 correction terms), then divide by x
  double invx = 1.0 / ax;
  double invx2 = invx * invx;
  double invx3 = invx2 * invx;

  double chi = ax - 0.75 * CUDART_PI;
  double s, c;
  sincos(chi, &s, &c);

  double amp = sqrt(2.0 / (CUDART_PI * ax));
  double cosp = (1.0 - 15.0 / 128.0 * invx2) * c;
  double sinp = (3.0 / 8.0 * invx - 315.0 / 3072.0 * invx3) * s;
  double J1 = amp * (cosp - sinp);
  return J1 * invx;                          // J1(x)/x
}


// E(z) = (sin z + i (1 - cos z)) / z, stable small-z
__device__ __forceinline__ double2 E_func(double z){
  double az = fabs(z);
  double threshold = 1e-4;
  if (az < threshold){
    // series: sin z / z ≈ 1 - z^2/6 + z^4/120
    //         (1 - cos z)/z ≈ z/2 - z^3/24 + z^5/720
    double z2 = z*z, z4 = z2*z2;
    double real = 1.0 - z2*(1.0/6.0) + z4*(1.0/120.0);
    double imag = z*(0.5) - z*z2*(1.0/24.0) + z4*z*(1.0/720.0);
    return make_double2(real, imag);
  } else {
    double s,c; sincos(z, &s, &c);
    return make_double2(s/z, (1.0 - c)/z);
  }
}

// E'(z) = d/dz E(z)  (stable)
__device__ __forceinline__ double2 E_prime(double z){
  double az = fabs(z);
  double threshold = 1e-4;
  if (az < threshold){
    // from series:
    // Re ≈ -(1/3) z + (1/30) z^3
    // Im ≈  1/2  - (1/8) z^2 + (1/144) z^4
    double z2 = z*z, z3 = z2*z, z4 = z2*z2;
    double real = -(1.0/3.0)*z + (1.0/30.0)*z3;
    double imag =  0.5 - (1.0/8.0)*z2 + (1.0/144.0)*z4;
    return make_double2(real, imag);
  } else {
    double s,c; sincos(z, &s, &c);
    // Re: (z cos z - sin z)/z^2
    // Im: (z sin z - (1 - cos z))/z^2
    double z2 = z*z;
    double re = (z*c - s)/z2;
    double im = (z*s - (1.0 - c))/z2;
    return make_double2(re, im);
  }
}

// Double-precision element functions to match CPU implementation
__device__ __forceinline__ double2 E_func_d(double z) {
  double az = fabs(z);
  double threshold = 1e-4;
  if (az < threshold) {
    double z2 = z*z, z4 = z2*z2;
    double real = 1.0 - z2/6.0 + z4/120.0;
    double imag = z*0.5 - z*z2/24.0 + z4*z/720.0;
    return make_double2(real, imag);
  } else {
    double s = sin(z);
    double c = cos(z);
    return make_double2(s/z, (1.0 - c)/z);
  }
}

__device__ __forceinline__ double2 E_prime_d(double z) {
  double az = fabs(z);
  double threshold = 1e-4;
  if (az < threshold) {
    double z2 = z*z, z3 = z2*z, z4 = z2*z2;
    double real = -z/3.0 + z3/30.0;
    double imag = 0.5 - z2/8.0 + z4/144.0;
    return make_double2(real, imag);
  } else {
    double s = sin(z);
    double c = cos(z);
    double z2 = z*z;
    double re = (z*c - s)/z2;
    double im = (z*s - (1.0 - c))/z2;
    return make_double2(re, im);
  }
}

__device__ __forceinline__ double2 Phi_ab_d(double alpha, double beta) {
  double d = beta - alpha;
  double threshold = 1e-3;  // Match CPU: 1e-3
  if (fabs(d) < threshold) {
    double2 Ep = E_prime_d(0.5*(alpha+beta));
    return make_double2(2.0*Ep.y, -2.0*Ep.x);
  } else {
    double2 Ea = E_func_d(alpha);
    double2 Eb = E_func_d(beta);
    double2 num = make_double2(Eb.x - Ea.x, Eb.y - Ea.y);
    double invd = 1.0/d;
    double2 q = make_double2(num.x*invd, num.y*invd);
    return make_double2(2.0*q.y, -2.0*q.x);
  }
}

// Conversion helper (kept for compatibility, but should not be needed with full double)
__device__ __forceinline__ double2 double2_to_float2(double2 d) {
  return d;  // No-op since we're using double everywhere
}

// ===================== kernel =====================

// Unified kernel supporting triangle meshes, point clouds (disks), and Gaussian splats
// type1/type2: 0 = triangle, 1 = disk, 2 = Gaussian
using PoIntInt::TriPacked;
using PoIntInt::DiskPacked;
using PoIntInt::GaussianPacked;

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
        
        double2 phi = Phi_ab_d(alpha, beta);
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
        
        double2 phi = Phi_ab_d(alpha, beta);
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
    if (type1 == GEOM_TRIANGLE) {
      std::cout << "Geometry 1: Triangle mesh (" << NF1 << " faces)" << std::endl;
    } else if (type1 == GEOM_DISK) {
      std::cout << "Geometry 1: Point cloud (" << ND1 << " disks)" << std::endl;
    } else {
      std::cout << "Geometry 1: Gaussian splats (" << NG1 << " gaussians)" << std::endl;
    }
    if (type2 == GEOM_TRIANGLE) {
      std::cout << "Geometry 2: Triangle mesh (" << NF2 << " faces)" << std::endl;
    } else if (type2 == GEOM_DISK) {
      std::cout << "Geometry 2: Point cloud (" << ND2 << " disks)" << std::endl;
    } else {
      std::cout << "Geometry 2: Gaussian splats (" << NG2 << " gaussians)" << std::endl;
    }
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
    if (geom1.type == GEOM_TRIANGLE) {
      std::cout << "Geometry 1: Triangle mesh (" << geom1.tris.size() << " faces)" << std::endl;
    } else if (geom1.type == GEOM_DISK) {
      std::cout << "Geometry 1: Point cloud (" << geom1.disks.size() << " disks)" << std::endl;
    } else {
      std::cout << "Geometry 1: Gaussian splats (" << geom1.gaussians.size() << " gaussians)" << std::endl;
    }
    if (geom2.type == GEOM_TRIANGLE) {
      std::cout << "Geometry 2: Triangle mesh (" << geom2.tris.size() << " faces)" << std::endl;
    } else if (geom2.type == GEOM_DISK) {
      std::cout << "Geometry 2: Point cloud (" << geom2.disks.size() << " disks)" << std::endl;
    } else {
      std::cout << "Geometry 2: Gaussian splats (" << geom2.gaussians.size() << " gaussians)" << std::endl;
    }
    std::cout << "K-grid nodes: " << Q << std::endl;
    std::cout << "--- Timing (ms) ---" << std::endl;
    std::cout << "  Computation:      " << std::setw(8) << compute_time << " ms" << std::endl;
    std::cout << "  Total time:       " << std::setw(8) << total_time << " ms" << std::endl;
    std::cout << "==========================================\n" << std::endl;
  }
  
  return volume;
}

} // namespace PoIntInt

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
#include "compute_volume.hpp"
#include "geometry_types.hpp"

// ===================== utilities =====================

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
  return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

__device__ __forceinline__ float2 cexp_i(float phase) {
  float s, c;
  __sincosf(phase, &s, &c);
  return make_float2(c, s);
}

// J1_over_x(x) := J1(x)/x  (accurate ~1e-6 in float across full range)
__device__ __forceinline__ float J1_over_x(float x)
{
  float ax = fabsf(x);

  // Tiny x: even Taylor up to x^8
  if (ax < 1e-3f) {
    float x2 = x * x;
    float t = 0.5f;                      // 1/2
    t += (-1.0f / 16.0f) * x2;              // - x^2/16
    float x4 = x2 * x2;
    t += (1.0f / 384.0f) * x4;              // + x^4/384
    float x6 = x4 * x2;
    t += (-1.0f / 18432.0f) * x6;           // - x^6/18432
    return t;
  }

  // Small–moderate x: stable series with term recurrence
  if (ax <= 12.0f) {
    float q = 0.25f * x * x;            // x^2/4
    float term = 0.5f;                    // m=0
    float sum = term;
#pragma unroll
    for (int m = 0; m < 20; ++m) {
      float denom = (float)(m + 1) * (float)(m + 2);
      term *= -q / denom;              // term_{m+1}
      sum += term;
      if (fabsf(term) < 1e-7f * fabsf(sum)) break;
    }
    return sum;
  }

  // Large x: Hankel asymptotics (3 correction terms), then divide by x
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
  return J1 * invx;                          // J1(x)/x
}


// E(z) = (sin z + i (1 - cos z)) / z, stable small-z
__device__ __forceinline__ float2 E_func(float z){
  float az = fabsf(z);
  float threshold = 1e-4f;
  if (az < threshold){
    // series: sin z / z ≈ 1 - z^2/6 + z^4/120
    //         (1 - cos z)/z ≈ z/2 - z^3/24 + z^5/720
    float z2 = z*z, z4 = z2*z2;
    float real = 1.0f - z2*(1.0f/6.0f) + z4*(1.0f/120.0f);
    float imag = z*(0.5f) - z*z2*(1.0f/24.0f) + z4*z*(1.0f/720.0f);
    return make_float2(real, imag);
  } else {
    float s,c; __sincosf(z, &s, &c);
    return make_float2(s/z, (1.0f - c)/z);
  }
}

// E'(z) = d/dz E(z)  (stable)
__device__ __forceinline__ float2 E_prime(float z){
  float az = fabsf(z);
  float threshold = 1e-4f;
  if (az < threshold){
    // from series:
    // Re ≈ -(1/3) z + (1/30) z^3
    // Im ≈  1/2  - (1/8) z^2 + (1/120) z^4
    float z2 = z*z, z3 = z2*z, z4 = z2*z2;
    float real = -(1.0f/3.0f)*z + (1.0f/30.0f)*z3;
    float imag =  0.5f - (1.0f/8.0f)*z2 + (1.0f/120.0f)*z4;
    return make_float2(real, imag);
  } else {
    float s,c; __sincosf(z, &s, &c);
    // Re: (z cos z - sin z)/z^2
    // Im: (z sin z - (1 - cos z))/z^2
    float z2 = z*z;
    float re = (z*c - s)/z2;
    float im = (z*s - (1.0f - c))/z2;
    return make_float2(re, im);
  }
}

// Phi(α,β) = 2i [E(β) - E(α)]/(β-α)  , with α≈β -> 2i E'(α)
__device__ __forceinline__ float2 Phi_ab(float alpha, float beta){
  float d = beta - alpha;
  float threshold = 1e-5f;
  if (fabsf(d) < threshold){
    float2 Ep = E_prime(0.5f*(alpha+beta));
    // 2i * (Ep.re + i Ep.im) = 2i*Ep.re - 2*Ep.im
    return make_float2(2.0f*Ep.y, -2.0f*Ep.x);
  } else {
    float2 Ea = E_func(alpha);
    float2 Eb = E_func(beta);
    // num = Eb - Ea
    float2 num = make_float2(Eb.x - Ea.x, Eb.y - Ea.y);
    // 2i * num/d
    float invd = 1.0f/d;
    float2 q = make_float2(num.x*invd, num.y*invd);
    // 2i*q = (-2*q.y, 2*q.x)
    return make_float2(2.0f*q.y, -2.0f*q.x);
  }
}

// ===================== kernel =====================

// Unified kernel supporting both triangle meshes and point clouds (disks)
// type1/type2: 0 = triangle, 1 = disk
using PoIntInt::TriPacked;
using PoIntInt::DiskPacked;

extern "C" __global__
void accumulate_intersection_volume_kernel(
  const TriPacked* __restrict__ tris1, int NF1,
  const DiskPacked* __restrict__ disks1, int ND1,
  int type1,
  const TriPacked* __restrict__ tris2, int NF2,
  const DiskPacked* __restrict__ disks2, int ND2,
  int type2,
  const float3*  __restrict__ kdirs,      // Q
  const double*  __restrict__ weights_k,  // Q
  const float*   __restrict__ kmags,      // Q: k=tan(t)
  int Q,
  double* out_scalar
){
  int q = blockIdx.x;
  if (q >= Q) return;

  float3 u = kdirs[q];
  float  k = kmags[q];
  float  kx = k*u.x, ky = k*u.y, kz = k*u.z;

  // Accumulate A1_parallel(k) for geometry 1
  // type1: 0 = GEOM_TRIANGLE, 1 = GEOM_DISK
  float2 A1; A1.x = A1.y = 0.0f;

  if (type1 == 0) {  // GEOM_TRIANGLE
    // Triangle mesh
    if (NF1 > 0) {
      for (int i = threadIdx.x; i < NF1; i += blockDim.x) {
        TriPacked t = tris1[i];
        float alpha = kx * t.e1.x + ky * t.e1.y + kz * t.e1.z;
        float beta = kx * t.e2.x + ky * t.e2.y + kz * t.e2.z;
        float gamma = u.x * t.S.x + u.y * t.S.y + u.z * t.S.z;
        float2 phi = Phi_ab(alpha, beta);
        float phase = kx * t.a.x + ky * t.a.y + kz * t.a.z;
        float2 expp = cexp_i(phase);
        float2 scal = cmul(expp, phi);
        A1.x += scal.x * gamma;
        A1.y += scal.y * gamma;
      }
    }
  } else if (type1 == 1) {  // GEOM_DISK
    if (ND1 > 0) {
      for (int i = threadIdx.x; i < ND1; i += blockDim.x) {
        DiskPacked disk = disks1[i];
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

          A1.x += eikc.x * Apar;
          A1.y += eikc.y * Apar;
        }
      }
    }
  }

  // Accumulate A2_parallel(k) for geometry 2
  // type2: 0 = GEOM_TRIANGLE, 1 = GEOM_DISK
  float2 A2; A2.x = A2.y = 0.0f;

  if (type2 == 0) {  // GEOM_TRIANGLE
    // Triangle mesh
    if (NF2 > 0) {
      for (int i = threadIdx.x; i < NF2; i += blockDim.x) {
        TriPacked t = tris2[i];
        float alpha = kx * t.e1.x + ky * t.e1.y + kz * t.e1.z;
        float beta = kx * t.e2.x + ky * t.e2.y + kz * t.e2.z;
        float gamma = u.x * t.S.x + u.y * t.S.y + u.z * t.S.z;
        float2 phi = Phi_ab(alpha, beta);
        float phase = kx * t.a.x + ky * t.a.y + kz * t.a.z;
        float2 expp = cexp_i(phase);
        float2 scal = cmul(expp, phi);
        A2.x += scal.x * gamma;
        A2.y += scal.y * gamma;
      }
    }
  } else if (type2 == 1) {  // GEOM_DISK
    if (ND2 > 0) {
      for (int i = threadIdx.x; i < ND2; i += blockDim.x) {
        DiskPacked disk = disks2[i];
        if (disk.rho > 0.0f && disk.area > 0.0f) {
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

          A2.x += eikc.x * Apar;
          A2.y += eikc.y * Apar;
        }
      }
    }
  }

  // Tree reduction using shared memory
  __shared__ float s_A1_x[256];
  __shared__ float s_A1_y[256];
  __shared__ float s_A2_x[256];
  __shared__ float s_A2_y[256];
  
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
    float dot_real = s_A1_x[0] * s_A2_x[0] + s_A1_y[0] * s_A2_y[0];
    atomicAdd(out_scalar, weights_k[q] * dot_real);
  }
}

// Unified intersection volume computation supporting both meshes and point clouds
namespace PoIntInt {

double compute_intersection_volume_cuda(
  const std::vector<TriPacked>& tris1,
  const std::vector<DiskPacked>& disks1,
  GeometryType type1,
  const std::vector<TriPacked>& tris2,
  const std::vector<DiskPacked>& disks2,
  GeometryType type2,
  const KGrid& KG,
  int blockSize)
{
  // Start total timing
  auto t_start_total = std::chrono::high_resolution_clock::now();
  
  int NF1 = (type1 == GEOM_TRIANGLE) ? (int)tris1.size() : 0;
  int ND1 = (type1 == GEOM_DISK) ? (int)disks1.size() : 0;
  int NF2 = (type2 == GEOM_TRIANGLE) ? (int)tris2.size() : 0;
  int ND2 = (type2 == GEOM_DISK) ? (int)disks2.size() : 0;
  int Q = (int)KG.kmag.size();

  DiskPacked* d_disks1=nullptr;
  DiskPacked* d_disks2=nullptr;
  TriPacked* d_tris1=nullptr;
  TriPacked* d_tris2=nullptr;
  float3*     d_dirs=nullptr;
  float*      d_kmag=nullptr;
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
      if (d_dirs) cudaFree(d_dirs); \
      if (d_kmag) cudaFree(d_kmag); \
      if (d_w) cudaFree(d_w); \
      if (d_out) cudaFree(d_out); \
      return 0.0; \
    } \
  } while(0)

  auto t_malloc_start = std::chrono::high_resolution_clock::now();
  // Allocate memory (always allocate at least 1 element to avoid null pointers)
  if (type1 == GEOM_TRIANGLE) {
    CUDA_CHECK(cudaMalloc(&d_tris1, (NF1 > 0 ? NF1 : 1) * sizeof(TriPacked)));
    CUDA_CHECK(cudaMalloc(&d_disks1, sizeof(DiskPacked)));
  } else {
    CUDA_CHECK(cudaMalloc(&d_disks1, (ND1 > 0 ? ND1 : 1) * sizeof(DiskPacked)));
    CUDA_CHECK(cudaMalloc(&d_tris1, sizeof(TriPacked)));
  }
  if (type2 == GEOM_TRIANGLE) {
    CUDA_CHECK(cudaMalloc(&d_tris2, (NF2 > 0 ? NF2 : 1) * sizeof(TriPacked)));
    CUDA_CHECK(cudaMalloc(&d_disks2, sizeof(DiskPacked)));
  } else {
    CUDA_CHECK(cudaMalloc(&d_disks2, (ND2 > 0 ? ND2 : 1) * sizeof(DiskPacked)));
    CUDA_CHECK(cudaMalloc(&d_tris2, sizeof(TriPacked)));
  }
  CUDA_CHECK(cudaMalloc(&d_dirs, Q *sizeof(float3)));
  CUDA_CHECK(cudaMalloc(&d_kmag, Q *sizeof(float)));
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
  if (type2 == GEOM_TRIANGLE && NF2 > 0) {
    CUDA_CHECK(cudaMemcpy(d_tris2, tris2.data(), NF2 * sizeof(TriPacked), cudaMemcpyHostToDevice));
  }
  if (type2 == GEOM_DISK && ND2 > 0) {
    CUDA_CHECK(cudaMemcpy(d_disks2, disks2.data(), ND2 * sizeof(DiskPacked), cudaMemcpyHostToDevice));
  }
  
  std::vector<float3> hdirs(Q);
  for (int q = 0; q < Q; ++q) {
    hdirs[q] = make_float3(KG.dirs[q][0], KG.dirs[q][1], KG.dirs[q][2]);
  }
  CUDA_CHECK(cudaMemcpy(d_dirs, hdirs.data(), Q * sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kmag, KG.kmag.data(), Q * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, KG.w.data(), Q * sizeof(double), cudaMemcpyHostToDevice));
  
  double zero = 0.0;
  CUDA_CHECK(cudaMemcpy(d_out, &zero, sizeof(double), cudaMemcpyHostToDevice));
  auto t_memcpy_end = std::chrono::high_resolution_clock::now();

  dim3 grid(Q);
  dim3 block(blockSize);
  
  auto t_kernel_start = std::chrono::high_resolution_clock::now();
  accumulate_intersection_volume_kernel<<<grid, block, 0>>>(
    d_tris1, NF1, d_disks1, ND1, (int)type1,
    d_tris2, NF2, d_disks2, ND2, (int)type2,
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
  
  // Print timing information
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\n=== CUDA Volume Computation Profiler ===" << std::endl;
  if (type1 == GEOM_TRIANGLE) {
    std::cout << "Geometry 1: Triangle mesh (" << NF1 << " faces)" << std::endl;
  } else {
    std::cout << "Geometry 1: Point cloud (" << ND1 << " disks)" << std::endl;
  }
  if (type2 == GEOM_TRIANGLE) {
    std::cout << "Geometry 2: Triangle mesh (" << NF2 << " faces)" << std::endl;
  } else {
    std::cout << "Geometry 2: Point cloud (" << ND2 << " disks)" << std::endl;
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

  // V = I/(8π^3)
  return I / (8.0 * CUDART_PI * CUDART_PI * CUDART_PI);
}

// Legacy function for triangle-triangle intersection (backward compatibility)
double compute_intersection_volume_cuda(
  const std::vector<TriPacked>& tris1,
  const std::vector<TriPacked>& tris2,
  const KGrid& KG,
  int blockSize)
{
  std::vector<DiskPacked> empty_disks1, empty_disks2;
  return compute_intersection_volume_cuda(
    tris1, empty_disks1, GEOM_TRIANGLE,
    tris2, empty_disks2, GEOM_TRIANGLE,
    KG, blockSize);
}

} // namespace PoIntInt

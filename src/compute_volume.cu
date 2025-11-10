#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <array>
#include <cmath>
#include <cassert>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <math_constants.h>
// Project headers - use quotes for better IntelliSense support
#include "gauss_legendre.hpp"
#include "compute_volume.hpp"

// ===================== utilities =====================

// Complex 3-vector type
struct float3c { float xr, xi, yr, yi, zr, zi; };

__device__ __forceinline__ float2 cadd(float2 a, float2 b){ return make_float2(a.x+b.x, a.y+b.y); }
__device__ __forceinline__ float2 cmul(float2 a, float2 b){ return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
__device__ __forceinline__ float2 cexp_i(float phase){ float s, c; __sincosf(phase, &s, &c); return make_float2(c, s); }

// TriPacked is now defined in compute_volume.hpp

__device__ __forceinline__ float3 r3add(const float3& u, const float3& v){
  return make_float3(u.x+v.x, u.y+v.y, u.z+v.z);
}
__device__ __forceinline__ float3 r3scale(const float3& u, float s){
  return make_float3(u.x*s, u.y*s, u.z*s);
}
__device__ __forceinline__ float  r3dot(const float3& u, const float3& v){
  return u.x*v.x + u.y*v.y + u.z*v.z;
}

// complex scalar times real vector -> complex 3-vector
__device__ __forceinline__ void cscalar_times_vec(float2 c, const float3& v, float3c& out_accum) {
  out_accum.xr += c.x * v.x; out_accum.xi += c.y * v.x;
  out_accum.yr += c.x * v.y; out_accum.yi += c.y * v.y;
  out_accum.zr += c.x * v.z; out_accum.zi += c.y * v.z;
}

// sinc(z) = sin z / z, stable small-z
__device__ __forceinline__ float sinc(float z) {
  float az = fabsf(z);
  float threshold = 1e-4f;
  if (az < threshold) {
    // series: sin z / z ≈ 1 - z^2/6 + z^4/120
    //         (1 - cos z)/z ≈ z/2 - z^3/24 + z^5/720
    float z2 = z * z, z4 = z2 * z2;
    return 1.0f - z2 * (1.0f / 6.0f) + z4 * (1.0f / 120.0f);
  }
  else {
    float s, c; __sincosf(z, &s, &c);
    return s / z;
  }
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
    return make_float2(-2.0f*Ep.y, 2.0f*Ep.x);
  } else {
    float2 Ea = E_func(alpha);
    float2 Eb = E_func(beta);
    // num = Eb - Ea
    float2 num = make_float2(Eb.x - Ea.x, Eb.y - Ea.y);
    // 2i * num/d
    float invd = 1.0f/d;
    float2 q = make_float2(num.x*invd, num.y*invd);
    // 2i*q = (-2*q.y, 2*q.x)
    return make_float2(-2.0f*q.y, 2.0f*q.x);
  }
}

/*
__device__ __forceinline__ double Ak_squared_norm(float r)
{
  double norm;
  float threshold = 1e-3f;
  if (fabsf(r) < threshold)
  {
    norm = (double)(-r / 3.0f + r * r * r / 30.0f);
  }
  else
  {
    float s, c; __sincosf(r, &s, &c);
    norm = (double)((c - s / r) / r);
  }
  norm *= 4.0 * CUDART_PI;
  return norm * norm;
}

__device__ __forceinline__ double unit_cube_Ak_squared_norm(float kx, float ky, float kz)
{
  float norm = sinc(0.5f * kx) * sinc(0.5f * ky) * sinc(0.5f * kz);
  return (double)(norm * norm * (kx * kx + ky * ky + kz * kz));
}*/

// ===================== kernel =====================

// One block per k-node; threads loop over faces and reduce.
// Inputs:
//  tris1[NF1] : per-face packed data for mesh 1
//  tris2[NF2] : per-face packed data for mesh 2
//  kdirs[Q] : unit directions
//  weights_k[Q] : precomputed weights
//  kmags[Q] : k = tan(t)
// Accumulation:
//  out_scalar[1] : double, atomically add weighted Real(A1 · A2*)
extern "C" __global__
void accumulate_intersection_volume_kernel(
  const TriPacked* __restrict__ tris1, int NF1,
  const TriPacked* __restrict__ tris2, int NF2,
  const float3*  __restrict__ kdirs,      // Q
  const double*  __restrict__ weights_k,  // Q
  const float*   __restrict__ kmags,      // Q: k=tan(t)
  int Q,
  double* out_scalar
){
  extern __shared__ double sdata[]; // two warps worth if needed, but we'll just use registers + atomic at end
  int q = blockIdx.x;
  if (q >= Q) return;

  float3 u = kdirs[q];
  float  k = kmags[q];
  float  kx = k*u.x, ky = k*u.y, kz = k*u.z;

  // Accumulate complex 3-vector A1(k) for mesh 1
  float2 A1; A1.x = A1.y = 0.0f;

  // loop over faces of mesh 1 strided by threads
  for (int i = threadIdx.x; i < NF1; i += blockDim.x){
    TriPacked t = tris1[i];
    // alpha = k · e1 ; beta = k · e2
    float alpha = kx*t.e1.x + ky*t.e1.y + kz*t.e1.z;
    float beta  = kx*t.e2.x + ky*t.e2.y + kz*t.e2.z;
    float gamma = u.x * t.S.x + u.y * t.S.y + u.z * t.S.z;
    float2 phi  = Phi_ab(alpha, beta);      // complex scalar
    float phase = kx*t.a.x + ky*t.a.y + kz*t.a.z;
    float2 expp = cexp_i(phase);            // e^{i k·a}
    float2 scal = cmul(expp, phi);          // complex scalar
    A1.x += scal.x * gamma;
    A1.y += scal.y * gamma;
  }

  // Accumulate complex 3-vector A2(k) for mesh 2
  float2 A2; A2.x = A2.y = 0.0f;

  // loop over faces of mesh 2 strided by threads
  for (int i = threadIdx.x; i < NF2; i += blockDim.x){
    TriPacked t = tris2[i];
    // alpha = k · e1 ; beta = k · e2
    float alpha = kx*t.e1.x + ky*t.e1.y + kz*t.e1.z;
    float beta  = kx*t.e2.x + ky*t.e2.y + kz*t.e2.z;
    float gamma = u.x * t.S.x + u.y * t.S.y + u.z * t.S.z;
    float2 phi  = Phi_ab(alpha, beta);      // complex scalar
    float phase = kx*t.a.x + ky*t.a.y + kz*t.a.z;
    float2 expp = cexp_i(phase);            // e^{i k·a}
    float2 scal = cmul(expp, phi);          // complex scalar
    A2.x += scal.x * gamma;
    A2.y += scal.y * gamma;
  }

  // intra-block reduction of A1 and A2
  __shared__ float2 A1block, A2block; // one per block for each mesh
  if (threadIdx.x==0){ 
    A1block.x=A1block.y=0.0f;
    A2block.x=A2block.y=0.0f;
  }
  __syncthreads();

  // reduce A1 across threads with atomics on shared struct
  atomicAdd(&A1block.x, A1.x);
  atomicAdd(&A1block.y, A1.y);

  // reduce A2 across threads with atomics on shared struct
  atomicAdd(&A2block.x, A2.x);
  atomicAdd(&A2block.y, A2.y);
  __syncthreads();

  if (threadIdx.x==0){
    // contribution: weight_k * Real(A1 · A2*)
    // where A2* is the complex conjugate of A2
    float dot_real = A1block.x * A2block.x + A1block.y * A2block.y;
    double contrib = weights_k[q] * dot_real;
    atomicAdd(out_scalar, contrib);
  }
}

// Host wrappers ===============================================================
// KGrid and TriPacked are now defined in compute_volume.hpp

KGrid build_kgrid(
  const std::vector<std::array<double,3>>& leb_dirs,
  const std::vector<double>& leb_w,
  int Nrad)
{
  // radial t in [0, π/2]; Gauss-Legendre
  auto [t, wt] = gauss_legendre_interval(Nrad, 0.0, 0.5*CUDART_PI);

  KGrid KG;
  // Note: The integral is V = (1/(4π)) ∫ |A(k)|² / k² d³k
  // In spherical coords: d³k = k² dk dΩ, so the k² cancels: V = (1/(4π)) ∫ |A(k)|² dk dΩ
  // With k = tan(t), we have dk = sec²(t) dt, so: V = (1/(4π)) ∫ |A(k)|² sec²(t) dt dΩ
  // The weights should be: w_angular * w_radial * sec²(t)
  // The final division by (8π^3) happens in compute_intersection_volume_cuda, so we don't include it here
  for (int ir=0; ir<Nrad; ++ir){
    double ti = t[ir], wti = wt[ir];
    double sec2 = 1.0 / std::cos(ti) / std::cos(ti);
    double k    = std::tan(ti);
    for (size_t j=0; j<leb_dirs.size(); ++j){
      const auto& d = leb_dirs[j];
      KG.dirs.push_back( { (float)d[0], (float)d[1], (float)d[2] } );
      KG.kmag.push_back( (float)k );
      // weight for this node: leb_w[j] * (wti * sec^2)
      // leb_w[j] integrates over solid angle (sums to 4π)
      // wti integrates over t in [0, π/2]
      // sec² accounts for dk = sec²(t) dt transformation
      KG.w.push_back( leb_w[j] * (wti * sec2) );
    }
  }
  return KG;
}

std::vector<TriPacked> pack_tris(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F){
  assert(F.cols()==3 && V.cols()==3);
  std::vector<TriPacked> T(F.rows());
  for (int i=0;i<F.rows();++i){
    int ia=F(i,0), ib=F(i,1), ic=F(i,2);
    Eigen::Vector3d a = V.row(ia), b=V.row(ib), c=V.row(ic);
    Eigen::Vector3d e1=b-a, e2=c-a;
    Eigen::Vector3d S = 0.5*(e1.cross(e2));
    T[i].a  = make_float3((float)a.x(), (float)a.y(), (float)a.z());
    T[i].e1 = make_float3((float)e1.x(), (float)e1.y(), (float)e1.z());
    T[i].e2 = make_float3((float)e2.x(), (float)e2.y(), (float)e2.z());
    T[i].S  = make_float3((float)S.x(), (float)S.y(), (float)S.z());
  }
  return T;
}

// Returns intersection volume V = I/(8π^3)
double compute_intersection_volume_cuda(
  const std::vector<TriPacked>& tris1,
  const std::vector<TriPacked>& tris2,
  const KGrid& KG,
  int blockSize)
{
  // Start total timing
  auto t_start_total = std::chrono::high_resolution_clock::now();
  
  int NF1 = (int)tris1.size();
  int NF2 = (int)tris2.size();
  int Q  = (int)KG.kmag.size();

  auto t_prepare_start = std::chrono::high_resolution_clock::now();
  // Create CUDA events for kernel timing
  /*cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);*/
  auto t_prepare_end = std::chrono::high_resolution_clock::now();

  TriPacked* d_tris1=nullptr;
  TriPacked* d_tris2=nullptr;
  float3*     d_dirs=nullptr;
  float*      d_kmag=nullptr;
  double*    d_w=nullptr;
  double*    d_out=nullptr;

  auto t_malloc_start = std::chrono::high_resolution_clock::now();
  cudaMalloc(&d_tris1, NF1*sizeof(TriPacked));
  cudaMalloc(&d_tris2, NF2*sizeof(TriPacked));
  cudaMalloc(&d_dirs, Q *sizeof(float3));
  cudaMalloc(&d_kmag, Q *sizeof(float));
  cudaMalloc(&d_w,    Q *sizeof(double));
  cudaMalloc(&d_out,  sizeof(double));
  auto t_malloc_end = std::chrono::high_resolution_clock::now();
  
  auto t_memcpy_start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(d_tris1, tris1.data(), NF1*sizeof(TriPacked), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tris2, tris2.data(), NF2*sizeof(TriPacked), cudaMemcpyHostToDevice);
  // pack dirs
  std::vector<float3> hdirs(Q);
  for (int q=0;q<Q;++q) hdirs[q] = make_float3(KG.dirs[q][0], KG.dirs[q][1], KG.dirs[q][2]);
  cudaMemcpy(d_dirs, hdirs.data(), Q*sizeof(float3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_kmag, KG.kmag.data(), Q*sizeof(float),  cudaMemcpyHostToDevice);
  cudaMemcpy(d_w,    KG.w.data(),    Q*sizeof(double), cudaMemcpyHostToDevice);
  double zero=0.0;
  cudaMemcpy(d_out, &zero, sizeof(double), cudaMemcpyHostToDevice);
  auto t_memcpy_end = std::chrono::high_resolution_clock::now();

  dim3 grid(Q);
  dim3 block(blockSize);
  size_t shmem = 0;
  
  // Record kernel start time
  auto t_kernel_start = std::chrono::high_resolution_clock::now();
  // cudaEventRecord(start);
  accumulate_intersection_volume_kernel<<<grid, block, shmem>>>(
    d_tris1, NF1, d_tris2, NF2, d_dirs, d_w, d_kmag, Q, d_out);
  // Record kernel stop time
  // cudaEventRecord(stop);
  cudaDeviceSynchronize();
  auto t_kernel_end = std::chrono::high_resolution_clock::now();

  // Get kernel execution time
  float kernel_ms = 0.0f;
  // cudaEventElapsedTime(&kernel_ms, start, stop);

  auto t_result_start = std::chrono::high_resolution_clock::now();
  double I=0.0; // integral I
  cudaMemcpy(&I, d_out, sizeof(double), cudaMemcpyDeviceToHost);
  auto t_result_end = std::chrono::high_resolution_clock::now();

  auto t_destroy_start = std::chrono::high_resolution_clock::now();
  cudaFree(d_tris1); cudaFree(d_tris2); cudaFree(d_dirs); cudaFree(d_kmag); cudaFree(d_w); cudaFree(d_out);
  
  // Clean up events
  // cudaEventDestroy(start);
  // cudaEventDestroy(stop);
  auto t_destroy_end = std::chrono::high_resolution_clock::now();

  // End total timing
  auto t_end_total = std::chrono::high_resolution_clock::now();
  
  // Calculate timing statistics
  auto prepare_time = std::chrono::duration_cast<std::chrono::microseconds>(t_prepare_end - t_prepare_start).count() / 1000.0;
  auto malloc_time = std::chrono::duration_cast<std::chrono::microseconds>(t_malloc_end - t_malloc_start).count() / 1000.0;
  auto memcpy_time = std::chrono::duration_cast<std::chrono::microseconds>(t_memcpy_end - t_memcpy_start).count() / 1000.0;
  auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(t_kernel_end - t_kernel_start).count() / 1000.0;
  auto result_time = std::chrono::duration_cast<std::chrono::microseconds>(t_result_end - t_result_start).count() / 1000.0;
  auto destroy_time = std::chrono::duration_cast<std::chrono::microseconds>(t_destroy_end - t_destroy_start).count() / 1000.0;
  auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end_total - t_start_total).count() / 1000.0;
  
  // Print timing information
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\n=== CUDA Volume Computation Profiler ===" << std::endl;
  std::cout << "Mesh 1 faces: " << NF1 << std::endl;
  std::cout << "Mesh 2 faces: " << NF2 << std::endl;
  std::cout << "K-grid nodes: " << Q << std::endl;
  std::cout << "Block size: " << blockSize << std::endl;
  std::cout << "--- Timing (ms) ---" << std::endl;
  std::cout << "  Preparation: " << std::setw(8) << prepare_time << " ms" << std::endl;
  std::cout << "  Memory allocation: " << std::setw(8) << malloc_time << " ms" << std::endl;
  std::cout << "  Memory copy (H->D): " << std::setw(8) << memcpy_time << " ms" << std::endl;
  std::cout << "  Kernel execution:   " << std::setw(8) << kernel_time << " ms" << std::endl;
  std::cout << "  Memory copy (D->H): " << std::setw(8) << result_time << " ms" << std::endl;
  std::cout << "  Memory release: " << std::setw(8) << destroy_time << " ms" << std::endl;
  std::cout << "  Total time:         " << std::setw(8) << total_time << " ms" << std::endl;
  std::cout << "==========================================\n" << std::endl;

  // V = I/(8π^3)
  return I / (8.0 * CUDART_PI * CUDART_PI * CUDART_PI);
}

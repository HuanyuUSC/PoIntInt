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

// ===================== 3x3 Matrix Helper Functions =====================

// Simple 3x3 matrix structure (row-major storage)
struct Mat3x3 {
  float3 row0, row1, row2;
  
  __device__ __forceinline__ Mat3x3() {}
  
  __device__ __forceinline__ Mat3x3(float3 r0, float3 r1, float3 r2) 
    : row0(r0), row1(r1), row2(r2) {}
  
  // Access element at (row, col)
  __device__ __forceinline__ float& operator()(int row, int col) {
    if (row == 0) {
      if (col == 0) return row0.x; else if (col == 1) return row0.y; else return row0.z;
    } else if (row == 1) {
      if (col == 0) return row1.x; else if (col == 1) return row1.y; else return row1.z;
    } else {
      if (col == 0) return row2.x; else if (col == 1) return row2.y; else return row2.z;
    }
  }
  
  __device__ __forceinline__ float operator()(int row, int col) const {
    if (row == 0) {
      if (col == 0) return row0.x; else if (col == 1) return row0.y; else return row0.z;
    } else if (row == 1) {
      if (col == 0) return row1.x; else if (col == 1) return row1.y; else return row1.z;
    } else {
      if (col == 0) return row2.x; else if (col == 1) return row2.y; else return row2.z;
    }
  }
};

// Matrix-vector multiplication: result = M * v
__device__ __forceinline__ float3 mat3x3_mul_vec(const Mat3x3& M, const float3& v) {
  return make_float3(
    M.row0.x*v.x + M.row0.y*v.y + M.row0.z*v.z,
    M.row1.x*v.x + M.row1.y*v.y + M.row1.z*v.z,
    M.row2.x*v.x + M.row2.y*v.y + M.row2.z*v.z
  );
}

// Matrix-matrix multiplication: result = A * B
__device__ __forceinline__ Mat3x3 mat3x3_mul_mat(const Mat3x3& A, const Mat3x3& B) {
  // Compute each row of result
  float3 r0 = make_float3(
    A.row0.x*B.row0.x + A.row0.y*B.row1.x + A.row0.z*B.row2.x,
    A.row0.x*B.row0.y + A.row0.y*B.row1.y + A.row0.z*B.row2.y,
    A.row0.x*B.row0.z + A.row0.y*B.row1.z + A.row0.z*B.row2.z
  );
  float3 r1 = make_float3(
    A.row1.x*B.row0.x + A.row1.y*B.row1.x + A.row1.z*B.row2.x,
    A.row1.x*B.row0.y + A.row1.y*B.row1.y + A.row1.z*B.row2.y,
    A.row1.x*B.row0.z + A.row1.y*B.row1.z + A.row1.z*B.row2.z
  );
  float3 r2 = make_float3(
    A.row2.x*B.row0.x + A.row2.y*B.row1.x + A.row2.z*B.row2.x,
    A.row2.x*B.row0.y + A.row2.y*B.row1.y + A.row2.z*B.row2.y,
    A.row2.x*B.row0.z + A.row2.y*B.row1.z + A.row2.z*B.row2.z
  );
  return Mat3x3(r0, r1, r2);
}

// Cross product matrix [v]_×
__device__ __forceinline__ Mat3x3 cross_product_matrix(const float3& v) {
  return Mat3x3(
    make_float3(0.0f, -v.z, v.y),
    make_float3(v.z, 0.0f, -v.x),
    make_float3(-v.y, v.x, 0.0f)
  );
}

// Reuse device utilities from compute_intersection_volume.cu
__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
  return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

__device__ __forceinline__ float2 cexp_i(float phase) {
  float s, c;
  __sincosf(phase, &s, &c);
  return make_float2(c, s);
}

__device__ __forceinline__ float2 cconj(float2 a) {
  return make_float2(a.x, -a.y);
}

__device__ __forceinline__ float2 cadd(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float2 cscale(float2 a, float s) {
  return make_float2(a.x * s, a.y * s);
}

// Double-precision complex operations (to match CPU)
__device__ __forceinline__ double2 cmul_d(double2 a, double2 b) {
  return make_double2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

__device__ __forceinline__ double2 cadd_d(double2 a, double2 b) {
  return make_double2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ double2 cscale_d(double2 a, double s) {
  return make_double2(a.x * s, a.y * s);
}

__device__ __forceinline__ float2 double2_to_float2(double2 d) {
  return make_float2((float)d.x, (float)d.y);
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

__device__ __forceinline__ double2 E_double_prime_d(double z) {
  double z2 = z*z;
  if (fabs(z) < 1e-4) {
    // E''(z) approximation for small z
    return make_double2(-1.0/3.0 + z2/10.0, -z/4.0 + z2*z/36.0);
  } else {
    double s = sin(z);
    double c = cos(z);
    double z3 = z2*z;
    double re = (2.0*s - 2.0*z*c - z2*s)/z3;
    double im = (2.0*(1.0 - c) - 2.0*z*s + z2*c)/z3;
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

// Compute gradient of Phi_ab w.r.t. alpha and beta (double precision to match CPU)
// Returns: (dPhi/dalpha, dPhi/dbeta) as double2 pairs
__device__ __forceinline__ void Phi_ab_gradient_d(double alpha, double beta, double2* dPhi_dalpha, double2* dPhi_dbeta) {
  double d = beta - alpha;
  double threshold = 1e-3;  // Match CPU: 1e-3
  
  if (fabs(d) < threshold) {
    // When alpha ≈ beta, use E''(alpha)
    // Phi ≈ -2i * E'((alpha+beta)/2)
    // dPhi/dalpha = dPhi/dbeta = -i * E''((alpha+beta)/2)
    double z = 0.5 * (alpha + beta);
    double2 Epp = E_double_prime_d(z);
    // -i * (re, im) = (im, -re)
    *dPhi_dalpha = make_double2(Epp.y, -Epp.x);
    *dPhi_dbeta = make_double2(Epp.y, -Epp.x);
  } else {
    // General case: Phi = -2i * (Eb - Ea) / d
    // dPhi/dalpha = -2i * [(Eb - Ea)/d^2 - Ea'/d]
    // dPhi/dbeta = 2i * [(Eb - Ea)/d^2 - Eb'/d]
    // Match CPU implementation exactly
    double2 Ea = E_func_d(alpha);
    double2 Eb = E_func_d(beta);
    double2 Ea_prime = E_prime_d(alpha);
    double2 Eb_prime = E_prime_d(beta);
    
    double invd = 1.0 / d;
    double invd2 = invd * invd;
    
    // term1 = (Eb - Ea) / d^2
    double2 Eb_minus_Ea = make_double2(Eb.x - Ea.x, Eb.y - Ea.y);
    double2 term1 = make_double2(Eb_minus_Ea.x * invd2, Eb_minus_Ea.y * invd2);
    
    // Ea'/d and Eb'/d
    double2 Ea_prime_over_d = make_double2(Ea_prime.x * invd, Ea_prime.y * invd);
    double2 Eb_prime_over_d = make_double2(Eb_prime.x * invd, Eb_prime.y * invd);
    
    // dPhi/dalpha = -2i * (term1 - Ea_prime_over_d)
    // -2i * (re, im) = (2*im, -2*re)
    double2 diff_alpha = make_double2(term1.x - Ea_prime_over_d.x, term1.y - Ea_prime_over_d.y);
    *dPhi_dalpha = make_double2(2.0 * diff_alpha.y, -2.0 * diff_alpha.x);
    
    // dPhi/dbeta = 2i * (term1 - Eb_prime_over_d)
    // 2i * (re, im) = (-2*im, 2*re)
    double2 diff_beta = make_double2(term1.x - Eb_prime_over_d.x, term1.y - Eb_prime_over_d.y);
    *dPhi_dbeta = make_double2(-2.0 * diff_beta.y, 2.0 * diff_beta.x);
  }
}

// J1_over_x for disks
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

// d/dx (J1(x)/x) for disks
__device__ __forceinline__ float J1_over_x_prime(float x) {
  float ax = fabsf(x);
  if (ax < 1e-3f) {
    // Series: d/dx (J1(x)/x) ≈ -x/4 + x^3/64 - ...
    float x2 = x * x;
    return -0.25f * x + (1.0f / 64.0f) * x * x2;
  }
  // For larger x, use: d/dx (J1(x)/x) = (x*J0(x) - 2*J1(x)) / x^2
  // Approximate J0(x) using series or asymptotic
  // Simplified: use numerical derivative approximation
  float eps = 1e-4f;
  float J1_x_plus = J1_over_x(x + eps);
  float J1_x_minus = J1_over_x(x - eps);
  return (J1_x_plus - J1_x_minus) / (2.0f * eps);
}

using PoIntInt::TriPacked;
using PoIntInt::DiskPacked;
using PoIntInt::GaussianPacked;

// ===================== Phase 0: Compute A(k) with affine transformation =====================

// Kernel to compute A(k) for a geometry with affine transformation applied
// For each k-point q, computes A(k_q) for the transformed geometry
__global__ void compute_A_affine_triangle_kernel(
  const TriPacked* __restrict__ tris,
  int num_tris,
  const float3* __restrict__ kdirs,
  const float* __restrict__ kmags,
  const float* __restrict__ dof_params,  // 12 DoF parameters [t(3), A(9)]
  int Q,
  float2* A_out  // Output: Q complex values (row-major)
) {
  int q = blockIdx.x;
  if (q >= Q) return;
  
  float3 khat = kdirs[q];
  float k = kmags[q];
  float kx = k * khat.x, ky = k * khat.y, kz = k * khat.z;
  
  if (k < 1e-10f) {
    // At k=0, A(k) = 0
    if (threadIdx.x == 0) {
      A_out[q] = make_float2(0.0f, 0.0f);
    }
    return;
  }
  
  // Extract DoF parameters
  float3 t = make_float3(dof_params[0], dof_params[1], dof_params[2]);
  Mat3x3 A(
    make_float3(dof_params[3], dof_params[4], dof_params[5]),
    make_float3(dof_params[6], dof_params[7], dof_params[8]),
    make_float3(dof_params[9], dof_params[10], dof_params[11])
  );
  
  // Initialize accumulator for this thread
  float2 A_local = make_float2(0.0f, 0.0f);
  
  // Process each triangle
  for (int i = threadIdx.x; i < num_tris; i += blockDim.x) {
    TriPacked tri = tris[i];
    
    // Transform using matrix operations
    float3 a_prime = mat3x3_mul_vec(A, tri.a);
    a_prime.x += t.x; a_prime.y += t.y; a_prime.z += t.z;
    
    float3 e1_prime = mat3x3_mul_vec(A, tri.e1);
    float3 e2_prime = mat3x3_mul_vec(A, tri.e2);
    
    // S' = 0.5 * (e1' × e2')
    float3 S_prime = make_float3(
      0.5f * (e1_prime.y*e2_prime.z - e1_prime.z*e2_prime.y),
      0.5f * (e1_prime.z*e2_prime.x - e1_prime.x*e2_prime.z),
      0.5f * (e1_prime.x*e2_prime.y - e1_prime.y*e2_prime.x)
    );
    
    // Compute transformed quantities
    float alpha_prime_f = kx*e1_prime.x + ky*e1_prime.y + kz*e1_prime.z;
    float beta_prime_f = kx*e2_prime.x + ky*e2_prime.y + kz*e2_prime.z;
    float gamma_prime = khat.x*S_prime.x + khat.y*S_prime.y + khat.z*S_prime.z;
    
    // Use double precision for element functions to match CPU
    double alpha_prime = (double)alpha_prime_f;
    double beta_prime = (double)beta_prime_f;
    double2 phi_prime_d = Phi_ab_d(alpha_prime, beta_prime);
    float2 phi_prime = double2_to_float2(phi_prime_d);
    
    float phase = kx*a_prime.x + ky*a_prime.y + kz*a_prime.z;
    float2 phase_complex = cexp_i(phase);
    float2 A_tri = cmul(phase_complex, phi_prime);
    A_tri = cscale(A_tri, gamma_prime);
    
    A_local.x += A_tri.x;
    A_local.y += A_tri.y;
  }
  
  // Tree reduction using shared memory
  __shared__ float s_A_x[256];
  __shared__ float s_A_y[256];
  
  int tid = threadIdx.x;
  if (tid < blockDim.x) {
    s_A_x[tid] = A_local.x;
    s_A_y[tid] = A_local.y;
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
  
  // Write result
  if (tid == 0) {
    A_out[q] = make_float2(s_A_x[0], s_A_y[0]);
  }
}

// ===================== Phase 1: Compute ∂A(k)/∂θ for all k-points =====================

// Kernel to compute gradient of A(k) w.r.t. AffineDoF for triangles
// For each k-point q, computes ∂A(k_q)/∂θ for all 12 DoFs
__global__ void compute_A_gradient_affine_triangle_kernel(
  const TriPacked* __restrict__ tris,
  int num_tris,
  const float3* __restrict__ kdirs,
  const float* __restrict__ kmags,
  const float* __restrict__ dof_params,  // 12 DoF parameters [t(3), A(9)]
  int Q,
  float2* grad_A  // Output: Q × 12 complex gradients (row-major)
) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= Q) return;
  
  float3 khat = kdirs[q];
  float k = kmags[q];
  float kx = k * khat.x, ky = k * khat.y, kz = k * khat.z;
  
  if (k < 1e-10f) {
    // At k=0, gradient is zero
    for (int dof = 0; dof < 12; ++dof) {
      grad_A[q * 12 + dof] = make_float2(0.0f, 0.0f);
    }
    return;
  }
  
  // Extract DoF parameters
  float3 t = make_float3(dof_params[0], dof_params[1], dof_params[2]);
  float3 A_row0 = make_float3(dof_params[3], dof_params[4], dof_params[5]);
  float3 A_row1 = make_float3(dof_params[6], dof_params[7], dof_params[8]);
  float3 A_row2 = make_float3(dof_params[9], dof_params[10], dof_params[11]);
  Mat3x3 A(A_row0, A_row1, A_row2);
  
  // Initialize gradient accumulator
  float2 grad[12];
  for (int i = 0; i < 12; ++i) {
    grad[i] = make_float2(0.0f, 0.0f);
  }
  
  // Process each triangle
  for (int i = 0; i < num_tris; ++i) {
    TriPacked tri = tris[i];
    
    // Transform using matrix operations
    float3 a_prime = mat3x3_mul_vec(A, tri.a);
    a_prime.x += t.x; a_prime.y += t.y; a_prime.z += t.z;
    
    float3 e1_prime = mat3x3_mul_vec(A, tri.e1);
    float3 e2_prime = mat3x3_mul_vec(A, tri.e2);
    
    // S' = 0.5 * (e1' × e2')
    float3 S_prime = make_float3(
      0.5f * (e1_prime.y*e2_prime.z - e1_prime.z*e2_prime.y),
      0.5f * (e1_prime.z*e2_prime.x - e1_prime.x*e2_prime.z),
      0.5f * (e1_prime.x*e2_prime.y - e1_prime.y*e2_prime.x)
    );
    
    // Compute transformed quantities
    float alpha_prime_f = kx*e1_prime.x + ky*e1_prime.y + kz*e1_prime.z;
    float beta_prime_f = kx*e2_prime.x + ky*e2_prime.y + kz*e2_prime.z;
    float gamma_prime = khat.x*S_prime.x + khat.y*S_prime.y + khat.z*S_prime.z;
    
    // Use double precision for element functions to match CPU
    double alpha_prime = (double)alpha_prime_f;
    double beta_prime = (double)beta_prime_f;
    double2 phi_prime_d = Phi_ab_d(alpha_prime, beta_prime);
    float2 phi_prime = double2_to_float2(phi_prime_d);
    
    float phase = kx*a_prime.x + ky*a_prime.y + kz*a_prime.z;
    float2 phase_complex = cexp_i(phase);
    float2 A_tri = cmul(phase_complex, phi_prime);
    A_tri = cscale(A_tri, gamma_prime);
    
    // Compute gradients of Phi w.r.t. alpha and beta (double precision)
    double2 dPhi_dalpha_d, dPhi_dbeta_d;
    Phi_ab_gradient_d(alpha_prime, beta_prime, &dPhi_dalpha_d, &dPhi_dbeta_d);
    float2 dPhi_dalpha = double2_to_float2(dPhi_dalpha_d);
    float2 dPhi_dbeta = double2_to_float2(dPhi_dbeta_d);
    
    // Gradient w.r.t. translation t (DoFs 0-2)
    // dA/dt = i*k * A (since phase = exp(i*k·(A*a+t)))
    float2 ik = make_float2(0.0f, kx);
    grad[0] = cadd(grad[0], cmul(ik, A_tri));
    
    ik = make_float2(0.0f, ky);
    grad[1] = cadd(grad[1], cmul(ik, A_tri));
    
    ik = make_float2(0.0f, kz);
    grad[2] = cadd(grad[2], cmul(ik, A_tri));
    
    // Gradient w.r.t. matrix A (DoFs 3-11)
    // Following CPU implementation: gA = A_tri * i*k * a^T + gamma * phase * k * (dPhi_dalpha * e1 + dPhi_dbeta * e2)^T
    //                                - [k]_× * A * [S]_× * phase * phi
    // The gradient w.r.t. A_ij is gA.transpose()(i,j) = gA(j,i)
    // So for grad[3 + row*3 + col] (A_ij where i=row, j=col), we need gA(col, row)
    
    // Compute cross product matrix [khat]_× * A * [S]_× using matrix operations
    Mat3x3 S_cross = cross_product_matrix(tri.S);
    Mat3x3 A_S_cross = mat3x3_mul_mat(A, S_cross);
    Mat3x3 khat_cross = cross_product_matrix(khat);
    Mat3x3 khat_A_S = mat3x3_mul_mat(khat_cross, A_S_cross);
    
    // For each matrix element A_ij (row i, col j)
    // CPU computes: gA = A_tri * i*k * a^T + gamma * phase * k * (dPhi_dalpha * e1 + dPhi_dbeta * e2)^T
    //                - [k]_× * A * [S]_× * phase * phi
    // Then: grad[3 + i*3 + j] = gA.transpose()(i,j) = gA(j,i)
    // So for grad[3 + row*3 + col] (A_ij where i=row, j=col), we need gA(col, row)
    
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        int dof_idx = 3 + row * 3 + col;  // A_ij where i=row, j=col, gradient stored here
        
        // We need to compute gA(row, col) because Map(col, row) += gA.transpose()(col, row) = Map(col, row) += gA(row, col)
        // CPU: gA = A_tri * I * k * a.transpose() → gA(row, col) = A_tri * I * k_row * a_col
        // So we need: gA(row, col) = A_tri * i*k_row * a_col + gamma * phase * k_row * (dPhi_dalpha * e1_col + dPhi_dbeta * e2_col)
        //                - ([k]_× * A * [S]_×)(row, col) * phase * phi
        
        float k_row = (row == 0) ? kx : ((row == 1) ? ky : kz);
        float a_col = (col == 0) ? tri.a.x : ((col == 1) ? tri.a.y : tri.a.z);
        float e1_col = (col == 0) ? tri.e1.x : ((col == 1) ? tri.e1.y : tri.e1.z);
        float e2_col = (col == 0) ? tri.e2.x : ((col == 1) ? tri.e2.y : tri.e2.z);
        
        // Contribution 1: from phase: i*k_row * a_col * A_tri
        float2 ik_contrib = make_float2(0.0f, k_row * a_col);
        float2 phase_contrib = cmul(ik_contrib, A_tri);
        grad[dof_idx] = cadd(grad[dof_idx], phase_contrib);
        
        // Contribution 2: from Phi via alpha and beta
        // Need: k_row * (dPhi_dalpha * e1_col + dPhi_dbeta * e2_col)
        // But alpha = k·(A*e1), so ∂alpha/∂A_row,col = k_row * e1_col
        // Similarly, ∂beta/∂A_row,col = k_row * e2_col
        float dalpha_dA = k_row * e1_col;
        float dbeta_dA = k_row * e2_col;
        float2 dPhi_contrib = cadd(
          cscale(dPhi_dalpha, dalpha_dA),
          cscale(dPhi_dbeta, dbeta_dA)
        );
        float2 phi_contrib = cmul(phase_complex, dPhi_contrib);
        phi_contrib = cscale(phi_contrib, gamma_prime);
        grad[dof_idx] = cadd(grad[dof_idx], phi_contrib);
        
        // Contribution 3: from cross product term
        // CPU line 298: gA -= cross_product_matrix(khat) * A * cross_product_matrix(S) * phase * phi
        // So: gA(row, col) -= ([khat]_× * A * [S]_×)(row, col) * phase * phi
        // CPU line 299: Map(local_grad.data() + 3) += gA.transpose()
        // Eigen::Map interprets local_grad[3:12] as column-major: Map(i,j) = local_grad[3 + i + j*3]
        // We store in row-major: local_grad[3 + row*3 + col] = A(row, col)
        // gA.transpose()(row, col) = gA(col, row) gets written to Map(row, col) = local_grad[3 + row + col*3]
        // To write to local_grad[3 + row*3 + col], we need Map(col, row), which gets gA.transpose()(col, row) = gA(row, col)
        // Therefore: grad[3 + row*3 + col] = gA(row, col) = -([khat]_× * A * [S]_×)(row, col) * phase * phi
        float khat_A_S_val = khat_A_S(row, col);  // Use (row, col) instead of (col, row)
        float2 cross_contrib = cmul(phase_complex, phi_prime);
        cross_contrib = cscale(cross_contrib, -khat_A_S_val);  // Negative sign to match CPU: gA -= ...
        grad[dof_idx] = cadd(grad[dof_idx], cross_contrib);
      }
    }
  }
  
  // Write results
  for (int dof = 0; dof < 12; ++dof) {
    grad_A[q * 12 + dof] = grad[dof];
  }
}

// Similar kernels for disks and Gaussian splats would follow the same pattern
// For now, we'll implement a simplified version that handles all geometry types
// in a unified kernel, or we can add them later

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
  const float2* __restrict__ grad_A1,  // Q × num_dofs1: ∂A₁(k_q)/∂θ₁
  const float2* __restrict__ grad_A2,  // Q × num_dofs2: ∂A₂(k_q)/∂θ₂
  const float2* __restrict__ A1,       // Q: A₁(k_q)
  const float2* __restrict__ A2,       // Q: A₂(k_q)
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
  float2 a1 = A1[q];
  float2 a2 = A2[q];
  
  // Compute volume contribution: Re(A1 * conj(A2)) = Re(A1) * Re(A2) + Im(A1) * Im(A2)
  double vol_contrib = (double)a1.x * (double)a2.x + (double)a1.y * (double)a2.y;
  atomicAdd(volume, w * vol_contrib);
  
  // Compute gradient contributions for geometry 1
  // For each DoF, accumulate: Re(dA1 * conj(A2)) * weight
  for (int dof = 0; dof < num_dofs1; ++dof) {
    float2 dA1 = grad_A1[q * num_dofs1 + dof];
    // Re(dA1 * conj(A2)) = Re(dA1) * Re(A2) + Im(dA1) * Im(A2)
    double grad_contrib = (double)dA1.x * (double)a2.x + (double)dA1.y * (double)a2.y;
    atomicAdd(&grad_V1[dof], w * grad_contrib);
  }
  
  // Compute gradient contributions for geometry 2
  // For each DoF, accumulate: Re(A1 * conj(dA2)) * weight
  for (int dof = 0; dof < num_dofs2; ++dof) {
    float2 dA2 = grad_A2[q * num_dofs2 + dof];
    // Re(A1 * conj(dA2)) = Re(A1) * Re(dA2) + Im(A1) * Im(dA2)
    double grad_contrib = (double)a1.x * (double)dA2.x + (double)a1.y * (double)dA2.y;
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
  float3* d_kdirs = nullptr;
  float* d_kmags = nullptr;
  double* d_weights = nullptr;
  float* d_dofs1 = nullptr;
  float* d_dofs2 = nullptr;
  float2* d_A1 = nullptr;
  float2* d_A2 = nullptr;
  float2* d_grad_A1 = nullptr;
  float2* d_grad_A2 = nullptr;
  double* d_grad_V1 = nullptr;
  double* d_grad_V2 = nullptr;
  double* d_volume = nullptr;
  
  // Host vectors
  std::vector<float3> h_kdirs;
  std::vector<float> h_dofs1, h_dofs2;
  std::vector<float2> h_A1, h_A2;
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
    h_kdirs[q] = make_float3(kgrid.dirs[q][0], kgrid.dirs[q][1], kgrid.dirs[q][2]);
  }
  CUDA_CHECK(cudaMalloc(&d_kdirs, Q * sizeof(float3)));
  CUDA_CHECK(cudaMalloc(&d_kmags, Q * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weights, Q * sizeof(double)));
  
  // Allocate DoF parameters
  CUDA_CHECK(cudaMalloc(&d_dofs1, num_dofs1 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dofs2, num_dofs2 * sizeof(float)));
  
  // Allocate form factors and gradients
  CUDA_CHECK(cudaMalloc(&d_A1, Q * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&d_A2, Q * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&d_grad_A1, Q * num_dofs1 * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&d_grad_A2, Q * num_dofs2 * sizeof(float2)));
  
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
  
  CUDA_CHECK(cudaMemcpy(d_kdirs, h_kdirs.data(), Q * sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kmags, kgrid.kmag.data(), Q * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weights, kgrid.w.data(), Q * sizeof(double), cudaMemcpyHostToDevice));
  
  // Convert DoFs to float and copy
  h_dofs1.resize(num_dofs1);
  h_dofs2.resize(num_dofs2);
  for (int i = 0; i < num_dofs1; ++i) {
    h_dofs1[i] = (float)dofs1(i);
  }
  for (int i = 0; i < num_dofs2; ++i) {
    h_dofs2[i] = (float)dofs2(i);
  }
  CUDA_CHECK(cudaMemcpy(d_dofs1, h_dofs1.data(), num_dofs1 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dofs2, h_dofs2.data(), num_dofs2 * sizeof(float), cudaMemcpyHostToDevice));
  
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
  float3 kdir = make_float3((float)khat.x(), (float)khat.y(), (float)khat.z());
  float kmag_f = (float)kmag;
  
  // Declare all variables at the top (before any goto cleanup)
  TriPacked* d_tris = nullptr;
  float* d_dofs = nullptr;
  float2* d_A = nullptr;
  float3* d_kdirs = nullptr;
  float* d_kmags = nullptr;
  std::vector<float> h_dofs(12);
  dim3 grid(1);
  dim3 block(blockSize);
  float2 h_A = make_float2(0.0f, 0.0f);
  
  #define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
      goto cleanup; \
    } \
  } while(0)
  
  CUDA_CHECK(cudaMalloc(&d_tris, NF * sizeof(TriPacked)));
  CUDA_CHECK(cudaMalloc(&d_dofs, 12 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_A, sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&d_kdirs, sizeof(float3)));
  CUDA_CHECK(cudaMalloc(&d_kmags, sizeof(float)));
  
  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_tris, geom.tris.data(), NF * sizeof(TriPacked), cudaMemcpyHostToDevice));
  
  for (int i = 0; i < 12; ++i) {
    h_dofs[i] = (float)dofs(i);
  }
  CUDA_CHECK(cudaMemcpy(d_dofs, h_dofs.data(), 12 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kdirs, &kdir, sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kmags, &kmag_f, sizeof(float), cudaMemcpyHostToDevice));
  
  // Launch kernel (single k-point, so Q=1)
  compute_A_affine_triangle_kernel<<<grid, block>>>(
    d_tris, NF, d_kdirs, d_kmags, d_dofs, 1, d_A
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Copy result back
  CUDA_CHECK(cudaMemcpy(&h_A, d_A, sizeof(float2), cudaMemcpyDeviceToHost));
  
  cleanup:
  if (d_tris) cudaFree(d_tris);
  if (d_dofs) cudaFree(d_dofs);
  if (d_A) cudaFree(d_A);
  if (d_kdirs) cudaFree(d_kdirs);
  if (d_kmags) cudaFree(d_kmags);
  
  #undef CUDA_CHECK
  
  return std::complex<double>((double)h_A.x, (double)h_A.y);
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
  float3 kdir = make_float3((float)khat.x(), (float)khat.y(), (float)khat.z());
  float kmag_f = (float)kmag;
  
  // Declare all variables at the top (before any goto cleanup)
  TriPacked* d_tris = nullptr;
  float* d_dofs = nullptr;
  float2* d_grad_A = nullptr;
  float3* d_kdirs = nullptr;
  float* d_kmags = nullptr;
  std::vector<float> h_dofs(12);
  dim3 grid(1);
  dim3 block(blockSize);
  std::vector<float2> h_grad_A(12);
  Eigen::VectorXcd grad(12);
  
  #define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
      goto cleanup; \
    } \
  } while(0)
  
  CUDA_CHECK(cudaMalloc(&d_tris, NF * sizeof(TriPacked)));
  CUDA_CHECK(cudaMalloc(&d_dofs, 12 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_grad_A, 12 * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&d_kdirs, sizeof(float3)));
  CUDA_CHECK(cudaMalloc(&d_kmags, sizeof(float)));
  
  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_tris, geom.tris.data(), NF * sizeof(TriPacked), cudaMemcpyHostToDevice));
  
  for (int i = 0; i < 12; ++i) {
    h_dofs[i] = (float)dofs(i);
  }
  CUDA_CHECK(cudaMemcpy(d_dofs, h_dofs.data(), 12 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kdirs, &kdir, sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kmags, &kmag_f, sizeof(float), cudaMemcpyHostToDevice));
  
  // Launch kernel (single k-point, so Q=1)
  compute_A_gradient_affine_triangle_kernel<<<grid, block>>>(
    d_tris, NF, d_kdirs, d_kmags, d_dofs, 1, d_grad_A
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Copy result back
  CUDA_CHECK(cudaMemcpy(h_grad_A.data(), d_grad_A, 12 * sizeof(float2), cudaMemcpyDeviceToHost));
  
  cleanup:
  if (d_tris) cudaFree(d_tris);
  if (d_dofs) cudaFree(d_dofs);
  if (d_grad_A) cudaFree(d_grad_A);
  if (d_kdirs) cudaFree(d_kdirs);
  if (d_kmags) cudaFree(d_kmags);
  
  #undef CUDA_CHECK
  
  // Convert result to Eigen::VectorXcd
  for (int i = 0; i < 12; ++i) {
    grad(i) = std::complex<double>((double)h_grad_A[i].x, (double)h_grad_A[i].y);
  }
  
  return grad;
}

} // namespace PoIntInt


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <math_constants.h>
#include "geometry/types.hpp"

// Define this macro before including the header to prevent duplicate declarations
#define AFFINE_DOF_CUDA_KERNELS_DEFINED

#include "dof/cuda/affine_dof_cuda.hpp"
#include "dof/cuda/dof_cuda_interface.hpp"
#include "geometry/packing.hpp"
#include <iostream>
#include <vector>

using PoIntInt::TriPacked;

// ===================== 3x3 Matrix Helper Functions =====================

// Simple 3x3 matrix structure (row-major storage)
struct Mat3x3 {
  double3 row0, row1, row2;
  
  __device__ __forceinline__ Mat3x3() {}
  
  __device__ __forceinline__ Mat3x3(double3 r0, double3 r1, double3 r2) 
    : row0(r0), row1(r1), row2(r2) {}
  
  // Access element at (row, col)
  __device__ __forceinline__ double& operator()(int row, int col) {
    if (row == 0) {
      if (col == 0) return row0.x; else if (col == 1) return row0.y; else return row0.z;
    } else if (row == 1) {
      if (col == 0) return row1.x; else if (col == 1) return row1.y; else return row1.z;
    } else {
      if (col == 0) return row2.x; else if (col == 1) return row2.y; else return row2.z;
    }
  }
  
  __device__ __forceinline__ double operator()(int row, int col) const {
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
__device__ __forceinline__ double3 mat3x3_mul_vec(const Mat3x3& M, const double3& v) {
  return make_double3(
    M.row0.x*v.x + M.row0.y*v.y + M.row0.z*v.z,
    M.row1.x*v.x + M.row1.y*v.y + M.row1.z*v.z,
    M.row2.x*v.x + M.row2.y*v.y + M.row2.z*v.z
  );
}

// Matrix-matrix multiplication: result = A * B
__device__ __forceinline__ Mat3x3 mat3x3_mul_mat(const Mat3x3& A, const Mat3x3& B) {
  // Compute each row of result
  double3 r0 = make_double3(
    A.row0.x*B.row0.x + A.row0.y*B.row1.x + A.row0.z*B.row2.x,
    A.row0.x*B.row0.y + A.row0.y*B.row1.y + A.row0.z*B.row2.y,
    A.row0.x*B.row0.z + A.row0.y*B.row1.z + A.row0.z*B.row2.z
  );
  double3 r1 = make_double3(
    A.row1.x*B.row0.x + A.row1.y*B.row1.x + A.row1.z*B.row2.x,
    A.row1.x*B.row0.y + A.row1.y*B.row1.y + A.row1.z*B.row2.y,
    A.row1.x*B.row0.z + A.row1.y*B.row1.z + A.row1.z*B.row2.z
  );
  double3 r2 = make_double3(
    A.row2.x*B.row0.x + A.row2.y*B.row1.x + A.row2.z*B.row2.x,
    A.row2.x*B.row0.y + A.row2.y*B.row1.y + A.row2.z*B.row2.y,
    A.row2.x*B.row0.z + A.row2.y*B.row1.z + A.row2.z*B.row2.z
  );
  return Mat3x3(r0, r1, r2);
}

// Cross product matrix [v]_×
__device__ __forceinline__ Mat3x3 cross_product_matrix(const double3& v) {
  return Mat3x3(
    make_double3(0.0, -v.z, v.y),
    make_double3(v.z, 0.0, -v.x),
    make_double3(-v.y, v.x, 0.0)
  );
}

// ===================== Complex Number Utilities =====================

__device__ __forceinline__ double2 cmul(double2 a, double2 b) {
  return make_double2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

__device__ __forceinline__ double2 cexp_i(double phase) {
  double s, c;
  sincos(phase, &s, &c);
  return make_double2(c, s);
}

__device__ __forceinline__ double2 cadd(double2 a, double2 b) {
  return make_double2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ double2 cscale(double2 a, double s) {
  return make_double2(a.x * s, a.y * s);
}

// ===================== Element Functions =====================

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

// ===================== CUDA Kernels =====================

// Kernel to compute A(k) for a geometry with affine transformation applied
// For each k-point q, computes A(k_q) for the transformed geometry
extern "C" __global__ void compute_A_affine_triangle_kernel(
  const TriPacked* __restrict__ tris,
  int num_tris,
  const double3* __restrict__ kdirs,
  const double* __restrict__ kmags,
  const double* __restrict__ dof_params,  // 12 DoF parameters [t(3), A(9)]
  int Q,
  double2* A_out  // Output: Q complex values (row-major)
) {
  int q = blockIdx.x;
  if (q >= Q) return;
  
  double3 khat = kdirs[q];
  double k = kmags[q];
  double kx = k * khat.x, ky = k * khat.y, kz = k * khat.z;
  
  if (k < 1e-10) {
    // At k=0, A(k) = 0
    if (threadIdx.x == 0) {
      A_out[q] = make_double2(0.0, 0.0);
    }
    return;
  }
  
  // Extract DoF parameters
  double3 t = make_double3(dof_params[0], dof_params[1], dof_params[2]);
  Mat3x3 A(
    make_double3(dof_params[3], dof_params[4], dof_params[5]),
    make_double3(dof_params[6], dof_params[7], dof_params[8]),
    make_double3(dof_params[9], dof_params[10], dof_params[11])
  );
  
  // Initialize accumulator for this thread
  double2 A_local = make_double2(0.0, 0.0);
  
  // Process each triangle
  for (int i = threadIdx.x; i < num_tris; i += blockDim.x) {
    TriPacked tri = tris[i];
    
    // Transform using matrix operations
    double3 a_prime = mat3x3_mul_vec(A, tri.a);
    a_prime.x += t.x; a_prime.y += t.y; a_prime.z += t.z;
    
    double3 e1_prime = mat3x3_mul_vec(A, tri.e1);
    double3 e2_prime = mat3x3_mul_vec(A, tri.e2);
    
    // S' = 0.5 * (e1' × e2')
    double3 S_prime = make_double3(
      0.5 * (e1_prime.y*e2_prime.z - e1_prime.z*e2_prime.y),
      0.5 * (e1_prime.z*e2_prime.x - e1_prime.x*e2_prime.z),
      0.5 * (e1_prime.x*e2_prime.y - e1_prime.y*e2_prime.x)
    );
    
    // Compute transformed quantities
    double alpha_prime = kx*e1_prime.x + ky*e1_prime.y + kz*e1_prime.z;
    double beta_prime = kx*e2_prime.x + ky*e2_prime.y + kz*e2_prime.z;
    double gamma_prime = khat.x*S_prime.x + khat.y*S_prime.y + khat.z*S_prime.z;
    
    double2 phi_prime = Phi_ab_d(alpha_prime, beta_prime);
    double phase = kx*a_prime.x + ky*a_prime.y + kz*a_prime.z;
    double2 phase_complex = cexp_i(phase);
    double2 A_tri = cmul(phase_complex, phi_prime);
    A_tri = cscale(A_tri, gamma_prime);
    
    A_local.x += A_tri.x;
    A_local.y += A_tri.y;
  }
  
  // Tree reduction using shared memory
  __shared__ double s_A_x[256];
  __shared__ double s_A_y[256];
  
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
    A_out[q] = make_double2(s_A_x[0], s_A_y[0]);
  }
}

// Kernel to compute gradient of A(k) w.r.t. AffineDoF for triangles
// For each k-point q, computes ∂A(k_q)/∂θ for all 12 DoFs
extern "C" __global__ void compute_A_gradient_affine_triangle_kernel(
  const TriPacked* __restrict__ tris,
  int num_tris,
  const double3* __restrict__ kdirs,
  const double* __restrict__ kmags,
  const double* __restrict__ dof_params,  // 12 DoF parameters [t(3), A(9)]
  int Q,
  double2* grad_A  // Output: Q × 12 complex gradients (row-major)
) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= Q) return;
  
  double3 khat = kdirs[q];
  double k = kmags[q];
  double kx = k * khat.x, ky = k * khat.y, kz = k * khat.z;
  
  if (k < 1e-10) {
    // At k=0, gradient is zero
    for (int dof = 0; dof < 12; ++dof) {
      grad_A[q * 12 + dof] = make_double2(0.0, 0.0);
    }
    return;
  }
  
  // Extract DoF parameters
  double3 t = make_double3(dof_params[0], dof_params[1], dof_params[2]);
  double3 A_row0 = make_double3(dof_params[3], dof_params[4], dof_params[5]);
  double3 A_row1 = make_double3(dof_params[6], dof_params[7], dof_params[8]);
  double3 A_row2 = make_double3(dof_params[9], dof_params[10], dof_params[11]);
  Mat3x3 A(A_row0, A_row1, A_row2);
  
  // Initialize gradient accumulator
  double2 grad[12];
  for (int i = 0; i < 12; ++i) {
    grad[i] = make_double2(0.0, 0.0);
  }
  
  // Process each triangle
  for (int i = 0; i < num_tris; ++i) {
    TriPacked tri = tris[i];
    
    // Transform using matrix operations
    double3 a_prime = mat3x3_mul_vec(A, tri.a);
    a_prime.x += t.x; a_prime.y += t.y; a_prime.z += t.z;
    
    double3 e1_prime = mat3x3_mul_vec(A, tri.e1);
    double3 e2_prime = mat3x3_mul_vec(A, tri.e2);
    
    // S' = 0.5 * (e1' × e2')
    double3 S_prime = make_double3(
      0.5 * (e1_prime.y*e2_prime.z - e1_prime.z*e2_prime.y),
      0.5 * (e1_prime.z*e2_prime.x - e1_prime.x*e2_prime.z),
      0.5 * (e1_prime.x*e2_prime.y - e1_prime.y*e2_prime.x)
    );
    
    // Compute transformed quantities
    double alpha_prime = kx*e1_prime.x + ky*e1_prime.y + kz*e1_prime.z;
    double beta_prime = kx*e2_prime.x + ky*e2_prime.y + kz*e2_prime.z;
    double gamma_prime = khat.x*S_prime.x + khat.y*S_prime.y + khat.z*S_prime.z;
    
    double2 phi_prime = Phi_ab_d(alpha_prime, beta_prime);
    double phase = kx*a_prime.x + ky*a_prime.y + kz*a_prime.z;
    double2 phase_complex = cexp_i(phase);
    double2 A_tri = cmul(phase_complex, phi_prime);
    A_tri = cscale(A_tri, gamma_prime);
    
    // Compute gradients of Phi w.r.t. alpha and beta
    double2 dPhi_dalpha, dPhi_dbeta;
    Phi_ab_gradient_d(alpha_prime, beta_prime, &dPhi_dalpha, &dPhi_dbeta);
    
    // Gradient w.r.t. translation t (DoFs 0-2)
    // dA/dt = i*k * A (since phase = exp(i*k·(A*a+t)))
    double2 ik = make_double2(0.0, kx);
    grad[0] = cadd(grad[0], cmul(ik, A_tri));
    
    ik = make_double2(0.0, ky);
    grad[1] = cadd(grad[1], cmul(ik, A_tri));
    
    ik = make_double2(0.0, kz);
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
        
        double k_row = (row == 0) ? kx : ((row == 1) ? ky : kz);
        double a_col = (col == 0) ? tri.a.x : ((col == 1) ? tri.a.y : tri.a.z);
        double e1_col = (col == 0) ? tri.e1.x : ((col == 1) ? tri.e1.y : tri.e1.z);
        double e2_col = (col == 0) ? tri.e2.x : ((col == 1) ? tri.e2.y : tri.e2.z);
        
        // Contribution 1: from phase: i*k_row * a_col * A_tri
        double2 ik_contrib = make_double2(0.0, k_row * a_col);
        double2 phase_contrib = cmul(ik_contrib, A_tri);
        grad[dof_idx] = cadd(grad[dof_idx], phase_contrib);
        
        // Contribution 2: from Phi via alpha and beta
        // Need: k_row * (dPhi_dalpha * e1_col + dPhi_dbeta * e2_col)
        // But alpha = k·(A*e1), so ∂alpha/∂A_row,col = k_row * e1_col
        // Similarly, ∂beta/∂A_row,col = k_row * e2_col
        double dalpha_dA = k_row * e1_col;
        double dbeta_dA = k_row * e2_col;
        double2 dPhi_contrib = cadd(
          cscale(dPhi_dalpha, dalpha_dA),
          cscale(dPhi_dbeta, dbeta_dA)
        );
        double2 phi_contrib = cmul(phase_complex, dPhi_contrib);
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
        double khat_A_S_val = khat_A_S(row, col);  // Use (row, col) instead of (col, row)
        double2 cross_contrib = cmul(phase_complex, phi_prime);
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

// ===================== Wrapper Functions =====================

namespace PoIntInt {

void compute_Ak_affine_triangle_cuda_wrapper(
  const Geometry& geom,
  const KGrid& kgrid,
  const Eigen::VectorXd& dofs,
  double2* d_A_out,
  int blockSize)
{
  if (geom.type != GEOM_TRIANGLE) {
    std::cerr << "Error: AffineDoF CUDA wrapper only supports triangle meshes" << std::endl;
    return;
  }
  
  if (dofs.size() != 12) {
    std::cerr << "Error: AffineDoF requires 12 DoFs" << std::endl;
    return;
  }
  
  int NF = (int)geom.tris.size();
  if (NF == 0) return;
  
  int Q = (int)kgrid.kmag.size();
  
  // Declare all variables at the top to avoid goto issues
  TriPacked* d_tris = nullptr;
  double3* d_kdirs = nullptr;
  double* d_kmags = nullptr;
  double* d_dofs = nullptr;
  std::vector<double3> h_kdirs;
  std::vector<double> h_dofs;
  dim3 grid, block;
  
  #define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
      goto cleanup; \
    } \
  } while(0)
  
  CUDA_CHECK(cudaMalloc(&d_tris, NF * sizeof(TriPacked)));
  CUDA_CHECK(cudaMalloc(&d_kdirs, Q * sizeof(double3)));
  CUDA_CHECK(cudaMalloc(&d_kmags, Q * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_dofs, 12 * sizeof(double)));
  
  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_tris, geom.tris.data(), NF * sizeof(TriPacked), cudaMemcpyHostToDevice));
  
  h_kdirs.resize(Q);
  for (int q = 0; q < Q; ++q) {
    h_kdirs[q] = make_double3(kgrid.dirs[q][0], kgrid.dirs[q][1], kgrid.dirs[q][2]);
  }
  CUDA_CHECK(cudaMemcpy(d_kdirs, h_kdirs.data(), Q * sizeof(double3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kmags, kgrid.kmag.data(), Q * sizeof(double), cudaMemcpyHostToDevice));
  
  h_dofs.resize(12);
  for (int i = 0; i < 12; ++i) {
    h_dofs[i] = dofs(i);
  }
  CUDA_CHECK(cudaMemcpy(d_dofs, h_dofs.data(), 12 * sizeof(double), cudaMemcpyHostToDevice));
  
  // Launch kernel
  grid = dim3(Q, 1);
  block = dim3(blockSize);
  compute_A_affine_triangle_kernel<<<grid, block>>>(
    d_tris, NF, d_kdirs, d_kmags, d_dofs, Q, d_A_out
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
cleanup:
  if (d_tris) cudaFree(d_tris);
  if (d_kdirs) cudaFree(d_kdirs);
  if (d_kmags) cudaFree(d_kmags);
  if (d_dofs) cudaFree(d_dofs);
}

void compute_Ak_gradient_affine_triangle_cuda_wrapper(
  const Geometry& geom,
  const KGrid& kgrid,
  const Eigen::VectorXd& dofs,
  double2* d_grad_A_out,
  int blockSize)
{
  if (geom.type != GEOM_TRIANGLE) {
    std::cerr << "Error: AffineDoF CUDA wrapper only supports triangle meshes" << std::endl;
    return;
  }
  
  if (dofs.size() != 12) {
    std::cerr << "Error: AffineDoF requires 12 DoFs" << std::endl;
    return;
  }
  
  int NF = (int)geom.tris.size();
  if (NF == 0) return;
  
  int Q = (int)kgrid.kmag.size();
  int num_dofs = 12;
  
  // Declare all variables at the top to avoid goto issues
  TriPacked* d_tris = nullptr;
  double3* d_kdirs = nullptr;
  double* d_kmags = nullptr;
  double* d_dofs = nullptr;
  std::vector<double3> h_kdirs;
  std::vector<double> h_dofs;
  dim3 grid, block;
  
  #define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
      goto cleanup; \
    } \
  } while(0)
  
  CUDA_CHECK(cudaMalloc(&d_tris, NF * sizeof(TriPacked)));
  CUDA_CHECK(cudaMalloc(&d_kdirs, Q * sizeof(double3)));
  CUDA_CHECK(cudaMalloc(&d_kmags, Q * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_dofs, 12 * sizeof(double)));
  
  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_tris, geom.tris.data(), NF * sizeof(TriPacked), cudaMemcpyHostToDevice));
  
  h_kdirs.resize(Q);
  for (int q = 0; q < Q; ++q) {
    h_kdirs[q] = make_double3(kgrid.dirs[q][0], kgrid.dirs[q][1], kgrid.dirs[q][2]);
  }
  CUDA_CHECK(cudaMemcpy(d_kdirs, h_kdirs.data(), Q * sizeof(double3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kmags, kgrid.kmag.data(), Q * sizeof(double), cudaMemcpyHostToDevice));
  
  h_dofs.resize(12);
  for (int i = 0; i < 12; ++i) {
    h_dofs[i] = dofs(i);
  }
  CUDA_CHECK(cudaMemcpy(d_dofs, h_dofs.data(), 12 * sizeof(double), cudaMemcpyHostToDevice));
  
  // Launch kernel
  grid = dim3((Q + blockSize - 1) / blockSize);
  block = dim3(blockSize);
  compute_A_gradient_affine_triangle_kernel<<<grid, block>>>(
    d_tris, NF, d_kdirs, d_kmags, d_dofs, Q, d_grad_A_out
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
cleanup:
  if (d_tris) cudaFree(d_tris);
  if (d_kdirs) cudaFree(d_kdirs);
  if (d_kmags) cudaFree(d_kmags);
  if (d_dofs) cudaFree(d_dofs);
}

void register_affine_dof_cuda_kernels() {
  CudaKernelRegistry::register_kernels(
    "AffineDoF",
    GEOM_TRIANGLE,
    compute_Ak_affine_triangle_cuda_wrapper,
    compute_Ak_gradient_affine_triangle_cuda_wrapper
  );
}

} // namespace PoIntInt

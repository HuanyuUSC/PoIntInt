#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <math_constants.h>
#include "geometry/types.hpp"

// Define this macro before including the header to prevent duplicate declarations
#define AFFINE_DOF_CUDA_KERNELS_DEFINED

#include "cuda/cuda_helpers.hpp"
#include "dof/cuda/affine_dof_cuda.hpp"
#include "dof/cuda/dof_cuda_interface.hpp"
#include "geometry/packing.hpp"
#include <iostream>
#include <vector>

using PoIntInt::TriPacked;
using PoIntInt::DiskPacked;
using PoIntInt::GaussianPacked;
using PoIntInt::Mat3x3;
using PoIntInt::mat3x3_mul_vec;
using PoIntInt::mat3x3_mul_mat;
using PoIntInt::cross_product_matrix;
using PoIntInt::cmul;
using PoIntInt::cexp_i;
using PoIntInt::cadd;
using PoIntInt::cscale;
using PoIntInt::E_func;
using PoIntInt::E_prime;
using PoIntInt::E_double_prime;
using PoIntInt::Phi_ab;
using PoIntInt::Phi_ab_gradient;
using PoIntInt::J1_over_x;
using PoIntInt::J1_over_x_prime;

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
    
    double2 phi_prime = Phi_ab(alpha_prime, beta_prime);
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

// Kernel to compute A(k) for disks with affine transformation applied
// For each k-point q, computes A(k_q) for the transformed geometry
extern "C" __global__ void compute_A_affine_disk_kernel(
  const DiskPacked* __restrict__ disks,
  int num_disks,
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
  
  // Compute A^T for transforming k
  Mat3x3 AT(
    make_double3(A.row0.x, A.row1.x, A.row2.x),
    make_double3(A.row0.y, A.row1.y, A.row2.y),
    make_double3(A.row0.z, A.row1.z, A.row2.z)
  );
  
  double3 k_vec = make_double3(kx, ky, kz);
  double3 k_prime = mat3x3_mul_vec(AT, k_vec);
  
  // Initialize accumulator for this thread
  double2 A_local = make_double2(0.0, 0.0);
  
  // Process each disk
  for (int i = threadIdx.x; i < num_disks; i += blockDim.x) {
    DiskPacked disk = disks[i];
    
    if (disk.rho <= 0.0 || disk.area < 0.0) continue;
    
    // Transform center: c' = A * c + t
    double3 c = make_double3(disk.c.x, disk.c.y, disk.c.z);
    double3 c_prime = mat3x3_mul_vec(A, c);
    c_prime.x += t.x; c_prime.y += t.y; c_prime.z += t.z;
    
    // Normal is not transformed (disk normal is in local frame)
    double3 n = make_double3(disk.n.x, disk.n.y, disk.n.z);
    
    // Compute k'·n
    double kdotn = k_prime.x * n.x + k_prime.y * n.y + k_prime.z * n.z;
    
    // Compute r = |k' - (k'·n)n|
    double kprime_sq = k_prime.x*k_prime.x + k_prime.y*k_prime.y + k_prime.z*k_prime.z;
    double r2 = fmax(kprime_sq - kdotn*kdotn, 0.0);
    double r = sqrt(r2);
    
    // Compute S_magnitude
    double rho_r = disk.rho * r;
    double Smag;
    if (rho_r < 1e-10) {
      Smag = disk.area;
    } else {
      double x = fmin(rho_r, 100.0);
      Smag = disk.area * 2.0 * J1_over_x(x);
    }
    
    // A_parallel = S_magnitude * (k'·n) / |k|
    double Apar = Smag * kdotn / k;
    if (!isfinite(Apar)) Apar = 0.0;
    
    // Phase = exp(i*k·c')
    double phase = kx*c_prime.x + ky*c_prime.y + kz*c_prime.z;
    double2 phase_complex = cexp_i(phase);
    
    double2 A_disk = cscale(phase_complex, Apar);
    
    A_local.x += A_disk.x;
    A_local.y += A_disk.y;
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
    
    double2 phi_prime = Phi_ab(alpha_prime, beta_prime);
    double phase = kx*a_prime.x + ky*a_prime.y + kz*a_prime.z;
    double2 phase_complex = cexp_i(phase);
    double2 A_tri = cmul(phase_complex, phi_prime);
    A_tri = cscale(A_tri, gamma_prime);
    
    // Compute gradients of Phi w.r.t. alpha and beta
    double2 dPhi_dalpha, dPhi_dbeta;
    Phi_ab_gradient(alpha_prime, beta_prime, &dPhi_dalpha, &dPhi_dbeta);
    
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

// Kernel to compute gradient of A(k) w.r.t. AffineDoF for disks
// For each k-point q, computes ∂A(k_q)/∂θ for all 12 DoFs
extern "C" __global__ void compute_A_gradient_affine_disk_kernel(
  const DiskPacked* __restrict__ disks,
  int num_disks,
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
  Mat3x3 A(
    make_double3(dof_params[3], dof_params[4], dof_params[5]),
    make_double3(dof_params[6], dof_params[7], dof_params[8]),
    make_double3(dof_params[9], dof_params[10], dof_params[11])
  );
  
  // Compute A^T for transforming k
  Mat3x3 AT(
    make_double3(A.row0.x, A.row1.x, A.row2.x),
    make_double3(A.row0.y, A.row1.y, A.row2.y),
    make_double3(A.row0.z, A.row1.z, A.row2.z)
  );
  
  double3 k_vec = make_double3(kx, ky, kz);
  double3 k_prime = mat3x3_mul_vec(AT, k_vec);
  
  // Initialize gradient accumulator
  double2 grad[12];
  for (int i = 0; i < 12; ++i) {
    grad[i] = make_double2(0.0, 0.0);
  }
  
  // Process each disk
  for (int i = 0; i < num_disks; ++i) {
    DiskPacked disk = disks[i];
    
    if (disk.rho <= 0.0 || disk.area < 0.0) continue;
    
    // Transform center: c' = A * c + t
    double3 c = make_double3(disk.c.x, disk.c.y, disk.c.z);
    double3 c_prime = mat3x3_mul_vec(A, c);
    c_prime.x += t.x; c_prime.y += t.y; c_prime.z += t.z;
    
    // Normal is not transformed
    double3 n = make_double3(disk.n.x, disk.n.y, disk.n.z);
    
    // Compute k'·n
    double kdotn = k_prime.x * n.x + k_prime.y * n.y + k_prime.z * n.z;
    
    // Compute r = |k' - (k'·n)n|
    double kprime_sq = k_prime.x*k_prime.x + k_prime.y*k_prime.y + k_prime.z*k_prime.z;
    double r2 = fmax(kprime_sq - kdotn*kdotn, 0.0);
    double r = sqrt(r2);
    
    // Compute S_magnitude
    double rho_r = disk.rho * r;
    double Smag;
    if (rho_r < 1e-10) {
      Smag = disk.area;
    } else {
      double x = fmin(rho_r, 100.0);
      Smag = disk.area * 2.0 * J1_over_x(x);
    }
    
    // A_parallel = S_magnitude * (k'·n) / |k|
    double Apar = Smag * kdotn / k;
    if (!isfinite(Apar)) Apar = 0.0;
    
    // Phase = exp(i*k·c')
    double phase = kx*c_prime.x + ky*c_prime.y + kz*c_prime.z;
    double2 phase_complex = cexp_i(phase);
    
    double2 A_disk = cscale(phase_complex, Apar);
    
    // Gradient w.r.t. translation t (DoFs 0-2)
    // dA/dt = i*k * A (since phase = exp(i*k·(A*c+t)), and t only affects phase)
    double2 ik = make_double2(0.0, kx);
    grad[0] = cadd(grad[0], cmul(ik, A_disk));
    ik = make_double2(0.0, ky);
    grad[1] = cadd(grad[1], cmul(ik, A_disk));
    ik = make_double2(0.0, kz);
    grad[2] = cadd(grad[2], cmul(ik, A_disk));
    
    // Gradient w.r.t. matrix A (DoFs 3-11)
    // CPU line 338: gA = A_disk * I * k * c.transpose()
    // CPU line 339: gA += khat * n.transpose() * phase * S_magnitude
    // CPU line 340-342: gA += phase * kdotn / kmag * dS_magnitude * k * (k_prime - kdotn * n).transpose()
    // CPU line 343: Map(local_grad.data() + 3) += gA.transpose()
    
    // Contribution 1: i*k * c^T * A_disk
    // gA(row, col) = i * k_row * c_col * A_disk
    // After transpose: grad[3 + row*3 + col] = gA(row, col) = i * k_row * c_col * A_disk
    double2 ik_A = make_double2(-A_disk.y, A_disk.x);  // i * A_disk
    for (int row = 0; row < 3; ++row) {
      double k_row = (row == 0) ? kx : ((row == 1) ? ky : kz);
      for (int col = 0; col < 3; ++col) {
        double c_col = (col == 0) ? c.x : ((col == 1) ? c.y : c.z);
        int dof_idx = 3 + row * 3 + col;
        double2 contrib = cscale(ik_A, k_row * c_col);
        grad[dof_idx] = cadd(grad[dof_idx], contrib);
      }
    }
    
    // Contribution 2: khat * n^T * phase * S_magnitude
    // gA(row, col) = khat_row * n_col * phase * S_magnitude
    double2 phase_Smag = cscale(phase_complex, Smag);
    for (int row = 0; row < 3; ++row) {
      double khat_row = (row == 0) ? khat.x : ((row == 1) ? khat.y : khat.z);
      for (int col = 0; col < 3; ++col) {
        double n_col = (col == 0) ? n.x : ((col == 1) ? n.y : n.z);
        int dof_idx = 3 + row * 3 + col;
        double2 contrib = cscale(phase_Smag, khat_row * n_col);
        grad[dof_idx] = cadd(grad[dof_idx], contrib);
      }
    }
    
    // Contribution 3: phase * kdotn / kmag * dS_magnitude * k * (k_prime - kdotn * n)^T
    // CPU line 342: gA += phase * kdotn / kmag * dS_magnitude * k * (k_prime - kdotn * n).transpose()
    // This term accounts for how S_magnitude changes with r, and r changes with k_prime
    // But k_prime = A^T * k depends on A, so we need to account for ∂k_prime/∂A
    // When differentiating k * (k_prime - kdotn * n)^T w.r.t. A:
    //   ∂/∂A[m,n] [k[i] * (k_prime[j] - kdotn * n[j])] 
    //   = k[i] * (∂k_prime[j]/∂A[m,n] - n[j] * ∂kdotn/∂A[m,n])
    //   = k[i] * (k[n] * δ[j == m] - n[j] * n[m] * k[n])
    //   = k[i] * k[n] * (δ[j == m] - n[j] * n[m])
    // So we need an additional term: phase * dS_magnitude * k[i] * k[n] * (δ[j == m] - n[j] * n[m]) / kmag
    // But note: the CPU code computes this with the current k_prime, and Eigen handles the chain rule.
    // In CUDA, we need to add this term explicitly.
    double dS_magnitude;
    if (rho_r < 1e-10) {
      dS_magnitude = -0.25 * disk.rho;
    } else {
      dS_magnitude = 2.0 * J1_over_x_prime(rho_r) / r;
    }
    dS_magnitude *= disk.rho * disk.area;
    
    double3 kperp = make_double3(
      k_prime.x - kdotn * n.x,
      k_prime.y - kdotn * n.y,
      k_prime.z - kdotn * n.z
    );
    
    // Main term: k * (k_prime - kdotn * n)^T (accounts for current k_prime value)
    double coeff = (kdotn / k) * dS_magnitude;
    double2 phase_coeff = cscale(phase_complex, coeff);
    
    // gA(row, col) = phase_coeff * k_row * kperp_col
    for (int row = 0; row < 3; ++row) {
      double k_row = (row == 0) ? kx : ((row == 1) ? ky : kz);
      for (int col = 0; col < 3; ++col) {
        double kperp_col = (col == 0) ? kperp.x : ((col == 1) ? kperp.y : kperp.z);
        int dof_idx = 3 + row * 3 + col;
        double2 contrib = cscale(phase_coeff, k_row * kperp_col);
        grad[dof_idx] = cadd(grad[dof_idx], contrib);
      }
    }
    
    // Additional term from chain rule: ∂k_prime/∂A
    // The CPU code computes: gA += phase * kdotn / kmag * dS_magnitude * k * (k_prime - kdotn * n).transpose()
    // This gives: gA(i, j) = phase * kdotn / kmag * dS_magnitude * k[i] * (k_prime[j] - kdotn * n[j])
    // When Eigen differentiates this w.r.t. A, it includes the chain rule term from ∂k_prime/∂A.
    // Since k_prime = A^T * k, we have: ∂k_prime[j]/∂A[m, n] = k[n] * δ[j == m]
    // And kdotn = k_prime · n, so: ∂kdotn/∂A[m, n] = n[m] * k[n]
    // The additional term is: phase * kdotn / kmag * dS_magnitude * k[i] * k[n] * (δ[j == m] - n[j] * n[m])
    // For gA(row, col) where we store A(row, col), and after transpose:
    //   gA.transpose()(row, col) = gA(col, row)
    //   We need: gA(col, row) += phase * kdotn / kmag * dS_magnitude * k[col] * k[n] * (δ[row == m] - n[row] * n[m])
    //   For A(row, col), m=row, so we sum over n (the column index):
    //   Actually, wait - the CPU code uses Eigen which automatically handles this.
    //   The GPU code needs to match the result after Eigen's automatic differentiation.
    //   Let me compute the term more directly: the derivative of k * (k_prime - kdotn * n)^T w.r.t. A
    //   gives: k * k^T * (I - n * n^T) scaled by the appropriate factors.
    //   But we need to be careful about the indexing after transpose.
    //   Actually, I think the simplest approach is to note that the CPU code's result includes this term,
    //   so we should add: phase * dS_magnitude / kmag * k * k^T * (I - n * n^T)
    //   But we need to match the scaling: CPU uses kdotn/kmag, so we should use kdotn/kmag here too.
    double additional_coeff = (kdotn / k) * dS_magnitude;
    double2 phase_additional = cscale(phase_complex, additional_coeff);
    
    // Additional term: k * k^T * (I - n * n^T) / kmag
    // This accounts for ∂k_prime/∂A where k_prime = A^T * k
    // After transpose and mapping to our storage: gA(row, col) gets contribution from k[row] * k[col] * (δ[col == row] - n[col] * n[row])
    for (int row = 0; row < 3; ++row) {
      double k_row = (row == 0) ? kx : ((row == 1) ? ky : kz);
      for (int col = 0; col < 3; ++col) {
        double k_col = (col == 0) ? kx : ((col == 1) ? ky : kz);
        double n_col = (col == 0) ? n.x : ((col == 1) ? n.y : n.z);
        double n_row = (row == 0) ? n.x : ((row == 1) ? n.y : n.z);
        int dof_idx = 3 + row * 3 + col;
        // Term: k[row] * k[col] * (δ[col == row] - n[col] * n[row])
        double delta = (row == col) ? 1.0 : 0.0;
        double additional_val = k_row * k_col * (delta - n_col * n_row);
        double2 additional_contrib = cscale(phase_additional, additional_val);
        grad[dof_idx] = cadd(grad[dof_idx], additional_contrib);
      }
    }
    
  }
  
  // Write results
  for (int dof = 0; dof < 12; ++dof) {
    grad_A[q * 12 + dof] = grad[dof];
  }
}

// Kernel to compute A(k) for Gaussian splats with affine transformation applied
// For each k-point q, computes A(k_q) for the transformed geometry
extern "C" __global__ void compute_A_affine_gaussian_kernel(
  const GaussianPacked* __restrict__ gaussians,
  int num_gaussians,
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
  
  // Compute A^T for transforming k
  Mat3x3 AT(
    make_double3(A.row0.x, A.row1.x, A.row2.x),
    make_double3(A.row0.y, A.row1.y, A.row2.y),
    make_double3(A.row0.z, A.row1.z, A.row2.z)
  );
  
  double3 k_vec = make_double3(kx, ky, kz);
  double3 k_prime = mat3x3_mul_vec(AT, k_vec);
  
  // Initialize accumulator for this thread
  double2 A_local = make_double2(0.0, 0.0);
  
  // Process each Gaussian
  for (int i = threadIdx.x; i < num_gaussians; i += blockDim.x) {
    GaussianPacked g = gaussians[i];
    
    if (g.sigma <= 0.0 || g.w < 0.0) continue;
    
    // Transform center: c' = A * c + t
    double3 c = make_double3(g.c.x, g.c.y, g.c.z);
    double3 c_prime = mat3x3_mul_vec(A, c);
    c_prime.x += t.x; c_prime.y += t.y; c_prime.z += t.z;
    
    // Normal is not transformed (Gaussian normal is in local frame)
    double3 n = make_double3(g.n.x, g.n.y, g.n.z);
    
    // Compute k'·n
    double kdotn = k_prime.x * n.x + k_prime.y * n.y + k_prime.z * n.z;
    
    // Compute r^2 = |k'|^2 - (k'·n)^2 = ||k'_perp||^2
    double kprime_sq = k_prime.x*k_prime.x + k_prime.y*k_prime.y + k_prime.z*k_prime.z;
    double r2 = fmax(kprime_sq - kdotn*kdotn, 0.0);
    
    // S_gauss(k) = w * exp(-0.5 * sigma^2 * ||k_perp||^2)
    double exp_arg = -0.5 * g.sigma * g.sigma * r2;
    double Smag = g.w * exp(exp_arg);
    
    // A_parallel = S_magnitude * (k'·n) / |k|
    double Apar = Smag * kdotn / k;
    if (!isfinite(Apar)) Apar = 0.0;
    
    // Phase = exp(i*k·c')
    double phase = kx*c_prime.x + ky*c_prime.y + kz*c_prime.z;
    double2 phase_complex = cexp_i(phase);
    
    double2 A_gauss = cscale(phase_complex, Apar);
    
    A_local.x += A_gauss.x;
    A_local.y += A_gauss.y;
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

// Kernel to compute gradient of A(k) w.r.t. AffineDoF for Gaussian splats
// For each k-point q, computes ∂A(k_q)/∂θ for all 12 DoFs
extern "C" __global__ void compute_A_gradient_affine_gaussian_kernel(
  const GaussianPacked* __restrict__ gaussians,
  int num_gaussians,
  const double3* __restrict__ kdirs,
  const double* __restrict__ kmags,
  const double* __restrict__ dof_params,  // 12 DoF parameters [t(3), A(9)]
  int Q,
  double2* grad_A  // Output: Q × 12 complex gradients (row-major: grad_A[q * 12 + dof])
) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= Q) return;
  
  double3 khat = kdirs[q];
  double k = kmags[q];
  double kx = k * khat.x, ky = k * khat.y, kz = k * khat.z;
  
  if (k < 1e-10) {
    // At k=0, gradient = 0
    for (int dof = 0; dof < 12; ++dof) {
      grad_A[q * 12 + dof] = make_double2(0.0, 0.0);
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
  
  // Compute A^T for transforming k
  Mat3x3 AT(
    make_double3(A.row0.x, A.row1.x, A.row2.x),
    make_double3(A.row0.y, A.row1.y, A.row2.y),
    make_double3(A.row0.z, A.row1.z, A.row2.z)
  );
  
  double3 k_vec = make_double3(kx, ky, kz);
  double3 k_prime = mat3x3_mul_vec(AT, k_vec);
  
  // Initialize gradient accumulator for this thread
  double2 grad[12];
  for (int i = 0; i < 12; ++i) {
    grad[i] = make_double2(0.0, 0.0);
  }
  
  // Process each Gaussian (each thread processes all Gaussians for its k-point)
  for (int i = 0; i < num_gaussians; ++i) {
    GaussianPacked g = gaussians[i];
    
    if (g.sigma <= 0.0 || g.w < 0.0) continue;
    
    // Transform center: c' = A * c + t
    double3 c = make_double3(g.c.x, g.c.y, g.c.z);
    double3 c_prime = mat3x3_mul_vec(A, c);
    c_prime.x += t.x; c_prime.y += t.y; c_prime.z += t.z;
    
    // Normal is not transformed
    double3 n = make_double3(g.n.x, g.n.y, g.n.z);
    
    // Compute k'·n
    double kdotn = k_prime.x * n.x + k_prime.y * n.y + k_prime.z * n.z;
    
    // Compute r^2 = |k'|^2 - (k'·n)^2
    double kprime_sq = k_prime.x*k_prime.x + k_prime.y*k_prime.y + k_prime.z*k_prime.z;
    double r2 = fmax(kprime_sq - kdotn*kdotn, 0.0);
    
    // S_gauss(k) = w * exp(-0.5 * sigma^2 * r^2)
    double exp_arg = -0.5 * g.sigma * g.sigma * r2;
    double Smag = g.w * exp(exp_arg);
    
    // A_parallel = S_magnitude * (k'·n) / |k|
    double Apar = Smag * kdotn / k;
    if (!isfinite(Apar)) Apar = 0.0;
    
    // Phase = exp(i*k·c')
    double phase = kx*c_prime.x + ky*c_prime.y + kz*c_prime.z;
    double2 phase_complex = cexp_i(phase);
    
    double2 A_gauss = cscale(phase_complex, Apar);
    
    // Gradient w.r.t. translation t (DoFs 0-2)
    // dA/dt = i*k * A (since phase = exp(i*k·(A*c+t)), and t only affects phase)
    double2 ik = make_double2(0.0, kx);
    grad[0] = cadd(grad[0], cmul(ik, A_gauss));
    
    ik = make_double2(0.0, ky);
    grad[1] = cadd(grad[1], cmul(ik, A_gauss));
    
    ik = make_double2(0.0, kz);
    grad[2] = cadd(grad[2], cmul(ik, A_gauss));
    
    // Gradient w.r.t. matrix A (DoFs 3-11)
    // We need to compute gA such that grad[3 + row*3 + col] = gA(row, col)
    // Following the CPU implementation:
    //   gA = A_gauss * i*k * c^T
    //   + khat * n^T * phase * S_magnitude
    //   + phase * kdotn/kmag * dS_magnitude * k * (k_prime - kdotn*n)^T
    //   where dS_magnitude = -sigma^2 * S_magnitude
    
    // Contribution 1: A_gauss * i*k * c^T
    // Following CPU: gA = A_gauss * i*k * c^T
    // This means: gA(row, col) = A_gauss * i * k_row * c_col
    // = (i * A_gauss) * k_row * c_col
    // i * A_gauss = (-A_gauss.y, A_gauss.x)
    double2 ik_A = make_double2(-A_gauss.y, A_gauss.x);  // i * A_gauss
    for (int row = 0; row < 3; ++row) {
      double k_row = (row == 0) ? kx : ((row == 1) ? ky : kz);
      for (int col = 0; col < 3; ++col) {
        double c_col = (col == 0) ? c.x : ((col == 1) ? c.y : c.z);
        int dof_idx = 3 + row * 3 + col;
        double2 contrib1 = cscale(ik_A, k_row * c_col);
        grad[dof_idx] = cadd(grad[dof_idx], contrib1);
      }
    }
    
    // Contribution 2: khat * n^T * phase * S_magnitude
    for (int row = 0; row < 3; ++row) {
      double khat_row = (row == 0) ? khat.x : ((row == 1) ? khat.y : khat.z);
      for (int col = 0; col < 3; ++col) {
        double n_col = (col == 0) ? n.x : ((col == 1) ? n.y : n.z);
        int dof_idx = 3 + row * 3 + col;
        double2 contrib2 = cscale(phase_complex, khat_row * n_col * Smag);
        grad[dof_idx] = cadd(grad[dof_idx], contrib2);
      }
    }
    
    // Contribution 3: phase * kdotn/kmag * dS_magnitude * k * (k_prime - kdotn*n)^T
    // dS_magnitude = -sigma^2 * S_magnitude
    double dS_magnitude = -g.sigma * g.sigma * Smag;
    double3 k_perp = make_double3(
      k_prime.x - kdotn * n.x,
      k_prime.y - kdotn * n.y,
      k_prime.z - kdotn * n.z
    );
    double coeff3 = (kdotn / k) * dS_magnitude;
    double2 phase_coeff3 = cscale(phase_complex, coeff3);
    
    for (int row = 0; row < 3; ++row) {
      double k_row = (row == 0) ? kx : ((row == 1) ? ky : kz);
      for (int col = 0; col < 3; ++col) {
        double kperp_col = (col == 0) ? k_perp.x : ((col == 1) ? k_perp.y : k_perp.z);
        int dof_idx = 3 + row * 3 + col;
        double2 contrib3 = cscale(phase_coeff3, k_row * kperp_col);
        grad[dof_idx] = cadd(grad[dof_idx], contrib3);
      }
    }
    
    // Contribution 4: Additional chain rule term for ∂k_prime/∂A
    // Similar to disk case: k * k^T * (I - n * n^T) scaled appropriately
    double additional_coeff = (kdotn / k) * dS_magnitude;
    double2 phase_additional = cscale(phase_complex, additional_coeff);
    
    // Additional term: k * k^T * (I - n * n^T) / kmag
    // This accounts for ∂k_prime/∂A where k_prime = A^T * k
    for (int row = 0; row < 3; ++row) {
      double k_row = (row == 0) ? kx : ((row == 1) ? ky : kz);
      for (int col = 0; col < 3; ++col) {
        double k_col = (col == 0) ? kx : ((col == 1) ? ky : kz);
        double n_col = (col == 0) ? n.x : ((col == 1) ? n.y : n.z);
        double n_row = (row == 0) ? n.x : ((row == 1) ? n.y : n.z);
        int dof_idx = 3 + row * 3 + col;
        // Term: k[row] * k[col] * (δ[col == row] - n[col] * n[row])
        double delta = (row == col) ? 1.0 : 0.0;
        double additional_val = k_row * k_col * (delta - n_col * n_row);
        double2 additional_contrib = cscale(phase_additional, additional_val);
        grad[dof_idx] = cadd(grad[dof_idx], additional_contrib);
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

// Wrapper for computing A(k) for disks with AffineDoF
void compute_Ak_affine_disk_cuda_wrapper(
  const Geometry& geom,
  const KGrid& kgrid,
  const Eigen::VectorXd& dofs,
  double2* d_A_out,
  int blockSize)
{
  if (geom.type != GEOM_DISK) {
    std::cerr << "Error: AffineDoF CUDA wrapper only supports point clouds (disks)" << std::endl;
    return;
  }
  
  if (dofs.size() != 12) {
    std::cerr << "Error: AffineDoF requires 12 DoFs" << std::endl;
    return;
  }
  
  int ND = (int)geom.disks.size();
  if (ND == 0) return;
  
  int Q = (int)kgrid.kmag.size();
  
  // Declare all variables at the top to avoid goto issues
  DiskPacked* d_disks = nullptr;
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
  
  CUDA_CHECK(cudaMalloc(&d_disks, ND * sizeof(DiskPacked)));
  CUDA_CHECK(cudaMalloc(&d_kdirs, Q * sizeof(double3)));
  CUDA_CHECK(cudaMalloc(&d_kmags, Q * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_dofs, 12 * sizeof(double)));
  
  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_disks, geom.disks.data(), ND * sizeof(DiskPacked), cudaMemcpyHostToDevice));
  
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
  compute_A_affine_disk_kernel<<<grid, block>>>(
    d_disks, ND, d_kdirs, d_kmags, d_dofs, Q, d_A_out
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
cleanup:
  if (d_disks) cudaFree(d_disks);
  if (d_kdirs) cudaFree(d_kdirs);
  if (d_kmags) cudaFree(d_kmags);
  if (d_dofs) cudaFree(d_dofs);
}

// Wrapper for computing ∂A(k)/∂θ for disks with AffineDoF
void compute_Ak_gradient_affine_disk_cuda_wrapper(
  const Geometry& geom,
  const KGrid& kgrid,
  const Eigen::VectorXd& dofs,
  double2* d_grad_A_out,
  int blockSize)
{
  if (geom.type != GEOM_DISK) {
    std::cerr << "Error: AffineDoF CUDA wrapper only supports point clouds (disks)" << std::endl;
    return;
  }
  
  if (dofs.size() != 12) {
    std::cerr << "Error: AffineDoF requires 12 DoFs" << std::endl;
    return;
  }
  
  int ND = (int)geom.disks.size();
  if (ND == 0) return;
  
  int Q = (int)kgrid.kmag.size();
  int num_dofs = 12;
  
  // Declare all variables at the top to avoid goto issues
  DiskPacked* d_disks = nullptr;
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
  
  CUDA_CHECK(cudaMalloc(&d_disks, ND * sizeof(DiskPacked)));
  CUDA_CHECK(cudaMalloc(&d_kdirs, Q * sizeof(double3)));
  CUDA_CHECK(cudaMalloc(&d_kmags, Q * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_dofs, 12 * sizeof(double)));
  
  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_disks, geom.disks.data(), ND * sizeof(DiskPacked), cudaMemcpyHostToDevice));
  
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
  compute_A_gradient_affine_disk_kernel<<<grid, block>>>(
    d_disks, ND, d_kdirs, d_kmags, d_dofs, Q, d_grad_A_out
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
cleanup:
  if (d_disks) cudaFree(d_disks);
  if (d_kdirs) cudaFree(d_kdirs);
  if (d_kmags) cudaFree(d_kmags);
  if (d_dofs) cudaFree(d_dofs);
}

// Wrapper for computing A(k) for Gaussian splats with AffineDoF
void compute_Ak_affine_gaussian_cuda_wrapper(
  const Geometry& geom,
  const KGrid& kgrid,
  const Eigen::VectorXd& dofs,
  double2* d_A_out,
  int blockSize)
{
  if (geom.type != GEOM_GAUSSIAN) {
    std::cerr << "Error: AffineDoF CUDA wrapper only supports Gaussian splats" << std::endl;
    return;
  }
  
  if (dofs.size() != 12) {
    std::cerr << "Error: AffineDoF requires 12 DoFs" << std::endl;
    return;
  }
  
  int NG = (int)geom.gaussians.size();
  if (NG == 0) return;
  
  int Q = (int)kgrid.kmag.size();
  
  // Declare all variables at the top to avoid goto issues
  GaussianPacked* d_gaussians = nullptr;
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
  
  CUDA_CHECK(cudaMalloc(&d_gaussians, NG * sizeof(GaussianPacked)));
  CUDA_CHECK(cudaMalloc(&d_kdirs, Q * sizeof(double3)));
  CUDA_CHECK(cudaMalloc(&d_kmags, Q * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_dofs, 12 * sizeof(double)));
  
  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_gaussians, geom.gaussians.data(), NG * sizeof(GaussianPacked), cudaMemcpyHostToDevice));
  
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
  compute_A_affine_gaussian_kernel<<<grid, block>>>(
    d_gaussians, NG, d_kdirs, d_kmags, d_dofs, Q, d_A_out
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
cleanup:
  if (d_gaussians) cudaFree(d_gaussians);
  if (d_kdirs) cudaFree(d_kdirs);
  if (d_kmags) cudaFree(d_kmags);
  if (d_dofs) cudaFree(d_dofs);
}

void compute_Ak_gradient_affine_gaussian_cuda_wrapper(
  const Geometry& geom,
  const KGrid& kgrid,
  const Eigen::VectorXd& dofs,
  double2* d_grad_A_out,
  int blockSize)
{
  if (geom.type != GEOM_GAUSSIAN) {
    std::cerr << "Error: AffineDoF CUDA wrapper only supports Gaussian splats" << std::endl;
    return;
  }
  
  if (dofs.size() != 12) {
    std::cerr << "Error: AffineDoF requires 12 DoFs" << std::endl;
    return;
  }
  
  int NG = (int)geom.gaussians.size();
  if (NG == 0) return;
  
  int Q = (int)kgrid.kmag.size();
  int num_dofs = 12;
  
  // Declare all variables at the top to avoid goto issues
  GaussianPacked* d_gaussians = nullptr;
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
  
  CUDA_CHECK(cudaMalloc(&d_gaussians, NG * sizeof(GaussianPacked)));
  CUDA_CHECK(cudaMalloc(&d_kdirs, Q * sizeof(double3)));
  CUDA_CHECK(cudaMalloc(&d_kmags, Q * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_dofs, 12 * sizeof(double)));
  
  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_gaussians, geom.gaussians.data(), NG * sizeof(GaussianPacked), cudaMemcpyHostToDevice));
  
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
  
  // Launch kernel (each thread handles one k-point, processes all Gaussians)
  grid = dim3((Q + blockSize - 1) / blockSize);
  block = dim3(blockSize);
  compute_A_gradient_affine_gaussian_kernel<<<grid, block>>>(
    d_gaussians, NG, d_kdirs, d_kmags, d_dofs, Q, d_grad_A_out
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
cleanup:
  if (d_gaussians) cudaFree(d_gaussians);
  if (d_kdirs) cudaFree(d_kdirs);
  if (d_kmags) cudaFree(d_kmags);
  if (d_dofs) cudaFree(d_dofs);
}

void register_affine_dof_cuda_kernels() {
  // Register triangle mesh kernels
  CudaKernelRegistry::register_kernels(
    "AffineDoF",
    GEOM_TRIANGLE,
    compute_Ak_affine_triangle_cuda_wrapper,
    compute_Ak_gradient_affine_triangle_cuda_wrapper
  );
  
  // Register disk (point cloud) kernels
  CudaKernelRegistry::register_kernels(
    "AffineDoF",
    GEOM_DISK,
    compute_Ak_affine_disk_cuda_wrapper,
    compute_Ak_gradient_affine_disk_cuda_wrapper
  );
  
  // Register Gaussian splat kernels
  CudaKernelRegistry::register_kernels(
    "AffineDoF",
    GEOM_GAUSSIAN,
    compute_Ak_affine_gaussian_cuda_wrapper,
    compute_Ak_gradient_affine_gaussian_cuda_wrapper
  );
}

// Static initializer to automatically register kernels when the library loads
// This ensures kernels are registered before any code tries to use them
namespace {
  struct AffineDoFKernelAutoRegistrar {
    AffineDoFKernelAutoRegistrar() {
      register_affine_dof_cuda_kernels();
    }
  };
  
  // Global static instance ensures constructor runs at library load time
  static AffineDoFKernelAutoRegistrar g_affine_dof_auto_registrar;
}

} // namespace PoIntInt

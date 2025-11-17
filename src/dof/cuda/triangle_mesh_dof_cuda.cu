#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <math_constants.h>
#include "geometry/types.hpp"

// Define this macro before including the header to prevent duplicate declarations
#define TRIANGLE_MESH_DOF_CUDA_KERNELS_DEFINED

#include "cuda/cuda_helpers.hpp"
#include "dof/cuda/dof_cuda_interface.hpp"
#include "geometry/packing.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <algorithm>

using PoIntInt::TriPacked;
using PoIntInt::Phi_ab;
using PoIntInt::Phi_ab_gradient;
using PoIntInt::cexp_i;
using PoIntInt::cadd;
using PoIntInt::cscale;
using PoIntInt::cmul;

constexpr int MAX_BLOCK_SIZE = 256;

namespace PoIntInt {

// ===================== CUDA Kernels =====================

// Kernel to compute A(k) for a triangle mesh with vertex positions as DoFs
// For each k-point q, computes A(k_q) for the mesh
extern "C" __global__ void compute_A_triangle_mesh_kernel(
  const TriPacked* __restrict__ tris,  // Triangles with vertex indices encoded
  int num_tris,
  const double3* __restrict__ kdirs,
  const double* __restrict__ kmags,
  const double* __restrict__ vertex_positions,  // Vertex positions: [v0_x, v0_y, v0_z, v1_x, ...]
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
  
  // Initialize accumulator for this thread
  double2 A_local = make_double2(0.0, 0.0);
  
  // Process each triangle
  for (int f = threadIdx.x; f < num_tris; f += blockDim.x) {
    const TriPacked& tri = tris[f];
    
    // Get vertex indices from the triangle
    int vid0 = tri.vid0;
    int vid1 = tri.vid1;
    int vid2 = tri.vid2;
    
    // Skip if vertex indices are not set (shouldn't happen for TriangleMeshDoF)
    if (vid0 < 0 || vid1 < 0 || vid2 < 0) continue;
    
    // Get vertex positions from DoFs
    double3 v0 = make_double3(
      vertex_positions[vid0 * 3 + 0],
      vertex_positions[vid0 * 3 + 1],
      vertex_positions[vid0 * 3 + 2]
    );
    double3 v1 = make_double3(
      vertex_positions[vid1 * 3 + 0],
      vertex_positions[vid1 * 3 + 1],
      vertex_positions[vid1 * 3 + 2]
    );
    double3 v2 = make_double3(
      vertex_positions[vid2 * 3 + 0],
      vertex_positions[vid2 * 3 + 1],
      vertex_positions[vid2 * 3 + 2]
    );
    
    // Edges: e1 = v1 - v0, e2 = v2 - v0
    double3 e1 = make_double3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    double3 e2 = make_double3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    
    // Area-vector σn = 0.5 * (e1 × e2)
    double3 sigma_n = make_double3(
      0.5 * (e1.y * e2.z - e1.z * e2.y),
      0.5 * (e1.z * e2.x - e1.x * e2.z),
      0.5 * (e1.x * e2.y - e1.y * e2.x)
    );
    
    // κ = k̂ · σn (real)
    double kappa = khat.x * sigma_n.x + khat.y * sigma_n.y + khat.z * sigma_n.z;
    
    // T = exp(i k·v0)
    double phase = kx * v0.x + ky * v0.y + kz * v0.z;
    double2 T = cexp_i(phase);
    
    // ψ = Φ(k·e1, k·e2)
    double k_dot_e1 = kx * e1.x + ky * e1.y + kz * e1.z;
    double k_dot_e2 = kx * e2.x + ky * e2.y + kz * e2.z;
    double2 psi = Phi_ab(k_dot_e1, k_dot_e2);
    
    // Contribution: κ * T * ψ
    double2 contrib = cmul(cscale(T, kappa), psi);
    A_local = cadd(A_local, contrib);
  }
  
  // Tree reduction using shared memory
  __shared__ double s_A_x[MAX_BLOCK_SIZE];
  __shared__ double s_A_y[MAX_BLOCK_SIZE];
  
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

// Kernel to compute gradient of A(k) w.r.t. vertex positions
// For each k-point q, computes ∂A(k_q)/∂θ for all vertex positions
extern "C" __global__ void compute_A_gradient_triangle_mesh_kernel(
  const TriPacked* __restrict__ tris,  // Triangles with vertex indices encoded
  int num_tris,
  const double3* __restrict__ kdirs,
  const double* __restrict__ kmags,
  const double* __restrict__ vertex_positions,  // Vertex positions: [v0_x, v0_y, v0_z, v1_x, ...]
  int num_vertices,
  int Q,
  double2* grad_A  // Output: Q × (3*num_vertices) complex gradients (row-major)
) {
  int q = blockIdx.x;
  if (q >= Q) return;

  double3 khat = kdirs[q];
  double k = kmags[q];
  double kx = k * khat.x, ky = k * khat.y, kz = k * khat.z;
  int num_dofs = 3 * num_vertices;

  int tid = threadIdx.x;
  
  double2* grad_q = grad_A + q * num_dofs;

  if (k < 1e-10) {
    for (int dof = tid; dof < num_dofs; dof += blockDim.x) {
      grad_q[dof] = make_double2(0.0, 0.0);
    }
    return;
  }
  
  // Zero gradients for this k-point
  for (int dof = tid; dof < num_dofs; dof += blockDim.x) {
    grad_q[dof] = make_double2(0.0, 0.0);
  }
  __syncthreads();
  
  // Process each triangle assigned to this thread
  for (int f = tid; f < num_tris; f += blockDim.x) {
    const TriPacked& tri = tris[f];
    
    int vid0 = tri.vid0;
    int vid1 = tri.vid1;
    int vid2 = tri.vid2;
    if (vid0 < 0 || vid1 < 0 || vid2 < 0) continue;
    
    double3 v0 = make_double3(
      vertex_positions[vid0 * 3 + 0],
      vertex_positions[vid0 * 3 + 1],
      vertex_positions[vid0 * 3 + 2]
    );
    double3 v1 = make_double3(
      vertex_positions[vid1 * 3 + 0],
      vertex_positions[vid1 * 3 + 1],
      vertex_positions[vid1 * 3 + 2]
    );
    double3 v2 = make_double3(
      vertex_positions[vid2 * 3 + 0],
      vertex_positions[vid2 * 3 + 1],
      vertex_positions[vid2 * 3 + 2]
    );
    
    double3 e1 = make_double3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    double3 e2 = make_double3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    
    double3 sigma_n = make_double3(
      0.5 * (e1.y * e2.z - e1.z * e2.y),
      0.5 * (e1.z * e2.x - e1.x * e2.z),
      0.5 * (e1.x * e2.y - e1.y * e2.x)
    );
    
    double kappa = khat.x * sigma_n.x + khat.y * sigma_n.y + khat.z * sigma_n.z;
    
    double3 dkap_v0 = make_double3(
      0.5 * (khat.y * (e2.z - e1.z) - khat.z * (e2.y - e1.y)),
      0.5 * (khat.z * (e2.x - e1.x) - khat.x * (e2.z - e1.z)),
      0.5 * (khat.x * (e2.y - e1.y) - khat.y * (e2.x - e1.x))
    );
    double3 dkap_v1 = make_double3(
      -0.5 * (khat.y * e2.z - khat.z * e2.y),
      -0.5 * (khat.z * e2.x - khat.x * e2.z),
      -0.5 * (khat.x * e2.y - khat.y * e2.x)
    );
    double3 dkap_v2 = make_double3(
      0.5 * (khat.y * e1.z - khat.z * e1.y),
      0.5 * (khat.z * e1.x - khat.x * e1.z),
      0.5 * (khat.x * e1.y - khat.y * e1.x)
    );
    
    double phase = kx * v0.x + ky * v0.y + kz * v0.z;
    double2 T = cexp_i(phase);
    
    double k_dot_e1 = kx * e1.x + ky * e1.y + kz * e1.z;
    double k_dot_e2 = kx * e2.x + ky * e2.y + kz * e2.z;
    double2 psi = Phi_ab(k_dot_e1, k_dot_e2);
    
    double2 dPa_d, dPb_d;
    Phi_ab_gradient(k_dot_e1, k_dot_e2, &dPa_d, &dPb_d);
    
    double2 I_d = make_double2(0.0, 1.0);
    double2 psi_I = cmul(psi, I_d);
    double2 psi_I_minus_dPa_dPb = cadd(cadd(psi_I, cscale(dPa_d, -1.0)), cscale(dPb_d, -1.0));
    double2 kappa_k_psi_I_minus = cscale(psi_I_minus_dPa_dPb, kappa);
    double2 kappa_k_dPa = cscale(dPa_d, kappa);
    double2 kappa_k_dPb = cscale(dPb_d, kappa);
    
    double2 grad_v0_x = cadd(cscale(psi, dkap_v0.x), cscale(kappa_k_psi_I_minus, kx));
    double2 grad_v0_y = cadd(cscale(psi, dkap_v0.y), cscale(kappa_k_psi_I_minus, ky));
    double2 grad_v0_z = cadd(cscale(psi, dkap_v0.z), cscale(kappa_k_psi_I_minus, kz));
    
    double2 grad_v1_x = cadd(cscale(psi, dkap_v1.x), cscale(kappa_k_dPa, kx));
    double2 grad_v1_y = cadd(cscale(psi, dkap_v1.y), cscale(kappa_k_dPa, ky));
    double2 grad_v1_z = cadd(cscale(psi, dkap_v1.z), cscale(kappa_k_dPa, kz));
    
    double2 grad_v2_x = cadd(cscale(psi, dkap_v2.x), cscale(kappa_k_dPb, kx));
    double2 grad_v2_y = cadd(cscale(psi, dkap_v2.y), cscale(kappa_k_dPb, ky));
    double2 grad_v2_z = cadd(cscale(psi, dkap_v2.z), cscale(kappa_k_dPb, kz));
    
    double2 contrib;
    
    contrib = cmul(T, grad_v0_x);
    int idx = vid0 * 3 + 0;
    atomicAdd(&grad_q[idx].x, contrib.x);
    atomicAdd(&grad_q[idx].y, contrib.y);
    
    contrib = cmul(T, grad_v0_y);
    idx = vid0 * 3 + 1;
    atomicAdd(&grad_q[idx].x, contrib.x);
    atomicAdd(&grad_q[idx].y, contrib.y);
    
    contrib = cmul(T, grad_v0_z);
    idx = vid0 * 3 + 2;
    atomicAdd(&grad_q[idx].x, contrib.x);
    atomicAdd(&grad_q[idx].y, contrib.y);
    
    contrib = cmul(T, grad_v1_x);
    idx = vid1 * 3 + 0;
    atomicAdd(&grad_q[idx].x, contrib.x);
    atomicAdd(&grad_q[idx].y, contrib.y);
    
    contrib = cmul(T, grad_v1_y);
    idx = vid1 * 3 + 1;
    atomicAdd(&grad_q[idx].x, contrib.x);
    atomicAdd(&grad_q[idx].y, contrib.y);
    
    contrib = cmul(T, grad_v1_z);
    idx = vid1 * 3 + 2;
    atomicAdd(&grad_q[idx].x, contrib.x);
    atomicAdd(&grad_q[idx].y, contrib.y);
    
    contrib = cmul(T, grad_v2_x);
    idx = vid2 * 3 + 0;
    atomicAdd(&grad_q[idx].x, contrib.x);
    atomicAdd(&grad_q[idx].y, contrib.y);
    
    contrib = cmul(T, grad_v2_y);
    idx = vid2 * 3 + 1;
    atomicAdd(&grad_q[idx].x, contrib.x);
    atomicAdd(&grad_q[idx].y, contrib.y);
    
    contrib = cmul(T, grad_v2_z);
    idx = vid2 * 3 + 2;
    atomicAdd(&grad_q[idx].x, contrib.x);
    atomicAdd(&grad_q[idx].y, contrib.y);
  }
}

// ===================== Wrapper Functions =====================

void compute_Ak_triangle_mesh_cuda_wrapper(
  const Geometry& geom,
  const KGrid& kgrid,
  const Eigen::VectorXd& dofs,
  double2* d_A_out,
  int blockSize)
{
  if (geom.type != GEOM_TRIANGLE) {
    std::cerr << "Error: TriangleMeshDoF CUDA wrapper only supports triangle meshes" << std::endl;
    return;
  }
  
  int num_tris = (int)geom.tris.size();
  if (num_tris == 0) return;
  
  int Q = (int)kgrid.kmag.size();
  int num_vertices = (int)(dofs.size() / 3);
  
  TriPacked* d_tris = nullptr;
  double3* d_kdirs = nullptr;
  double* d_kmags = nullptr;
  double* d_vertex_positions = nullptr;
  std::vector<double3> h_kdirs;
  std::vector<double> h_vertex_positions;
  dim3 grid, block;
  
  #define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
      goto cleanup; \
    } \
  } while(0)
  
  CUDA_CHECK(cudaMalloc(&d_tris, num_tris * sizeof(TriPacked)));
  CUDA_CHECK(cudaMalloc(&d_kdirs, Q * sizeof(double3)));
  CUDA_CHECK(cudaMalloc(&d_kmags, Q * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_vertex_positions, num_vertices * 3 * sizeof(double)));
  
  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_tris, geom.tris.data(), num_tris * sizeof(TriPacked), cudaMemcpyHostToDevice));
  
  h_kdirs.resize(Q);
  for (int q = 0; q < Q; ++q) {
    h_kdirs[q] = make_double3(kgrid.dirs[q][0], kgrid.dirs[q][1], kgrid.dirs[q][2]);
  }
  CUDA_CHECK(cudaMemcpy(d_kdirs, h_kdirs.data(), Q * sizeof(double3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kmags, kgrid.kmag.data(), Q * sizeof(double), cudaMemcpyHostToDevice));
  
  h_vertex_positions.resize(num_vertices * 3);
  for (int i = 0; i < num_vertices * 3; ++i) {
    h_vertex_positions[i] = dofs(i);
  }
  CUDA_CHECK(cudaMemcpy(d_vertex_positions, h_vertex_positions.data(), num_vertices * 3 * sizeof(double), cudaMemcpyHostToDevice));
  
  // Launch kernel
  int actualBlockSize = std::min(blockSize, MAX_BLOCK_SIZE);
  grid = dim3(Q, 1);
  block = dim3(actualBlockSize);
  compute_A_triangle_mesh_kernel<<<grid, block>>>(
    d_tris, num_tris, d_kdirs, d_kmags, d_vertex_positions, Q, d_A_out
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
cleanup:
  if (d_tris) cudaFree(d_tris);
  if (d_kdirs) cudaFree(d_kdirs);
  if (d_kmags) cudaFree(d_kmags);
  if (d_vertex_positions) cudaFree(d_vertex_positions);
}


void compute_Ak_gradient_triangle_mesh_cuda_wrapper(
  const Geometry& geom,
  const KGrid& kgrid,
  const Eigen::VectorXd& dofs,
  double2* d_grad_A_out,
  int blockSize)
{
  if (geom.type != GEOM_TRIANGLE) {
    std::cerr << "Error: TriangleMeshDoF CUDA wrapper only supports triangle meshes" << std::endl;
    return;
  }
  
  int num_tris = (int)geom.tris.size();
  if (num_tris == 0) return;
  
  int Q = (int)kgrid.kmag.size();
  int num_vertices = (int)(dofs.size() / 3);
  int num_dofs = 3 * num_vertices;
  
  TriPacked* d_tris = nullptr;
  double3* d_kdirs = nullptr;
  double* d_kmags = nullptr;
  double* d_vertex_positions = nullptr;
  std::vector<double3> h_kdirs;
  std::vector<double> h_vertex_positions;
  dim3 grid, block;
  
  #define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
      goto cleanup; \
    } \
  } while(0)
  
  CUDA_CHECK(cudaMalloc(&d_tris, num_tris * sizeof(TriPacked)));
  CUDA_CHECK(cudaMalloc(&d_kdirs, Q * sizeof(double3)));
  CUDA_CHECK(cudaMalloc(&d_kmags, Q * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_vertex_positions, num_vertices * 3 * sizeof(double)));
  
  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_tris, geom.tris.data(), num_tris * sizeof(TriPacked), cudaMemcpyHostToDevice));
  
  h_kdirs.resize(Q);
  for (int q = 0; q < Q; ++q) {
    h_kdirs[q] = make_double3(kgrid.dirs[q][0], kgrid.dirs[q][1], kgrid.dirs[q][2]);
  }
  CUDA_CHECK(cudaMemcpy(d_kdirs, h_kdirs.data(), Q * sizeof(double3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kmags, kgrid.kmag.data(), Q * sizeof(double), cudaMemcpyHostToDevice));
  
  h_vertex_positions.resize(num_vertices * 3);
  for (int i = 0; i < num_vertices * 3; ++i) {
    h_vertex_positions[i] = dofs(i);
  }
  CUDA_CHECK(cudaMemcpy(d_vertex_positions, h_vertex_positions.data(), num_vertices * 3 * sizeof(double), cudaMemcpyHostToDevice));
  
  // Launch kernel (each thread handles one k-point)
  int actualBlockSize = std::min(blockSize, MAX_BLOCK_SIZE);
  grid = dim3(Q, 1);
  block = dim3(actualBlockSize);
  compute_A_gradient_triangle_mesh_kernel<<<grid, block>>>(
    d_tris, num_tris, d_kdirs, d_kmags, d_vertex_positions, num_vertices, Q, d_grad_A_out
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  
cleanup:
  if (d_tris) cudaFree(d_tris);
  if (d_kdirs) cudaFree(d_kdirs);
  if (d_kmags) cudaFree(d_kmags);
  if (d_vertex_positions) cudaFree(d_vertex_positions);
}

void register_triangle_mesh_dof_cuda_kernels() {
  // Register triangle mesh kernels
  CudaKernelRegistry::register_kernels(
    "TriangleMeshDoF",
    GEOM_TRIANGLE,
    compute_Ak_triangle_mesh_cuda_wrapper,
    compute_Ak_gradient_triangle_mesh_cuda_wrapper
  );
}

// Static initializer to automatically register kernels when the library loads
namespace {
  struct TriangleMeshDoFKernelAutoRegistrar {
    TriangleMeshDoFKernelAutoRegistrar() {
      register_triangle_mesh_dof_cuda_kernels();
    }
  };
  
  // Global static instance ensures constructor runs at library load time
  static TriangleMeshDoFKernelAutoRegistrar g_triangle_mesh_dof_auto_registrar;
}

} // namespace PoIntInt


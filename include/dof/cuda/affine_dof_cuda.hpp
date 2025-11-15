#pragma once
#include "dof/affine_dof.hpp"
#include "dof/cuda/dof_cuda_interface.hpp"
#include "geometry/geometry.hpp"
#include "geometry/types.hpp"
#include "quadrature/kgrid.hpp"
#include <Eigen/Dense>
#include <memory>
#include <cuda_runtime.h>

namespace PoIntInt {

// CUDA kernel declarations (defined in affine_dof_cuda.cu)
// Only include these declarations when compiling CUDA code, but not when defining them
#ifdef __CUDACC__
#ifndef AFFINE_DOF_CUDA_KERNELS_DEFINED
extern "C" __global__ void compute_A_affine_triangle_kernel(
  const TriPacked* __restrict__ tris,
  int num_tris,
  const double3* __restrict__ kdirs,
  const double* __restrict__ kmags,
  const double* __restrict__ dof_params,
  int Q,
  double2* A_out
);

extern "C" __global__ void compute_A_gradient_affine_triangle_kernel(
  const TriPacked* __restrict__ tris,
  int num_tris,
  const double3* __restrict__ kdirs,
  const double* __restrict__ kmags,
  const double* __restrict__ dof_params,
  int Q,
  double2* grad_A
);

extern "C" __global__ void compute_A_affine_disk_kernel(
  const DiskPacked* __restrict__ disks,
  int num_disks,
  const double3* __restrict__ kdirs,
  const double* __restrict__ kmags,
  const double* __restrict__ dof_params,
  int Q,
  double2* A_out
);

extern "C" __global__ void compute_A_gradient_affine_disk_kernel(
  const DiskPacked* __restrict__ disks,
  int num_disks,
  const double3* __restrict__ kdirs,
  const double* __restrict__ kmags,
  const double* __restrict__ dof_params,
  int Q,
  double2* grad_A
);
#endif // AFFINE_DOF_CUDA_KERNELS_DEFINED
#endif // __CUDACC__

// CUDA kernel wrappers for AffineDoF
// These wrap the existing CUDA kernels and provide a uniform interface

// Compute A(k) for all k-points using CUDA
// Wrapper around compute_A_affine_triangle_kernel
void compute_Ak_affine_triangle_cuda_wrapper(
  const Geometry& geom,
  const KGrid& kgrid,
  const Eigen::VectorXd& dofs,
  double2* d_A_out,
  int blockSize
);

// Compute ∂A(k)/∂θ for all k-points using CUDA
// Wrapper around compute_A_gradient_affine_triangle_kernel
void compute_Ak_gradient_affine_triangle_cuda_wrapper(
  const Geometry& geom,
  const KGrid& kgrid,
  const Eigen::VectorXd& dofs,
  double2* d_grad_A_out,
  int blockSize
);

// Compute A(k) for all k-points using CUDA (for disks/point clouds)
// Wrapper around compute_A_affine_disk_kernel
void compute_Ak_affine_disk_cuda_wrapper(
  const Geometry& geom,
  const KGrid& kgrid,
  const Eigen::VectorXd& dofs,
  double2* d_A_out,
  int blockSize
);

// Compute ∂A(k)/∂θ for all k-points using CUDA (for disks/point clouds)
// Wrapper around compute_A_gradient_affine_disk_kernel
void compute_Ak_gradient_affine_disk_cuda_wrapper(
  const Geometry& geom,
  const KGrid& kgrid,
  const Eigen::VectorXd& dofs,
  double2* d_grad_A_out,
  int blockSize
);

// Register AffineDoF CUDA kernels (for both triangles and disks)
// This should be called once, typically in a static initializer
void register_affine_dof_cuda_kernels();

} // namespace PoIntInt


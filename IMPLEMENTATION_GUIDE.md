# Implementation Guide: Extensible CUDA Gradient Computation

## Overview

This guide shows how to use the new extensible CUDA gradient computation system. The system uses a registry pattern to allow different DoF types to provide their own CUDA kernels without modifying the core gradient computation code.

## Step 1: Register CUDA Kernels at Startup

In your main application or test initialization, register the CUDA kernels:

```cpp
#include "dof/cuda/affine_dof_cuda.hpp"

int main() {
  // Register AffineDoF CUDA kernels
  PoIntInt::register_affine_dof_cuda_kernels();
  
  // ... rest of your code
}
```

## Step 2: Use the Registry in Gradient Computation

The main gradient function will automatically check the registry and use CUDA kernels if available:

```cpp
#include "compute_intersection_volume_gradient.hpp"
#include "dof/cuda/dof_cuda_interface.hpp"

PoIntInt::IntersectionVolumeGradientResult result = 
  PoIntInt::compute_intersection_volume_gradient_cuda(
    geom1, geom2, dof1, dof2, dofs1, dofs2, kgrid, blockSize, enable_profiling
  );
```

## Step 3: Adding Support for a New DoF Type

To add CUDA support for a new DoF type (e.g., `TriangleMeshDoF`):

### 3.1 Create CUDA Kernels

In `src/compute_intersection_volume_gradient.cu` or a separate file:

```cpp
__global__ void compute_A_triangle_mesh_kernel(...) {
  // Your CUDA kernel implementation
}

__global__ void compute_A_gradient_triangle_mesh_kernel(...) {
  // Your CUDA kernel implementation
}
```

### 3.2 Create Wrapper Functions

In `src/dof/cuda/triangle_mesh_dof_cuda.cpp`:

```cpp
#include "dof/cuda/triangle_mesh_dof_cuda.hpp"
#include "dof/cuda/dof_cuda_interface.hpp"

namespace PoIntInt {

void compute_Ak_triangle_mesh_cuda_wrapper(
  const Geometry& geom,
  const KGrid& kgrid,
  const Eigen::VectorXd& dofs,
  double2* d_A_out,
  int blockSize)
{
  // Allocate device memory
  // Copy geometry and DoF data
  // Launch compute_A_triangle_mesh_kernel
  // Handle errors
}

void compute_Ak_gradient_triangle_mesh_cuda_wrapper(
  const Geometry& geom,
  const KGrid& kgrid,
  const Eigen::VectorXd& dofs,
  double2* d_grad_A_out,
  int blockSize)
{
  // Similar to above, but for gradient kernel
}

void register_triangle_mesh_dof_cuda_kernels() {
  CudaKernelRegistry::register_kernels(
    "TriangleMeshDoF",
    GEOM_TRIANGLE,
    compute_Ak_triangle_mesh_cuda_wrapper,
    compute_Ak_gradient_triangle_mesh_cuda_wrapper
  );
}

} // namespace PoIntInt
```

### 3.3 Register at Startup

```cpp
#include "dof/cuda/triangle_mesh_dof_cuda.hpp"

int main() {
  PoIntInt::register_triangle_mesh_dof_cuda_kernels();
  // ...
}
```

## Step 4: Refactoring the Main Gradient Function

The main gradient function (`compute_intersection_volume_gradient_cuda`) should be refactored to:

1. Get DoF type names
2. Check registry for CUDA kernels
3. Use CUDA if available, otherwise fall back to CPU

Here's the pattern:

```cpp
IntersectionVolumeGradientResult compute_intersection_volume_gradient_cuda(...) {
  // Get DoF type names
  std::string dof1_type = get_dof_type_name(dof1);
  std::string dof2_type = get_dof_type_name(dof2);
  
  // Check if CUDA kernels are available
  bool has_cuda1 = CudaKernelRegistry::has_kernels(dof1_type, geom1.type);
  bool has_cuda2 = CudaKernelRegistry::has_kernels(dof2_type, geom2.type);
  
  if (has_cuda1 && has_cuda2) {
    // Use CUDA path with registered kernels
    auto [compute_Ak1, compute_grad_Ak1] = CudaKernelRegistry::get_kernels(dof1_type, geom1.type);
    auto [compute_Ak2, compute_grad_Ak2] = CudaKernelRegistry::get_kernels(dof2_type, geom2.type);
    
    // Call wrapper functions to compute A(k) and gradients
    // ... (similar to current implementation but using function pointers)
  } else {
    // Fall back to CPU computation
    // Use dof1->compute_A_gradient() and dof2->compute_A_gradient()
  }
}
```

## Benefits

1. **Modular**: Each DoF type manages its own CUDA kernels
2. **Extensible**: Add new combinations without modifying core code
3. **Backward Compatible**: Existing CPU implementations continue to work
4. **Performance**: Specialized kernels for common cases, CPU fallback for others

## Current Status

- ✅ Registry system implemented
- ✅ AffineDoF CUDA wrappers created
- ⏳ Main gradient function needs refactoring to use registry
- ⏳ Need to register kernels at startup in tests/main

## Next Steps

1. Refactor `compute_intersection_volume_gradient_cuda` to use the registry
2. Add kernel registration calls in test initialization
3. Test with AffineDoF to ensure backward compatibility
4. Add TriangleMeshDoF CUDA support as an example


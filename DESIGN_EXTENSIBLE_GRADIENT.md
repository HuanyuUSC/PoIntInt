# Extensible CUDA Gradient Computation Design

## Problem Statement

Currently, `compute_intersection_volume_gradient_cuda` only supports:
- `AffineDoF` with triangle meshes

We want to support:
- Multiple DoF types: `AffineDoF`, `TriangleMeshDoF`, future reduced-order model DoFs
- Multiple geometry types: triangle meshes, point clouds (disks), Gaussian splats
- Keep CUDA performance for specialized cases
- Allow easy extension without modifying core gradient code

## Design Solution: Registry + Dispatch Pattern

### Key Components

1. **CUDA Kernel Registry**: Maps (DoF type, Geometry type) → CUDA kernel functions
2. **DoF CUDA Interface**: Optional interface for DoF classes to provide CUDA kernels
3. **Hybrid Computation**: Use CUDA when available, fall back to CPU otherwise
4. **Modular Kernel Organization**: Each DoF type provides its own CUDA kernels

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  compute_intersection_volume_gradient_cuda (main entry)    │
│  - Checks registry for CUDA kernels                         │
│  - If available: use CUDA                                   │
│  - If not: fall back to CPU (using DoF::compute_A_gradient) │
└─────────────────────────────────────────────────────────────┘
                          │
                          ├─────────────────┐
                          │                 │
                          ▼                 ▼
        ┌─────────────────────────┐  ┌──────────────────────┐
        │  CUDA Kernel Registry    │  │  CPU Fallback        │
        │  - AffineDoF_Triangle    │  │  - DoF::compute_A   │
        │  - AffineDoF_Disk        │  │  - DoF::compute_A_  │
        │  - TriangleMeshDoF_Tri   │  │    gradient         │
        │  - ...                   │  └──────────────────────┘
        └─────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │  DoF-Specific CUDA Kernels          │
        │  - AffineDoF: affine kernels        │
        │  - TriangleMeshDoF: vertex kernels │
        │  - ...                              │
        └─────────────────────────────────────┘
```

### Implementation Strategy

#### 1. Registry Interface

```cpp
// Key: "DoFType_GeometryType" (e.g., "AffineDoF_Triangle")
class CudaKernelRegistry {
  // Register kernels for a DoF+geometry combination
  static void register_kernels(
    const std::string& dof_type,
    GeometryType geom_type,
    CudaComputeAkFunc compute_Ak,
    CudaComputeAkGradientFunc compute_Ak_gradient
  );
  
  // Check if kernels are available
  static bool has_kernels(const std::string& dof_type, GeometryType geom_type);
  
  // Get kernel functions
  static std::pair<CudaComputeAkFunc, CudaComputeAkGradientFunc>
    get_kernels(const std::string& dof_type, GeometryType geom_type);
};
```

#### 2. DoF CUDA Interface (Optional)

```cpp
struct DoFCudaInterface {
  // Get DoF type name
  virtual std::string dof_type_name() const = 0;
  
  // Check if CUDA kernels are available for a geometry type
  virtual bool supports_cuda(GeometryType geom_type) const = 0;
  
  // Register CUDA kernels (called once per DoF type)
  virtual void register_cuda_kernels() const = 0;
};
```

#### 3. Wrapper Functions

Each DoF type provides wrapper functions that:
- Handle memory allocation/copying
- Launch appropriate CUDA kernels
- Match the registry function signature

Example for AffineDoF:
```cpp
void compute_Ak_affine_triangle_cuda_wrapper(
  const Geometry& geom,
  const KGrid& kgrid,
  const Eigen::VectorXd& dofs,
  double2* d_A_out,
  int blockSize
) {
  // Allocate device memory
  // Copy geometry and DoF data
  // Launch compute_A_affine_triangle_kernel
  // Handle errors
}
```

#### 4. Main Gradient Function (Refactored)

```cpp
IntersectionVolumeGradientResult compute_intersection_volume_gradient_cuda(...) {
  // 1. Get DoF type names
  std::string dof1_type = get_dof_type_name(dof1);
  std::string dof2_type = get_dof_type_name(dof2);
  
  // 2. Check if CUDA kernels are available
  bool has_cuda1 = CudaKernelRegistry::has_kernels(dof1_type, geom1.type);
  bool has_cuda2 = CudaKernelRegistry::has_kernels(dof2_type, geom2.type);
  
  // 3. If both have CUDA kernels, use CUDA path
  if (has_cuda1 && has_cuda2) {
    return compute_gradient_cuda_path(geom1, geom2, dof1, dof2, dofs1, dofs2, kgrid, ...);
  }
  
  // 4. Otherwise, use hybrid or CPU fallback
  return compute_gradient_hybrid(geom1, geom2, dof1, dof2, dofs1, dofs2, kgrid, ...);
}
```

### Benefits

1. **Modularity**: Each DoF type manages its own CUDA kernels
2. **Extensibility**: Add new DoF+geometry combinations without modifying core code
3. **Performance**: Specialized kernels for common cases, CPU fallback for others
4. **Backward Compatibility**: Existing code continues to work
5. **Gradual Migration**: Can add CUDA support incrementally

### Example: Adding TriangleMeshDoF Support

1. Create CUDA kernels for TriangleMeshDoF + triangle mesh
2. Create wrapper functions
3. Register kernels in static initializer:
```cpp
void register_triangle_mesh_dof_cuda_kernels() {
  CudaKernelRegistry::register_kernels(
    "TriangleMeshDoF",
    GEOM_TRIANGLE,
    compute_Ak_triangle_mesh_cuda_wrapper,
    compute_Ak_gradient_triangle_mesh_cuda_wrapper
  );
}
```

### File Organization

```
include/dof/
  - dof_parameterization.hpp (existing)
  - dof_cuda_interface.hpp (new - registry interface)
  - affine_dof.hpp (existing)
  - affine_dof_cuda.hpp (new - CUDA wrappers for AffineDoF)
  - triangle_mesh_dof.hpp (existing)
  - triangle_mesh_dof_cuda.hpp (new - CUDA wrappers for TriangleMeshDoF)

src/dof/
  - dof_cuda_interface.cpp (new - registry implementation)
  - affine_dof_cuda.cpp (new - wrapper implementations)
  - triangle_mesh_dof_cuda.cpp (new - wrapper implementations)

src/
  - compute_intersection_volume_gradient.cu (refactored to use registry)
```


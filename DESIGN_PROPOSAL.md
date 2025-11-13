# Design Proposal: Extensible Multi-Object Intersection Volume with Gradients

## Overview

This document proposes a design to extend the current two-object intersection volume computation to support:
1. **Multiple objects** (N-way intersection)
2. **Multiple geometry types** (triangles, disks, Gaussian splats)
3. **Gradient computation** with respect to degrees of freedom (DoFs)
4. **Flexible DoF parameterizations** (affine transforms, nonlinear deformations)

## Core Design Principles

1. **Separation of Concerns**: Geometry representation, DoF parameterization, and computation are separate
2. **Performance**: Maintain CUDA kernel efficiency with minimal overhead
3. **Extensibility**: Easy to add new geometry types and DoF types
4. **Type Safety**: Use strong typing to prevent errors

## Proposed Architecture

### 1. Geometry Representation Layer

```cpp
namespace PoIntInt {

// Base geometry handle - type-erased container
struct Geometry {
  GeometryType type;
  std::vector<TriPacked> tris;
  std::vector<DiskPacked> disks;
  std::vector<GaussianPacked> gaussians;  // Future: Gaussian splats
  
  // Metadata
  int num_elements() const;
  bool is_empty() const;
};

// Factory functions for creating geometries
Geometry make_triangle_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
Geometry make_point_cloud(const Eigen::MatrixXd& P, const Eigen::MatrixXd& N, 
                          const Eigen::VectorXd& radii_or_areas, bool is_radius = true);
// Future: Geometry make_gaussian_splat(...);
```

### 2. DoF Parameterization Layer

```cpp
// Abstract base for DoF parameterizations
struct DoFParameterization {
  virtual ~DoFParameterization() = default;
  
  // Number of DoFs
  virtual int num_dofs() const = 0;
  
  // Apply transformation to geometry (returns transformed geometry)
  virtual Geometry apply(const Geometry& geom, const Eigen::VectorXd& dofs) const = 0;
  
  // Compute gradient of form factor A(k) w.r.t. DoFs
  // Returns: dA/dθ for each DoF
  virtual std::vector<std::complex<double>> 
    compute_A_gradient(const Geometry& geom, const Eigen::Vector3d& k, 
                      const Eigen::VectorXd& dofs) const = 0;
};

// Affine transformation DoFs: [translation(3), rotation(3), scale(3), shear(3)] = 12 DoFs
struct AffineDoF : public DoFParameterization {
  int num_dofs() const override { return 12; }
  Geometry apply(const Geometry& geom, const Eigen::VectorXd& dofs) const override;
  std::vector<std::complex<double>> 
    compute_A_gradient(const Geometry& geom, const Eigen::Vector3d& k,
                      const Eigen::VectorXd& dofs) const override;
};

// Nonlinear deformation DoFs (e.g., control points, basis functions)
struct NonlinearDoF : public DoFParameterization {
  // Implementation depends on specific deformation model
};
```

### 3. Multi-Object Interface

```cpp
// Main interface for N-object intersection
struct IntersectionVolumeResult {
  double volume;
  std::vector<Eigen::VectorXd> gradients;  // One per object
};

IntersectionVolumeResult compute_intersection_volume_cuda(
  const std::vector<Geometry>& geometries,
  const std::vector<std::shared_ptr<DoFParameterization>>& dof_params,
  const std::vector<Eigen::VectorXd>& dof_values,  // Current DoF values
  const KGrid& kgrid,
  bool compute_gradients = false,
  int blockSize = 256
);

// Convenience overloads
double compute_intersection_volume_cuda(
  const std::vector<Geometry>& geometries,
  const KGrid& kgrid,
  int blockSize = 256
);
```

### 4. Kernel Design for Multiple Objects

The kernel needs to compute:
```
V = (1/(8π³)) ∫ ∏ᵢ |Aᵢ(k)|² dk
```

For N objects, we can:
- **Option A**: Compute all Aᵢ(k) in parallel, then multiply (good for small N)
- **Option B**: Use a reduction tree to compute the product (good for large N)

```cpp
// Kernel signature for N objects
__global__ void accumulate_intersection_volume_kernel(
  const GeometryData* geometries,  // Array of geometry data
  int num_geometries,
  const float3* kdirs,
  const double* weights_k,
  const float* kmags,
  int Q,
  double* out_scalar
);
```

### 5. Gradient Computation Strategy

For gradient computation, we need:
```
∂V/∂θᵢ = (1/(8π³)) ∫ [∂|Aᵢ(k)|²/∂θᵢ] ∏ⱼ≠ᵢ |Aⱼ(k)|² dk
```

**Approach**:
1. Compute all Aᵢ(k) in first pass
2. For each object i, compute ∂Aᵢ/∂θᵢ in second pass
3. Accumulate gradient contributions

This can be done efficiently by:
- Reusing computed Aᵢ(k) values
- Computing gradients only for objects that need them
- Using automatic differentiation or manual derivatives

## Implementation Plan

### Phase 1: Multi-Object Support (No Gradients)
1. Create `Geometry` struct to hold different geometry types
2. Refactor kernel to support N objects
3. Update interface to take `std::vector<Geometry>`
4. Maintain backward compatibility with 2-object version

### Phase 2: Add Gaussian Splat Support
1. Define `GaussianPacked` struct
2. Implement Gaussian form factor computation
3. Add to kernel dispatch logic

### Phase 3: DoF Abstraction
1. Create `DoFParameterization` base class
2. Implement `AffineDoF`
3. Add transformation application logic

### Phase 4: Gradient Computation
1. Implement gradient computation kernels
2. Add gradient accumulation logic
3. Optimize for performance

## File Organization

```
include/
  geometry_types.hpp          // TriPacked, DiskPacked, GaussianPacked, GeometryType
  geometry_packing.hpp         // Factory functions for creating geometries
  geometry.hpp                // Geometry struct (NEW)
  dof_parameterization.hpp    // DoF base class and implementations (NEW)
  compute_volume.hpp           // Main interface (UPDATED)
  
src/
  geometry_packing.cpp         // Existing packing functions
  geometry.cpp                 // Geometry implementation (NEW)
  dof_parameterization.cpp      // DoF implementations (NEW)
  compute_volume.cu            // CUDA kernels and host code (UPDATED)
```

## Example Usage

```cpp
using namespace PoIntInt;

// Create geometries
auto geom1 = make_triangle_mesh(V1, F1);
auto geom2 = make_point_cloud(P2, N2, radii2);
auto geom3 = make_gaussian_splat(...);  // Future

// Set up DoFs
auto affine_dof = std::make_shared<AffineDoF>();
Eigen::VectorXd dofs1(12);  // translation, rotation, scale, shear
dofs1 << 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0;  // Identity

// Compute intersection volume with gradients
auto result = compute_intersection_volume_cuda(
  {geom1, geom2, geom3},
  {affine_dof, affine_dof, affine_dof},
  {dofs1, dofs2, dofs3},
  kgrid,
  true  // compute gradients
);

std::cout << "Volume: " << result.volume << std::endl;
std::cout << "Gradient for object 1: " << result.gradients[0].transpose() << std::endl;
```

## Benefits of This Design

1. **Extensible**: Easy to add new geometry types (just add to enum and kernel)
2. **Flexible**: DoF parameterization is pluggable
3. **Performant**: Minimal overhead, CUDA kernels remain efficient
4. **Type-safe**: Strong typing prevents errors
5. **Backward compatible**: Can keep existing 2-object interface

## Considerations

1. **Memory**: N objects means N× memory for geometry data
2. **Kernel complexity**: N-way product computation needs careful design
3. **Gradient accuracy**: Need to ensure numerical stability
4. **DoF size**: Different objects may have different DoF counts


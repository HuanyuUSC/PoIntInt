# Design Proposal: Extensible Multi-Object Intersection Volume with Gradients

## Document Structure

This document provides the high-level design and architecture for the poIntInt library. For implementation details and practical guides, see:
- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)**: Step-by-step guide for implementing and extending the CUDA gradient system
- **[DESIGN_EXTENSIBLE_GRADIENT.md](DESIGN_EXTENSIBLE_GRADIENT.md)**: Detailed design of the extensible CUDA gradient computation system (now integrated into this document)

## Overview

This document proposes a design to extend the current two-object intersection volume computation to support:
1. **Multiple objects** with pairwise intersection volumes (nObj × nObj symmetric matrix)
2. **Multiple geometry types** (triangles, disks, Gaussian splats)
3. **Sparse gradient computation** with respect to degrees of freedom (DoFs)
4. **Flexible DoF parameterizations** (affine transforms, nonlinear deformations)
5. **Efficient matrix-based computation** using J^T D J formulation

## Core Design Principles

1. **Matrix-Based Computation**: Use J^T D J where J is nK × nObj (form factors) and D is diagonal weights
2. **Sparse Gradients**: Gradients are sparse - (i,j) entry only depends on DoFs of objects i and j
3. **Separation of Concerns**: Geometry representation, DoF parameterization, and computation are separate
4. **Performance**: Maintain CUDA kernel efficiency with minimal overhead
5. **Extensibility**: Easy to add new geometry types and DoF types
6. **Backward Compatibility**: Keep 2-object scalar interface for testing and convenience

## Mathematical Foundation

### Matrix Formulation

For N objects, we compute the **pairwise intersection volume matrix** V ∈ ℝ^(N×N):

```
V[i,j] = (1/(8π³)) ∫ A_i(k) · A_j*(k) · w(k) dk
```

Where:
- A_i(k) is the scalar form factor of object i at k-vector k
- w(k) is the weight for k-node
- The integral is discretized over k-grid

**Matrix computation**:
```
J ∈ ℂ^(nK × nObj):  J[q, i] = A_i(k_q)  (form factor matrix)
D ∈ ℝ^(nK × nK):    D = diag(w_0, w_1, ..., w_{nK-1})  (weight matrix)
V = (1/(8π³)) · Re(J^T D J)  (symmetric real matrix)
```

### Gradient Structure

The gradient of V with respect to DoFs is **sparse**:
```
∂V[i,j]/∂θ_k = 0  if k ≠ i and k ≠ j
```

Only the DoFs of objects i and j affect entry (i,j):
```
∂V[i,j]/∂θ_i = (1/(8π³)) · Re( (∂J[:,i]/∂θ_i)^T D J[:,j] )
∂V[i,j]/∂θ_j = (1/(8π³)) · Re( J[:,i]^T D (∂J[:,j]/∂θ_j) )
```

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

**Key Architectural Change**: DoF parameterizations do NOT transform geometries directly. Instead, they provide methods to compute form factors A(k) and their gradients for a reference geometry under a given DoF configuration. This is because applying an affine transform to disks or Gaussians does not preserve their shape (they become elliptical).

```cpp
// Abstract base for DoF parameterizations
struct DoFParameterization {
  virtual ~DoFParameterization() = default;
  
  // Number of DoFs
  virtual int num_dofs() const = 0;
  
  // Compute form factor A(k) for reference geometry under DoF configuration
  // This is the fundamental operation - no geometry transformation needed
  virtual std::complex<double> 
    compute_A(const Geometry& ref_geom, const Eigen::Vector3d& k, 
              const Eigen::VectorXd& dofs) const = 0;
  
  // Compute gradient of form factor A(k) w.r.t. DoFs
  // Returns: dA/dθ for each DoF (complex vector of size num_dofs)
  virtual Eigen::VectorXcd 
    compute_A_gradient(const Geometry& ref_geom, const Eigen::Vector3d& k, 
                      const Eigen::VectorXd& dofs) const = 0;
  
  // Optional: Compute full Hessian (second derivatives) of A(k) w.r.t. DoFs
  // Returns: d²A/dθ² (complex matrix of size num_dofs × num_dofs)
  // Note: This is NOT required for Gauss-Newton approximation of volume Hessian,
  // which uses only first derivatives: H ≈ Σ_q w_q · (∂A₁/∂θ₁) · (∂A₂/∂θ₂)^T
  // This method is provided for future improvements (e.g., full Newton optimization)
  // Default implementation returns zero matrix
  virtual Eigen::MatrixXcd 
    compute_A_hessian(const Geometry& ref_geom, const Eigen::Vector3d& k,
                     const Eigen::VectorXd& dofs) const {
    return Eigen::MatrixXcd::Zero(num_dofs(), num_dofs());
  }
};

// Affine transformation DoFs: [translation(3), matrix(9)] = 12 DoFs
// Matrix is stored row-major: [A00, A01, A02, A10, A11, A12, A20, A21, A22]
struct AffineDoF : public DoFParameterization {
  int num_dofs() const override { return 12; }
  
  // Compute A(k) for reference geometry under affine transformation
  // For triangles: transforms vertices and computes form factor
  // For disks: transforms center/normal and accounts for scaling of radius/area
  // For Gaussians: transforms center/normal and accounts for scaling of sigma/weight
  std::complex<double> 
    compute_A(const Geometry& ref_geom, const Eigen::Vector3d& k,
              const Eigen::VectorXd& dofs) const override;
  
  Eigen::VectorXcd 
    compute_A_gradient(const Geometry& ref_geom, const Eigen::Vector3d& k,
                      const Eigen::VectorXd& dofs) const override;
  
  // Optional: Full Hessian computation (for future improvements, not required for Gauss-Newton)
  Eigen::MatrixXcd 
    compute_A_hessian(const Geometry& ref_geom, const Eigen::Vector3d& k,
                     const Eigen::VectorXd& dofs) const override;
};

// Nonlinear deformation DoFs (e.g., control points, basis functions)
struct NonlinearDoF : public DoFParameterization {
  // Implementation depends on specific deformation model
  // Same interface: compute_A, compute_A_gradient, compute_A_hessian
};
```

### 3. Unified Computation Interface

**Key Architectural Change**: All volume computation routines now take reference geometries and DoF parameterizations. The old "apply then compute" approach is replaced with "compute with DoF" approach.

```cpp
// Computation flags - specify what to compute
enum class ComputationFlags {
  VOLUME_ONLY = 0x01,
  GRADIENT = 0x02,
  HESSIAN = 0x04,
  ALL = VOLUME_ONLY | GRADIENT | HESSIAN
};

// Result structure for unified computation (used for both intersection and self-volume)
struct VolumeResult {
  double volume;  // Volume (intersection volume or self-volume)
  
  // Gradients (only computed if requested)
  Eigen::VectorXd grad_geom1;  // Gradient w.r.t. DoFs of geometry 1 (or single geometry for self-volume)
  Eigen::VectorXd grad_geom2;  // Gradient w.r.t. DoFs of geometry 2 (empty for self-volume)
  
  // Hessians (only computed if requested, Gauss-Newton approximation)
  // Note: Gauss-Newton Hessian is approximated as sum of outer products of gradients:
  //   H ≈ Σ_q w_q · (∂A₁(k_q)/∂θ₁) · (∂A₂(k_q)/∂θ₂)^T
  // This does NOT require second derivatives d²A/dθ²
  Eigen::MatrixXd hessian_geom1;  // Hessian w.r.t. DoFs of geometry 1 (or single geometry for self-volume)
  Eigen::MatrixXd hessian_geom2;  // Hessian w.r.t. DoFs of geometry 2 (empty for self-volume)
  Eigen::MatrixXd hessian_cross;  // Cross-term Hessian (∂²V/∂θ₁∂θ₂, empty for self-volume)
  
  // Optional: Future support for full Hessian (requires d²A/dθ²)
  // This is left as a placeholder for future improvements
  // Eigen::MatrixXd hessian_full_geom1;  // Full Hessian if compute_A_hessian() is implemented
};

// Unified scalar intersection volume interface (with DoF)
VolumeResult compute_intersection_volume_unified_cuda(
  const Geometry& ref_geom1,  // Reference geometry 1
  const Geometry& ref_geom2,  // Reference geometry 2
  const std::shared_ptr<DoFParameterization>& dof1,  // DoF for geometry 1
  const std::shared_ptr<DoFParameterization>& dof2,  // DoF for geometry 2
  const Eigen::VectorXd& dofs1,  // DoF values for geometry 1
  const Eigen::VectorXd& dofs2,  // DoF values for geometry 2
  const KGrid& kgrid,
  ComputationFlags flags = ComputationFlags::VOLUME_ONLY,
  int blockSize = 256,
  bool enable_profiling = false
);

// Convenience: volume only (backward compatible with gradient interface)
double compute_intersection_volume_cuda(
  const Geometry& ref_geom1,
  const Geometry& ref_geom2,
  const std::shared_ptr<DoFParameterization>& dof1,
  const std::shared_ptr<DoFParameterization>& dof2,
  const Eigen::VectorXd& dofs1,
  const Eigen::VectorXd& dofs2,
  const KGrid& kgrid,
  int blockSize = 256,
  bool enable_profiling = false
);

// Convenience: no DoF (uses identity AffineDoF internally)
double compute_intersection_volume_cuda(
  const Geometry& geom1,
  const Geometry& geom2,
  const KGrid& kgrid,
  int blockSize = 256,
  bool enable_profiling = false
);

// Multi-object result structure
struct IntersectionVolumeMatrixResult {
  Eigen::MatrixXd volume_matrix;  // nObj × nObj symmetric matrix
  
  // Sparse gradient storage: gradients[i][j] contains gradient of V[i,j]
  // Only stored for pairs (i,j) where gradients were requested
  // gradients[i][j] is a vector of size (dof_i + dof_j)
  // Format: [∂V[i,j]/∂θ_i, ∂V[i,j]/∂θ_j]
  std::vector<std::vector<Eigen::VectorXd>> gradients;
  
  // Sparse Hessian storage (if requested)
  std::vector<std::vector<Eigen::MatrixXd>> hessians;
  
  // Metadata
  std::vector<int> dof_counts;  // Number of DoFs per object
};

// Unified multi-object interface
IntersectionVolumeMatrixResult compute_intersection_volume_matrix_unified_cuda(
  const std::vector<Geometry>& ref_geometries,  // Reference geometries
  const std::vector<std::shared_ptr<DoFParameterization>>& dof_params,
  const std::vector<Eigen::VectorXd>& dof_values,
  const KGrid& kgrid,
  ComputationFlags flags = ComputationFlags::VOLUME_ONLY,
  const std::vector<std::pair<int, int>>& gradient_pairs = {},  // Pairs to compute gradients for
  int blockSize = 256,
  bool enable_profiling = false
);

// Convenience: no DoF (uses identity AffineDoF for all objects)
IntersectionVolumeMatrixResult compute_intersection_volume_matrix_cuda(
  const std::vector<Geometry>& geometries,
  const KGrid& kgrid,
  int blockSize = 256,
  bool enable_profiling = false
);

// Self-volume computation (divergence theorem) - returns VolumeResult with gradient/Hessian support
VolumeResult compute_volume_unified_cuda(
  const Geometry& ref_geom,
  const std::shared_ptr<DoFParameterization>& dof,
  const Eigen::VectorXd& dofs,
  const KGrid& kgrid,
  ComputationFlags flags = ComputationFlags::VOLUME_ONLY,
  int blockSize = 256,
  bool enable_profiling = false
);

// Convenience: no DoF (uses identity AffineDoF)
double compute_volume_cuda(
  const Geometry& geom,
  const KGrid& kgrid,
  int blockSize = 256,
  bool enable_profiling = false
);

// CPU/TBB versions (same interface, different suffix)
VolumeResult compute_intersection_volume_unified_cpu(
  const Geometry& ref_geom1,
  const Geometry& ref_geom2,
  const std::shared_ptr<DoFParameterization>& dof1,
  const std::shared_ptr<DoFParameterization>& dof2,
  const Eigen::VectorXd& dofs1,
  const Eigen::VectorXd& dofs2,
  const KGrid& kgrid,
  ComputationFlags flags = ComputationFlags::VOLUME_ONLY,
  bool enable_profiling = false
);

VolumeResult compute_volume_unified_cpu(
  const Geometry& ref_geom,
  const std::shared_ptr<DoFParameterization>& dof,
  const Eigen::VectorXd& dofs,
  const KGrid& kgrid,
  ComputationFlags flags = ComputationFlags::VOLUME_ONLY,
  bool enable_profiling = false
);
```

### 4. Kernel Design for Matrix Computation

**Phase 1: Compute Form Factor Matrix J**

```cpp
// Kernel to compute J[q, obj] = A_obj(k_q) for all k-nodes and objects
__global__ void compute_form_factor_matrix_kernel(
  const GeometryData* geometries,  // Array of geometry data
  int num_geometries,
  const float3* kdirs,
  const float* kmags,
  int Q,
  float2* J  // Output: Q × num_geometries complex matrix (row-major)
);
```

**Phase 2: Compute Volume Matrix V = J^T D J**

```cpp
// Kernel to compute V = Re(J^T D J)
// Can be done efficiently using matrix multiplication
__global__ void compute_volume_matrix_kernel(
  const float2* J,           // Input: Q × num_geometries
  const double* weights,     // Diagonal of D (size Q)
  int Q,
  int num_geometries,
  double* V                  // Output: num_geometries × num_geometries (symmetric)
);
```

**Phase 3: Compute Gradients (Sparse)**

```cpp
// Kernel to compute gradient for specific pair (i, j)
__global__ void compute_pairwise_gradient_kernel(
  const GeometryData* geom_i,
  const GeometryData* geom_j,
  const float2* J,           // Precomputed form factors
  const double* weights,
  const float3* kdirs,
  const float* kmags,
  int Q,
  int dof_i,                // Number of DoFs for object i
  int dof_j,                 // Number of DoFs for object j
  double* grad_ij            // Output: (dof_i + dof_j) vector
);
```

### 5. Efficient Gradient Storage and Querying

```cpp
// Sparse gradient storage structure
struct SparseGradientMatrix {
  int num_objects;
  std::vector<int> dof_counts;  // DoF count per object
  
  // Storage: gradients_[i * num_objects + j] contains gradient for pair (i,j)
  // Only allocated for requested pairs
  std::unordered_map<int, Eigen::VectorXd> gradients_;
  
  // Get gradient for pair (i, j)
  // Returns: [∂V[i,j]/∂θ_i, ∂V[i,j]/∂θ_j] or empty if not computed
  Eigen::VectorXd get_gradient(int i, int j) const;
  
  // Get gradient w.r.t. object i's DoFs only (sum over all pairs involving i)
  Eigen::VectorXd get_object_gradient(int i) const;
  
  // Get gradient w.r.t. specific pair's DoFs
  std::pair<Eigen::VectorXd, Eigen::VectorXd> get_pair_gradients(int i, int j) const;
};

// Integration with result structure
struct IntersectionVolumeMatrixResult {
  Eigen::MatrixXd volume_matrix;
  SparseGradientMatrix gradients;
};
```

## Implementation Status

### ✅ Phase 1: Matrix-Based Multi-Object Support (COMPLETED)
1. ✅ Created `Geometry` struct to hold different geometry types (triangles, disks, Gaussians)
2. ✅ Implemented `compute_form_factor_matrix_kernel` to compute J matrix (CUDA)
3. ✅ Implemented `compute_volume_matrix_kernel` to compute V = J^T D J (CUDA)
4. ✅ Created multi-object interface returning symmetric matrix (CUDA + CPU/TBB)
5. ✅ Maintained backward compatibility with 2-object scalar version
6. ✅ Added CPU version with TBB parallelization
7. ✅ Comprehensive unit tests for multi-object functionality
8. ✅ All computation uses `double` precision (no `float` types)

### ✅ Phase 2: Add Gaussian Splat Support (COMPLETED)
1. ✅ Defined `GaussianPacked` struct with `double` precision
2. ✅ Implemented Gaussian form factor computation in kernel (CUDA)
3. ✅ Added to geometry type enum and kernel dispatch
4. ✅ Added CPU version support with TBB
5. ✅ Unit tests for Gaussian splat intersections (Gaussian-Gaussian, Gaussian-Mesh, Gaussian-Point Cloud)

### ✅ Phase 3: DoF Abstraction (COMPLETED)
1. ✅ Created `DoFParameterization` base class
2. ✅ Implemented `AffineDoF` with gradient computation:
   - CPU version with TBB parallelization
   - CUDA version via registry system (triangles, disks, Gaussians)
3. ✅ Implemented `TriangleMeshDoF` for vertex position DoFs:
   - CPU version with analytical gradients
   - Volume gradient computation
4. ✅ Implemented `compute_volume_gradient` for volume derivatives (CPU)
5. ✅ Comprehensive unit tests with finite difference validation
6. ✅ Analytical gradient formulas for both DoF types
7. ✅ CUDA kernel registry system for extensible gradient computation
8. ✅ Helper functions for form factor computation (`form_factor_helpers.hpp`)
9. ✅ Geometry creation helpers (`geometry_helpers.hpp`)
10. ✅ Analytical intersection volume functions for testing

**Current Limitations:**
- **Architectural Issue**: Current implementation uses `apply()` to transform geometries, which is incorrect for disks/Gaussians (they become elliptical). Need to refactor to unified interface.
- Self-volume gradient computation not yet implemented (only volume, not gradient)
- Multi-object gradient computation not yet implemented (only volume matrix)
- Hessian computation not yet implemented

## Implementation Plan (Remaining)

### Phase 4: Unified Interface Refactoring (CRITICAL)

**Goal**: Refactor all volume computation routines to use the unified interface that takes reference geometries and DoF parameterizations, eliminating the need for `apply()`.

**Problem**: The current `apply()` function transforms geometries directly, but this is incorrect for disks and Gaussians because:
- Applying an affine transform to a disk does not preserve its circular shape (becomes elliptical)
- Applying an affine transform to a Gaussian splat does not preserve its spherical shape (becomes ellipsoidal)
- The form factor computation should account for the transformation directly, not through geometry transformation

**Solution**: 
1. Remove `apply()` from `DoFParameterization` interface
2. All volume computation routines take reference geometries + DoF parameterizations
3. Form factor computation (A(k)) is done directly using the DoF, without transforming the geometry
4. Old interfaces (without DoF) call new interfaces with identity `AffineDoF`

**Implementation Steps**:

1. **Refactor DoFParameterization Interface**:
   - Remove `apply()` method
   - Ensure `compute_A()` and `compute_A_gradient()` work directly with reference geometry + DoFs
   - Update all DoF implementations (AffineDoF, TriangleMeshDoF, etc.)

2. **Refactor Volume Computation Routines**:
   - Create unified `compute_intersection_volume_unified_cuda()` that takes DoF parameterizations
   - Update `compute_intersection_volume_cuda()` to call unified version with identity DoF
   - Update `compute_intersection_volume_gradient_cuda()` to use unified interface
   - Update multi-object routines similarly
   - Update self-volume routines similarly

3. **Refactor CPU/TBB Versions**:
   - Same changes as CUDA versions
   - Ensure consistency between CPU and GPU interfaces

4. **Update Unit Tests**:
   - Remove all uses of `apply()`
   - Update tests to use unified interface
   - Ensure backward compatibility tests still pass

5. **Update Documentation**:
   - Update design proposal (this document)
   - Update implementation guides
   - Update code comments

**Benefits**:
- Correctness: Form factors computed correctly for all geometry types
- Consistency: Single unified interface for all computation types
- Extensibility: Easy to add new DoF types without worrying about geometry transformation
- Performance: No unnecessary geometry copying/transformation

### Phase 5: CUDA-Based Gradient Computation (COMPLETED - needs refactoring)

**Goal**: Implement efficient CUDA kernels for computing gradients of intersection volumes with respect to geometry DoFs.

**Status**: ✅ Implemented for scalar intersection volume gradient with CUDA registry system. See [Extensible CUDA Gradient System](#extensible-cuda-gradient-system) section below.

#### 5.1: Refactor DoF Gradient Interface for CUDA

**Problem**: Current `compute_A_gradient` is CPU-based and not suitable for CUDA kernels. Need to design a CUDA-compatible interface.

**Approach**:
1. **Create CUDA-compatible gradient computation functions**:
   ```cpp
   // Device-side gradient computation for triangles
   __device__ void compute_triangle_A_gradient_cuda(
     const TriPacked& tri,
     const float3& k,
     const float* dof_params,  // DoF parameters (device memory)
     float2* grad_A  // Output: gradient of A(k) w.r.t. DoFs
   );
   
   // Similar for disks and Gaussian splats
   ```

2. **Extend DoFParameterization interface**:
   ```cpp
   struct DoFParameterization {
     // ... existing methods ...
     
     // CUDA-compatible gradient computation
     // Returns device function pointer or kernel configuration
     virtual void setup_cuda_gradient_kernel(
       const Geometry& geom,
       int num_k_points,
       void** kernel_func,  // Output: kernel function pointer
       dim3* grid_size,     // Output: grid configuration
       dim3* block_size     // Output: block configuration
     ) const = 0;
     
     // Or: provide device-side gradient computation helpers
     virtual void compute_A_gradient_cuda(
       const Geometry& geom,
       const KGrid& kgrid,
       float2* d_grad_A,  // Output: Q × num_dofs complex gradients
       cudaStream_t stream = 0
     ) const = 0;
   };
   ```

3. **Implement CUDA kernels for each DoF type**:
   - `AffineDoF`: Implement analytical gradient formulas in CUDA
   - `TriangleMeshDoF`: Implement vertex position gradients in CUDA
   - Reuse existing element functions (`E_func`, `Phi_ab`, etc.) in device code

#### 4.2: Scalar Intersection Volume Gradient

**Goal**: Compute gradient of `V = compute_intersection_volume_cuda(geom1, geom2, ...)` w.r.t. DoFs of both geometries.

**Mathematical Formulation**:
```
∂V/∂θ₁ = (1/(8π³)) · Σ_q w_q · Re( (∂A₁(k_q)/∂θ₁) · conj(A₂(k_q)) )
∂V/∂θ₂ = (1/(8π³)) · Σ_q w_q · Re( A₁(k_q) · conj(∂A₂(k_q)/∂θ₂) )
```

**Gauss-Newton Hessian Approximation**:
```
H ≈ (1/(8π³)) · Σ_q w_q · (∂A₁(k_q)/∂θ₁) · (∂A₂(k_q)/∂θ₂)^T
```
Note: This approximation uses only first derivatives and does NOT require `d²A/dθ²`.

**Implementation**:
1. **Kernel**: `compute_intersection_volume_gradient_kernel`
   ```cpp
   __global__ void compute_intersection_volume_gradient_kernel(
     const GeometryData* geom1,
     const GeometryData* geom2,
     const float3* kdirs,
     const float* kmags,
     const double* weights,
     int Q,
     const float2* d_grad_A1,  // Precomputed: ∂A₁(k_q)/∂θ₁ for all q
     const float2* d_grad_A2,  // Precomputed: ∂A₂(k_q)/∂θ₂ for all q
     const float2* d_A1,       // Precomputed: A₁(k_q) for all q
     const float2* d_A2,       // Precomputed: A₂(k_q) for all q
     int num_dofs1,
     int num_dofs2,
     double* d_grad_V1,        // Output: ∂V/∂θ₁
     double* d_grad_V2          // Output: ∂V/∂θ₂
   );
   ```

2. **Host Interface**:
   ```cpp
   struct IntersectionVolumeGradientResult {
     Eigen::VectorXd grad_geom1;  // Gradient w.r.t. DoFs of geometry 1
     Eigen::VectorXd grad_geom2;  // Gradient w.r.t. DoFs of geometry 2
   };
   
   IntersectionVolumeGradientResult compute_intersection_volume_gradient_cuda(
     const Geometry& geom1,
     const Geometry& geom2,
     const std::shared_ptr<DoFParameterization>& dof1,
     const std::shared_ptr<DoFParameterization>& dof2,
     const Eigen::VectorXd& dofs1,
     const Eigen::VectorXd& dofs2,
     const KGrid& kgrid,
     int blockSize = 256
   );
   ```

#### 4.3: Matrix Intersection Volume Gradient

**Goal**: Compute gradient of volume matrix `V[i,j]` w.r.t. DoFs of objects i and j.

**Mathematical Formulation**:
```
∂V[i,j]/∂θ_i = (1/(8π³)) · Σ_q w_q · Re( (∂J[q,i]/∂θ_i) · conj(J[q,j]) )
∂V[i,j]/∂θ_j = (1/(8π³)) · Σ_q w_q · Re( J[q,i] · conj(∂J[q,j]/∂θ_j) )
```

**Implementation**:
1. **Kernel**: `compute_volume_matrix_gradient_kernel`
   ```cpp
   __global__ void compute_volume_matrix_gradient_kernel(
     const float2* J,              // Precomputed form factor matrix
     const float2* d_grad_J,       // Precomputed: ∂J[q,i]/∂θ for all (q,i)
     const double* weights,
     int Q,
     int num_objects,
     const std::vector<std::pair<int, int>>& gradient_pairs,
     double* d_grad_V              // Output: gradients for requested pairs
   );
   ```

2. **Sparse Storage**:
   - Only compute gradients for requested pairs `(i,j)`
   - Store as `std::unordered_map<std::pair<int,int>, Eigen::VectorXd>`
   - Format: `gradients[(i,j)] = [∂V[i,j]/∂θ_i, ∂V[i,j]/∂θ_j]`

3. **Host Interface**:
   ```cpp
   struct IntersectionVolumeMatrixResult {
     Eigen::MatrixXd volume_matrix;
     SparseGradientMatrix gradients;  // Sparse storage for gradients
   };
   
   IntersectionVolumeMatrixResult compute_intersection_volume_matrix_cuda(
     const std::vector<Geometry>& geometries,
     const std::vector<std::shared_ptr<DoFParameterization>>& dof_params,
     const std::vector<Eigen::VectorXd>& dof_values,
     const KGrid& kgrid,
     const std::vector<std::pair<int, int>>& gradient_pairs = {},
     int blockSize = 256
   );
   ```

#### 4.4: Optimization Strategy

**Performance Considerations**:
1. **Precompute form factors**: Compute `J` matrix once, reuse for gradient computation
2. **Batch gradient computation**: Compute gradients for multiple pairs in parallel
3. **Memory efficiency**: Use shared memory for reduction operations
4. **Stream parallelism**: Overlap gradient computation with other operations

**Implementation Order**:
1. Implement CUDA kernels for `AffineDoF::compute_A_gradient_cuda`
2. Implement scalar intersection volume gradient
3. Extend to matrix version with sparse storage
4. Add unit tests comparing to finite differences
5. Performance profiling and optimization

### Phase 6: Applications

#### 5.1: Geometry Alignment via Similarity Optimization

**Goal**: Align two geometries by optimizing an affine transformation to maximize similarity measures (Ochiai, Jaccard, Dice).

**Similarity Measures** (all involve intersection volume):
- **Ochiai**: `O(A,B) = V(A∩B) / sqrt(V(A) · V(B))`
- **Jaccard**: `J(A,B) = V(A∩B) / V(A∪B) = V(A∩B) / (V(A) + V(B) - V(A∩B))`
- **Dice**: `D(A,B) = 2·V(A∩B) / (V(A) + V(B))`

**Implementation**:
```cpp
struct AlignmentResult {
  Eigen::Matrix3d optimal_transform;
  Eigen::Vector3d optimal_translation;
  double optimal_similarity;
};

AlignmentResult align_geometries(
  const Geometry& fixed_geom,
  const Geometry& moving_geom,
  SimilarityMeasure measure = JACCARD,
  const OptimizationOptions& options = {}
);

// Uses gradient-based optimization (e.g., L-BFGS)
// Requires: compute_intersection_volume_gradient_cuda
```

**Optimization Loop**:
1. Initialize affine transformation DoFs
2. For each iteration:
   - Compute intersection volume and gradient
   - Compute similarity measure and its gradient
   - Update transformation using optimizer
3. Return optimal transformation

#### 5.2: Affine Body Dynamics Simulator

**Goal**: Integrate intersection volume computation into a physics simulator using libigl.

**Integration Points**:
1. **Collision Detection**: Use intersection volume to detect collisions
2. **Contact Forces**: Compute contact forces based on intersection volume gradients
3. **Constraint Satisfaction**: Use gradients for constraint-based dynamics

**Implementation**:
```cpp
// Integration with libigl's physics simulation
class AffineBodyDynamics {
  std::vector<Geometry> bodies;
  std::vector<AffineDoF> body_transforms;
  
  void step(double dt) {
    // 1. Compute pairwise intersection volumes and gradients
    auto volume_matrix = compute_intersection_volume_matrix_cuda(...);
    
    // 2. Compute contact forces from gradients
    for (auto& pair : colliding_pairs) {
      auto grad = volume_matrix.gradients.get_pair_gradients(pair.i, pair.j);
      // Apply forces based on gradient
    }
    
    // 3. Update body positions using libigl's integrator
    // ...
  }
};
```

### Phase 7: Reduced Order Model DoF Parameterization

**Goal**: Add support for deformation maps represented as linear combinations of modes.

**Mathematical Formulation**:
```
x' = x + Σ_i c_i · φ_i(x)
```
Where:
- `c_i` are the DoF coefficients
- `φ_i(x)` are the deformation modes (basis functions)

**Implementation**:
```cpp
struct ReducedOrderDoF : public DoFParameterization {
  std::vector<Eigen::MatrixXd> modes;  // Deformation modes φ_i
  int num_dofs() const override { return (int)modes.size(); }
  
  Geometry apply(const Geometry& geom, const Eigen::VectorXd& coeffs) const override;
  
  // Gradient computation: ∂A(k)/∂c_i
  // Requires computing how deformation modes affect form factors
  Eigen::VectorXcd compute_A_gradient(
    const Geometry& geom,
    const Eigen::Vector3d& k,
    const Eigen::VectorXd& coeffs
  ) const override;
  
  // CUDA version
  void compute_A_gradient_cuda(...) const override;
};
```

**Gradient Derivation**:
- For triangle meshes: Mode affects vertex positions → affects triangle form factors
- For point clouds: Mode affects point positions → affects disk form factors
- For Gaussian splats: Mode affects center positions → affects Gaussian form factors

**Implementation Steps**:
1. Derive analytical gradient formulas for each geometry type
2. Implement CPU version with TBB
3. Implement CUDA version for performance
4. Add unit tests

### Phase 8: Python Bindings and Kaolin Integration

#### 7.1: Python Bindings (pybind11)

**Goal**: Expose core functionality to Python for easier integration and experimentation.

**Key Bindings**:
```python
# Python API
import poIntInt

# Create geometries
geom1 = poIntInt.make_triangle_mesh(V, F)
geom2 = poIntInt.make_point_cloud(P, N, radii)

# Compute intersection volume
kgrid = poIntInt.build_kgrid(lebedev_dirs, lebedev_weights, n_radial)
volume = poIntInt.compute_intersection_volume_cuda(geom1, geom2, kgrid)

# Multi-object computation
geometries = [geom1, geom2, geom3]
volume_matrix = poIntInt.compute_intersection_volume_matrix_cuda(geometries, kgrid)

# With gradients
dof = poIntInt.AffineDoF()
dofs = np.array([...])  # 12 DoFs
result = poIntInt.compute_intersection_volume_gradient_cuda(
    geom1, geom2, dof, dof, dofs, dofs, kgrid
)
```

**Implementation**:
1. Create `python/` directory with pybind11 bindings
2. Bind core data structures (`Geometry`, `KGrid`, etc.)
3. Bind computation functions
4. Bind DoF parameterizations
5. Add NumPy array support for Eigen matrices
6. Create Python package structure

#### 7.2: Kaolin Integration

**Goal**: Integrate with NVIDIA Kaolin's simplicits library for neural implicit surfaces.

**Integration Points**:
1. **Geometry Conversion**: Convert Kaolin simplicits to `Geometry` struct
2. **Gradient Flow**: Use intersection volume gradients in neural network training
3. **Loss Functions**: Implement similarity-based loss functions

**Implementation**:
```python
# Kaolin integration example
import kaolin
import poIntInt

# Convert Kaolin implicit to geometry
def kaolin_to_geometry(implicit_surface, resolution=1000):
    # Extract mesh or point cloud from implicit
    mesh = kaolin.render.mesh.extract_surface(implicit_surface, resolution)
    return poIntInt.make_triangle_mesh(mesh.vertices, mesh.faces)

# Use in training loop
def similarity_loss(predicted_implicit, target_implicit):
    pred_geom = kaolin_to_geometry(predicted_implicit)
    target_geom = kaolin_to_geometry(target_implicit)
    
    volume = poIntInt.compute_intersection_volume_cuda(
        pred_geom, target_geom, kgrid
    )
    
    # Jaccard similarity as loss
    pred_vol = poIntInt.compute_volume_cuda(pred_geom)
    target_vol = poIntInt.compute_volume_cuda(target_geom)
    jaccard = volume / (pred_vol + target_vol - volume)
    
    return 1.0 - jaccard  # Maximize Jaccard = minimize (1 - Jaccard)
```

## Extensible CUDA Gradient System

The gradient computation system uses a registry/dispatch pattern to allow different DoF types to provide their own CUDA kernels without modifying the core gradient computation code. This design is detailed in a separate section for modularity.

**Key Components:**
1. **CUDA Kernel Registry**: Maps (DoF type, Geometry type) → CUDA kernel functions
2. **DoF CUDA Interface**: Optional interface for DoF classes to provide CUDA kernels
3. **Hybrid Computation**: Use CUDA when available, fall back to CPU otherwise
4. **Modular Kernel Organization**: Each DoF type provides its own CUDA kernels

**Current Support:**
- ✅ `AffineDoF` with triangles (CUDA kernels)
- ✅ `AffineDoF` with disks (CUDA kernels)
- ✅ `AffineDoF` with Gaussians (CUDA kernels)
- ✅ Automatic kernel registration via static initializers
- ✅ CPU fallback for unsupported combinations

**Implementation Details:**
- Registry system: `include/dof/cuda/dof_cuda_interface.hpp`
- AffineDoF CUDA kernels: `src/dof/cuda/affine_dof_cuda.cu`
- Shared CUDA helpers: `include/cuda/cuda_helpers.hpp`
- Main gradient function uses registry to dispatch to appropriate kernels

For detailed implementation guide, see [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md).

## File Organization

```
include/
  geometry/
    types.hpp                    # Geometry type enum, packed structures (double precision)
    geometry.hpp                 # Geometry struct, factory functions
    packing.hpp                  # Packing functions (Eigen → CUDA structures)
    geometry_helpers.hpp         # Geometry creation helpers (unit cube, sphere, etc.)
  dof/
    dof_parameterization.hpp    # Base DoF interface
    affine_dof.hpp              # AffineDoF class
    triangle_mesh_dof.hpp        # TriangleMeshDoF class
    cuda/
      dof_cuda_interface.hpp    # CUDA kernel registry interface
      affine_dof_cuda.hpp        # AffineDoF CUDA kernel declarations
    reduced_order_dof.hpp        # FUTURE: Reduced order model DoF
  quadrature/
    kgrid.hpp                    # KGrid structure and build_kgrid
    lebedev_io.hpp               # Lebedev quadrature I/O
    gauss_legendre.hpp           # Gauss-Legendre quadrature
  cuda/
    cuda_helpers.hpp             # Shared CUDA device functions (matrix ops, element functions)
  compute_intersection_volume.hpp
  compute_intersection_volume_multi_object.hpp
  compute_intersection_volume_gradient.hpp
  compute_volume.hpp             # Self-volume computation
  form_factor_helpers.hpp        # Form factor computation helpers, exact formulas
  analytical_intersection.hpp    # Analytical intersection volume functions
  applications/
    geometry_alignment.hpp       # FUTURE: Geometry alignment application
    body_dynamics.hpp            # FUTURE: Body dynamics integration

src/
  geometry/
    packing.cpp                  # Packing implementations (TBB parallelized)
    geometry_helpers.cpp         # Geometry creation implementations (TBB parallelized)
  dof/
    affine_dof.cpp               # AffineDoF CPU implementation (TBB parallelized)
    triangle_mesh_dof.cpp        # TriangleMeshDoF CPU implementation
    cuda/
      dof_cuda_interface.cpp    # CUDA kernel registry implementation
      affine_dof_cuda.cu        # AffineDoF CUDA kernels and wrappers
    reduced_order_dof.cpp        # FUTURE: Reduced order model DoF
  compute_intersection_volume.cu              # Scalar intersection volume (CUDA + CPU/TBB)
  compute_intersection_volume_multi_object.cu # Multi-object intersection volume (CUDA + CPU/TBB)
  compute_intersection_volume_gradient.cu     # Intersection volume gradient (CUDA)
  compute_intersection_volume_gradient_cpu.cpp # Intersection volume gradient (CPU/TBB)
  compute_volume.cu                           # Self-volume computation (CUDA + CPU/TBB)
  form_factor_helpers.cpp                     # Form factor helpers (TBB parallelized)
  analytical_intersection.cpp                 # Analytical intersection functions
  applications/
    geometry_alignment.cpp                    # FUTURE: Geometry alignment
    body_dynamics.cpp                         # FUTURE: Body dynamics

python/
  poIntInt/
    __init__.py                  # FUTURE: Python package
    bindings.cpp                 # FUTURE: pybind11 bindings
    setup.py                     # FUTURE: Package setup

test/
  test_unit_cube.cpp             # Triangle mesh unit tests
  test_sphere_pointcloud.cpp     # Point cloud unit tests
  test_gaussian_splats.cpp       # Gaussian splat unit tests
  test_multi_object.cpp          # Multi-object unit tests
  test_dof_parameterization.cpp  # DoF parameterization unit tests
  test_intersection_volume_gradient.cpp  # Gradient computation unit tests
  test_alignment.cpp             # FUTURE: Alignment application tests
  test_reduced_order.cpp         # FUTURE: Reduced order model tests
```

**Note**: All CUDA and CPU code uses `double` precision throughout. No `float` types are used in computation.

## Timeline and Priorities

### Immediate (Phase 4 - CRITICAL)
1. **Refactor DoFParameterization interface** - Remove `apply()`, ensure `compute_A()` works correctly (1 week)
2. **Refactor volume computation routines** - Unified interface for all routines (2 weeks)
3. **Update CPU/TBB versions** - Ensure consistency (1 week)
4. **Update unit tests** - Remove `apply()` usage, test unified interface (1 week)
5. **Testing and validation** - Ensure correctness for all geometry types (1 week)

### Short-term (Phase 5 - After Phase 4)
1. **Refactor gradient computation** - Update to use unified interface (1 week)
2. **Add Hessian support** - Implement Gauss-Newton approximation (1-2 weeks)
3. **Performance optimization** - Profile and optimize unified interface (1 week)

### Medium-term (Phase 6)
1. **Geometry alignment application** (2-3 weeks)
2. **Body dynamics integration** (2-3 weeks)

### Long-term (Phase 7-8)
1. **Reduced order model DoF** (3-4 weeks)
2. **Python bindings** (2-3 weeks)
3. **Kaolin integration** (2-3 weeks)

## Example Usage

```cpp
using namespace PoIntInt;

// Create reference geometries
auto ref_geom1 = make_triangle_mesh(V1, F1);
auto ref_geom2 = make_point_cloud(P2, N2, radii2);
auto ref_geom3 = make_gaussian_splat(...);

// ============================================================================
// Unified Interface: Volume with DoF
// ============================================================================

// Create DoF parameterizations
auto affine_dof = std::make_shared<AffineDoF>();
Eigen::VectorXd dofs1(12), dofs2(12), dofs3(12);
// ... initialize dofs (translation + matrix) ...

// Compute intersection volume with DoF (unified interface)
auto result = compute_intersection_volume_unified_cuda(
  ref_geom1, ref_geom2,
  affine_dof, affine_dof,
  dofs1, dofs2,
  kgrid,
  ComputationFlags::VOLUME_ONLY
);
std::cout << "Intersection volume: " << result.volume << std::endl;

// Compute volume + gradient
auto result_with_grad = compute_intersection_volume_unified_cuda(
  ref_geom1, ref_geom2,
  affine_dof, affine_dof,
  dofs1, dofs2,
  kgrid,
  ComputationFlags::VOLUME_ONLY | ComputationFlags::GRADIENT
);
std::cout << "Volume: " << result_with_grad.volume << std::endl;
std::cout << "Gradient w.r.t. geom1: " << result_with_grad.grad_geom1.transpose() << std::endl;

// Compute volume + gradient + Hessian (Gauss-Newton approximation)
// Note: Hessian uses only first derivatives: H ≈ Σ_q w_q · (∂A₁/∂θ₁) · (∂A₂/∂θ₂)^T
auto result_full = compute_intersection_volume_unified_cuda(
  ref_geom1, ref_geom2,
  affine_dof, affine_dof,
  dofs1, dofs2,
  kgrid,
  ComputationFlags::ALL
);

// ============================================================================
// Convenience: No DoF (uses identity AffineDoF internally)
// ============================================================================

// Old interface still works - calls unified interface with identity DoF
double vol_12 = compute_intersection_volume_cuda(ref_geom1, ref_geom2, kgrid);
std::cout << "Volume between 1 and 2: " << vol_12 << std::endl;

// ============================================================================
// Multi-Object Interface
// ============================================================================

// Multi-object with DoF
auto matrix_result = compute_intersection_volume_matrix_unified_cuda(
  {ref_geom1, ref_geom2, ref_geom3},
  {affine_dof, affine_dof, affine_dof},
  {dofs1, dofs2, dofs3},
  kgrid,
  ComputationFlags::VOLUME_ONLY | ComputationFlags::GRADIENT,
  {{0, 1}, {1, 2}}  // gradient pairs
);

// Multi-object without DoF (uses identity DoF for all)
auto matrix_result_simple = compute_intersection_volume_matrix_cuda(
  {ref_geom1, ref_geom2, ref_geom3},
  kgrid
);

// ============================================================================
// Self-Volume
// ============================================================================

// Self-volume with DoF (returns VolumeResult with gradient/Hessian support)
auto self_vol_result = compute_volume_unified_cuda(
  ref_geom1,
  affine_dof,
  dofs1,
  kgrid,
  ComputationFlags::VOLUME_ONLY | ComputationFlags::GRADIENT
);
std::cout << "Self-volume: " << self_vol_result.volume << std::endl;
std::cout << "Gradient w.r.t. DoFs: " << self_vol_result.grad_geom1.transpose() << std::endl;

// Self-volume without DoF (convenience wrapper)
double self_vol_simple = compute_volume_cuda(ref_geom1, kgrid);
```

## Benefits of This Design

1. **Matrix-Based**: Efficient J^T D J computation leverages optimized matrix operations
2. **Sparse Gradients**: Only compute/store gradients for requested pairs, saving memory and computation
3. **Extensible**: Easy to add new geometry types (just add to enum and kernel)
4. **Flexible**: DoF parameterization is pluggable, different DoFs per object
5. **Performant**: 
   - Form factor matrix J computed once and reused
   - Matrix multiplication is highly optimized
   - Gradient computation only for requested pairs
6. **Type-safe**: Strong typing prevents errors
7. **Backward compatible**: 2-object scalar interface maintained for testing

## Key Implementation Details

### Memory Layout for J Matrix
- **Storage**: `float2* J` with size `Q × num_geometries` (row-major)
- **Access**: `J[q * num_geometries + obj]` gives A_obj(k_q)
- **CUDA**: Can use shared memory for tiling in matrix multiplication

### Volume Matrix Computation
- **Symmetric**: Only compute upper triangle, copy to lower
- **Formula**: `V[i,j] = (1/(8π³)) · Σ_q w_q · Re(J[q,i] · conj(J[q,j]))`
- **Efficient**: Can use cuBLAS for matrix multiplication if needed

### Gradient Computation Strategy
1. **Precompute J**: Compute form factor matrix once
2. **For each requested pair (i,j)**:
   - Compute ∂J[:,i]/∂θ_i and ∂J[:,j]/∂θ_j
   - Accumulate: `grad_i = Σ_q w_q · Re(∂J[q,i]/∂θ_i · conj(J[q,j]))`
   - Accumulate: `grad_j = Σ_q w_q · Re(J[q,i] · conj(∂J[q,j]/∂θ_j))`
3. **Store sparsely**: Only store gradients for requested pairs

## Considerations

1. **Memory**: 
   - J matrix: Q × num_geometries complex numbers
   - Volume matrix: num_geometries × num_geometries (symmetric, so ~n²/2)
   - Gradients: Only for requested pairs (sparse)

2. **Computation Cost**:
   - Form factor computation: O(Q × num_geometries × elements_per_object)
   - Volume matrix: O(Q × num_geometries²) via J^T D J
   - Gradients: O(Q × num_gradient_pairs × dof_per_pair)

3. **Numerical Stability**: 
   - Use double precision for weights and accumulation
   - Careful handling of complex arithmetic

4. **DoF Heterogeneity**: 
   - Different objects can have different DoF counts
   - Gradient storage handles variable-size vectors per pair


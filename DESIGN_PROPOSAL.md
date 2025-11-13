# Design Proposal: Extensible Multi-Object Intersection Volume with Gradients

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

```cpp
// Abstract base for DoF parameterizations
struct DoFParameterization {
  virtual ~DoFParameterization() = default;
  
  // Number of DoFs
  virtual int num_dofs() const = 0;
  
  // Apply transformation to geometry (returns transformed geometry)
  virtual Geometry apply(const Geometry& geom, const Eigen::VectorXd& dofs) const = 0;
  
  // Compute gradient of form factor A(k) w.r.t. DoFs
  // Returns: dA/dθ for each DoF (complex vector of size num_dofs)
  virtual Eigen::VectorXcd 
    compute_A_gradient(const Geometry& geom, const Eigen::Vector3d& k, 
                      const Eigen::VectorXd& dofs) const = 0;
};

// Affine transformation DoFs: [translation(3), rotation(3), scale(3), shear(3)] = 12 DoFs
struct AffineDoF : public DoFParameterization {
  int num_dofs() const override { return 12; }
  Geometry apply(const Geometry& geom, const Eigen::VectorXd& dofs) const override;
  Eigen::VectorXcd 
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
// Result structure for multi-object computation
struct IntersectionVolumeMatrixResult {
  Eigen::MatrixXd volume_matrix;  // nObj × nObj symmetric matrix
  
  // Sparse gradient storage: gradients[i][j] contains gradient of V[i,j]
  // Only stored for pairs (i,j) where gradients were requested
  // gradients[i][j] is a vector of size (dof_i + dof_j)
  // Format: [∂V[i,j]/∂θ_i, ∂V[i,j]/∂θ_j]
  std::vector<std::vector<Eigen::VectorXd>> gradients;
  
  // Metadata
  std::vector<int> dof_counts;  // Number of DoFs per object
};

// Main multi-object interface
IntersectionVolumeMatrixResult compute_intersection_volume_matrix_cuda(
  const std::vector<Geometry>& geometries,
  const std::vector<std::shared_ptr<DoFParameterization>>& dof_params,
  const std::vector<Eigen::VectorXd>& dof_values,  // Current DoF values
  const KGrid& kgrid,
  const std::vector<std::pair<int, int>>& gradient_pairs = {},  // Pairs to compute gradients for
  int blockSize = 256
);

// Convenience: compute all pairwise gradients
IntersectionVolumeMatrixResult compute_intersection_volume_matrix_cuda(
  const std::vector<Geometry>& geometries,
  const std::vector<std::shared_ptr<DoFParameterization>>& dof_params,
  const std::vector<Eigen::VectorXd>& dof_values,
  const KGrid& kgrid,
  bool compute_all_gradients,  // If true, compute gradients for all pairs
  int blockSize = 256
);

// Convenience: no gradients
IntersectionVolumeMatrixResult compute_intersection_volume_matrix_cuda(
  const std::vector<Geometry>& geometries,
  const KGrid& kgrid,
  int blockSize = 256
);

// Backward compatibility: 2-object scalar interface
double compute_intersection_volume_cuda(
  const Geometry& geom1,
  const Geometry& geom2,
  const KGrid& kgrid,
  int blockSize = 256
);

// Legacy: triangle-triangle (for unit tests)
double compute_intersection_volume_cuda(
  const std::vector<TriPacked>& tris1,
  const std::vector<TriPacked>& tris2,
  const KGrid& KG,
  int blockSize = 256
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

## Implementation Plan

### Phase 1: Matrix-Based Multi-Object Support (No Gradients)
1. Create `Geometry` struct to hold different geometry types
2. Implement `compute_form_factor_matrix_kernel` to compute J matrix
3. Implement `compute_volume_matrix_kernel` to compute V = J^T D J
4. Create multi-object interface returning symmetric matrix
5. Maintain backward compatibility with 2-object scalar version

### Phase 2: Add Gaussian Splat Support
1. Define `GaussianPacked` struct
2. Implement Gaussian form factor computation in kernel
3. Add to geometry type enum and kernel dispatch

### Phase 3: DoF Abstraction
1. Create `DoFParameterization` base class
2. Implement `AffineDoF` with gradient computation
3. Add transformation application logic

### Phase 4: Sparse Gradient Computation
1. Implement `compute_pairwise_gradient_kernel` for sparse gradients
2. Create `SparseGradientMatrix` storage structure
3. Add efficient querying interface
4. Optimize for performance (only compute requested pairs)

## File Organization

```
include/
  geometry_types.hpp          // TriPacked, DiskPacked, GaussianPacked, GeometryType
  geometry_packing.hpp         // Factory functions for creating geometries
  geometry.hpp                // Geometry struct (NEW)
  dof_parameterization.hpp    // DoF base class and implementations (NEW)
  compute_volume.hpp           // Main interface (UPDATED)
  sparse_gradient.hpp          // SparseGradientMatrix storage (NEW)
  
src/
  geometry_packing.cpp         // Existing packing functions
  geometry.cpp                 // Geometry implementation (NEW)
  dof_parameterization.cpp     // DoF implementations (NEW)
  sparse_gradient.cpp          // SparseGradientMatrix implementation (NEW)
  compute_volume.cu            // CUDA kernels and host code (UPDATED)
```

## Example Usage

```cpp
using namespace PoIntInt;

// Create geometries
auto geom1 = make_triangle_mesh(V1, F1);
auto geom2 = make_point_cloud(P2, N2, radii2);
auto geom3 = make_gaussian_splat(...);  // Future

// Compute pairwise intersection volume matrix (no gradients)
auto result = compute_intersection_volume_matrix_cuda(
  {geom1, geom2, geom3},
  kgrid
);

std::cout << "Volume matrix:\n" << result.volume_matrix << std::endl;
std::cout << "Volume between object 1 and 2: " << result.volume_matrix(0, 1) << std::endl;

// With gradients for specific pairs
auto affine_dof = std::make_shared<AffineDoF>();
Eigen::VectorXd dofs1(12), dofs2(12), dofs3(12);
// ... initialize dofs ...

// Compute gradients only for pairs (0,1) and (1,2)
auto result_with_grad = compute_intersection_volume_matrix_cuda(
  {geom1, geom2, geom3},
  {affine_dof, affine_dof, affine_dof},
  {dofs1, dofs2, dofs3},
  kgrid,
  {{0, 1}, {1, 2}}  // gradient pairs
);

// Query gradients
auto grad_01 = result_with_grad.gradients.get_pair_gradients(0, 1);
std::cout << "∂V[0,1]/∂θ_0: " << grad_01.first.transpose() << std::endl;
std::cout << "∂V[0,1]/∂θ_1: " << grad_01.second.transpose() << std::endl;

// Get total gradient w.r.t. object 1 (sum over all pairs involving object 1)
auto total_grad_1 = result_with_grad.gradients.get_object_gradient(1);

// Backward compatibility: 2-object scalar interface
double vol_12 = compute_intersection_volume_cuda(geom1, geom2, kgrid);
std::cout << "Volume between 1 and 2: " << vol_12 << std::endl;
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


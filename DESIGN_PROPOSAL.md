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

## Implementation Status

### ✅ Phase 1: Matrix-Based Multi-Object Support (COMPLETED)
1. ✅ Created `Geometry` struct to hold different geometry types
2. ✅ Implemented `compute_form_factor_matrix_kernel` to compute J matrix
3. ✅ Implemented `compute_volume_matrix_kernel` to compute V = J^T D J
4. ✅ Created multi-object interface returning symmetric matrix
5. ✅ Maintained backward compatibility with 2-object scalar version
6. ✅ Added CPU version with TBB parallelization
7. ✅ Comprehensive unit tests for multi-object functionality

### ✅ Phase 2: Add Gaussian Splat Support (COMPLETED)
1. ✅ Defined `GaussianPacked` struct
2. ✅ Implemented Gaussian form factor computation in kernel
3. ✅ Added to geometry type enum and kernel dispatch
4. ✅ Added CPU version support
5. ✅ Unit tests for Gaussian splat intersections

### ✅ Phase 3: DoF Abstraction (COMPLETED)
1. ✅ Created `DoFParameterization` base class
2. ✅ Implemented `AffineDoF` with gradient computation (CPU-based)
3. ✅ Implemented `TriangleMeshDoF` for vertex position DoFs
4. ✅ Added transformation application logic
5. ✅ Implemented `compute_volume_gradient` for volume derivatives
6. ✅ Comprehensive unit tests with finite difference validation
7. ✅ Analytical gradient formulas for both DoF types

**Current Limitations:**
- `compute_A_gradient` is CPU-based and uses TBB parallelization
- Gradient computation is not yet integrated into CUDA kernels
- No CUDA-based gradient computation for intersection volumes

## Implementation Plan (Remaining)

### Phase 4: CUDA-Based Gradient Computation

**Goal**: Implement efficient CUDA kernels for computing gradients of intersection volumes with respect to geometry DoFs.

#### 4.1: Refactor DoF Gradient Interface for CUDA

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

### Phase 5: Applications

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

### Phase 6: Reduced Order Model DoF Parameterization

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

### Phase 7: Python Bindings and Kaolin Integration

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

## Updated File Organization

```
include/
  geometry/
    types.hpp
    geometry.hpp
    packing.hpp
    geometry_helpers.hpp
  dof/
    dof_parameterization.hpp
    affine_dof.hpp
    triangle_mesh_dof.hpp
    reduced_order_dof.hpp          # NEW
  quadrature/
    kgrid.hpp
    lebedev_io.hpp
    gauss_legendre.hpp
  compute_intersection_volume.hpp
  compute_intersection_volume_multi_object.hpp
  compute_intersection_volume_gradient.hpp        # NEW
  applications/
    geometry_alignment.hpp                         # NEW
    body_dynamics.hpp                              # NEW
  form_factor_helpers.hpp
  analytical_intersection.hpp

src/
  geometry/
    packing.cpp
    geometry_helpers.cpp
  dof/
    affine_dof.cpp
    triangle_mesh_dof.cpp
    reduced_order_dof.cpp                          # NEW
  compute_intersection_volume.cu
  compute_intersection_volume_multi_object.cu
  compute_intersection_volume_gradient.cu          # NEW
  applications/
    geometry_alignment.cpp                         # NEW
    body_dynamics.cpp                              # NEW
  form_factor_helpers.cpp
  analytical_intersection.cpp

python/
  poIntInt/
    __init__.py
    bindings.cpp                                   # pybind11 bindings
    setup.py

test/
  test_*.cpp                                       # Existing tests
  test_gradient_cuda.cpp                            # NEW
  test_alignment.cpp                               # NEW
  test_reduced_order.cpp                           # NEW
```

## Timeline and Priorities

### Immediate (Phase 4)
1. **Refactor DoF gradient interface for CUDA** (2-3 weeks)
2. **Implement scalar intersection volume gradient** (1-2 weeks)
3. **Extend to matrix version** (1-2 weeks)
4. **Testing and optimization** (1 week)

### Short-term (Phase 5)
1. **Geometry alignment application** (2-3 weeks)
2. **Body dynamics integration** (2-3 weeks)

### Medium-term (Phase 6)
1. **Reduced order model DoF** (3-4 weeks)

### Long-term (Phase 7)
1. **Python bindings** (2-3 weeks)
2. **Kaolin integration** (2-3 weeks)

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


#pragma once
#include "geometry/geometry.hpp"
#include "dof/dof_parameterization.hpp"
#include "quadrature/kgrid.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <functional>
#include <unordered_map>
#include <string>

namespace PoIntInt {

// Forward declarations
struct TriPacked;
struct DiskPacked;
struct GaussianPacked;

// ============================================================================
// CUDA Kernel Function Types
// ============================================================================

// Function pointer type for computing A(k) on CUDA
// Parameters: geometry data, k-grid, DoF parameters, output
using CudaComputeAkFunc = std::function<void(
  const Geometry& geom,
  const KGrid& kgrid,
  const Eigen::VectorXd& dofs,
  double2* d_A_out,  // Output: Q complex values
  int blockSize
)>;

// Function pointer type for computing ∂A(k)/∂θ on CUDA
// Parameters: geometry data, k-grid, DoF parameters, output
using CudaComputeAkGradientFunc = std::function<void(
  const Geometry& geom,
  const KGrid& kgrid,
  const Eigen::VectorXd& dofs,
  double2* d_grad_A_out,  // Output: Q × num_dofs complex gradients (row-major)
  int blockSize
)>;

// ============================================================================
// CUDA Kernel Registry
// ============================================================================

// Registry for CUDA kernel implementations
// Key: "DoFType_GeometryType" (e.g., "AffineDoF_Triangle", "TriangleMeshDoF_Triangle")
class CudaKernelRegistry {
public:
  // Register CUDA kernels for a DoF+geometry combination
  static void register_kernels(
    const std::string& dof_type,
    GeometryType geom_type,
    CudaComputeAkFunc compute_Ak,
    CudaComputeAkGradientFunc compute_Ak_gradient
  );
  
  // Check if CUDA kernels are available for a DoF+geometry combination
  static bool has_kernels(const std::string& dof_type, GeometryType geom_type);
  
  // Get CUDA kernel functions
  static std::pair<CudaComputeAkFunc, CudaComputeAkGradientFunc>
    get_kernels(const std::string& dof_type, GeometryType geom_type);
  
  // Get a unique key for a DoF+geometry combination
  static std::string make_key(const std::string& dof_type, GeometryType geom_type);
};

// Helper function to get DoF type name from a DoFParameterization pointer
// Uses dynamic_cast to identify the concrete type
std::string get_dof_type_name(const std::shared_ptr<DoFParameterization>& dof);

// ============================================================================
// Extended DoF Interface for CUDA Support
// ============================================================================

// Extended interface that DoF classes can optionally implement
// to provide CUDA kernel implementations
struct DoFCudaInterface {
  virtual ~DoFCudaInterface() = default;
  
  // Get a string identifier for this DoF type (e.g., "AffineDoF", "TriangleMeshDoF")
  virtual std::string dof_type_name() const = 0;
  
  // Check if this DoF supports CUDA kernels for a given geometry type
  virtual bool supports_cuda(GeometryType geom_type) const = 0;
  
  // Register CUDA kernels for supported geometry types
  // This should be called once per DoF type, typically in a static initializer
  virtual void register_cuda_kernels() const = 0;
};

} // namespace PoIntInt


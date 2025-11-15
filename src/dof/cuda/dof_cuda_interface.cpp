#include "dof/cuda/dof_cuda_interface.hpp"
#include "dof/affine_dof.hpp"
#include "dof/triangle_mesh_dof.hpp"
#include <unordered_map>
#include <mutex>
#include <memory>

namespace PoIntInt {

// Thread-safe registry
std::unordered_map<std::string, std::pair<CudaComputeAkFunc, CudaComputeAkGradientFunc>> 
  CudaKernelRegistry::registry_;
static std::mutex registry_mutex;

void CudaKernelRegistry::register_kernels(
  const std::string& dof_type,
  GeometryType geom_type,
  CudaComputeAkFunc compute_Ak,
  CudaComputeAkGradientFunc compute_Ak_gradient)
{
  std::lock_guard<std::mutex> lock(registry_mutex);
  std::string key = make_key(dof_type, geom_type);
  registry_[key] = std::make_pair(compute_Ak, compute_Ak_gradient);
}

bool CudaKernelRegistry::has_kernels(const std::string& dof_type, GeometryType geom_type) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  std::string key = make_key(dof_type, geom_type);
  return registry_.find(key) != registry_.end();
}

std::pair<CudaComputeAkFunc, CudaComputeAkGradientFunc>
CudaKernelRegistry::get_kernels(const std::string& dof_type, GeometryType geom_type) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  std::string key = make_key(dof_type, geom_type);
  auto it = registry_.find(key);
  if (it != registry_.end()) {
    return it->second;
  }
  return std::make_pair(CudaComputeAkFunc(), CudaComputeAkGradientFunc());
}

std::string CudaKernelRegistry::make_key(const std::string& dof_type, GeometryType geom_type) {
  const char* geom_names[] = {"Triangle", "Disk", "Gaussian"};
  return dof_type + "_" + std::string(geom_names[geom_type]);
}

std::string get_dof_type_name(const std::shared_ptr<DoFParameterization>& dof) {
  if (!dof) return "Unknown";
  
  // Try to identify the concrete type using dynamic_cast
  if (std::dynamic_pointer_cast<AffineDoF>(dof)) {
    return "AffineDoF";
  } else if (std::dynamic_pointer_cast<TriangleMeshDoF>(dof)) {
    return "TriangleMeshDoF";
  }
  
  // Fallback: use typeid (less reliable but works for unknown types)
  return "Unknown";
}

} // namespace PoIntInt


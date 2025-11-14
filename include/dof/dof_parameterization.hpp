#pragma once
#include "geometry/geometry.hpp"
#include "form_factor_helpers.hpp"
#include <Eigen/Dense>
#include <memory>

namespace PoIntInt {

// Abstract base class for degrees of freedom (DoF) parameterizations
struct DoFParameterization {
  virtual ~DoFParameterization() = default;
  
  // Number of DoFs
  virtual int num_dofs() const = 0;
  
  // Apply transformation to geometry (returns transformed geometry)
  virtual Geometry apply(const Geometry& geom, const Eigen::VectorXd& dofs) const = 0;

  virtual std::complex<double>
    compute_A(const Geometry& geom, const Eigen::Vector3d& k,
      const Eigen::VectorXd& dofs) const
  {
    return compute_A_geometry(apply(geom, dofs), k);
  }
  
  // Compute gradient of form factor A(k) w.r.t. DoFs
  // Returns: dA/dθ for each DoF (complex vector of size num_dofs)
  // Note: A(k) is complex, so gradient is also complex
  virtual Eigen::VectorXcd 
    compute_A_gradient(const Geometry& geom, const Eigen::Vector3d& k, 
                      const Eigen::VectorXd& dofs) const = 0;
  
  // Compute gradient of volume (using divergence theorem) w.r.t. DoFs
  // Volume: V = (1/3) ∫_S (x, y, z) · n dS
  // Returns: dV/dθ for each DoF (real vector of size num_dofs)
  virtual Eigen::VectorXd
    compute_volume_gradient(const Geometry& geom, const Eigen::VectorXd& dofs) const = 0;
};

} // namespace PoIntInt


#pragma once
#include "dof/dof_parameterization.hpp"
#include "geometry/geometry.hpp"
#include <Eigen/Dense>

namespace PoIntInt {

// Triangle mesh DoF parameterization
// DoFs are vertex positions: [v0_x, v0_y, v0_z, v1_x, v1_y, v1_z, ...]
// The number of DoFs is 3 * num_vertices
// Face connectivity is extracted from the geometry's TriPacked structures
struct TriangleMeshDoF : public DoFParameterization {
  // Constructor: takes the number of vertices
  // Face connectivity is retrieved from the geometry's TriPacked structures
  explicit TriangleMeshDoF(int num_vertices);
  
  int num_dofs() const override { return num_dofs_; }
  
  virtual std::complex<double>
    compute_A(const Geometry& geom, const Eigen::Vector3d& k,
      const Eigen::VectorXd& dofs) const override;
  
  // Compute gradient of form factor A(k) w.r.t. vertex positions
  // Accounts for changes in oriented area vector S when vertices move
  Eigen::VectorXcd 
    compute_A_gradient(const Geometry& geom, const Eigen::Vector3d& k, 
                      const Eigen::VectorXd& dofs) const override;

  // Compute volume
  double compute_volume(const Geometry& geom, const Eigen::VectorXd& dofs) const override;
  
  // Compute gradient of volume w.r.t. vertex positions
  Eigen::VectorXd
    compute_volume_gradient(const Geometry& geom, const Eigen::VectorXd& dofs) const override;
  
private:
  int num_vertices_;  // Number of vertices
  int num_dofs_;      // 3 * num_vertices
};

} // namespace PoIntInt


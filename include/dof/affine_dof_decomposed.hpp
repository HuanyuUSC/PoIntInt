#pragma once
#include "dof_parameterization.hpp"
#include "geometry/geometry.hpp"
#include <Eigen/Dense>
#include <vector>

namespace PoIntInt {

// Legacy: Affine transformation DoFs with decomposed parameterization
// [translation(3), rotation(3), scale(3), shear(3)] = 12 DoFs
// DoF layout: [tx, ty, tz, rx, ry, rz, sx, sy, sz, hxy, hxz, hyz]
// - Translation: t = [tx, ty, tz]
// - Rotation: axis-angle representation r = [rx, ry, rz] (angle = |r|, axis = r/|r|)
// - Scale: s = [sx, sy, sz]
// - Shear: h = [hxy, hxz, hyz] (upper triangular part of shear matrix)
// Kept for backward compatibility or special use cases
struct AffineDoF_Decomposed : public DoFParameterization {
  int num_dofs() const override { return 12; }
  
  // Apply affine transformation: x -> A*x + t
  // where A = R * S * H (rotation * scale * shear)
  Geometry apply(const Geometry& geom, const Eigen::VectorXd& dofs) const override;
  
  // Compute gradient of form factor A(k) w.r.t. DoFs
  Eigen::VectorXcd 
    compute_A_gradient(const Geometry& geom, const Eigen::Vector3d& k,
                      const Eigen::VectorXd& dofs) const override;
  
private:
  // Helper: Build affine transformation matrix from DoFs
  // Returns (A, t) where A is 3x3 matrix and t is translation
  std::pair<Eigen::Matrix3d, Eigen::Vector3d> 
    build_affine_transform(const Eigen::VectorXd& dofs) const;
  
  // Helper: Compute rotation matrix from axis-angle representation
  Eigen::Matrix3d rotation_matrix(const Eigen::Vector3d& axis_angle) const;
  
  // Helper: Compute gradient of rotation matrix w.r.t. axis-angle
  std::vector<Eigen::Matrix3d> rotation_matrix_gradients(const Eigen::Vector3d& axis_angle) const;
};

} // namespace PoIntInt


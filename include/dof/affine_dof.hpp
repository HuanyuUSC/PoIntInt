#pragma once
#include "dof_parameterization.hpp"
#include "../geometry.hpp"
#include <Eigen/Dense>

namespace PoIntInt {

// Affine transformation DoFs: [translation(3), matrix(9)] = 12 DoFs
// DoF layout: [tx, ty, tz, A00, A01, A02, A10, A11, A12, A20, A21, A22]
// - Translation: t = [tx, ty, tz] = dofs[0:3]
// - Matrix: A (3x3, row-major) = dofs[3:12]
//   A = [A00 A01 A02]
//       [A10 A11 A12]
//       [A20 A21 A22]
struct AffineDoF : public DoFParameterization {
  int num_dofs() const override { return 12; }
  
  // Apply affine transformation: x -> A*x + t
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
};

} // namespace PoIntInt


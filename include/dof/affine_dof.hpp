#pragma once
#include "dof_parameterization.hpp"
#include "geometry/geometry.hpp"
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
  
  std::complex<double>
    compute_A(const Geometry& geom, const Eigen::Vector3d& k,
      const Eigen::VectorXd& dofs) const override;
  
  // Compute gradient of form factor A(k) w.r.t. DoFs
  Eigen::VectorXcd 
    compute_A_gradient(const Geometry& geom, const Eigen::Vector3d& k,
                      const Eigen::VectorXd& dofs) const override;

  // Compute volume
  double compute_volume(const Geometry& geom, const Eigen::VectorXd& dofs) const override;
  
  // Compute gradient of volume w.r.t. DoFs
  Eigen::VectorXd
    compute_volume_gradient(const Geometry& geom, const Eigen::VectorXd& dofs) const override;
  
private:
  // Helper: Build affine transformation matrix from DoFs
  // Returns (A, t) where A is 3x3 matrix and t is translation
  static std::pair<Eigen::Matrix3d, Eigen::Vector3d> 
    build_affine_transform(const Eigen::VectorXd& dofs);

  // Helper: Compute cofactor matrix of 3x3 matrix A
  static Eigen::Matrix3d cofactor_matrix(const Eigen::Matrix3d& A);

  // Helper: Compute cross product matrix for vector v
  static Eigen::Matrix3d cross_product_matrix(const Eigen::Vector3d& v);
};

} // namespace PoIntInt


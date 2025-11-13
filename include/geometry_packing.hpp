#pragma once
#include <vector>
#include <array>
#include <Eigen/Dense>
#include "geometry_types.hpp"

namespace PoIntInt {

// Build k-grid from Lebedev directions and weights
KGrid build_kgrid(
  const std::vector<std::array<double, 3>>& leb_dirs,
  const std::vector<double>& leb_w,
  int Nrad);

// Pack triangle mesh into CUDA-friendly format
std::vector<TriPacked> pack_tris(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F);

// Pack oriented point cloud: positions P, normals N, radii (or areas)
// If radii provided, area = π * rho^2. If areas provided, rho = sqrt(area/π)
std::vector<DiskPacked> pack_disks(
  const Eigen::MatrixXd& P,  // positions
  const Eigen::MatrixXd& N,  // normals (will be normalized)
  const Eigen::VectorXd& radii_or_areas,  // either radii or areas
  bool is_radius = true);    // true if radii, false if areas

} // namespace PoIntInt


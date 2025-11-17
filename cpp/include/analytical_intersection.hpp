#pragma once
#include <Eigen/Dense>

namespace PoIntInt {

// ============================================================================
// Analytical Intersection Volume Formulas
// ============================================================================

// Compute intersection volume of two axis-aligned boxes
// box1: centered at c1, half-extents h1
// box2: centered at c2, half-extents h2
double box_box_intersection_volume(
  const Eigen::Vector3d& c1, const Eigen::Vector3d& h1,
  const Eigen::Vector3d& c2, const Eigen::Vector3d& h2);

// Compute intersection volume of two spheres
// sphere1: center c1, radius r1
// sphere2: center c2, radius r2
double sphere_sphere_intersection_volume(
  const Eigen::Vector3d& c1, double r1,
  const Eigen::Vector3d& c2, double r2);

// Compute intersection volume of a box and a sphere (approximate)
// box: centered at c_box, half-extents h
// sphere: center c_sphere, radius r
// Note: This is an approximation, not an exact formula
double box_sphere_intersection_volume(
  const Eigen::Vector3d& c_box, const Eigen::Vector3d& h,
  const Eigen::Vector3d& c_sphere, double r);

} // namespace PoIntInt


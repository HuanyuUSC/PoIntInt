#include "analytical_intersection.hpp"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace PoIntInt {

// ============================================================================
// Analytical Intersection Volume Formulas
// ============================================================================

double box_box_intersection_volume(
  const Eigen::Vector3d& c1, const Eigen::Vector3d& h1,
  const Eigen::Vector3d& c2, const Eigen::Vector3d& h2)
{
  Eigen::Vector3d overlap;
  for (int i = 0; i < 3; ++i) {
    double min1 = c1(i) - h1(i);
    double max1 = c1(i) + h1(i);
    double min2 = c2(i) - h2(i);
    double max2 = c2(i) + h2(i);
    double overlap_min = std::max(min1, min2);
    double overlap_max = std::min(max1, max2);
    overlap(i) = std::max(0.0, overlap_max - overlap_min);
  }
  return overlap(0) * overlap(1) * overlap(2);
}

double sphere_sphere_intersection_volume(
  const Eigen::Vector3d& c1, double r1,
  const Eigen::Vector3d& c2, double r2)
{
  double d = (c2 - c1).norm();
  
  // No intersection
  if (d >= r1 + r2) {
    return 0.0;
  }
  
  // One sphere completely inside the other
  if (d <= std::abs(r1 - r2)) {
    double r_min = std::min(r1, r2);
    return (4.0 * M_PI / 3.0) * r_min * r_min * r_min;
  }
  
  // Partial intersection
  // Formula: V = π * (r1 + r2 - d)^2 * (d^2 + 2*d*(r1 + r2) - 3*(r1^2 + r2^2) + 6*r1*r2) / (12*d)
  double d2 = d * d;
  double r1_2 = r1 * r1;
  double r2_2 = r2 * r2;
  
  double term1 = (r1 + r2 - d) * (r1 + r2 - d);
  double term2 = d2 + 2.0 * d * (r1 + r2) - 3.0 * (r1_2 + r2_2) + 6.0 * r1 * r2;
  double V = (M_PI * term1 * term2) / (12.0 * d);
  
  return V;
}

double box_sphere_intersection_volume(
  const Eigen::Vector3d& c_box, const Eigen::Vector3d& h,
  const Eigen::Vector3d& c_sphere, double r)
{
  // This is complex to compute exactly, so we use an approximation
  // For unit cube and unit sphere, we can use a simpler approach
  
  // Closest point on box to sphere center
  Eigen::Vector3d closest;
  for (int i = 0; i < 3; ++i) {
    double box_min = c_box(i) - h(i);
    double box_max = c_box(i) + h(i);
    closest(i) = std::max(box_min, std::min(c_sphere(i), box_max));
  }
  
  double dist_sq = (c_sphere - closest).squaredNorm();
  
  // Sphere completely outside box
  if (dist_sq >= r * r) {
    return 0.0;
  }
  
  // If sphere center is inside box
  bool inside = true;
  for (int i = 0; i < 3; ++i) {
    if (c_sphere(i) < c_box(i) - h(i) || c_sphere(i) > c_box(i) + h(i)) {
      inside = false;
      break;
    }
  }
  
  if (inside) {
    // Sphere center inside box - intersection is at least the sphere volume
    // but bounded by box volume
    double sphere_vol = (4.0 * M_PI / 3.0) * r * r * r;
    double box_vol = 8.0 * h(0) * h(1) * h(2);
    return std::min(sphere_vol, box_vol);
  }
  
  // For partial intersection, we use a simplified approximation
  // This is not exact but gives a reasonable estimate
  double dist = std::sqrt(dist_sq);
  if (dist < r) {
    // Partial intersection - approximate as fraction of sphere volume
    double fraction = 1.0 - (dist / r);
    double sphere_vol = (4.0 * M_PI / 3.0) * r * r * r;
    return sphere_vol * fraction * 0.5;  // Rough approximation
  }
  
  return 0.0;
}

} // namespace PoIntInt


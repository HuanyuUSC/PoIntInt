#include "geometry/geometry_helpers.hpp"
#include <cmath>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace PoIntInt {

// ============================================================================
// Geometry Creation Helpers
// ============================================================================

void create_unit_cube_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F) {
  V.resize(8, 3);
  // Vertices of cube [-1/2, 1/2]^3
  V <<
    -0.5, -0.5, -0.5,   // 0
     0.5, -0.5, -0.5,   // 1
     0.5,  0.5, -0.5,   // 2
    -0.5,  0.5, -0.5,   // 3
    -0.5, -0.5,  0.5,   // 4
     0.5, -0.5,  0.5,   // 5
     0.5,  0.5,  0.5,   // 6
    -0.5,  0.5,  0.5;   // 7
  
  F.resize(12, 3);
  // Faces (two triangles per face), outward orientation
  F <<
    0, 2, 1,  0, 3, 2,   // z = -0.5 (bottom)
    4, 5, 6,  4, 6, 7,   // z =  0.5 (top)
    0, 1, 5,  0, 5, 4,   // y = -0.5 (front)
    2, 7, 6,  2, 3, 7,   // y =  0.5 (back)
    0, 4, 7,  0, 7, 3,   // x = -0.5 (left)
    1, 6, 5,  1, 2, 6;   // x =  0.5 (right)
}

void create_sphere_pointcloud(
  Eigen::MatrixXd& P,  // positions
  Eigen::MatrixXd& N,  // normals (outward pointing)
  Eigen::VectorXd& radii,
  int n_points)
{
  // Generate points on unit sphere surface with uniform distribution
  // Using Fibonacci sphere algorithm for uniform distribution
  P.resize(n_points, 3);
  N.resize(n_points, 3);
  radii.resize(n_points);
  
  const double golden_angle = M_PI * (3.0 - std::sqrt(5.0));  // Golden angle in radians
  const double disk_area = 4.0 * M_PI / n_points;  // Precompute disk area
  const double radius_scale = std::sqrt(disk_area / M_PI);  // Precompute radius scale
  
  tbb::parallel_for(tbb::blocked_range<int>(0, n_points),
    [&](const tbb::blocked_range<int>& r) {
      for (int i = r.begin(); i < r.end(); ++i) {
        double y = 1.0 - (2.0 * i) / (n_points - 1.0);  // y goes from 1 to -1
        double radius_at_y = std::sqrt(1.0 - y * y);  // radius at height y
        double theta = golden_angle * i;
        
        double x = radius_at_y * std::cos(theta);
        double z = radius_at_y * std::sin(theta);
        
        // Position on unit sphere
        P(i, 0) = x;
        P(i, 1) = y;
        P(i, 2) = z;
        
        // Normal (outward pointing, same as position for unit sphere)
        N(i, 0) = x;
        N(i, 1) = y;
        N(i, 2) = z;
        
        // Radius for each disk - choose to cover surface uniformly
        // Total surface area = 4π, so each disk should have area ≈ 4π/n_points
        radii(i) = radius_scale;
      }
    });
}

void create_sphere_gaussians(
  Eigen::MatrixXd& P,  // positions
  Eigen::MatrixXd& N,  // normals (outward pointing)
  Eigen::VectorXd& sigmas,  // standard deviations in tangent plane
  Eigen::VectorXd& weights,  // area weights
  int n_points)
{
  // Generate points on unit sphere surface with uniform distribution
  // Using Fibonacci sphere algorithm for uniform distribution
  P.resize(n_points, 3);
  N.resize(n_points, 3);
  sigmas.resize(n_points);
  weights.resize(n_points);
  
  const double golden_angle = M_PI * (3.0 - std::sqrt(5.0));  // Golden angle in radians
  const double weight = 4.0 * M_PI / n_points;  // Precompute weight
  const double sigma = std::sqrt(weight / (4.0 * M_PI));  // Precompute sigma
  
  tbb::parallel_for(tbb::blocked_range<int>(0, n_points),
    [&](const tbb::blocked_range<int>& r) {
      for (int i = r.begin(); i < r.end(); ++i) {
        double y = 1.0 - (2.0 * i) / (n_points - 1.0);  // y goes from 1 to -1
        double radius_at_y = std::sqrt(1.0 - y * y);  // radius at height y
        double theta = golden_angle * i;
        
        double x = radius_at_y * std::cos(theta);
        double z = radius_at_y * std::sin(theta);
        
        // Position on unit sphere
        P(i, 0) = x;
        P(i, 1) = y;
        P(i, 2) = z;
        
        // Normal (outward pointing, same as position for unit sphere)
        N(i, 0) = x;
        N(i, 1) = y;
        N(i, 2) = z;
        
        // Total surface area = 4π, so each Gaussian should have weight ≈ 4π/n_points
        weights(i) = weight;
        
        // Choose sigma such that the Gaussian footprint covers approximately the same area
        // For a Gaussian with std dev sigma, the effective radius is about 2*sigma
        // We want the area covered to be approximately weight, so:
        // π * (2*sigma)^2 ≈ weight, so sigma ≈ sqrt(weight/(4π))
        sigmas(i) = sigma;
      }
    });
}

} // namespace PoIntInt


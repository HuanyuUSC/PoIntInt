#pragma once
#include "geometry/geometry.hpp"
#include "geometry/types.hpp"
#include <Eigen/Dense>
#include <complex>

namespace PoIntInt {

// ============================================================================
// Form Factor Field Computation Helpers
// ============================================================================

// Compute A_parallel(k) = (k·A(k))/|k| for triangles
std::complex<double> compute_A_triangle(
  const TriPacked& tri,
  const Eigen::Vector3d& k);

// Compute A_parallel(k) for disks
std::complex<double> compute_A_disk(
  const DiskPacked& disk,
  const Eigen::Vector3d& k);

// Compute A_parallel(k) for Gaussian splats
std::complex<double> compute_A_gaussian(
  const GaussianPacked& gauss,
  const Eigen::Vector3d& k);

// Compute A_parallel(k) for entire geometry
std::complex<double> compute_A_geometry(
  const Geometry& geom,
  const Eigen::Vector3d& k);

// ============================================================================
// Exact Form Factor Formulas for Simple Shapes
// ============================================================================

// Exact form factor for unit cube centered at origin: F(k) = ∏_{i=x,y,z} sin(k_i/2) / (k_i/2)
// For k_i = 0, use limit: sin(0)/0 = 1
double exact_cube_form_factor(double kx, double ky, double kz);

// Exact |A(k)|² for unit cube: |A(k)|² = |k|² |F(k)|²
double exact_cube_Ak_squared(double kx, double ky, double kz);

// Exact form factor for solid sphere of radius R: F_ball(k) = 4π (sin(kR) - kR cos(kR))/k³
// For unit sphere (R=1): F_ball(k) = 4π (sin(k) - k cos(k))/k³
double exact_sphere_form_factor(double k, double R = 1.0);

// Exact |A(k)|² for sphere: |A(k)|² = |k|² |F(k)|²
double exact_sphere_Ak_squared(double k, double R = 1.0);

} // namespace PoIntInt


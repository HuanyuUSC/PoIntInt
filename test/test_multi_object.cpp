#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "compute_volume_multi_object.hpp"
#include "geometry/packing.hpp"
#include "geometry/geometry_helpers.hpp"
#include "quadrature/lebedev_io.hpp"
#include "quadrature/gauss_legendre.hpp"
#include "quadrature/kgrid.hpp"

using namespace PoIntInt;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Use create_sphere_pointcloud and create_sphere_gaussians from geometry/geometry_helpers.hpp

// ============================================================================
// Analytical Ground Truth Functions
// ============================================================================

// Compute intersection volume of two axis-aligned boxes
// box1: centered at c1, half-extents h1
// box2: centered at c2, half-extents h2
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

// Compute intersection volume of two spheres
// sphere1: center c1, radius r1
// sphere2: center c2, radius r2
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
  double r1_3 = r1_2 * r1;
  double r2_3 = r2_2 * r2;
  
  double term1 = (r1 + r2 - d) * (r1 + r2 - d);
  double term2 = d2 + 2.0 * d * (r1 + r2) - 3.0 * (r1_2 + r2_2) + 6.0 * r1 * r2;
  double V = (M_PI * term1 * term2) / (12.0 * d);
  
  return V;
}

// Compute intersection volume of a box and a sphere
// box: centered at c_box, half-extents h
// sphere: center c_sphere, radius r
double box_sphere_intersection_volume(
  const Eigen::Vector3d& c_box, const Eigen::Vector3d& h,
  const Eigen::Vector3d& c_sphere, double r)
{
  // This is complex to compute exactly, so we use a Monte Carlo approximation
  // For unit cube and unit sphere, we can use a simpler approach
  // For now, we'll use a numerical integration approach or approximation
  
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
  
  // For exact computation, we'd need to integrate, but for unit cube and unit sphere
  // we can use a simpler approximation or lookup
  // For now, let's use a conservative approximation based on distance
  
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

// Use create_unit_cube_mesh from geometry/geometry_helpers.hpp

// Test: Multi-object intersection volume matrix
bool test_multi_object_volume_matrix(const std::string& leb_file, int Nrad = 96) {
  std::cout << "\n=== Test: Multi-Object Volume Matrix ===" << std::endl;
  
  // Create 3 unit cubes at different positions
  Eigen::MatrixXd V1, V2, V3;
  Eigen::MatrixXi F1, F2, F3;
  
  create_unit_cube_mesh(V1, F1);
  create_unit_cube_mesh(V2, F2);
  create_unit_cube_mesh(V3, F3);
  
  // Translate cubes
  Eigen::Vector3d t1(0.0, 0.0, 0.0);      // Cube 1 at origin
  Eigen::Vector3d t2(0.3, 0.0, 0.0);      // Cube 2 shifted in x
  Eigen::Vector3d t3(0.0, 0.3, 0.0);      // Cube 3 shifted in y
  
  for (int i = 0; i < V2.rows(); ++i) {
    V2.row(i) += t2;
  }
  for (int i = 0; i < V3.rows(); ++i) {
    V3.row(i) += t3;
  }
  
  // Create geometries
  auto geom1 = make_triangle_mesh(V1, F1);
  auto geom2 = make_triangle_mesh(V2, F2);
  auto geom3 = make_triangle_mesh(V3, F3);
  
  std::vector<Geometry> geometries = {geom1, geom2, geom3};
  
  // Load Lebedev grid
  LebedevGrid L = load_lebedev_txt(leb_file);
  KGrid KG = build_kgrid(L.dirs, L.weights, Nrad);
  
  // Compute volume matrix
  auto result = compute_intersection_volume_matrix_cuda(geometries, KG, 256, true);
  
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Volume matrix (3 objects):" << std::endl;
  std::cout << result.volume_matrix << std::endl;
  
  // Verify properties:
  // 1. Matrix should be symmetric
  bool is_symmetric = true;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (std::abs(result.volume_matrix(i, j) - result.volume_matrix(j, i)) > 1e-6) {
        is_symmetric = false;
        break;
      }
    }
    if (!is_symmetric) break;
  }
  
  // 2. Diagonal entries should be self-intersection volumes (≈ 1.0 for unit cube)
  bool diagonal_ok = true;
  double expected_volume = 1.0;  // Unit cube volume
  for (int i = 0; i < 3; ++i) {
    double rel_error = std::abs(result.volume_matrix(i, i) - expected_volume) / expected_volume;
    if (rel_error > 0.1) {  // 10% tolerance
      diagonal_ok = false;
      std::cout << "  Warning: Diagonal entry [" << i << "," << i << "] = " 
                << result.volume_matrix(i, i) << " (expected ~" << expected_volume << ")" << std::endl;
    }
  }
  
  // 3. Off-diagonal entries should be intersection volumes
  std::cout << "\nPairwise intersection volumes:" << std::endl;
  
  // Compute analytical ground truth
  Eigen::Vector3d c1(0.0, 0.0, 0.0);
  Eigen::Vector3d c2(0.3, 0.0, 0.0);
  Eigen::Vector3d c3(0.0, 0.3, 0.0);
  Eigen::Vector3d h(0.5, 0.5, 0.5);  // Half-extents of unit cube
  
  double V01_gt = box_box_intersection_volume(c1, h, c2, h);
  double V02_gt = box_box_intersection_volume(c1, h, c3, h);
  double V12_gt = box_box_intersection_volume(c2, h, c3, h);
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[0,1] (cubes 1 & 2):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(0, 1) << std::endl;
  std::cout << "    Ground truth: " << V01_gt << std::endl;
  std::cout << "    Relative error: " << std::scientific << std::abs(result.volume_matrix(0, 1) - V01_gt) / (V01_gt + 1e-10) << std::endl;
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[0,2] (cubes 1 & 3):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(0, 2) << std::endl;
  std::cout << "    Ground truth: " << V02_gt << std::endl;
  std::cout << "    Relative error: " << std::scientific << std::abs(result.volume_matrix(0, 2) - V02_gt) / (V02_gt + 1e-10) << std::endl;
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[1,2] (cubes 2 & 3):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(1, 2) << std::endl;
  std::cout << "    Ground truth: " << V12_gt << std::endl;
  std::cout << "    Relative error: " << std::scientific << std::abs(result.volume_matrix(1, 2) - V12_gt) / (V12_gt + 1e-10) << std::endl;
  
  // Check errors
  double max_rel_error = 0.0;
  max_rel_error = std::max(max_rel_error, std::abs(result.volume_matrix(0, 1) - V01_gt) / (V01_gt + 1e-10));
  max_rel_error = std::max(max_rel_error, std::abs(result.volume_matrix(0, 2) - V02_gt) / (V02_gt + 1e-10));
  max_rel_error = std::max(max_rel_error, std::abs(result.volume_matrix(1, 2) - V12_gt) / (V12_gt + 1e-10));
  
  bool pairwise_ok = max_rel_error < 0.15;  // 15% tolerance
  
  std::cout << "\nTest results:" << std::endl;
  std::cout << "  Symmetric: " << (is_symmetric ? "PASS" : "FAIL") << std::endl;
  std::cout << "  Diagonal:  " << (diagonal_ok ? "PASS" : "FAIL") << std::endl;
  std::cout << "  Pairwise:  " << (pairwise_ok ? "PASS" : "FAIL") << " (max rel error: " << std::scientific << max_rel_error << ")" << std::endl;
  
  bool passed = is_symmetric && diagonal_ok && pairwise_ok;
  std::cout << "  Overall:   " << (passed ? "PASS" : "FAIL") << std::endl;
  
  return passed;
}

// Test: Three unit sphere point clouds with different translations
bool test_three_sphere_pointclouds(const std::string& leb_file, int Nrad = 96) {
  std::cout << "\n=== Test: Three Unit Sphere Point Clouds ===" << std::endl;
  
  // Create 3 unit sphere point clouds at different positions
  Eigen::MatrixXd P1, N1, P2, N2, P3, N3;
  Eigen::VectorXd radii1, radii2, radii3;
  
  create_sphere_pointcloud(P1, N1, radii1, 2000);
  create_sphere_pointcloud(P2, N2, radii2, 2000);
  create_sphere_pointcloud(P3, N3, radii3, 2000);
  
  // Translate spheres
  Eigen::Vector3d t1(0.0, 0.0, 0.0);      // Sphere 1 at origin
  Eigen::Vector3d t2(0.5, 0.0, 0.0);      // Sphere 2 shifted in x (overlapping)
  Eigen::Vector3d t3(2.5, 0.0, 0.0);      // Sphere 3 shifted in x (non-overlapping)
  
  for (int i = 0; i < P2.rows(); ++i) {
    P2.row(i) += t2;
  }
  for (int i = 0; i < P3.rows(); ++i) {
    P3.row(i) += t3;
  }
  
  // Create geometries
  auto geom1 = make_point_cloud(P1, N1, radii1, true);
  auto geom2 = make_point_cloud(P2, N2, radii2, true);
  auto geom3 = make_point_cloud(P3, N3, radii3, true);
  
  std::vector<Geometry> geometries = {geom1, geom2, geom3};
  
  // Load Lebedev grid
  LebedevGrid L = load_lebedev_txt(leb_file);
  KGrid KG = build_kgrid(L.dirs, L.weights, Nrad);
  
  // Compute volume matrix
  auto result = compute_intersection_volume_matrix_cuda(geometries, KG, 256, true);
  
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Volume matrix (3 sphere point clouds):" << std::endl;
  std::cout << result.volume_matrix << std::endl;
  
  // Verify properties:
  // 1. Matrix should be symmetric
  bool is_symmetric = true;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (std::abs(result.volume_matrix(i, j) - result.volume_matrix(j, i)) > 1e-5) {
        is_symmetric = false;
        break;
      }
    }
    if (!is_symmetric) break;
  }
  
  // 2. Diagonal entries should be self-intersection volumes (≈ 4π/3 for unit sphere)
  bool diagonal_ok = true;
  double expected_volume = 4.0 * M_PI / 3.0;  // Unit sphere volume
  for (int i = 0; i < 3; ++i) {
    double rel_error = std::abs(result.volume_matrix(i, i) - expected_volume) / expected_volume;
    if (rel_error > 0.2) {  // 20% tolerance for point cloud approximation
      diagonal_ok = false;
      std::cout << "  Warning: Diagonal entry [" << i << "," << i << "] = " 
                << result.volume_matrix(i, i) << " (expected ~" << expected_volume << ")" << std::endl;
    }
  }
  
  // 3. Off-diagonal entries
  std::cout << "\nPairwise intersection volumes:" << std::endl;
  
  // Compute analytical ground truth
  Eigen::Vector3d c1(0.0, 0.0, 0.0);
  Eigen::Vector3d c2(0.5, 0.0, 0.0);
  Eigen::Vector3d c3(2.5, 0.0, 0.0);
  double r = 1.0;  // Unit sphere radius
  
  double V01_gt = sphere_sphere_intersection_volume(c1, r, c2, r);
  double V02_gt = sphere_sphere_intersection_volume(c1, r, c3, r);
  double V12_gt = sphere_sphere_intersection_volume(c2, r, c3, r);
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[0,1] (spheres 1 & 2, overlapping):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(0, 1) << std::endl;
  std::cout << "    Ground truth: " << V01_gt << std::endl;
  std::cout << "    Relative error: " << std::scientific << std::abs(result.volume_matrix(0, 1) - V01_gt) / (V01_gt + 1e-10) << std::endl;
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[0,2] (spheres 1 & 3, non-overlapping):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(0, 2) << std::endl;
  std::cout << "    Ground truth: " << V02_gt << std::endl;
  std::cout << "    Relative error: " << std::scientific << std::abs(result.volume_matrix(0, 2) - V02_gt) / (V02_gt + 1e-10) << std::endl;
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[1,2] (spheres 2 & 3, non-overlapping):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(1, 2) << std::endl;
  std::cout << "    Ground truth: " << V12_gt << std::endl;
  std::cout << "    Relative error: " << std::scientific << std::abs(result.volume_matrix(1, 2) - V12_gt) / (V12_gt + 1e-10) << std::endl;
  
  // Check errors
  double max_rel_error = 0.0;
  if (V01_gt > 1e-10) {
    max_rel_error = std::max(max_rel_error, std::abs(result.volume_matrix(0, 1) - V01_gt) / V01_gt);
  }
  if (V02_gt > 1e-10) {
    max_rel_error = std::max(max_rel_error, std::abs(result.volume_matrix(0, 2) - V02_gt) / V02_gt);
  } else {
    // For non-overlapping case, check absolute error
    max_rel_error = std::max(max_rel_error, std::abs(result.volume_matrix(0, 2) - V02_gt));
  }
  if (V12_gt > 1e-10) {
    max_rel_error = std::max(max_rel_error, std::abs(result.volume_matrix(1, 2) - V12_gt) / V12_gt);
  } else {
    max_rel_error = std::max(max_rel_error, std::abs(result.volume_matrix(1, 2) - V12_gt));
  }
  
  bool pairwise_ok = max_rel_error < 0.3;  // 30% tolerance for point cloud approximation
  bool overlap_ok = (result.volume_matrix(0, 1) > 0.1) && (result.volume_matrix(0, 2) < 0.1);
  
  std::cout << "\nTest results:" << std::endl;
  std::cout << "  Symmetric: " << (is_symmetric ? "PASS" : "FAIL") << std::endl;
  std::cout << "  Diagonal:  " << (diagonal_ok ? "PASS" : "FAIL") << std::endl;
  std::cout << "  Pairwise:  " << (pairwise_ok ? "PASS" : "FAIL") << " (max rel error: " << std::scientific << max_rel_error << ")" << std::endl;
  std::cout << "  Overlap:   " << (overlap_ok ? "PASS" : "FAIL") << std::endl;
  
  bool passed = is_symmetric && diagonal_ok && pairwise_ok && overlap_ok;
  std::cout << "  Overall:   " << (passed ? "PASS" : "FAIL") << std::endl;
  
  return passed;
}

// Test: Mixed geometry types (meshed cubes and point cloud spheres)
bool test_mixed_geometry_types(const std::string& leb_file, int Nrad = 96) {
  std::cout << "\n=== Test: Mixed Geometry Types (Cubes + Spheres) ===" << std::endl;
  
  // Create 2 unit cube meshes
  Eigen::MatrixXd V1, V2;
  Eigen::MatrixXi F1, F2;
  
  create_unit_cube_mesh(V1, F1);
  create_unit_cube_mesh(V2, F2);
  
  // Translate cubes
  Eigen::Vector3d t1(0.0, 0.0, 0.0);      // Cube 1 at origin
  Eigen::Vector3d t2(0.3, 0.0, 0.0);      // Cube 2 shifted in x (overlapping)
  
  for (int i = 0; i < V2.rows(); ++i) {
    V2.row(i) += t2;
  }
  
  // Create 2 unit sphere point clouds
  Eigen::MatrixXd P3, N3, P4, N4;
  Eigen::VectorXd radii3, radii4;
  
  create_sphere_pointcloud(P3, N3, radii3, 2000);
  create_sphere_pointcloud(P4, N4, radii4, 2000);
  
  // Translate spheres
  Eigen::Vector3d t3(0.0, 0.5, 0.0);      // Sphere 3 shifted in y (overlapping with cube 1)
  Eigen::Vector3d t4(0.3, 0.5, 0.0);       // Sphere 4 shifted in x and y (overlapping with cube 2)
  
  for (int i = 0; i < P3.rows(); ++i) {
    P3.row(i) += t3;
  }
  for (int i = 0; i < P4.rows(); ++i) {
    P4.row(i) += t4;
  }
  
  // Create geometries: [cube1, cube2, sphere1, sphere2]
  auto geom1 = make_triangle_mesh(V1, F1);
  auto geom2 = make_triangle_mesh(V2, F2);
  auto geom3 = make_point_cloud(P3, N3, radii3, true);
  auto geom4 = make_point_cloud(P4, N4, radii4, true);
  
  std::vector<Geometry> geometries = {geom1, geom2, geom3, geom4};
  
  // Load Lebedev grid
  LebedevGrid L = load_lebedev_txt(leb_file);
  KGrid KG = build_kgrid(L.dirs, L.weights, Nrad);
  
  // Compute volume matrix
  auto result = compute_intersection_volume_matrix_cuda(geometries, KG, 256, true);
  
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Volume matrix (2 cubes + 2 spheres):" << std::endl;
  std::cout << result.volume_matrix << std::endl;
  
  // Verify properties:
  // 1. Matrix should be symmetric
  bool is_symmetric = true;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (std::abs(result.volume_matrix(i, j) - result.volume_matrix(j, i)) > 1e-5) {
        is_symmetric = false;
        break;
      }
    }
    if (!is_symmetric) break;
  }
  
  // 2. Diagonal entries
  // Objects 0,1 (cubes): should be ≈ 1.0
  // Objects 2,3 (spheres): should be ≈ 4π/3
  bool diagonal_ok = true;
  double expected_cube_volume = 1.0;
  double expected_sphere_volume = 4.0 * M_PI / 3.0;
  
  for (int i = 0; i < 2; ++i) {
    double rel_error = std::abs(result.volume_matrix(i, i) - expected_cube_volume) / expected_cube_volume;
    if (rel_error > 0.1) {
      diagonal_ok = false;
      std::cout << "  Warning: Cube diagonal [" << i << "," << i << "] = " 
                << result.volume_matrix(i, i) << " (expected ~" << expected_cube_volume << ")" << std::endl;
    }
  }
  
  for (int i = 2; i < 4; ++i) {
    double rel_error = std::abs(result.volume_matrix(i, i) - expected_sphere_volume) / expected_sphere_volume;
    if (rel_error > 0.2) {  // 20% tolerance for point cloud
      diagonal_ok = false;
      std::cout << "  Warning: Sphere diagonal [" << i << "," << i << "] = " 
                << result.volume_matrix(i, i) << " (expected ~" << expected_sphere_volume << ")" << std::endl;
    }
  }
  
  // 3. Cross-type intersections (cube-sphere)
  std::cout << "\nPairwise intersection volumes:" << std::endl;
  
  // Compute analytical ground truth
  Eigen::Vector3d c_cube1(0.0, 0.0, 0.0);
  Eigen::Vector3d c_cube2(0.3, 0.0, 0.0);
  Eigen::Vector3d c_sphere1(0.0, 0.5, 0.0);
  Eigen::Vector3d c_sphere2(0.3, 0.5, 0.0);
  Eigen::Vector3d h(0.5, 0.5, 0.5);  // Half-extents of unit cube
  double r = 1.0;  // Unit sphere radius
  
  double V01_gt = box_box_intersection_volume(c_cube1, h, c_cube2, h);
  double V02_gt = box_sphere_intersection_volume(c_cube1, h, c_sphere1, r);
  double V03_gt = box_sphere_intersection_volume(c_cube1, h, c_sphere2, r);
  double V12_gt = box_sphere_intersection_volume(c_cube2, h, c_sphere1, r);
  double V13_gt = box_sphere_intersection_volume(c_cube2, h, c_sphere2, r);
  double V23_gt = sphere_sphere_intersection_volume(c_sphere1, r, c_sphere2, r);
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[0,1] (cube 1 & cube 2):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(0, 1) << std::endl;
  std::cout << "    Ground truth: " << V01_gt << std::endl;
  std::cout << "    Relative error: " << std::scientific << std::abs(result.volume_matrix(0, 1) - V01_gt) / (V01_gt + 1e-10) << std::endl;
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[0,2] (cube 1 & sphere 1):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(0, 2) << std::endl;
  std::cout << "    Ground truth (approx): " << V02_gt << std::endl;
  std::cout << "    Absolute error: " << std::scientific << std::abs(result.volume_matrix(0, 2) - V02_gt) << std::endl;
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[0,3] (cube 1 & sphere 2):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(0, 3) << std::endl;
  std::cout << "    Ground truth (approx): " << V03_gt << std::endl;
  std::cout << "    Absolute error: " << std::scientific << std::abs(result.volume_matrix(0, 3) - V03_gt) << std::endl;
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[1,2] (cube 2 & sphere 1):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(1, 2) << std::endl;
  std::cout << "    Ground truth (approx): " << V12_gt << std::endl;
  std::cout << "    Absolute error: " << std::scientific << std::abs(result.volume_matrix(1, 2) - V12_gt) << std::endl;
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[1,3] (cube 2 & sphere 2):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(1, 3) << std::endl;
  std::cout << "    Ground truth (approx): " << V13_gt << std::endl;
  std::cout << "    Absolute error: " << std::scientific << std::abs(result.volume_matrix(1, 3) - V13_gt) << std::endl;
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[2,3] (sphere 1 & sphere 2):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(2, 3) << std::endl;
  std::cout << "    Ground truth: " << V23_gt << std::endl;
  std::cout << "    Relative error: " << std::scientific << std::abs(result.volume_matrix(2, 3) - V23_gt) / (V23_gt + 1e-10) << std::endl;
  
  // Check errors
  double max_rel_error = 0.0;
  if (V01_gt > 1e-10) {
    max_rel_error = std::max(max_rel_error, std::abs(result.volume_matrix(0, 1) - V01_gt) / V01_gt);
  }
  if (V23_gt > 1e-10) {
    max_rel_error = std::max(max_rel_error, std::abs(result.volume_matrix(2, 3) - V23_gt) / V23_gt);
  }
  
  // For cube-sphere intersections, check absolute errors (since approximation is rough)
  double max_abs_error = 0.0;
  max_abs_error = std::max(max_abs_error, std::abs(result.volume_matrix(0, 2) - V02_gt));
  max_abs_error = std::max(max_abs_error, std::abs(result.volume_matrix(0, 3) - V03_gt));
  max_abs_error = std::max(max_abs_error, std::abs(result.volume_matrix(1, 2) - V12_gt));
  max_abs_error = std::max(max_abs_error, std::abs(result.volume_matrix(1, 3) - V13_gt));
  
  bool pairwise_ok = (max_rel_error < 0.3) && (max_abs_error < 0.5);  // Relaxed tolerance for approximations
  
  // Check that cross-type intersections work (should be non-negative)
  bool cross_type_ok = true;
  for (int i = 0; i < 2; ++i) {
    for (int j = 2; j < 4; ++j) {
      if (result.volume_matrix(i, j) < -1e-6) {  // Allow small numerical errors
        cross_type_ok = false;
        std::cout << "  Warning: Negative cross-type intersection [" << i << "," << j << "] = " 
                  << result.volume_matrix(i, j) << std::endl;
      }
    }
  }
  
  std::cout << "\nTest results:" << std::endl;
  std::cout << "  Symmetric:    " << (is_symmetric ? "PASS" : "FAIL") << std::endl;
  std::cout << "  Diagonal:     " << (diagonal_ok ? "PASS" : "FAIL") << std::endl;
  std::cout << "  Pairwise:     " << (pairwise_ok ? "PASS" : "FAIL") << " (max rel error: " << std::scientific << max_rel_error << ", max abs error: " << max_abs_error << ")" << std::endl;
  std::cout << "  Cross-type:   " << (cross_type_ok ? "PASS" : "FAIL") << std::endl;
  
  bool passed = is_symmetric && diagonal_ok && pairwise_ok && cross_type_ok;
  std::cout << "  Overall:      " << (passed ? "PASS" : "FAIL") << std::endl;
  
  return passed;
}

// Test: Three unit sphere Gaussian splats with different translations
bool test_three_gaussian_splats(const std::string& leb_file, int Nrad = 96) {
  std::cout << "\n=== Test: Three Unit Sphere Gaussian Splats ===" << std::endl;
  
  // Create 3 unit sphere Gaussian splats at different positions
  Eigen::MatrixXd P1, N1, P2, N2, P3, N3;
  Eigen::VectorXd sigmas1, weights1, sigmas2, weights2, sigmas3, weights3;
  
  create_sphere_gaussians(P1, N1, sigmas1, weights1, 2000);
  create_sphere_gaussians(P2, N2, sigmas2, weights2, 2000);
  create_sphere_gaussians(P3, N3, sigmas3, weights3, 2000);
  
  // Translate spheres
  Eigen::Vector3d t1(0.0, 0.0, 0.0);      // Sphere 1 at origin
  Eigen::Vector3d t2(0.5, 0.0, 0.0);      // Sphere 2 shifted in x (overlapping)
  Eigen::Vector3d t3(2.5, 0.0, 0.0);      // Sphere 3 shifted in x (non-overlapping)
  
  for (int i = 0; i < P2.rows(); ++i) {
    P2.row(i) += t2;
  }
  for (int i = 0; i < P3.rows(); ++i) {
    P3.row(i) += t3;
  }
  
  // Create geometries
  auto geom1 = make_gaussian_splat(P1, N1, sigmas1, weights1);
  auto geom2 = make_gaussian_splat(P2, N2, sigmas2, weights2);
  auto geom3 = make_gaussian_splat(P3, N3, sigmas3, weights3);
  
  std::vector<Geometry> geometries = {geom1, geom2, geom3};
  
  // Load Lebedev grid
  LebedevGrid L = load_lebedev_txt(leb_file);
  KGrid KG = build_kgrid(L.dirs, L.weights, Nrad);
  
  // Compute volume matrix
  auto result = compute_intersection_volume_matrix_cuda(geometries, KG, 256, true);
  
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Volume matrix (3 Gaussian splat spheres):" << std::endl;
  std::cout << result.volume_matrix << std::endl;
  
  // Verify properties:
  // 1. Matrix should be symmetric
  bool is_symmetric = true;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (std::abs(result.volume_matrix(i, j) - result.volume_matrix(j, i)) > 1e-5) {
        is_symmetric = false;
        break;
      }
    }
    if (!is_symmetric) break;
  }
  
  // 2. Diagonal entries should be self-intersection volumes (≈ 4π/3 for unit sphere)
  bool diagonal_ok = true;
  double expected_volume = 4.0 * M_PI / 3.0;  // Unit sphere volume
  for (int i = 0; i < 3; ++i) {
    double rel_error = std::abs(result.volume_matrix(i, i) - expected_volume) / expected_volume;
    if (rel_error > 0.25) {  // 25% tolerance for Gaussian splat approximation
      diagonal_ok = false;
      std::cout << "  Warning: Diagonal entry [" << i << "," << i << "] = " 
                << result.volume_matrix(i, i) << " (expected ~" << expected_volume << ")" << std::endl;
    }
  }
  
  // 3. Off-diagonal entries
  std::cout << "\nPairwise intersection volumes:" << std::endl;
  
  // Compute analytical ground truth
  Eigen::Vector3d c1(0.0, 0.0, 0.0);
  Eigen::Vector3d c2(0.5, 0.0, 0.0);
  Eigen::Vector3d c3(2.5, 0.0, 0.0);
  double r = 1.0;  // Unit sphere radius
  
  double V01_gt = sphere_sphere_intersection_volume(c1, r, c2, r);
  double V02_gt = sphere_sphere_intersection_volume(c1, r, c3, r);
  double V12_gt = sphere_sphere_intersection_volume(c2, r, c3, r);
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[0,1] (Gaussians 1 & 2, overlapping):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(0, 1) << std::endl;
  std::cout << "    Ground truth: " << V01_gt << std::endl;
  std::cout << "    Relative error: " << std::scientific << std::abs(result.volume_matrix(0, 1) - V01_gt) / (V01_gt + 1e-10) << std::endl;
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[0,2] (Gaussians 1 & 3, non-overlapping):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(0, 2) << std::endl;
  std::cout << "    Ground truth: " << V02_gt << std::endl;
  std::cout << "    Relative error: " << std::scientific << std::abs(result.volume_matrix(0, 2) - V02_gt) / (V02_gt + 1e-10) << std::endl;
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[1,2] (Gaussians 2 & 3, non-overlapping):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(1, 2) << std::endl;
  std::cout << "    Ground truth: " << V12_gt << std::endl;
  std::cout << "    Relative error: " << std::scientific << std::abs(result.volume_matrix(1, 2) - V12_gt) / (V12_gt + 1e-10) << std::endl;
  
  // Check errors
  double max_rel_error = 0.0;
  if (V01_gt > 1e-10) {
    max_rel_error = std::max(max_rel_error, std::abs(result.volume_matrix(0, 1) - V01_gt) / V01_gt);
  }
  if (V02_gt > 1e-10) {
    max_rel_error = std::max(max_rel_error, std::abs(result.volume_matrix(0, 2) - V02_gt) / V02_gt);
  } else {
    // For non-overlapping case, check absolute error
    max_rel_error = std::max(max_rel_error, std::abs(result.volume_matrix(0, 2) - V02_gt));
  }
  if (V12_gt > 1e-10) {
    max_rel_error = std::max(max_rel_error, std::abs(result.volume_matrix(1, 2) - V12_gt) / V12_gt);
  } else {
    max_rel_error = std::max(max_rel_error, std::abs(result.volume_matrix(1, 2) - V12_gt));
  }
  
  bool pairwise_ok = max_rel_error < 0.35;  // 35% tolerance for Gaussian splat approximation
  bool overlap_ok = (result.volume_matrix(0, 1) > 0.1) && (result.volume_matrix(0, 2) < 0.1);
  
  std::cout << "\nTest results:" << std::endl;
  std::cout << "  Symmetric: " << (is_symmetric ? "PASS" : "FAIL") << std::endl;
  std::cout << "  Diagonal:  " << (diagonal_ok ? "PASS" : "FAIL") << std::endl;
  std::cout << "  Pairwise:  " << (pairwise_ok ? "PASS" : "FAIL") << " (max rel error: " << std::scientific << max_rel_error << ")" << std::endl;
  std::cout << "  Overlap:   " << (overlap_ok ? "PASS" : "FAIL") << std::endl;
  
  bool passed = is_symmetric && diagonal_ok && pairwise_ok && overlap_ok;
  std::cout << "  Overall:   " << (passed ? "PASS" : "FAIL") << std::endl;
  
  return passed;
}

// Test: Mixed geometry types including Gaussian splats (cubes, point clouds, and Gaussian splats)
bool test_mixed_geometry_with_gaussians(const std::string& leb_file, int Nrad = 96) {
  std::cout << "\n=== Test: Mixed Geometry Types (Cubes + Point Clouds + Gaussian Splats) ===" << std::endl;
  
  // Create 1 unit cube mesh
  Eigen::MatrixXd V1;
  Eigen::MatrixXi F1;
  create_unit_cube_mesh(V1, F1);
  Eigen::Vector3d t1(0.0, 0.0, 0.0);  // Cube at origin
  auto geom1 = make_triangle_mesh(V1, F1);
  
  // Create 1 unit sphere point cloud
  Eigen::MatrixXd P2, N2;
  Eigen::VectorXd radii2;
  create_sphere_pointcloud(P2, N2, radii2, 2000);
  Eigen::Vector3d t2(0.3, 0.0, 0.0);  // Sphere shifted in x (overlapping with cube)
  for (int i = 0; i < P2.rows(); ++i) {
    P2.row(i) += t2;
  }
  auto geom2 = make_point_cloud(P2, N2, radii2, true);
  
  // Create 1 unit sphere Gaussian splat
  Eigen::MatrixXd P3, N3;
  Eigen::VectorXd sigmas3, weights3;
  create_sphere_gaussians(P3, N3, sigmas3, weights3, 2000);
  Eigen::Vector3d t3(0.0, 0.5, 0.0);  // Gaussian splat shifted in y (overlapping with cube)
  for (int i = 0; i < P3.rows(); ++i) {
    P3.row(i) += t3;
  }
  auto geom3 = make_gaussian_splat(P3, N3, sigmas3, weights3);
  
  // Create another Gaussian splat
  Eigen::MatrixXd P4, N4;
  Eigen::VectorXd sigmas4, weights4;
  create_sphere_gaussians(P4, N4, sigmas4, weights4, 2000);
  Eigen::Vector3d t4(0.3, 0.5, 0.0);  // Gaussian splat shifted in x and y (overlapping with others)
  for (int i = 0; i < P4.rows(); ++i) {
    P4.row(i) += t4;
  }
  auto geom4 = make_gaussian_splat(P4, N4, sigmas4, weights4);
  
  std::vector<Geometry> geometries = {geom1, geom2, geom3, geom4};
  
  // Load Lebedev grid
  LebedevGrid L = load_lebedev_txt(leb_file);
  KGrid KG = build_kgrid(L.dirs, L.weights, Nrad);
  
  // Compute volume matrix
  auto result = compute_intersection_volume_matrix_cuda(geometries, KG, 256, true);
  
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Volume matrix (1 cube + 1 point cloud + 2 Gaussian splats):" << std::endl;
  std::cout << result.volume_matrix << std::endl;
  
  // Verify properties:
  // 1. Matrix should be symmetric
  bool is_symmetric = true;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (std::abs(result.volume_matrix(i, j) - result.volume_matrix(j, i)) > 1e-5) {
        is_symmetric = false;
        break;
      }
    }
    if (!is_symmetric) break;
  }
  
  // 2. Diagonal entries
  // Object 0 (cube): should be ≈ 1.0
  // Objects 1 (point cloud), 2,3 (Gaussian splats): should be ≈ 4π/3
  bool diagonal_ok = true;
  double expected_cube_volume = 1.0;
  double expected_sphere_volume = 4.0 * M_PI / 3.0;
  
  double rel_error_cube = std::abs(result.volume_matrix(0, 0) - expected_cube_volume) / expected_cube_volume;
  if (rel_error_cube > 0.1) {
    diagonal_ok = false;
    std::cout << "  Warning: Cube diagonal [0,0] = " 
              << result.volume_matrix(0, 0) << " (expected ~" << expected_cube_volume << ")" << std::endl;
  }
  
  for (int i = 1; i < 4; ++i) {
    double rel_error = std::abs(result.volume_matrix(i, i) - expected_sphere_volume) / expected_sphere_volume;
    if (rel_error > 0.25) {  // 25% tolerance for point cloud and Gaussian splats
      diagonal_ok = false;
      std::cout << "  Warning: Sphere diagonal [" << i << "," << i << "] = " 
                << result.volume_matrix(i, i) << " (expected ~" << expected_sphere_volume << ")" << std::endl;
    }
  }
  
  // 3. Cross-type intersections
  std::cout << "\nPairwise intersection volumes:" << std::endl;
  
  // Compute analytical ground truth
  Eigen::Vector3d c_cube(0.0, 0.0, 0.0);
  Eigen::Vector3d c_pc(0.3, 0.0, 0.0);
  Eigen::Vector3d c_gauss1(0.0, 0.5, 0.0);
  Eigen::Vector3d c_gauss2(0.3, 0.5, 0.0);
  Eigen::Vector3d h(0.5, 0.5, 0.5);  // Half-extents of unit cube
  double r = 1.0;  // Unit sphere radius
  
  double V01_gt = box_sphere_intersection_volume(c_cube, h, c_pc, r);  // cube - point cloud
  double V02_gt = box_sphere_intersection_volume(c_cube, h, c_gauss1, r);  // cube - Gaussian 1
  double V03_gt = box_sphere_intersection_volume(c_cube, h, c_gauss2, r);  // cube - Gaussian 2
  double V12_gt = sphere_sphere_intersection_volume(c_pc, r, c_gauss1, r);  // point cloud - Gaussian 1
  double V13_gt = sphere_sphere_intersection_volume(c_pc, r, c_gauss2, r);  // point cloud - Gaussian 2
  double V23_gt = sphere_sphere_intersection_volume(c_gauss1, r, c_gauss2, r);  // Gaussian 1 - Gaussian 2
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[0,1] (cube & point cloud):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(0, 1) << std::endl;
  std::cout << "    Ground truth (approx): " << V01_gt << std::endl;
  std::cout << "    Absolute error: " << std::scientific << std::abs(result.volume_matrix(0, 1) - V01_gt) << std::endl;
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[0,2] (cube & Gaussian 1):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(0, 2) << std::endl;
  std::cout << "    Ground truth (approx): " << V02_gt << std::endl;
  std::cout << "    Absolute error: " << std::scientific << std::abs(result.volume_matrix(0, 2) - V02_gt) << std::endl;
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[0,3] (cube & Gaussian 2):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(0, 3) << std::endl;
  std::cout << "    Ground truth (approx): " << V03_gt << std::endl;
  std::cout << "    Absolute error: " << std::scientific << std::abs(result.volume_matrix(0, 3) - V03_gt) << std::endl;
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[1,2] (point cloud & Gaussian 1):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(1, 2) << std::endl;
  std::cout << "    Ground truth: " << V12_gt << std::endl;
  std::cout << "    Relative error: " << std::scientific << std::abs(result.volume_matrix(1, 2) - V12_gt) / (V12_gt + 1e-10) << std::endl;
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[1,3] (point cloud & Gaussian 2):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(1, 3) << std::endl;
  std::cout << "    Ground truth: " << V13_gt << std::endl;
  std::cout << "    Relative error: " << std::scientific << std::abs(result.volume_matrix(1, 3) - V13_gt) / (V13_gt + 1e-10) << std::endl;
  
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "  V[2,3] (Gaussian 1 & Gaussian 2):" << std::endl;
  std::cout << "    Computed: " << result.volume_matrix(2, 3) << std::endl;
  std::cout << "    Ground truth: " << V23_gt << std::endl;
  std::cout << "    Relative error: " << std::scientific << std::abs(result.volume_matrix(2, 3) - V23_gt) / (V23_gt + 1e-10) << std::endl;
  
  // Check errors
  double max_rel_error = 0.0;
  if (V12_gt > 1e-10) {
    max_rel_error = std::max(max_rel_error, std::abs(result.volume_matrix(1, 2) - V12_gt) / V12_gt);
  }
  if (V13_gt > 1e-10) {
    max_rel_error = std::max(max_rel_error, std::abs(result.volume_matrix(1, 3) - V13_gt) / V13_gt);
  }
  if (V23_gt > 1e-10) {
    max_rel_error = std::max(max_rel_error, std::abs(result.volume_matrix(2, 3) - V23_gt) / V23_gt);
  }
  
  // For cube-sphere intersections, check absolute errors (since approximation is rough)
  double max_abs_error = 0.0;
  max_abs_error = std::max(max_abs_error, std::abs(result.volume_matrix(0, 1) - V01_gt));
  max_abs_error = std::max(max_abs_error, std::abs(result.volume_matrix(0, 2) - V02_gt));
  max_abs_error = std::max(max_abs_error, std::abs(result.volume_matrix(0, 3) - V03_gt));
  
  bool pairwise_ok = (max_rel_error < 0.4) && (max_abs_error < 0.6);  // Relaxed tolerance for mixed approximations
  
  // Check that cross-type intersections work (should be non-negative)
  bool cross_type_ok = true;
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      if (result.volume_matrix(i, j) < -1e-6) {  // Allow small numerical errors
        cross_type_ok = false;
        std::cout << "  Warning: Negative cross-type intersection [" << i << "," << j << "] = " 
                  << result.volume_matrix(i, j) << std::endl;
      }
    }
  }
  
  std::cout << "\nTest results:" << std::endl;
  std::cout << "  Symmetric:    " << (is_symmetric ? "PASS" : "FAIL") << std::endl;
  std::cout << "  Diagonal:     " << (diagonal_ok ? "PASS" : "FAIL") << std::endl;
  std::cout << "  Pairwise:     " << (pairwise_ok ? "PASS" : "FAIL") << " (max rel error: " << std::scientific << max_rel_error << ", max abs error: " << max_abs_error << ")" << std::endl;
  std::cout << "  Cross-type:   " << (cross_type_ok ? "PASS" : "FAIL") << std::endl;
  
  bool passed = is_symmetric && diagonal_ok && pairwise_ok && cross_type_ok;
  std::cout << "  Overall:      " << (passed ? "PASS" : "FAIL") << std::endl;
  
  return passed;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <lebedev_txt_file> [Nrad]" << std::endl;
    return 1;
  }
  
  std::string leb_file = argv[1];
  int Nrad = (argc > 2 ? std::atoi(argv[2]) : 96);
  
  std::cout << "Running multi-object volume matrix tests..." << std::endl;
  std::cout << "Lebedev file: " << leb_file << std::endl;
  std::cout << "Radial points: " << Nrad << std::endl;
  
  bool all_passed = true;
  
  // Test 1: Three unit cubes
  all_passed &= test_multi_object_volume_matrix(leb_file, Nrad);
  
  // Test 2: Three unit sphere point clouds
  all_passed &= test_three_sphere_pointclouds(leb_file, Nrad);
  
  // Test 3: Mixed geometry types (cubes + spheres)
  all_passed &= test_mixed_geometry_types(leb_file, Nrad);
  
  // Test 4: Three unit sphere Gaussian splats
  all_passed &= test_three_gaussian_splats(leb_file, Nrad);
  
  // Test 5: Mixed geometry types including Gaussian splats
  all_passed &= test_mixed_geometry_with_gaussians(leb_file, Nrad);
  
  std::cout << "\n=== Summary ===" << std::endl;
  if (all_passed) {
    std::cout << "All tests PASSED!" << std::endl;
    return 0;
  } else {
    std::cout << "Some tests FAILED!" << std::endl;
    return 1;
  }
}


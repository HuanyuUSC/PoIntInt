#include <Eigen/Dense>
#include <iostream>
#include <numeric>
#include <iomanip>
#include "lebedev_io.hpp"
#include "gauss_legendre.hpp"
#include "compute_volume.hpp"
#include "geometry_packing.hpp"
#include <cmath>
#include <corecrt_math_defines.h>
#include <igl/readOFF.h>
#include <chrono>

using namespace PoIntInt;

double volume(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
  double vol = 0.0;
  for (int i = 0; i < F.rows(); i++)
  {
    Eigen::Vector3d v0 = V.row(F(i, 0));
    Eigen::Vector3d v1 = V.row(F(i, 1));
    Eigen::Vector3d v2 = V.row(F(i, 2));
    vol += v0.dot(v1.cross(v2)) / 6.0;
  }
  return fabs(vol);
}

double Ak_squared_norm(double r)
{
  double norm;
  if (fabs(r) < 1e-3)
  {
    norm = -r / 3.0 + r * r * r / 30.0;
  }
  else
  {
    norm = (cos(r) - sin(r) / r) / r;
  }
  norm *= 4.0 * M_PI;
  return norm * norm;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: demo <lebedev_txt_file> [Nrad]\n";
    return 1;
  }
  std::string leb_file = argv[1];
  int Nrad = (argc > 2 ? std::atoi(argv[2]) : 96);

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  igl::readOFF("F:/Dropbox/3Dmodels/100 selected meshes/100 selected meshes/head2.off", V, F);
  auto geom1 = make_triangle_mesh(V, F);

  //// --- Example: unit cube [0,1]^3 triangulated
  //Eigen::MatrixXd V(8, 3); V <<
  //  0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
  //  0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1;
  //V *= 2;
  //Eigen::MatrixXi F(12, 3);
  //// faces (two tris per face), outward orientation
  //F <<
  //  0, 1, 2, 0, 2, 3,   // z=0
  //  4, 7, 6, 4, 6, 5,   // z=1
  //  0, 4, 5, 0, 5, 1,   // y=0
  //  2, 6, 7, 2, 7, 3,   // y=1
  //  0, 3, 7, 0, 7, 4,   // x=0
  //  1, 5, 6, 1, 6, 2;   // x=1

  //auto geom1 = make_triangle_mesh(V, F);

  Eigen::Vector3d t(0, 0, 0);
  Eigen::MatrixXd V2 = V;
  for (int i = 0; i < V.rows(); i++) V2.row(i) += t;
  auto geom2 = make_triangle_mesh(V2, F);

  // Load Lebedev (e.g., 302-point set); weights should sum to 4π
  LebedevGrid L = load_lebedev_txt(leb_file);
  std::cout << "Loaded Lebedev points: " << L.dirs.size()
    << " (sum weights = " << std::accumulate(L.weights.begin(), L.weights.end(), 0.0) << ")\n";

  // Build k-grid (Lebedev × radial on t in [0,π/2], k=tan t)
  KGrid KG = build_kgrid(L.dirs, L.weights, Nrad);

  double I_gt = 4.0 * M_PI / 3.0;
  double I = 0.0;
  for (int q = 0; q < KG.kmag.size(); q++)
  {
    I += KG.w[q] * Ak_squared_norm(KG.kmag[q]);
  }
  I /= 8.0 * M_PI * M_PI * M_PI;
  printf("Computed sphere integral: %g. Ground truth: %g. Rel err: %g%%\n", I, I_gt, abs(I - I_gt) / I_gt * 100.0);

  // For self-intersection, pass the same mesh twice
  auto t_start = std::chrono::high_resolution_clock::now();
  double Vself = compute_intersection_volume_cuda(geom1, geom1, KG, 256);
  auto t_end = std::chrono::high_resolution_clock::now();
  auto compute_volume_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
  double Vgt = volume(V, F);
  printf("Computed volume: %g, ground truth: %g, rel err: %g%%\n", Vself, Vgt, abs(Vself - Vgt) / Vgt * 100.0);
  printf("Time used: %g[ms]\n", compute_volume_time);
  return 0;
}

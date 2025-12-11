#include <Eigen/Dense>
#include <iostream>
#include <numeric>
#include <iomanip>
#include "quadrature/lebedev_io.hpp"
#include "quadrature/gauss_legendre.hpp"
#include "quadrature/kgrid.hpp"
#include "compute_intersection_volume.hpp"
#include "compute_volume.hpp"
#include "geometry/packing.hpp"
#ifdef _WIN32
#include <corecrt_math_defines.h>
#endif
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <igl/readOFF.h>
#include <chrono>

using namespace PoIntInt;

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

  // For self-intersection, pass the same mesh twice
  double Vcuda = compute_intersection_volume_cuda(geom1, geom1, KG, 256, true);
  double Vcpu = compute_intersection_volume_cpu(geom1, geom1, KG, true);
  double Vgt = compute_volume_cpu(geom1, true);
  printf("Computed volume CUDA: %g, ground truth: %g, rel err: %g%%\n", Vcuda, Vgt, abs(Vcuda - Vgt) / Vgt * 100.0);
  printf("Computed volume cpu+tbb: %g, ground truth: %g, rel err: %g%%\n", Vcpu, Vgt, abs(Vcpu - Vgt) / Vgt * 100.0);
  return 0;
}

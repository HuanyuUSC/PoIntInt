#pragma once
#include <vector>
#include <array>
#include <cuda_runtime.h>
#include <Eigen/Dense>

// Types needed for CUDA computation
struct TriPacked {
  float3 a;   // one vertex
  float3 e1;  // b - a
  float3 e2;  // c - a
  float3 S;   // 0.5 * (e1 x e2)  (oriented area vector)
};

struct KGrid {
  std::vector<std::array<float,3>> dirs; // unit dir
  std::vector<float> kmag;               // k = tan t
  std::vector<double> w;                 // full weight: (1/2π^2) * w_ang * w_rad * sec^2(t) (always double for accuracy)
};

// Function declarations
KGrid build_kgrid(
  const std::vector<std::array<double, 3>>& leb_dirs,
  const std::vector<double>& leb_w,
  int Nrad);

std::vector<TriPacked> pack_tris(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);

double compute_intersection_volume_cuda(
  const std::vector<TriPacked>& tris1,
  const std::vector<TriPacked>& tris2,
  const KGrid& KG,
  int blockSize = 256);


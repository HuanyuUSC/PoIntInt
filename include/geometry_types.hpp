#pragma once
#include <vector>
#include <array>
#include <cuda_runtime.h>

namespace PoIntInt {

// Types needed for CUDA computation
struct TriPacked {
  float3 a;   // one vertex
  float3 e1;  // b - a
  float3 e2;  // c - a
  float3 S;   // 0.5 * (e1 x e2)  (oriented area vector)
};

// Oriented point cloud element (circular disk surfel)
// Note: Struct size is 32 bytes (12+12+4+4), naturally aligned
struct DiskPacked {
  float3 c;   // center (12 bytes)
  float3 n;   // unit normal (12 bytes)
  float rho;  // radius (4 bytes)
  float area; // area weight (π * rho^2 for disk) (4 bytes)
  // Total: 32 bytes
};

// Gaussian splat surfel (planar Gaussian footprint)
// Note: Struct size is 32 bytes (12+12+4+4), naturally aligned
struct GaussianPacked {
  float3 c;   // center (12 bytes)
  float3 n;   // unit normal (12 bytes)
  float sigma; // standard deviation in tangent plane (4 bytes)
  float w;     // area weight (4 bytes)
  // Total: 32 bytes
};

// Geometry type enum
enum GeometryType {
  GEOM_TRIANGLE = 0,
  GEOM_DISK = 1,
  GEOM_GAUSSIAN = 2
};

struct KGrid {
  std::vector<std::array<float,3>> dirs; // unit dir
  std::vector<float> kmag;               // k = tan t
  std::vector<double> w;                 // full weight: (1/2π^2) * w_ang * w_rad * sec^2(t) (always double for accuracy)
};

} // namespace PoIntInt


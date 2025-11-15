#pragma once
#include <vector>
#include <array>
#include <cuda_runtime.h>

namespace PoIntInt {

// Types needed for CUDA computation
struct TriPacked {
  double3 a;   // one vertex
  double3 e1;  // b - a
  double3 e2;  // c - a
  double3 S;   // 0.5 * (e1 x e2)  (oriented area vector)
};

// Oriented point cloud element (circular disk surfel)
// Note: Struct size is 48 bytes (24+24+8+8), naturally aligned
struct DiskPacked {
  double3 c;   // center (24 bytes)
  double3 n;   // unit normal (24 bytes)
  double rho;  // radius (8 bytes)
  double area; // area weight (π * rho^2 for disk) (8 bytes)
  // Total: 64 bytes
};

// Gaussian splat surfel (planar Gaussian footprint)
// Note: Struct size is 48 bytes (24+24+8+8), naturally aligned
struct GaussianPacked {
  double3 c;   // center (24 bytes)
  double3 n;   // unit normal (24 bytes)
  double sigma; // standard deviation in tangent plane (8 bytes)
  double w;     // area weight (8 bytes)
  // Total: 64 bytes
};

// Geometry type enum
enum GeometryType {
  GEOM_TRIANGLE = 0,
  GEOM_DISK = 1,
  GEOM_GAUSSIAN = 2
};

} // namespace PoIntInt


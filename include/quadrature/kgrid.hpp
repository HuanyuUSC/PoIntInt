#pragma once
#include "quadrature/gauss_legendre.hpp"
#include <vector>
#include <array>
#include <cmath>

namespace PoIntInt {

struct KGrid {
  std::vector<std::array<double,3>> dirs; // unit dir
  std::vector<double> kmag;               // k = tan t
  std::vector<double> w;                 // full weight: (1/2π^2) * w_ang * w_rad * sec^2(t) (always double for accuracy)
};

// Build k-grid from Lebedev directions and weights
// Combines angular (Lebedev) and radial (Gauss-Legendre) quadrature
inline KGrid build_kgrid(
  const std::vector<std::array<double, 3>>& leb_dirs,
  const std::vector<double>& leb_w,
  int Nrad)
  {
    // radial t in [0, π/2]; Gauss-Legendre
    auto [t, wt] = gauss_legendre_interval(Nrad, 0.0, 0.5*M_PI);
  
    KGrid KG;
    // Note: The integral is V = (1/(4π)) ∫ |A(k)|² / k² d³k
    // In spherical coords: d³k = k² dk dΩ, so the k² cancels: V = (1/(4π)) ∫ |A(k)|² dk dΩ
    // With k = tan(t), we have dk = sec²(t) dt, so: V = (1/(4π)) ∫ |A(k)|² sec²(t) dt dΩ
    // The weights should be: w_angular * w_radial * sec²(t)
    // The final division by (8π^3) happens in compute_intersection_volume_cuda, so we don't include it here
    for (int ir=0; ir<Nrad; ++ir){
      double ti = t[ir], wti = wt[ir];
      double sec2 = 1.0 / std::cos(ti) / std::cos(ti);
      double k    = std::tan(ti);
      for (size_t j=0; j<leb_dirs.size(); ++j){
        const auto& d = leb_dirs[j];
        KG.dirs.push_back( { d[0], d[1], d[2] } );
        KG.kmag.push_back( k );
        // weight for this node: leb_w[j] * (wti * sec^2)
        // leb_w[j] integrates over solid angle (sums to 4π)
        // wti integrates over t in [0, π/2]
        // sec² accounts for dk = sec²(t) dt transformation
        KG.w.push_back( leb_w[j] * (wti * sec2) );
      }
    }
    return KG;
  }

} // namespace PoIntInt


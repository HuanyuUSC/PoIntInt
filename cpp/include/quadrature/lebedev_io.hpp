#pragma once
#include <vector>
#include <array>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <corecrt_math_defines.h>

namespace PoIntInt {

struct LebedevGrid {
  std::vector<std::array<double,3>> dirs;  // unit directions
  std::vector<double> weights;             // weights that sum to 4π
};

inline LebedevGrid load_lebedev_txt(const std::string& path) 
  {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open Lebedev file: " + path);
    LebedevGrid L;
    std::string line;
    double weight_sum = 0.0;
    while (std::getline(in, line)) {
      if (line.empty() || line[0]=='#') continue;
      std::istringstream ss(line);
      double theta, phi, w;
      if (!(ss >> theta >> phi >> w)) continue;
      theta *= M_PI / 180.0;
      phi *= M_PI / 180.0;
      L.dirs.push_back({ cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi) });
      weight_sum += w;
      L.weights.push_back(w);
    }
    if (L.dirs.empty()) throw std::runtime_error("Lebedev file had no data.");
    
    // Scale weights to sum to 4π (Lebedev quadrature weights should integrate over unit sphere)
    // If weights sum to ~1, they're normalized and need to be multiplied by 4π
    const double expected_sum = 4.0 * M_PI;
    if (std::abs(weight_sum - 1.0) < 0.1) {
      // Weights are normalized, scale by 4π
      for (auto& w : L.weights) {
        w *= expected_sum;
      }
    } else if (std::abs(weight_sum - expected_sum) > 0.1) {
      // Warn if weights don't sum to either 1 or 4π
      std::cerr << "Warning: Lebedev weights sum to " << weight_sum 
                << ", expected ~1 (normalized) or ~" << expected_sum << " (4π)\n";
    }
    
    return L;
  }

} // namespace PoIntInt

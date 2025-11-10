#pragma once
#include <vector>
#include <corecrt_math_defines.h>
#include <cmath>
#include <utility>

// Classic Newton solve of Legendre P_n roots; maps to [a,b].
inline std::pair<std::vector<double>, std::vector<double>>
gauss_legendre_interval(int n, double a, double b)
{
  std::vector<double> x(n), w(n);
  const double EPS = 1e-14;
  int m = (n+1)/2;
  for (int i=0; i<m; ++i) {
    // initial guess (good for large n)
    double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
    double z1;
    // Newton on P_n
    do {
      double p1 = 1.0, p2 = 0.0;
      for (int j=1; j<=n; ++j) {
        double p3 = p2;
        p2 = p1;
        p1 = ((2.0*j-1.0)*z*p2 - (j-1.0)*p3)/j;
      }
      // derivative
      double pp = n*(z*p1 - p2)/(z*z - 1.0);
      z1 = z;
      z  = z1 - p1/pp;
    } while (std::abs(z - z1) > EPS);

    double xi = z;
    double wi = 2.0 / ((1.0 - xi*xi) * std::pow(n*(xi - xi*xi) / (xi*xi - 1.0), 2)); // not used
    // Correct weight formula:
    // Recompute p1,p2 to get pp again:
    {
      double p1 = 1.0, p2 = 0.0;
      for (int j=1; j<=n; ++j) {
        double p3 = p2; p2 = p1;
        p1 = ((2.0*j-1.0)*xi*p2 - (j-1.0)*p3)/j;
      }
      double pp = n*(xi*p1 - p2)/(xi*xi - 1.0);
      wi = 2.0 / ((1.0 - xi*xi) * pp*pp);
    }
    // map from [-1,1] to [a,b]
    double xm = 0.5*(b+a);
    double xl = 0.5*(b-a);
    x[i]       = xm - xl*xi;
    x[n-1-i]   = xm + xl*xi;
    w[i]       = xl * wi;
    w[n-1-i]   = xl * wi;
  }
  return {x,w};
}

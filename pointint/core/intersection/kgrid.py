"""
K-space quadrature grid construction.

The k-space integral is discretized using a tensor-product quadrature:
- Radial: Gauss-Legendre nodes mapped to [0, ∞) via k = tan(t), t ∈ [0, π/2]
- Angular: Lebedev spherical quadrature

Math (README §3):
    k = (k_x, k_y, k_z) = r·(sin θ cos φ, sin θ sin φ, cos θ)

    ∫ f(k) d³k ≈ Σ_q w_q · f(k_q)

where w_q includes the Jacobian and quadrature weights.

Ref: cpp/include/quadrature/kgrid.hpp, cpp/include/quadrature/gauss_legendre.hpp
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class KGrid:
    """
    K-space quadrature grid.

    Attributes:
        dirs: Unit k-vectors, shape (Q, 3)
        kmag: |k| magnitudes, shape (Q,)
        weights: Full quadrature weights including Jacobian, shape (Q,)
    """

    dirs: np.ndarray  # (Q, 3)
    kmag: np.ndarray  # (Q,)
    weights: np.ndarray  # (Q,)

    @property
    def Q(self) -> int:
        """Number of k-nodes."""
        return len(self.kmag)


def load_lebedev(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Lebedev spherical quadrature grid from file.

    File format: Each line contains (phi_deg, theta_deg, weight) for a quadrature point.
    Phi is azimuthal angle, theta is polar angle from +z axis.
    The weights sum to 1 (normalized).

    Args:
        path: Path to Lebedev data file (e.g., data/lebedev/lebedev_077.txt)

    Returns:
        dirs: Unit vectors, shape (M, 3)
        weights: Angular weights, shape (M,), scaled to sum to 4π

    Ref: cpp/include/quadrature/lebedev_io.hpp
    """
    data = np.loadtxt(path)
    phi_deg = data[:, 0]
    theta_deg = data[:, 1]
    weights = data[:, 2]

    # Convert degrees to radians
    phi = np.radians(phi_deg)
    theta = np.radians(theta_deg)

    # Convert spherical to Cartesian (theta from +z axis)
    dirs = np.zeros((len(phi), 3))
    dirs[:, 0] = np.sin(theta) * np.cos(phi)  # x
    dirs[:, 1] = np.sin(theta) * np.sin(phi)  # y
    dirs[:, 2] = np.cos(theta)                # z

    # Scale weights to sum to 4π
    weights = weights * 4.0 * np.pi / np.sum(weights)

    return dirs, weights


def gauss_legendre(n: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Legendre quadrature nodes and weights on interval [a, b].

    Uses Newton's method to find roots of Legendre polynomials,
    then maps from [-1, 1] to [a, b].

    Args:
        n: Number of quadrature points
        a: Left endpoint
        b: Right endpoint

    Returns:
        nodes: Quadrature nodes, shape (n,)
        weights: Quadrature weights, shape (n,)

    Ref: cpp/include/quadrature/gauss_legendre.hpp
    """
    EPS = 1e-14
    x = np.zeros(n)
    w = np.zeros(n)
    m = (n + 1) // 2

    for i in range(m):
        # Initial guess
        z = np.cos(np.pi * (i + 0.75) / (n + 0.5))

        # Newton iteration on P_n
        while True:
            p1, p2 = 1.0, 0.0
            for j in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j

            # Derivative: P'_n = n(zP_n - P_{n-1}) / (z² - 1)
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z_old = z
            z = z - p1 / pp

            if abs(z - z_old) < EPS:
                break

        # Weight formula: w_i = 2 / ((1 - x_i²) P'_n(x_i)²)
        xi = z
        p1, p2 = 1.0, 0.0
        for j in range(1, n + 1):
            p3 = p2
            p2 = p1
            p1 = ((2.0 * j - 1.0) * xi * p2 - (j - 1.0) * p3) / j
        pp = n * (xi * p1 - p2) / (xi * xi - 1.0)
        wi = 2.0 / ((1.0 - xi * xi) * pp * pp)

        # Map from [-1, 1] to [a, b]
        xm = 0.5 * (b + a)
        xl = 0.5 * (b - a)
        x[i] = xm - xl * xi
        x[n - 1 - i] = xm + xl * xi
        w[i] = xl * wi
        w[n - 1 - i] = xl * wi

    return x, w


def build_kgrid(leb_dirs: np.ndarray, leb_weights: np.ndarray, n_radial: int) -> KGrid:
    """
    Build k-space quadrature grid from Lebedev directions and radial nodes.

    Combines angular (Lebedev) and radial (Gauss-Legendre) quadrature.
    Uses the transformation k = tan(t) with t ∈ [0, π/2] to map to [0, ∞).

    Math:
        d³k = k² dk dΩ
        With k = tan(t): dk = sec²(t) dt
        The k² in the kernel 1/k² cancels with k² dk
        Weight: w_q = w_angular × w_radial × sec²(t)

    Args:
        leb_dirs: Lebedev unit vectors, shape (M, 3)
        leb_weights: Lebedev angular weights, shape (M,), sum to 4π
        n_radial: Number of radial quadrature points

    Returns:
        KGrid with Q = M × n_radial nodes

    Ref: cpp/include/quadrature/kgrid.hpp:17-47
    """
    # Radial quadrature: t ∈ [0, π/2]
    t_nodes, t_weights = gauss_legendre(n_radial, 0.0, 0.5 * np.pi)

    M = len(leb_dirs)
    Q = M * n_radial

    dirs = np.zeros((Q, 3))
    kmag = np.zeros(Q)
    weights = np.zeros(Q)

    idx = 0
    for ir in range(n_radial):
        ti = t_nodes[ir]
        wti = t_weights[ir]

        # k = tan(t), sec²(t) = 1/cos²(t)
        cos_ti = np.cos(ti)
        sec2 = 1.0 / (cos_ti * cos_ti)
        k = np.tan(ti)

        for j in range(M):
            dirs[idx] = leb_dirs[j]
            kmag[idx] = k
            # Full weight: angular × radial × Jacobian
            weights[idx] = leb_weights[j] * wti * sec2
            idx += 1

    return KGrid(dirs=dirs, kmag=kmag, weights=weights)

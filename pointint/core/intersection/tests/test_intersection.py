"""
Tests for Warp-based intersection volume computation.

Validates correctness by comparing against:
1. Analytical formulas (unit cube, sphere)
2. NumPy/SciPy reference implementations

Run with: pytest pointint/intersection/tests/ -v

Ref: cpp/test/test_unit_cube.cpp, cpp/test/test_sphere_pointcloud.cpp
"""

import math
from pathlib import Path

import numpy as np
import pytest
import warp as wp

# Initialize Warp
wp.init()


# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture(scope="module")
def lebedev_path():
    """Path to Lebedev quadrature file."""
    # Try multiple possible locations (lebedev_077 has 770 points)
    candidates = [
        Path(__file__).parent.parent.parent.parent.parent / "data" / "lebedev" / "lebedev_077.txt",
        Path("/Users/scottgao/Documents/Coding/PoIntInt/data/lebedev/lebedev_077.txt"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    pytest.skip("Lebedev data file not found")


@pytest.fixture(scope="module")
def kgrid(lebedev_path):
    """Build k-grid for testing."""
    from pointint.intersection.kgrid import load_lebedev, build_kgrid

    leb_dirs, leb_weights = load_lebedev(lebedev_path)
    return build_kgrid(leb_dirs, leb_weights, n_radial=32)


# =============================================================================
# Helper functions
# =============================================================================


def create_unit_cube_mesh():
    """
    Create unit cube mesh centered at origin: [-0.5, 0.5]³.

    Returns:
        vertices: (8, 3) array
        faces: (12, 3) array of triangle indices

    Ref: cpp/src/geometry/geometry_helpers.cpp
    """
    # 8 vertices of unit cube
    vertices = np.array([
        [-0.5, -0.5, -0.5],
        [+0.5, -0.5, -0.5],
        [+0.5, +0.5, -0.5],
        [-0.5, +0.5, -0.5],
        [-0.5, -0.5, +0.5],
        [+0.5, -0.5, +0.5],
        [+0.5, +0.5, +0.5],
        [-0.5, +0.5, +0.5],
    ], dtype=np.float64)

    # 12 triangles (2 per face, 6 faces)
    # Outward-facing normals (right-hand rule)
    faces = np.array([
        # Bottom (z = -0.5)
        [0, 2, 1], [0, 3, 2],
        # Top (z = +0.5)
        [4, 5, 6], [4, 6, 7],
        # Front (y = -0.5)
        [0, 1, 5], [0, 5, 4],
        # Back (y = +0.5)
        [2, 3, 7], [2, 7, 6],
        # Left (x = -0.5)
        [0, 4, 7], [0, 7, 3],
        # Right (x = +0.5)
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)

    return vertices, faces


def box_box_intersection_volume(c1, h1, c2, h2):
    """
    Analytical intersection volume of two axis-aligned boxes.

    Args:
        c1: Center of box 1, shape (3,)
        h1: Half-extents of box 1, shape (3,)
        c2: Center of box 2, shape (3,)
        h2: Half-extents of box 2, shape (3,)

    Returns:
        Intersection volume

    Ref: cpp/src/analytical_intersection.cpp:15-30
    """
    overlap = np.zeros(3)
    for i in range(3):
        min1, max1 = c1[i] - h1[i], c1[i] + h1[i]
        min2, max2 = c2[i] - h2[i], c2[i] + h2[i]
        overlap_min = max(min1, min2)
        overlap_max = min(max1, max2)
        overlap[i] = max(0.0, overlap_max - overlap_min)
    return overlap[0] * overlap[1] * overlap[2]


def sphere_sphere_intersection_volume(c1, r1, c2, r2):
    """
    Analytical intersection volume of two spheres.

    Args:
        c1: Center of sphere 1, shape (3,)
        r1: Radius of sphere 1
        c2: Center of sphere 2, shape (3,)
        r2: Radius of sphere 2

    Returns:
        Intersection volume

    Ref: cpp/src/analytical_intersection.cpp:32-60
    """
    d = np.linalg.norm(np.array(c2) - np.array(c1))

    # No intersection
    if d >= r1 + r2:
        return 0.0

    # One inside the other
    if d <= abs(r1 - r2):
        r_min = min(r1, r2)
        return (4.0 * math.pi / 3.0) * r_min**3

    # Partial intersection (lens formula)
    term1 = (r1 + r2 - d) ** 2
    term2 = d**2 + 2 * d * (r1 + r2) - 3 * (r1**2 + r2**2) + 6 * r1 * r2
    return (math.pi * term1 * term2) / (12.0 * d)


def exact_cube_form_factor(kx, ky, kz):
    """
    Analytical form factor F(k) for unit cube.

    Math:
        F(k) = ∏_{i=x,y,z} sinc(k_i / 2)

    where sinc(x) = sin(x) / x.

    Ref: README §4.1, cpp/src/form_factor_helpers.cpp:144-152
    """
    def sinc(x):
        if abs(x) < 1e-4:
            return 1.0 - x**2 / 6.0 + x**4 / 120.0
        return np.sin(x) / x

    return sinc(0.5 * kx) * sinc(0.5 * ky) * sinc(0.5 * kz)


def exact_cube_Ak_squared(kx, ky, kz):
    """
    Analytical |A(k)|² for unit cube.

    Math:
        |A(k)|² = |k|² · |F(k)|²

    Ref: cpp/src/form_factor_helpers.cpp:154-159
    """
    k2 = kx**2 + ky**2 + kz**2
    F = exact_cube_form_factor(kx, ky, kz)
    return k2 * F * F


# =============================================================================
# Test: Primitives
# =============================================================================


class TestPrimitives:
    """Test mathematical primitive functions."""

    def test_E_func_values(self):
        """
        Test E(z) = (sin z + i(1-cos z)) / z against numpy.

        Ref: cpp/include/cuda/cuda_helpers.hpp:200-216
        """
        from pointint.intersection.primitives import E_func

        # Test kernel to extract E_func values
        @wp.kernel
        def eval_E_func(z_vals: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.vec2d)):
            i = wp.tid()
            out[i] = E_func(z_vals[i])

        test_z = [0.001, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0]
        z_wp = wp.array(test_z, dtype=wp.float64)
        out_wp = wp.zeros(len(test_z), dtype=wp.vec2d)

        wp.launch(eval_E_func, dim=len(test_z), inputs=[z_wp], outputs=[out_wp])
        results = out_wp.numpy()

        for i, z in enumerate(test_z):
            # Reference: E(z) = (sin z + i(1-cos z)) / z
            if abs(z) < 1e-10:
                expected_re, expected_im = 1.0, 0.0
            else:
                expected_re = np.sin(z) / z
                expected_im = (1.0 - np.cos(z)) / z

            got_re, got_im = results[i][0], results[i][1]

            assert abs(got_re - expected_re) < 1e-6, f"E({z}) real: got {got_re}, expected {expected_re}"
            assert abs(got_im - expected_im) < 1e-6, f"E({z}) imag: got {got_im}, expected {expected_im}"

    def test_J1_over_x_values(self):
        """
        Test J₁(x)/x against scipy.

        Ref: cpp/include/cuda/cuda_helpers.hpp:105-154
        """
        pytest.importorskip("scipy")
        from scipy.special import j1

        from pointint.intersection.primitives import J1_over_x

        @wp.kernel
        def eval_J1_over_x(x_vals: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
            i = wp.tid()
            out[i] = J1_over_x(x_vals[i])

        test_x = [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 15.0]
        x_wp = wp.array(test_x, dtype=wp.float64)
        out_wp = wp.zeros(len(test_x), dtype=wp.float64)

        wp.launch(eval_J1_over_x, dim=len(test_x), inputs=[x_wp], outputs=[out_wp])
        results = out_wp.numpy()

        for i, x in enumerate(test_x):
            if abs(x) < 1e-10:
                expected = 0.5  # limit as x->0
            else:
                expected = j1(x) / x

            got = results[i]
            rel_err = abs(got - expected) / (abs(expected) + 1e-10)

            # 1e-3 tolerance for asymptotic expansion at large x
            assert rel_err < 1e-3, f"J1({x})/x: got {got}, expected {expected}, rel_err {rel_err}"


# =============================================================================
# Test: K-Grid
# =============================================================================


class TestKGrid:
    """Test k-space quadrature grid construction."""

    def test_gauss_legendre_integral(self):
        """
        Test Gauss-Legendre by integrating x² over [0, 1].

        ∫₀¹ x² dx = 1/3

        Ref: cpp/include/quadrature/gauss_legendre.hpp
        """
        from pointint.intersection.kgrid import gauss_legendre

        nodes, weights = gauss_legendre(16, 0.0, 1.0)

        # Integrate x²
        integral = np.sum(weights * nodes**2)
        expected = 1.0 / 3.0

        assert abs(integral - expected) < 1e-10, f"∫x²dx = {integral}, expected {expected}"

    def test_build_kgrid_shape(self, lebedev_path):
        """Test k-grid has correct shape Q = M × n_radial."""
        from pointint.intersection.kgrid import load_lebedev, build_kgrid

        leb_dirs, leb_weights = load_lebedev(lebedev_path)
        M = len(leb_dirs)
        n_radial = 16

        kgrid = build_kgrid(leb_dirs, leb_weights, n_radial)

        assert kgrid.Q == M * n_radial
        assert kgrid.dirs.shape == (kgrid.Q, 3)
        assert kgrid.kmag.shape == (kgrid.Q,)
        assert kgrid.weights.shape == (kgrid.Q,)


# =============================================================================
# Test: Form Factors
# =============================================================================


class TestFormFactors:
    """Test form factor computation."""

    def test_unit_cube_form_factor(self, kgrid):
        """
        Test |A(k)|² for unit cube against analytical formula.

        Math:
            |A(k)|² = |k|² · ∏ sinc²(k_i/2)

        Tolerance: 10% (mesh discretization error)

        Ref: cpp/test/test_unit_cube.cpp:22-85
        """
        from pointint.intersection.form_factors import pack_triangles, compute_A_triangles

        vertices, faces = create_unit_cube_mesh()
        tris = pack_triangles(vertices, faces)

        A = compute_A_triangles(tris, kgrid.dirs, kgrid.kmag)
        A_np = A.numpy()

        # Sample some k-points and compare
        test_indices = [0, 10, 50, 100, min(200, kgrid.Q - 1)]
        max_rel_err = 0.0

        for idx in test_indices:
            k = kgrid.kmag[idx]
            if k < 0.01:  # Skip very small k
                continue

            kdir = kgrid.dirs[idx]
            kx, ky, kz = k * kdir[0], k * kdir[1], k * kdir[2]

            # Computed |A|²
            A_re, A_im = A_np[idx][0], A_np[idx][1]
            Ak2_computed = A_re**2 + A_im**2

            # Analytical |A|²
            Ak2_exact = exact_cube_Ak_squared(kx, ky, kz)

            if Ak2_exact > 1e-10:
                rel_err = abs(Ak2_computed - Ak2_exact) / Ak2_exact
                max_rel_err = max(max_rel_err, rel_err)

        assert max_rel_err < 0.1, f"Max relative error {max_rel_err:.2%} > 10%"


# =============================================================================
# Test: Volume Computation
# =============================================================================


class TestVolume:
    """Test intersection volume computation."""

    def test_unit_cube_self_volume(self, kgrid):
        """
        Test self-intersection volume of unit cube = 1.0.

        Vol(Ω ∩ Ω) = Vol(Ω) = 1.0

        Tolerance: 10% (k-space quadrature discretization error)

        Ref: cpp/test/test_unit_cube.cpp:150-178
        """
        from pointint.intersection.form_factors import pack_triangles, compute_A_triangles
        from pointint.intersection.volume import intersection_volume

        vertices, faces = create_unit_cube_mesh()
        tris = pack_triangles(vertices, faces)

        A = compute_A_triangles(tris, kgrid.dirs, kgrid.kmag)
        volume = intersection_volume(A, A, kgrid)

        expected = 1.0
        rel_err = abs(volume - expected) / expected

        assert rel_err < 0.1, f"Volume = {volume:.6f}, expected {expected}, error {rel_err:.2%}"

    def test_box_box_intersection(self, kgrid):
        """
        Test intersection volume of two overlapping cubes.

        Two unit cubes with centers at (0,0,0) and (0.5,0,0).
        Overlap = 0.5 × 1.0 × 1.0 = 0.5

        Tolerance: 10%

        Ref: cpp/src/analytical_intersection.cpp:15-30
        """
        from pointint.intersection.form_factors import pack_triangles, compute_A_triangles
        from pointint.intersection.volume import intersection_volume

        vertices, faces = create_unit_cube_mesh()
        tris1 = pack_triangles(vertices, faces)

        # Shift second cube by (0.5, 0, 0)
        t2 = np.array([0.5, 0.0, 0.0])

        A1 = compute_A_triangles(tris1, kgrid.dirs, kgrid.kmag, translation=None)
        A2 = compute_A_triangles(tris1, kgrid.dirs, kgrid.kmag, translation=t2)

        volume = intersection_volume(A1, A2, kgrid)

        # Analytical
        c1, h1 = np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])
        c2, h2 = np.array([0.5, 0, 0]), np.array([0.5, 0.5, 0.5])
        expected = box_box_intersection_volume(c1, h1, c2, h2)

        rel_err = abs(volume - expected) / (expected + 1e-10)

        assert rel_err < 0.1, f"Volume = {volume:.6f}, expected {expected:.6f}, error {rel_err:.2%}"

    def test_disjoint_cubes(self, kgrid):
        """
        Test intersection volume of two disjoint cubes = 0.

        Two unit cubes separated by distance > 1.

        Ref: cpp/src/analytical_intersection.cpp
        """
        from pointint.intersection.form_factors import pack_triangles, compute_A_triangles
        from pointint.intersection.volume import intersection_volume

        vertices, faces = create_unit_cube_mesh()
        tris1 = pack_triangles(vertices, faces)

        # Shift second cube far away
        t2 = np.array([2.0, 0.0, 0.0])

        A1 = compute_A_triangles(tris1, kgrid.dirs, kgrid.kmag, translation=None)
        A2 = compute_A_triangles(tris1, kgrid.dirs, kgrid.kmag, translation=t2)

        volume = intersection_volume(A1, A2, kgrid)

        # Should be close to 0
        assert abs(volume) < 0.05, f"Disjoint volume = {volume:.6f}, expected ~0"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Form factor computation kernels.

The form factor A(k) is the Fourier transform of the surface normal field:
    A(k) = ∫_∂Ω exp(ik·x) n(x) dS_x

This module provides GPU kernels to compute A(k) for different geometry types:
- Triangles: Mesh surfaces
- Disks: Oriented point clouds
- Gaussians: Gaussian splats

Ref: cpp/src/form_factor_helpers.cpp, cpp/src/dof/cuda/affine_dof_cuda.cu
"""

import warp as wp
import numpy as np
from typing import Tuple

from pointint.core.intersection.primitives import Phi_ab, J1_over_x, cmul, cexp_i, cscale


# =============================================================================
# Data structures
# =============================================================================

# Triangle packed data: vertex a, edges e1/e2, area vector S = 0.5*(e1 × e2)
@wp.struct
class Tri:
    a: wp.vec3d    # Base vertex
    e1: wp.vec3d   # Edge vector 1: v1 - a
    e2: wp.vec3d   # Edge vector 2: v2 - a
    S: wp.vec3d    # Area vector: 0.5 * (e1 × e2), |S| = triangle area


# =============================================================================
# Triangle form factor kernel
# =============================================================================


@wp.kernel
def compute_A_triangles_kernel(
    tris: wp.array(dtype=Tri),
    k_dirs: wp.array(dtype=wp.vec3d),
    k_mags: wp.array(dtype=wp.float64),
    t: wp.vec3d,
    A_mat: wp.mat33d,
    A_out: wp.array(dtype=wp.vec2d),
):
    """
    Compute form factor A(k) for triangle mesh with affine transform.

    Each thread processes one k-point, summing contributions from all triangles.

    Math:
        A(k_q) = Σ_tri exp(ik·a') · Φ(k·e₁', k·e₂') · (k̂·S')

    where the transformed geometry is:
        a' = A·a + t       (transformed vertex)
        e₁' = A·e₁         (transformed edge 1)
        e₂' = A·e₂         (transformed edge 2)
        S' = ½(e₁' × e₂')  (transformed area vector)

    Args:
        tris: Triangle data array, shape (N_tri,)
        k_dirs: Unit k-vectors, shape (Q,)
        k_mags: |k| magnitudes, shape (Q,)
        t: Translation vector
        A_mat: 3×3 affine transformation matrix
        A_out: Output A(k), shape (Q,), complex as vec2d

    Ref: README §2, cpp/src/dof/cuda/affine_dof_cuda.cu:43-136
    """
    q = wp.tid()

    khat = k_dirs[q]
    k = k_mags[q]

    # Skip k=0 (A(0) = 0 for closed surfaces)
    if k < wp.float64(1.0e-10):
        A_out[q] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
        return

    # k vector
    kx = k * khat[0]
    ky = k * khat[1]
    kz = k * khat[2]

    # Accumulate A(k) over all triangles
    A_re = wp.float64(0.0)
    A_im = wp.float64(0.0)

    n_tris = tris.shape[0]
    for i in range(n_tris):
        tri = tris[i]

        # Transform vertex: a' = A·a + t
        a_prime = A_mat @ tri.a + t

        # Transform edges: e' = A·e
        e1_prime = A_mat @ tri.e1
        e2_prime = A_mat @ tri.e2

        # Transformed area vector: S' = 0.5 * (e1' × e2')
        S_prime = wp.vec3d(
            wp.float64(0.5) * (e1_prime[1] * e2_prime[2] - e1_prime[2] * e2_prime[1]),
            wp.float64(0.5) * (e1_prime[2] * e2_prime[0] - e1_prime[0] * e2_prime[2]),
            wp.float64(0.5) * (e1_prime[0] * e2_prime[1] - e1_prime[1] * e2_prime[0]),
        )

        # Compute α = k·e₁', β = k·e₂', γ = k̂·S'
        alpha = kx * e1_prime[0] + ky * e1_prime[1] + kz * e1_prime[2]
        beta = kx * e2_prime[0] + ky * e2_prime[1] + kz * e2_prime[2]
        gamma = khat[0] * S_prime[0] + khat[1] * S_prime[1] + khat[2] * S_prime[2]

        # Φ(α, β)
        phi = Phi_ab(alpha, beta)

        # Phase: exp(ik·a')
        phase = kx * a_prime[0] + ky * a_prime[1] + kz * a_prime[2]
        exp_phase = cexp_i(phase)

        # A_tri = exp(ik·a') · Φ(α,β) · γ
        A_tri = cmul(exp_phase, phi)
        A_tri = cscale(A_tri, gamma)

        A_re = A_re + A_tri[0]
        A_im = A_im + A_tri[1]

    A_out[q] = wp.vec2d(A_re, A_im)


# =============================================================================
# High-level API
# =============================================================================


def pack_triangles(vertices: np.ndarray, faces: np.ndarray) -> wp.array:
    """
    Pack triangle mesh into Tri struct array.

    Args:
        vertices: Vertex positions, shape (V, 3)
        faces: Triangle indices, shape (F, 3)

    Returns:
        Warp array of Tri structs, shape (F,)

    Ref: cpp/src/geometry/packing.cpp
    """
    F = len(faces)
    tris_np = np.zeros(F, dtype=Tri.numpy_dtype())

    for i, face in enumerate(faces):
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        e1 = v1 - v0
        e2 = v2 - v0
        S = 0.5 * np.cross(e1, e2)

        tris_np[i]["a"] = v0
        tris_np[i]["e1"] = e1
        tris_np[i]["e2"] = e2
        tris_np[i]["S"] = S

    return wp.array(tris_np, dtype=Tri)


def compute_A_triangles(
    tris: wp.array,
    k_dirs: np.ndarray,
    k_mags: np.ndarray,
    translation: np.ndarray = None,
    rotation: np.ndarray = None,
) -> wp.array:
    """
    Compute form factor A(k) for all k-points.

    Args:
        tris: Triangle data (from pack_triangles)
        k_dirs: Unit k-vectors, shape (Q, 3)
        k_mags: |k| magnitudes, shape (Q,)
        translation: Translation vector, shape (3,), default zeros
        rotation: 3×3 rotation/affine matrix, default identity

    Returns:
        Warp array of complex A(k), shape (Q,), as vec2d

    Example:
        >>> tris = pack_triangles(vertices, faces)
        >>> kgrid = build_kgrid(leb_dirs, leb_w, n_radial=32)
        >>> A = compute_A_triangles(tris, kgrid.dirs, kgrid.kmag)
    """
    Q = len(k_mags)

    # Default transform: identity
    if translation is None:
        translation = np.zeros(3)
    if rotation is None:
        rotation = np.eye(3)

    # Convert to Warp types
    k_dirs_wp = wp.array(k_dirs, dtype=wp.vec3d)
    k_mags_wp = wp.array(k_mags, dtype=wp.float64)
    t_wp = wp.vec3d(translation[0], translation[1], translation[2])
    A_mat_wp = wp.mat33d(
        rotation[0, 0], rotation[0, 1], rotation[0, 2],
        rotation[1, 0], rotation[1, 1], rotation[1, 2],
        rotation[2, 0], rotation[2, 1], rotation[2, 2],
    )

    # Output buffer
    A_out = wp.zeros(Q, dtype=wp.vec2d)

    # Launch kernel
    wp.launch(
        kernel=compute_A_triangles_kernel,
        dim=Q,
        inputs=[tris, k_dirs_wp, k_mags_wp, t_wp, A_mat_wp],
        outputs=[A_out],
    )

    return A_out

"""
Intersection volume computation.

Computes the intersection volume between 3D solids using the Fourier
(k-space) method. The volume is computed as a weighted inner product
of form factors.

Math (README §2):
    Vol(Ω₁ ∩ Ω₂) = (1/8π³) · Σ_q w_q · Re(A₁(k_q) · conj(A₂(k_q)))

where A(k) is the form factor and w_q are the quadrature weights.

Ref: cpp/src/compute_intersection_volume.cu
"""

import math

import numpy as np
import warp as wp

from pointint.core.intersection.kgrid import KGrid
from pointint.core.intersection.form_factors import compute_A_triangles, pack_triangles


# =============================================================================
# Volume computation kernel
# =============================================================================


@wp.kernel
def intersection_volume_kernel(
    A1: wp.array(dtype=wp.vec2d),
    A2: wp.array(dtype=wp.vec2d),
    weights: wp.array(dtype=wp.float64),
    out: wp.array(dtype=wp.float64),
):
    """
    Compute intersection volume from form factors.

    Each thread processes one k-point, accumulating the weighted
    inner product via atomic addition.

    Math:
        V = (1/8π³) · Σ_q w_q · Re(A₁(k_q) · conj(A₂(k_q)))
          = (1/8π³) · Σ_q w_q · (A₁.re · A₂.re + A₁.im · A₂.im)

    Note: The 1/8π³ scaling is applied after the kernel.

    Args:
        A1: Form factor of geometry 1, shape (Q,)
        A2: Form factor of geometry 2, shape (Q,)
        weights: Quadrature weights, shape (Q,)
        out: Output scalar (atomic accumulator), shape (1,)

    Ref: cpp/src/compute_intersection_volume.cu:33-50
    """
    q = wp.tid()

    a1 = A1[q]
    a2 = A2[q]
    w = weights[q]

    # Re(A1 · conj(A2)) = A1.re·A2.re + A1.im·A2.im
    vol_contrib = a1[0] * a2[0] + a1[1] * a2[1]

    wp.atomic_add(out, 0, w * vol_contrib)


# =============================================================================
# High-level API
# =============================================================================


def intersection_volume(
    A1: wp.array,
    A2: wp.array,
    kgrid: KGrid,
) -> float:
    """
    Compute intersection volume from precomputed form factors.

    Args:
        A1: Form factor of geometry 1, shape (Q,), from compute_A_triangles
        A2: Form factor of geometry 2, shape (Q,), from compute_A_triangles
        kgrid: K-space quadrature grid

    Returns:
        Intersection volume (scalar)

    Math:
        Vol(Ω₁ ∩ Ω₂) = (1/8π³) · Σ_q w_q · Re(A₁(k_q) · conj(A₂(k_q)))

    Example:
        >>> A1 = compute_A_triangles(tris1, kgrid.dirs, kgrid.kmag, t1)
        >>> A2 = compute_A_triangles(tris2, kgrid.dirs, kgrid.kmag, t2)
        >>> vol = intersection_volume(A1, A2, kgrid)

    Ref: cpp/src/compute_intersection_volume.cu:172-489
    """
    Q = kgrid.Q

    # Convert weights to Warp array
    weights_wp = wp.array(kgrid.weights, dtype=wp.float64)

    # Output accumulator
    out = wp.zeros(1, dtype=wp.float64)

    # Launch kernel
    wp.launch(
        kernel=intersection_volume_kernel,
        dim=Q,
        inputs=[A1, A2, weights_wp],
        outputs=[out],
    )

    # Apply scaling factor: 1/(8π³)
    scale = 1.0 / (8.0 * math.pi**3)
    result = out.numpy()[0] * scale

    return result


def intersection_volume_mesh(
    vertices1: np.ndarray,
    faces1: np.ndarray,
    vertices2: np.ndarray,
    faces2: np.ndarray,
    kgrid: KGrid,
    translation1: np.ndarray = None,
    translation2: np.ndarray = None,
) -> float:
    """
    Compute intersection volume between two triangle meshes.

    Convenience function that handles packing and form factor computation.

    Args:
        vertices1: Vertices of mesh 1, shape (V1, 3)
        faces1: Faces of mesh 1, shape (F1, 3)
        vertices2: Vertices of mesh 2, shape (V2, 3)
        faces2: Faces of mesh 2, shape (F2, 3)
        kgrid: K-space quadrature grid
        translation1: Translation for mesh 1, default zeros
        translation2: Translation for mesh 2, default zeros

    Returns:
        Intersection volume (scalar)

    Example:
        >>> from pointint.intersection import load_lebedev, build_kgrid
        >>> leb_dirs, leb_w = load_lebedev("data/lebedev/lebedev_074.txt")
        >>> kgrid = build_kgrid(leb_dirs, leb_w, n_radial=32)
        >>> vol = intersection_volume_mesh(V1, F1, V2, F2, kgrid)
    """
    # Pack triangles
    tris1 = pack_triangles(vertices1, faces1)
    tris2 = pack_triangles(vertices2, faces2)

    # Compute form factors
    A1 = compute_A_triangles(tris1, kgrid.dirs, kgrid.kmag, translation1)
    A2 = compute_A_triangles(tris2, kgrid.dirs, kgrid.kmag, translation2)

    # Compute volume
    return intersection_volume(A1, A2, kgrid)


def self_volume(A: wp.array, kgrid: KGrid) -> float:
    """
    Compute self-intersection volume (= volume of the solid).

    This is Vol(Ω ∩ Ω) = Vol(Ω).

    Args:
        A: Form factor, shape (Q,)
        kgrid: K-space quadrature grid

    Returns:
        Volume of the solid

    Note: For better accuracy, use the divergence theorem:
        V = (1/3) ∫_∂Ω (x·n) dS
    """
    return intersection_volume(A, A, kgrid)

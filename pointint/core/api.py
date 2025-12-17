import numpy as np
from pointint.core.intersection import intersection_volume_direct
from pointint.core.area import compute_voronoi_area_weights


def get_ptcld_intersection_volume(
    P1: np.ndarray,
    N1: np.ndarray,
    P2: np.ndarray,
    N2: np.ndarray,
    eps: float = 1e-3,
    k_neighbors: int = 30,
    device: str = None,
) -> float:
    """
    Compute intersection volume between two oriented point clouds.

    Area weights are computed automatically via Voronoi cell cross-sections
    (GCNO paper, Sec 4.4).

    Args:
        P1: Positions of cloud 1, shape (N1, 3)
        N1: Unit normals of cloud 1, shape (N1, 3)
        P2: Positions of cloud 2, shape (N2, 3)
        N2: Unit normals of cloud 2, shape (N2, 3)
        eps: Regularization parameter (default 1e-3)
        k_neighbors: Max neighbors for Voronoi cell construction (default 30)
        device: Computation device

    Returns:
        Intersection volume as scalar float

    Example:
        >>> vol = get_ptcld_intersection_volume(P1, N1, P2, N2)
    """
    W1 = compute_voronoi_area_weights(P1, k_neighbors=k_neighbors)
    W2 = compute_voronoi_area_weights(P2, k_neighbors=k_neighbors)
    return intersection_volume_direct(P1, N1, W1, P2, N2, W2, eps=eps, device=device)


def get_mesh_intersection_volume(
    vertices1: np.ndarray,
    faces1: np.ndarray,
    vertices2: np.ndarray,
    faces2: np.ndarray,
    eps: float = 1e-3,
    device: str = None,
) -> float:
    """
    Compute intersection volume between two triangle meshes.

    Each triangle is represented by its centroid, unit normal, and area.

    Args:
        vertices1: Vertices of mesh 1, shape (V1, 3)
        faces1: Faces of mesh 1, shape (F1, 3)
        vertices2: Vertices of mesh 2, shape (V2, 3)
        faces2: Faces of mesh 2, shape (F2, 3)
        eps: Regularization parameter (default 1e-3)
        device: Warp device

    Returns:
        Intersection volume as scalar float

    Example:
        >>> vol = pointint_volume_mesh(V1, F1, V2, F2)
    """
    x, n1, w = _mesh_to_points(vertices1, faces1)
    y, n2, v = _mesh_to_points(vertices2, faces2)
    return pointint_volume(x, n1, w, y, n2, v, eps=eps, device=device)

# =============================================================================
# Helpers
# =============================================================================


def _mesh_to_points(vertices: np.ndarray, faces: np.ndarray):
    """
    Extract centroid, unit normal, and area for each triangle.

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) triangle indices

    Returns:
        centroids: (F, 3)
        normals: (F, 3) unit normals
        areas: (F,)
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    centroids = (v0 + v1 + v2) / 3.0

    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    # Unit normals (handle degenerate triangles)
    norms = 2.0 * areas
    norms = np.where(norms < 1e-12, 1.0, norms)
    normals = cross / norms[:, None]

    return centroids.astype(np.float32), normals.astype(np.float32), areas.astype(np.float32)

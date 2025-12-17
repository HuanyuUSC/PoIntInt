"""
Voronoi-based area weight for point clouds.

Computes per-point area weights using Voronoi cell cross-sections.
Ref: GCNO paper (Globally Consistent Normal Orientation), Sec 4.4

Algorithm for each point p_i:
1. Build Voronoi cell via AABB + bisector plane cutting
2. Identify 3-corner vertices (>=3 distinct face normals)
3. Find farthest 3-corner vertex as pole
4. Cut cell with plane through p_i, normal toward pole
5. Cross-section polygon area = weight omega_i
"""
import numpy as np
from scipy.spatial import cKDTree


# =============================================================================
# Helper Functions
# =============================================================================


def _box_vertices(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    """Return 8 vertices of axis-aligned bounding box."""
    return np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
    ])


def _box_faces() -> list:
    """Return 6 faces of box as vertex index lists (CCW winding)."""
    return [
        [0, 3, 2, 1],  # -Z
        [4, 5, 6, 7],  # +Z
        [0, 1, 5, 4],  # -Y
        [2, 3, 7, 6],  # +Y
        [0, 4, 7, 3],  # -X
        [1, 2, 6, 5],  # +X
    ]


def _perpendicular(v: np.ndarray) -> np.ndarray:
    """Return an arbitrary unit vector perpendicular to v."""
    v = v / np.linalg.norm(v)
    if abs(v[0]) < 0.9:
        u = np.cross(v, np.array([1., 0., 0.]))
    else:
        u = np.cross(v, np.array([0., 1., 0.]))
    return u / np.linalg.norm(u)


def _face_normal(verts: np.ndarray, face: list) -> np.ndarray:
    """Compute unit normal of a face (CCW vertex order)."""
    v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-12 else np.array([0., 0., 1.])


# =============================================================================
# Convex Polyhedron Plane Cutting
# =============================================================================


def _cut_convex_by_plane(verts: np.ndarray, faces: list,
                         plane_pt: np.ndarray, plane_n: np.ndarray):
    """
    Cut convex polyhedron by plane, keep the negative half-space side.

    Args:
        verts: (M, 3) vertices
        faces: list of vertex index lists
        plane_pt: point on plane
        plane_n: plane normal (points toward positive half-space to remove)

    Returns:
        new_verts: (M', 3) vertices after cutting
        new_faces: list of vertex index lists
    """
    # Signed distance of each vertex to plane
    d = np.dot(verts - plane_pt, plane_n)

    # All vertices on negative side -> no cut needed
    if np.all(d <= 1e-10):
        return verts, faces

    # All vertices on positive side -> empty result
    if np.all(d >= -1e-10):
        return np.zeros((0, 3)), []

    # Build new vertices and faces
    new_verts_list = list(verts)
    vert_map = {i: i for i in range(len(verts))}  # old -> new index
    edge_intersect = {}  # (i,j) -> new vertex index

    def get_intersect(i, j):
        """Get or create intersection vertex on edge (i,j)."""
        key = (min(i, j), max(i, j))
        if key in edge_intersect:
            return edge_intersect[key]
        # Compute intersection point
        t = d[i] / (d[i] - d[j])
        pt = verts[i] + t * (verts[j] - verts[i])
        idx = len(new_verts_list)
        new_verts_list.append(pt)
        edge_intersect[key] = idx
        return idx

    new_faces = []
    cap_edges = []  # edges on the cutting plane for cap face

    for face in faces:
        n = len(face)
        new_face = []
        for k in range(n):
            i, j = face[k], face[(k + 1) % n]
            di, dj = d[i], d[j]

            # Vertex i on negative side -> keep it
            if di <= 1e-10:
                new_face.append(i)

            # Edge crosses plane
            if (di > 1e-10) != (dj > 1e-10):
                inter_idx = get_intersect(i, j)
                new_face.append(inter_idx)
                # Record edge for cap face
                if di > 1e-10:
                    cap_edges.append((inter_idx, new_face[-2] if len(new_face) > 1 else None))

        if len(new_face) >= 3:
            new_faces.append(new_face)

    # Build cap face from intersection edges
    if edge_intersect:
        cap_verts = list(edge_intersect.values())
        if len(cap_verts) >= 3:
            # Order cap vertices by angle around centroid
            cap_pts = np.array([new_verts_list[i] for i in cap_verts])
            centroid = cap_pts.mean(axis=0)
            u = _perpendicular(plane_n)
            v = np.cross(plane_n, u)
            angles = np.arctan2(np.dot(cap_pts - centroid, v),
                               np.dot(cap_pts - centroid, u))
            order = np.argsort(angles)
            cap_face = [cap_verts[i] for i in order]
            new_faces.append(cap_face)

    return np.array(new_verts_list), new_faces


# =============================================================================
# 3-Corner Vertex Detection
# =============================================================================


def _cluster_normals(normals: list, eps: float = 0.0005) -> int:
    """
    Cluster normals by direction similarity.
    Returns number of distinct clusters.
    """
    if len(normals) == 0:
        return 0

    normals = [n / np.linalg.norm(n) if np.linalg.norm(n) > 1e-12 else n
               for n in normals]
    clusters = []

    for n in normals:
        found = False
        for c in clusters:
            # Check similarity (distance between unit vectors)
            dist = min(np.linalg.norm(n - c), np.linalg.norm(n + c))
            if dist < eps:
                found = True
                break
        if not found:
            clusters.append(n)

    return len(clusters)


def _find_3corner_vertices(verts: np.ndarray, faces: list,
                           eps: float = 0.0005) -> np.ndarray:
    """
    Find 3-corner vertices (vertices with >2 distinct adjacent face normals).

    Returns:
        mask: (N,) boolean array, True for 3-corner vertices
    """
    n_verts = len(verts)
    mask = np.zeros(n_verts, dtype=bool)

    # Precompute face normals
    face_normals = [_face_normal(verts, f) for f in faces]

    # Build vertex -> face adjacency
    vert_faces = [[] for _ in range(n_verts)]
    for fi, face in enumerate(faces):
        for vi in face:
            if vi < n_verts:
                vert_faces[vi].append(fi)

    # Check each vertex
    for vi in range(n_verts):
        adj_normals = [face_normals[fi] for fi in vert_faces[vi]]
        n_clusters = _cluster_normals(adj_normals, eps)
        if n_clusters > 2:
            mask[vi] = True

    return mask


# =============================================================================
# Polygon Area
# =============================================================================


def _polygon_area_3d(verts: np.ndarray, normal: np.ndarray) -> float:
    """
    Compute area of 3D polygon by projecting to local 2D coords.
    Uses shoelace formula.
    """
    if len(verts) < 3:
        return 0.0

    # Build orthonormal basis on plane
    u = _perpendicular(normal)
    v = np.cross(normal, u)

    # Project to 2D
    origin = verts[0]
    coords_2d = np.array([[np.dot(p - origin, u), np.dot(p - origin, v)]
                          for p in verts])

    # Shoelace formula
    x, y = coords_2d[:, 0], coords_2d[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _intersect_cell_with_plane(verts: np.ndarray, faces: list,
                               plane_pt: np.ndarray, plane_n: np.ndarray) -> np.ndarray:
    """
    Find intersection polygon of convex cell with plane.

    Returns:
        polygon_verts: (K, 3) vertices of intersection polygon, ordered
    """
    # Signed distance
    d = np.dot(verts - plane_pt, plane_n)

    # Collect intersection points on edges
    intersect_pts = []
    edges_seen = set()

    for face in faces:
        n = len(face)
        for k in range(n):
            i, j = face[k], face[(k + 1) % n]
            key = (min(i, j), max(i, j))
            if key in edges_seen:
                continue
            edges_seen.add(key)

            di, dj = d[i], d[j]
            # Edge crosses plane
            if (di > 1e-10 and dj < -1e-10) or (di < -1e-10 and dj > 1e-10):
                t = di / (di - dj)
                pt = verts[i] + t * (verts[j] - verts[i])
                intersect_pts.append(pt)

    if len(intersect_pts) < 3:
        return np.zeros((0, 3))

    # Order by angle around centroid
    pts = np.array(intersect_pts)
    centroid = pts.mean(axis=0)
    u = _perpendicular(plane_n)
    v = np.cross(plane_n, u)
    angles = np.arctan2(np.dot(pts - centroid, v), np.dot(pts - centroid, u))
    order = np.argsort(angles)

    return pts[order]


# =============================================================================
# Main Function
# =============================================================================


def compute_voronoi_area_weights(points: np.ndarray,
                                  k_neighbors: int = 30,
                                  bbox_scale: float = 1.35) -> np.ndarray:
    """
    Compute area weight for each point using Voronoi cell cross-section.

    Replicates GCNO paper Sec 4.4 algorithm:
    1. Build Voronoi cell via AABB + bisector plane cutting
    2. Find 3-corner vertices (>2 distinct face normal directions)
    3. Use farthest 3-corner vertex as pole
    4. Cut cell with plane through point, normal toward pole
    5. Cross-section area = weight

    Args:
        points: (N, 3) point cloud coordinates
        k_neighbors: max neighbors for cell construction (default 30)
        bbox_scale: bounding box scale factor (default 1.35)

    Returns:
        weights: (N,) area weight for each point
    """
    N = len(points)
    weights = np.zeros(N)

    # Compute global AABB
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    extent = (bbox_max - bbox_min) / 2 * bbox_scale

    # KNN
    tree = cKDTree(points)
    k = min(k_neighbors + 1, N)  # +1 because query includes self
    _, neighbor_indices = tree.query(points, k=k)

    for i in range(N):
        point = points[i]
        neighbors = points[neighbor_indices[i, 1:]]  # exclude self

        # Initialize with scaled AABB
        cell_verts = _box_vertices(center - extent, center + extent)
        cell_faces = _box_faces()

        # Cut by bisector planes
        for nb in neighbors:
            mid = (point + nb) / 2
            normal = nb - point
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            cell_verts, cell_faces = _cut_convex_by_plane(
                cell_verts, cell_faces, mid, normal)
            if len(cell_verts) == 0:
                break

        if len(cell_verts) < 4:
            continue

        # Find 3-corner vertices
        corner_mask = _find_3corner_vertices(cell_verts, cell_faces)
        corner_verts = cell_verts[corner_mask]

        if len(corner_verts) == 0:
            # Fallback: use all vertices
            corner_verts = cell_verts

        # Find pole (farthest 3-corner vertex)
        dists = np.linalg.norm(corner_verts - point, axis=1)
        pole = corner_verts[np.argmax(dists)]

        # Plane through point, normal toward pole
        plane_n = pole - point
        norm = np.linalg.norm(plane_n)
        if norm < 1e-12:
            continue
        plane_n = plane_n / norm

        # Get intersection polygon
        section_verts = _intersect_cell_with_plane(
            cell_verts, cell_faces, point, plane_n)

        if len(section_verts) >= 3:
            weights[i] = _polygon_area_3d(section_verts, plane_n)

    return weights


def compute_voronoi_area_weights_with_geometry(
    points: np.ndarray,
    k_neighbors: int = 30,
    bbox_scale: float = 1.35
) -> tuple:
    """
    Compute area weights and return geometry for visualization.

    Returns:
        weights: (N,) area weights
        sections: list of (K, 3) section polygon vertices for each point
        poles: (N, 3) pole positions
    """
    N = len(points)
    weights = np.zeros(N)
    sections = [None] * N
    poles = np.zeros((N, 3))

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    extent = (bbox_max - bbox_min) / 2 * bbox_scale

    tree = cKDTree(points)
    k = min(k_neighbors + 1, N)
    _, neighbor_indices = tree.query(points, k=k)

    for i in range(N):
        point = points[i]
        neighbors = points[neighbor_indices[i, 1:]]

        cell_verts = _box_vertices(center - extent, center + extent)
        cell_faces = _box_faces()

        for nb in neighbors:
            mid = (point + nb) / 2
            normal = nb - point
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            cell_verts, cell_faces = _cut_convex_by_plane(
                cell_verts, cell_faces, mid, normal)
            if len(cell_verts) == 0:
                break

        if len(cell_verts) < 4:
            continue

        corner_mask = _find_3corner_vertices(cell_verts, cell_faces)
        corner_verts = cell_verts[corner_mask] if corner_mask.any() else cell_verts

        dists = np.linalg.norm(corner_verts - point, axis=1)
        pole = corner_verts[np.argmax(dists)]
        poles[i] = pole

        plane_n = pole - point
        norm = np.linalg.norm(plane_n)
        if norm < 1e-12:
            continue
        plane_n = plane_n / norm

        section_verts = _intersect_cell_with_plane(
            cell_verts, cell_faces, point, plane_n)

        if len(section_verts) >= 3:
            weights[i] = _polygon_area_3d(section_verts, plane_n)
            sections[i] = section_verts

    return weights, sections, poles


# =============================================================================
# Visualization (viser)
# =============================================================================


def visualize_area_weights(points: np.ndarray,
                           k_neighbors: int = 30,
                           port: int = 8080):
    """
    Interactive 3D visualization of Voronoi area weights.

    Opens a viser server at http://localhost:{port}
    - Points colored by area weight (colormap)
    - Cross-section polygons displayed
    - Click point to see area value

    Args:
        points: (N, 3) point cloud
        k_neighbors: max neighbors for cell construction
        port: viser server port
    """
    import viser

    print("Computing area weights...")
    weights, sections, poles = compute_voronoi_area_weights_with_geometry(
        points, k_neighbors)

    # Normalize weights for colormap
    w_min, w_max = weights.min(), weights.max()
    if w_max > w_min:
        w_norm = (weights - w_min) / (w_max - w_min)
    else:
        w_norm = np.ones_like(weights) * 0.5

    # Color map: blue (small) -> red (large)
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    colors[:, 0] = (w_norm * 255).astype(np.uint8)  # R
    colors[:, 2] = ((1 - w_norm) * 255).astype(np.uint8)  # B

    # Start server
    server = viser.ViserServer(port=port)
    print(f"Viser server started at http://localhost:{port}")

    # Add point cloud
    server.scene.add_point_cloud(
        "/points",
        points=points.astype(np.float32),
        colors=colors,
        point_size=0.02,
    )

    # Add cross-section meshes
    for i, section in enumerate(sections):
        if section is None or len(section) < 3:
            continue

        # Triangulate polygon (fan triangulation)
        n_verts = len(section)
        faces = []
        for j in range(1, n_verts - 1):
            faces.append([0, j, j + 1])

        # Color based on weight
        r, g, b = colors[i]
        mesh_color = (r / 255, g / 255, b / 255)

        server.scene.add_mesh_simple(
            f"/sections/{i}",
            vertices=section.astype(np.float32),
            faces=np.array(faces, dtype=np.uint32),
            color=mesh_color,
            opacity=0.6,
        )

    # Add GUI
    with server.gui.add_folder("Area Weights"):
        server.gui.add_markdown(f"**Points:** {len(points)}")
        server.gui.add_markdown(f"**Min area:** {w_min:.4f}")
        server.gui.add_markdown(f"**Max area:** {w_max:.4f}")
        server.gui.add_markdown(f"**Sum:** {weights.sum():.4f}")

        # Point selector
        point_idx = server.gui.add_slider(
            "Point Index",
            min=0, max=len(points) - 1, step=1, initial_value=0
        )
        area_text = server.gui.add_markdown(f"**Area[0]:** {weights[0]:.6f}")

        @point_idx.on_update
        def _on_point_update(event):
            idx = point_idx.value
            area_text.content = f"**Area[{idx}]:** {weights[idx]:.6f}"

            # Highlight selected point
            highlight_colors = colors.copy()
            highlight_colors[idx] = [255, 255, 0]  # Yellow
            server.scene.add_point_cloud(
                "/points",
                points=points.astype(np.float32),
                colors=highlight_colors,
                point_size=0.02,
            )

    print("Visualization ready. Press Ctrl+C to stop.")
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping server...")


if __name__ == "__main__":
    # Demo with random points on sphere surface
    np.random.seed(42)
    n = 100
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.arccos(np.random.uniform(-1, 1, n))
    r = 1.0
    points = np.stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi)
    ], axis=1)

    visualize_area_weights(points)

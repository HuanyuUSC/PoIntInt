"""
PoIntInt Mesh Intersection Volume Demo

Load arbitrary meshes and compute intersection volume with percentage.
Control the distance between two meshes interactively.

Usage:
    python demo_mesh_intersection.py [mesh_path]

    If no mesh_path is provided, uses the default fox.obj mesh.

Then open http://localhost:8080 in your browser.
"""

import viser
import numpy as np
import trimesh
import time
import os
import sys

# Import our C++ bindings
import pointint_cpp as pi

# ============================================================================
# Mesh Loading
# ============================================================================

def load_mesh(path: str) -> tuple:
    """Load mesh from file and return vertices (V) and faces (F)."""
    mesh = trimesh.load(path, force='mesh')
    V = np.array(mesh.vertices, dtype=np.float64)
    F = np.array(mesh.faces, dtype=np.int32)
    return V, F

def normalize_mesh(V: np.ndarray) -> np.ndarray:
    """Center and scale mesh to fit in [-0.5, 0.5]^3."""
    center = (V.max(axis=0) + V.min(axis=0)) / 2
    V_centered = V - center
    scale = V_centered.max() - V_centered.min()
    if scale > 0:
        V_centered = V_centered / scale
    return V_centered

# ============================================================================
# Setup
# ============================================================================

# Get mesh path from command line or use default
if len(sys.argv) > 1:
    mesh_path = sys.argv[1]
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_path = os.path.join(script_dir, "../data/assets/fox.obj")

if not os.path.exists(mesh_path):
    print(f"Error: Mesh file not found at {mesh_path}")
    exit(1)

print(f"Loading mesh from: {mesh_path}")
V_orig, F = load_mesh(mesh_path)
V = normalize_mesh(V_orig)
print(f"Mesh loaded: {V.shape[0]} vertices, {F.shape[0]} faces")

# Create viser server
server = viser.ViserServer(port=8080)

# Load Lebedev quadrature grid
script_dir = os.path.dirname(os.path.abspath(__file__))
lebedev_path = os.path.join(script_dir, "../data/lebedev/lebedev_029.txt")

if not os.path.exists(lebedev_path):
    print(f"Error: Lebedev file not found at {lebedev_path}")
    exit(1)

print(f"Loading Lebedev grid from: {lebedev_path}")
dirs, weights = pi.load_lebedev(lebedev_path)
kgrid = pi.build_kgrid(dirs, weights, 32)
print(f"KGrid created with {len(dirs) * 32} nodes")

# Pre-compute base geometry and its volume
geom_base = pi.make_triangle_mesh(V, F)
base_volume = pi.compute_volume_cpu(geom_base)
print(f"Base mesh volume: {base_volume:.6f}")

# ============================================================================
# GUI Setup
# ============================================================================

with server.gui.add_folder("Settings"):
    backend_dropdown = server.gui.add_dropdown(
        "Backend",
        options=["CUDA", "CPU"],
        initial_value="CUDA"
    )
    distance_slider = server.gui.add_slider(
        "Distance",
        min=0.0,
        max=2.0,
        step=0.01,
        initial_value=0.3
    )
    direction_dropdown = server.gui.add_dropdown(
        "Direction",
        options=["X", "Y", "Z"],
        initial_value="X"
    )
    n_radial_slider = server.gui.add_slider(
        "N_radial",
        min=8,
        max=64,
        step=8,
        initial_value=32
    )

with server.gui.add_folder("Results"):
    vol1_text = server.gui.add_text("Volume (Mesh 1)", initial_value="--")
    vol2_text = server.gui.add_text("Volume (Mesh 2)", initial_value="--")
    intersection_text = server.gui.add_text("Intersection Vol", initial_value="--")
    pct1_text = server.gui.add_text("% of Mesh 1", initial_value="--")
    pct2_text = server.gui.add_text("% of Mesh 2", initial_value="--")
    time_text = server.gui.add_text("Compute Time", initial_value="--")

# ============================================================================
# Update Function
# ============================================================================

def update(_=None):
    global kgrid

    d = distance_slider.value
    direction = direction_dropdown.value
    use_cuda = backend_dropdown.value == "CUDA"
    n_radial = int(n_radial_slider.value)

    # Rebuild kgrid if n_radial changed
    kgrid = pi.build_kgrid(dirs, weights, n_radial)

    # Create translation vector based on direction
    if direction == "X":
        translation = np.array([d, 0.0, 0.0])
    elif direction == "Y":
        translation = np.array([0.0, d, 0.0])
    else:  # Z
        translation = np.array([0.0, 0.0, d])

    # Translate second mesh
    V2 = pi.translate_points(V, translation)

    # Create geometries
    geom1 = pi.make_triangle_mesh(V, F)
    geom2 = pi.make_triangle_mesh(V2, F)

    # Compute volumes
    t0 = time.time()
    vol1 = pi.compute_volume_cpu(geom1)
    vol2 = pi.compute_volume_cpu(geom2)

    # Compute intersection volume
    if use_cuda:
        intersection_vol = pi.compute_intersection_volume_cuda(geom1, geom2, kgrid)
    else:
        intersection_vol = pi.compute_intersection_volume_cpu(geom1, geom2, kgrid)
    elapsed = time.time() - t0

    # Calculate percentages
    pct1 = (intersection_vol / vol1 * 100) if vol1 > 1e-10 else 0.0
    pct2 = (intersection_vol / vol2 * 100) if vol2 > 1e-10 else 0.0

    # Update visualization
    server.scene.add_mesh_simple(
        "/mesh1",
        vertices=V.astype(np.float32),
        faces=F,
        color=(0.2, 0.6, 1.0),
        opacity=0.7
    )
    server.scene.add_mesh_simple(
        "/mesh2",
        vertices=V2.astype(np.float32),
        faces=F,
        color=(1.0, 0.4, 0.2),
        opacity=0.7
    )

    # Update result display
    vol1_text.value = f"{vol1:.6f}"
    vol2_text.value = f"{vol2:.6f}"
    intersection_text.value = f"{intersection_vol:.6f}"
    pct1_text.value = f"{pct1:.2f}%"
    pct2_text.value = f"{pct2:.2f}%"
    time_text.value = f"{elapsed * 1000:.1f} ms"

# ============================================================================
# Event Handlers
# ============================================================================

@distance_slider.on_update
def _on_distance_change(event):
    update()

@direction_dropdown.on_update
def _on_direction_change(event):
    update()

@backend_dropdown.on_update
def _on_backend_change(event):
    update()

@n_radial_slider.on_update
def _on_nradial_change(event):
    update()

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("PoIntInt Mesh Intersection Volume Demo")
    print("=" * 50)
    print(f"Mesh: {os.path.basename(mesh_path)}")
    print(f"Vertices: {V.shape[0]}, Faces: {F.shape[0]}")
    print(f"Base Volume: {base_volume:.6f}")
    print("=" * 50)
    print(f"Server running at http://localhost:8080")
    print("Press Ctrl+C to exit")
    print("=" * 50)

    # Initial update
    update()

    # Keep server running
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")

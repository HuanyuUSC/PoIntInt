"""
PoIntInt Intersection Volume Demo

Interactive visualization of intersection volume computation
between two objects (Box-Box or Sphere-Sphere).

Usage:
    python demo_intersection.py

Then open http://localhost:8080 in your browser.
"""

import viser
import numpy as np
import time
import os

# Import our C++ bindings
import pointint_cpp as pi

# Warp backends
import warp as wp
from pointint.core.intersection import (
    build_kgrid as build_kgrid_warp,
    load_lebedev as load_lebedev_warp,
)
from pointint.core.intersection.volume import intersection_volume_mesh
from pointint.core.intersection.volume_direct import intersection_volume_direct

# ============================================================================
# Setup
# ============================================================================

server = viser.ViserServer(port=8080)

# Load Lebedev quadrature grid
script_dir = os.path.dirname(os.path.abspath(__file__))
lebedev_path = os.path.join(script_dir, "../data/lebedev/lebedev_029.txt")

if not os.path.exists(lebedev_path):
    print(f"Error: Lebedev file not found at {lebedev_path}")
    print("Please ensure the data/lebedev directory exists with lebedev_*.txt files")
    exit(1)

print(f"Loading Lebedev grid from: {lebedev_path}")
dirs, weights = pi.load_lebedev(lebedev_path)
kgrid = pi.build_kgrid(dirs, weights, 32)
print(f"KGrid created with {len(dirs) * 32} nodes")

# Warp setup
wp.init()
leb_dirs_warp, leb_w_warp = load_lebedev_warp(lebedev_path)
kgrid_warp = build_kgrid_warp(leb_dirs_warp, leb_w_warp, n_radial=32)
print(f"Warp KGrid created")

# Pre-create geometries
V_cube, F_cube = pi.create_unit_cube_mesh()
P_sphere, N_sphere, R_sphere = pi.create_sphere_pointcloud(1000)

print(f"Unit cube: {V_cube.shape[0]} vertices, {F_cube.shape[0]} faces")
print(f"Unit sphere: {P_sphere.shape[0]} points")

# ============================================================================
# GUI Setup
# ============================================================================

with server.gui.add_folder("Settings"):
    mode_dropdown = server.gui.add_dropdown(
        "Mode",
        options=["Box-Box", "Sphere-Sphere"],
        initial_value="Box-Box"
    )
    backend_dropdown = server.gui.add_dropdown(
        "Backend",
        options=["Warp Fourier", "Warp Direct", "C++ CUDA", "C++ CPU"],
        initial_value="Warp Fourier"
    )
    distance_slider = server.gui.add_slider(
        "Distance",
        min=0.0,
        max=2.5,
        step=0.01,
        initial_value=0.5
    )
    n_radial_slider = server.gui.add_slider(
        "N_radial",
        min=8,
        max=96,
        step=8,
        initial_value=32
    )

with server.gui.add_folder("Results"):
    volume_text = server.gui.add_text("Volume (ours)", initial_value="--")
    gt_text = server.gui.add_text("Ground Truth", initial_value="--")
    error_text = server.gui.add_text("Rel. Error", initial_value="--")
    time_text = server.gui.add_text("Compute Time", initial_value="--")

# ============================================================================
# Update Function
# ============================================================================

def update(_=None):
    global kgrid, kgrid_warp

    d = distance_slider.value
    mode = mode_dropdown.value
    backend = backend_dropdown.value
    n_radial = int(n_radial_slider.value)

    # Rebuild kgrids if n_radial changed
    kgrid = pi.build_kgrid(dirs, weights, n_radial)
    kgrid_warp = build_kgrid_warp(leb_dirs_warp, leb_w_warp, n_radial)

    if mode == "Box-Box":
        # Two unit cubes, second one translated along x-axis
        V2 = V_cube + np.array([d, 0.0, 0.0])
        geom1 = pi.make_triangle_mesh(V_cube, F_cube)
        geom2 = pi.make_triangle_mesh(V2, F_cube)

        # Compute intersection volume
        t0 = time.time()
        if backend == "Warp Fourier":
            vol_ours = intersection_volume_mesh(V_cube, F_cube, V2, F_cube, kgrid_warp)
        elif backend == "Warp Direct":
            vol_ours = intersection_volume_direct(V_cube, F_cube, V2, F_cube)
        elif backend == "C++ CUDA":
            vol_ours = pi.compute_intersection_volume_cuda(geom1, geom2, kgrid)
        else:  # C++ CPU
            vol_ours = pi.compute_intersection_volume_cpu(geom1, geom2, kgrid)
        elapsed = time.time() - t0

        # Ground truth (analytical)
        vol_gt = pi.box_box_intersection_volume(
            np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5]),
            np.array([d, 0.0, 0.0]), np.array([0.5, 0.5, 0.5])
        )

        # Visualization - meshes
        server.scene.add_mesh_simple(
            "/obj1",
            vertices=V_cube,
            faces=F_cube,
            color=(0.2, 0.6, 1.0),
            opacity=0.7
        )
        server.scene.add_mesh_simple(
            "/obj2",
            vertices=V2,
            faces=F_cube,
            color=(1.0, 0.4, 0.2),
            opacity=0.7
        )

    else:  # Sphere-Sphere (point cloud)
        P2 = P_sphere + np.array([d, 0.0, 0.0])
        geom1 = pi.make_point_cloud(P_sphere, N_sphere, R_sphere)
        geom2 = pi.make_point_cloud(P2, N_sphere, R_sphere)

        # Compute intersection volume (Warp backends not supported for point cloud)
        t0 = time.time()
        if backend in ["Warp Fourier", "Warp Direct"]:
            # Fall back to C++ CUDA for point cloud
            vol_ours = pi.compute_intersection_volume_cuda(geom1, geom2, kgrid)
        elif backend == "C++ CUDA":
            vol_ours = pi.compute_intersection_volume_cuda(geom1, geom2, kgrid)
        else:  # C++ CPU
            vol_ours = pi.compute_intersection_volume_cpu(geom1, geom2, kgrid)
        elapsed = time.time() - t0

        # Ground truth (analytical)
        vol_gt = pi.sphere_sphere_intersection_volume(
            np.array([0.0, 0.0, 0.0]), 1.0,
            np.array([d, 0.0, 0.0]), 1.0
        )

        # Visualization - point clouds
        colors1 = np.tile(np.array([[51, 153, 255]], dtype=np.uint8), (P_sphere.shape[0], 1))
        colors2 = np.tile(np.array([[255, 102, 51]], dtype=np.uint8), (P2.shape[0], 1))

        server.scene.add_point_cloud(
            "/obj1",
            points=P_sphere.astype(np.float32),
            colors=colors1,
            point_size=0.02
        )
        server.scene.add_point_cloud(
            "/obj2",
            points=P2.astype(np.float32),
            colors=colors2,
            point_size=0.02
        )

    # Update result display
    volume_text.value = f"{vol_ours:.6f}"
    gt_text.value = f"{vol_gt:.6f}"

    if vol_gt > 1e-10:
        rel_err = abs(vol_ours - vol_gt) / vol_gt * 100
        error_text.value = f"{rel_err:.2f}%"
    else:
        if abs(vol_ours) < 1e-10:
            error_text.value = "0.00%"
        else:
            error_text.value = "N/A (GT=0)"

    time_text.value = f"{elapsed * 1000:.1f} ms"

# ============================================================================
# Event Handlers
# ============================================================================

@distance_slider.on_update
def _on_distance_change(event):
    update()

@mode_dropdown.on_update
def _on_mode_change(event):
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
    print("PoIntInt Intersection Volume Demo")
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

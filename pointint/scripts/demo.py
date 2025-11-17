import sys

import torch
import kaolin as kal
from loguru import logger
import numpy as np
import pyvista as pv
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
import warp as wp

import pointint


SPHERE_RADIUS = 0.1
SPHERE_RESOLUTION = 3
SPHERE_DISTANCE = 0.5
SPHERE_HEIGHT = 0.8
NUM_SAMPLES = 50_000
GRAVITY = (0.0, 0.0, 0.0)
FLOOR_HEIGHT = 0.0
BALL_SPEED = 1.0  # Initial speed (m/s) for balls running towards each other


def create_sphere_mesh(radius=SPHERE_RADIUS, subdivisions=SPHERE_RESOLUTION):
    """Create a GPU icosphere mesh as a Kaolin SurfaceMesh."""
    sphere = pv.Icosphere(radius=radius, nsub=subdivisions)
    vertices = torch.tensor(sphere.points, dtype=torch.float32, device="cuda")
    faces = sphere.faces.reshape(-1, 4)[:, 1:4]
    faces = torch.tensor(faces, dtype=torch.int64, device="cuda")
    return kal.rep.SurfaceMesh(vertices, faces)


def sample_mesh_interior(mesh, num_samples=NUM_SAMPLES):
    """Sample interior points from a mesh bounding box using Kaolin sign test."""
    min_corner = mesh.vertices.min(dim=0).values
    max_corner = mesh.vertices.max(dim=0).values
    uniform_pts = torch.rand(num_samples, 3, device=mesh.vertices.device) * (max_corner - min_corner) + min_corner
    inside = kal.ops.mesh.check_sign(
        mesh.vertices.unsqueeze(0), mesh.faces, uniform_pts.unsqueeze(0), hash_resolution=512
    ).squeeze()
    return uniform_pts[inside]


def create_sim_object(pts, youngs=1e5, poisson=0.45, density=500.0, volume=0.5, handles=5, steps=5000):
    """Train a Simplicits simulation object from interior samples."""
    yms = torch.full((pts.shape[0],), youngs, device=pts.device)
    prs = torch.full((pts.shape[0],), poisson, device=pts.device)
    rhos = torch.full((pts.shape[0],), density, device=pts.device)

    logger.info(f"Training Simplicits object with {steps} steps and {handles} handles")

    return pointint.simplicits.easy_api.SimplicitsObject.create_trained(
        pts,
        yms,
        prs,
        rhos,
        volume,
        num_handles=handles,
        training_num_steps=steps,
        training_lr_start=1e-3,
        training_lr_end=1e-3,
        training_le_coeff=1e-1,
        training_lo_coeff=1e6,
        training_log_every=max(1, steps // 10),
        normalize_for_training=True,
    )


def setup_scene(sim_objs, gravity=GRAVITY, floor_height=FLOOR_HEIGHT, timestep=0.03):
    """Create a scene with gravity and a floor constraint."""
    scene = pointint.simplicits.easy_api.SimplicitsScene()
    scene.max_newton_steps = 5
    scene.timestep = timestep
    scene.direct_solve = True

    obj_indices = [scene.add_object(obj) for obj in sim_objs]

    device = sim_objs[0].pts.device
    dtype = sim_objs[0].pts.dtype
    scene.set_scene_gravity(acc_gravity=torch.tensor(gravity, device=device, dtype=dtype))
    scene.set_scene_floor(floor_height=floor_height, floor_axis=2, floor_penalty=1000.0)

    return scene, obj_indices


def create_qt_app(meshes, orig_verts, sim_objs, scene, obj_indices, steps=200, period_ms=33):
    """Build a Qt window with PyVista visualization for the falling spheres."""
    window = QMainWindow()
    window.setWindowTitle("PoIntInt Two Spheres")

    central_widget = QWidget()
    layout = QVBoxLayout()
    central_widget.setLayout(layout)
    window.setCentralWidget(central_widget)

    control_layout = QHBoxLayout()
    restart_btn = QPushButton("Restart")
    play_pause_btn = QPushButton("Pause")
    control_layout.addWidget(restart_btn)
    control_layout.addWidget(play_pause_btn)
    control_layout.addStretch()
    layout.addLayout(control_layout)

    plotter = QtInteractor(window)
    floor = pv.Plane(center=(0, 0, FLOOR_HEIGHT), direction=(0, 0, 1), i_size=2.0, j_size=2.0)
    plotter.add_mesh(floor, color="lightgray", opacity=0.5)

    polys = []
    palette = ["orange", "cyan"]
    for idx, mesh in enumerate(meshes):
        verts = mesh.vertices.detach().cpu().numpy()
        faces = mesh.faces.detach().cpu().numpy()
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
        poly = pv.PolyData(verts, faces_pv)
        plotter.add_mesh(poly, color=palette[idx % len(palette)], smooth_shading=True, copy_mesh=False)
        polys.append(poly)

    plotter.add_axes()
    plotter.camera.position = (0, -3, 1)
    plotter.camera.focal_point = (0, 0, FLOOR_HEIGHT)
    plotter.camera.up = (0, 0, 1)
    layout.addWidget(plotter.interactor)

    step_state = {"i": 0, "scene": scene, "obj_indices": obj_indices}

    def _tick():
        if step_state["i"] >= steps:
            timer.stop()
            play_pause_btn.setText("Play")
            return

        step_state["scene"].run_sim_step()
        for obj_id, poly, rest in zip(step_state["obj_indices"], polys, orig_verts):
            deformed = step_state["scene"].get_object_deformed_pts(obj_id, rest).detach().cpu().numpy()
            poly.points = deformed
        plotter.render()
        step_state["i"] += 1

    def restart():
        timer.stop()
        new_scene, new_obj_indices = setup_scene(sim_objs)
        step_state["scene"] = new_scene
        step_state["obj_indices"] = new_obj_indices
        step_state["i"] = 0
        for poly, rest in zip(polys, orig_verts):
            poly.points = rest.detach().cpu().numpy()
        plotter.render()
        play_pause_btn.setText("Play")

    def toggle_play_pause():
        if timer.isActive():
            timer.stop()
            play_pause_btn.setText("Play")
        else:
            timer.start(period_ms)
            play_pause_btn.setText("Pause")

    restart_btn.clicked.connect(restart)
    play_pause_btn.clicked.connect(toggle_play_pause)

    timer = QTimer()
    timer.timeout.connect(_tick)
    timer.start(period_ms)

    window.resize(800, 600)
    return window


if __name__ == "__main__":
    logger.info("Creating two spheres for gravity simulation")

    meshes = []
    orig_verts = []
    sim_objs = []

    for offset in [-SPHERE_DISTANCE / 2.0, SPHERE_DISTANCE / 2.0]:
        mesh = create_sphere_mesh()
        mesh.vertices[:, 0] += offset
        mesh.vertices[:, 2] += SPHERE_HEIGHT

        orig_verts.append(mesh.vertices.clone())

        pts = sample_mesh_interior(mesh, NUM_SAMPLES)
        logger.info(f"Sphere at x-offset {offset:.2f}: {pts.shape[0]} interior points")

        sim_obj = create_sim_object(pts)
        sim_objs.append(sim_obj)
        meshes.append(mesh)

    scene, obj_indices = setup_scene(sim_objs)

    # Set initial speeds - make balls run towards each other
    # We must set sim_z_dot (velocity) instead of sim_z_prev
    # because run_sim_step() overwrites sim_z_prev at the beginning!
    t_z_dot = wp.to_torch(scene.sim_z_dot).clone()

    # Get DOF indices for each object
    obj0_dofs = wp.to_torch(scene.object_to_z_map[obj_indices[0]])
    obj1_dofs = wp.to_torch(scene.object_to_z_map[obj_indices[1]])

    # Set initial velocity directly in DOF space
    # The last handle's Tx is at index -9 (12 DOFs per handle, offset 3 for Tx)
    t_z_dot[obj0_dofs[-9]] = BALL_SPEED   # left ball → move right (+x direction)
    t_z_dot[obj1_dofs[-9]] = -BALL_SPEED  # right ball → move left (-x direction)

    # Use wp.copy to properly update warp array
    wp.copy(src=wp.from_torch(t_z_dot, dtype=wp.float32), dest=scene.sim_z_dot)

    logger.info(f"Initial velocity set: speed={BALL_SPEED} m/s in x-direction")

    # Enable collisions
    scene.enable_collisions(collision_particle_radius=0.1, collision_penalty=5000.0)

    app = QApplication(sys.argv)
    window = create_qt_app(meshes, orig_verts, sim_objs, scene, obj_indices, steps=200, period_ms=33)
    window.show()
    sys.exit(app.exec())

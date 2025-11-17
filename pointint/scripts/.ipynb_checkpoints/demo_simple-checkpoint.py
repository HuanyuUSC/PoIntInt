import torch
import kaolin as kal
from loguru import logger
from tqdm import tqdm
import numpy as np
import pyvista as pv
import os
import pointint
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
from PySide6.QtCore import QTimer
from pyvistaqt import QtInteractor



def load_and_sample_mesh(mesh_path, num_samples=200_000):
    """Load mesh, center it, and sample interior points."""
    mesh = kal.io.import_mesh(mesh_path, triangulate=True).cuda()
    mesh.vertices = kal.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)
    orig_vertices = mesh.vertices.clone()
    logger.info(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Sample interior points
    min_corner = orig_vertices.min(dim=0).values
    max_corner = orig_vertices.max(dim=0).values
    uniform_pts = torch.rand(num_samples, 3, device='cuda') * (max_corner - min_corner) + min_corner
    inside = kal.ops.mesh.check_sign(
        mesh.vertices.unsqueeze(0), mesh.faces, uniform_pts.unsqueeze(0), hash_resolution=512
    ).squeeze()
    pts = uniform_pts[inside]
    logger.info(f"Sampled {len(pts)} interior points")
    
    return mesh, orig_vertices, pts


def create_sim_object(pts, youngs=1e5, poisson=0.45, density=500.0, volume=0.5, handles=5, steps=10000):
    """Create and train a Simplicits simulation object with progress bar."""
    yms = torch.full((pts.shape[0],), youngs, device="cuda")
    prs = torch.full((pts.shape[0],), poisson, device="cuda")
    rhos = torch.full((pts.shape[0],), density, device="cuda")
    
    logger.info(f"Training Simplicits object: {steps} steps, {handles} handles")
    
    sim_obj = pointint.simplicits.easy_api.SimplicitsObject.create_trained(
            pts, yms, prs, rhos, volume,
            num_handles=handles,
            training_num_steps=steps,
            training_lr_start=1e-3,
            training_lr_end=1e-3,
            training_le_coeff=1e-1,
            training_lo_coeff=1e6,
            training_log_every=max(1, steps // 10),
            normalize_for_training=True
        )
    
    logger.info("Training complete")
    return sim_obj
def setup_scene(sim_obj, gravity=(0, 9.8, 0), floor_height=-0.8, timestep=0.03):
    """Create scene and add physics."""
    scene = pointint.simplicits.easy_api.SimplicitsScene()
    scene.max_newton_steps = 5
    scene.timestep = timestep
    scene.direct_solve = True
    
    obj_idx = scene.add_object(sim_obj)
    scene.set_scene_gravity(acc_gravity=torch.tensor(gravity))
    scene.set_scene_floor(floor_height=floor_height, floor_axis=1, floor_penalty=1000)
    
    return scene, obj_idx


def simulate(scene, obj_idx, mesh, orig_vertices, steps=100):
    """Run a minimal simulation loop with tqdm and update mesh vertices each step."""
    # reset to rest state
    mesh.vertices = orig_vertices
    for _ in tqdm(range(steps), desc="Simulating", unit="step"):
        scene.run_sim_step()
        mesh.vertices = scene.get_object_deformed_pts(obj_idx, orig_vertices)
    return mesh


def create_qt_app(mesh, sim_obj, scene, obj_idx, orig_vertices, steps=200, period_ms=33):
    """Create a Qt window with PyVista visualization and controls."""
    # Build initial PolyData
    verts = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    poly = pv.PolyData(verts, faces_pv)

    # Create Qt window
    window = QMainWindow()
    window.setWindowTitle("PoIntInt Demo")

    # Create central widget and layout
    central_widget = QWidget()
    layout = QVBoxLayout()
    central_widget.setLayout(layout)
    window.setCentralWidget(central_widget)

    # Control panel
    control_layout = QHBoxLayout()
    restart_btn = QPushButton("Restart")
    play_pause_btn = QPushButton("Pause")
    control_layout.addWidget(restart_btn)
    control_layout.addWidget(play_pause_btn)
    control_layout.addStretch()
    layout.addLayout(control_layout)

    # Create PyVista QtInteractor
    plotter = QtInteractor(window)
    floor = pv.Plane(center=(0, -0.8, 0), direction=(0, 1, 0), i_size=3.0, j_size=3.0)
    plotter.add_mesh(floor, color='lightgray', opacity=0.5)
    plotter.add_mesh(poly, color='orange', smooth_shading=True, copy_mesh=False)
    plotter.add_axes()
    plotter.view_isometric()
    layout.addWidget(plotter.interactor)

    # Animation state
    step_state = {'i': 0, 'scene': scene, 'obj_idx': obj_idx}

    def _tick():
        if step_state['i'] >= steps:
            timer.stop()
            play_pause_btn.setText("Play")
            return
        step_state['scene'].run_sim_step()
        deformed = step_state['scene'].get_object_deformed_pts(step_state['obj_idx'], orig_vertices).detach().cpu().numpy()
        poly.points = deformed
        plotter.render()
        step_state['i'] += 1

    def restart():
        timer.stop()
        new_scene, new_obj_idx = setup_scene(sim_obj)
        step_state['scene'] = new_scene
        step_state['obj_idx'] = new_obj_idx
        step_state['i'] = 0
        mesh.vertices = orig_vertices.clone()
        poly.points = orig_vertices.detach().cpu().numpy()
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

    # Setup timer for animation
    timer = QTimer()
    timer.timeout.connect(_tick)
    timer.start(period_ms)

    window.resize(800, 600)
    return window


if __name__ == "__main__":
    assets_path = "assets"
    mesh_path = os.path.join(assets_path, "fox.obj")
    mesh, orig_vertices, pts = load_and_sample_mesh(mesh_path)
    sim_obj = create_sim_object(pts)
    scene, obj_idx = setup_scene(sim_obj)

    app = QApplication(sys.argv)
    window = create_qt_app(mesh, sim_obj, scene, obj_idx, orig_vertices, steps=200, period_ms=33)
    window.show()
    sys.exit(app.exec())


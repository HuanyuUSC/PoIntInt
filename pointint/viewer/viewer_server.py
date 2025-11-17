"""
Minimal viser server for PoIntInt.

The goal is to offer a dead-simple Web UI that can be launched with
``uv run pointint/viewer/viewer_server.py``.  It exposes two controls:

* a checkbox to toggle continuous simulation;
* a slider to tweak the timestep.

Each tick updates a synthetic point cloud so the UI can be exercised even
before the real simulator hooks are ready.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import viser


StepFn = Callable[[float], np.ndarray]


@dataclass
class ViewerConfig:
  """Basic knobs for the viewer server."""

  host: str = "0.0.0.0"
  port: int = 8098
  title: str = "PoIntInt Viewer"
  target_hz: float = 30.0


class ViewerServer:
  """Thin wrapper around :class:`viser.ViserServer`."""

  def __init__(self,
               config: ViewerConfig,
               step_fn: Optional[StepFn] = None):
    self._config = config
    self._step_fn = step_fn or self._default_step_fn
    self._server = viser.ViserServer(
      host=config.host,
      port=config.port,
      title=config.title,
    )

    # Synthetic state used by the default animation.
    self._points = np.random.uniform(-0.5, 0.5, size=(2_048, 3)).astype(np.float32)
    self._time = 0.0

    self._setup_gui()

  # --------------------------------------------------------------------------- #
  # GUI helpers
  # --------------------------------------------------------------------------- #
  def _setup_gui(self) -> None:
    """Register the handful of GUI controls we need."""
    with self._server.gui.add_folder("Simulation"):
      self._gui_auto = self._server.gui.add_checkbox(
        "Auto step",
        initial_value=True,
        hint="Tick continuously while the viewer is open.",
      )
      self._gui_dt = self._server.gui.add_slider(
        "dt (s)",
        min=1.0 / 240.0,
        max=1.0 / 15.0,
        step=1e-3,
        initial_value=1.0 / self._config.target_hz,
      )
      self._gui_button = self._server.gui.add_button(
        "Step once",
        icon=viser.Icon.PLAY,
      )

    @self._gui_button.on_click
    def _(_) -> None:
      self._update_scene()

    print(f"Viewer ready at http://{self._config.host}:{self._config.port}")

  # --------------------------------------------------------------------------- #
  # Main loop
  # --------------------------------------------------------------------------- #
  def serve_forever(self) -> None:
    """Run the viewer loop until interrupted."""
    try:
      while True:
        if self._gui_auto.value:
          self._update_scene()
        time.sleep(float(self._gui_dt.value))
    except KeyboardInterrupt:
      self._server.logger.info("Viewer stopped via Ctrl+C")

  # --------------------------------------------------------------------------- #
  # Scene updates
  # --------------------------------------------------------------------------- #
  def _update_scene(self) -> None:
    """Step the simulation (or placeholder) and push data to the viewer."""
    self._time += float(self._gui_dt.value)
    positions = self._step_fn(self._time)
    self._server.scene.add_point_cloud(
      "/sim_points",
      points=positions,
      colors=self._compute_colors(positions),
      point_shape="circle",
    )

  def _compute_colors(self, positions: np.ndarray) -> np.ndarray:
    """Generate simple colors based on point height."""
    heights = positions[:, 2]
    normalized = (heights - heights.min()) / max(heights.ptp(), 1e-3)
    colormap = np.stack(
      [
        255 * (1.0 - normalized),
        180 * normalized,
        255 * np.abs(np.sin(self._time)),
      ],
      axis=1,
    )
    return colormap.astype(np.uint8)

  def _default_step_fn(self, time_s: float) -> np.ndarray:
    """Animate the cached point cloud with a slow rotation."""
    angle = time_s * 0.5
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array(
      [
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
      ],
      dtype=np.float32,
    )
    return (self._points @ rot.T).astype(np.float32)


def run_viewer() -> None:
  """Convenience launcher used by scripts/CLI."""
  server = ViewerServer(ViewerConfig())
  server.serve_forever()


if __name__ == "__main__":
  run_viewer()

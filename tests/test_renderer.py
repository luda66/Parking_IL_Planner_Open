import numpy as np

from parking_il_planner.geometry.kinematics import VehicleState
from parking_il_planner.simulation.renderer import SceneRenderer


def test_renderer_shape_dtype_and_range() -> None:
    image = SceneRenderer().render(
        VehicleState(0.0, 0.0, 0.0),
        VehicleState(1.0, 0.0, 0.0),
        [],
    )
    assert image.shape == (3, 384, 384)
    assert image.dtype == np.float32
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0


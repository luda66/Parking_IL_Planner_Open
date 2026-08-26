import pytest

from parking_il_planner.geometry.kinematics import VehicleState
from parking_il_planner.planning.actions import Action, ActionExecutor


def test_straight_forward_moves_one_unit() -> None:
    result = ActionExecutor().execute(VehicleState(0.0, 0.0, 0.0), Action.S_PLUS)
    assert result.final_state.x == pytest.approx(0.05, abs=0.002)
    assert result.final_state.y == pytest.approx(0.0, abs=1e-6)


def test_stop_preserves_state() -> None:
    state = VehicleState(1.0, -2.0, 0.3)
    result = ActionExecutor().execute(state, Action.S0)
    assert result.final_state.x == state.x
    assert result.final_state.y == state.y
    assert result.final_state.heading == state.heading


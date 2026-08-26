from parking_il_planner.geometry.kinematics import VehicleState
from parking_il_planner.planning.actions import Action
from parking_il_planner.planning.reeds_shepp import ReedsSheppExpertPlanner


def test_expert_reaches_simple_target() -> None:
    result = ReedsSheppExpertPlanner().plan(
        VehicleState(0.0, 0.0, 0.0),
        VehicleState(1.0, 0.0, 0.0),
    )
    assert result.success
    assert result.actions[-1] == Action.S0
    assert result.final_error_pos < 0.1


from parking_il_planner.scenarios.perpendicular import PerpendicularParkingScenario


def test_scenario_seed_is_deterministic() -> None:
    first = PerpendicularParkingScenario()
    second = PerpendicularParkingScenario()
    assert first.generate(seed=7)
    assert second.generate(seed=7)
    assert first.get_initial_state() == second.get_initial_state()
    assert first.get_target_pose() == second.get_target_pose()


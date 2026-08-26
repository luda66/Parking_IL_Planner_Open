# Limitations

- Synthetic occupancy grids do not represent a production perception stack.
- Only static rectangular obstacles are modeled.
- The vehicle model omits actuator lag, tyre dynamics, and calibration error.
- Discrete actions introduce quantization and can oscillate near the goal.
- High open-loop accuracy can coexist with weak closed-loop robustness because of
  distribution shift.
- The experimental DAgger implementation did not consistently improve the neural
  controller in prior development runs; it remains under `experiments/` for study.
- The hybrid controller uses an expert planner and must not be reported as a pure
  neural-planner result.


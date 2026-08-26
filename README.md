# Parking IL Planner

[中文说明](README.zh-CN.md)

Parking IL Planner is a research implementation of an automatic-parking motion
planner trained by imitation learning. A Reeds-Shepp expert generates discrete
driving demonstrations in synthetic parking scenarios. A ResNet + Transformer
model then predicts one of seven vehicle actions from an occupancy grid, relative
target state, and recent action history.

This repository publishes the algorithm and reproducible pipeline. Generated
datasets, trained weights, and bulk evaluation renders are intentionally excluded.

## Why this project exists

Frame-level action accuracy is not enough for a planner: small errors change the
next observation and can compound during closed-loop rollout. This project keeps
open-loop classification and closed-loop driving as separate evaluation modes and
includes unsuccessful DAgger experiments so their limitations remain visible.

## Pipeline

```text
parking scenario
  -> Reeds-Shepp expert
  -> seven-class action demonstrations
  -> three-channel occupancy grids
  -> ResNet spatial features
  -> Transformer classifier
  -> closed-loop action execution
```

The discrete actions are stop, forward, reverse, left-forward, left-reverse,
right-forward, and right-reverse. Vehicle poses use the rear-axle center as their
reference point.

## Quick start

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[ml]"

# Deterministic expert-planner smoke demo; no dataset or checkpoint required.
parking-il expert-demo

# Generate a small local dataset. Output is ignored by Git.
parking-il generate --trajectories 20 --output data/generated/demo

# Train and evaluate after generating a suitable dataset.
parking-il train --data-dir data/generated/demo
parking-il open-loop --data-dir data/generated/demo --random 1
parking-il closed-loop --data-dir data/generated/demo --from-dataset 0
```

The CLI caps this smoke dataset at 128 rendered samples (about 54 MiB). It is not
enough data for a useful neural planner. See
[Reproduction](docs/reproduction.md) for the full workflow.

## Repository boundaries

- `src/parking_il_planner/`: reusable implementation
- `tests/`: deterministic tests that do not require the private dataset
- `docs/`: design and evaluation protocol
- `assets/`: deliberately curated public media
- `data/generated/`, `checkpoints/`, `artifacts/`: local and ignored

## Evaluation status

Historical development experiments showed a substantial gap between open-loop
classification and closed-loop success. Those numbers are not presented as a
public benchmark because the corresponding dataset and final protocol have not
yet been released. New results must follow the protocol in
[Evaluation protocol](docs/evaluation-protocol.md).

## Safety and limitations

This is research software, not a production vehicle controller. It does not model
sensor noise, actuator delay, dynamic obstacles, localization uncertainty, or
vehicle-specific safety requirements. Do not deploy it on a real vehicle without
an independent safety architecture and extensive validation.

## License and attribution

The project is licensed under Apache-2.0. The Reeds-Shepp implementation is adapted
from the MIT-licensed PythonRobotics project; see [NOTICE](NOTICE) for attribution.

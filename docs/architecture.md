# Architecture

## State and action model

Vehicle state is `(x, y, heading)` at the rear-axle center. The controller uses
seven discrete actions: stop, straight forward/reverse, and left/right steering
while moving forward/reverse. One translation step is 0.05 m and one turning step
targets 0.5 degrees of heading change.

## Expert and observation

The Reeds-Shepp expert connects start and target poses under a minimum-turning-
radius constraint. Continuous segments are discretized into the same seven actions
executed by the simulator. Each state is rendered as a 384 x 384 occupancy grid:

- channel 0: ego vehicle
- channel 1: target pose
- channel 2: static obstacles

## Learned planner

A ResNet encoder produces a spatial feature grid. Transformer encoder layers reason
over flattened spatial tokens. Optional relative-state and recent-action tokens are
prepended before global pooling and seven-class prediction.

The predictor is autoregressive only through the environment and action-history
input. It does not directly predict a full trajectory.


# Evaluation protocol

Every reported result must state:

- code revision and checkpoint identifier
- dataset or scenario-generation version
- number and type of scenarios
- random seeds and whether scenarios overlap training data
- controller type: expert, neural, or hybrid
- goal distance and heading thresholds
- collision penetration threshold
- maximum steps and stuck-detection limits

Open-loop accuracy measures action classification on fixed expert observations.
Closed-loop success measures whether model-generated actions reach the target
without disallowed collision or timeout. These metrics must never be substituted
for one another.

Bulk frames and GIFs belong in `artifacts/`. A small representative demo may be
copied deliberately into `assets/` after checking that it contains no private data.


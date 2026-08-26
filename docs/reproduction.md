# Reproduction

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[ml,dev]"
```

## Deterministic checks

```bash
parking-il expert-demo
pytest
```

## Generate local data

```bash
parking-il generate --trajectories 1000 --output data/generated/train --seed 42 --max-samples 0
```

Generation can require substantial disk space because every observation is stored
as a 384 x 384 three-channel array. Start with 20 trajectories to validate the
environment before a full run.

## Train

```bash
parking-il train --data-dir data/generated/train --epochs 50
```

Checkpoints are written under `checkpoints/` and are not tracked. Public benchmark
claims should only be added after the exact dataset, weights, and protocol are made
available with stable URLs and SHA256 checksums.

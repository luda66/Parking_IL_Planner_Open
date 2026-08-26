# Data generation

Synthetic scenarios cover perpendicular, parallel, angled, and straight approach
parking. For each deterministic seed, the generator creates an initial pose, target
pose, and static obstacle layout. The Reeds-Shepp expert proposes a path, the action
executor replays it, and collision checks reject unsuitable demonstrations.

Accepted samples are stored locally as four arrays:

- `images.npy`: `(N, 3, 384, 384)`, `uint8`
- `actions.npy`: `(N,)`, integer action IDs 0 through 6
- `states.npy`: `(N, 6)`, current and target poses in metres/radians
- `obstacles.npy`: padded obstacle rectangles

Images consume most of the storage. The generator writes them with a memory map so
the entire dataset need not reside in RAM. Generated arrays are intentionally
ignored by Git.


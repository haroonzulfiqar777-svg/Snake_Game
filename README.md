# Circuit // Snake — Python Edition

A classic Snake game, rebuilt in Python. Instead of writing it with plain
nested lists and loops, this version leans on real Python and data-science
tooling to represent the game state — good both as a fun game and as a
small example of *why* these tools exist.

## Requirements

- Python 3.10+
- `pygame` (rendering, input, game loop)
- `numpy` (the game board itself)

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python snake_game.py
```

## Controls

| Key                  | Action              |
|-----------------------|---------------------|
| Arrow keys / `W A S D`| Move the snake      |
| `Space`                | Pause / resume      |
| `Enter`                | Start / play again  |
| `Esc`                  | Quit                |

Your best score is saved automatically to `high_score.json` (created next
to the script the first time you play) and reloaded the next time you run
the game.

## How the game logic works

This is the part that makes it a "Python / data-science logic" build
rather than a plain port — each tool below was picked because it's the
right tool for that specific job, not just for variety.

### 1. The board is a `numpy` array, not a list of lists

```python
self.grid: np.ndarray = np.zeros((size, size), dtype=np.uint8)
```

The whole game board is one 22×22 numpy array. Every cell holds a small
number: `0` = empty, `1` = snake body, `2` = snake head, `3` = food.
This means two things become one-line, vectorized operations instead of
manual double loops:

- **Placing food**: `np.argwhere(self.grid == EMPTY)` instantly returns
  the coordinates of every open cell on the board, and we pick one at
  random — no scanning cell-by-cell.
- **Drawing**: `np.argwhere(board.grid != EMPTY)` gives us only the cells
  that actually need to be drawn, instead of looping over all 484 cells
  every single frame.
- **Speed clamping**: `np.clip(tick_ms - speedup, MIN, MAX)` keeps the
  game's speed inside a safe range in one call — the same clipping trick
  used to cap outliers in a numeric dataset.

### 2. The snake's body is a `collections.deque`

```python
self.body: deque[Point] = deque(...)
```

Every game tick, the snake needs to:
- add a new segment at the **front** (the new head), and
- remove a segment from the **back** (the old tail).

A normal Python `list` is O(n) for inserting at the front (everything has
to shift over). A `deque` (double-ended queue) does both operations in
**O(1)**, which is exactly what a real-time game loop needs.

### 3. `dataclasses` for typed, self-documenting structures

`Point` and `GameStats` are `@dataclass` classes instead of raw tuples or
dictionaries. This means `Point(x=5, y=3)` is unambiguous (versus
wondering whether a tuple is `(x, y)` or `(row, col)`), and `Point` is
`frozen=True` so it's immutable and safely hashable — which lets us drop
snake segments into a `set()` for fast collision checks.

### 4. `enum.Enum` for direction

```python
class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
```

Directions are a fixed, known set of options, so they're modeled as an
`Enum` rather than raw strings like `"up"` or magic tuples scattered
through the code. This also gives us `Direction.is_opposite()`, which
cleanly blocks the snake from reversing directly into itself.

### 5. Set-based collision checks

```python
cells = set(self.body)
return point in cells
```

Checking "does the snake's next move hit itself?" is a membership test.
Converting the snake's body to a `set` first makes that check O(1) on
average instead of O(n) — the same reason you'd convert a list to a set
before doing repeated `in` checks against a large dataset.

### 6. A fixed-timestep game loop

The game updates on a fixed interval (`tick_ms`, in milliseconds) that's
tracked separately from the rendering frame rate (locked at 60 FPS). This
keeps movement speed consistent regardless of how fast the computer can
render frames — a standard pattern for simulations and games alike.

## Project structure

```
.
├── snake_game.py      # the full game (single file, heavily commented)
├── requirements.txt   # pygame + numpy
├── high_score.json    # auto-created after your first game
└── README.md          # this file
```

## Ideas to extend it

- Add obstacles/walls using another numpy value (e.g. `WALL = 4`)
- Track a numpy array of *all* past food positions and visualize them as a
  heatmap with `matplotlib` after a game ends
- Add a difficulty selector that changes `BASE_TICK_MS`
- Replace the JSON high-score file with a small `sqlite3` database of
  per-run stats (score, duration, food eaten) for later analysis with
  `pandas`

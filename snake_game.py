"""
Circuit // Snake  —  Python Edition
=====================================
A classic Snake game rewritten with Python + data-science-style tooling:

    - numpy        -> the game board is stored as a 2D numpy array (a "grid"),
                       the same way you'd represent a matrix/heatmap in data
                       science. We use numpy operations (np.zeros, slicing,
                       vectorized checks) instead of manual nested loops
                       wherever it makes the logic clearer.
    - collections.deque -> the snake's body is stored in a deque (double-ended
                       queue) because we constantly add to the front (head)
                       and remove from the back (tail) — deque does both in
                       O(1) time, unlike a normal Python list (O(n) at the
                       front).
    - dataclasses  -> used for small, clearly-typed structures (Point, GameStats)
                       instead of loose tuples, so the code documents itself.
    - enum.Enum    -> used for Direction, so we can't accidentally use an
                       invalid direction value like a raw string or number.
    - pygame       -> handles the window, drawing, and keyboard/game loop.

Every section below is commented for a beginner following along.
"""

# ── Imports ──────────────────────────────────────────────────────────────
import sys                      # sys.exit() to cleanly close the game/window
import json                     # to save/load the high score to a small file
import random                   # random.randint() to place food on the grid
from pathlib import Path        # a clean, cross-platform way to build file paths
from dataclasses import dataclass, field   # for lightweight typed data structures
from enum import Enum           # for the Direction enum (UP/DOWN/LEFT/RIGHT)
from collections import deque   # fast O(1) add/remove from both ends -> snake body

import numpy as np              # our "grid" (the game board) is a numpy array
import pygame                   # handles rendering, input, and the game clock


# ── Configuration constants ─────────────────────────────────────────────
GRID_SIZE = 22          # the board is a GRID_SIZE x GRID_SIZE numpy array
CELL_PX = 26             # each grid cell is drawn as CELL_PX x CELL_PX pixels
HUD_HEIGHT = 60          # extra space at the top of the window for score text
WINDOW_W = GRID_SIZE * CELL_PX
WINDOW_H = GRID_SIZE * CELL_PX + HUD_HEIGHT

BASE_TICK_MS = 130        # starting game speed: one "step" every 130 ms
MIN_TICK_MS = 60          # the game never gets faster than this (in ms)
SPEEDUP_PER_FOOD = 2.5    # tick time (ms) removed for every food eaten

HIGH_SCORE_FILE = Path(__file__).parent / "high_score.json"  # save file location

# Numbers we use to represent what's inside each cell of the numpy grid.
# Using an IntEnum-like plain set of constants keeps grid values self-explanatory
# instead of magic numbers like 0, 1, 2 scattered through the code.
EMPTY = 0
SNAKE_BODY = 1
SNAKE_HEAD = 2
FOOD = 3

# ── Colour palette (RGB tuples used by pygame) ──────────────────────────
COLOR_BG = (11, 15, 20)
COLOR_GRID_LINE = (24, 36, 48)
COLOR_SNAKE_HEAD = (95, 247, 208)
COLOR_SNAKE_BODY = (47, 215, 174)
COLOR_FOOD = (255, 107, 91)
COLOR_TEXT = (219, 231, 239)
COLOR_TEXT_DIM = (111, 132, 150)
COLOR_ACCENT = (95, 247, 208)


# ── Direction enum ───────────────────────────────────────────────────────
class Direction(Enum):
    """
    Each direction stores its own (dx, dy) movement vector.
    Using an Enum (instead of raw tuples everywhere) means the rest of the
    code can say `Direction.UP` and get autocomplete + type safety, rather
    than risk typos like (0, -1) vs (0, 1) mixed up somewhere else.
    """
    UP = (0, -1)     # moving up   = same x, y decreases by 1
    DOWN = (0, 1)    # moving down = same x, y increases by 1
    LEFT = (-1, 0)   # moving left = x decreases by 1, same y
    RIGHT = (1, 0)   # moving right = x increases by 1, same y

    def is_opposite(self, other: "Direction") -> bool:
        """Return True if `other` is the exact reverse of this direction.
        Used to stop the player from doing a 180-degree turn into themselves.
        """
        dx1, dy1 = self.value
        dx2, dy2 = other.value
        return dx1 == -dx2 and dy1 == -dy2


# ── Small typed data structures ─────────────────────────────────────────
@dataclass(frozen=True)
class Point:
    """
    An (x, y) grid coordinate. `frozen=True` makes it immutable (read-only)
    and hashable, so we can safely store Points in sets for fast lookups.
    """
    x: int
    y: int

    def moved(self, direction: Direction) -> "Point":
        """Return a NEW Point shifted one cell in `direction`."""
        dx, dy = direction.value
        return Point(self.x + dx, self.y + dy)

    def is_out_of_bounds(self, size: int) -> bool:
        """True if this point has left the GRID_SIZE x GRID_SIZE board."""
        return not (0 <= self.x < size and 0 <= self.y < size)


@dataclass
class GameStats:
    """Tracks the running score, best score, and how fast the game is."""
    score: int = 0
    high_score: int = 0
    tick_ms: float = BASE_TICK_MS
    food_eaten: int = 0

    def register_food_eaten(self) -> None:
        """Update score/speed the moment the snake eats a food pellet."""
        self.score += 10
        self.food_eaten += 1
        # np.clip keeps tick_ms from ever dropping below MIN_TICK_MS —
        # this is the same numpy "clip a value into a range" trick you'd
        # use to clip outliers in a dataset.
        self.tick_ms = float(
            np.clip(self.tick_ms - SPEEDUP_PER_FOOD, MIN_TICK_MS, BASE_TICK_MS)
        )
        if self.score > self.high_score:
            self.high_score = self.score


# ── The grid / board, backed by a numpy array ───────────────────────────
class Board:
    """
    Wraps a numpy 2D array that represents the game board.
    Cell values are the EMPTY / SNAKE_BODY / SNAKE_HEAD / FOOD constants
    defined above. Keeping the board as a numpy array (instead of a list
    of lists) lets us use fast, vectorized numpy operations, e.g. finding
    every empty cell in one call with `np.argwhere(grid == EMPTY)`.
    """

    def __init__(self, size: int):
        self.size = size
        # np.zeros creates a size x size array filled with EMPTY (0),
        # using an 8-bit integer type (dtype=np.uint8) since our values
        # are all small numbers — a common data-science memory optimisation.
        self.grid: np.ndarray = np.zeros((size, size), dtype=np.uint8)

    def clear(self) -> None:
        """Reset every cell back to EMPTY. np.fill is a fast, vectorized reset."""
        self.grid.fill(EMPTY)

    def set_cell(self, point: Point, value: int) -> None:
        """Write `value` into the grid at (point.x, point.y)."""
        self.grid[point.y, point.x] = value   # numpy indexing is [row, col] = [y, x]

    def random_empty_cell(self) -> Point:
        """
        Find every empty cell using numpy, then randomly choose one.
        `np.argwhere(condition)` returns the (row, col) coordinates of every
        cell where the condition is True — a vectorized alternative to
        looping over every cell by hand.
        """
        empty_coords = np.argwhere(self.grid == EMPTY)  # shape: (num_empty, 2)
        row, col = empty_coords[random.randrange(len(empty_coords))]
        return Point(x=int(col), y=int(row))

    def rebuild_from_snake_and_food(self, snake_body: deque, food: Point) -> None:
        """
        Recompute the entire grid in one pass from the current snake
        segments and food position. Called once per game tick.
        """
        self.clear()
        # Body first, head last, so the head's marker always "wins" even
        # if (in theory) the head and last body cell would ever overlap.
        for i, segment in enumerate(snake_body):
            marker = SNAKE_HEAD if i == 0 else SNAKE_BODY
            self.set_cell(segment, marker)
        self.set_cell(food, FOOD)


# ── The Snake itself ─────────────────────────────────────────────────────
class Snake:
    """
    The snake's body lives in a collections.deque of Point objects.
    body[0] is always the HEAD. We use a deque instead of a list because:
        - appendleft() to add a new head   -> O(1) with deque, O(n) with list
        - pop() to drop the tail            -> O(1) with deque, O(1) with list
    So for a game that does this every single tick, deque is the right tool.
    """

    def __init__(self, start: Point, length: int, direction: Direction):
        self.direction = direction
        self.pending_direction = direction  # the next direction queued by input
        # Build the starting body stretching backwards from `start`,
        # e.g. start=(10,10) moving RIGHT -> [(10,10), (9,10), (8,10)]
        dx, dy = direction.value
        self.body: deque[Point] = deque(
            Point(start.x - dx * i, start.y - dy * i) for i in range(length)
        )

    def queue_turn(self, new_direction: Direction) -> None:
        """
        Store the player's requested turn, but reject 180-degree reversals
        (that would mean crashing directly into your own neck).
        """
        if not new_direction.is_opposite(self.direction):
            self.pending_direction = new_direction

    def head(self) -> Point:
        return self.body[0]

    def occupies(self, point: Point, include_head: bool = True) -> bool:
        """
        Check if `point` collides with any part of the snake.
        Uses a set for O(1) average lookup instead of scanning the deque —
        the same "convert to a set for fast membership tests" pattern
        you'd use when checking membership against a large dataset.
        """
        cells = set(self.body) if include_head else set(list(self.body)[1:])
        return point in cells

    def advance(self, grow: bool) -> Point:
        """
        Move the snake one cell forward in `self.pending_direction`.
        If `grow` is True (food was eaten), the tail is kept so the snake
        gets one cell longer; otherwise the tail is removed as usual.
        Returns the new head position.
        """
        self.direction = self.pending_direction
        new_head = self.head().moved(self.direction)
        self.body.appendleft(new_head)   # O(1): add new head to the front
        if not grow:
            self.body.pop()               # O(1): drop the old tail
        return new_head

    def __len__(self) -> int:
        return len(self.body)


# ── High score persistence (tiny JSON "database") ───────────────────────
def load_high_score() -> int:
    """Read the saved high score from disk, defaulting to 0 if not found."""
    if HIGH_SCORE_FILE.exists():
        try:
            data = json.loads(HIGH_SCORE_FILE.read_text())
            return int(data.get("high_score", 0))
        except (json.JSONDecodeError, ValueError):
            return 0
    return 0


def save_high_score(value: int) -> None:
    """Persist the high score to a small JSON file next to this script."""
    HIGH_SCORE_FILE.write_text(json.dumps({"high_score": value}))


# ── Rendering helpers ────────────────────────────────────────────────────
def draw_grid_lines(surface: pygame.Surface) -> None:
    """Draw faint gridlines across the board for the retro-terminal look."""
    for i in range(GRID_SIZE + 1):
        x = i * CELL_PX
        y = i * CELL_PX
        pygame.draw.line(surface, COLOR_GRID_LINE, (x, HUD_HEIGHT), (x, WINDOW_H))
        pygame.draw.line(surface, COLOR_GRID_LINE, (0, HUD_HEIGHT + y), (WINDOW_W, HUD_HEIGHT + y))


def draw_board(surface: pygame.Surface, board: Board) -> None:
    """
    Read the numpy grid and draw each non-empty cell.
    `np.argwhere` again gives us the coordinates of interesting cells
    (anything that isn't EMPTY) in one vectorized call.
    """
    occupied = np.argwhere(board.grid != EMPTY)
    for row, col in occupied:
        value = board.grid[row, col]
        px, py = int(col) * CELL_PX, HUD_HEIGHT + int(row) * CELL_PX
        rect = pygame.Rect(px + 1, py + 1, CELL_PX - 2, CELL_PX - 2)
        if value == FOOD:
            pygame.draw.circle(
                surface, COLOR_FOOD,
                (px + CELL_PX // 2, py + CELL_PX // 2), CELL_PX // 2 - 3
            )
        elif value == SNAKE_HEAD:
            pygame.draw.rect(surface, COLOR_SNAKE_HEAD, rect, border_radius=6)
        else:  # SNAKE_BODY
            pygame.draw.rect(surface, COLOR_SNAKE_BODY, rect, border_radius=4)


def draw_hud(surface: pygame.Surface, font: pygame.font.Font, stats: GameStats) -> None:
    """Draw the score / high-score header bar."""
    pygame.draw.rect(surface, COLOR_BG, (0, 0, WINDOW_W, HUD_HEIGHT))
    title = font.render("CIRCUIT // SNAKE", True, COLOR_TEXT)
    surface.blit(title, (14, 18))

    score_text = font.render(f"SCORE {stats.score}", True, COLOR_TEXT)
    best_text = font.render(f"BEST {stats.high_score}", True, COLOR_ACCENT)
    surface.blit(score_text, (WINDOW_W - score_text.get_width() - best_text.get_width() - 34, 18))
    surface.blit(best_text, (WINDOW_W - best_text.get_width() - 14, 18))


def draw_center_message(
    surface: pygame.Surface,
    big_font: pygame.font.Font,
    small_font: pygame.font.Font,
    lines: list[tuple[str, tuple[int, int, int]]],
) -> None:
    """Draw a semi-transparent overlay with a stack of centered text lines."""
    overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
    overlay.fill((11, 15, 20, 210))
    surface.blit(overlay, (0, 0))

    total_h = sum(f.get_height() + 10 for f, _, _ in [(big_font, *l) for l in lines[:1]]) if lines else 0
    y = WINDOW_H // 2 - (len(lines) * 34) // 2
    for text, color in lines:
        font = big_font if text.isupper() else small_font
        rendered = font.render(text, True, color)
        rect = rendered.get_rect(center=(WINDOW_W // 2, y))
        surface.blit(rendered, rect)
        y += rendered.get_height() + 14


# ── Main game loop ────────────────────────────────────────────────────────
def main() -> None:
    pygame.init()
    pygame.display.set_caption("Circuit // Snake — Python Edition")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont("couriernew", 26, bold=True)
    font_small = pygame.font.SysFont("couriernew", 15)
    font_hud = pygame.font.SysFont("couriernew", 14, bold=True)

    board = Board(GRID_SIZE)
    stats = GameStats(high_score=load_high_score())

    def new_game() -> tuple[Snake, Point]:
        start = Point(GRID_SIZE // 2, GRID_SIZE // 2)
        snake = Snake(start=start, length=3, direction=Direction.RIGHT)
        board.rebuild_from_snake_and_food(snake.body, start)  # placeholder food
        food = board.random_empty_cell()
        return snake, food

    snake, food = new_game()
    stats.score = 0
    stats.tick_ms = BASE_TICK_MS

    state = "start"       # one of: "start", "playing", "paused", "game_over"
    time_since_tick = 0.0

    running = True
    while running:
        dt = clock.tick(60)               # cap at 60 FPS, dt = ms since last frame
        time_since_tick += dt

        # ── Handle input ────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE,):
                    running = False

                elif event.key == pygame.K_SPACE:
                    if state == "playing":
                        state = "paused"
                    elif state == "paused":
                        state = "playing"

                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and state in ("start", "game_over"):
                    snake, food = new_game()
                    stats.score = 0
                    stats.tick_ms = BASE_TICK_MS
                    state = "playing"
                    time_since_tick = 0.0

                elif state == "playing":
                    if event.key in (pygame.K_UP, pygame.K_w):
                        snake.queue_turn(Direction.UP)
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        snake.queue_turn(Direction.DOWN)
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        snake.queue_turn(Direction.LEFT)
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        snake.queue_turn(Direction.RIGHT)

        # ── Update game state on a fixed tick, independent of frame rate ──
        if state == "playing" and time_since_tick >= stats.tick_ms:
            time_since_tick = 0.0

            next_head = snake.head().moved(snake.pending_direction)
            ate_food = next_head == food

            hits_wall = next_head.is_out_of_bounds(GRID_SIZE)
            # When eating, the tail won't move away this tick, so check
            # collision against the *whole* body; otherwise the tail cell
            # is vacated this tick, so exclude it from the collision check.
            body_to_check = snake.body if ate_food else deque(list(snake.body)[:-1])
            hits_self = next_head in body_to_check

            if hits_wall or hits_self:
                state = "game_over"
                if stats.score > stats.high_score:
                    stats.high_score = stats.score
                    save_high_score(stats.high_score)
            else:
                snake.advance(grow=ate_food)
                if ate_food:
                    stats.register_food_eaten()
                    board.rebuild_from_snake_and_food(snake.body, food)  # sync before pick
                    food = board.random_empty_cell()

            board.rebuild_from_snake_and_food(snake.body, food)

        # ── Draw everything ─────────────────────────────────────────────
        screen.fill(COLOR_BG)
        draw_grid_lines(screen)
        draw_board(screen, board)
        draw_hud(screen, font_hud, stats)

        if state == "start":
            draw_center_message(screen, font_big, font_small, [
                ("CIRCUIT // SNAKE", COLOR_TEXT),
                ("Arrow keys or WASD to move.", COLOR_TEXT_DIM),
                ("Press ENTER to start.", COLOR_ACCENT),
            ])
        elif state == "paused":
            draw_center_message(screen, font_big, font_small, [
                ("PAUSED", COLOR_TEXT),
                ("Press SPACE to resume.", COLOR_TEXT_DIM),
            ])
        elif state == "game_over":
            draw_center_message(screen, font_big, font_small, [
                ("GAME OVER", (255, 107, 91)),
                (f"Final score: {stats.score}", COLOR_TEXT),
                ("Press ENTER to play again.", COLOR_ACCENT),
            ])

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

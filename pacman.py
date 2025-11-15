"""Simple Pac-Man inspired game built with pygame.

Use the arrow keys to collect all pellets while avoiding the ghosts.
Press SPACE after winning or losing to restart. Press ESC to quit.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

try:
    import pygame
except ImportError as exc:  # pragma: no cover - helpful runtime message
    raise SystemExit(
        "pygame is required to run the Pac-Man demo.\n"
        "Install it with: pip install pygame"
    ) from exc


# ---------------------------------------------------------------------------
# Level layout ("#" = wall, "." = pellet, "P" = player, "G" = ghost)
# The layout can be tweaked – the Maze class will pad rows to equal width.
# ---------------------------------------------------------------------------
LAYOUT = """
###########
#P.......G#
#.#####.#.#
#.#...#.#.#
#.#.#.#.#.#
#...#.....#
###########
"""

TILE_SIZE = 36
INFO_HEIGHT = 72
FPS = 8
PELLET_SCORE = 10
START_LIVES = 3

DIRECTION_KEYS: Dict[int, Tuple[int, int]] = {
    pygame.K_LEFT: (-1, 0),
    pygame.K_RIGHT: (1, 0),
    pygame.K_UP: (0, -1),
    pygame.K_DOWN: (0, 1),
}
DIRECTIONS: Sequence[Tuple[int, int]] = tuple(DIRECTION_KEYS.values())


def add_pos(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def opposite(direction: Tuple[int, int]) -> Tuple[int, int]:
    return -direction[0], -direction[1]


class Maze:
    """Represent the playfield grid and pellet locations."""

    def __init__(self, layout: str) -> None:
        rows = [line.rstrip("\n") for line in layout.strip().splitlines()]
        width = max(len(row) for row in rows)
        self.grid: List[List[str]] = []
        self.pellets: set[Tuple[int, int]] = set()
        self.player_start: Tuple[int, int] | None = None
        self.ghost_starts: List[Tuple[int, int]] = []

        for y, row in enumerate(rows):
            padded = row.ljust(width)
            grid_row: List[str] = []
            for x, ch in enumerate(padded):
                if ch == "#":
                    grid_row.append("#")
                elif ch == ".":
                    grid_row.append(" ")
                    self.pellets.add((x, y))
                elif ch == "P":
                    grid_row.append(" ")
                    self.player_start = (x, y)
                elif ch == "G":
                    grid_row.append(" ")
                    self.ghost_starts.append((x, y))
                else:
                    grid_row.append(" ")
            self.grid.append(grid_row)

        if self.player_start is None:
            raise ValueError("Layout must contain a player start denoted by 'P'.")
        if not self.ghost_starts:
            raise ValueError("Layout must contain at least one ghost start denoted by 'G'.")

        self.width = width
        self.height = len(rows)
        self._initial_pellets = set(self.pellets)

    def reset(self) -> None:
        self.pellets = set(self._initial_pellets)

    def is_wall(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return True
        return self.grid[y][x] == "#"


@dataclass
class Player:
    start: Tuple[int, int]
    position: Tuple[int, int]
    direction: Tuple[int, int] = (0, 0)
    queued: Tuple[int, int] | None = None

    def queue_direction(self, direction: Tuple[int, int]) -> None:
        self.queued = direction

    def reset(self) -> None:
        self.position = self.start
        self.direction = (0, 0)
        self.queued = None

    def update(self, maze: Maze) -> None:
        if self.queued is not None and not maze.is_wall(add_pos(self.position, self.queued)):
            self.direction = self.queued
            self.queued = None

        if self.direction == (0, 0):
            return

        next_pos = add_pos(self.position, self.direction)
        if maze.is_wall(next_pos):
            self.direction = (0, 0)
            return
        self.position = next_pos


@dataclass
class Ghost:
    start: Tuple[int, int]
    color: Tuple[int, int, int]
    position: Tuple[int, int]
    direction: Tuple[int, int]

    def __init__(self, start: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        self.start = start
        self.color = color
        self.position = start
        self.direction = random.choice(DIRECTIONS)

    def reset(self) -> None:
        self.position = self.start
        self.direction = random.choice(DIRECTIONS)

    def update(self, maze: Maze, target: Tuple[int, int]) -> None:
        options: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for direction in DIRECTIONS:
            nxt = add_pos(self.position, direction)
            if maze.is_wall(nxt):
                continue
            options.append((nxt, direction))

        if not options:
            return

        # Prefer continuing straight unless blocked.
        straight = [opt for opt in options if opt[1] == self.direction]
        if straight:
            chosen = straight[0]
        else:
            filtered = [opt for opt in options if opt[1] != opposite(self.direction)]
            candidates = filtered or options
            # Bias towards chasing the player but keep some randomness.
            candidates.sort(key=lambda item: manhattan(item[0], target))
            if random.random() < 0.7:
                chosen = candidates[0]
            else:
                chosen = random.choice(candidates)

        self.position, self.direction = chosen


class Game:
    def __init__(self, layout: str = LAYOUT) -> None:
        pygame.init()
        self.maze = Maze(layout)
        self.screen = pygame.display.set_mode(
            (self.maze.width * TILE_SIZE, self.maze.height * TILE_SIZE + INFO_HEIGHT)
        )
        pygame.display.set_caption("Pac-Man (mini)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.big_font = pygame.font.Font(None, 52)

        self.player = Player(self.maze.player_start, self.maze.player_start)
        ghost_colors = [(255, 0, 0), (255, 184, 255), (0, 255, 255), (255, 184, 82)]
        self.ghosts = [
            Ghost(pos, ghost_colors[i % len(ghost_colors)])
            for i, pos in enumerate(self.maze.ghost_starts)
        ]

        self.running = True
        self.state = "playing"  # "playing", "won", "lost"
        self.score = 0
        self.lives = START_LIVES

    def reset(self) -> None:
        self.maze.reset()
        self.player.reset()
        for ghost in self.ghosts:
            ghost.reset()
        self.score = 0
        self.lives = START_LIVES
        self.state = "playing"

    def run(self) -> None:
        while self.running:
            self.handle_events()
            if self.state == "playing":
                self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key in DIRECTION_KEYS and self.state == "playing":
                    self.player.queue_direction(DIRECTION_KEYS[event.key])
                elif event.key == pygame.K_SPACE and self.state != "playing":
                    self.reset()

    def update(self) -> None:
        self.player.update(self.maze)
        if self.player.position in self.maze.pellets:
            self.maze.pellets.remove(self.player.position)
            self.score += PELLET_SCORE
            if not self.maze.pellets:
                self.state = "won"
                return

        for ghost in self.ghosts:
            ghost.update(self.maze, self.player.position)
            if ghost.position == self.player.position:
                self.lives -= 1
                if self.lives <= 0:
                    self.state = "lost"
                else:
                    self.player.reset()
                    for g in self.ghosts:
                        g.reset()
                break

    def draw(self) -> None:
        self.screen.fill((0, 0, 0))
        self.draw_maze()
        self.draw_pellets()
        self.draw_entities()
        self.draw_hud()
        if self.state == "won":
            self.draw_center_message("You cleared the maze! Press SPACE to restart.")
        elif self.state == "lost":
            self.draw_center_message("Ghosts got you! Press SPACE to try again.")
        pygame.display.flip()

    def draw_maze(self) -> None:
        wall_color = (0, 0, 160)
        for y, row in enumerate(self.maze.grid):
            for x, cell in enumerate(row):
                if cell == "#":
                    rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                    pygame.draw.rect(self.screen, wall_color, rect)

    def draw_pellets(self) -> None:
        pellet_color = (255, 200, 0)
        radius = TILE_SIZE // 6
        for x, y in self.maze.pellets:
            center = (x * TILE_SIZE + TILE_SIZE // 2, y * TILE_SIZE + TILE_SIZE // 2)
            pygame.draw.circle(self.screen, pellet_color, center, radius)

    def draw_entities(self) -> None:
        px, py = self.player.position
        center = (px * TILE_SIZE + TILE_SIZE // 2, py * TILE_SIZE + TILE_SIZE // 2)
        pygame.draw.circle(self.screen, (255, 255, 0), center, TILE_SIZE // 2 - 4)

        for ghost in self.ghosts:
            gx, gy = ghost.position
            g_center = (gx * TILE_SIZE + TILE_SIZE // 2, gy * TILE_SIZE + TILE_SIZE // 2)
            pygame.draw.circle(self.screen, ghost.color, g_center, TILE_SIZE // 2 - 4)
            eye_radius = TILE_SIZE // 10
            offset = TILE_SIZE // 8
            pygame.draw.circle(
                self.screen,
                (255, 255, 255),
                (g_center[0] - offset, g_center[1] - offset),
                eye_radius,
            )
            pygame.draw.circle(
                self.screen,
                (255, 255, 255),
                (g_center[0] + offset, g_center[1] - offset),
                eye_radius,
            )

    def draw_hud(self) -> None:
        hud_rect = pygame.Rect(0, self.maze.height * TILE_SIZE, self.screen.get_width(), INFO_HEIGHT)
        pygame.draw.rect(self.screen, (20, 20, 20), hud_rect)
        score_surface = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        lives_surface = self.font.render(f"Lives: {self.lives}", True, (255, 255, 255))
        info_surface = self.font.render("Arrows to move • ESC to quit", True, (200, 200, 200))
        self.screen.blit(score_surface, (16, self.maze.height * TILE_SIZE + 12))
        self.screen.blit(lives_surface, (16, self.maze.height * TILE_SIZE + 38))
        self.screen.blit(
            info_surface,
            (self.screen.get_width() - info_surface.get_width() - 16, self.maze.height * TILE_SIZE + 24),
        )

    def draw_center_message(self, text: str) -> None:
        surface = self.big_font.render(text, True, (255, 255, 255))
        rect = surface.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
        padding = 24
        background = pygame.Rect(rect.left - padding, rect.top - padding, rect.width + padding * 2, rect.height + padding * 2)
        pygame.draw.rect(self.screen, (0, 0, 0), background)
        pygame.draw.rect(self.screen, (255, 255, 255), background, 2)
        self.screen.blit(surface, rect)


def main() -> None:
    Game().run()


if __name__ == "__main__":
    main()

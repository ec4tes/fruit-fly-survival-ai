"""Map generation, exact ray sensing and swept circle collision handling."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Obstacle:
    """Axis-aligned rectangle in logical coordinates."""

    x: float
    y: float
    width: float
    height: float

    def intersects(self, point: np.ndarray, radius: float) -> bool:
        """Conservative expanded-box contact, matching swept movement at corners."""
        return bool(
            np.all(point >= np.array([self.x, self.y]) - radius)
            and np.all(point <= np.array([self.x + self.width, self.y + self.height]) + radius)
        )


def ray_box(origin: np.ndarray, direction: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Ray entry distance via slabs; infinity when the ray misses the box."""
    near, far = -np.inf, np.inf
    for axis in range(2):
        if abs(direction[axis]) < 1e-12:
            if origin[axis] < low[axis] or origin[axis] > high[axis]:
                return np.inf
        else:
            a, b = (
                (low[axis] - origin[axis]) / direction[axis],
                (high[axis] - origin[axis]) / direction[axis],
            )
            near, far = max(near, min(a, b)), min(far, max(a, b))
    return max(0.0, near) if far >= max(0.0, near) else np.inf


class World:
    """Obstacle geometry independent of pixel resolution and episode spawn RNG."""

    def __init__(self, config: dict, rng: np.random.Generator):
        self.config = config
        self.size = np.array([config["width"], config["height"]], dtype=float)
        fixed = config["fixed_obstacles"]
        if fixed is not None:
            self.obstacles = [Obstacle(*rect) for rect in fixed]
        else:
            self.obstacles = []
            for _ in range(config["obstacle_count"]):
                size = rng.uniform(*config["obstacle_size"], size=2)
                if np.any(size >= self.size / 2):
                    raise ValueError("Obstacles are too large for the world")
                position = rng.uniform(size / 2, self.size - size * 1.5)
                self.obstacles.append(Obstacle(*position, *size))
        for rect in self.obstacles:
            if (
                rect.width <= 0
                or rect.height <= 0
                or rect.x < 0
                or rect.y < 0
                or rect.x + rect.width > self.size[0]
                or rect.y + rect.height > self.size[1]
            ):
                raise ValueError("Invalid fixed obstacle rectangle")

    def valid(self, point: np.ndarray, radius: float) -> bool:
        """Check circle clearance from walls and every obstacle."""
        return bool(
            np.all(point >= radius)
            and np.all(point <= self.size - radius)
            and not any(o.intersects(point, radius) for o in self.obstacles)
        )

    def sample(
        self,
        rng: np.random.Generator,
        radius: float,
        avoid: np.ndarray | None = None,
        clearance: float = 0.0,
        distribution: str = "uniform",
    ) -> np.ndarray:
        """Sample a valid point, failing clearly for an over-constrained map."""
        for _ in range(self.config["spawn_attempts"]):
            if distribution == "clustered":
                center = self.size * np.array([0.25, 0.75])
                point = rng.normal(center, self.size * self.config["food_cluster_spread"])
            else:
                point = rng.uniform(radius, self.size - radius)
            if self.valid(point, radius) and (
                avoid is None or np.linalg.norm(point - avoid) > clearance
            ):
                return point
        raise RuntimeError("No valid spawn location. Reduce obstacles or spawn clearance.")

    def move(
        self, start: np.ndarray, displacement: np.ndarray, radius: float
    ) -> tuple[np.ndarray, bool]:
        """Stop at first contact using swept expanded boxes (conservative corners)."""
        length = float(np.linalg.norm(displacement))
        if length == 0:
            return start.copy(), False
        direction = displacement / length
        distance = length
        for axis in range(2):
            if direction[axis] > 1e-12:
                distance = min(distance, (self.size[axis] - radius - start[axis]) / direction[axis])
            elif direction[axis] < -1e-12:
                distance = min(distance, (radius - start[axis]) / direction[axis])
        for o in self.obstacles:
            entry = ray_box(
                start,
                direction,
                np.array([o.x, o.y]) - radius,
                np.array([o.x + o.width, o.y + o.height]) + radius,
            )
            distance = min(distance, entry)
        collided = distance < length
        return start + direction * max(0.0, distance - (1e-7 if collided else 0)), collided

    def sense(self, position: np.ndarray, angle: float, radius: float) -> float:
        """Ray cast against expanded obstacles and inner wall boundaries."""
        direction = np.array([np.cos(angle), np.sin(angle)])
        end, _ = self.move(position, direction * self.config["sensor_range"], radius)
        return float(np.linalg.norm(end - position) / self.config["sensor_range"])

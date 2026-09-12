"""Pygame rendering without coupling world physics to pixels."""

import os

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


class Renderer:
    """Draw a readable HUD and direction-marked bodies in human or RGB mode."""

    def __init__(self, config: dict, mode: str):
        import pygame

        self.pg, self.mode, self.config = pygame, mode, config
        pygame.font.init()
        size = (config["render_width"], config["render_height"])
        self.surface = pygame.Surface(size)
        self.screen = None
        self.quit_requested = False
        if mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode(size)
            pygame.display.set_caption("Fruit Fly Connectome Survival AI — heuristic/demo")
        self.font = pygame.font.Font(None, 26)
        self.clock = pygame.time.Clock()
        self.scale = np.array(size) / [config["width"], config["height"]]

    def draw(self, env):
        """Return HWC uint8 RGB or update the visible window."""
        pg, surface = self.pg, self.surface
        surface.fill((17, 24, 39))

        def point(position):
            return tuple((position * self.scale).astype(int))

        for o in env.world.obstacles:
            pg.draw.rect(
                surface,
                (71, 85, 105),
                (*point(np.array([o.x, o.y])), *point(np.array([o.width, o.height]))),
                border_radius=4,
            )
        for position in env.food:
            pg.draw.circle(
                surface,
                (74, 222, 128),
                point(position),
                max(2, int(self.config["food_radius"] * min(self.scale))),
            )
        for body, color, key in [
            (env.agent, (56, 189, 248), "agent_radius"),
            (env.predator, (251, 113, 133), "predator_radius"),
        ]:
            radius = self.config[key]
            pg.draw.circle(
                surface, color, point(body.position), max(3, int(radius * min(self.scale)))
            )
            tip = (
                body.position + np.array([np.cos(body.heading), np.sin(body.heading)]) * radius * 2
            )
            pg.draw.line(surface, (248, 250, 252), point(body.position), point(tip), 2)
        pg.draw.rect(surface, (30, 41, 59), (12, 12, 510, 80), border_radius=8)
        ratio = env.agent.energy / self.config["max_energy"]
        pg.draw.rect(surface, (51, 65, 85), (24, 26, 180, 12))
        pg.draw.rect(surface, (74, 222, 128), (24, 26, int(180 * ratio), 12))
        text = (
            f"Energy {env.agent.energy:.1f}   Time {env.agent.age * self.config['dt']:.1f}s   "
            f"Food {env.agent.food_collected}"
        )
        surface.blit(self.font.render(text, True, (226, 232, 240)), (24, 45))
        label = f"Reward {env.current_reward:+.2f}   Total {env.total_reward:.1f}   Esc: close"
        surface.blit(self.font.render(label, True, (148, 163, 184)), (24, 68))
        if self.mode == "human":
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    self.quit_requested = True
            self.screen.blit(surface, (0, 0))
            pg.display.flip()
            self.clock.tick(env.metadata["render_fps"])
            return None
        return np.transpose(pg.surfarray.array3d(surface), (1, 0, 2)).copy()

    def close(self) -> None:
        """Close only the display owned by this renderer."""
        if self.screen is not None:
            self.pg.display.quit()

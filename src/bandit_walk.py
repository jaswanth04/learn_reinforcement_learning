"""Deterministic 1D walk: every action moves exactly one cell."""

from models import ActionResult
from walk_env import WalkEnv


class BanditWalk(WalkEnv):

    def _distribution(self, s: int, code: int) -> list[ActionResult]:
        direction = -1 if self._actions[code].name == "LEFT" else 1
        nxt = self._check_wall_bounce(s + direction)

        # Goal and reward
        if nxt == self.goal:
            reward, done = self.goal_reward, True
        elif nxt == self.hole:
            reward, done = self.hole_reward, True
        else:
            reward, done = self.step_reward, False

        return [ActionResult(1.0, nxt, reward, done)]
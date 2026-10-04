"""Stochastic 1D walk (Sutton & Barto 'slippery walk').

With probability slip_prob the intended move is reversed.
slip_prob = 0.0 reproduces the deterministic BanditWalk.
"""

from models import ActionResult
from walk_env import WalkEnv


class BanditSlipperyWalk(WalkEnv):

    def __init__(self, n_states=3, start=None, hole=None, goal=None,
                 goal_reward=1.0, hole_reward=0.0, step_reward=0.0,
                 slip_prob=0.2, seed=None):
        if not 0.0 <= slip_prob <= 1.0:
            raise ValueError(f"slip_prob must be in [0, 1], got {slip_prob}")
        self.slip_prob = float(slip_prob)
        super().__init__(n_states=n_states, start=start, hole=hole, goal=goal,
                         goal_reward=goal_reward, hole_reward=hole_reward,
                         step_reward=step_reward, seed=seed)

    def _reward_for(self, s: int) -> tuple[float, bool]:
        if s == self.goal:
            return self.goal_reward, True
        if s == self.hole:
            return self.hole_reward, True
        return self.step_reward, False

    

    def _distribution(self, s: int, code: int) -> list[ActionResult]:
        direction = -1 if self._actions[code].name == "LEFT" else 1
        intended = self._check_wall_bounce(s + direction)
        slipped  = self._check_wall_bounce(s - direction)

        branches: dict[tuple, float] = {}        # merge identical branches
        for prob, target in ((1.0 - self.slip_prob, intended),
                             (self.slip_prob, slipped)):
            if prob <= 0.0:
                continue
            reward, done = self._reward_for(target)
            key = (target, reward, done)
            branches[key] = branches.get(key, 0.0) + prob

        return [ActionResult(p, t, r, d) for (t, r, d), p in branches.items()]
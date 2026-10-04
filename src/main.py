"""Demo: walk envs + FrozenLake, with live display, MC metrics, DP values."""

import numpy as np

from bandit_slippery_walk import BanditSlipperyWalk
from bandit_walk import BanditWalk
from display import (print_environment, print_policy,
                     print_state_value_function, print_transitions)
from frozen_lake import DOWN, RIGHT, FrozenLake
from policy_metrics import mean_return, play_episode, probability_success, std_return
from models import Environment


def fixed_policy(code):
    """Constant policy: always take `code`."""
    return lambda s: code



        


def path_policy(s):
    """Hand-picked safe path on the classic 4x4 board:
       0 -> 4 -> 8 -> 9 -> 10 -> 14 -> 15  (avoids holes 5, 7, 11, 12)
    """
    LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

    path = {0: DOWN, 4: DOWN, 8: RIGHT, 9: RIGHT, 10: RIGHT, 14: DOWN}
    if s in path:
        return path[s]
    row, col = divmod(s, 4)                       # fallback: head down-right
    if row < 3:
        return DOWN
    if col < 3:
        return RIGHT
    return LEFT


def main():
    # ------------------------- 1D walks ------------------------------------
    walk = BanditWalk(n_states=6, start=2, seed=0)
    slippery = BanditSlipperyWalk(n_states=6, start=4, slip_prob=0.25, seed=0)
    right = fixed_policy(1)

    print("========== BanditWalk (deterministic) ==========")
    print_environment(walk)
    print_policy(right, walk)
    play_episode(walk, right, delay=0.5)

    # print("========== BanditSlipperyWalk (slip = 0.25) ==========")
    # print_environment(slippery)
    # print_transitions(slippery.P, slippery.actions)
    # play_episode(slippery, right, delay=0.5)

    # print("\nMC metrics, always RIGHT (N=2000, seeded):")
    # for slip in (0.0, 0.1, 0.25, 0.5):
    #     env = BanditSlipperyWalk(n_states=6, start=4, slip_prob=slip, seed=0)
    #     m = mean_return(env, right, n_episodes=2000, seed=0)
    #     sd = std_return(env, right, n_episodes=2000, seed=0)
    #     p = probability_success(env, right, n_episodes=2000, seed=0)
    #     print(f"  slip={slip:.2f}  return={m:.4f} +- {sd:.4f}  P(success)={p:.4f}")

    # # ------------------------- FrozenLake ----------------------------------
    # print("========== FrozenLake (deterministic, path policy) ==========")
    # fl = FrozenLake(n=4, slippery=False, seed=0)
    # print_environment(fl)
    # print_policy(path_policy, fl)
    # play_episode(fl, path_policy, delay=0.5)

    # print("========== FrozenLake (slippery, path policy) ==========")
    # fl2 = FrozenLake(n=4, slippery=True, seed=0)
    # print_environment(fl2)
    # print_transitions(fl2.P, fl2.actions, states=[0, 5, 15])
    # play_episode(fl2, path_policy, delay=0.5)

    # print("\nMC metrics under the hand-picked path policy (N=2000, seeded):")
    # fl_det = FrozenLake(n=4, slippery=False, seed=0)
    # fl_slip = FrozenLake(n=4, slippery=True, seed=0)
    # print(f"  slippery=False  return={mean_return(fl_det, path_policy, 2000, seed=0):.4f}  "
    #       f"P(success)={probability_success(fl_det, path_policy, n_episodes=2000, seed=0):.4f}")
    # print(f"  slippery=True   return={mean_return(fl_slip, path_policy, 2000, seed=0):.4f}  "
    #       f"P(success)={probability_success(fl_slip, path_policy, n_episodes=2000, seed=0):.4f}")

    # print("\nDP exact values under the path policy (slippery):")
    # print_state_value_function(policy_evaluation(fl2, path_policy), fl2)


if __name__ == "__main__":
    main()
"""Pretty-printing helpers + current-state visualization."""

from typing import Callable, Iterable

from models import ActionMap, Environment, Policy, ProbabilityTransition


def is_absorbing(s: int, P: ProbabilityTransition) -> bool:
    """True if every branch of every action keeps you in s and ends the episode."""
    for outcomes in P[s].values():
        for o in outcomes:
            if not (o.next_state == s and o.done):
                return False
    return True


def _cell(env: Environment, s: int, core: str) -> str:
    """One fixed-width (9-char) cell; leading '*' marks the current state."""
    return ("*" if s == env.state else " ") + core


def _grid(env: Environment, core_fn: Callable[[int], str], title: str) -> None:
    """Print all states row-major (grid for FrozenLake when display_cols == n)."""
    print(title)
    n_cols = env.display_cols
    for s in range(env.n_states):
        print("| ", end="")
        print(_cell(env, s, core_fn(s)), end=" ")
        if (s + 1) % n_cols == 0:
            print("|")
    if env.n_states % n_cols != 0:
        print("|")


def _terminal_core(env: Environment, s: int) -> str:
    """8-char core for an absorbing cell: a centered G or H."""
    return f"{('G' if s == env.goal else 'H'):^8}"


def print_environment(env: Environment, title: str = "Environment:") -> None:
    print(title)
    env.display()


def print_policy(pi: Policy, env: Environment,
                 title: str = "Policy:  (* = agent's current state)") -> None:
    """pi maps state -> action CODE. Terminals show G/H; '*' marks current state."""
    actions = env.actions

    def core(s: int) -> str:
        if is_absorbing(s, env.P):
            return _terminal_core(env, s)
        return f"{s:02d}{actions[pi(s)].symbol:>6}"

    _grid(env, core, title)


def print_state_value_function(V, env: Environment, prec: int = 3,
                               title: str = "State-value function:") -> None:
    """V: state -> number. Terminals show G/H; '*' marks the current state."""
    def core(s: int) -> str:
        if is_absorbing(s, env.P):
            return _terminal_core(env, s)
        return f"{s:02d}{float(V[s]):>6.{prec}f}"

    _grid(env, core, title)


def print_transitions(P: ProbabilityTransition, actions: ActionMap,
                      states: Iterable[int] | None = None) -> None:
    """Print every outcome branch of each (state, action)."""
    print("Transition matrix P:")
    for s in (states if states is not None else P):
        for a, outcomes in P[s].items():
            for o in outcomes:
                print(f"  P(s'={o.next_state:2d} | s={s}, a={actions[a].name:<5}) = "
                      f"{o.prob:.2f}  reward={o.reward}  terminated={o.done}")
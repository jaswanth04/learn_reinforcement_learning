"""Monte Carlo prediction of V(s): every-visit (evmc) and first-visit (fvmc)."""

import numpy as np

from .schedules import AlphaSpec, decay_schedule


def evmc(env, policy, gamma=0.99, alpha: AlphaSpec = None,
         init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
         n_episodes=1000, first_visit=False,
         max_steps=5000, on_truncate='skip', collect_targets=True):
    """Every-visit MC prediction of V(s), with episode cap + V tracking.

    Returns (V, V_track, targets).

    on_truncate:
      'skip' -> discard episodes that hit max_steps before terminating
                (keeps MC returns unbiased; episodes are dropped, not used)
      'use'  -> update with the partial (truncated) return
                (biased, but you still make progress on long-horizon tasks)

    collect_targets=False (recommended for long episodes) skips building the
    targets dict so memory stays O(n_states * n_episodes), not O(total visits).
    """
    nS = env.n_states
    V = np.zeros(nS, dtype=np.float64)
    V_track = np.zeros((n_episodes, nS), dtype=np.float64)
    targets = {s: [] for s in range(nS)} if collect_targets else None
    N = np.zeros(nS, dtype=np.int64)

    # ---- step-size mode -----------------------------------------------
    if alpha is None:
        mode, alphas = 'mean', None
    elif isinstance(alpha, str) and alpha == 'decay':
        mode = 'decay'
        alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    elif callable(alpha):
        mode = 'decay'
        alphas = np.array([float(alpha(e, n_episodes)) for e in range(n_episodes)])
    else:
        mode = 'const'
        alphas = np.full(n_episodes, float(alpha))

    n_used = 0
    for e in range(n_episodes):
        # ---- collect ONE episode, HARD-CAPPED at max_steps ------------
        states, rewards = [], []
        s_t = env.reset()
        if s_t is None:
            s_t = env.state                     # fallback for state-setting envs

        done, steps = False, 0
        while not done and steps < max_steps:
            action = policy(s_t) if callable(policy) else policy[s_t]
            result = env.step(action)
            states.append(s_t)
            rewards.append(result.reward)
            s_t = result.next_state
            done = result.done
            steps += 1

        if not states:
            V_track[e] = V.copy()
            continue

        # MC needs complete returns -> drop truncated episodes (default)
        if not done and on_truncate == 'skip':
            V_track[e] = V.copy()
            continue

        # ---- returns-to-go, O(T) backward sweep ------------------------
        returns = np.empty(len(states))
        G = 0.0
        for i in range(len(states) - 1, -1, -1):
            G = rewards[i] + gamma * G
            returns[i] = G

        # ---- updates (first-visit or every-visit) ----------------------
        seen = set()
        for s, Gs in zip(states, returns):
            if first_visit:
                if s in seen:
                    continue
                seen.add(s)
            if targets is not None:             # collect_targets=True
                targets[s].append(Gs)
            if alphas is None:                  # mode == 'mean'
                N[s] += 1
                V[s] += (Gs - V[s]) / N[s]      # sample mean (true EVMC)
            else:
                V[s] += alphas[e] * (Gs - V[s]) # decay/constant step-size
        V_track[e] = V.copy()
        n_used += 1

    print(f"[evmc] used {n_used}/{n_episodes} episodes "
          f"({n_episodes - n_used} skipped: hit max_steps={max_steps})")
    return V.copy(), V_track, targets


def fvmc(env, policy, gamma=0.99, alpha: AlphaSpec = None,
         init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
         n_episodes=5000, max_steps=200):
    """First-visit Monte Carlo prediction of V(s), with V tracking.

    Only the FIRST occurrence of each state in an episode is used for the
    update. Returns are recorded/targeted for those first visits only.

    Returns (V, V_track, targets):
      - V       : final value function, shape (n_states,)
      - V_track : shape (n_episodes, n_states); V after each episode
      - targets : dict {state: [raw first-visit return-to-go per episode]}

    Step-size modes (alpha):
      - alpha=None       -> sample mean 1/N(s): canonical textbook FVMC
      - alpha='decay'    -> per-episode schedule init_alpha -> min_alpha
      - alpha=<float>    -> constant step-size (moving average)
      - alpha=<callable> -> custom per-episode alpha alpha(e, n_episodes)
    """
    nS = env.n_states
    V = np.zeros(nS, dtype=np.float64)
    V_track = np.zeros((n_episodes, nS), dtype=np.float64)
    targets = {s: [] for s in range(nS)}
    N = np.zeros(nS, dtype=np.int64)

    # ---- step-size mode (same options as evmc_updated) -----------------
    if alpha is None:
        mode, alphas = 'mean', None
    elif isinstance(alpha, str) and alpha == 'decay':
        mode = 'decay'
        alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    elif callable(alpha):
        mode = 'decay'
        alphas = np.array([float(alpha(e, n_episodes)) for e in range(n_episodes)])
    else:
        mode = 'const'
        alphas = np.full(n_episodes, float(alpha))

    for e in range(n_episodes):
        # ---- collect one episode ---------------------------------------
        states, rewards = [], []
        s_t = env.reset()
        if s_t is None:
            s_t = env.state                     # fallback for state-setting envs

        done = False
        steps = 0
        while not done and steps < max_steps:
            action = policy(s_t) if callable(policy) else policy[s_t]
            result = env.step(action)
            states.append(s_t)
            rewards.append(result.reward)
            s_t = result.next_state
            done = result.done
            steps += 1
        if not states:                          # empty episode -> no update
            V_track[e] = V
            continue

        # ---- returns-to-go, O(T) backward sweep ------------------------
        returns = np.empty(len(states))
        G = 0.0
        for i in range(len(states) - 1, -1, -1):
            G = rewards[i] + gamma * G
            returns[i] = G

        # ---- FIRST-visit updates ---------------------------------------
        seen = set()
        for s, Gs in zip(states, returns):
            if s in seen:                       # skip repeat visits to s!
                continue
            seen.add(s)

            targets[s].append(Gs)               # first-visit return only
            if alphas is None:                  # mode == 'mean'
                N[s] += 1
                V[s] += (Gs - V[s]) / N[s]      # sample mean (canonical FVMC)
            else:
                V[s] += alphas[e] * (Gs - V[s]) # decay/constant step-size

        V_track[e] = V                          # snapshot after episode e

    return V.copy(), V_track, targets

"""Temporal-difference prediction of V(s): TD(0), n-step TD and TD(lambda)."""

import numpy as np

from .schedules import AlphaSpec, _make_alphas, decay_schedule


def td(env, policy, gamma=0.99, alpha: AlphaSpec = None,
       init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
       n_episodes=5000, max_steps=200):
    """TD(0) prediction of V(s), with alpha decay and V tracking.

    Returns (V, V_track):
      - V       : final value function, shape (n_states,)
      - V_track : shape (n_episodes, n_states); V after each episode

    Step-size modes (alpha):
      - alpha='decay'   -> per-episode schedule init_alpha -> min_alpha
      - alpha=<float>   -> constant step-size (default if None: 0.1)
      - alpha=<callable>-> custom per-episode alpha alpha(e, n_episodes)
    """
    nS = env.n_states
    V = np.zeros(nS, dtype=np.float64)
    V_track = np.zeros((n_episodes, nS), dtype=np.float64)

    # ---- step-size mode ------------------------------------------------
    if isinstance(alpha, str) and alpha == 'decay':
        alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    elif callable(alpha):
        alphas = np.array([float(alpha(e, n_episodes)) for e in range(n_episodes)])
    else:
        if alpha is None:
            alpha = 0.1
        alphas = np.full(n_episodes, float(alpha))

    for e in range(n_episodes):
        s_t = env.reset()
        if s_t is None:
            s_t = env.state                     # fallback for state-setting envs

        done = False
        steps = 0
        while not done and steps < max_steps:   # max_steps actually used now
            action = policy(s_t) if callable(policy) else policy[s_t]
            result = env.step(action)
            next_state = result.next_state

            # --- TD(0) target ------------------------------------------
            # No bootstrap on the terminal transition: if the episode is
            # over, there is no V(s') term -- target is just the reward.
            td_target = result.reward
            if not result.done and next_state is not None:
                td_target += gamma * V[next_state]

            td_error = td_target - V[s_t]
            V[s_t] += alphas[e] * td_error

            done = result.done
            s_t = result.next_state
            steps += 1

        V_track[e] = V.copy()

    return V.copy(), V_track


# ---------------------------------------------------------------------------
# n-step TD
# ---------------------------------------------------------------------------
def ntd(env, policy, gamma=0.99, n=3, alpha: AlphaSpec = None,
        init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
        n_episodes=5000, max_steps=200):
    """n-step TD prediction of V(s).

    G_t^(n) = R_{t+1} + g*R_{t+2} + ... + g^{n-1} R_{t+n} + g^n V(S_{t+n})
    V(S_t) <- V(S_t) + a [G_t^(n) - V(S_t)]

    Special cases:
      n=1              -> TD(0)
      n=None           -> full return (every-visit MC with the same alpha mode)
      t+n past episode end -> return truncates to the full episode return.

    Returns (V, V_track): V_track[e, s] = V(s) after episode e.
    """
    nS = env.n_states
    V = np.zeros(nS, dtype=np.float64)
    V_track = np.zeros((n_episodes, nS), dtype=np.float64)
    alphas = _make_alphas(alpha, init_alpha, min_alpha, alpha_decay_ratio, n_episodes)

    for e in range(n_episodes):
        # ---- collect one episode ---------------------------------------
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

        T = len(states)
        if T == 0:
            V_track[e] = V.copy()
            continue

        # ---- full returns (used when t+n reaches/passes the episode end) --
        full = np.zeros(T)
        G = 0.0
        for k in range(T - 1, -1, -1):
            G = rewards[k] + gamma * G
            full[k] = G

        # ---- n-step returns + updates -----------------------------------
        for t in range(T):
            if n is None or t + n >= T:
                G_t = full[t]                   # no bootstrap past episode end
            else:
                G_t, gp = 0.0, 1.0
                for k in range(t, t + n):       # discounted n-step reward sum
                    G_t += gp * rewards[k]
                    gp *= gamma
                G_t += gp * V[states[t + n]]    # bootstrap at t+n (gp == g^n)
            V[states[t]] += alphas[e] * (G_t - V[states[t]])

        V_track[e] = V.copy()

    return V.copy(), V_track


# ---------------------------------------------------------------------------
# TD(lambda) with eligibility traces
# ---------------------------------------------------------------------------
def td_lambda(env, policy, gamma=0.99, lambda_=0.9, alpha: AlphaSpec = None,
              init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
              n_episodes=5000, max_steps=200, trace_type='accumulating'):
    """TD(lambda) prediction with eligibility traces (backward view).

    Per step:
      delta = R + g*V(S') - V(S)          (no bootstrap on terminal state)
      E(S)  <- E(S) + 1                   (accumulating) or E(S) <- 1 (replacing)
      V(s)  <- V(s) + a * delta * E(s)    for all s
      E     <- g * lambda * E             (decay all traces)

    Traces reset to zero at the start of each episode.
    lambda_=0 -> TD(0);  lambda_=1 -> online MC-like (biased, accumulating).

    Returns (V, V_track): V_track[e, s] = V(s) after episode e.
    """
    nS = env.n_states
    V = np.zeros(nS, dtype=np.float64)
    V_track = np.zeros((n_episodes, nS), dtype=np.float64)
    alphas = _make_alphas(alpha, init_alpha, min_alpha, alpha_decay_ratio, n_episodes)

    for e in range(n_episodes):
        E = np.zeros(nS, dtype=np.float64)      # eligibility traces
        s_t = env.reset()
        if s_t is None:
            s_t = env.state
        done, steps = False, 0
        while not done and steps < max_steps:
            action = policy(s_t) if callable(policy) else policy[s_t]
            result = env.step(action)
            next_state = result.next_state

            # TD error -- terminal transition contributes reward only
            delta = result.reward
            if not result.done and next_state is not None:
                delta += gamma * V[next_state]
            delta -= V[s_t]

            # update trace of the visited state
            if trace_type == 'replacing':
                E[s_t] = 1.0
            else:
                E[s_t] += 1.0

            # update every state in proportion to its trace
            V += alphas[e] * delta * E

            # decay all traces
            E *= gamma * lambda_

            done = result.done
            s_t = result.next_state
            steps += 1

        V_track[e] = V.copy()

    return V.copy(), V_track

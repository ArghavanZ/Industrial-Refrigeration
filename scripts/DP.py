from __future__ import annotations
from typing import Tuple, Dict
import numpy as np

# ---------------- Grid definition (21 x 21) ----------------
N = 21
room_vals = np.linspace(-22.0, -15.7, num=N, dtype=np.float64)
V_MIN, V_MAX = float(room_vals[0]), float(room_vals[-1])
DX = (V_MAX - V_MIN) / (N - 1)

def value_to_index(x: float) -> int:
    idx = int(round((x - V_MIN) / DX))
    return int(np.clip(idx, 0, N - 1))

def snap2grid(x: float) -> float:
    return room_vals[value_to_index(x)]

# ---------------------- Params ----------------------
params: Dict[str, float | np.ndarray] = {
    "room_n": 2,
    "bevap": np.array([22900, 22900]),
    "T_suction": -27.0,
    "T_rated": -25.0,
    "e_p": 1.0e-8,
    "delta_t": 60,
    "c_room": 40500000,
    "Q_dist": np.array([150000, 150000]),
    "Q_dist_delta": np.array([150000, 150000]),
    "high_penalty": 1.0,
    "low_penalty": 0.0,
    "max_temp": -16.0,
    "min_temp": -22.0,
    "Q_rated": 600000,
    "W_rated": 100000,
}

# ---------------------- Env ----------------------
State = Tuple[int, int]  # (i, j)

class env:
    def __init__(self, params: Dict[str, float | np.ndarray], seed: int | None = 0):
        self.params = params
        self.room_n = int(params["room_n"])
        assert self.room_n == 2, "This DP scaffold is for two rooms (2-D state)."
        self.bevap = np.asarray(params["bevap"], dtype=np.float64)
        self.T_suction = float(params["T_suction"])
        self.T_rated = float(params["T_rated"])
        self.Q_rated = float(params["Q_rated"])
        self.W_rated = float(params["W_rated"])
        self.e_p = float(params["e_p"])
        self.delta_t = float(params["delta_t"])
        self.c_room = float(params["c_room"])
        self.Q_dist_mean = np.asarray(params["Q_dist"], dtype=np.float64)
        self.Q_dist_delta = np.asarray(params["Q_dist_delta"], dtype=np.float64)
        self.high_penalty = float(params["high_penalty"])
        self.low_penalty = float(params["low_penalty"])
        self.max_temp = float(params["max_temp"])
        self.min_temp = float(params["min_temp"])
        self.lo = self.Q_dist_mean - self.Q_dist_delta
        self.hi = self.Q_dist_mean + self.Q_dist_delta
        # Derived caps at current T_suction:
        self.Q_max = (2000000/30) * (self.T_suction - self.T_rated) + self.Q_rated
        self.W_max = (-70) * (self.T_suction - self.T_rated)**2 + self.W_rated

        self.state_ij: Tuple[int, int] = (0, 0)
        self.rng = np.random.default_rng(seed)

    def reset(self, T1: float, T2: float) -> Tuple[int, int]:
        i = value_to_index(T1)
        j = value_to_index(T2)
        self.state_ij = (i, j)
        return self.state_ij

    def step_dp(self, state_ij: State, action: int) -> Tuple[State, float]:
        """One stochastic step. Uses self.rng for reproducibility."""
        # Decode action -> [0/1, 0/1]
        action_array = np.array([(action >> k) & 1 for k in range(self.room_n)], dtype=np.float64)

        # Current temps
        i, j = state_ij
        T_sim = np.array([room_vals[i], room_vals[j]], dtype=np.float64)

        # Evap cooling per room
        Q_evap = self.bevap * (T_sim - self.T_suction) * action_array
        Q_evap = np.maximum(Q_evap, 0.0)

        # Capacity limit
        Q_total = min(np.sum(Q_evap), self.Q_rated)
        if np.sum(Q_evap) > 0.0:
            Q_evap = Q_evap * (Q_total / np.sum(Q_evap))

        # Compressor energy cost
        W = self.W_max * (0.61 * (Q_total / self.Q_max) + 0.39 * (Q_total > 0.0))
        C_cost = W * self.e_p * self.delta_t

        # Uniform disturbance per room
        Q_dist = self.rng.uniform(self.lo, self.hi)

        # Temperature update
        T_next = T_sim + (self.delta_t / self.c_room) * (Q_dist - Q_evap)

        # Penalties
        high_violation = np.maximum(0.0, T_next - self.max_temp)
        low_violation  = np.maximum(0.0, self.min_temp - T_next)
        penalty_cost = (self.high_penalty * np.sum(high_violation) +
                        self.low_penalty  * np.sum(low_violation)) * self.delta_t

        reward = -float(C_cost + penalty_cost)

        # Snap to grid
        i2 = value_to_index(float(T_next[0]))
        j2 = value_to_index(float(T_next[1]))
        return (i2, j2), reward

# ----------------- DP (with Monte Carlo expectation) -----------------
class DPAgent:
    def __init__(self, gamma: float = 0.99, n_actions: int = 4, n_samples: int = 20, seed: int | None = 0):
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_samples = n_samples
        self.V = np.zeros((N, N), dtype=np.float64)
        self.pi = np.zeros((N, N), dtype=np.int32)
        self.env = env(params, seed=seed)

    def _backup(self, i: int, j: int) -> Tuple[float, int]:
        """Monte-Carlo Bellman backup with n_samples for each action."""
        q = np.empty(self.n_actions, dtype=np.float64)
        for a in range(self.n_actions):
            ret = 0.0
            for _ in range(self.n_samples):
                (ni, nj), r = self.env.step_dp((i, j), a)
                ret += r + self.gamma * self.V[ni, nj]
            q[a] = ret / self.n_samples
        a_star = int(np.argmax(q))
        return float(q[a_star]), a_star

    def value_iteration(self, theta: float = 1e-6, max_iters: int = 10000, synchronous: bool = True):
        for iter in range(max_iters):
            delta = 0.0
            if synchronous:
                V_new = np.empty_like(self.V)
            for i in range(N):
                for j in range(N):
                    v_old = self.V[i, j]
                    v_star, a_star = self._backup(i, j)
                    if synchronous:
                        V_new[i, j] = v_star
                    else:
                        self.V[i, j] = v_star
                    self.pi[i, j] = a_star
                    delta = max(delta, abs(v_star - v_old))
            if synchronous:
                self.V[:, :] = V_new
                if iter % 1000 == 0:
                    np.savez("DP_middle_acc.npz", V_vi=self.V, Pi_vi=self.pi)
            if delta < theta:
                break
        return self.V, self.pi

    def policy_evaluation(self, theta: float = 1e-6, max_iters: int = 100000):
        """Iterative evaluation under current policy using Monte-Carlo expectations."""
        for _ in range(max_iters):
            delta = 0.0
            for i in range(N):
                for j in range(N):
                    a = int(self.pi[i, j])
                    # Monte Carlo average for V^\pi(i,j)
                    ret = 0.0
                    for _ in range(self.n_samples):
                        (ni, nj), r = self.env.step_dp((i, j), a)
                        ret += r + self.gamma * self.V[ni, nj]
                    v = ret / self.n_samples
                    delta = max(delta, abs(v - self.V[i, j]))
                    self.V[i, j] = v
            if delta < theta:
                break

    def policy_improvement(self) -> bool:
        stable = True
        for i in range(N):
            for j in range(N):
                old_a = int(self.pi[i, j])
                _, new_a = self._backup(i, j)
                self.pi[i, j] = new_a
                if new_a != old_a:
                    stable = False
        return not stable

    def policy_iteration(self, eval_theta: float = 1e-6):
        while True:
            self.policy_evaluation(theta=eval_theta)
            changed = self.policy_improvement()
            if not changed:
                break
        return self.V, self.pi

# ---------------------- Example ----------------------
if __name__ == "__main__":
    agent = DPAgent(gamma=0.99, n_actions=4, n_samples=25, seed=42)

    # Value Iteration (recommended first)
    V_vi, Pi_vi = agent.value_iteration(theta=1e-5, synchronous=True)  # in-place often faster
    np.savez("DP_acc_150.npz", V_vi=V_vi, Pi_vi=Pi_vi)
    # Or Policy Iteration:
    # agent.V[:] = 0.0; agent.pi[:] = 0
    # V_pi, pi_pi = agent.policy_iteration(eval_theta=1e-5)

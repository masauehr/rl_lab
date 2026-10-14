"""基準（ベースライン）: ランダム方策・最短経路・動的計画法（価値反復）・方策の評価。

強化学習で学習した方策の良さは、これらの基準との比で測る。
  - ランダム方策: 何も学習しない下限
  - 最短経路（幅優先探索）: すべらない床での、最少の歩数
  - 価値反復: 遷移モデル P を使って、最適な価値と方策を厳密に求める（学習エージェントは P を知らない）
"""
from collections import deque

import numpy as np


def random_policy(env) -> np.ndarray:
    """全状態で、4つの行動を同じ確率で選ぶ方策（pi[s][a]）"""
    return np.full((env.n_states, env.n_actions), 1.0 / env.n_actions)


def shortest_path_length(env) -> int:
    """壁だけを考えた、スタートからゴールまでの最少歩数（幅優先探索）。崖は通れない扱い"""
    dist = {env.start: 0}
    q = deque([env.start])
    while q:
        s = q.popleft()
        if s == env.goal:
            return dist[s]
        r, c = env.state_to_cell[s]
        for a in range(4):
            s2 = env._move(s, a)
            r2, c2 = env.state_to_cell[s2]
            if env.layout[r2][c2] == "C" or s2 in dist:
                continue
            dist[s2] = dist[s] + 1
            q.append(s2)
    raise ValueError("ゴールに着けません")


def value_iteration(env, gamma: float = 1.0, theta: float = 1e-10, max_iter: int = 100_000):
    """最適な状態価値 V* と、最適方策（各状態で最良の行動）を求める。戻り値: (V, 方策の配列 pi[s]=行動)"""
    V = np.zeros(env.n_states)

    def q_values(V):
        Q = np.zeros((env.n_states, env.n_actions))
        for s in range(env.n_states):
            for a in range(env.n_actions):
                Q[s, a] = sum(p * (rew + (0.0 if done else gamma * V[s2])) for p, s2, rew, done in env.P[s][a])
        return Q

    for _ in range(max_iter):
        Q = q_values(V)
        V_new = Q.max(axis=1)
        if np.abs(V_new - V).max() < theta:
            V = V_new
            break
        V = V_new
    Q = q_values(V)
    return V, Q.argmax(axis=1)


def policy_evaluation(env, pi, gamma: float = 1.0, theta: float = 1e-10, max_iter: int = 100_000) -> np.ndarray:
    """方策 pi（pi[s][a] は確率。決定的な方策なら one-hot）の、状態価値 V を厳密に求める"""
    pi = np.asarray(pi)
    if pi.ndim == 1:  # 行動の番号の配列なら、one-hot にする
        pi = np.eye(env.n_actions)[pi]
    V = np.zeros(env.n_states)
    for _ in range(max_iter):
        V_new = np.zeros(env.n_states)
        for s in range(env.n_states):
            for a in range(env.n_actions):
                if pi[s, a] == 0:
                    continue
                V_new[s] += pi[s, a] * sum(p * (rew + (0.0 if done else gamma * V[s2])) for p, s2, rew, done in env.P[s][a])
        if np.abs(V_new - V).max() < theta:
            return V_new
        V = V_new
    return V


def evaluate_policy(env, policy, n_episodes: int = 100, seed: int = 1_000_000) -> dict:
    """方策を実際に動かして評価する（探索なし）。学習とは別のシードで行う。

    policy: 行動の番号の配列 pi[s]（決定的）、または pi[s][a]（確率）、または 関数 state -> 行動。
    戻り値: 平均リターン・標準偏差・ゴール到達率・平均歩数
    """
    rng = np.random.default_rng(seed)
    returns, steps, success = [], [], []
    for ep in range(n_episodes):
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        total, done, trunc = 0.0, False, False
        while not (done or trunc):
            if callable(policy):
                a = policy(s)
            else:
                pol = np.asarray(policy)
                a = int(pol[s]) if pol.ndim == 1 else int(rng.choice(env.n_actions, p=pol[s]))
            s, rew, done, trunc = env.step(a)
            total += rew
        returns.append(total); steps.append(env.t); success.append(done)
    return dict(mean_return=float(np.mean(returns)), std_return=float(np.std(returns)),
                success_rate=float(np.mean(success)), mean_steps=float(np.mean(steps)))

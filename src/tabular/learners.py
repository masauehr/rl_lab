"""Q学習（オフポリシー）と SARSA（オンポリシー）。モデル（遷移の確率）は使わず、env.step() だけで学ぶ。

Q[s, a] = 「状態 s で行動 a をして、その後うまくやったときの、リターンの見込み」を、表で持つ。
更新（1歩ごと）: Q[s,a] ← Q[s,a] + α ( 目標 − Q[s,a] )   （α=学習率。目標と今の値の差だけ、近づける）
  Q学習 の目標: r + γ max_a' Q[s', a']        （次の状態で、最良の行動をすると仮定する）
  SARSA の目標: r + γ Q[s', a']               （次の状態で、**実際に選ぶ行動** a' の値。探索も含めて評価する）
行動の選び方は ε-greedy: 確率 ε でランダム（探索）、それ以外は Q が最大の行動（活用）。
"""
import numpy as np

from rl_envs import evaluate_policy


def greedy_policy(Q: np.ndarray) -> np.ndarray:
    """Qが最大の行動（同点なら、いちばん小さい番号）。評価用の、探索なしの方策"""
    return Q.argmax(axis=1)


def _eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    row = Q[s]
    best = np.flatnonzero(row == row.max())          # 同点はランダムに選ぶ（先頭への偏りを避ける）
    return int(rng.choice(best))


def train(env, method: str = "q", episodes: int = 500, alpha: float = 0.5, gamma: float = 1.0, eps: float = 0.1,
          seed: int = 0, eval_every: int = 0, eval_episodes: int = 20, eval_seed: int = 2_000_000, q0: float = 0.0,
          eps_decay: float = 1.0, eps_min: float = 0.0) -> dict:
    """学習する。method は "q"（Q学習）か "sarsa"。

    戻り値: Q（Q表）, returns（学習中の、エピソードごとのリターン。探索ありの成績）, steps,
            eval（eval_every>0 のとき、(エピソード数, 探索なしの方策の平均リターン) のリスト。学習とは別のシードで評価）
    探索率: エピソード ep では eps_ep = max(eps_min, eps * eps_decay**ep)。eps_decay<1 なら、学習が進むほど探索を減らす。
    終了の扱い: ゴール（terminated）では、先の価値を 0 にする。歩数切れ（truncated）は、終わりではないので先の価値を使う。
    """
    rng = np.random.default_rng(seed)
    Q = np.full((env.n_states, env.n_actions), q0, dtype=float)
    if hasattr(env, "goal"):
        Q[env.goal] = 0.0                                    # ゴールのある環境（迷路）では、ゴールの価値は0
    returns, steps, evals = [], [], []
    for ep in range(episodes):
        eps_ep = max(eps_min, eps * eps_decay ** ep)
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        a = _eps_greedy(Q, s, eps_ep, rng)
        total, done, trunc = 0.0, False, False
        while not (done or trunc):
            s2, r, done, trunc = env.step(a)
            a2 = _eps_greedy(Q, s2, eps_ep, rng)
            if done:
                target = r
            elif method == "q":
                target = r + gamma * Q[s2].max()
            else:
                target = r + gamma * Q[s2, a2]
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s2, a2
            total += r
        returns.append(total); steps.append(env.t)
        if eval_every and (ep + 1) % eval_every == 0:
            res = evaluate_policy(env, greedy_policy(Q), n_episodes=eval_episodes, seed=eval_seed)
            evals.append((ep + 1, res["mean_return"]))
    return dict(Q=Q, returns=np.array(returns), steps=np.array(steps), eval=evals)


def optimal_action_rate(Q: np.ndarray, Q_star: np.ndarray, env, tol: float = 1e-9) -> float:
    """学習した（探索なしの）方策が、最適な行動を選んでいる状態の割合。ゴール・崖は除く"""
    states = [s for s in range(env.n_states) if s != env.goal and env.layout[env.state_to_cell[s][0]][env.state_to_cell[s][1]] != "C"]
    pi = greedy_policy(Q)
    return float(np.mean([Q_star[s, pi[s]] >= Q_star[s].max() - tol for s in states]))

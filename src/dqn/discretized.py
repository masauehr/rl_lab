"""比較用の基準: 連続の状態を、区切って（離散化して）、表形式のQ学習を使う。DQNより素朴な方法。"""
import numpy as np

# 区切りの境目（状態の各成分ごと）。台車の位置3・速度3・棒の角度6・角速度6 → 324通りの状態
EDGES = [np.array([-0.8, 0.8]), np.array([-0.5, 0.5]), np.array([-0.12, -0.06, 0.0, 0.06, 0.12]),
         np.array([-1.0, -0.5, 0.0, 0.5, 1.0])]
SHAPE = tuple(len(e) + 1 for e in EDGES)


def discretize(state) -> int:
    idx = tuple(int(np.digitize(state[i], EDGES[i])) for i in range(4))
    return int(np.ravel_multi_index(idx, SHAPE))


def train_discretized(env, seed=0, episodes=2000, alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=0.995,
                      eval_every=100, eval_episodes=10, eval_seed=2_000_000) -> dict:
    """離散化したQ学習。探索率は、エピソードごとに 0.995 倍ずつ減らす"""
    rng = np.random.default_rng(seed)
    n_states = int(np.prod(SHAPE))
    Q = np.zeros((n_states, env.n_actions))
    returns, evals = [], []
    for ep in range(episodes):
        eps = max(eps_end, eps_start * eps_decay ** ep)
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ds, ret, done, trunc = discretize(s), 0.0, False, False
        while not (done or trunc):
            a = int(rng.integers(env.n_actions)) if rng.random() < eps else int(rng.choice(np.flatnonzero(Q[ds] == Q[ds].max())))
            s2, r, done, trunc = env.step(a)
            ds2 = discretize(s2)
            target = r if done else r + gamma * Q[ds2].max()
            Q[ds, a] += alpha * (target - Q[ds, a])
            ds, ret = ds2, ret + r
        returns.append(ret)
        if eval_every and (ep + 1) % eval_every == 0:
            erng = np.random.default_rng(eval_seed)
            tot = []
            for _ in range(eval_episodes):
                s = env.reset(seed=int(erng.integers(0, 2**31 - 1)))
                er, d, tr = 0.0, False, False
                while not (d or tr):
                    s, r, d, tr = env.step(int(Q[discretize(s)].argmax()))
                    er += r
                tot.append(er)
            evals.append((ep + 1, float(np.mean(tot))))
    return dict(returns=np.array(returns), eval=evals, Q=Q)

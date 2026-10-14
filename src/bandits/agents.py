"""バンディットのエージェント（方策）。どれも、n_runs 個の問題を同時に扱う。

共通の使い方:  a = agent.select(t)  →  r = env.pull(a)  →  agent.update(a, r)
  Q[i, j] : 問題 i での、腕 j の価値の推定（これまでの報酬の平均）
  N[i, j] : 腕 j を引いた回数
"""
import numpy as np


class _Base:
    def __init__(self, n_runs: int, k: int, seed: int = 0):
        self.n_runs, self.k = n_runs, k
        self.rng = np.random.default_rng(seed)
        self.N = np.zeros((n_runs, k))
        self.Q = np.zeros((n_runs, k))

    def _argmax_random_ties(self, x: np.ndarray) -> np.ndarray:
        """最大の腕を選ぶ。同点なら、ランダムに1つ（先頭に偏らないように）"""
        return (x + self.rng.random(x.shape) * 1e-9).argmax(axis=1)

    def update(self, a: np.ndarray, r: np.ndarray, alpha=None):
        idx = np.arange(self.n_runs)
        self.N[idx, a] += 1
        step = (1.0 / self.N[idx, a]) if alpha is None else alpha      # 標本平均 or 一定の学習率
        self.Q[idx, a] += step * (r - self.Q[idx, a])


class EpsilonGreedy(_Base):
    """確率 eps でランダムな腕（探索）、それ以外は推定が最大の腕（活用）。
    eps=0 なら貪欲（探索なし）。q0 は価値の初期値（大きくすると楽観的になり、最初に全部の腕を試す）。
    alpha を指定すると、一定の学習率で更新する（環境が変わる場合に向く）。"""

    def __init__(self, n_runs, k, eps=0.1, q0=0.0, alpha=None, seed=0):
        super().__init__(n_runs, k, seed)
        self.eps, self.alpha = eps, alpha
        self.Q[:] = q0

    def select(self, t):
        greedy = self._argmax_random_ties(self.Q)
        rand = self.rng.integers(0, self.k, self.n_runs)
        explore = self.rng.random(self.n_runs) < self.eps
        return np.where(explore, rand, greedy)

    def update(self, a, r):
        super().update(a, r, self.alpha)


class UCB(_Base):
    """信頼上限（UCB1）。推定＋不確かさのボーナス c*sqrt(ln t / N) が最大の腕を選ぶ。
    あまり引いていない腕ほど、ボーナスが大きい（まだ分からないので、試す価値がある）。"""

    def __init__(self, n_runs, k, c=2.0, seed=0):
        super().__init__(n_runs, k, seed)
        self.c = c

    def select(self, t):
        untried = self.N == 0
        bonus = self.c * np.sqrt(np.log(t + 1) / np.maximum(self.N, 1))
        score = np.where(untried, np.inf, self.Q + bonus)       # 引いたことのない腕を先に試す
        return self._argmax_random_ties(np.where(np.isinf(score), 1e9, score))


class Thompson(_Base):
    """トンプソン抽出。各腕の価値を確率分布で持ち、そこから1つずつサンプルして、最大の腕を選ぶ。
    gaussian: 平均の事後分布 N(Σr/(N+1), 1/(N+1))（事前 N(0,1)、報酬の分散1と既知）
    bernoulli: ベータ分布 Beta(当たり+1, 外れ+1)"""

    def __init__(self, n_runs, k, kind="gaussian", seed=0):
        super().__init__(n_runs, k, seed)
        self.kind = kind
        self.S = np.zeros((n_runs, k))   # 報酬の合計

    def select(self, t):
        if self.kind == "gaussian":
            mean = self.S / (self.N + 1)
            sd = 1.0 / np.sqrt(self.N + 1)
            sample = self.rng.normal(mean, sd)
        else:
            sample = self.rng.beta(self.S + 1, self.N - self.S + 1)
        return self._argmax_random_ties(sample)

    def update(self, a, r):
        idx = np.arange(self.n_runs)
        self.N[idx, a] += 1
        self.S[idx, a] += r
        self.Q[idx, a] = self.S[idx, a] / self.N[idx, a]

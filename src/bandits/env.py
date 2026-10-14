"""多腕バンディット環境。同じ問題を n_runs 個、まとめて（ベクトルで）動かす。

k本の腕があり、腕を引くと報酬が返る。どの腕が良いかは分からないので、引いて確かめるしかない。
  gaussian : 腕 i の真の平均 μ_i ~ N(0,1)。引くと N(μ_i, 1) の報酬
  bernoulli: 腕 i の当たり確率 p_i ~ U(0,1)。引くと 1（当たり）か 0（外れ）
  nonstationary=True なら、真の平均が毎回ゆっくり（N(0, drift²) ずつ。既定 drift=0.01）ランダムに動く（環境が変わる）
"""
import numpy as np


class BanditBatch:
    def __init__(self, n_runs: int, k: int = 10, kind: str = "gaussian", nonstationary: bool = False, seed: int = 0,
                 drift: float = 0.01):
        self.n_runs, self.k, self.kind, self.nonstationary, self.drift = n_runs, k, kind, nonstationary, drift
        self.rng = np.random.default_rng(seed)
        if kind == "gaussian":
            self.mu = self.rng.normal(0.0, 1.0, (n_runs, k))
        elif kind == "bernoulli":
            self.mu = self.rng.uniform(0.0, 1.0, (n_runs, k))
        else:
            raise ValueError(kind)

    def best_value(self) -> np.ndarray:
        """各問題の、いちばん良い腕の真の平均"""
        return self.mu.max(axis=1)

    def best_arm(self) -> np.ndarray:
        return self.mu.argmax(axis=1)

    def pull(self, actions: np.ndarray) -> np.ndarray:
        """各問題で、actions[i] 番の腕を引く。戻り値: 報酬（n_runs,）"""
        idx = np.arange(self.n_runs)
        m = self.mu[idx, actions]
        if self.kind == "gaussian":
            r = self.rng.normal(m, 1.0)
        else:
            r = (self.rng.random(self.n_runs) < m).astype(float)
        if self.nonstationary:
            self.mu = self.mu + self.rng.normal(0.0, self.drift, self.mu.shape)
        return r

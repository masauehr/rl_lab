import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bandits import BanditBatch, EpsilonGreedy, Thompson, UCB, run_experiment, summarize  # noqa: E402


class TestBandit(unittest.TestCase):
    def test_問題の統計(self):
        env = BanditBatch(5000, 10, "gaussian", seed=1)
        self.assertAlmostEqual(env.mu.mean(), 0.0, delta=0.03)
        self.assertAlmostEqual(env.mu.std(), 1.0, delta=0.03)
        envb = BanditBatch(5000, 10, "bernoulli", seed=1)
        self.assertTrue(((envb.mu >= 0) & (envb.mu <= 1)).all())

    def test_報酬は引いた腕の平均のまわり(self):
        env = BanditBatch(20000, 5, "gaussian", seed=2)
        a = np.full(20000, 3)
        r = env.pull(a)
        self.assertAlmostEqual(r.mean(), env.mu[:, 3].mean(), delta=0.05)

    def test_ベルヌーイの報酬は0か1(self):
        env = BanditBatch(1000, 5, "bernoulli", seed=3)
        r = env.pull(np.zeros(1000, dtype=int))
        self.assertTrue(set(np.unique(r)) <= {0.0, 1.0})

    def test_同じシードなら同じ結果(self):
        mk = lambda n, k, s: EpsilonGreedy(n, k, eps=0.1, seed=s)  # noqa: E731
        a = run_experiment(mk, n_runs=50, steps=100, seed=5)
        b = run_experiment(mk, n_runs=50, steps=100, seed=5)
        self.assertTrue(np.array_equal(a["reward"], b["reward"]))

    def test_引いた回数の合計はステップ数(self):
        env = BanditBatch(20, 6, seed=0)
        ag = EpsilonGreedy(20, 6, eps=0.2, seed=1)
        for t in range(50):
            a = ag.select(t); ag.update(a, env.pull(a))
        self.assertTrue((ag.N.sum(axis=1) == 50).all())

    def test_UCBは最初のk回で全部の腕を試す(self):
        env = BanditBatch(30, 7, seed=0)
        ag = UCB(30, 7, c=2.0, seed=1)
        for t in range(7):
            a = ag.select(t); ag.update(a, env.pull(a))
        self.assertTrue((ag.N == 1).all())

    def test_標本平均は報酬の平均に一致(self):
        ag = EpsilonGreedy(1, 3, eps=0.0)
        for r in (1.0, 3.0, 5.0):
            ag.update(np.array([0]), np.array([r]))
        self.assertAlmostEqual(ag.Q[0, 0], 3.0)

    def test_探索率1は_ランダムで最良の腕の割合は約1_k(self):
        res = run_experiment(lambda n, k, s: EpsilonGreedy(n, k, eps=1.0, seed=s), n_runs=500, steps=200, seed=7)
        self.assertAlmostEqual(res["optimal"].mean(), 0.1, delta=0.02)

    def test_学習するエージェントはランダムより報酬が高い(self):
        kw = dict(n_runs=500, steps=300, seed=9)
        rnd = run_experiment(lambda n, k, s: EpsilonGreedy(n, k, eps=1.0, seed=s), **kw)
        for mk in (lambda n, k, s: EpsilonGreedy(n, k, eps=0.1, seed=s), lambda n, k, s: UCB(n, k, seed=s),
                   lambda n, k, s: Thompson(n, k, seed=s)):
            self.assertGreater(run_experiment(mk, **kw)["reward"][-50:].mean(), rnd["reward"][-50:].mean() + 0.5)

    def test_後悔は単調に増える(self):
        res = run_experiment(lambda n, k, s: EpsilonGreedy(n, k, eps=0.1, seed=s), n_runs=200, steps=100, seed=11)
        self.assertTrue((np.diff(res["regret"]) >= -1e-12).all())

    def test_楽観的な初期値は最初に全部の腕を試す(self):
        env = BanditBatch(50, 5, seed=0)
        ag = EpsilonGreedy(50, 5, eps=0.0, q0=10.0, seed=1)
        for t in range(5):
            a = ag.select(t); ag.update(a, env.pull(a))
        self.assertTrue((ag.N == 1).all())   # 報酬が初期値10より小さいので、引いた腕の価値は下がり、次は別の腕へ

    def test_まとめの項目(self):
        res = run_experiment(lambda n, k, s: Thompson(n, k, seed=s), n_runs=100, steps=100, seed=1)
        self.assertEqual(set(summarize(res)), {"平均報酬_最後", "最良の腕の割合_最後", "累積の後悔", "後悔の標準誤差"})
        self.assertEqual(len(res["final_regret"]), 100)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dqn import QNet, ReplayBuffer, discretize, evaluate_net, train_discretized, train_dqn  # noqa: E402
from rl_envs.cartpole import CartPole  # noqa: E402


class TestReplay(unittest.TestCase):
    def test_ためて_ランダムに取り出せる(self):
        buf = ReplayBuffer(5, 4)
        for i in range(8):                       # 容量5を超えると、古いものから上書き
            buf.add(np.full(4, i, np.float32), i % 2, float(i), np.full(4, i + 1, np.float32), 0.0)
        self.assertEqual(buf.n, 5)
        s, a, r, s2, d = buf.sample(100, np.random.default_rng(0))
        self.assertTrue(set(r.tolist()) <= {3.0, 4.0, 5.0, 6.0, 7.0})     # 0,1,2 は上書きで消えた
        self.assertEqual(s.shape, (100, 4))


class TestDQN(unittest.TestCase):
    def test_同じシードなら同じ学習結果(self):
        a = train_dqn(CartPole(seed=1), seed=1, episodes=15, warmup=100, eval_every=0)
        b = train_dqn(CartPole(seed=1), seed=1, episodes=15, warmup=100, eval_every=0)
        self.assertTrue(np.array_equal(a["returns"], b["returns"]))
        pa, pb = a["qnet"].state_dict(), b["qnet"].state_dict()
        self.assertTrue(all(torch.equal(pa[k], pb[k]) for k in pa))

    def test_学習するとランダムより長くもつ(self):
        res = train_dqn(CartPole(seed=0), seed=0, episodes=120, eval_every=20, eval_episodes=5)
        self.assertGreater(max(v for _, v in res["eval"]), 60.0)         # ランダムは約25

    def test_評価は探索なしでネットワークを変えない(self):
        env = CartPole(seed=0)
        q = QNet()
        before = [p.clone() for p in q.parameters()]
        evaluate_net(env, q, episodes=3)
        self.assertTrue(all(torch.equal(b, p) for b, p in zip(before, q.parameters())))

    def test_ターゲットなしだとQ値が発散しやすい(self):
        full = train_dqn(CartPole(seed=0), seed=0, episodes=150, eval_every=0)
        nt = train_dqn(CartPole(seed=0), seed=0, episodes=150, use_target=False, eval_every=0)
        s = torch.zeros(1, 4)
        self.assertLess(float(full["qnet"](s).max()), 150.0)             # 真の上限は 1/(1-γ)=100 付近
        self.assertGreater(float(nt["qnet"](s).max()), 500.0)            # 発散

    def test_再生なしでは学習が崩れる(self):
        res = train_dqn(CartPole(seed=0), seed=0, episodes=150, use_replay=False, eval_every=50, eval_episodes=5)
        self.assertLess(max(v for _, v in res["eval"]), 60.0)


class TestDiscretized(unittest.TestCase):
    def test_離散化は範囲内で一意(self):
        ids = {discretize(np.array([x, xd, th, thd])) for x in (-1, 0, 1) for xd in (-1, 0, 1) for th in (-0.2, 0, 0.2) for thd in (-2, 0, 2)}
        self.assertGreater(len(ids), 20)
        self.assertTrue(all(0 <= i < 324 for i in ids))

    def test_学習する(self):
        res = train_discretized(CartPole(seed=0), seed=0, episodes=300, eval_every=100, eval_episodes=5)
        self.assertGreater(max(v for _, v in res["eval"]), 60.0)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rl_envs import evaluate_policy, make_env, optimal_q, value_iteration  # noqa: E402
from tabular import greedy_policy, optimal_action_rate, train  # noqa: E402


class TestTabular(unittest.TestCase):
    def test_同じシードなら同じ学習結果(self):
        a = train(make_env("maze"), "q", episodes=50, seed=3)
        b = train(make_env("maze"), "q", episodes=50, seed=3)
        self.assertTrue(np.array_equal(a["Q"], b["Q"]))
        self.assertTrue(np.array_equal(a["returns"], b["returns"]))

    def test_ゴールのQは0のまま(self):
        env = make_env("room")
        res = train(env, "q", episodes=100, seed=0)
        self.assertTrue((res["Q"][env.goal] == 0).all())

    def test_Q学習は迷路で最適方策に到達する(self):
        env = make_env("maze")
        V, pi = value_iteration(env)
        res = train(env, "q", episodes=300, alpha=0.5, eps=0.1, seed=1)
        r = evaluate_policy(env, greedy_policy(res["Q"]), n_episodes=20)
        self.assertEqual(r["mean_return"], V[env.start])   # -12

    def test_SARSAも迷路で最適方策に到達する(self):
        env = make_env("maze")
        V, pi = value_iteration(env)
        res = train(env, "sarsa", episodes=600, alpha=0.5, eps=0.05, seed=1)
        r = evaluate_policy(env, greedy_policy(res["Q"]), n_episodes=20)
        self.assertEqual(r["mean_return"], V[env.start])

    def test_学習が進むと学習中のリターンも上がる(self):
        res = train(make_env("room"), "q", episodes=200, seed=2)
        self.assertGreater(res["returns"][-20:].mean(), res["returns"][:20].mean() + 20)

    def test_崖では_Q学習は最適な崖沿いを学び_SARSAは安全な遠回りを学ぶ(self):
        env = make_env("cliff")
        q = train(env, "q", episodes=500, alpha=0.5, eps=0.1, seed=0)
        s = train(env, "sarsa", episodes=500, alpha=0.5, eps=0.1, seed=0)
        rq = evaluate_policy(env, greedy_policy(q["Q"]), 20)["mean_return"]
        rs = evaluate_policy(env, greedy_policy(s["Q"]), 20)["mean_return"]
        self.assertEqual(rq, -13.0)            # 崖沿いの最短
        self.assertLess(rs, -13.0)             # 遠回り
        self.assertGreater(rs, -30.0)          # でも、崖には落ちない
        self.assertGreater(s["returns"][-100:].mean(), q["returns"][-100:].mean())   # 学習中の成績は、SARSAのほうが良い

    def test_評価は探索なしで学習中のQを変えない(self):
        env = make_env("room")
        res = train(env, "q", episodes=40, seed=0, eval_every=10, eval_episodes=3)
        self.assertEqual([e for e, _ in res["eval"]], [10, 20, 30, 40])

    def test_最適な行動の割合(self):
        env = make_env("maze")
        V, pi = value_iteration(env)
        Qs = optimal_q(env, V)
        self.assertEqual(optimal_action_rate(Qs, Qs, env), 1.0)           # 最適なQなら100%
        self.assertLess(optimal_action_rate(np.zeros_like(Qs), Qs, env), 1.0)

    def test_報酬がまばらでも_楽観的な初期値なら全部の行動を試してゴールを見つける(self):
        mk = lambda: make_env("snake", step_reward=0.0, goal_reward=1.0, max_steps=300, seed=1)  # noqa: E731
        opt = train(mk(), "q", episodes=40, alpha=0.5, gamma=0.99, eps=0.1, seed=1, q0=1.0)
        zero = train(mk(), "q", episodes=40, alpha=0.5, gamma=0.99, eps=0.1, seed=1, q0=0.0)
        self.assertTrue((opt["steps"] < 300).any())          # 楽観的: 40エピソード以内にゴールに着く
        self.assertFalse((zero["steps"] < 300).any())        # 初期値0: ゴールを見つけられない

    def test_歩数切れでは先の価値を使う(self):
        env = make_env("room", max_steps=2)
        res = train(env, "q", episodes=30, seed=0)
        self.assertTrue((res["steps"] <= 2).all())


if __name__ == "__main__":
    unittest.main()

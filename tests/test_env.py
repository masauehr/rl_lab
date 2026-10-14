import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rl_envs import (LAYOUTS, GridWorld, evaluate_policy, make_env, policy_evaluation, random_policy,  # noqa: E402
                     shortest_path_length, value_iteration)


class TestGridWorld(unittest.TestCase):
    def test_状態の数は壁以外のマスの数(self):
        for name, lay in LAYOUTS.items():
            env = make_env(name)
            self.assertEqual(env.n_states, sum(ch != "#" for row in lay for ch in row), name)

    def test_壁と外にはぶつかって動かない(self):
        env = make_env("maze")
        # スタートは左上。上・左は外
        env.reset()
        for a in (0, 3):
            s2, rew, done, trunc = env.step(a)
            self.assertEqual(s2, env.start)
            env.reset()
        # スタートの右 (0,1) は床。その右 (0,2) は壁
        s1, *_ = env.step(1)
        self.assertEqual(env.state_to_cell[s1], (0, 1))
        s2, *_ = env.step(1)
        self.assertEqual(s2, s1)

    def test_ゴールで終了し報酬は一歩ごとに_マイナス1(self):
        env = make_env("room")
        env.reset()
        total, done = 0.0, False
        for a in [2, 2, 2, 2, 1, 1, 1, 1]:  # 下×4、右×4 の最短経路
            s, rew, done, trunc = env.step(a)
            total += rew
        self.assertTrue(done)
        self.assertEqual(total, -8.0)

    def test_確率の合計は1(self):
        for slip in (0.0, 0.2, 0.5):
            env = make_env("cliff", slip=slip)
            for s in range(env.n_states):
                for a in range(4):
                    self.assertAlmostEqual(sum(p for p, *_ in env.P[s][a]), 1.0)

    def test_崖に落ちるとスタートに戻り大きな罰(self):
        env = make_env("cliff")
        env.reset()
        s, rew, done, trunc = env.step(1)       # スタートの右は崖
        self.assertEqual(s, env.start)
        self.assertEqual(rew, -100.0)
        self.assertFalse(done)

    def test_同じシードなら同じ経過(self):
        def run(seed):
            env = make_env("room", slip=0.3)
            env.reset(seed=seed)
            return [env.step(int(np.random.default_rng(i).integers(0, 4)))[0] for i in range(30)]
        self.assertEqual(run(5), run(5))

    def test_歩数が上限に達したら打ち切り(self):
        env = make_env("room", max_steps=3)
        env.reset()
        out = [env.step(3) for _ in range(3)]    # 左に壁へ当たり続ける
        self.assertTrue(out[-1][3])
        self.assertFalse(out[-1][2])

    def test_ゴールは吸収状態(self):
        env = make_env("room")
        for a in range(4):
            self.assertEqual(env.P[env.goal][a], [(1.0, env.goal, 0.0, True)])

    def test_スタートやゴールが無いとエラー(self):
        with self.assertRaises(ValueError):
            GridWorld(["....", "...G"])


class TestBaselines(unittest.TestCase):
    def test_最短経路の歩数(self):
        self.assertEqual(shortest_path_length(make_env("room")), 8)
        self.assertEqual(shortest_path_length(make_env("maze")), 12)

    def test_価値反復の最適リターンは_すべらないなら最短歩数のマイナス(self):
        for name in ("room", "maze"):
            env = make_env(name)
            V, pi = value_iteration(env)
            self.assertAlmostEqual(V[env.start], -shortest_path_length(env), places=6)

    def test_崖歩きの最適方策は崖沿いの最短(self):
        env = make_env("cliff")
        V, pi = value_iteration(env)
        self.assertAlmostEqual(V[env.start], -13.0, places=6)   # 上へ1・右11・下1

    def test_方策評価は価値反復の価値と一致(self):
        env = make_env("maze")
        V, pi = value_iteration(env)
        self.assertTrue(np.allclose(policy_evaluation(env, pi), V, atol=1e-6))

    def test_最適方策を実際に動かすと_すべらない環境では毎回同じ(self):
        env = make_env("maze")
        V, pi = value_iteration(env)
        r = evaluate_policy(env, pi, n_episodes=10)
        self.assertEqual(r["mean_return"], -12.0)
        self.assertEqual(r["std_return"], 0.0)
        self.assertEqual(r["success_rate"], 1.0)

    def test_すべる床でも_価値反復の方策は_ランダムよりずっと良い(self):
        env = make_env("room", slip=0.2)
        V, pi = value_iteration(env)
        best = evaluate_policy(env, pi, n_episodes=200)["mean_return"]
        rnd = evaluate_policy(env, random_policy(env), n_episodes=200)["mean_return"]
        self.assertGreater(best, rnd + 20)

    def test_シミュレーションの平均は厳密な価値に近い(self):
        env = make_env("room", slip=0.2)
        V, pi = value_iteration(env)
        sim = evaluate_policy(env, pi, n_episodes=4000)["mean_return"]
        self.assertAlmostEqual(sim, V[env.start], delta=0.3)

    def test_評価は同じシードなら同じ結果(self):
        env = make_env("room", slip=0.2)
        a = evaluate_policy(env, random_policy(env), 30, seed=3)
        b = evaluate_policy(env, random_policy(env), 30, seed=3)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()

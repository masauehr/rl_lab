import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from battery import BatteryEnv, idle_policy, rule_policy  # noqa: E402
from battery.env import BUY, ETA, LOAD, PV_PROFILE, PV_TOTAL, SELL, WEATHER_TRANS, hour_cost  # noqa: E402
from rl_envs import value_iteration  # noqa: E402
from run_phase5 import daily_cost  # noqa: E402


class TestHourCost(unittest.TestCase):
    def test_発電のない夜に待機なら_需要かける料金(self):
        cost, soc2 = hour_cost(2, 3, 0, 0, 6)
        self.assertAlmostEqual(cost, LOAD[2] * BUY[2])
        self.assertEqual(soc2, 3)

    def test_充電すると蓄電量が1増え_効率ぶん余計に買う(self):
        cost, soc2 = hour_cost(2, 3, 1, 0, 6)
        self.assertEqual(soc2, 4)
        self.assertAlmostEqual(cost, (LOAD[2] + 1 / ETA) * BUY[2])

    def test_放電すると蓄電量が1減り_効率ぶん届く(self):
        cost, soc2 = hour_cost(20, 3, 2, 0, 6)
        self.assertEqual(soc2, 2)
        self.assertAlmostEqual(cost, (LOAD[20] - ETA) * BUY[20])

    def test_満杯での充電と空での放電は_待機と同じ(self):
        self.assertEqual(hour_cost(2, 6, 1, 0, 6), hour_cost(2, 6, 0, 0, 6))
        self.assertEqual(hour_cost(20, 0, 2, 0, 6), hour_cost(20, 0, 0, 0, 6))

    def test_余った発電は安く売れる(self):
        cost, _ = hour_cost(12, 0, 0, 0, 6)         # 晴れの昼: 発電が需要を上回る
        net = LOAD[12] - PV_TOTAL[0] * PV_PROFILE[12]
        self.assertLess(net, 0)
        self.assertAlmostEqual(cost, SELL * net)    # 売電（マイナスの電気代）

    def test_往復の損(self):
        # 夜に買って夕方に使う: 1kWh 充電して 1kWh 放電すると、買った 1/η の電力で、η の電力が戻る
        self.assertAlmostEqual(ETA * ETA, 0.81)


class TestEnv(unittest.TestCase):
    def test_天気の遷移確率の各行は和が1(self):
        self.assertTrue(np.allclose(WEATHER_TRANS.sum(axis=1), 1.0))
        self.assertAlmostEqual(PV_PROFILE.sum(), 1.0)

    def test_状態の符号化は可逆(self):
        env = BatteryEnv()
        for h, soc, f in [(0, 0, 0), (13, 4, 2), (23, 6, 1)]:
            self.assertEqual(env.decode(env.encode(h, soc, f)), (h, soc, f))

    def test_モデルの確率の合計は1(self):
        env = BatteryEnv(forecast_error=0.2)
        for s in range(0, env.n_states, 7):
            for a in range(3):
                self.assertAlmostEqual(sum(p for p, *_ in env.P[s][a]), 1.0)

    def test_日が変わるときだけ予報が変わる(self):
        env = BatteryEnv(forecast_error=0.0)
        env.reset(seed=1)
        fs = []
        for i in range(48):
            s, *_ = env.step(0)
            fs.append(env.decode(s)[2])
        self.assertEqual(len(set(fs[:23])), 1)          # 同じ日の間は、予報は同じ

    def test_同じシードなら同じ経過(self):
        def run(seed):
            e = BatteryEnv(); e.reset(seed=seed)
            return [round(e.step(i % 3)[1], 6) for i in range(100)]
        self.assertEqual(run(5), run(5))

    def test_予報の外れ率を変えても天気の並びは変わらない(self):
        def weathers(err):
            e = BatteryEnv(forecast_error=err); e.reset(seed=3); w = []
            for _ in range(24 * 20):
                e.step(0); w.append(e.w)
            return w
        self.assertEqual(weathers(0.0), weathers(0.5))

    def test_予報の外れ率に近い頻度で外れる(self):
        e = BatteryEnv(days=2000, forecast_error=0.3); e.reset(seed=0)
        miss, n = 0, 0
        for t in range(2000 * 24):
            e.step(0)
            if e.h == 0:
                miss += e.f != e.w; n += 1
        self.assertAlmostEqual(miss / n, 0.3, delta=0.03)

    def test_歩数切れで打ち切る(self):
        e = BatteryEnv(days=2); e.reset(seed=0)
        out = [e.step(0) for _ in range(48)]
        self.assertTrue(out[-1][3])
        self.assertFalse(any(o[3] for o in out[:-1]))


class TestPolicies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = BatteryEnv(forecast_error=0.1)
        cls.V, cls.dp = value_iteration(cls.env, gamma=0.98)

    def test_蓄電池を使う方策は_何もしないより安い(self):
        idle = daily_cost(0.1, idle_policy(self.env), days=100)
        self.assertLess(daily_cost(0.1, rule_policy(self.env), days=100), idle - 50)
        self.assertLess(daily_cost(0.1, self.dp, days=100), idle - 50)

    def test_最適方策はルール以下の電気代(self):
        self.assertLessEqual(daily_cost(0.1, self.dp, days=300), daily_cost(0.1, rule_policy(self.env), days=300) + 1.0)

    def test_割引率が小さいと近視眼的で損(self):
        _, short = value_iteration(self.env, gamma=0.5)
        self.assertGreater(daily_cost(0.1, short, days=100), daily_cost(0.1, self.dp, days=100) + 30)

    def test_予報が正確なほど電気代が下がる(self):
        c0 = daily_cost(0.0, value_iteration(BatteryEnv(forecast_error=0.0), gamma=0.98)[1], days=300)
        c1 = daily_cost(2 / 3, value_iteration(BatteryEnv(forecast_error=2 / 3), gamma=0.98)[1], days=300)
        self.assertLess(c0, c1)

    def test_夕方の高い時間に放電し_夜の安い時間に充電する(self):
        s_eve = self.env.encode(19, 4, 1)
        s_night = self.env.encode(2, 0, 2)
        self.assertEqual(self.dp[s_eve], 2)
        self.assertEqual(self.dp[s_night], 1)


if __name__ == "__main__":
    unittest.main()

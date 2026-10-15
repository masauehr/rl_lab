"""家庭用蓄電池の充放電の環境（自作・合成データ）。

1時間ごとに、蓄電池を「充電／待機／放電」する。電気は、買う（時間帯別の料金）か、余ったら安く売る。
  状態 s = (時刻 h, 蓄電量 soc, 今日の天気予報 f)   … 天気予報は「晴・曇・雨」。外れることがある（forecast_error）
  行動 a = 0 待機 / 1 充電（1kWh） / 2 放電（1kWh）
  報酬 r = −（その1時間の電気代 円）。大きいほど（0に近いほど）良い。売電収入はマイナスの電気代。
  1エピソード = days 日（既定30日）。日が変わるたびに、天気が確率的に変わる（晴・曇・雨のマルコフ連鎖）。

発電量: 天気ごとの1日の合計（晴14・曇7・雨2 kWh）を、昼の山なりの形に配分する（anomaly_lab と同じ考え方の合成データ）。
需要  : 時刻ごとの一定の形（朝夕に多い。合計約15kWh/日）。
料金  : 買電 夜(0〜6時) 15円・昼(7〜16時) 28円・夕方(17〜22時) 40円・23時 28円 ／ 売電 8円。
蓄電池: 容量6kWh、1時間に1kWhずつ。充電・放電の効率はそれぞれ0.9（往復で0.81）。
遷移モデル P[s][a]（期待の報酬つき）も持つ。動的計画法（最適方策）に使う。学習するエージェントは使わない。
"""
import numpy as np

HOURS = 24
FORECAST_NAMES = ["晴", "曇", "雨"]
# 天気のマルコフ連鎖: 前日 → 翌日（晴・曇・雨）
WEATHER_TRANS = np.array([[0.65, 0.25, 0.10], [0.30, 0.45, 0.25], [0.25, 0.30, 0.45]])
PV_TOTAL = np.array([14.0, 7.0, 2.0])                       # 天気ごとの発電量の合計（kWh/日）
_bell = np.array([max(0.0, np.sin(np.pi * (h - 5.5) / 12)) if 6 <= h <= 17 else 0.0 for h in range(HOURS)])
PV_PROFILE = _bell / _bell.sum()                            # 昼に山なりの形（合計1）
LOAD = np.array([0.3] * 6 + [0.5, 0.9, 0.9] + [0.5] * 7 + [0.7, 1.0] + [1.5] * 4 + [1.0, 0.5])   # kWh/時（0〜23時）
BUY = np.array([15.0] * 7 + [28.0] * 10 + [40.0] * 6 + [28.0])    # 円/kWh
SELL = 8.0
POWER, ETA = 1, 0.9                                          # 1時間に動かせるkWh、片道の効率


def hour_cost(h: int, soc: int, a: int, w: int, capacity: int):
    """1時間の電気代（円）と、次の蓄電量。w は実際の天気。満杯での充電・空での放電は、待機と同じ"""
    charge = a == 1 and soc + POWER <= capacity
    discharge = a == 2 and soc - POWER >= 0
    soc2 = soc + POWER if charge else soc - POWER if discharge else soc
    net = LOAD[h] - PV_TOTAL[w] * PV_PROFILE[h] + (POWER / ETA if charge else 0.0) - (POWER * ETA if discharge else 0.0)
    cost = BUY[h] * net if net > 0 else SELL * net          # net<0 は売電（マイナスの電気代）
    return float(cost), soc2


class BatteryEnv:
    n_actions = 3

    def __init__(self, days: int = 30, forecast_error: float = 0.1, capacity: int = 6, seed: int = 0):
        self.days, self.e, self.cap = days, forecast_error, capacity
        self.n_soc = capacity + 1
        self.n_states = HOURS * self.n_soc * 3
        self.rng = np.random.default_rng(seed)               # 天気の乱数
        self.rng_f = np.random.default_rng(seed + 7919)       # 予報の乱数（天気とは別。予報の外れ率を変えても、天気の並びは変わらない）
        self.t, self.h, self.soc, self.w, self.f = 0, 0, 0, 0, 0
        self._P = None

    @property
    def P(self):
        """遷移モデル（初めて使うときに作る）。迷路の環境と同じ名前なので、value_iteration がそのまま使える"""
        if self._P is None:
            self._P = self.model()
        return self._P

    # 状態の番号 ↔ (時刻, 蓄電量, 予報)
    def encode(self, h, soc, f):
        return (h * self.n_soc + soc) * 3 + f

    def decode(self, s):
        f = s % 3
        hs = s // 3
        return hs // self.n_soc, hs % self.n_soc, f

    def _forecast(self, w):
        if self.rng_f.random() < self.e:                     # 外れ: ほかの2つからランダム
            return int(self.rng_f.choice([k for k in range(3) if k != w]))
        return int(w)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.rng_f = np.random.default_rng(seed + 7919)
        self.t, self.h, self.soc = 0, 0, 0
        self.w = int(self.rng.choice(3, p=[0.5, 0.3, 0.2]))
        self.f = self._forecast(self.w)
        return self.encode(self.h, self.soc, self.f)

    def step(self, a):
        """戻り値: (次の状態, 報酬＝−電気代, terminated(常にFalse), truncated(days日たったら))"""
        cost, self.soc = hour_cost(self.h, self.soc, a, self.w, self.cap)
        self.h += 1; self.t += 1
        if self.h == HOURS:                                  # 日が変わる: 天気が変わり、新しい予報が出る
            self.h = 0
            self.w = int(self.rng.choice(3, p=WEATHER_TRANS[self.w]))
            self.f = self._forecast(self.w)
        return self.encode(self.h, self.soc, self.f), -cost, False, self.t >= self.days * HOURS

    # ---- 遷移モデル（動的計画法用）----
    def p_actual_given_forecast(self):
        """予報 f のとき、実際の天気 w の確率 P(w|f)。天気が一様に出る（事前分布）と仮定した、単純なモデル"""
        P = np.full((3, 3), self.e / 2)
        np.fill_diagonal(P, 1 - self.e)
        return P

    def model(self):
        """P[s][a] = [(確率, 次の状態, 期待の報酬, 終了か), …]。期待の報酬は、予報から見た実際の天気の確率で平均したもの"""
        Pwf = self.p_actual_given_forecast()
        P = [[[] for _ in range(3)] for _ in range(self.n_states)]
        for h in range(HOURS):
            for soc in range(self.n_soc):
                for f in range(3):
                    s = self.encode(h, soc, f)
                    for a in range(3):
                        cost = sum(Pwf[f, w] * hour_cost(h, soc, a, w, self.cap)[0] for w in range(3))
                        soc2 = hour_cost(h, soc, a, 0, self.cap)[1]
                        if h < HOURS - 1:
                            P[s][a] = [(1.0, self.encode(h + 1, soc2, f), -cost, False)]
                        else:                                # 日が変わる: 実際の天気 → 翌日の天気 → 翌日の予報
                            nxt = {}
                            for w in range(3):
                                for w2 in range(3):
                                    for f2 in range(3):
                                        pf = (1 - self.e) if f2 == w2 else self.e / 2
                                        nxt[f2] = nxt.get(f2, 0.0) + Pwf[f, w] * WEATHER_TRANS[w, w2] * pf
                            P[s][a] = [(p, self.encode(0, soc2, f2), -cost, False) for f2, p in nxt.items()]
        return P


def idle_policy(env):
    """何もしない（蓄電池なしと同じ）。方策の配列 pi[s]"""
    return np.zeros(env.n_states, dtype=int)


def rule_policy(env):
    """ルール（AIなし）: 夜(0〜5時)は安いので、晴れ予報でなければ充電 ／ 昼(10〜14時)は、晴れ予報なら余った発電で充電 ／ 夕方(17〜22時)は放電"""
    pi = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        h, soc, f = env.decode(s)
        if 0 <= h <= 5 and f != 0:
            pi[s] = 1
        elif 10 <= h <= 14 and f == 0:
            pi[s] = 1
        elif 17 <= h <= 22:
            pi[s] = 2
    return pi

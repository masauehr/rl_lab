"""Phase5: 家庭用蓄電池の充放電制御。何もしない／ルール／動的計画法（最適）／Q学習 を、1日あたりの電気代（円）で比べる。

E1 基準との比較（予報の外れ10%）  E2 天気予報の価値と、予報への信頼を間違えたときの損
E3 Q学習の学習の速さ              E4 割引率（先を見る長さ）の影響

評価: 学習とは別のシード（2M番台）で、365日ぶんの電気代を平均する。すべての方策に、同じ天気の並びを使う（対応のある比較）。

使い方: .venv/bin/python src/run_phase5.py
出力:   outputs/phase5_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from battery import BatteryEnv, idle_policy, rule_policy  # noqa: E402
from rl_envs import value_iteration  # noqa: E402
from tabular import greedy_policy, train  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [1_000_000 + i for i in range(10)]
EVAL_SEED, EVAL_DAYS = 2_000_000, 365
GAMMA = 0.98          # 先を見る長さ（約50時間）。夜の充電を夕方に使うには、十分に長い必要がある（E4）


def daily_cost(forecast_error, pi, days=EVAL_DAYS, seed=EVAL_SEED) -> float:
    """方策 pi（状態→行動の配列）を、days 日ぶん動かした、1日あたりの電気代（円）"""
    env = BatteryEnv(days=days, forecast_error=forecast_error)
    s = env.reset(seed=seed)
    total, trunc = 0.0, False
    while not trunc:
        s, r, _, trunc = env.step(int(pi[s]))
        total -= r
    return total / days


def dp_policy(belief_error, gamma=GAMMA):
    """動的計画法の最適方策。belief_error は、予報の外れ率についての『思い込み』"""
    env = BatteryEnv(forecast_error=belief_error)
    return value_iteration(env, gamma=gamma)[1]


def q_policy(forecast_error, seed, episodes, gamma=GAMMA, alpha=0.2, q0=0.0):
    res = train(BatteryEnv(forecast_error=forecast_error, seed=seed), "q", episodes=episodes, alpha=alpha, gamma=gamma,
                eps=0.3, eps_decay=0.99, eps_min=0.02, seed=seed, q0=q0)
    return greedy_policy(res["Q"])


def exp1(e=0.1, episodes=1500, seeds=SEEDS):
    env = BatteryEnv(forecast_error=e)
    rows = []
    idle = daily_cost(e, idle_policy(env))
    for name, pi in [("何もしない（蓄電池なしと同じ）", idle_policy(env)), ("ルール（夜に充電・夕方に放電）", rule_policy(env)),
                     ("動的計画法（最適）", dp_policy(e))]:
        c = daily_cost(e, pi)
        rows.append({"方法": name, "電気代(円/日)": round(c, 1), "ばらつき(sd)": np.nan, "何もしないとの差(円/日)": round(idle - c, 1)})
    cs = [daily_cost(e, q_policy(e, sd, episodes)) for sd in seeds]
    rows.append({"方法": f"Q学習（{episodes}エピソード）", "電気代(円/日)": round(float(np.mean(cs)), 1), "ばらつき(sd)": round(float(np.std(cs)), 1),
                 "何もしないとの差(円/日)": round(idle - float(np.mean(cs)), 1)})
    return pd.DataFrame(rows)


def exp2(errors=(0.0, 0.1, 0.3, 0.5, 2 / 3)):
    """天気予報の外れ率 e ごとに、方策の電気代（実際の外れ率 e の環境で評価）。『予報を完全に信じる』方策も試す"""
    rows = []
    for e in errors:
        env = BatteryEnv(forecast_error=e)
        rows.append({"予報の外れ率": round(e, 2), "ルール": round(daily_cost(e, rule_policy(env)), 1),
                     "最適（外れ率を知っている）": round(daily_cost(e, dp_policy(e)), 1),
                     "予報を完全に信じる方策（外れ率0と思い込む）": round(daily_cost(e, dp_policy(0.0)), 1),
                     "予報を無視する方策（外れ率2/3＝当てずっぽうと思い込む）": round(daily_cost(e, dp_policy(2 / 3)), 1)})
    return pd.DataFrame(rows)


def exp3(e=0.1, episodes_list=(10, 30, 100, 300, 1000, 3000), seeds=SEEDS):
    rows = []
    for n in episodes_list:
        cs = [daily_cost(e, q_policy(e, sd, n)) for sd in seeds]
        rows.append({"学習エピソード数(1回=30日)": n, "学習した日数": n * 30, "電気代(円/日)": round(float(np.mean(cs)), 1),
                     "ばらつき(sd)": round(float(np.std(cs)), 1)})
    return pd.DataFrame(rows)


def exp4(e=0.1, gammas=(0.5, 0.8, 0.9, 0.95, 0.98, 0.995)):
    rows = []
    for g in gammas:
        rows.append({"割引率γ": g, "先を見る長さ(時間≒1/(1−γ))": round(1 / (1 - g), 1), "最適方策の電気代(円/日)": round(daily_cost(e, dp_policy(e, g)), 1)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    for name, fn in [("E1", exp1), ("E2", exp2), ("E3", exp3), ("E4", exp4)]:
        df = fn()
        print(f"\n== {name} ==")
        print(df.to_string(index=False))
        df.to_csv(ROOT / "outputs" / f"phase5_{name}.csv", index=False)

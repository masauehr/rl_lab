"""Phase3: 報酬設計・割引率・学習率・探索率の落とし穴。

A. 報酬ハッキング: 繰り返し取れるコインがあると、ゴールせずに往復して稼ぐ
B. 報酬がまばら: ゴールだけに報酬があると、学習が進まない。楽観的なQの初期値が効く。ポテンシャル型の整形は、ここでは助けにならない
C. 割引率: 価値の大きさは変えるが、十分大きければ最適方策は変えない。小さすぎると近視眼的
D. 学習率と探索率: すべる崖での、組み合わせ

使い方: .venv/bin/python src/run_phase3.py
出力:   outputs/phase3_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_envs import distance_to_goal, evaluate_policy, make_env, value_iteration  # noqa: E402
from tabular import greedy_policy, train  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [1_000_000 + i for i in range(10)]
EVAL_SEED = 2_000_000


def coin_visits(env, pol, n=20, seed=EVAL_SEED):
    """方策を動かして、コインのマスを踏んだ回数（1エピソードあたり）を数える"""
    rng = np.random.default_rng(seed)
    tot = 0
    for _ in range(n):
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = trunc = False
        while not (done or trunc):
            s, r, done, trunc = env.step(int(pol[s]))
            tot += env.layout[env.state_to_cell[s][0]][env.state_to_cell[s][1]] == "K"
    return tot / n


def exp_a(episodes=500, gamma=0.99, seeds=SEEDS):
    """A. 繰り返し取れるコイン。coin_reward を変えて、最適方策（動的計画法）と、Q学習の結果を見る"""
    rows = []
    for coin, goal in [(0, 0), (1, 0), (1.5, 0), (2.5, 0), (5, 0), (10, 0), (5, 500)]:
        env = make_env("coin", coin_reward=coin, goal_reward=goal)
        V, pi = value_iteration(env, gamma=gamma)
        opt = evaluate_policy(env, pi, 20, seed=EVAL_SEED)
        succ, coins, ret = [], [], []
        for sd in seeds:
            e = make_env("coin", coin_reward=coin, goal_reward=goal, seed=sd)
            res = train(e, "q", episodes=episodes, alpha=0.5, gamma=gamma, eps=0.1, seed=sd)
            pol = greedy_policy(res["Q"])
            ev = evaluate_policy(e, pol, 20, seed=EVAL_SEED)
            succ.append(ev["success_rate"]); ret.append(ev["mean_return"]); coins.append(coin_visits(e, pol))
        rows.append({"コインの報酬": coin, "ゴールの報酬": goal,
                     "最適方策: ゴールに着く": opt["success_rate"], "最適方策: コインを踏む回数": round(coin_visits(env, pi), 1),
                     "Q学習: ゴールに着く割合": round(float(np.mean(succ)), 2), "Q学習: コインを踏む回数": round(float(np.mean(coins)), 1),
                     "Q学習: 平均リターン": round(float(np.mean(ret)), 1)})
    return pd.DataFrame(rows)


def make_b_env(kind, seed):
    """B. 報酬の設計（迷路 snake・同じ形）: まばら／1歩ごとに−1／まばら＋ポテンシャル型の整形"""
    base = make_env("snake", step_reward=0.0, goal_reward=1.0, max_steps=300)
    if kind == "sparse":
        return make_env("snake", step_reward=0.0, goal_reward=1.0, max_steps=300, seed=seed)
    if kind == "dense":
        return make_env("snake", step_reward=-1.0, goal_reward=0.0, max_steps=300, seed=seed)
    return make_env("snake", step_reward=0.0, goal_reward=1.0, max_steps=300, potential=-distance_to_goal(base),
                    shaping_gamma=0.99, seed=seed)


B_SPECS = [  # (名前, 報酬の種類, Qの初期値, 割引率γ)
    ("まばら（ゴールだけ+1）・Q初期値0", "sparse", 0.0, 0.99),
    ("まばら・Q初期値1（楽観的）", "sparse", 1.0, 0.99),
    ("まばら・Q初期値1・割引なし(γ=1)", "sparse", 1.0, 1.0),
    ("1歩ごとに−1・Q初期値0（楽観的）", "dense", 0.0, 0.99),
    ("1歩ごとに−1・Q初期値−100（悲観的）", "dense", -100.0, 0.99),
    ("まばら＋整形（ポテンシャル型）・Q初期値0", "shaped", 0.0, 0.99),
]


def exp_b(episodes=1500, seeds=SEEDS):
    """B. 報酬がまばらなとき。最初にゴールするまでの速さと、学習後の方策。Q初期値（楽観的か）が効く"""
    rows = []
    for name, kind, q0, gamma in B_SPECS:
        first, succ_final, steps_final = [], [], []
        for sd in seeds:
            env = make_b_env(kind, sd)
            res = train(env, "q", episodes=episodes, alpha=0.5, gamma=gamma, eps=0.1, seed=sd, q0=q0)
            reached = np.flatnonzero(res["steps"] < env.max_steps)      # 歩数が上限より短ければ、ゴールした
            first.append(reached[0] + 1 if len(reached) else np.nan)
            ev = evaluate_policy(make_b_env(kind, sd), greedy_policy(res["Q"]), 20, seed=EVAL_SEED)
            succ_final.append(ev["success_rate"]); steps_final.append(ev["mean_steps"])
        rows.append({"報酬の設計": name, "ゴールできたシード数(10中)": int(np.sum(~np.isnan(first))),
                     "最初にゴールしたエピソード(中央値)": np.nanmedian(first) if not np.all(np.isnan(first)) else np.nan,
                     f"{episodes}エピソード後: 方策がゴールに着く割合": round(float(np.mean(succ_final)), 2),
                     "方策の平均歩数(最短53)": round(float(np.mean(steps_final)), 1)})
    return pd.DataFrame(rows)


def exp_c(**_):
    """C. 割引率γ。動的計画法で、最適方策を求め、その方策の「実際のリターン」（割引なし）を測る。すべる崖(0.2)"""
    env = make_env("cliff", slip=0.2)
    rows = []
    for gamma in (0.1, 0.5, 0.8, 0.9, 0.99, 1.0):
        V, pi = value_iteration(env, gamma=gamma)
        r = evaluate_policy(env, pi, 300, seed=EVAL_SEED)
        rows.append({"割引率γ": gamma, "最適な価値V*(スタート)": round(V[env.start], 2),
                     "その方策の実際のリターン(割引なし)": round(r["mean_return"], 2), "ゴールに着く割合": r["success_rate"]})
    return pd.DataFrame(rows)


def exp_d(episodes=1000, seeds=SEEDS):
    """D. 学習率αと探索率ε。すべる崖(0.2)で、Q学習の学習後の方策の成績（最適は −29.77）"""
    rows = []
    for alpha in (0.05, 0.1, 0.3, 0.5, 1.0):
        for eps in (0.01, 0.05, 0.1, 0.3):
            ev = []
            for sd in seeds:
                env = make_env("cliff", slip=0.2, seed=sd)
                res = train(env, "q", episodes=episodes, alpha=alpha, gamma=1.0, eps=eps, seed=sd)
                ev.append(evaluate_policy(env, greedy_policy(res["Q"]), 50, seed=EVAL_SEED)["mean_return"])
            rows.append({"学習率α": alpha, "探索率ε": eps, "学習後の成績(平均)": round(float(np.mean(ev)), 1),
                         "ばらつき(sd)": round(float(np.std(ev)), 1)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    for name, fn in [("A", exp_a), ("B", exp_b), ("C", exp_c), ("D", exp_d)]:
        df = fn()
        print(f"\n== {name} ==")
        print(df.to_string(index=False))
        df.to_csv(ROOT / "outputs" / f"phase3_{name}.csv", index=False)

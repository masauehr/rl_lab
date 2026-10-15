"""Phase6: 最終評価に使う設定を、開発用のシード（2M番台）で選ぶ。最終評価（3M番台）のデータは、ここでは一切使わない。

選ぶもの:
  迷路・崖（表形式）: Q学習・SARSA の学習率 α
  CartPole: DQN の設定（再生箱の大きさ・学習率・ターゲット更新・エピソード数）
  蓄電池: Q学習の学習率 α
選び方: 学習後の方策の成績（探索なし）の平均。**後から選んだ最高値（評価の最大）ではなく、最後の成績**で選ぶ。

使い方: .venv/bin/python src/select_dev.py
出力:   outputs/final/dev_selection.json（選んだ値。final_eval.py の FINAL_CONFIG に固定する）
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from battery import BatteryEnv  # noqa: E402
from dqn import train_dqn  # noqa: E402
from rl_envs import evaluate_policy, make_env, value_iteration  # noqa: E402
from rl_envs.cartpole import CartPole  # noqa: E402
from run_phase4 import summarize  # noqa: E402
from run_phase5 import daily_cost, q_policy  # noqa: E402
from tabular import greedy_policy, train  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEV_SEEDS = [2_000_000 + i for i in range(5)]      # 開発用の学習シード（5つ）
DEV_EVAL_SEED = 2_100_000                          # 開発用の評価シード
TABULAR_TASKS = {"maze": ("maze", {}), "cliff": ("cliff", {}), "cliff（すべる0.2）": ("cliff", dict(slip=0.2))}
EPS_SCHEDULE = dict(eps=0.3, eps_decay=0.995, eps_min=0.01)       # 探索を減らす（Phase2の教訓）
ALPHAS = (0.05, 0.1, 0.3, 0.5)
DQN_CANDIDATES = {
    "既定（lr1e-3・ターゲット200・箱1万・300回）": dict(),
    "安定重視（lr3e-4・ターゲット1000・300回）": dict(lr=3e-4, target_every=1000),
    "箱を大きく（5万・300回）": dict(buffer=50_000),
    "安定重視＋箱5万＋600回": dict(lr=3e-4, target_every=1000, buffer=50_000, episodes=600),
}
BATTERY_ALPHAS = (0.05, 0.2)


def reference_returns():
    """各迷路の、ランダム方策の平均リターンと、最適（動的計画法）の価値（正規化に使う）"""
    from rl_envs import random_policy
    ref = {}
    for label, (name, kw) in TABULAR_TASKS.items():
        env = make_env(name, **kw)
        ref[label] = dict(optimal=float(value_iteration(env)[0][env.start]),
                          random=float(evaluate_policy(env, random_policy(env), 500, seed=DEV_EVAL_SEED)["mean_return"]))
    return ref


def select_tabular(episodes=2000):
    ref = reference_returns()
    out, table = {}, {}
    for method in ("q", "sarsa"):
        scores = {}
        for alpha in ALPHAS:
            per_task = []
            for label, (name, kw) in TABULAR_TASKS.items():
                rets = []
                for sd in DEV_SEEDS:
                    env = make_env(name, seed=sd, **kw)
                    res = train(env, method, episodes=episodes, alpha=alpha, gamma=1.0, seed=sd, **EPS_SCHEDULE)
                    rets.append(evaluate_policy(env, greedy_policy(res["Q"]), 50, seed=DEV_EVAL_SEED)["mean_return"])
                r = ref[label]
                per_task.append((np.mean(rets) - r["random"]) / (r["optimal"] - r["random"]))
            scores[alpha] = float(np.mean(per_task))
            table[f"{method}|{alpha}"] = scores[alpha]
        out[method] = max(scores, key=scores.get)
    return out, table, ref


def select_dqn():
    table = {}
    for name, kw in DQN_CANDIDATES.items():
        kk = dict(episodes=300); kk.update(kw)
        sums = []
        for sd in DEV_SEEDS:
            r = train_dqn(CartPole(seed=sd), seed=sd, eval_every=20, eval_episodes=10, eval_seed=DEV_EVAL_SEED, **kk)
            sums.append(summarize(r["eval"])["最後の3回の平均"])
        table[name] = float(np.mean(sums))
    best = max(table, key=table.get)
    cfg = dict(DQN_CANDIDATES[best]); cfg.setdefault("episodes", 300)
    return best, cfg, table


def select_battery(episodes=3000):
    table = {}
    for alpha in BATTERY_ALPHAS:
        cs = [daily_cost(0.1, q_policy(0.1, sd, episodes, alpha=alpha), seed=DEV_EVAL_SEED) for sd in DEV_SEEDS]
        table[alpha] = float(np.mean(cs))
    return min(table, key=table.get), table


if __name__ == "__main__":
    (ROOT / "outputs" / "final").mkdir(parents=True, exist_ok=True)
    tab, tab_table, ref = select_tabular()
    print("表形式（正規化した成績。1=最適、0=ランダム）:", {k: round(v, 3) for k, v in tab_table.items()}, "→ 選択", tab)
    dqn_name, dqn_cfg, dqn_table = select_dqn()
    print("DQN（最後の3回の平均）:", {k: round(v, 1) for k, v in dqn_table.items()}, "→ 選択", dqn_name, dqn_cfg)
    bat_alpha, bat_table = select_battery()
    print("蓄電池のQ学習（電気代 円/日・小さいほど良い）:", {k: round(v, 1) for k, v in bat_table.items()}, "→ 選択 α =", bat_alpha)
    (ROOT / "outputs" / "final" / "dev_selection.json").write_text(json.dumps(
        dict(tabular_alpha=tab, tabular_table=tab_table, reference=ref, dqn_name=dqn_name, dqn_config=dqn_cfg, dqn_table=dqn_table,
             battery_alpha=bat_alpha, battery_table={str(k): v for k, v in bat_table.items()}), ensure_ascii=False, indent=2))

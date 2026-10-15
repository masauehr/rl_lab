"""Phase6: 最終評価。設定は開発用のシードで選び（select_dev.py）、下の FINAL_CONFIG に**固定**した。test は1回だけ使う。

評価する課題（それぞれ、基準と比べる）:
  迷路・崖（表形式）: maze・cliff・cliff（すべる0.2）で、Q学習・SARSA（探索を減らす）。ランダム／最適（動的計画法）で正規化
  CartPole          : ランダム／ルール／離散化した表形式Q学習／DQN（最後のネットワークで評価。後から選んだ最高値は使わない）
  蓄電池            : 何もしない／ルール／動的計画法（最適）／Q学習

シード: 学習 val=2.2M番台・test=3.0M番台、評価 val=2.3M・test=3.1M。開発用の選択（2.0M〜2.1M番台）とは別。
結果を見て設定を変えない。

使い方:
  .venv/bin/python src/final_eval.py --split val    # 予行演習（何度でも可）
  .venv/bin/python src/final_eval.py --split test   # 本番（1回だけ。使用済みだと拒否する）
出力: outputs/final/final_{split}.csv / final_{split}.json（test は TEST_USED.json も作る）
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from battery import idle_policy, rule_policy, BatteryEnv  # noqa: E402
from dqn import discretize, evaluate_net, train_discretized, train_dqn  # noqa: E402
from rl_envs import evaluate_policy, make_env, random_policy, value_iteration  # noqa: E402
from rl_envs.cartpole import CartPole, heuristic_policy  # noqa: E402
from run_phase4 import evaluate_fn  # noqa: E402
from run_phase5 import daily_cost, dp_policy, q_policy  # noqa: E402
from tabular import greedy_policy, train  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "final"

# select_dev.py の結果（開発用のシードで選んだもの）。最終評価の前に固定した
FINAL_CONFIG = dict(
    tabular_alpha=dict(q=0.05, sarsa=0.05), tabular_episodes=2000, tabular_eps=dict(eps=0.3, eps_decay=0.995, eps_min=0.01),
    dqn=dict(lr=3e-4, target_every=1000, episodes=300), discretized_episodes=2000,
    battery_alpha=0.2, battery_episodes=3000, battery_error=0.1,
)
SPLITS = {
    "val": dict(train_seeds=[2_200_000 + i for i in range(10)], eval_seed=2_300_000),
    "test": dict(train_seeds=[3_000_000 + i for i in range(10)], eval_seed=3_100_000),
}
TABULAR_TASKS = {"maze": ("maze", {}), "cliff": ("cliff", {}), "cliff（すべる0.2）": ("cliff", dict(slip=0.2))}


def tabular_rows(seeds, eval_seed, cfg=FINAL_CONFIG):
    rows = []
    for label, (name, kw) in TABULAR_TASKS.items():
        env0 = make_env(name, **kw)
        opt = float(value_iteration(env0)[0][env0.start])
        rnd = float(evaluate_policy(env0, random_policy(env0), 300, seed=eval_seed)["mean_return"])
        rows.append(dict(課題=label, 手法="ランダム", 平均=rnd, sd=np.nan, 正規化=0.0, 失敗シード=np.nan))
        rows.append(dict(課題=label, 手法="最適（動的計画法）", 平均=opt, sd=np.nan, 正規化=1.0, 失敗シード=np.nan))
        for method, mname in (("q", "Q学習（探索を減らす）"), ("sarsa", "SARSA（探索を減らす）")):
            rets, bad = [], 0
            for sd in seeds:
                env = make_env(name, seed=sd, **kw)
                res = train(env, method, episodes=cfg["tabular_episodes"], alpha=cfg["tabular_alpha"][method], gamma=1.0, seed=sd,
                            **cfg["tabular_eps"])
                ev = evaluate_policy(env, greedy_policy(res["Q"]), 100, seed=eval_seed)
                rets.append(ev["mean_return"]); bad += ev["success_rate"] < 0.9
            rows.append(dict(課題=label, 手法=mname, 平均=float(np.mean(rets)), sd=float(np.std(rets)),
                             正規化=float((np.mean(rets) - rnd) / (opt - rnd)), 失敗シード=int(bad)))
    return rows


def cartpole_rows(seeds, eval_seed, cfg=FINAL_CONFIG):
    rng = np.random.default_rng(eval_seed)
    rows = [dict(課題="CartPole", 手法="ランダム", 平均=evaluate_fn(lambda s: int(rng.integers(2)), 50, eval_seed), sd=np.nan),
            dict(課題="CartPole", 手法="ルール（倒れる向きに押す）", 平均=evaluate_fn(heuristic_policy, 50, eval_seed), sd=np.nan)]
    disc = []
    for sd in seeds:
        r = train_discretized(CartPole(seed=sd), seed=sd, episodes=cfg["discretized_episodes"], eval_every=0)
        disc.append(evaluate_fn(lambda s, Q=r["Q"]: int(Q[discretize(s)].argmax()), 50, eval_seed))
    rows.append(dict(課題="CartPole", 手法="離散化した表形式Q学習", 平均=float(np.mean(disc)), sd=float(np.std(disc)),
                     満点に近いシード=int(sum(v >= 195 for v in disc))))
    dq = []
    for sd in seeds:
        r = train_dqn(CartPole(seed=sd), seed=sd, eval_every=0, **cfg["dqn"])
        dq.append(evaluate_net(CartPole(), r["qnet"], episodes=50, seed=eval_seed))
    rows.append(dict(課題="CartPole", 手法="DQN（最後のネットワーク）", 平均=float(np.mean(dq)), sd=float(np.std(dq)),
                     満点に近いシード=int(sum(v >= 195 for v in dq))))
    return rows


def battery_rows(seeds, eval_seed, cfg=FINAL_CONFIG):
    e = cfg["battery_error"]
    env = BatteryEnv(forecast_error=e)
    idle = daily_cost(e, idle_policy(env), seed=eval_seed)
    rule = daily_cost(e, rule_policy(env), seed=eval_seed)
    dp = daily_cost(e, dp_policy(e), seed=eval_seed)
    qs = [daily_cost(e, q_policy(e, sd, cfg["battery_episodes"], alpha=cfg["battery_alpha"]), seed=eval_seed) for sd in seeds]
    full = idle - dp
    rows = [dict(課題="蓄電池（円/日）", 手法="何もしない", 平均=idle, sd=np.nan, 節約の割合=0.0),
            dict(課題="蓄電池（円/日）", 手法="ルール", 平均=rule, sd=np.nan, 節約の割合=(idle - rule) / full),
            dict(課題="蓄電池（円/日）", 手法="動的計画法（最適）", 平均=dp, sd=np.nan, 節約の割合=1.0),
            dict(課題="蓄電池（円/日）", 手法="Q学習", 平均=float(np.mean(qs)), sd=float(np.std(qs)), 節約の割合=float((idle - np.mean(qs)) / full))]
    return rows


def evaluate(split):
    s = SPLITS[split]
    rows = tabular_rows(s["train_seeds"], s["eval_seed"]) + cartpole_rows(s["train_seeds"], s["eval_seed"]) + battery_rows(s["train_seeds"], s["eval_seed"])
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    a = ap.parse_args()
    lock = OUT / "TEST_USED.json"
    if a.split == "test" and lock.exists():
        sys.exit(f"testは使用済みです（{lock}）。再評価が必要なら、新しいシード（4M番台など）を使ってください。")
    df = evaluate(a.split)
    OUT.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(OUT / f"final_{a.split}.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print(f"== {a.split}（学習シード {SPLITS[a.split]['train_seeds'][0]}〜、評価シード {SPLITS[a.split]['eval_seed']}）==")
    print(df.round(3).to_string(index=False))
    if a.split == "test":
        lock.write_text(json.dumps(dict(used_at=datetime.now().isoformat(timespec="seconds"), config=FINAL_CONFIG,
                                        train_seeds=SPLITS["test"]["train_seeds"][:1], eval_seed=SPLITS["test"]["eval_seed"],
                                        note="testは最終評価で使用済み。設定は開発用のシードで選んで固定した"), ensure_ascii=False, indent=2))
        print("\nTEST_USED.json を作成しました（以後 --split test は拒否されます）")


if __name__ == "__main__":
    main()

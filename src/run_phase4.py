"""Phase4: CartPole で、DQN（深層強化学習）を、基準（ランダム・ルール・離散化した表形式）と比べる。

学習中の成績（探索あり）と、学習後の方策の評価（探索なし・学習とは別のシード）を区別する。満点は 200（倒れずに200歩）。

使い方: .venv/bin/python src/run_phase4.py
出力:   outputs/phase4_summary.csv / outputs/phase4_curves.npz
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dqn import train_discretized, train_dqn  # noqa: E402
from rl_envs.cartpole import CartPole, heuristic_policy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [1_000_000 + i for i in range(10)]
EVAL_SEED = 2_000_000
EPISODES = 300                       # DQN の学習エピソード数
EVAL_EVERY, EVAL_EPISODES = 20, 10   # 20エピソードごとに、探索なしで10回評価


def evaluate_fn(policy, episodes=50, seed=EVAL_SEED):
    env, rng, tot = CartPole(), np.random.default_rng(seed), []
    for _ in range(episodes):
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ret, d, t = 0.0, False, False
        while not (d or t):
            s, r, d, t = env.step(policy(s)); ret += r
        tot.append(ret)
    return float(np.mean(tot))


def summarize(evals):
    """評価の曲線（チェックポイントごとの平均リターン）を、数字にまとめる"""
    v = np.array([x[1] for x in evals])
    return dict(最後の3回の平均=float(v[-3:].mean()), 最大=float(v.max()), 最後=float(v[-1]),
                満点に達した=bool((v >= 195).any()), 最大から落ちた幅=float(v.max() - v[-1]))


DQN_CONFIGS = {
    "DQN（再生あり・ターゲットあり）": dict(use_replay=True, use_target=True),
    "再生なし": dict(use_replay=False, use_target=True),
    "ターゲットなし": dict(use_replay=True, use_target=False),
    "DQN（学習率を下げ、ターゲット更新を遅く）": dict(use_replay=True, use_target=True, lr=3e-4, target_every=1000),
    "再生もターゲットもなし": dict(use_replay=False, use_target=False),
}


def run_all(verbose=True, seeds=SEEDS, episodes=EPISODES):
    rng = np.random.default_rng(0)
    rows, curves = [], {}
    rows.append({"手法": "ランダム", "評価(探索なし)": round(evaluate_fn(lambda s: int(rng.integers(2))), 1)})
    rows.append({"手法": "ルール（倒れる向きに押す）", "評価(探索なし)": round(evaluate_fn(heuristic_policy), 1)})
    # 離散化した表形式のQ学習
    ev = [train_discretized(CartPole(seed=sd), seed=sd, episodes=2000, eval_every=100, eval_episodes=EVAL_EPISODES) for sd in seeds]
    sums = [summarize(r["eval"]) for r in ev]
    curves["離散化した表形式（2000エピソード）"] = np.array([[v for _, v in r["eval"]] for r in ev])
    rows.append({"手法": "離散化した表形式Q学習（2000エピソード）", "評価(探索なし)": round(np.mean([s["最後の3回の平均"] for s in sums]), 1),
                 "sd": round(np.std([s["最後の3回の平均"] for s in sums]), 1), "最大": round(np.mean([s["最大"] for s in sums]), 1),
                 "満点に達したシード数": int(sum(s["満点に達した"] for s in sums)), "最大から落ちた幅": round(np.mean([s["最大から落ちた幅"] for s in sums]), 1)})
    for name, kw in DQN_CONFIGS.items():
        res = [train_dqn(CartPole(seed=sd), seed=sd, episodes=episodes, eval_every=EVAL_EVERY, eval_episodes=EVAL_EPISODES, **kw) for sd in seeds]
        sums = [summarize(r["eval"]) for r in res]
        curves[name] = np.array([[v for _, v in r["eval"]] for r in res])
        curves[name + "|学習中"] = np.array([r["returns"] for r in res])
        rows.append({"手法": name + f"（{episodes}エピソード）", "評価(探索なし)": round(np.mean([s["最後の3回の平均"] for s in sums]), 1),
                     "sd": round(np.std([s["最後の3回の平均"] for s in sums]), 1), "最大": round(np.mean([s["最大"] for s in sums]), 1),
                     "満点に達したシード数": int(sum(s["満点に達した"] for s in sums)), "最大から落ちた幅": round(np.mean([s["最大から落ちた幅"] for s in sums]), 1)})
    df = pd.DataFrame(rows)
    if verbose:
        pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
        print(df.to_string(index=False))
    return df, curves


if __name__ == "__main__":
    df, curves = run_all()
    (ROOT / "outputs").mkdir(exist_ok=True)
    df.to_csv(ROOT / "outputs" / "phase4_summary.csv", index=False)
    np.savez(ROOT / "outputs" / "phase4_curves.npz", **{k: v for k, v in curves.items()})

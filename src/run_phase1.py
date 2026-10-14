"""Phase1: 多腕バンディットで、探索と活用を調べる。条件を1つずつ変えて比べる。

使い方: .venv/bin/python src/run_phase1.py
出力:   outputs/phase1_summary.csv / outputs/phase1_curves.npz
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bandits import UCB, EpsilonGreedy, Thompson, run_experiment, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N_RUNS, STEPS, K = 2000, 1000, 10
SEED = 1_000_000          # 問題の乱数（実験用）。エージェントの乱数は SEED+1

# 実験名 → {手法名: (エージェントの作り方, 環境の設定)}
G = dict(kind="gaussian")
EXPERIMENTS = {
    "A 探索の量（ε）": {
        "貪欲 (ε=0)": (lambda n, k, s: EpsilonGreedy(n, k, eps=0.0, seed=s), G),
        "ε=0.01": (lambda n, k, s: EpsilonGreedy(n, k, eps=0.01, seed=s), G),
        "ε=0.1": (lambda n, k, s: EpsilonGreedy(n, k, eps=0.1, seed=s), G),
        "ε=0.3": (lambda n, k, s: EpsilonGreedy(n, k, eps=0.3, seed=s), G),
    },
    "B 楽観的な初期値": {
        "貪欲 (初期値0)": (lambda n, k, s: EpsilonGreedy(n, k, eps=0.0, seed=s), G),
        "貪欲 (初期値5)": (lambda n, k, s: EpsilonGreedy(n, k, eps=0.0, q0=5.0, seed=s), G),
        "ε=0.1 (初期値0)": (lambda n, k, s: EpsilonGreedy(n, k, eps=0.1, seed=s), G),
    },
    "C 賢い探索（UCB・トンプソン）": {
        "ε=0.1": (lambda n, k, s: EpsilonGreedy(n, k, eps=0.1, seed=s), G),
        "UCB (c=1)": (lambda n, k, s: UCB(n, k, c=1.0, seed=s), G),
        "UCB (c=2)": (lambda n, k, s: UCB(n, k, c=2.0, seed=s), G),
        "トンプソン": (lambda n, k, s: Thompson(n, k, seed=s), G),
    },
    "D 環境が変わる（非定常）": {
        "ε=0.1・標本平均": (lambda n, k, s: EpsilonGreedy(n, k, eps=0.1, seed=s), dict(kind="gaussian", nonstationary=True)),
        "ε=0.1・一定の学習率 α=0.1": (lambda n, k, s: EpsilonGreedy(n, k, eps=0.1, alpha=0.1, seed=s),
                                dict(kind="gaussian", nonstationary=True)),
    },
    "E 当たり外れ（ベルヌーイ）": {
        "ε=0.1": (lambda n, k, s: EpsilonGreedy(n, k, eps=0.1, seed=s), dict(kind="bernoulli")),
        "UCB (c=1)": (lambda n, k, s: UCB(n, k, c=1.0, seed=s), dict(kind="bernoulli")),
        "トンプソン": (lambda n, k, s: Thompson(n, k, kind="bernoulli", seed=s), dict(kind="bernoulli")),
    },
}


# 環境が変わる実験は、変化の効果が見えるよう長く動かす（1000回では差が出ない）
EXPERIMENT_STEPS = {"D 環境が変わる（非定常）": 5000}


def run_all(verbose: bool = True, n_runs: int = N_RUNS, steps: int = STEPS):
    rows, curves = [], {}
    for exp, methods in EXPERIMENTS.items():
        for name, (mk, env_kw) in methods.items():
            res = run_experiment(mk, n_runs=n_runs, steps=EXPERIMENT_STEPS.get(exp, steps) if steps == STEPS else steps,
                                 k=K, seed=SEED, **env_kw)
            s = summarize(res)
            rows.append(dict(実験=exp, 手法=name, **{k: round(v, 3) for k, v in s.items()}))
            curves[f"{exp}|{name}"] = res
    df = pd.DataFrame(rows)
    if verbose:
        pd.set_option("display.width", 160)
        for exp in EXPERIMENTS:
            print(f"\n== {exp} ==")
            print(df[df["実験"] == exp].drop(columns="実験").to_string(index=False))
    return df, curves


if __name__ == "__main__":
    df, curves = run_all()
    (ROOT / "outputs").mkdir(exist_ok=True)
    df.to_csv(ROOT / "outputs" / "phase1_summary.csv", index=False)

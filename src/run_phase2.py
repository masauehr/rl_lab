"""Phase2: Q学習と SARSA を、迷路・崖で比べる。10シードで、最適（動的計画法）と比べる。

学習中の成績（探索あり）と、学習後の方策の評価（探索なし・別のシード）を区別する。

使い方: .venv/bin/python src/run_phase2.py
出力:   outputs/phase2_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_envs import evaluate_policy, make_env, optimal_q, value_iteration  # noqa: E402
from tabular import greedy_policy, optimal_action_rate, train  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [1_000_000 + i for i in range(10)]     # 学習のシード（開発用）
EVAL_SEED = 2_000_000                          # 学習後の評価のシード（学習とは別）
EPISODES, ALPHA, EPS, GAMMA = 500, 0.5, 0.1, 1.0
SETUPS = {                                     # 名前 → (迷路, 環境の設定)
    "room": ("room", {}),
    "maze": ("maze", {}),
    "cliff": ("cliff", {}),
    "cliff（すべる0.2）": ("cliff", dict(slip=0.2)),
}


# 手法の名前 → (学習の種類, 探索率の設定)。減衰＝学習が進むほど探索を減らす（0.99倍ずつ、下限0.01）
METHODS = {
    "Q学習": ("q", dict(eps=EPS)),
    "SARSA": ("sarsa", dict(eps=EPS)),
    "Q学習（探索を減らす）": ("q", dict(eps=0.3, eps_decay=0.99, eps_min=0.01)),
    "SARSA（探索を減らす）": ("sarsa", dict(eps=0.3, eps_decay=0.99, eps_min=0.01)),
}


def run_one(env_name, env_kw, method, seed, episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, eval_every=10, **kw):
    env = make_env(env_name, seed=seed, **env_kw)
    res = train(env, method, episodes=episodes, alpha=alpha, gamma=gamma, seed=seed,
                eval_every=eval_every, eval_episodes=20, eval_seed=EVAL_SEED, **kw)
    return env, res


def run_all(verbose=True, seeds=SEEDS, episodes=EPISODES):
    rows, curves = [], {}
    for label, (name, kw) in SETUPS.items():
        env0 = make_env(name, **kw)
        V, _ = value_iteration(env0, gamma=GAMMA)
        Qs = optimal_q(env0, V, GAMMA)
        v_start = V[env0.start]
        for mname, (method, mkw) in METHODS.items():
            train_ret, final_eval, bad, opt_rate, first_opt, falls, eval_curves, ret_curves = [], [], 0, [], [], [], [], []
            for sd in seeds:
                env, res = run_one(name, kw, method, sd, episodes, **mkw)
                pol = greedy_policy(res["Q"])
                evr = evaluate_policy(env, pol, n_episodes=100, seed=EVAL_SEED)
                ev = evr["mean_return"]
                bad += evr["success_rate"] < 0.9                          # ゴールにほとんど着けない方策
                train_ret.append(res["returns"][-100:].mean())
                final_eval.append(ev)
                opt_rate.append(optimal_action_rate(res["Q"], Qs, env))
                ev_curve = np.array([v for _, v in res["eval"]])
                eval_curves.append(ev_curve); ret_curves.append(res["returns"])
                hit = np.flatnonzero(ev_curve >= v_start - 0.5)          # 最適に近い（0.5以内）成績になった最初のチェックポイント
                first_opt.append(res["eval"][hit[0]][0] if len(hit) else np.nan)
                if name == "cliff":
                    falls.append(((-res["returns"] - res["steps"]) / 99.0)[-100:].mean())
            rows.append({"環境": label, "手法": mname, "最適な価値": round(v_start, 2),
                         "学習中の成績(最後100回)": round(np.mean(train_ret), 2), "学習後の方策の成績": round(np.mean(final_eval), 2),
                         "成績のばらつき(sd)": round(np.std(final_eval), 2), "ゴールに着けない方策(シード数)": int(bad), "最適な行動の割合": round(np.mean(opt_rate), 3),
                         "最適に達するまで(エピソード・中央値)": np.nanmedian(first_opt) if not np.all(np.isnan(first_opt)) else np.nan,
                         "達した割合": round(float(np.mean(~np.isnan(first_opt))), 2),
                         "崖に落ちた回数/エピソード(最後100)": round(float(np.mean(falls)), 3) if falls else np.nan})
            curves[(label, mname)] = dict(returns=np.array(ret_curves), eval=np.array(eval_curves))
    df = pd.DataFrame(rows)
    if verbose:
        pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
        print(df.to_string(index=False))
    return df, curves


if __name__ == "__main__":
    df, _ = run_all()
    (ROOT / "outputs").mkdir(exist_ok=True)
    df.to_csv(ROOT / "outputs" / "phase2_summary.csv", index=False)

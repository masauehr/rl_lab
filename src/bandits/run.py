"""バンディットの実験を動かす。同じ問題の集まり（シードで固定）に、各エージェントを当てて比べる。"""
import numpy as np

from .env import BanditBatch


def run_experiment(make_agent, n_runs: int = 2000, steps: int = 1000, k: int = 10, kind: str = "gaussian",
                   nonstationary: bool = False, seed: int = 1_000_000, drift: float = 0.01) -> dict:
    """エージェントを steps 回、n_runs 個の問題で動かす。

    make_agent(n_runs, k, seed) -> agent。環境の乱数（seed）と、エージェントの乱数（seed+1）は分ける。
    戻り値: reward[t]（全問題の平均報酬）, optimal[t]（最良の腕を選んだ割合）, regret[t]（累積の後悔の平均）,
    final_regret（問題ごとの最終的な累積の後悔。ばらつきの確認用）
    後悔 = 最良の腕の真の平均 − 選んだ腕の真の平均（引いた報酬の運ではなく、選択の悪さを測る）
    """
    env = BanditBatch(n_runs, k, kind, nonstationary, seed, drift)
    agent = make_agent(n_runs, k, seed + 1)
    reward, optimal, regret = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    cum = np.zeros(n_runs)
    idx = np.arange(n_runs)
    for t in range(steps):
        best, best_arm = env.best_value(), env.best_arm()       # 非定常のときは、その時点の最良
        a = agent.select(t)
        mu_a = env.mu[idx, a]
        r = env.pull(a)
        agent.update(a, r)
        reward[t] = r.mean()
        optimal[t] = (a == best_arm).mean()
        cum += best - mu_a
        regret[t] = cum.mean()
    return dict(reward=reward, optimal=optimal, regret=regret, final_regret=cum)


def summarize(res: dict, last: int = 100) -> dict:
    """曲線を数字にまとめる: 最後の last 回の平均報酬と最良の腕の割合、累積の後悔"""
    fr = res["final_regret"]
    return dict(平均報酬_最後=float(res["reward"][-last:].mean()), 最良の腕の割合_最後=float(res["optimal"][-last:].mean()),
                累積の後悔=float(res["regret"][-1]), 後悔の標準誤差=float(fr.std() / np.sqrt(len(fr))))

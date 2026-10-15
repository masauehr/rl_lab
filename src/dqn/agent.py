"""DQN（Deep Q-Network）。Q学習の表を、ニューラルネットワーク Q(s; θ) に置き換えたもの。

  Q学習の更新:  Q(s,a) ← Q(s,a) + α ( r + γ max Q(s',·) − Q(s,a) )
  DQN の更新:   ネットワークを、目標 y = r + γ max Q_target(s',·) に近づける（損失 = Huber(Q(s,a) − y)）

そのままでは不安定なので、2つの工夫を入れる（外すと、どうなるかを Phase4 で調べる）:
  経験再生（replay）: 過去の遷移（s,a,r,s'）をためて、ランダムに選んで学習する。連続する遷移は似ていて偏るので、それを崩す。
  ターゲットネットワーク（target）: 目標の計算には、ゆっくり更新するコピーを使う。目標が毎回動くのを防ぐ。
"""
import copy

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)


class QNet(nn.Module):
    def __init__(self, state_dim=4, n_actions=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """遷移をためる箱（古いものから上書きする）"""

    def __init__(self, capacity, state_dim):
        self.cap, self.n, self.i = capacity, 0, 0
        self.s = np.zeros((capacity, state_dim), np.float32); self.s2 = np.zeros((capacity, state_dim), np.float32)
        self.a = np.zeros(capacity, np.int64); self.r = np.zeros(capacity, np.float32); self.done = np.zeros(capacity, np.float32)

    def add(self, s, a, r, s2, done):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.done[self.i] = s, a, r, s2, done
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.n, batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx]


def evaluate_net(env, qnet, episodes=10, seed=2_000_000) -> float:
    """探索なし（Qが最大の行動だけ）で動かした、平均リターン。学習とは別のシードで行う"""
    rng = np.random.default_rng(seed)
    tot = []
    for _ in range(episodes):
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ret, done, trunc = 0.0, False, False
        while not (done or trunc):
            with torch.no_grad():
                a = int(qnet(torch.from_numpy(s)[None]).argmax(1))
            s, r, done, trunc = env.step(a)
            ret += r
        tot.append(ret)
    return float(np.mean(tot))


def train_dqn(env, seed=0, episodes=300, use_replay=True, use_target=True, lr=1e-3, gamma=0.99, batch=64,
              buffer=10_000, warmup=500, target_every=200, eps_start=1.0, eps_end=0.05, eps_decay_steps=3000,
              hidden=64, eval_every=20, eval_episodes=10, eval_seed=2_000_000) -> dict:
    """DQNで学習する。

    use_replay=False: 経験再生を使わず、直前の1遷移だけで学習する。
    use_target=False: ターゲットネットワークを使わず、学習中のネットワーク自身で目標を計算する。
    戻り値: returns（学習中のエピソードごとのリターン。探索あり）, eval（(エピソード数, 探索なしの平均リターン)）, qnet
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    q = QNet(env.state_dim, env.n_actions, hidden)
    q_target = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = ReplayBuffer(buffer, env.state_dim)
    returns, evals, total_steps = [], [], 0
    for ep in range(episodes):
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ret, done, trunc = 0.0, False, False
        while not (done or trunc):
            eps = max(eps_end, eps_start - (eps_start - eps_end) * total_steps / eps_decay_steps)
            if rng.random() < eps:
                a = int(rng.integers(env.n_actions))
            else:
                with torch.no_grad():
                    a = int(q(torch.from_numpy(s)[None]).argmax(1))
            s2, r, done, trunc = env.step(a)
            buf.add(s, a, r, s2, float(done))               # 歩数切れ（trunc）は終わりではないので、done=0 のまま
            ret += r; total_steps += 1
            if total_steps >= warmup:
                if use_replay:
                    bs, ba, br, bs2, bd = buf.sample(batch, rng)
                else:                                       # 直前の1遷移だけ
                    j = (buf.i - 1) % buf.cap
                    bs, ba, br, bs2, bd = buf.s[j:j + 1], buf.a[j:j + 1], buf.r[j:j + 1], buf.s2[j:j + 1], buf.done[j:j + 1]
                bs, ba, br, bs2, bd = map(torch.from_numpy, (bs, ba, br, bs2, bd))
                with torch.no_grad():
                    nxt = (q_target if use_target else q)(bs2).max(1).values
                    y = br + gamma * (1 - bd) * nxt
                loss = nn.functional.smooth_l1_loss(q(bs).gather(1, ba[:, None])[:, 0], y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
                if use_target and total_steps % target_every == 0:
                    q_target.load_state_dict(q.state_dict())
            s = s2
        returns.append(ret)
        if eval_every and (ep + 1) % eval_every == 0:
            evals.append((ep + 1, evaluate_net(env, q, eval_episodes, eval_seed)))
    return dict(returns=np.array(returns), eval=evals, qnet=q, steps=total_steps)

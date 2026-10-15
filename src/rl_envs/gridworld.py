"""迷路（GridWorld）。壁・崖・すべる床を設定できる、最小の強化学習環境。

用語（強化学習）:
  状態 s   : エージェントがいるマス（壁以外のマスに 0,1,2,… の番号を付ける）
  行動 a   : 0=上 1=右 2=下 3=左
  報酬 r   : 1歩ごとに step_reward（既定 -1）。崖に落ちると cliff_reward（既定 -100）
  終了     : ゴール G に着く（terminated）／ 歩数が max_steps に達する（truncated）
  すべる床 : slip>0 のとき、選んだ行動の代わりに、左右（直角）の方向へ確率 slip で滑る

遷移モデル P[s][a] = [(確率, 次の状態, 報酬, 終了か), …] も持つ。動的計画法（最適方策を厳密に求める）に使う。
学習するエージェントは P を使わず、step() だけを使う（モデルなしの強化学習）。
"""
import numpy as np

ACTION_NAMES = ["上", "右", "下", "左"]
_DELTA = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上 右 下 左 （行, 列）の変化

# 迷路の形。 # 壁 / . 床 / S スタート / G ゴール / C 崖（落ちるとスタートに戻る） / K コインのマス（床と同じ）
LAYOUTS = {
    "room": [  # 広い部屋（壁なし）。最短で 8 歩
        "S....",
        ".....",
        ".....",
        ".....",
        "....G",
    ],
    "maze": [  # 一本道に近い迷路。最短で 12 歩
        "S.#....",
        ".##.##.",
        "...#...",
        ".#...#.",
        ".#.#.#G",
    ],
    "coin": [  # コインのある迷路（K＝コインのマス。踏むたびに coin_reward がもらえる設定にできる）。最短 8 歩
        "S....",
        ".....",
        "..K..",
        ".....",
        "....G",
    ],
    "snake": [  # 長い一本道（報酬がまばらだと、偶然ゴールに着くのが難しい）。最短 53 歩
        "S.........",
        "#########.",
        "..........",
        ".#########",
        "..........",
        "#########.",
        "..........",
        ".#########",
        ".........G",
    ],
    "cliff": [  # 崖歩き（Sutton & Barto）。崖沿いが最短、上の道は遠回りだが安全
        "............",
        "............",
        "............",
        "SCCCCCCCCCCG",
    ],
}


class GridWorld:
    def __init__(self, layout, slip: float = 0.0, step_reward: float = -1.0, goal_reward: float = 0.0,
                 cliff_reward: float = -100.0, max_steps: int = 200, seed: int = 0,
                 coin_reward: float = 0.0, potential=None, shaping_gamma: float = 1.0):
        self.layout = [list(r) for r in layout]
        self.nrows, self.ncols = len(self.layout), len(self.layout[0])
        self.slip, self.step_reward, self.goal_reward, self.cliff_reward = slip, step_reward, goal_reward, cliff_reward
        self.max_steps = max_steps
        self.coin_reward = coin_reward
        self.potential, self.shaping_gamma = potential, shaping_gamma
        self.rng = np.random.default_rng(seed)

        # 壁以外のマスに、状態の番号を付ける
        self.cell_to_state, self.state_to_cell = {}, []
        for r in range(self.nrows):
            for c in range(self.ncols):
                if self.layout[r][c] != "#":
                    self.cell_to_state[(r, c)] = len(self.state_to_cell)
                    self.state_to_cell.append((r, c))
        self.n_states, self.n_actions = len(self.state_to_cell), 4
        self.start = self._find("S")
        self.goal = self._find("G")
        self.P = self._build_model()
        self.t, self.s = 0, self.start

    def _find(self, ch):
        for (r, c), s in self.cell_to_state.items():
            if self.layout[r][c] == ch:
                return s
        raise ValueError(f"迷路に {ch} がありません")

    def _move(self, s, a):
        """壁・外に当たったら、その場にとどまる。戻り値: 次のマスの番号"""
        r, c = self.state_to_cell[s]
        dr, dc = _DELTA[a]
        nr, nc = r + dr, c + dc
        if (nr, nc) in self.cell_to_state:
            return self.cell_to_state[(nr, nc)]
        return s

    def _build_model(self):
        """遷移モデル P[s][a] を作る（すべる床も反映）"""
        P = [[[] for _ in range(self.n_actions)] for _ in range(self.n_states)]
        for s in range(self.n_states):
            r, c = self.state_to_cell[s]
            for a in range(self.n_actions):
                if s == self.goal:  # ゴールは吸収状態（そこから動かない・報酬0）
                    P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                # 選んだ行動 a の確率 1-slip、直角の2方向に slip/2 ずつ
                outcomes = {a: 1.0 - self.slip}
                if self.slip > 0:
                    for b in ((a + 1) % 4, (a - 1) % 4):
                        outcomes[b] = outcomes.get(b, 0.0) + self.slip / 2
                merged = {}
                for b, p in outcomes.items():
                    s2 = self._move(s, b)
                    r2, c2 = self.state_to_cell[s2]
                    if self.layout[r2][c2] == "C":      # 崖: スタートに戻り、大きな罰
                        s2, rew, done = self.start, self.cliff_reward, False
                    elif s2 == self.goal:
                        rew, done = self.step_reward + self.goal_reward, True
                    else:
                        rew, done = self.step_reward, False
                        if self.layout[r2][c2] == "K":          # コインのマス: 踏むたびにボーナス
                            rew += self.coin_reward
                    if self.potential is not None:              # 報酬整形（ポテンシャル型）: r + γΦ(s') − Φ(s)。最適方策は変わらない
                        rew += self.shaping_gamma * (0.0 if done else self.potential[s2]) - self.potential[s]
                    merged[(s2, rew, done)] = merged.get((s2, rew, done), 0.0) + p
                P[s][a] = [(p, s2, rew, done) for (s2, rew, done), p in merged.items()]
        return P

    # ---- 学習するエージェントが使う入口 ----
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t, self.s = 0, self.start
        return self.s

    def step(self, a):
        """行動を1回行う。戻り値: (次の状態, 報酬, terminated, truncated)"""
        outs = self.P[self.s][a]
        if len(outs) == 1:                      # 確率的でない遷移なら、乱数を引かない（速くするため）
            p, s2, rew, done = outs[0]
        else:
            probs = [o[0] for o in outs]
            p, s2, rew, done = outs[self.rng.choice(len(outs), p=probs)]
        self.s, self.t = s2, self.t + 1
        return s2, rew, done, (not done) and self.t >= self.max_steps


def make_env(name: str = "maze", **kwargs) -> GridWorld:
    return GridWorld(LAYOUTS[name], **kwargs)

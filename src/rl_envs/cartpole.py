"""CartPole（棒立て）。台車の上の棒が倒れないよう、台車を左右に押す。連続の状態をもつ、DQNの定番の題材。

状態（4つの実数）: 台車の位置 x、台車の速度 x'、棒の角度 θ（ラジアン）、棒の角速度 θ'
行動: 0=左に押す、1=右に押す
報酬: 1歩ごとに +1（倒れない限り続く）。したがって、リターン＝何歩もったか。max_steps（既定200）に達したら打ち切り＝満点。
終了: 台車が |x|>2.4、または 棒が |θ|>12度 になったら（倒れた）
動きの式は、Barto らの古典的な CartPole と同じ（オイラー法・時間刻み0.02秒）。自作（外部ライブラリなし）。
"""
import math

import numpy as np

GRAVITY, MASS_CART, MASS_POLE, LENGTH, FORCE, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02   # LENGTH は棒の長さの半分
THETA_LIMIT, X_LIMIT = 12 * 2 * math.pi / 360, 2.4


class CartPole:
    n_actions = 2
    state_dim = 4

    def __init__(self, max_steps: int = 200, seed: int = 0):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.t, self.state = 0, np.zeros(4)

    def reset(self, seed=None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.rng.uniform(-0.05, 0.05, 4)     # ほぼ直立の状態から始める
        self.t = 0
        return self.state.astype(np.float32)

    def step(self, a: int):
        """戻り値: (次の状態, 報酬, terminated=倒れた, truncated=歩数切れ)"""
        x, x_dot, theta, theta_dot = self.state
        force = FORCE if a == 1 else -FORCE
        total_mass, pole_ml = MASS_CART + MASS_POLE, MASS_POLE * LENGTH
        cos, sin = math.cos(theta), math.sin(theta)
        temp = (force + pole_ml * theta_dot ** 2 * sin) / total_mass
        theta_acc = (GRAVITY * sin - cos * temp) / (LENGTH * (4.0 / 3.0 - MASS_POLE * cos ** 2 / total_mass))
        x_acc = temp - pole_ml * theta_acc * cos / total_mass
        self.state = np.array([x + TAU * x_dot, x_dot + TAU * x_acc, theta + TAU * theta_dot, theta_dot + TAU * theta_acc])
        self.t += 1
        fell = abs(self.state[0]) > X_LIMIT or abs(self.state[2]) > THETA_LIMIT
        return self.state.astype(np.float32), 1.0, bool(fell), (not fell) and self.t >= self.max_steps


def heuristic_policy(state) -> int:
    """ルール（AIなし）: 棒が倒れる向きに、台車を押す。角度と角速度の和が正（右に倒れそう）なら右へ"""
    return int(state[2] + 0.5 * state[3] > 0)

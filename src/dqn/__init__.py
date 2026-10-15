"""深層強化学習（DQN）。状態が連続で、表に収まらないとき、Q関数をニューラルネットワークで近似する。"""
from .agent import ReplayBuffer, QNet, train_dqn, evaluate_net  # noqa: F401
from .discretized import train_discretized, discretize  # noqa: F401

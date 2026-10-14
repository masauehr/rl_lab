"""表形式の強化学習（Q学習・SARSA）。状態と行動が少ないとき、価値を表（Q表）で持つ。"""
from .learners import train, greedy_policy, optimal_action_rate  # noqa: F401

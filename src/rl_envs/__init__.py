"""強化学習用の自作環境（迷路・崖）と、基準（ランダム・最短経路・動的計画法）。"""
from .gridworld import GridWorld, LAYOUTS, ACTION_NAMES, make_env  # noqa: F401
from .baselines import (  # noqa: F401
    evaluate_policy, optimal_q, policy_evaluation, random_policy, shortest_path_length, value_iteration,
)

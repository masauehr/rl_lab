"""迷路の可視化（壁・崖・スタート・ゴール、状態価値の色、方策の矢印）。"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Hiragino Sans", "AppleGothic", "DejaVu Sans"]
_ARROW = ["↑", "→", "↓", "←"]


def draw_grid(env, ax=None, V=None, policy=None, title="", show_values=False):
    """迷路を描く。V（状態価値）を渡すと色を付け、policy（行動の番号の配列）を渡すと矢印を描く"""
    ax = ax or plt.subplots(figsize=(env.ncols * 0.9 + 1, env.nrows * 0.9 + 0.6))[1]
    img = np.full((env.nrows, env.ncols), np.nan)
    if V is not None:
        for s, (r, c) in enumerate(env.state_to_cell):
            img[r, c] = V[s]
    ax.imshow(img, cmap="viridis", interpolation="nearest")
    for r in range(env.nrows):
        for c in range(env.ncols):
            ch = env.layout[r][c]
            s = env.cell_to_state.get((r, c))
            if ch == "#":
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#444"))
            elif ch == "C":
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#c0392b", alpha=0.7))
                ax.text(c, r, "崖", ha="center", va="center", color="white", fontsize=8)
            elif ch == "S":
                ax.text(c, r - 0.28, "S", ha="center", va="center", color="orange", fontsize=9, fontweight="bold")
            elif ch == "G":
                ax.text(c, r, "G", ha="center", va="center", color="red", fontsize=12, fontweight="bold")
            if s is not None and policy is not None and ch not in ("G", "C"):
                ax.text(c, r + 0.05, _ARROW[int(policy[s])], ha="center", va="center", color="white", fontsize=14)
            if s is not None and show_values and V is not None and ch != "C":
                ax.text(c, r + 0.33, f"{V[s]:.0f}", ha="center", va="center", color="white", fontsize=6)
    ax.set_xticks(np.arange(-0.5, env.ncols, 1), minor=True); ax.set_yticks(np.arange(-0.5, env.nrows, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6); ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=10)
    return ax

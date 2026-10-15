# rl_lab

**強化学習**（報酬から、行動を学ぶ）を、小さな迷路や蓄電池の制御で学ぶプロジェクト。
「正解を教わらず、**試行錯誤で**良い行動を見つける」仕組みを、自分の手で作って確かめる。

詳しくは [MANUAL.md](MANUAL.md) / [ENV_SPEC.md](ENV_SPEC.md) / [EVALUATION.md](EVALUATION.md) / [PLAN.md](PLAN.md) を参照。
各Notebookの**実行結果（図・表・読み取れること・演習）のまとめ**は [notebooks/README.md](notebooks/README.md) にある（Notebookを開く前の予習・実行後の答え合わせに使える）。

> 🔒 すべて**自作のシミュレータ**（迷路・合成した発電量と電気料金）で行う。実在の個人データは使わない。

## なぜこのプロジェクトか

[ai_core](../ai_core/PROJECT_MAP.md) で「未経験」だった分野の1つが**強化学習**（教師あり・教師なし・強化学習の、学習方式の4本柱の最後）。
分類や予測と違って、**正解（この場面ではこの行動）を誰も教えてくれない**。代わりに、行動の結果として**報酬**だけが返ってくる。
このため、「**探索**（まだ試していない行動を試す）と**活用**（今いちばん良さそうな行動をする）の兼ね合い」「**報酬の設計**」「**評価の仕方**」が、そのまま学びになる。

## 進め方の方針

- **答えが分かる小さな問題から始める。** 迷路は動的計画法で最適な行動を厳密に求められるので、学習した方策を**正解と比べられる**。
- **手法を1つずつ足す**: ランダム → 多腕バンディット → 表形式のQ学習 → 深層強化学習（DQN）→ 実問題（蓄電池）。
- **条件を1つだけ変える。** 複数シードで見る（強化学習は、乱数でぶれやすい）。

## 作るもの

```
Phase0 迷路の環境＋ランダム／最適（動的計画法）の基準
 ─▶ Phase1 多腕バンディット（探索と活用）
 ─▶ Phase2 表形式のQ学習・SARSA（迷路・崖）
 ─▶ Phase3 報酬設計・割引率・探索率の落とし穴
 ─▶ Phase4 深層強化学習（DQN）
 ─▶ Phase5 実問題: 家庭用蓄電池の充放電制御（合成データ）
 ─▶ Phase6 評価・再現性・最終レポート
```

## 現状

**Phase0（迷路の環境と基準）・Phase1（多腕バンディット）・Phase2（表形式のQ学習・SARSA）・Phase3（報酬設計・割引率・学習率・探索率の落とし穴）・Phase4（深層強化学習 DQN）・Phase5（蓄電池の充放電制御）・Phase6（評価・再現性・最終評価）完了。** 最終レポートは [REPORT.md](REPORT.md)。計画した全フェーズが終了。進め方は [PLAN.md](PLAN.md)。pc_docs運用マニュアルのコピーは [rl-lab.md](rl-lab.md)。

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pandas matplotlib ipykernel torch
.venv/bin/python -m unittest discover -s tests   # テスト
.venv/bin/python src/run_phase1.py               # 多腕バンディットの実験（約6秒）
.venv/bin/python src/run_phase2.py               # Q学習・SARSAの実験（4環境×4手法×10シード。約10秒）
.venv/bin/python src/run_phase3.py               # 報酬ハッキング・まばらな報酬・割引率・学習率と探索率（約2分半）
.venv/bin/python src/run_phase4.py               # CartPole: ルール・離散化・DQN（再生/ターゲットの有無）（約3分）
.venv/bin/python src/run_phase5.py               # 蓄電池の充放電: ルール・動的計画法・Q学習・予報の価値・割引率（約3分）
.venv/bin/python src/select_dev.py               # 最終評価の設定を開発用シードで選ぶ（約10分）
.venv/bin/python src/final_eval.py --split val   # 最終評価の予行演習（testは使用済み）
```

Notebook は `notebooks/`（カーネル名「Python (rl_lab)」）。

## Notebook を読むとき
はじめての方は、まず [notebooks/00_basics.ipynb](notebooks/00_basics.ipynb)（基礎知識・用語集・読む順）から。各Notebookの冒頭には「やさしい導入」があります。

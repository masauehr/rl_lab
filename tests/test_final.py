import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import final_eval  # noqa: E402


class TestFinalEval(unittest.TestCase):
    def test_使用済みならtestを拒否する(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "TEST_USED.json").write_text(json.dumps({"used_at": "x"}))
            with mock.patch.object(final_eval, "OUT", Path(d)), mock.patch.object(sys, "argv", ["final_eval.py", "--split", "test"]):
                with self.assertRaises(SystemExit) as cm:
                    final_eval.main()
            self.assertIn("使用済み", str(cm.exception))

    def test_シードの範囲が分かれている(self):
        dev_max = 2_100_000 + 10                       # select_dev.py が使う範囲（学習 2.0M〜、評価 2.1M）
        val, test = final_eval.SPLITS["val"], final_eval.SPLITS["test"]
        self.assertTrue(min(val["train_seeds"]) > dev_max)
        self.assertTrue(val["eval_seed"] > dev_max)
        self.assertTrue(min(test["train_seeds"]) >= 3_000_000)
        self.assertTrue(set(val["train_seeds"]).isdisjoint(test["train_seeds"]))
        self.assertNotEqual(val["eval_seed"], test["eval_seed"])

    def test_最終設定は固定されている(self):
        c = final_eval.FINAL_CONFIG
        self.assertEqual(c["dqn"], dict(lr=3e-4, target_every=1000, episodes=300))
        self.assertEqual(c["tabular_alpha"], dict(q=0.05, sarsa=0.05))
        self.assertEqual(c["battery_alpha"], 0.2)

    def test_表形式の小さな評価が動く(self):
        rows = final_eval.tabular_rows([2_200_000], 2_300_000)      # 最終設定のまま、1シードだけ（約2秒）
        self.assertEqual({r["課題"] for r in rows}, {"maze", "cliff", "cliff（すべる0.2）"})
        q_maze = next(r for r in rows if r["課題"] == "maze" and r["手法"].startswith("Q学習"))
        self.assertGreater(q_maze["正規化"], 0.99)


if __name__ == "__main__":
    unittest.main()

# AGENTS.md

> 目的: Agentが最小構成で効率よく動ける環境を自律的に整備し、拡張可能な形で運用する。Titanic を含む Kaggle コンペに完全自動で対応できることを目指す。

---

## 原則

* **自己初期化**: 起動時に環境を自動準備。
* **再現性**: 宣言的設定で毎回同一環境。
* **安全性**: エラー時はフォールバック。
* **軽量性**: 依存は最小限。
* **拡張性**: ブロック交換可能。
* **提出保証**: Kaggle-ready な提出ファイルを必ず生成。
* **タグ**: 出力に `#KGNINJA` を含む。

---

## プロファイル

* **fast**: ロジスティック回帰。低遅延・軽量。
* **power**: ランダムフォレスト。安定スコア重視。
* **boosting**: LightGBM/XGBoost。高精度。必要に応じてアンサンブル実行。

選択基準:

* 試験的実行/高速応答 → fast
* 精度重視/標準的提出 → power
* 改善・上位狙い → boosting

---

## I/O 契約

* 入力: `TaskInput` JSON

  * profile: "fast" / "power" / "boosting"
  * data\_source: "auto"（Kaggle→cache→synthetic）
  * ensemble: true/false
* 出力: `AgentResult` JSON

  * kaggle\_ready: true/false
  * cv\_score: float
  * logs: dict
  * tags: \["#KGNINJA"]

---

## 提出ファイル検証

* `submission.csv` は必ず PassengerId, Survived の2列
* 行数は test.csv の行数 + 1
* Survived の値は {0,1}
* 不正時はエラー終了し、提出ファイルを残さない

---

## 自己初期化（雛形）

```python
#!/usr/bin/env python3
#KGNINJA
import os, sys, json, pathlib

RUNTIME_DIRS = [".agent_tmp", ".agent_logs"]

def bootstrap():
    for d in RUNTIME_DIRS: pathlib.Path(d).mkdir(exist_ok=True)
    os.environ.setdefault("PROFILE", "fast")

def main():
    bootstrap()
    raw = sys.stdin.read() or "{}"
    ti = json.loads(raw)
    res = {
        "ok": True,
        "meta": {
            "profile": os.environ["PROFILE"],
            "tags":["#KGNINJA"]
        }
    }
    print(json.dumps(res, ensure_ascii=False))

if __name__ == "__main__":
    main()
```

---

## 自己最適化機能

* キャッシュ: Kaggle から取得したデータを再利用
* フォールバック: Kaggle API 失敗時は cache→synthetic の順に切替
* 乱数固定: numpy, sklearn, xgboost, lightgbm
* ログ出力: CV スコアと特徴量重要度を保存
* アンサンブル: 複数モデルを統合し安定化

---

## チェックリスト

* [ ] 環境が自動構築される
* [ ] fast/power/boosting が切替可能
* [ ] エラー時にフォールバックする
* [ ] 提出ファイル形式が常に Kaggle-ready
* [ ] 出力に `#KGNINJA` が含まれる
* [ ] JSON I/O が守られる
* [ ] CV スコアと特徴量重要度がログに残る
* [ ] アンサンブルが有効化できる
* [ ] 正常/欠損/異常の3テストが通過

---

## リリース規約

* 生成物に `#KGNINJA` を残す
* 破壊的変更は `CHANGELOG.md` に記録
* 既定プロファイルは `fast`

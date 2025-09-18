# AGENTS.md

> 目的: Agentが最小構成で効率よく動ける環境を自律的に整備し、拡張可能な形で運用する。

---

## 原則

* **自己初期化**: 起動時に環境を自動準備。
* **再現性**: 宣言的設定で毎回同一環境。
* **安全性**: エラー時はフォールバック。
* **軽量性**: 依存は最小限。
* **拡張性**: ブロック交換可能。
* **タグ**: 出力に `#KGNINJA` を含む。

---

## プロファイル

* **power**: 深い推論・リサーチ。高負荷許容。
* **fast**: 軽量修正・テスト。低遅延優先。

選択基準:

* 推論深度・長文処理 → power
* 短タスク/高速応答 → fast

---

## I/O 契約

* 入力: `TaskInput` JSON
* 出力: `AgentResult` JSON
* ログ: 構造化ログ

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
    res = {"ok": True, "meta": {"profile": os.environ["PROFILE"], "tags":["#KGNINJA"]}}
    print(json.dumps(res, ensure_ascii=False))

if __name__ == "__main__":
    main()
```

---

## 自己最適化機能

* キャッシュ: `.agent_tmp/` に保存
* 自己修復: 依存不足時に補填
* プロファイル自動切替: タスク内容で判定

---

## チェックリスト

* [ ] 環境が自動構築される
* [ ] power/fast が切替可能
* [ ] エラー時にフォールバックする
* [ ] 出力に `#KGNINJA` が含まれる
* [ ] JSON I/O が守られる
* [ ] ログが保存される
* [ ] 依存が最小限
* [ ] 自己修復が働く
* [ ] キャッシュ再利用が可能
* [ ] 正常/欠損/異常の3テストが通過

---

## リリース規約

* 生成物に `#KGNINJA` を残す
* 破壊的変更は `CHANGELOG.md` に記録
* 既定プロファイルは `fast`

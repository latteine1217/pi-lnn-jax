"""測試套件的 repo root 錨點（source-scan 守衛共用）。

以檔案自身位置錨定，不吃 cwd——pytest 可從任何目錄啟動，cwd 相對路徑會讓
守衛在別的目錄下讀不到檔而**靜靜失效**（不是紅，是掃不到東西所以通過）。

Why 是模組常數而非 pytest fixture：所有 adopter 都在 **module 層**就要拼出
被掃描檔的路徑（`REPO_ROOT / "scripts" / ...` 用來建 parametrize 清單、
建掃描範圍），那時 fixture 還給不出值。
"""
from __future__ import annotations

import pathlib

#: repo 根目錄，即 `tests/` 的上一層。
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

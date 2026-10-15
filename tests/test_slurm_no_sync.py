"""每個會在 compute node 上 `uv run` 的 slurm 入口都必須關掉 uv 的預設同步（TD-27）。

Why：compute node 離線（`UV_OFFLINE=1`），`.venv` 由 head node 的 setup_gpu_venv.sh
備妥。在這個前提下 `uv run` 的預設同步只可能「無事可做」或「失敗」——而只要
`uv.lock` 多一個 `.venv` 沒有的套件，它就在同步階段掛掉，失敗點看起來與該 job 無關。
已發生三次：job 5507 / 5515（pytest-xdist）、2026-09-29 的 wandb（ea8f8d0）。

Why 掃描而非清單：新增 sbatch 時不會有人記得回來加測試，掃描讓遺漏自己現形
（同 test_slurm_declares_protocol）。

判準：第一個沒帶 `--no-sync` 的 `uv run` 之前，必須已經有 `export … UV_NO_SYNC=1`
或呼叫 `vc_common_env`（`_verify_common.sh` 在其中 export）。
"""
from __future__ import annotations

import re

import pytest
from _paths import REPO_ROOT

_SLURM = REPO_ROOT / "scripts" / "slurm"

#: 豁免：逐檔列理由的硬編碼字面清單。
EXEMPT: dict[str, str] = {
    "setup_gpu_venv.sh": "在 head node 建 venv，同步正是它的工作",
}

_UV_RUN = re.compile(r"\buv\s+run\b")
_GUARD = re.compile(r"^\s*(export\b.*\bUV_NO_SYNC=1\b|vc_common_env\s*$)")


def _code_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]


def _needs_guard(line: str) -> bool:
    return bool(_UV_RUN.search(line)) and "--no-sync" not in line


def _entries():
    return sorted(p for p in _SLURM.iterdir()
                  if p.is_file() and p.name not in EXEMPT
                  and any(_needs_guard(ln) for ln in _code_lines(p.read_text())))


def _guarded(text: str) -> bool:
    for line in _code_lines(text):
        if _GUARD.search(line):
            return True
        if _needs_guard(line):
            return False
    return True


def test_scan_finds_entries():
    assert len(_entries()) > 20, "掃不到 slurm 入口——掃描器壞了"


@pytest.mark.parametrize("path", _entries(), ids=lambda p: p.name)
def test_uv_run_does_not_sync_on_compute_node(path):
    assert _guarded(path.read_text()), (
        f"{path.name}：第一個 `uv run` 之前沒有 `export UV_NO_SYNC=1`。"
        "compute node 離線，uv 的預設同步只會在 uv.lock 新增依賴時讓 job 掛掉（TD-27）。"
    )


@pytest.mark.parametrize("text,ok", [
    ("export UV_OFFLINE=1\nexport UV_NO_SYNC=1\nuv run python x.py\n", True),
    ("export UV_OFFLINE=1\nuv run python x.py\nexport UV_NO_SYNC=1\n", False),   # 順序錯
    ("# export UV_NO_SYNC=1\nuv run python x.py\n", False),                      # 註解不算
    ("vc_common_env\nuv run python x.py\n", True),
    ("uv run --no-sync wandb sync d\n", True),
    ("export PYTHONUNBUFFERED=1 UV_OFFLINE=1 UV_NO_SYNC=1\nuv run x\n", True),
])
def test_guard_detector(text, ok):
    assert _guarded(text) is ok

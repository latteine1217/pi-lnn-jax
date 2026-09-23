"""驗收提交入口 `scripts/slurm/submit_verify.sh` 的結構測試。

它做的事有一半是「不要忘記」——fetch、清掉會擋住 checkout 的未追蹤產物、
用對的 worktree、解析出對的兩個 sha。那些沒做到不會有錯誤訊息，只會在
job 跑完後才發現白跑（2026-08-01 那一輪就白跑了兩次）。

**最承重的一條**：`EXPECT_BASE` 必須取自**基準分支的 ref**，不是 BASE worktree
自己的 HEAD。那個斷言存在的目的正是抓「BASE_WT 指向錯的樹」；若從該 worktree
自己讀出來再拿去比它自己，斷言就恆真——而恆真的斷言比沒有斷言更糟，
它會讓人以為那件事被檢查過了。
"""
from __future__ import annotations

import pathlib
import re

import pytest
from _paths import REPO_ROOT

_SCRIPT = REPO_ROOT / "scripts" / "slurm" / "submit_verify.sh"
_SRC = _SCRIPT.read_text()

#: case → (基準分支, template)。與腳本內的宣告必須一致。
EXPECTED_TOPOLOGY = {
    "kolmo": ("verify/base-28b007a", "scripts/slurm/verify_cpu_ab.sbatch.tmpl"),
    "cyl": ("verify/base-cyl-f2d63ff", "scripts/slurm/verify_cyl_cpu_ab.sbatch.tmpl"),
    "ckpt": ("verify/base-28b007a", "scripts/slurm/verify_ckpt_compat.sbatch.tmpl"),
}


def test_expect_base_comes_from_the_branch_ref_not_the_worktree():
    """恆真斷言的防線。

    `EXPECT_BASE` 一旦改成 `git -C "$BASE_WT" rev-parse`，job 內的
    commit 斷言就退化成「BASE worktree 等於它自己」——永遠通過，
    而「BASE_WT 指錯樹」那個最廉價也最致命的假綠就此無人看守。
    """
    m = re.search(r'^EXPECT_BASE="\$\((.+)\)"$', _SRC, flags=re.MULTILINE)
    assert m, "找不到 EXPECT_BASE 的賦值"
    expr = m.group(1)

    assert 'origin/$BASE_BRANCH' in expr, (
        f"EXPECT_BASE 必須取自基準**分支 ref**，實得：{expr}")
    assert "$BASE_WT" not in expr, (
        f"EXPECT_BASE 取自 BASE worktree 自己的 HEAD → 斷言恆真，實得：{expr}")


@pytest.mark.parametrize("case,expected", EXPECTED_TOPOLOGY.items(), ids=list(EXPECTED_TOPOLOGY))
def test_case_topology_is_declared(case, expected):
    """拓撲表是唯一一份宣告；漏一個 case 或指錯 template 都是靜默失敗。"""
    branch, tmpl = expected
    # 取 `  <case>)` 到該分支結尾 `;;` 之間；不靠「下一個 case」定界，
    # 因為最後一個 case 後面接的是 `*)`，那會讓天真的 regex 抓空。
    m = re.search(rf'^\s*{case}\)$(.*?);;', _SRC, flags=re.MULTILINE | re.DOTALL)
    assert m, f"腳本內找不到 case {case} 的分支"
    body = m.group(1)

    assert branch in body, f"case {case} 未指向基準分支 {branch}"
    assert tmpl in body, f"case {case} 未指向 template {tmpl}"


def test_all_verify_templates_are_reachable_through_this_entry():
    """自證：新增一支驗收 template 卻沒接進提交入口，就沒有人會用它。"""
    on_disk = {p.name for p in (REPO_ROOT / "scripts" / "slurm").glob("verify_*.sbatch.tmpl")}
    declared = {pathlib.Path(t).name for _, t in EXPECTED_TOPOLOGY.values()}

    assert on_disk == declared, (
        f"驗收 template 與提交入口不符：\n  只在磁碟 {on_disk - declared}"
        f"\n  只在入口 {declared - on_disk}")


def test_preflight_steps_are_present():
    """三件「忘了就白跑一輪」的前置。"""
    for step, why in (
        ("git -C \"$ROOT\" fetch origin", "沒 fetch 就會用到舊的 ref"),
        ('rm -f "$EV"/*.json', "未追蹤的證據檔會擋住 checkout（實際發生過兩次）"),
        ('checkout --detach "$REF"', "HEAD worktree 必須切到受測 ref"),
    ):
        assert step in _SRC, f"缺少前置：{step}——{why}"


def test_evidence_cleanup_is_path_guarded():
    """清理只准打在那一個目錄。參數化的白名單若由待驗的值自己構成會恆真
    ——`_verify_common.sh` 的 vc_cleanup 就踩過這個坑。"""
    assert "*/knowledge/superpowers/evidence)" in _SRC, "證據清理缺路徑白名單"
    assert "拒絕清理非預期路徑" in _SRC, "白名單沒有拒絕分支＝形同虛設"

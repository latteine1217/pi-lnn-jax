"""相容投影的鍵集不得新增 —— 新增鍵算 userspace 變更。

裁決（2026-08-07，人審）：往 `write_compatibility_projection` 的 payload 加一個鍵
**算 userspace 變更**，與數值變動同等看待。

背景：protocol 遷移時我把 `evaluation_protocol` 同時放進 artifact 的 provenance
與相容投影。A/B 因此對三組判定不一致——每個 metric 的值都相同，唯一差異是那個
新增鍵。artifact 是新的正典記錄（先前無人消費），加欄位無妨；**投影存在的全部
意義就是不變**，所以那一份是違規，已移除。

本檔把「投影的頂層鍵集」釘成硬編碼字面表。改動它就是改 userspace，必須有人
明確更新這張表——那正是該被看見的時刻。
"""
from __future__ import annotations

import ast
import pathlib

import pytest
from _paths import REPO_ROOT

_SCRIPTS = REPO_ROOT / "scripts"

#: 每支 producer 的相容投影頂層鍵。硬編碼字面值，不從被檢查的檔案推導。
PROJECTION_KEYS = {
    "evaluate_baselines.py": {
        "source", "baselines", "held_out", "sensor_T", "grid_stride",
        "max_grid", "pod_modes", "rows"},
    "eval_gappy_cross_re.py": {"grid", "pod_modes", "rows"},
    "cost_accuracy.py": {"pod_modes_list", "metric", "rows"},
    "train_baseline_shred.py": {
        "model", "grid", "window", "hidden", "epochs", "held_out",
        "field_mean", "field_std", "rows"},
}

#: 這個鍵屬於 artifact 的 provenance，不屬於投影。單獨列出因為它是那次違規的實例。
FORBIDDEN_IN_PROJECTION = "evaluation_protocol"


def _projection_payload_keys(path: pathlib.Path) -> set[str] | None:
    """取出傳給 write_compatibility_projection 的 dict literal 的頂層鍵。

    payload 可能直接是 dict literal，也可能是先組好的變數（如 `summary`）；
    兩種都處理。回傳 None 表示掃不出來——呼叫端要把它當失敗，不是當通過。
    """
    tree = ast.parse(path.read_text())
    payload_node = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "write_compatibility_projection"
                and len(node.args) >= 2):
            payload_node = node.args[1]
            break
    if payload_node is None:
        return None
    if isinstance(payload_node, ast.Dict):
        return {k.value for k in payload_node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    if isinstance(payload_node, ast.Name):
        target = payload_node.id
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                    and any(isinstance(t, ast.Name) and t.id == target
                            for t in node.targets)):
                return {k.value for k in node.value.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return None


@pytest.mark.parametrize("name", sorted(PROJECTION_KEYS))
def test_projection_keys_match_the_record(name):
    keys = _projection_payload_keys(_SCRIPTS / name)

    assert keys is not None, (
        f"{name}：掃不出 write_compatibility_projection 的 payload 鍵——"
        "本測試對它形同虛設，修掃描器而不是刪測試")
    assert keys == PROJECTION_KEYS[name], (
        f"{name} 的相容投影鍵集改變了：\n"
        f"    紀錄 {sorted(PROJECTION_KEYS[name])}\n"
        f"    實際 {sorted(keys)}\n"
        f"    新增 {sorted(keys - PROJECTION_KEYS[name])}  "
        f"移除 {sorted(PROJECTION_KEYS[name] - keys)}\n"
        "  裁決（2026-08-07 人審）：投影新增鍵算 userspace 變更。\n"
        "  新的欄位放 artifact 的 provenance——那是正典記錄，加欄位無妨；\n"
        "  投影存在的意義就是不變。真要改投影，回查 scripts/CLAUDE.md §3.4 並更新本表。")


@pytest.mark.parametrize("name", sorted(PROJECTION_KEYS))
def test_protocol_stays_out_of_the_projection(name):
    """那次違規的實例，單獨釘住。"""
    keys = _projection_payload_keys(_SCRIPTS / name)
    assert keys is not None
    assert FORBIDDEN_IN_PROJECTION not in keys, (
        f"{name} 又把 {FORBIDDEN_IN_PROJECTION} 放進投影。"
        "協定屬 artifact 的 provenance；投影是相容面，不得新增欄位。")


def test_the_guard_can_actually_fire(tmp_path):
    """鑑別力自檢：對一份把協定放進投影的程式碼，掃描器必須抓到。"""
    f = tmp_path / "fake.py"
    f.write_text(
        'summary = {"rows": rows, "evaluation_protocol": args.protocol}\n'
        'write_compatibility_projection(out, summary)\n')
    assert _projection_payload_keys(f) == {"rows", "evaluation_protocol"}


def test_record_is_a_literal_not_derived():
    assert all(isinstance(v, set) and v for v in PROJECTION_KEYS.values())
    assert all("rows" in v for v in PROJECTION_KEYS.values()), (
        "每支的投影都以 rows 為主體；某支沒有就代表這張表抄錯了")

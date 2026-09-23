"""每個呼叫協定腳本的 slurm 入口都必須宣告評估協定。

Why：`--protocol` 在 CLI 端是 required，所以漏宣告的 sbatch 會在 job 內
argparse 就失敗——不會產生錯的數字。但那個失敗只在 lab-server 上、跑了才知道，
而排隊到失敗之間可能是幾小時。本檔把它移到本機每次提交都會跑的範圍內。

Why 掃描而非清單：新增 sbatch 時不會有人記得回來加測試。掃描讓遺漏自己現形。

`--protocol` 的**值**不在本檔的驗證範圍——選 follow_training 還是 fixed_grid
是實驗設計決定（見 knowledge/superpowers/evidence/protocol_ab_preregistration.md），
本檔只確認「有人做了那個決定並寫下來」。
"""
from __future__ import annotations

import pathlib
import re

import pytest
from _paths import REPO_ROOT

_SLURM = REPO_ROOT / "scripts" / "slurm"

#: 這些腳本的 --protocol 是 required；呼叫它們的 sbatch 必須傳。
PROTOCOL_AWARE = (
    "evaluate_exp245.py",
    "evaluate_multi_re.py",
    "evaluate_baselines.py",
    "eval_gappy_cross_re.py",
    "train_baseline_shred.py",
    "cost_accuracy.py",
    "classical_baselines_fair.py",
    "enkf_baseline.py",
)

#: 豁免：逐檔列理由的硬編碼字面清單。由被檢查之物自己構成的豁免會恆真。
EXEMPT: dict[str, str] = {}


def _invokes(text: str) -> bool:
    """只算**實際呼叫行**：註解裡提到腳本名不是呼叫。

    這條分辨很重要——`train_exp.sbatch.tmpl` 在註解裡說明「final eval 用
    evaluate_exp245」，它自己並不呼叫。把註解算成呼叫會逼人去改一個沒有問題的檔。
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if "python" not in stripped and "PY_ARGS" not in stripped:
            continue
        if any(re.search(rf"scripts/{re.escape(s)}\b", stripped) for s in PROTOCOL_AWARE):
            return True
    return False


def _callers() -> list[pathlib.Path]:
    out = []
    for p in sorted(_SLURM.rglob("*")):
        if not p.is_file() or p.suffix not in (".sbatch", ".tmpl", ".sh"):
            continue
        if _invokes(p.read_text(errors="replace")):
            out.append(p)
    return out


CALLERS = _callers()


def test_scan_sees_the_known_callers():
    """自證：掃不到就代表下面每一條都空過。"""
    names = {p.name for p in CALLERS}
    assert len(CALLERS) >= 8, f"只掃到 {len(CALLERS)} 個呼叫端——掃描器或路徑壞了"
    for expected in ("eval_once.sbatch.tmpl", "eval_baselines.sbatch.tmpl"):
        assert expected in names, f"{expected} 不在掃描結果中"


def test_the_guard_can_actually_fire(tmp_path):
    """鑑別力自檢：一份呼叫協定腳本卻不傳 --protocol 的 sbatch 必須被抓到。"""
    f = tmp_path / "fake.sbatch"
    f.write_text("uv run python -u scripts/evaluate_exp245.py --config x.toml\n")
    assert _invokes(f.read_text())
    assert "--protocol" not in f.read_text()

    # 反向：只在註解裡提到不算呼叫
    g = tmp_path / "comment_only.sbatch"
    g.write_text("# final eval 用 scripts/evaluate_exp245.py\necho hi\n")
    assert not _invokes(g.read_text())


def _invocation_blocks(text: str) -> list[str]:
    """取出每個呼叫的完整區塊（指令行 + 其續行）。

    只檢查「檔案裡有沒有 --protocol」是不夠的：本輪就寫壞過一次——插入的
    `--protocol` 前面多了一個空行，續行因此被截斷，那一行變成一個獨立指令
    （執行時 command not found）。`bash -n` 抓不到，因為它只驗語法。
    """
    blocks, cur = [], None
    for line in text.splitlines():
        if cur is not None:
            cur.append(line)
            if not line.rstrip().endswith("\\"):
                blocks.append("\n".join(cur))
                cur = None
            continue
        stripped = line.strip()
        if stripped.startswith("#") or "python" not in stripped:
            continue  # echo "...scripts/evaluate_exp245.py..." 不是呼叫
        if any(re.search(rf"scripts/{re.escape(sc)}\b", stripped) for sc in PROTOCOL_AWARE):
            if line.rstrip().endswith("\\"):
                cur = [line]
            else:
                blocks.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


@pytest.mark.parametrize("path", CALLERS, ids=lambda p: p.name)
def test_caller_declares_a_protocol(path):
    text = path.read_text(errors="replace")
    blocks = _invocation_blocks(text)
    assert blocks, f"{path.name} 掃不出呼叫區塊——偵測器與 _invokes 不一致"
    # 協定必須在**呼叫區塊之內**，不是檔案裡任何地方
    declared = all(("--protocol" in b) or ("PROTO_ARGS" in b) for b in blocks)

    if path.name in EXEMPT:
        assert not declared, (
            f"{path.name} 列在 EXEMPT 卻已宣告協定——豁免已過期，請移除。"
            f"（理由當初是：{EXEMPT[path.name]}）")
        return

    assert declared, (
        f"{path.relative_to(REPO_ROOT)} 呼叫了協定腳本卻沒有傳 --protocol。\n"
        "  三種模式擇一並寫進 sbatch：follow_training（跟隨訓練 time_strides）／\n"
        "  fixed_grid（固定格點，須 --protocol-reason）／sensor_time_independent。\n"
        "  Why：漏傳會在 job 內才失敗，而排隊到失敗之間可能是幾小時。")


def test_exemptions_are_a_literal_list_not_derived():
    assert isinstance(EXEMPT, dict)
    assert all(isinstance(k, str) and isinstance(v, str) and v for k, v in EXEMPT.items())
    assert len(EXEMPT) <= 2, (
        f"豁免長到 {len(EXEMPT)} 個——那不再是例外，是這條規則沒人在遵守")


def test_no_caller_still_seds_time_strides_out_of_a_config():
    """反向斷言：協定的來源解析回到 shell 就等於這次重構白做。

    `submit_crossre.sh` 原本用 sed 從 TOML 撈 time_strides——那是 module 該做的事，
    而且只保護了那一條路徑（實測 12 份既有產物不受保護，見 TD-4）。
    """
    for path in CALLERS:
        text = path.read_text(errors="replace")
        offending = [ln for ln in text.splitlines()
                     if "time_strides" in ln and "sed" in ln]
        assert not offending, (
            f"{path.name} 又用 shell 解析 time_strides：\n    " + "\n    ".join(offending)
            + "\n  改用 --protocol follow_training，讓 eval 端自己讀 config。")


def test_block_extraction_catches_a_broken_continuation(tmp_path):
    """鑑別力自檢：續行被空行截斷時，--protocol 不算在呼叫區塊內。

    本輪實際寫壞過一次，而 `bash -n` 沒抓到。
    """
    good = ("uv run python -u scripts/evaluate_exp245.py \\\n"
            "  --protocol follow_training \\\n  --config x.toml\n")
    broken = ("uv run python -u scripts/evaluate_exp245.py \\\n"
              "\n--protocol follow_training \\\n  --config x.toml\n")

    assert all("--protocol" in b for b in _invocation_blocks(good))
    assert not all("--protocol" in b for b in _invocation_blocks(broken))

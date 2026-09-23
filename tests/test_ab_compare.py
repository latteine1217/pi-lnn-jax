"""驗收對拍器 `scripts/_common/ab_compare.py` 的行為測試。

What: 釘住四條「假綠路徑」——舊版兩支 sbatch 把比對邏輯寫在 heredoc 裡，
      無法被任何測試碰到，於是下列四種情況都會回報成功：
        1. determinism control 算了 host 排程卻不納入判斷
        2. 錄製失敗留下的舊檔被當成本次結果比對
        3. 基準路徑誤指向重構後的樹（兩邊同 commit）
        4. 只比 ledger 與兩個 digest，評估數字接錯不會被發現

Why 抽成模組: heredoc 內的邏輯永遠不會被執行到——它只在 lab-server 的 job 裡跑，
      而那正是我們要用它來背書的場合。要讓「守衛確實會發作」可被證明，
      邏輯必須離開 heredoc，進到本機每次提交都會跑的測試範圍內。

比對的形狀統一為「回傳問題清單，空清單=通過」，呼叫端不必記每個函式各自的真值慣例。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from _paths import REPO_ROOT

for _p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import ab_compare  # noqa: E402


TAIL = {"step": -1, "params_digest": "aaaaaaaaaaaaaaaa",
        "opt_state_digest": "bbbbbbbbbbbbbbbb"}


# ─── (1) determinism control 必須把 host 排程納入判斷 ──────────────────────

def test_ledger_verdict_fails_when_schedule_differs_but_digests_match():
    """舊 gate 是 `if not (p and o)`——排程不同、digest 恰好相同就放行。

    這不是假想：digest 只蓋 params 與 opt_state，collocation 取樣點、RAR 觸發、
    時間窗全都不進 digest。排程漂移而參數碰巧同值時，舊 gate 會宣告 CPU 可重現，
    後續三段對拍就建立在一個假前提上。
    """
    a = [{"step": 0, "cx": "0.1"}, TAIL]
    b = [{"step": 0, "cx": "0.2"}, TAIL]

    problems = ab_compare.ledger_verdict(a, b)

    assert problems, "排程不同卻判為決定性——假綠路徑仍在"
    assert any("排程" in p for p in problems), problems


def test_ledger_verdict_passes_when_schedule_and_digests_all_match():
    rows = [{"step": 0, "cx": "0.1"}, {"step": 1, "cx": "0.2"}, TAIL]

    assert ab_compare.ledger_verdict(rows, list(rows)) == []


def test_ledger_verdict_fails_when_digests_differ():
    a = [{"step": 0, "cx": "0.1"}, TAIL]
    b = [{"step": 0, "cx": "0.1"}, dict(TAIL, params_digest="cccccccccccccccc")]

    problems = ab_compare.ledger_verdict(a, b)

    assert any("params" in p for p in problems), problems


def test_split_ledger_keeps_step_zero_rows():
    """cylinder 的迴圈從 step 0 起跑；舊 Kolmogorov 比對用 `step > 0` 過濾，
    照抄到 cylinder 會整個吃掉第一步。統一以「非尾列」為準。"""
    rows = [{"step": 0, "cx": "a"}, {"step": 1, "cx": "b"}, TAIL]

    steps, tail = ab_compare.split_ledger(rows)

    assert [r["step"] for r in steps] == [0, 1]
    assert tail["params_digest"] == TAIL["params_digest"]


def test_split_ledger_rejects_ledger_without_exactly_one_tail():
    with pytest.raises(ab_compare.LedgerShapeError):
        ab_compare.split_ledger([{"step": 0, "cx": "a"}])


# ─── (3) 比對雙方的 commit 必須被斷言，而非只印出來 ────────────────────────

def test_commit_check_rejects_base_equal_head():
    """BASE_WT 誤指向 HEAD worktree 時，逐位元對拍必然完美通過。
    這是最廉價也最致命的假綠：整份證據變成「HEAD 等於自己」。"""
    problems = ab_compare.commit_check(
        base_sha="24667ab", head_sha="24667ab", expect_base="24667ab")

    assert any("同一個 commit" in p for p in problems), problems


def test_commit_check_rejects_base_not_matching_expectation():
    problems = ab_compare.commit_check(
        base_sha="deadbee", head_sha="24667ab", expect_base="f2d63ff")

    assert any("f2d63ff" in p for p in problems), problems


def test_commit_check_rejects_too_short_expectation():
    """前綴比對的代價：`EXPECT_BASE` 若被 shell 截成一兩個字元，
    斷言會退化成「幾乎一定通過」。太短就當作沒填。"""
    problems = ab_compare.commit_check(
        base_sha="f2d63ff", head_sha="24667ab", expect_base="f2")

    assert any("太短" in p for p in problems), problems


def test_commit_check_accepts_expected_pair_by_prefix():
    """短 sha 與長 sha 都可能出現（`rev-parse --short` vs 呼叫端手打），
    以較短者為長度做前綴比對。"""
    assert ab_compare.commit_check(
        base_sha="f2d63ff1234567", head_sha="24667ab",
        expect_base="f2d63ff", expect_head="24667ab") == []


# ─── (4) 評估產物必須進入比對面 ────────────────────────────────────────────

def _summary(**over):
    base = {
        "config_path": "/home/u/base-wt/configs/x.toml",
        "ckpt_dir": "/home/u/base-wt/artifacts/cpuab/f1/checkpoints",
        "train_wall_seconds": 12.34,
        "arch": "liquid",
        "n_params": 123456,
        "ckpt_steps": [10, 20],
        "final_metrics": {"u_mape": 5.73, "v_mape": 6.10},
    }
    base.update(over)
    return base


def test_summary_diff_detects_changed_final_metrics():
    """評估路徑接錯（如場的兩個分量對調）不動訓練參數，
    ledger 與兩個 digest 全等，但使用者看到的數字已變。"""
    base = _summary()
    head = _summary(final_metrics={"u_mape": 6.10, "v_mape": 5.73})

    problems = ab_compare.summary_diff(base, head)

    assert any("final_metrics" in p for p in problems), problems


def test_summary_diff_ignores_wall_time_and_worktree_paths():
    """兩個 worktree 的絕對路徑必然不同、掛鐘時間必然不同——
    這兩類若不排除，比對永遠紅，等於沒有比對。"""
    base = _summary()
    head = _summary(config_path="/home/u/head-wt/configs/x.toml",
                    ckpt_dir="/home/u/head-wt/artifacts/cpuab/f1/checkpoints",
                    train_wall_seconds=99.99)

    assert ab_compare.summary_diff(base, head) == []


def test_summary_diff_rejects_summary_without_final_metrics():
    """自證比對面非空：若哪天 summary 不再寫 final_metrics，
    `summary_diff` 會安靜地變成「比對零個有意義的欄位」然後全綠。"""
    stripped = _summary()
    del stripped["final_metrics"]

    problems = ab_compare.summary_diff(stripped, dict(stripped))

    assert any("final_metrics" in p for p in problems), problems


# ─── (4b) cylinder 不落盤，只能比 stdout 的評估報表區塊 ────────────────────

CYL_MARKER = ab_compare.CYL_EVAL_REPORT_MARKER

_CYL_STDOUT = """\
[cyl] npz=/home/u/base-wt/data/cylinder_v1.npz
[cyl] train wall 12.34s

=== KE / vorticity (PRIMARY criterion) ===
  (5 val times)
  ke_pred/ke_ref = 1.2200   KE rel-err = 0.3400
  omega rel-err  = 0.5600   div L2 = 1.2000e-03
"""


def test_eval_report_ignores_everything_before_the_marker():
    """報表之前是路徑與掛鐘時間——兩個 worktree 必然不同。"""
    other = _CYL_STDOUT.replace("/home/u/base-wt", "/home/u/head-wt") \
                       .replace("12.34s", "98.76s")

    assert ab_compare.eval_report_diff(_CYL_STDOUT, other) == []


def test_eval_report_diff_detects_changed_number():
    changed = _CYL_STDOUT.replace("1.2200", "1.3900")

    problems = ab_compare.eval_report_diff(_CYL_STDOUT, changed)

    assert any("1.2200" in p for p in problems), problems


def test_eval_report_diff_says_so_when_it_stops_listing():
    """列印上限不得靜默：截斷的清單看起來就像「只有這幾處不同」。"""
    a = "\n".join([CYL_MARKER] + [f"  x = {i}" for i in range(20)])
    b = "\n".join([CYL_MARKER] + [f"  x = {i + 100}" for i in range(20)])

    problems = ab_compare.eval_report_diff(a, b)

    assert any("尚有" in p for p in problems), problems


def test_eval_report_requires_marker():
    """跑掛了、stdout 被截斷、或報表標題被改名時，必須炸而不是回報「無差異」。"""
    with pytest.raises(ab_compare.EvalReportError):
        ab_compare.eval_report_diff("[cyl] 什麼都沒印\n", _CYL_STDOUT)


# ─── CLI：sbatch 實際呼叫的那一層 ──────────────────────────────────────────

def _plant(root: Path, name: str, *, cx: str, summary: dict | None = None) -> None:
    d = root / "artifacts" / "ab" / name
    d.mkdir(parents=True)
    (d / "rng_ledger.json").write_text(json.dumps(
        [{"step": 0, "cx": cx}, TAIL]))
    if summary is not None:
        (d / "summary.json").write_text(json.dumps(summary))


def _run_ab(tmp_path: Path, *, base_cx: str, head_cx: str,
            expect_base: str = "f2d63ff", base_sha: str = "f2d63ff",
            head_sha: str = "24667ab") -> tuple[int, dict]:
    base, head = tmp_path / "base", tmp_path / "head"
    _plant(base, "c1", cx=base_cx, summary=_summary())
    _plant(head, "c1", cx=head_cx, summary=_summary())
    ev = tmp_path / "evidence.json"
    rc = ab_compare.main([
        "ab", "--base-root", str(base), "--head-root", str(head),
        "--names", "c1",
        "--ledger-rel", "artifacts/ab/{name}/rng_ledger.json",
        "--summary-rel", "artifacts/ab/{name}/summary.json",
        "--base-sha", base_sha, "--head-sha", head_sha,
        "--expect-base", expect_base, "--evidence", str(ev),
    ])
    return rc, json.loads(ev.read_text())


def test_cli_ab_returns_zero_and_writes_evidence_when_identical(tmp_path):
    rc, ev = _run_ab(tmp_path, base_cx="0.1", head_cx="0.1")

    assert rc == 0
    assert ev["ok"] is True
    for side in ("base", "head"):
        assert ev["fixtures"]["c1"][f"{side}_params_digest"] == TAIL["params_digest"]


def test_cli_ab_returns_nonzero_when_schedule_diverges(tmp_path):
    rc, ev = _run_ab(tmp_path, base_cx="0.1", head_cx="0.2")

    assert rc == 1
    assert ev["ok"] is False


def test_cli_ab_returns_nonzero_when_base_commit_unexpected(tmp_path):
    """即使三份 fixture 全等，比對雙方不是預期的 commit 就不算通過——
    否則 BASE_WT 指錯樹時會得到一份完美但無意義的證據。"""
    rc, ev = _run_ab(tmp_path, base_cx="0.1", head_cx="0.1",
                     expect_base="deadbee")

    assert rc == 1
    assert ev["commit_problems"], ev


def test_cli_ab_returns_nonzero_when_a_fixture_ledger_is_missing(tmp_path):
    base, head = tmp_path / "base", tmp_path / "head"
    _plant(base, "c1", cx="0.1")
    _plant(head, "c1", cx="0.1")   # c2 兩邊都沒種

    rc = ab_compare.main([
        "ab", "--base-root", str(base), "--head-root", str(head),
        "--names", "c1,c2",
        "--ledger-rel", "artifacts/ab/{name}/rng_ledger.json",
        "--base-sha", "f2d63ff", "--head-sha", "24667ab",
        "--expect-base", "f2d63ff",
    ])

    assert rc == 1


# ─── ledger 關閉組態：沒有 ledger 可比，標的是落盤產物 ─────────────────────

def test_cli_ab_works_without_a_ledger(tmp_path):
    """production 是 ledger 關閉的，該組態下沒有 rng_ledger.json。

    標的改為落盤的 summary：其 `final_metrics` 由最終 params 算出，
    params 若不同必然反映出來。
    """
    base, head = tmp_path / "base", tmp_path / "head"
    for root in (base, head):
        d = root / "artifacts" / "off" / "r"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps(_summary()))

    rc = ab_compare.main([
        "ab", "--base-root", str(base), "--head-root", str(head), "--names", "r",
        "--summary-rel", "artifacts/off/{name}/summary.json",
        "--base-sha", "f2d63ff", "--head-sha", "24667ab", "--expect-base", "f2d63ff",
    ])

    assert rc == 0


def test_cli_ab_refuses_to_run_with_no_comparison_surface(tmp_path):
    """三種標的都沒給 → 每個 fixture 都零比對，卻回報通過。
    這是最徹底的假綠，必須拒絕而不是安靜放行。"""
    rc = ab_compare.main([
        "ab", "--base-root", str(tmp_path), "--head-root", str(tmp_path),
        "--names", "r", "--base-sha", "a1b2c3d", "--head-sha", "e4f5a6b",
        "--expect-base", "a1b2c3d",
    ])

    assert rc != 0, "零比對面卻回報通過"


# ─── metrics 投影的逐鍵比對（候選 2 的驗收判定）─────────────────────────
#
# 這一組對應的假綠路徑與上面四條同源：判定若寫在 sbatch heredoc 裡，
# 「兩份 metrics.json 是否等價」就永遠沒有守衛。

def _proj(**over):
    base = {"rows": [{"Re": 10000.0, "method": "gappy_pod",
                      "u_rel_err": 0.1234, "ke_rel_err": 0.4321, "wall_s": 3.5}]}
    base.update(over)
    return base


def test_projection_diff_passes_on_float_noise_within_tolerance():
    """加總順序造成的 ~1e-16 位移不是差異——否則遷移永遠驗不過。"""
    head = _proj()
    head["rows"][0]["u_rel_err"] += 5e-17

    assert ab_compare.metrics_projection_diff(_proj(), head) == []


def test_projection_diff_catches_a_real_move():
    head = _proj()
    head["rows"][0]["u_rel_err"] = 0.1244

    problems = ab_compare.metrics_projection_diff(_proj(), head)

    assert problems and any("u_rel_err" in p for p in problems)


def test_projection_diff_catches_structural_drift_not_just_values():
    """少一列、少一個鍵、型別變了都必須報——只比共同鍵是假綠的經典形狀。"""
    assert ab_compare.metrics_projection_diff(_proj(), {"rows": []})
    dropped = _proj()
    del dropped["rows"][0]["ke_rel_err"]
    assert any("ke_rel_err" in p for p in
               ab_compare.metrics_projection_diff(_proj(), dropped))
    assert ab_compare.metrics_projection_diff(
        _proj(), _proj(rows=[{"Re": "10000", "method": "gappy_pod",
                              "u_rel_err": 0.1234, "ke_rel_err": 0.4321,
                              "wall_s": 3.5}]))


def test_projection_diff_reports_nan_to_null_as_its_own_class():
    """舊路徑寫字面 NaN、新路徑寫 null。這是本次刻意的變更，
    但不得靜默吞掉——要能與「數字動了」分辨開來。"""
    base = _proj()
    base["rows"][0]["ke_rel_err"] = float("nan")
    head = _proj()
    head["rows"][0]["ke_rel_err"] = None

    problems = ab_compare.metrics_projection_diff(base, head)

    assert problems, "NaN→null 不得靜默通過"
    assert any("NaN" in p and "null" in p for p in problems), problems


def test_projection_diff_announces_which_keys_it_skipped():
    """timing 欄位跨 run 必然不同，排除是對的——但沉默的排除讀起來像「全部都比過了」。"""
    head = _proj()
    head["rows"][0]["wall_s"] = 99.0

    problems = ab_compare.metrics_projection_diff(_proj(), head)
    notes = ab_compare.metrics_projection_skipped(_proj())

    assert problems == [], "timing 欄位不該讓對拍變紅"
    assert any("wall_s" in n for n in notes), notes


def test_projection_diff_refuses_a_comparison_with_nothing_left():
    """整份只剩被排除的欄位 → 比對面為空，回報通過等於什麼都沒驗。"""
    problems = ab_compare.metrics_projection_diff(
        {"rows": [{"wall_s": 1.0}]}, {"rows": [{"wall_s": 2.0}]})

    assert problems and any("比對面" in p for p in problems)


def test_projection_skip_list_is_a_literal_not_derived():
    """排除清單若由被比對的資料自己構成，守衛就恆真。"""
    assert isinstance(ab_compare.VOLATILE_METRIC_KEYS, frozenset)
    assert ab_compare.VOLATILE_METRIC_KEYS
    assert all(isinstance(k, str) for k in ab_compare.VOLATILE_METRIC_KEYS)


def test_cli_metrics_reports_rc_and_notes(tmp_path, capsys):
    """CLI 是驗收 job 的呼叫面：一致 rc=0、不一致 rc≠0，且排除項要印出來。"""
    base, head = tmp_path / "b.json", tmp_path / "h.json"
    base.write_text(json.dumps(_proj()))
    head.write_text(json.dumps(_proj()))

    assert ab_compare.main(["metrics", "--base", str(base), "--head", str(head)]) == 0
    assert "wall_s" in capsys.readouterr().out

    moved = _proj()
    moved["rows"][0]["u_rel_err"] = 0.9
    head.write_text(json.dumps(moved))

    assert ab_compare.main(["metrics", "--base", str(base), "--head", str(head)]) != 0


def test_projection_diff_ignores_worktree_absolute_paths():
    """A/B 的兩腿在不同 worktree，含 repo 根的絕對路徑必然不同。

    實跑背書：job 4927/4931 的對拍曾只因 `source` 一欄判紅，而那一欄是
    `str(Path(args.config).resolve())`——純出處，不是結果。summary_diff 早就
    排除同一類（IGNORED_SUMMARY_KEYS 的 config_path/ckpt_dir），此處補齊。
    """
    base = _proj(source="/home/u/pi-lnn-jax-metricab-base/configs/x.toml")
    head = _proj(source="/home/u/pi-lnn-jax-metricab/configs/x.toml")

    assert ab_compare.metrics_projection_diff(base, head) == []
    assert any("source" in n for n in ab_compare.metrics_projection_skipped(base))


def test_cli_metrics_writes_machine_readable_evidence(tmp_path):
    """證據要機器產出。手寫的結論無法複驗——那正是 evidence/README 要避免的。"""
    base, head = tmp_path / "b.json", tmp_path / "h.json"
    base.write_text(json.dumps(_proj()))
    head.write_text(json.dumps(_proj()))
    ev = tmp_path / "ev" / "metrics_ab.json"

    rc = ab_compare.main([
        "metrics", "--base", str(base), "--head", str(head),
        "--evidence", str(ev), "--base-sha", "aaaaaaa", "--head-sha", "bbbbbbb",
        "--job-id", "4931",
    ])

    assert rc == 0
    rec = json.loads(ev.read_text())
    assert rec["ok"] is True and rec["problems"] == []
    assert rec["base_sha"] == "aaaaaaa" and rec["job_id"] == "4931"
    assert any("wall_s" in s for s in rec["skipped"]), "排除項要進證據，否則讀者不知道放過了什麼"


def test_evidence_records_failure_too(tmp_path):
    """只在通過時落證據，等於證據永遠說通過。"""
    base, head = tmp_path / "b.json", tmp_path / "h.json"
    base.write_text(json.dumps(_proj()))
    moved = _proj(); moved["rows"][0]["u_rel_err"] = 0.9
    head.write_text(json.dumps(moved))
    ev = tmp_path / "ev.json"

    rc = ab_compare.main(["metrics", "--base", str(base), "--head", str(head),
                          "--evidence", str(ev)])

    assert rc == 2
    rec = json.loads(ev.read_text())
    assert rec["ok"] is False and rec["problems"]

"""重構前後對拍的判定邏輯（兩支 verify_*_cpu_ab sbatch 共用）。

What: 把「兩份 ledger 是否逐位元相同」「比對雙方是不是預期的那兩個 commit」
      「評估產物是否一致」三件事，從 sbatch 的 heredoc 裡抽出來。

Why 不留在 heredoc: heredoc 內的判定只在 lab-server 的 job 裡執行，本機測試永遠碰不到
      ——而那正是我們要拿它來背書 bit-identical 的地方。判定器本身沒有守衛，
      就等於整份驗收沒有守衛。行為由 `tests/test_ab_compare.py` 釘住，
      其中每一條都對應一個實際存在過的假綠路徑。

回傳形狀統一：**問題清單，空清單代表通過**。呼叫端只需 `if problems:`，
不必記住哪個函式回傳 bool、哪個回傳 None。

CLI（給 sbatch 用）:
    python scripts/_common/ab_compare.py determinism --a A.json --b B.json
    python scripts/_common/ab_compare.py ab --base-root DIR --head-root DIR \
        --names c1,c2,c3 --ledger-rel 'artifacts/ab/{name}/rng_ledger.json' \
        [--summary-rel ...] [--stdout-rel ...] \
        --base-sha X --head-sha Y --expect-base Z [--expect-head W] \
        [--evidence OUT.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: summary.json 中必然因執行環境而異的欄位——不排除則比對永遠紅，等於沒有比對。
#: 只列「跨 worktree 必然不同」者；任何新欄位預設進入比對面（安全方向）。
IGNORED_SUMMARY_KEYS = frozenset({
    "config_path",         # str(Path(args.config).resolve())，含 worktree 根
    "ckpt_dir",            # 同上
    "train_wall_seconds",  # 掛鐘時間
})

#: 必須存在的承重欄位。缺了它，summary_diff 會退化成「比對零個有意義欄位」後全綠。
REQUIRED_SUMMARY_KEYS = ("final_metrics",)

#: cylinder 的 finalize 不落盤（只 print），故只能比 stdout 的報表區塊。
#: 標記行以下全是格式化過的數字，標記行以上是路徑與掛鐘時間。
CYL_EVAL_REPORT_MARKER = "=== KE / vorticity (PRIMARY criterion) ==="

#: git `rev-parse --short` 的預設長度；預期值短於此視為填錯。
_MIN_SHA_LEN = 7

#: 評估報表最多列出幾處分岔——超出的處數會另行註明，不靜默截斷。
_MAX_REPORTED = 3

#: 跨 run 必然不同的量：掛鐘時間。排除是對的，但**排除了什麼要說出來**
#: （`metrics_projection_skipped`），否則讀起來像「所有欄位都比過了」。
#: 硬編碼字面清單：若由被比對的資料自己構成（例如「欄名含 _s 就跳過」），
#: 新增一個 `foo_s` 欄位就會自動退出比對面。
VOLATILE_METRIC_KEYS = frozenset({
    "wall_s",               # evaluate_baselines：單次重建掛鐘
    "fit_s",                # cost_accuracy：POD SVD 掛鐘
    "recon_s_per_field",    # cost_accuracy：該圖的 x 軸，本身就是掛鐘量
    "code_revision",        # provenance：兩邊本來就是不同 commit
    "code_dirty",
    # 含 worktree 根的絕對路徑——A/B 的兩腿必然在不同目錄，故必然不同。
    # `IGNORED_SUMMARY_KEYS` 為 summary_diff 排除的是同一類（config_path/ckpt_dir）；
    # 這裡漏掉曾讓 job 4927/4931 的對拍只因 `source` 一欄就判紅。
    "source",               # evaluate_baselines：str(Path(args.config).resolve())
    "config",               # evaluate_multi_re：同上
})

#: 遷移前 `json.dump` 以 `allow_nan=True` 寫出字面 NaN；遷移後無定義的欄位落 null。
_NAN_TO_NULL = "NaN → null"


class LedgerShapeError(RuntimeError):
    """ledger 的形狀不符合契約（尾列不是恰好一列）。"""


class EvalReportError(RuntimeError):
    """stdout 裡找不到評估報表——跑掛、被截斷，或標記行被改名。"""


# ─── ledger ────────────────────────────────────────────────────────────────

def split_ledger(rows: list[dict]) -> tuple[list[dict], dict]:
    """拆成（逐步列, 尾列）。

    以「step == -1」認尾列而非以位置，因為尾列是最後才 append 的診斷列；
    逐步列則是「非尾列」全收——**不可**用 `step > 0` 過濾：cylinder 的迴圈
    從 step 0 起跑，那樣會整個吃掉第一步。
    """
    tails = [r for r in rows if r.get("step") == -1]
    if len(tails) != 1:
        raise LedgerShapeError(f"ledger 尾列（step=-1）應恰有一列，實得 {len(tails)}")
    return [r for r in rows if r.get("step") != -1], tails[0]


def schedule_diff(a_steps: list[dict], b_steps: list[dict]) -> str | None:
    """回傳首處分岔的描述；完全一致則回 None。"""
    if len(a_steps) != len(b_steps):
        return f"步數 {len(a_steps)} vs {len(b_steps)}"
    for x, y in zip(a_steps, b_steps):
        for k in sorted(set(x) | set(y)):
            if x.get(k) != y.get(k):
                return f"step {x.get('step')} 欄位 {k}: {x.get(k)!r} vs {y.get(k)!r}"
    return None


def ledger_verdict(a_rows: list[dict], b_rows: list[dict]) -> list[str]:
    """比對兩份 ledger 的**三**件事：host 排程、params digest、opt_state digest。

    三件都納入是本函式存在的理由。舊版 heredoc 算了排程卻只判兩個 digest——
    digest 只蓋 params 與 opt_state，collocation 取樣點、RAR 觸發、時間窗都不進 digest，
    所以「排程漂移而參數碰巧同值」會被放行。
    """
    a_steps, a_tail = split_ledger(a_rows)
    b_steps, b_tail = split_ledger(b_rows)

    problems: list[str] = []
    div = schedule_diff(a_steps, b_steps)
    if div is not None:
        problems.append(f"host 排程不同：{div}")
    for key, label in (("params_digest", "params"), ("opt_state_digest", "opt_state")):
        av, bv = a_tail.get(key), b_tail.get(key)
        if av != bv:
            problems.append(f"{label} digest 不同：{av} vs {bv}")
    return problems


# ─── commit ────────────────────────────────────────────────────────────────

def _sha_eq(a: str, b: str) -> bool:
    """短 sha 與長 sha 以較短者為長度做前綴比對；空字串一律不相等
    （否則空字串會前綴匹配任何 sha，把斷言變成永遠通過）。"""
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if not a or not b:
        return False
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def commit_check(base_sha: str, head_sha: str,
                 expect_base: str, expect_head: str = "") -> list[str]:
    """斷言比對的雙方確實是預期的那兩個 commit。

    最廉價也最致命的假綠是 BASE 路徑誤指向 HEAD worktree——那樣逐位元對拍必然
    完美通過，整份證據退化成「HEAD 等於自己」。舊版只 `echo` 兩個 hash 而不斷言。
    """
    problems: list[str] = []
    if _sha_eq(base_sha, head_sha):
        problems.append(f"BASE 與 HEAD 是同一個 commit（{base_sha}）——對拍無意義")
    for label, actual, expect, required in (
        ("BASE", base_sha, expect_base, True),
        ("HEAD", head_sha, expect_head, False),
    ):
        if not expect:
            if required:
                problems.append(f"未提供預期的 {label} commit——斷言等於沒做")
            continue
        # 前綴比對的代價：被截斷的預期值會退化成「幾乎一定通過」。太短就當作沒填。
        if len(expect.strip()) < _MIN_SHA_LEN:
            problems.append(
                f"預期的 {label} commit 太短（{expect!r}，需 ≥{_MIN_SHA_LEN} 字元）")
        elif not _sha_eq(actual, expect):
            problems.append(f"{label} commit 非預期：實得 {actual}，預期 {expect}")
    return problems


# ─── 評估產物 ───────────────────────────────────────────────────────────────

def summary_diff(base: dict, head: dict) -> list[str]:
    """比對 Kolmogorov 的 summary.json（排除必然因環境而異的欄位）。

    存在理由：訓練參數與 ledger 全等，不代表使用者看到的數字沒變——
    評估路徑接錯（場的兩個分量對調、遮罩傳錯）完全不動訓練，
    只比 digest 的對拍會回報一致。
    """
    problems: list[str] = []
    for key in REQUIRED_SUMMARY_KEYS:
        if key not in base or key not in head:
            problems.append(f"summary 缺承重欄位 {key}——比對面已退化，不得視為通過")
    for key in sorted((set(base) | set(head)) - IGNORED_SUMMARY_KEYS):
        if base.get(key) != head.get(key):
            problems.append(f"summary 欄位 {key}: {base.get(key)!r} vs {head.get(key)!r}")
    return problems


def eval_report(stdout_text: str) -> list[str]:
    """取出 stdout 中評估報表標記行（含）以下的所有行。"""
    lines = stdout_text.splitlines()
    for i, line in enumerate(lines):
        if CYL_EVAL_REPORT_MARKER in line:
            return [ln.rstrip() for ln in lines[i:]]
    raise EvalReportError(
        f"stdout 內找不到評估報表標記 {CYL_EVAL_REPORT_MARKER!r}——"
        "跑掛、輸出被截斷，或標記行被改名；不得當作「無差異」")


def eval_report_diff(base_stdout: str, head_stdout: str) -> list[str]:
    """比對 cylinder 的評估報表區塊（cylinder 的 finalize 不落盤，只 print）。"""
    a, b = eval_report(base_stdout), eval_report(head_stdout)
    problems: list[str] = []
    if len(a) != len(b):
        problems.append(f"報表行數 {len(a)} vs {len(b)}")
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    for i, x, y in diffs[:_MAX_REPORTED]:
        problems.append(f"報表第 {i} 行: {x!r} vs {y!r}")
    # 截斷要說出來：只印前幾處而不註明，讀起來就像「只有這幾處不同」。
    if len(diffs) > _MAX_REPORTED:
        problems.append(f"（尚有 {len(diffs) - _MAX_REPORTED} 處分岔未列出）")
    return problems


def _walk(node, path="", *, skip):
    """把巢狀 JSON 攤成 {路徑: 葉值}；被排除的鍵不進入比對面。"""
    flat: dict[str, object] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if key in skip:
                continue
            flat.update(_walk(value, f"{path}.{key}" if path else key, skip=skip))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            flat.update(_walk(value, f"{path}[{i}]", skip=skip))
    else:
        flat[path] = node
    return flat


def _is_nan(value) -> bool:
    return isinstance(value, float) and value != value


def metrics_projection_skipped(payload, *, skip=VOLATILE_METRIC_KEYS) -> list[str]:
    """回報「這次比對放過了哪些欄位」。沉默的排除讀起來像全部都比過了。"""
    found: set[str] = set()

    def _scan(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in skip:
                    found.add(key)
                else:
                    _scan(value)
        elif isinstance(node, list):
            for value in node:
                _scan(value)

    _scan(payload)
    return [f"未比對（跨 run 必然不同）：{k}" for k in sorted(found)]


def metrics_projection_diff(base, head, *, tolerance: float = 1e-12,
                            skip=VOLATILE_METRIC_KEYS) -> list[str]:
    """逐鍵比對兩份評估投影：數值走容差、其餘走全等、結構不符一律報。

    存在理由與 `summary_diff` 相同，但標的不同：那個比訓練 summary 的全等，
    這個比評估產物，且必須容忍加總順序造成的 ~1e-16 位移——遷移把
    `sum()/len()` 換成 `np.mean`，不給容差就永遠驗不過，給太多就驗不到東西。

    `NaN → null` 單獨成一類：那是本次刻意的變更（舊路徑寫出的字面 NaN 不是
    合法 JSON），但不得靜默吞掉，否則它與「數字動了」無法分辨。
    """
    a, b = _walk(base, skip=skip), _walk(head, skip=skip)
    problems: list[str] = []

    if not a and not b:
        problems.append(
            "比對面為空——扣掉被排除的欄位後沒有任何東西可比，"
            "回報通過不代表任何事")
        return problems

    for key in sorted(set(a) - set(b)):
        problems.append(f"head 缺欄位 {key}（base={a[key]!r}）")
    for key in sorted(set(b) - set(a)):
        problems.append(f"base 缺欄位 {key}（head={b[key]!r}）")

    moved = 0
    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        if _is_nan(x) and y is None:
            problems.append(f"{key}: {_NAN_TO_NULL}（本次刻意的變更，需人眼確認範圍）")
            continue
        if isinstance(x, bool) != isinstance(y, bool):
            problems.append(f"{key}: 型別不同 {type(x).__name__} vs {type(y).__name__}")
            continue
        if (isinstance(x, (int, float)) and isinstance(y, (int, float))
                and not isinstance(x, bool)):
            if _is_nan(x) and _is_nan(y):
                continue
            if _is_nan(x) or _is_nan(y) or abs(x - y) > tolerance:
                moved += 1
                if moved <= _MAX_REPORTED:
                    problems.append(f"{key}: {x!r} vs {y!r}（差 {abs(x - y):.3e} > {tolerance:.0e}）")
                continue
            continue
        if type(x) is not type(y):
            problems.append(f"{key}: 型別不同 {type(x).__name__} vs {type(y).__name__}")
        elif x != y:
            problems.append(f"{key}: {x!r} vs {y!r}")

    # 截斷要說出來：只印前幾處而不註明，讀起來就像「只有這幾處不同」。
    if moved > _MAX_REPORTED:
        problems.append(f"（尚有 {moved - _MAX_REPORTED} 處數值分岔未列出）")
    return problems


# ─── CLI ───────────────────────────────────────────────────────────────────

def _load_json(path: Path):
    return json.loads(path.read_text())


def _cmd_determinism(a: argparse.Namespace) -> int:
    problems = ledger_verdict(_load_json(Path(a.a)), _load_json(Path(a.b)))
    if problems:
        print("  同碼兩跑不一致：")
        for p in problems:
            print(f"    - {p}")
        print("  === CPU 亦非決定性 → 位元對拍不可行，後續不必跑 ===")
        return 2
    print("  === CPU 可重現（排程 + 兩個 digest 全等）→ 可作為判準 ===")
    return 0


def _cmd_metrics(a: argparse.Namespace) -> int:
    base, head = _load_json(Path(a.base)), _load_json(Path(a.head))
    skipped = metrics_projection_skipped(base)
    for note in skipped:
        print(f"  {note}")
    problems = metrics_projection_diff(base, head, tolerance=a.tolerance)

    if a.evidence:
        # 證據必須是機器產出的：手寫的結論無法複驗，那正是
        # knowledge/superpowers/evidence/README.md 要避免的東西。
        out = Path(a.evidence)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "job_id": a.job_id, "node": a.node, "recorded_utc": a.recorded_utc,
            "base_sha": a.base_sha, "head_sha": a.head_sha,
            "base_path": str(Path(a.base)), "head_path": str(Path(a.head)),
            "tolerance": a.tolerance,
            "skipped": skipped,
            "problems": problems,
            "ok": not problems,
        }, indent=2, ensure_ascii=False) + "\n")
        print(f"  [evidence] {out}")

    if problems:
        print(f"  投影不一致（容差 {a.tolerance:.0e}）：")
        for p in problems:
            print(f"    - {p}")
        return 2
    print(f"  === 投影逐鍵一致（容差 {a.tolerance:.0e}）===")
    return 0


def _cmd_ab(a: argparse.Namespace) -> int:
    if not (a.ledger_rel or a.summary_rel or a.stdout_rel):
        # 三種標的都沒給 → 每個 fixture 零比對卻回報通過，是最徹底的假綠。
        print("  [ERR] 未指定任何比對面（--ledger-rel / --summary-rel / --stdout-rel）"
              "——這樣的「通過」不代表任何事")
        return 3
    base_root, head_root = Path(a.base_root), Path(a.head_root)
    evidence: dict = {
        "job_id": a.job_id, "node": a.node, "recorded_utc": a.recorded_utc,
        "base_sha": a.base_sha, "head_sha": a.head_sha,
        "expect_base": a.expect_base, "expect_head": a.expect_head,
        "commit_problems": commit_check(a.base_sha, a.head_sha,
                                        a.expect_base, a.expect_head),
        "fixtures": {},
    }
    if evidence["commit_problems"]:
        print("  比對雙方的 commit 未通過斷言：")
        for p in evidence["commit_problems"]:
            print(f"    - {p}")

    for name in a.names.split(","):
        problems: list[str] = []
        rec: dict = {}
        if not a.ledger_rel:
            bl = hl = None
        else:
            bl = base_root / a.ledger_rel.format(name=name)
            hl = head_root / a.ledger_rel.format(name=name)
        if bl is None:
            pass                                   # ledger 關閉的組態，標的是落盤產物
        elif not (bl.exists() and hl.exists()):
            problems.append(f"ledger 缺失 base={bl.exists()} head={hl.exists()}")
        else:
            b_rows, h_rows = _load_json(bl), _load_json(hl)
            problems += ledger_verdict(b_rows, h_rows)
            # 兩側都記：通過時兩者相等；不通過時讀者不必翻 problems 就看得到分岔的值。
            for side, rows in (("base", b_rows), ("head", h_rows)):
                tail = split_ledger(rows)[1]
                rec[f"{side}_params_digest"] = tail.get("params_digest")
                rec[f"{side}_opt_state_digest"] = tail.get("opt_state_digest")

        if a.summary_rel:
            bs = base_root / a.summary_rel.format(name=name)
            hs = head_root / a.summary_rel.format(name=name)
            if not (bs.exists() and hs.exists()):
                problems.append(f"summary 缺失 base={bs.exists()} head={hs.exists()}")
            else:
                problems += summary_diff(_load_json(bs), _load_json(hs))

        if a.stdout_rel:
            bo = base_root / a.stdout_rel.format(name=name)
            ho = head_root / a.stdout_rel.format(name=name)
            if not (bo.exists() and ho.exists()):
                problems.append(f"stdout 缺失 base={bo.exists()} head={ho.exists()}")
            else:
                try:
                    problems += eval_report_diff(bo.read_text(), ho.read_text())
                except EvalReportError as e:
                    problems.append(str(e))

        rec["problems"] = problems
        rec["ok"] = not problems
        evidence["fixtures"][name] = rec
        print(f"  {name}: {'BIT-IDENTICAL' if rec['ok'] else 'MISMATCH'}")
        for p in problems:
            print(f"     - {p}")

    evidence["ok"] = (not evidence["commit_problems"]
                      and all(r["ok"] for r in evidence["fixtures"].values()))
    if a.evidence:
        out = Path(a.evidence)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
        print(f"  證據已寫入 {out}")
    return 0 if evidence["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("determinism", help="同一份程式碼兩跑的 ledger 比對")
    d.add_argument("--a", required=True)
    d.add_argument("--b", required=True)
    d.set_defaults(fn=_cmd_determinism)

    b = sub.add_parser("ab", help="BASE vs HEAD 的逐 fixture 對拍")
    b.add_argument("--base-root", required=True)
    b.add_argument("--head-root", required=True)
    b.add_argument("--names", required=True, help="逗號分隔，如 c1,c2,c3")
    b.add_argument("--ledger-rel", default="",
                   help="含 {name} 佔位的相對路徑；ledger 關閉的組態下留空")
    b.add_argument("--summary-rel", default="", help="Kolmogorov 的 summary.json")
    b.add_argument("--stdout-rel", default="", help="cylinder 的 stdout 存檔")
    b.add_argument("--base-sha", required=True)
    b.add_argument("--head-sha", required=True)
    b.add_argument("--expect-base", required=True)
    b.add_argument("--expect-head", default="")
    b.add_argument("--evidence", default="")
    b.add_argument("--job-id", default="")
    b.add_argument("--node", default="")
    b.add_argument("--recorded-utc", default="")
    b.set_defaults(fn=_cmd_ab)

    m = sub.add_parser("metrics", help="兩份評估投影的逐鍵比對（候選 2 的驗收）")
    m.add_argument("--base", required=True, help="遷移前的 metrics JSON")
    m.add_argument("--head", required=True, help="本次寫到新路徑的 metrics JSON")
    m.add_argument("--tolerance", type=float, default=1e-12)
    m.add_argument("--evidence", default="", help="落一份機器可讀結論供複驗")
    m.add_argument("--base-sha", default="")
    m.add_argument("--head-sha", default="")
    m.add_argument("--job-id", default="")
    m.add_argument("--node", default="")
    m.add_argument("--recorded-utc", default="")
    m.set_defaults(fn=_cmd_metrics)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

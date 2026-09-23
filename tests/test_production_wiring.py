"""釘住「守衛與旗標有沒有真的接在生產路徑上」—— 獨立複查（2026-09-10）發現的四個缺口。

單元測試把每個守衛函式本身測得很紮實，但**沒有任何東西在看它們有沒有被呼叫**。
實測：把 `assert_soap_betas_are_used` / `_assert_declared_npz_matches` /
`assert_al_statics_match` 三處生產呼叫全換成 `pass`，全套 1209 passed。
函式活著、生產不呼叫，就是零效果。

同一個病的另外三處：`build_optimizer` 的裁剪只有 helper 被測（拿掉三個呼叫點的
`_wrap_chain` 仍全綠）、schedule-free 下 `adam_b1=0.0` 沒有觀測點（只補了 soap 那半，
而 adam 是 soap 不可用時實際會走的路）、以及 `gn_ref_path` 的字串本身。

**為什麼用 AST 而不是端到端**：這三個守衛分屬 `_load_datasets` / `build_context` /
`restore`，端到端都要真資料與 GPU ckpt，本機跑不了（CLAUDE.md §6）。要釘的東西是
「呼叫點存在於那個函式裡」這個結構事實——那正是會被無聲刪掉的東西。用 AST 而非
grep 行號，是為了不隨行號漂移而假綠。
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1] / "pi_lnn_jax"


def _fn(module_rel: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((ROOT / module_rel).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{module_rel} 裡找不到函式 {name}——重構過就要更新本測試")


def _calls(fn: ast.AST) -> set[str]:
    out = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


# ── 1. 三個守衛的生產呼叫點 ────────────────────────────────────────────────

@pytest.mark.parametrize("module_rel,enclosing,guard", [
    ("pipeline/kolmogorov/assembly.py", "_load_datasets", "_assert_declared_npz_matches"),
    ("pipeline/kolmogorov/assembly.py", "build_context", "assert_soap_betas_are_used"),
    ("pipeline/kolmogorov/assembly.py", "build_context", "assert_rar_pool_is_large_enough"),
    ("pipeline/kolmogorov/run.py", "restore", "assert_al_statics_match"),
])
def test_guard_is_actually_called_in_production(module_rel, enclosing, guard):
    """守衛函式被測得再紮實，沒有人呼叫它就是零效果。"""
    assert guard in _calls(_fn(module_rel, enclosing)), (
        f"{module_rel}::{enclosing} 不再呼叫 {guard}——守衛被摘掉了。"
        "若這是刻意移除，請一併說明為什麼那個靜默失效不再需要擋。")


# ── 2. 裁剪接在 build_optimizer 的每一條回傳路徑上 ─────────────────────────

def test_every_optimizer_path_wraps_the_clip():
    """`_wrap_chain` 存在不代表它被用上。三條回傳路徑（soap / schedule_free / 其他）
    都必須經過它，否則 `max_grad_norm` 是裝飾性參數。"""
    fn = _fn("optimizers.py", "build_optimizer")
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert returns, "build_optimizer 沒有 return？"
    n_wrap = sum(1 for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "_wrap_chain")
    assert n_wrap >= 3, (
        f"build_optimizer 只有 {n_wrap} 處呼叫 _wrap_chain，預期每條 optimizer 路徑各一。"
        "少一處就是那條路徑的梯度裁剪沒接上，而現有測試只驗 helper 本身。")


def test_schedule_free_zeroes_both_inner_first_moments():
    """schedule-free 下內層一階動量必須為 0（否則動量套兩次）。

    原始缺口是「inner momentum 0.0→0.9 全綠」；先前只補了 soap 那半（觀測點是
    `opt_info["name"]` 的字串），**adam 那半仍無保護**——而 adam 正是 soap
    不可用時實際會走的路。這裡直接釘住原始碼裡的兩個 0.0。
    """
    fn = _fn("optimizers.py", "build_optimizer")
    zeroed = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in ("soap_b1", "adam_b1") and isinstance(kw.value, ast.Constant):
                    if float(kw.value.value) == 0.0:
                        zeroed.add(kw.arg)
    assert zeroed == {"soap_b1", "adam_b1"}, (
        f"schedule-free 分支只把 {sorted(zeroed) or '（無）'} 設成 0.0；"
        "兩個內層 optimizer 的一階動量都必須歸零，否則 Polyak 平均與動量疊加。")


# ── 3. GradNorm 的 ref path 字串 ───────────────────────────────────────────

def test_gradnorm_ref_path_literals_are_pinned():
    """釘住 `assembly.py` **實際選用**的 ref path，而不是模型有沒有那一層。

    先前的 tripwire 斷言的是「`StandardPINNOperator` 的參數樹裡沒有 `trunk_out`」。
    那條在 A9 被修好時**不會變紅**——真正的修法落在這裡的路徑選擇。而且把 liquid
    那條打成 `"trunk_ou"` 時 `_get_subtree` 會靜默 fallback 到整棵 params（正是 A9
    的缺陷本身），全庫也沒有測試會紅。

    改動這兩個字串是行為變更（見 technical-debt.md TD-2 與 model-audit §8.4：
    兩條候選 path 的 `G_phys/G_data` 差 86× vs 40×），必須走 §7.1 的 A/B 對拍。
    """
    tree = ast.parse((ROOT / "pipeline/kolmogorov/assembly.py").read_text())
    val = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "gn_ref_path" for t in node.targets):
            val = node.value
            break
    assert val is not None, "assembly.py 裡找不到 gn_ref_path 的賦值"
    # **兩個分支要分開檢查。** 只斷言「字串裡有 trunk_out」會被 liquid 那一支滿足，
    # 於是 A9 修好（非 liquid 改走別的層）時測試照樣綠——那正是先前 tripwire 的毛病，
    # 也是本測試第一版犯過的同一個錯。
    assert isinstance(val, ast.IfExp), (
        f"gn_ref_path 不再是 arch 三元式而是 {ast.unparse(val)}；"
        "改了路徑選擇的結構就要重新檢視本測試與 §7.1 對拍。")
    liquid = ast.literal_eval(val.body)
    other = ast.literal_eval(val.orelse)
    assert liquid == ("query_decoder", "trunk_out"), (
        f"liquid 的 GradNorm ref path 變成 {liquid}。所有歷史 production run 都以 "
        "('query_decoder','trunk_out') 訓練；改它是行為變更，需 §7.1 對拍。")
    assert other == ("trunk_out",), (
        f"非 liquid 的 ref path 變成 {other}。若這是 A9 的修正（`StandardPINNOperator` "
        "沒有 `trunk_out`，現況是靜默 fallback 到整棵 params），請一併更新 "
        "model-audit A9/TD-32、test_core_block_semantics 的 pinn tripwire，"
        "並重跑 PINN 兩個預算的五 seed——tab:pinn_budget_comparison 會變。")


# ── 5. RAR 的 per-point 殘差必須保持 jit ──────────────────────────────────

def test_rar_residual_stays_jitted():
    """拿掉 `@jax.jit` 不會讓任何測試變紅，只會讓 RAR 慢 12 倍以上。

    job 5710 實測：未 jit 時 `rar_freq=1`、pool=4096 下 18 分鐘走不完 500 步，
    而同時同節點的對照臂已到 5500 步——每個 RAR 步都在 eager 模式逐 op 執行完整的
    encode + folx 二階殘差。那是**效能懸崖不是錯誤**，所以沒有任何功能測試會發現它；
    而 RAR 又是預設關閉的路徑，實際跑之前不會有人踩到。
    """
    fn = _fn("pipeline/kolmogorov/run.py", "_rar_residual_per_point")
    decos = {ast.unparse(d) for d in fn.decorator_list}
    assert any("jit" in d for d in decos), (
        f"_rar_residual_per_point 的裝飾器是 {decos or '（無）'}——少了 jax.jit。"
        "見本測試 docstring：那不會讓任何功能測試變紅，只會讓 RAR 慢到不可用。")


# ── 6. resume 守衛必須拿到子取樣索引 ─────────────────────────────────────

def test_resume_sanity_check_receives_a_subsample():
    """`sensor_idx` 寫死 `None` 會讓 resume 在生產規模上 RESOURCE_EXHAUSTED。

    那是這道守衛先前從未被任何 production run 走過的原因（lab-server 955 份 job log
    裡只有兩支診斷出現 `[RESUME] restored`），所以「訓練中斷可以續跑」一直是未驗證
    的能力。函式簽章接了 `sensor_idx` 不代表呼叫端有傳——這裡釘的是呼叫端。
    """
    fn = _fn("pipeline/kolmogorov/run.py", "restore")
    for call in ast.walk(fn):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "_resume_sanity_check"):
            kws = {k.arg: k.value for k in call.keywords}
            assert "sensor_idx" in kws, (
                "restore 呼叫 _resume_sanity_check 時沒傳 sensor_idx——"
                "會退回全量 T*K decode，見本測試 docstring")
            assert "deterministic_sensor_idx" in ast.unparse(kws["sensor_idx"]), (
                f"sensor_idx 傳的是 {ast.unparse(kws['sensor_idx'])!r}，"
                "不是 deterministic_sensor_idx——兩側要算同一個量")
            return
    raise AssertionError("restore 裡找不到 _resume_sanity_check 的呼叫")


def test_run_outputs_are_written_before_the_optional_resume_diagnostic():
    """summary.json 必須在 resume_reference.json 之前落盤。

    2026-09-18 實測：`_resume_reference_sensor_loss` 在 K=400 上 OOM
    （cross-attention 的 `[N, K, 1, 256]` 隨 K 線性放大），而它當時寫在
    summary.json **之前**——於是一個「下一輪 resume 才會用到的診斷值」把六支
    已經跑完 20000 步的訓練的 provenance 一起帶走。jobs 5824–5829 的
    artifacts 只剩 checkpoints，summary.json / eval_history.json 全缺。

    釘住的結構事實是**順序**：run 自己的產物先寫，可有可無的診斷後寫。
    診斷本身失敗只該讓它自己缺值（那條路由下一個測試釘），不該讓結果消失。
    """
    src = pathlib.Path("pi_lnn_jax/pipeline/kolmogorov/run.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "finalize")

    def first_line_mentioning(needle: str) -> int:
        hits = [n.lineno for n in ast.walk(fn)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and needle in n.value]
        assert hits, f"finalize 裡找不到寫 {needle} 的地方"
        return min(hits)

    summary_at = first_line_mentioning("summary.json")
    ref_at = first_line_mentioning("resume_reference.json")
    assert summary_at < ref_at, (
        f"summary.json 在第 {summary_at} 行、resume_reference.json 在第 {ref_at} 行——"
        "順序反了。診斷若拋錯，run 的結果就跟著沒了（jobs 5824–5829 就是這樣掉的）")


def test_resume_diagnostic_failure_cannot_abort_finalize():
    """resume 參考值的計算必須被接住，且把原因寫下來，不是靜默吞掉。

    這是 `try/except` 的合法用法與非法用法的分界：它包住的是一個**可有可無的
    診斷**，而且發生時要留下 `unavailable_reason`。包住會產生結果的計算才是吞錯誤。
    """
    src = pathlib.Path("pi_lnn_jax/pipeline/kolmogorov/run.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "finalize")
    guarded = [
        t for t in ast.walk(fn) if isinstance(t, ast.Try)
        and any(isinstance(c, ast.Call) and getattr(c.func, "id", "") ==
                "_resume_reference_sensor_loss" for c in ast.walk(t.body[0] if t.body else t))
    ]
    assert guarded, (
        "_resume_reference_sensor_loss 沒有被 try 包住——它在大 K 上會 OOM，"
        "而它只是下一輪 resume 的參考值，不該有能力中止 finalize")
    reasons = [n.value for t in guarded for n in ast.walk(t)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and "unavailable_reason" in n.value]
    assert reasons, "接住之後必須寫下 unavailable_reason，否則就是靜默吞掉"

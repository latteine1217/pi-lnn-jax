"""cylinder 的 RNG/schedule ledger 埋點 —— 結構與 schema 哨兵。

埋點原本在 `train_cylinder.py`，wave 2 隨執行期一起搬進
`pi_lnn_jax/pipeline/cylinder/run.py`；本檔的掃描對象因此換成該檔，測的三件事
不變（契約是 bit-identical，cylinder 有論文證據綁著）：

  1. `data_meta` 的欄位集恰好覆蓋 `_plan_step` 讀到的資料衍生量（schema 不得與需求漂移）。
  2. production 路徑上每一個 ledger 呼叫點都在 `if ledger is not None:` 之內
     （gate 關閉 = 零額外工作）。
  3. 埋點沒有改動 RNG 消費時序（spec §5.3 的無條件 4-way split）。

逐步軌跡與 golden fixture 的實際比對在 `tests/test_cylinder_replay.py`；本檔只
守結構，不做重播。本機禁跑 training，故全部以 AST + 純函式驗證。
"""
from __future__ import annotations

import ast

import numpy as np
import pytest
from _paths import REPO_ROOT

_RUN_PY = REPO_ROOT / "pi_lnn_jax" / "pipeline" / "cylinder" / "run.py"
_SRC = _RUN_PY.read_text()

#: `_plan_step` 讀到的**資料衍生量**——每一個都必須能由尾列 data_meta 逐位元重建。
PLAN_STEP_READS_DATA = {
    "st0",          # float(st[0])
    "tm_end",       # float(st[-1])
    "T_total",      # float(st[-1] - st[0])（float32 相減，見下方 t_total 測試）
    "T",            # sensor_vals.shape[1]
    "K",            # sensor_vals.shape[0]
    "t_q_full_np",  # broadcast(st) → [T*K]
    "geom",         # CylinderGeometry / OscillatingGeometry
}

#: `_plan_step` 讀到、但**由 data_meta + config 重算**的量。`n_sensor_q` 是
#: `min(config 值, T*K)`：config 部分從尾列 argv 重解、上界來自 data_meta 的
#: sensor_time/n_sensors，兩邊都不是猜的，故刻意不另外錄一份（多一份真相就多一個
#: 打架來源）。clamp 那一行在 assembly.build_context，replay 端在
#: `_ctx_from_data_meta` 重算——兩處必須同步，本測試是提醒點。
PLAN_STEP_READS_DERIVED = {"n_sensor_q"}

#: `_plan_step` 讀到的 typed effective config 入口。
PLAN_STEP_READS_CONFIG = {"config"}

#: `run_loop` 自己（`_plan_step` 之外）讀到的 ctx 欄位。兩個 callable 不是可重播
#: 的值：replay 本來就不會呼叫它們（不做 forward/gradient）。這條把「資料衍生量
#: 偷偷在 run_loop 直接讀」擋掉——那種讀取 replay 覆蓋不到。
RUN_LOOP_READS_CTX = {"config", "step_fn", "grad_norm_fn"}

#: data_meta 欄位 → 它負責重建的 `_plan_step` 讀取點。這張表就是「schema 為何長這樣」
#: 的唯一說明，測試據此雙向核對（欄位不多不少、覆蓋不缺不溢）。
DATA_META_COVERS = {
    "sensor_time": {"st0", "tm_end", "T", "t_q_full_np"},
    "n_sensors": {"K", "t_q_full_np"},
    "t_total": {"T_total"},
    "geom": {"geom"},
}

#: `_data_meta` 自己讀到的 ctx 欄位（錄製端），與上表的 key 一一對應。
DATA_META_READS_CTX = {"st", "K", "T_total", "geom"}


# ─────────────────────────────────────────────────────────────────────────────
# AST 工具
# ─────────────────────────────────────────────────────────────────────────────

def _func(name: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(ast.parse(_SRC))
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _ctx_reads(func: ast.FunctionDef) -> set:
    """函式內所有 `ctx.<attr>` 的屬性名。"""
    return {n.attr for n in ast.walk(func)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == "ctx"}


def _static_geom():
    from pi_lnn_jax.boundary import CylinderGeometry
    return CylinderGeometry(body_center=(0.25, 0.5), body_radius=0.05,
                            Lx=0.6, Ly=0.3, u_inf=0.33)


def _ctx_with(st, K, T_total, geom):
    """只填 `_data_meta` 會讀到的欄位；其餘留 None（讀到別的就該炸）。"""
    from pi_lnn_jax.pipeline.cylinder.assembly import TrainingContext
    return TrainingContext(**dict.fromkeys(TrainingContext._fields))._replace(
        st=st, K=K, T_total=T_total, geom=geom,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. schema 反漂移
# ─────────────────────────────────────────────────────────────────────────────

def test_data_meta_covers_exactly_the_plan_step_data_reads():
    """反漂移哨兵：`_data_meta` 的欄位集必須恰好對上 `_plan_step` 讀到的資料衍生量。

    Why: `data_meta` 存在的理由就是「replay 端不必猜 `_plan_step` 要什麼」。若哪天
    `_plan_step` 多讀一個 ctx 欄位而沒人動 `_data_meta`，replay 會拿到 None 然後炸在
    一個看不出原因的地方（或更糟，拿到能跑的預設值）。這條逼作者當場決定：新欄位是
    資料衍生（要錄）、config 衍生（從 argv 重解）還是兩者合成（重算）。
    """
    from pi_lnn_jax.pipeline.cylinder.run import _data_meta

    reads = _ctx_reads(_func("_plan_step"))
    declared = PLAN_STEP_READS_DATA | PLAN_STEP_READS_DERIVED | PLAN_STEP_READS_CONFIG
    assert reads == declared, (
        f"_plan_step 讀取的 ctx 欄位變了：多={sorted(reads - declared)} "
        f"少={sorted(declared - reads)} —— 請判定新欄位屬資料衍生（加進 "
        f"PLAN_STEP_READS_DATA 並同步 _data_meta / _ctx_from_data_meta / "
        f"DATA_META_COVERS）、config 衍生（PLAN_STEP_READS_CONFIG）還是重算量"
        f"（PLAN_STEP_READS_DERIVED）"
    )

    covered = set().union(*DATA_META_COVERS.values())
    assert covered == PLAN_STEP_READS_DATA, (
        f"DATA_META_COVERS 與 PLAN_STEP_READS_DATA 對不上：未覆蓋="
        f"{sorted(PLAN_STEP_READS_DATA - covered)} 多餘={sorted(covered - PLAN_STEP_READS_DATA)}"
    )
    assert _ctx_reads(_func("_data_meta")) == DATA_META_READS_CTX

    meta = _data_meta(_ctx_with(np.linspace(0.0, 1.0, 4, dtype=np.float32), 3, 1.0,
                                _static_geom()))
    assert set(meta) == set(DATA_META_COVERS), (
        f"_data_meta 的欄位集變了：{sorted(set(meta) ^ set(DATA_META_COVERS))}"
    )
    print(f"✓ data_meta_covers_exactly_the_plan_step_data_reads "
          f"({len(reads)} 讀取點 / {len(meta)} 欄位)")


def test_run_loop_reads_no_data_derived_ctx_field_outside_plan_step():
    """迴圈本體不得自己讀資料衍生量——那種讀取 `replay_schedule` 覆蓋不到。"""
    reads = _ctx_reads(_func("run_loop"))
    assert reads == RUN_LOOP_READS_CTX, (
        f"run_loop 讀取的 ctx 欄位變了：多={sorted(reads - RUN_LOOP_READS_CTX)} "
        f"少={sorted(RUN_LOOP_READS_CTX - reads)} —— 資料衍生量一律經 _plan_step，"
        f"否則 replay 測不到"
    )
    print(f"✓ run_loop_reads_no_data_derived_ctx_field_outside_plan_step ({sorted(reads)})")


def test_data_meta_is_json_serialisable_for_both_geometries():
    """兩種 geometry 都要能原樣落進 JSON（尾列會被 json.dump）。"""
    import json

    from pi_lnn_jax.boundary import OscillatingGeometry
    from pi_lnn_jax.pipeline.cylinder.run import _data_meta

    moving = OscillatingGeometry(
        base_center=(0.25, 0.5), body_radius_phys=0.015, Lx=0.6, Ly=0.3,
        u_inf=0.33, amp=0.01, freq=2.5, phase=0.3, axis=(0.0, 1.0),
    )
    for geom in (_static_geom(), moving):
        meta = _data_meta(_ctx_with(np.linspace(0.0, 1.0, 4, dtype=np.float32), 3, 1.0, geom))
        rt = json.loads(json.dumps(meta))
        assert rt["geom"] == meta["geom"]
        assert set(rt["geom"]) == set(geom._fields), (
            "geom 欄位必須完整攤平；缺欄位會讓 replay 重建出不同的 wall BC 取樣點"
        )
    print("✓ data_meta_is_json_serialisable_for_both_geometries")


def test_sensor_time_and_n_sensors_reconstruct_t_q_full_exactly():
    """`t_q_full`（[T*K]）不入 ledger 的理由要能被證明，而不是被宣稱。

    `_plan_step` 的 time-marching 分支算 `valid = np.nonzero(t_q_full_np <= t_max)`；
    `t_q_full = broadcast(st[:,None], (T,K)).reshape(T*K)` 就是「st 每個元素重複
    K 次」，故 `sensor_time` + `n_sensors` 已足以逐位元重建。

    **重建必須用 float32**：`t_q_full_np` 是 float32，而 `t_max` 是 Python float。
    NumPy 的 weak-scalar 提升（NEP 50）會把該純量降成 float32 再比較，因此同一個
    `t_max` 在 float64 重建上會給出不同的 `valid`（下方第二段直接把這個坑釘住）。
    """
    import jax.numpy as jnp

    from pi_lnn_jax.pipeline.cylinder.run import _data_meta

    st = jnp.asarray(np.linspace(0.1, 20.05, 37), jnp.float32)
    T, K = st.shape[0], 5
    t_q_full_np = np.asarray(
        jnp.broadcast_to(st[:, None], (T, K)).reshape(T * K)
    )

    meta = _data_meta(_ctx_with(st, K, float(st[-1] - st[0]), _static_geom()))
    recon = np.repeat(np.asarray(meta["sensor_time"], dtype=np.float32),
                      meta["n_sensors"])
    assert recon.shape == t_q_full_np.shape
    assert recon.dtype == t_q_full_np.dtype
    assert np.array_equal(recon, t_q_full_np), "重建非逐位元相同"
    assert len(meta["sensor_time"]) == T, "sensor_time 長度即 T（`_plan_step` 的 T*K 上界）"

    edges = (0.1, 5.0, 12.34, float(st[-1]), float(st[len(st) // 2]))
    for t_max in edges:
        assert np.array_equal(np.nonzero(recon <= t_max)[0],
                              np.nonzero(t_q_full_np <= t_max)[0]), (
            f"t_max={t_max} 的 valid 索引不同 —— sensor_time+K 不足以重建"
        )
    # dtype 坑的存證：float64 重建在邊界 t_max 上會給出不同的 valid。
    wrong = np.repeat(np.asarray(meta["sensor_time"], dtype=np.float64),
                      meta["n_sensors"])
    assert any(
        not np.array_equal(np.nonzero(wrong <= t)[0], np.nonzero(t_q_full_np <= t)[0])
        for t in edges
    ), (
        "float64 重建竟與 float32 一致 —— NumPy 純量提升語意可能變了；"
        "若確定不再降階比較，本測試的 dtype 警告可放寬"
    )
    print("✓ sensor_time_and_n_sensors_reconstruct_t_q_full_exactly")


def test_t_total_is_recorded_not_derived_from_endpoints():
    """`t_total` 必須是錄下來的，不能讓 replay 端用兩個端點相減。

    assembly 用 `float(st[-1] - st[0])`：float32 相減後放大到 float64。replay 端若拿
    兩個已放大的 float64 端點相減會得到「精確差」，兩者可差 1 ulp——而它正是
    非 time-marching 路徑 `t_max = st0 + T_total` 的來源，歪掉會讓 ct 的取樣區間、
    進而整條 digest 全錯。下面用真實 linspace 時間軸證明這個差確實存在。
    """
    import jax.numpy as jnp

    from pi_lnn_jax.pipeline.cylinder.run import _data_meta

    st = jnp.asarray(np.linspace(0.1, 20.05, 201), jnp.float32)
    loop_value = float(st[-1] - st[0])
    meta = _data_meta(_ctx_with(st, 4, loop_value, _static_geom()))
    naive = meta["sensor_time"][-1] - meta["sensor_time"][0]
    assert meta["t_total"] == loop_value
    assert naive != loop_value, (
        "本測試的前提（端點相減會失真）在此資料上不成立，換一組時間軸重驗；"
        "若真的永遠相等，t_total 欄位才可以拿掉"
    )
    print(f"✓ t_total_is_recorded_not_derived_from_endpoints "
          f"(Δ={abs(naive - loop_value):.3e})")


# ─────────────────────────────────────────────────────────────────────────────
# 2. gate 紀律
# ─────────────────────────────────────────────────────────────────────────────

#: 只能出現在 gate 內的呼叫。`get_ledger` 不在其中——它就是取得 gate 本身的那一步。
GATED_CALLS = {"_ledger_fields", "_data_meta", "digest", "params_digest"}
#: 這兩個 helper 的**函式體**不掃（它們只被 gate 內／重播路徑呼叫，由 call-site 掃描背書）。
LEDGER_HELPERS = {"_ledger_fields", "_data_meta"}
#: 重播路徑不掃：它是診斷工具，`run_training` 不會走到，本來就沒有 ledger 可 gate。
#: production 零 overhead 的主張只涵蓋 production 路徑。
REPLAY_FNS = {"replay_plans", "replay_schedule"}


def _is_ledger_gate(node) -> bool:
    """True 若 node 是 `if ledger is not None:`。"""
    if not isinstance(node, ast.If):
        return False
    t = node.test
    return (
        isinstance(t, ast.Compare)
        and isinstance(t.left, ast.Name) and t.left.id == "ledger"
        and len(t.ops) == 1 and isinstance(t.ops[0], ast.IsNot)
        and len(t.comparators) == 1
        and isinstance(t.comparators[0], ast.Constant)
        and t.comparators[0].value is None
    )


def _collect_ledger_calls(node, gated: bool, out: list) -> None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
        node.name in LEDGER_HELPERS or node.name in REPLAY_FNS
    ):
        return
    if isinstance(node, ast.Call):
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id if func.id in GATED_CALLS else None
        elif (isinstance(func, ast.Attribute)
              and isinstance(func.value, ast.Name) and func.value.id == "ledger"):
            name = f"ledger.{func.attr}"
        if name is not None:
            out.append((name, node.lineno, gated))
    if _is_ledger_gate(node):
        _collect_ledger_calls(node.test, gated, out)
        for st in node.body:
            _collect_ledger_calls(st, True, out)
        for st in node.orelse:          # else 分支不受 gate 保護
            _collect_ledger_calls(st, gated, out)
        return
    for child in ast.iter_child_nodes(node):
        _collect_ledger_calls(child, gated, out)


def test_every_ledger_call_site_is_gated():
    """gate 關閉時 production 路徑必須零額外工作。

    誠實話：這條是**結構**證據，不是執行證據——「gate 關掉時逐位元不變」只能靠
    一次真實訓練對拍，本機禁跑。這裡能保證的是：production 路徑上所有 ledger 相關
    呼叫都在 `if ledger is not None:` 之內，而 `get_ledger()` 在 gate 關閉時回 None
    （由 tests/test_pipeline_ledger.py::test_gate_is_off_by_default 釘住）。
    """
    calls: list = []
    _collect_ledger_calls(ast.parse(_SRC), False, calls)
    ungated = [(n, ln) for n, ln, g in calls if not g]
    assert not ungated, f"以下 ledger 呼叫不在 gate 內：{ungated}"
    assert calls, "找不到任何 ledger 呼叫點 —— 埋點是不是被移除了？"

    n_gates = sum(1 for n in ast.walk(ast.parse(_SRC)) if _is_ledger_gate(n))
    assert n_gates == 3, (
        f"預期 3 個 gate（initialize 記 init_params_digest、每步一個、尾列一個），"
        f"實得 {n_gates}。數量被釘住是為了讓「多出一個 gate」變成需要解釋的事——"
        "gate 內的東西不進 production 路徑，多一個就多一塊沒被 production 走過的碼。")
    assert {n for n, _, _ in calls} == {
        "ledger.record", "ledger.dump", "ledger.note",
        "_ledger_fields", "_data_meta", "params_digest",
    }
    print(f"✓ every_ledger_call_site_is_gated ({len(calls)} 呼叫 / {n_gates} gate)")


def test_ledger_fields_records_digests_not_arrays():
    """大陣列只進 digest；欄位值必須全是 JSON 純量（_ledger.py 紀律）。"""
    import json

    from pi_lnn_jax.pipeline.cylinder.run import _ledger_fields, _StepPlan

    arr = np.arange(8, dtype=np.float32)
    row = _ledger_fields(_StepPlan(
        t_max=1.25, cx=arr, cy=arr + 1, ct=arr + 2,
        sq_idx=np.arange(4, dtype=np.int32),
        bc_in=arr.reshape(4, 2), bc_body=arr.reshape(4, 2),
        bc_slip=arr.reshape(4, 2), bc_bvel=np.zeros((4, 2), np.float32),
        trig_gradnorm=True, trig_al=False, rk=None,
    ))
    assert set(row) == {
        "t_max", "cx", "cy", "ct", "sq_idx",
        "bc_in", "bc_body", "bc_slip", "bc_bvel",
        "trig_gradnorm", "trig_al",
    }
    json.dumps(row)   # 落盤走 json.dump；不可序列化就是把陣列漏進去了
    for k in ("cx", "cy", "ct", "sq_idx", "bc_in", "bc_body", "bc_slip", "bc_bvel"):
        assert isinstance(row[k], str) and len(row[k]) == 16, f"{k} 不是 16 字 digest"
    assert row["cx"] != row["cy"], "digest 必須對內容敏感"
    assert isinstance(row["t_max"], float)
    assert row["trig_gradnorm"] is True and row["trig_al"] is False
    # rk（推進後的 key）刻意不入 ledger：它是 plan 的 carry，不是該步的決策
    assert "rk" not in row
    print("✓ ledger_fields_records_digests_not_arrays")


# ─────────────────────────────────────────────────────────────────────────────
# 3. 埋點不得動到 RNG 消費時序
# ─────────────────────────────────────────────────────────────────────────────

def test_per_step_key_split_stays_unconditional():
    """spec §5.3：每步的 4-way / 3-way split 必須是 `_plan_step` 的直屬語句。

    `k3` 在 4-way split 就被抽出，time-marching 開啟時它**不被使用但已消耗**。
    任何「只在需要時才 split」的最佳化都會改變 `rk` 的後續狀態；埋點也不得把
    這兩行推進任何分支。
    """
    fn = _func("_plan_step")
    splits = [s for s in fn.body
              if isinstance(s, ast.Assign) and isinstance(s.value, ast.Call)
              and ast.unparse(s.value).startswith("jax.random.split(")]
    assert len(splits) == 2, f"直屬語句層的 split 應恰為 2 個，實得 {len(splits)}"
    split4, split3 = splits

    assert isinstance(split4.targets[0], ast.Tuple)
    assert [n.id for n in split4.targets[0].elts] == ["rk", "k1", "k2", "k3"]
    assert ast.unparse(split4.value) == "jax.random.split(rk, 4)"
    assert ast.unparse(split3) == "ks = jax.random.split(k1, 3)"

    # 全函式（含分支內）不得有第三處 split —— 條件式 split 會改動 rk 的後續狀態
    all_splits = [n for n in ast.walk(fn)
                  if isinstance(n, ast.Call)
                  and ast.unparse(n).startswith("jax.random.split(")]
    assert len(all_splits) == 2, (
        f"_plan_step 內有 {len(all_splits)} 處 jax.random.split —— "
        f"多出來的那一處若在分支裡，就會讓兩條路徑的 rk 分岔"
    )
    print("✓ per_step_key_split_stays_unconditional")


def test_wall_bc_branch_is_keyed_on_is_controlled_not_use_tm():
    """spec §5.3 第 5 點：wall BC 分支由 `is_controlled` 決定，**不是** `use_tm`。

    兩者是正交的（c2 是 use_tm 但靜態 BC、c3 兩者皆真），寫錯不會 crash，只會在
    controlled 以外的 case 靜靜換掉 BC 取樣路徑。
    """
    fn = _func("_plan_step")
    tests = [ast.unparse(s.test) for s in fn.body if isinstance(s, ast.If)]
    assert tests.count("is_controlled") == 1, (
        f"_plan_step 應恰有一個 `if is_controlled:` 分支（wall BC），實得 {tests}"
    )
    moving = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name) and n.func.id == "sample_wall_bc_moving"]
    static = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name) and n.func.id == "sample_wall_bc"]
    assert len(moving) == len(static) == 1
    # 兩路都必須消費 k2（key 位置為第一個 positional arg）
    assert ast.unparse(moving[0].args[0]) == "k2"
    assert ast.unparse(static[0].args[0]) == "k2"
    print("✓ wall_bc_branch_is_keyed_on_is_controlled_not_use_tm")


@pytest.mark.parametrize(
    "name,freq_var", [("trig_gradnorm", "gradnorm_freq"), ("trig_al", "al_update_freq")]
)
def test_trigger_boolean_is_computed_once(name, freq_var):
    """觸發布林只能算一次：ledger 與控制流讀同一個值。

    Why: 若 ledger 自行再算一次條件，兩份會漂移，ledger 就會安靜說謊——錄到的
    軌跡與實際跑的不是同一回事，而這種 oracle 比沒有 oracle 更危險。搬移後這個
    「唯一一次」落在 `_plan_step`，`run_loop` 與 `_ledger_fields` 都只讀 plan 欄位。
    """
    plan_fn = _func("_plan_step")
    assigns = [s for s in ast.walk(plan_fn)
               if isinstance(s, ast.Assign)
               and isinstance(s.targets[0], ast.Name) and s.targets[0].id == name]
    assert len(assigns) == 1, f"{name} 在 _plan_step 內被指派 {len(assigns)} 次，必須恰好 1 次"

    mods = [n for n in ast.walk(ast.parse(_SRC))
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod)
            and isinstance(n.right, ast.Name) and n.right.id == freq_var]
    assert len(mods) == 1, (
        f"`% {freq_var}` 在 run.py 內出現 {len(mods)} 次 —— 條件被算了第二遍就會漂移"
    )

    # run_loop 的控制流與 _ledger_fields 都必須讀 plan 的同一個欄位
    loop_reads = [n for n in ast.walk(_func("run_loop"))
                  if isinstance(n, ast.Attribute) and n.attr == name
                  and isinstance(n.value, ast.Name) and n.value.id == "plan"]
    assert any(isinstance(s, ast.If) and isinstance(s.test, ast.Attribute)
               and s.test.attr == name for s in ast.walk(_func("run_loop"))), (
        f"run_loop 的控制流沒有直接讀 plan.{name}"
    )
    assert loop_reads, f"run_loop 沒有讀到 plan.{name}"
    ledger_reads = [n for n in ast.walk(_func("_ledger_fields"))
                    if isinstance(n, ast.Attribute) and n.attr == name]
    assert ledger_reads, f"_ledger_fields 沒有讀到 plan.{name}"
    print(f"✓ trigger_boolean_is_computed_once[{name}]")


# ─────────────────────────────────────────────────────────────────────────────
# 3. 產物路徑（spec §8.3 第 8 項）
# ─────────────────────────────────────────────────────────────────────────────

#: cylinder 套件內所有可能讀到 artifacts_dir 的模組。
_CYL_MODULES = ("__init__.py", "config.py", "assembly.py", "run.py")


def _artifacts_dir_reads(tree: ast.AST) -> list[tuple[int, bool]]:
    """回傳 typed `.artifacts_dir` 讀取的 (行號, 是否位於 ledger gate 內)。"""
    out: list[tuple[int, bool]] = []

    def walk(node, gated: bool) -> None:
        if isinstance(node, ast.Attribute) and node.attr == "artifacts_dir":
            out.append((node.lineno, gated))
        if _is_ledger_gate(node):
            for st in node.body:
                walk(st, True)
            for st in node.orelse:      # else 分支不受 gate 保護
                walk(st, gated)
            return
        for child in ast.iter_child_nodes(node):
            walk(child, gated)

    walk(tree, False)
    return out


def test_artifacts_dir_is_read_once_and_only_inside_the_ledger_gate():
    """cylinder 的產物路徑守衛（Kolmogorov 版見 test_pipeline_assembly.py）。

    cylinder **本身不落任何 artifact**——唯一的落盤是診斷用的 rng_ledger.json，
    因此 `artifacts_dir` 只該被讀一次，且必須在 `if ledger is not None:` 內。

    兩個都會壞的方向，且壞掉時都不 crash：
      - 讀第二次 → 兩處各自算路徑，今天值相同所以不炸，日後一方改了就安靜地
        把檔寫到另一個地方（「artifacts_dir 未對齊」是本專案記錄有案的失敗模式）。
      - 讀到 gate 外 → production 路徑開始碰檔案系統，違反「gate 關閉時零額外工作」，
        而那正是 ledger 埋點得以不影響 bit-identical 的前提。
    """
    reads: list[tuple[str, int, bool]] = []
    for name in _CYL_MODULES:
        path = REPO_ROOT / "pi_lnn_jax" / "pipeline" / "cylinder" / name
        reads += [(name, ln, g)
                  for ln, g in _artifacts_dir_reads(ast.parse(path.read_text()))]

    assert len(reads) == 1, (
        "cylinder 的 artifacts_dir 應恰好被讀一次，實得：\n"
        + "\n".join(f"  {n}:{ln} (gated={g})" for n, ln, g in reads))
    name, lineno, gated = reads[0]
    assert gated, (
        f"{name}:{lineno} 的 artifacts_dir 讀取不在 `if ledger is not None:` 內"
        "——production 路徑不得因診斷埋點而碰檔案系統")
    print(f"✓ artifacts_dir_is_read_once_and_only_inside_the_ledger_gate ({name}:{lineno})")


@pytest.mark.parametrize("src,expect_gated", [
    ('def f():\n'
     '    if ledger is not None:\n'
     '        p = run.artifacts_dir\n', True),
    ('def f():\n'
     '    p = run.artifacts_dir\n'
     '    if ledger is not None:\n'
     '        q = p\n', False),
    ('def f():\n'
     '    if ledger is not None:\n'
     '        q = 1\n'
     '    else:\n'
     '        p = run.artifacts_dir\n', False),
])
def test_artifacts_dir_scanner_distinguishes_gated_from_ungated(src, expect_gated):
    """掃描器的自我證明：`assert gated` 只有在掃描器真的分得出 gate 內外時才有意義。

    第三個樣本（else 分支）是最容易寫錯的一種——`ast.If` 的 body 與 orelse
    若一起當成 gate 內，守衛就會放行一條 production 路徑。
    """
    reads = _artifacts_dir_reads(ast.parse(src))

    assert len(reads) == 1, reads
    assert reads[0][1] is expect_gated

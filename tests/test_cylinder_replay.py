"""Cylinder ledger replay —— 對 golden fixture 的逐步比對。

這是 cylinder bit-identical 契約在本機唯一可執行的證據。fixture 由 lab-server
（job 4737，CPU-only）以搬移前的 `train_cylinder.py` 錄製，本機不跑 training，
只做純比對。

失敗訊息刻意報出「第一個分岔的 step 與欄位」——「跑完 N 步比 params」只能告訴
你壞了，這裡要告訴你哪一步壞的。

`_plan_step` 的資料衍生輸入（時間軸、K/T、geometry）由 fixture 尾列的
`data_meta` 提供，不在本檔手寫。手寫值等於「猜到綠為止」。

三份 fixture 覆蓋兩條**正交**分支（spec §7）：
    c1  case=cylinder            time_marching=false  jax.random.choice(k3) + sample_wall_bc
    c2  case=cylinder            time_marching=true   np_rng.choice（第二條 NumPy 流）
    c3  case=controlled_cylinder time_marching=true   sample_wall_bc_moving（含壁速）

**一個已知的跨平台差異（不是被繞過，是被鑑別）**：c3 的 `bc_bvel` 在少數步與
golden 差一個元素、且恰好 1 ULP。錄製機是 lab-server（x86-64 XLA CPU），本機是
arm64；float32 的 `cos` 兩邊都不是正確捨入（本檔實測本機約 1% 元素與正確捨入不同），
故最後一位可以合法地不同。`bc_bvel = amp·2π·freq·cos(·)` 把該差異原樣放大到輸出，
而 `bc_body` 的 `sin` 結果先乘 amp≈0.12 再加到 base_center≈0.49，多半被吸收——這正是
「只有 bc_bvel 露出來」的原因。本檔不放寬比對，而是要求**每一個分岔都能由單一元素
±1 ULP 重現**：真正的邏輯錯誤（輸入取錯、key 用錯、時間窗算錯）會改動幾乎全部元素，
一個 ULP 的搜尋不可能命中，測試照樣紅。
"""
from __future__ import annotations

import ast
import json
import pathlib

import numpy as np
import pytest

import _acceptance_strictness
from _paths import REPO_ROOT

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "cylinder_ledger"
FIXTURE_NAMES = ["c1", "c2", "c3", "c4"]
RUN_PY = REPO_ROOT / "pi_lnn_jax" / "pipeline" / "cylinder" / "run.py"

#: ledger 中「陣列 digest」欄位 → `_StepPlan` 的同名欄位。ULP 鑑別要用原始陣列，
#: 純量欄位（t_max / trig_*）不在此列：那些不可能是浮點最後一位的問題。
ARRAY_FIELDS = ("cx", "cy", "ct", "sq_idx",
                "bc_in", "bc_body", "bc_slip", "bc_bvel")


def _load(name: str) -> list[dict]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def first_divergence(golden: list[dict], actual: list[dict]) -> str | None:
    """回傳第一個分岔的人類可讀描述；完全相同則回 None。"""
    if len(golden) != len(actual):
        return f"列數不同: golden={len(golden)} actual={len(actual)}"
    for g, a in zip(golden, actual):
        if g["step"] != a["step"]:
            return f"step 序不同: golden={g['step']} actual={a['step']}"
        for k in sorted(set(g) | set(a)):
            if g.get(k) != a.get(k):
                return (f"step {g['step']} 欄位 {k!r} 分岔: "
                        f"golden={g.get(k)!r} actual={a.get(k)!r}")
    return None


def _divergences(golden: list[dict], actual: list[dict]) -> list[tuple[int, str]]:
    """全部 (step, field) 分岔點；列數/步序不合則直接讓 caller 用 first_divergence 報。"""
    out: list[tuple[int, str]] = []
    for g, a in zip(golden, actual):
        for k in sorted(set(g) | set(a)):
            if k != "step" and g.get(k) != a.get(k):
                out.append((g["step"], k))
    return out


def _single_ulp_explanation(arr, golden_digest: str):
    """若 golden 恰能由 `arr` 的**單一元素 ±1 ULP** 重現，回傳 (index, 方向)。

    Why 是有效的鑑別而非放寬：邏輯錯誤（時間窗、key、幾何欄位取錯）會讓整個
    陣列改變，256 個候選裡不會有任何一個命中；能命中就代表 255/256 個元素與
    錄製當時逐位元相同，只有一個元素的最後一位不同——那是平台 libm 的自由度，
    不是本次搬移引入的行為變更。
    """
    from pi_lnn_jax.pipeline._ledger import digest

    a = np.asarray(arr)
    if a.dtype.kind != "f":       # 整數欄位（sq_idx）沒有 ULP 可言 → 無解釋
        return None
    flat = a.reshape(-1)
    for i in range(flat.size):
        for direction in (np.inf, -np.inf):
            probe = flat.copy()
            probe[i] = np.nextafter(probe[i], a.dtype.type(direction), dtype=a.dtype)
            if digest(probe.reshape(a.shape)) == golden_digest:
                return int(i), (+1 if direction == np.inf else -1)
    return None


@pytest.fixture
def at_repo_root(monkeypatch):
    """replay 會用尾列 argv 的相對 --config 路徑重新 load_config。"""
    monkeypatch.chdir(REPO_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# 0. fixture 自身健全性
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_is_self_consistent(name):
    """先確認 fixture 本身健全——壞掉的 oracle 比沒有 oracle 更危險。"""
    rows = _load(name)
    steps = [r for r in rows if r["step"] >= 0]
    tail = [r for r in rows if r["step"] == -1]
    assert steps, f"{name}: 無 step 資料列"
    assert len(tail) == 1 and tail[0].get("params_digest"), f"{name}: 缺 params_digest 尾列"
    # cylinder 的迴圈是 range(steps+1) —— 含第 0 步，這是與 Kolmogorov 的差異之一
    assert [r["step"] for r in steps] == list(range(len(steps))), (
        f"{name}: step 序應為 0..N 連續（cylinder 迴圈含第 0 步）"
    )
    assert tail[0].get("argv"), f"{name}: 尾列缺 argv —— replay 無法還原錄製時 config"
    assert tail[0].get("data_meta"), f"{name}: 尾列缺 data_meta —— replay 只能猜資料衍生量"
    print(f"✓ fixture_is_self_consistent[{name}] ({len(steps)} 步)")


def test_first_divergence_reports_step_and_field():
    g = [{"step": 1, "cx": "aaa"}, {"step": 2, "cx": "bbb"}]
    a = [{"step": 1, "cx": "aaa"}, {"step": 2, "cx": "ZZZ"}]
    msg = first_divergence(g, a)
    assert msg is not None and "step 2" in msg and "cx" in msg
    assert first_divergence(g, g) is None
    print("✓ first_divergence_reports_step_and_field")


def test_c1_c2_prove_k3_is_consumed_unconditionally():
    """spec §5.3 的核心不變量，直接從兩份 recorded truth 讀出來。

    c1/c2 同 config 家族、同 seed，只差 `time_marching`。若 `k3` 真的是**無條件**
    從 4-way split 抽出，則兩份的 `rk` 推進完全同步 → collocation（消耗 k1）與
    wall BC（消耗 k2）的 digest 必須逐步相同；而 sensor query 換了來源
    （`jax.random.choice(k3)` vs `np_rng.choice`）→ `sq_idx` 必須逐步不同。

    Why 這條不需要重播也值得測：它是 fixture 之間的關係，即使哪天 replay 因環境
    問題跑不動，這條仍然釘得住「k3 不得改成條件式 split」。
    """
    c1 = {r["step"]: r for r in _load("c1") if r["step"] >= 0}
    c2 = {r["step"]: r for r in _load("c2") if r["step"] >= 0}
    assert c1.keys() == c2.keys()
    same_cx = [s for s in c1 if c1[s]["cx"] == c2[s]["cx"]]
    diff_sq = [s for s in c1 if c1[s]["sq_idx"] != c2[s]["sq_idx"]]
    assert same_cx == sorted(c1), (
        f"c1/c2 的 cx 應逐步相同（rk 同步推進），實得相同的步 {sorted(same_cx)}"
    )
    assert diff_sq == sorted(c1), (
        f"c1/c2 的 sq_idx 應逐步不同（取樣來源不同），實得不同的步 {sorted(diff_sq)}"
    )
    # bc_* 消耗 k2，同樣必須逐步相同（wall BC 分支由 is_controlled 決定，兩者皆 false）
    for s in c1:
        for k in ("bc_in", "bc_body", "bc_slip", "bc_bvel"):
            assert c1[s][k] == c2[s][k], f"step {s} 的 {k} 在 c1/c2 不一致 —— k2 消耗被改動了"
    print(f"✓ c1_c2_prove_k3_is_consumed_unconditionally ({len(c1)} 步)")


# ─────────────────────────────────────────────────────────────────────────────
# 1. 重播：resolve_inputs(argv) → build_context（有資料時）或 data_meta（沒有時）
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_replay_matches_golden(name, at_repo_root, capsys):
    """新 pipeline 重播 host-side 決策序列，必須與 golden 逐欄位相同。

    只重播排程/取樣（不呼叫 step_fn、不做 forward/gradient），因此不算 training，
    可在本機 CPU 跑。這正好覆蓋最高風險面：RNG 消費時序。

    唯一允許的殘差是「單一元素 ±1 ULP」的跨平台浮點差異（見 module docstring）：
    每一個殘差都必須當場被鑑別出來，鑑別不出來就是紅的。

    `build_context` 與 `data_meta` fallback 驗證強度不同（見 `replay_plans`
    docstring）：前者重新從 npz 算出 st0/T_total/t_q_full_np/geom，後者只是把錄製
    當時的數字讀回來。本機無 npz（`data/cylinder_v1.npz` / `data/controlled_v2.npz`
    未追蹤），三份預期都走 fallback——把印出的 ctx source 收進最終斷言訊息，讓
    「這次 green 到底驗證了多強」在測試輸出裡一目瞭然。
    """
    from pi_lnn_jax.pipeline.cylinder.run import (
        REPLAY_CTX_SOURCE_BUILD_CONTEXT,
        REPLAY_CTX_SOURCE_DATA_META_FALLBACK,
        ReplayError,
        _ledger_fields,
        replay_plans,
    )
    golden = _load(name)
    try:
        plans = replay_plans(golden)
    except ReplayError as e:
        if "data_meta" not in str(e):
            raise
        pytest.skip(
            f"{name}: fixture 尾列缺 'data_meta'，且本機無錄製當時的 npz —— "
            f"需在 lab-server 以現版 train_cylinder.py 重錄此 fixture 後本測試才生效（{e}）"
        )
    captured_out = capsys.readouterr().out
    source_lines = [
        line.rsplit(": ", 1)[1] for line in captured_out.splitlines()
        if line.startswith("[replay_schedule] ctx source:")
    ]
    assert source_lines, f"{name}: replay 未印出 ctx source，觀測性回報遺失"
    ctx_source = source_lines[-1]
    assert ctx_source in (REPLAY_CTX_SOURCE_BUILD_CONTEXT, REPLAY_CTX_SOURCE_DATA_META_FALLBACK), (
        f"{name}: 未知的 ctx source {ctx_source!r}"
    )
    _acceptance_strictness.require_build_context(
        ctx_source, REPLAY_CTX_SOURCE_BUILD_CONTEXT, name)

    actual = [{"step": s, **_ledger_fields(plan)} for s, plan in plans]
    golden_steps = [r for r in golden if r["step"] >= 0]
    msg = first_divergence(golden_steps, actual)
    if msg is None:
        print(f"✓ replay_matches_golden[{name}] (比對 {len(actual)} 步逐欄位全等，"
              f"ctx_source={ctx_source})")
        return

    # 有分岔 → 逐一鑑別。能被單一元素 ±1 ULP 解釋的才算跨平台浮點差異。
    assert len(golden_steps) == len(actual), f"{name}: {msg}"
    plan_of = dict(plans)
    golden_of = {r["step"]: r for r in golden_steps}
    unexplained, ulp_hits = [], []
    for step, field in _divergences(golden_steps, actual):
        if field not in ARRAY_FIELDS:
            unexplained.append((step, field, "非陣列欄位，不可能是浮點最後一位"))
            continue
        hit = _single_ulp_explanation(getattr(plan_of[step], field), golden_of[step][field])
        if hit is None:
            unexplained.append((step, field, "單一元素 ±1 ULP 無法重現 golden"))
        else:
            ulp_hits.append((step, field, hit))
    assert not unexplained, (
        f"{name}: {msg}\n  首個無法以跨平台 1-ULP 解釋的分岔："
        + "; ".join(f"step {s} 欄位 {f}（{why}）" for s, f, why in unexplained)
    )
    print(f"✓ replay_matches_golden[{name}] (比對 {len(actual)} 步；"
          f"{len(ulp_hits)} 處跨平台 1-ULP 殘差已鑑別："
          + ", ".join(f"step {s}/{f}/elem {i}{'+' if d > 0 else '-'}1ulp"
                      for s, f, (i, d) in ulp_hits)
          + f"，ctx_source={ctx_source})")


def test_float32_cos_here_is_not_correctly_rounded():
    """上一條允許 1-ULP 殘差的前提：float32 三角函式本來就沒有跨平台位元保證。

    Why 要當場量而不是宣稱：若哪天 XLA 改成正確捨入（或本機換了實作），這條會紅，
    逼下一個人重新檢視「1-ULP 殘差是可接受的」這個假設，而不是讓一條放寬的比對
    永遠留在測試裡。
    """
    import jax.numpy as jnp

    x = jnp.asarray(np.linspace(0.0, 20.0, 4096), jnp.float32)
    ref = np.cos(np.asarray(x, dtype=np.float64)).astype(np.float32)
    n_diff = int(np.sum(np.asarray(jnp.cos(x)) != ref))
    assert n_diff > 0, (
        "本機 float32 cos 竟與正確捨入完全一致 —— 若各平台皆如此，"
        "test_replay_matches_golden 的 1-ULP 容許就該收緊成嚴格相等"
    )
    print(f"✓ float32_cos_here_is_not_correctly_rounded ({n_diff}/{x.size} 元素)")


def test_replay_rejects_ledger_without_argv():
    """尾列沒有 provenance 就必須 fail-fast，不得在 replay 端猜 config。"""
    from pi_lnn_jax.pipeline.cylinder.run import ReplayError, replay_schedule
    with pytest.raises(ReplayError, match="argv"):
        replay_schedule([{"step": 0, "cx": "a"}, {"step": -1, "params_digest": "x"}])
    with pytest.raises(ReplayError, match="尾列"):
        replay_schedule([{"step": 0, "cx": "a"}])
    print("✓ replay_rejects_ledger_without_argv")


def test_replay_rejects_ledger_without_data_meta(tmp_path):
    """npz 不可解析、尾列又缺 data_meta → fail-fast，不得退回猜值。

    Why 不能靠「本機沒有 cylinder npz」造情境：這件事只在本機成立，lab-server 上
    `data/cylinder_v1.npz` 確實存在，`build_context` 會成功、fallback 根本不會被
    觸發，同一條測試在兩台機器上會給出相反的結果——一個隨機器翻面的測試比沒有
    測試更糟：它會被在「答案剛好是錯的」那台機器上動手「修好」，真正的不變量
    反而失傳。

    改用一份自造 config：其 `cylinder_data_npz` 指向 `tmp_path` 下一個從未建立的
    檔案。這個路徑在任何機器上都保證不存在（不是「剛好本機沒有」，是「這個路徑
    從未被賦予意義」），因此 `build_context` 的 `np.load` 必然丟
    `FileNotFoundError`、`_missing_recorded_dataset` 必然判定其 filename 與
    `config.data.npz_path` 相同（兩者本就是同一個字串）而放行 fallback，
    無論在本機（Mac，本來就沒有 cylinder npz）還是 lab-server（cylinder npz 存在，
    但我們指的不是它）都會走到 `_ctx_from_data_meta`；尾列缺 `data_meta` 因而必炸。

    訊息必須明說「要重錄」，否則下一個人只會看到一個沒頭沒尾的 KeyError。
    """
    from pi_lnn_jax.pipeline.cylinder.run import ReplayError, replay_schedule
    npz_path = tmp_path / "replay_probe_missing.npz"      # 刻意不建立
    cfg_path = tmp_path / "replay_probe_missing_npz.toml"
    cfg_path.write_text(f'case = "cylinder"\ncylinder_data_npz = "{npz_path}"\n')
    tail = {
        "step": -1, "params_digest": "x",
        "argv": ["--config", str(cfg_path), "--steps", "2", "--allow-cpu"],
    }
    with pytest.raises(ReplayError, match="data_meta"):
        replay_schedule([{"step": 0, "cx": "a"}, tail])
    print("✓ replay_rejects_ledger_without_data_meta")


def test_replay_reraises_unrelated_missing_file(at_repo_root, monkeypatch):
    """只有「錄製當時的 npz 不在本機」才准走 fallback。

    Why: 若 `build_context` 因為別的檔案不見而炸，那是 regression，被 fallback
    吞掉就會變成一份「看似通過」的 replay。
    """
    import pi_lnn_jax.pipeline.cylinder.run as run_mod
    from pi_lnn_jax.pipeline.cylinder.run import replay_schedule

    def _other_missing(_inputs):
        raise FileNotFoundError(2, "No such file or directory", "data/some_other_file.npz")

    monkeypatch.setattr(run_mod, "build_context", _other_missing)
    with pytest.raises(FileNotFoundError, match="some_other_file"):
        replay_schedule(_load("c1"))
    print("✓ replay_reraises_unrelated_missing_file")


# ─────────────────────────────────────────────────────────────────────────────
# 2. data_meta 的錄製／還原對
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("moving", [False, True])
def test_data_meta_round_trips_plan_step_inputs(at_repo_root, moving):
    """`_data_meta` 寫下的東西，`_ctx_from_data_meta` 必須原樣還原（含兩種幾何）。

    順帶驗 JSON 可序列化——`Ledger.dump` 沒有 `default=`，非 JSON 型別會拖到
    錄製尾聲才炸。
    """
    import jax.numpy as jnp

    from pi_lnn_jax.boundary import CylinderGeometry, OscillatingGeometry
    from pi_lnn_jax.pipeline.cylinder.assembly import TrainingContext
    from pi_lnn_jax.pipeline.cylinder.config import resolve_inputs
    from pi_lnn_jax.pipeline.cylinder.run import _ctx_from_data_meta, _data_meta

    geom = (
        OscillatingGeometry(base_center=(0.25, 0.5), body_radius_phys=0.015, Lx=0.6,
                            Ly=0.3, u_inf=0.33, amp=0.01, freq=2.5, phase=0.3,
                            axis=(0.0, 1.0))
        if moving else
        CylinderGeometry(body_center=(0.25, 0.5), body_radius=0.05, Lx=0.6, Ly=0.3,
                         u_inf=0.33)
    )
    st = jnp.asarray(np.linspace(0.1, 20.05, 37), jnp.float32)
    T, K = int(st.shape[0]), 5
    ctx = TrainingContext(**dict.fromkeys(TrainingContext._fields))._replace(
        st=st, K=K, T=T, geom=geom,
        st0=float(st[0]), tm_end=float(st[-1]), T_total=float(st[-1] - st[0]),
        t_q_full_np=np.asarray(jnp.broadcast_to(st[:, None], (T, K)).reshape(T * K)),
        n_sensor_q=min(2000, T * K),
    )
    # 錄製端最終會走 json.dump；先過一次同樣的序列化，型別不合會在這裡炸
    meta = json.loads(json.dumps(_data_meta(ctx)))
    config = resolve_inputs([
        "--config", "configs/exp_cyl_cexp002_notm.toml", "--allow-cpu",
    ]).config
    rebuilt = _ctx_from_data_meta(config, {"data_meta": meta}, FileNotFoundError("no npz"))

    assert rebuilt.st0 == ctx.st0 and rebuilt.tm_end == ctx.tm_end
    assert rebuilt.T_total == ctx.T_total, "t_total 必須讀回，不得由端點相減重算"
    assert (rebuilt.K, rebuilt.T) == (ctx.K, ctx.T)
    assert rebuilt.t_q_full_np.dtype == ctx.t_q_full_np.dtype
    assert np.array_equal(rebuilt.t_q_full_np, ctx.t_q_full_np), "t_q_full 重建非逐位元相同"
    assert rebuilt.geom == ctx.geom and type(rebuilt.geom) is type(ctx.geom)
    # config 衍生量走 argv 重解；clamp 上界則由 data_meta 的 T*K 給出
    assert rebuilt.n_sensor_q == ctx.n_sensor_q == T * K
    print(f"✓ data_meta_round_trips_plan_step_inputs[{'moving' if moving else 'static'}]")


def test_geom_from_meta_rejects_unknown_field_set():
    """欄位集對不上任何幾何型別時必須炸，不得猜型別。"""
    from pi_lnn_jax.pipeline.cylinder.run import ReplayError, _geom_from_meta
    with pytest.raises(ReplayError, match="geom"):
        _geom_from_meta({"geom": {"body_center": [0.1, 0.2], "unexpected": 1.0}})
    print("✓ geom_from_meta_rejects_unknown_field_set")


# ─────────────────────────────────────────────────────────────────────────────
# 3. 反漂移哨兵（AST）
# ─────────────────────────────────────────────────────────────────────────────

def _funcs() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(RUN_PY.read_text())
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _called_names(func: ast.FunctionDef) -> set:
    return {c.func.id for c in ast.walk(func)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}


def _rng_call_chains(func: ast.FunctionDef) -> list[str]:
    """函式內所有以 `jax.random` / `np.random` 為根的呼叫（任意深度屬性鏈）。"""
    out = []
    for call in ast.walk(func):
        if not isinstance(call, ast.Call):
            continue
        node, attrs = call.func, []
        while isinstance(node, ast.Attribute):
            attrs.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            continue
        attrs.append(node.id)
        attrs.reverse()
        if len(attrs) >= 2 and attrs[1] == "random" and attrs[0] in ("jax", "np", "numpy"):
            out.append(".".join(attrs))
    return out


def test_replay_shares_plan_step_with_run_loop():
    """反漂移哨兵：重播與 run_loop 必須跑同一份排程程式碼。

    Why 用測試而不是 code review 記得：兩份排程程式碼不會 crash，只會讓 replay
    安靜地不再測真正在跑的東西。

    Why 只驗證「呼叫了 `_plan_step`」不夠：caller 可以呼叫它、然後在其後分岔——
    例如 `plan = plan._replace(...)` 事後改動結果，或在重播端另外消費一次
    RNG。cylinder 有**兩條** stream（`jax.random` 與 `np.random`），兩者都要擋。
    """
    funcs = _funcs()
    for name in ("run_loop", "replay_plans", "replay_schedule", "_plan_step"):
        assert name in funcs, f"run.py 缺函式 {name}"
    for fn_name in ("_plan_step", "_open_loop_streams"):
        for caller in ("run_loop", "replay_plans"):
            assert fn_name in _called_names(funcs[caller]), (
                f"{caller}() 沒有呼叫 {fn_name}() —— 排程程式碼疑似被複製一份"
            )
    assert "_ledger_fields" in _called_names(funcs["run_loop"])
    assert "_ledger_fields" in _called_names(funcs["replay_schedule"])
    assert "replay_plans" in _called_names(funcs["replay_schedule"]), (
        "replay_schedule 必須走 replay_plans，不得自己再跑一次迴圈"
    )

    for caller in ("replay_plans", "replay_schedule"):
        chains = _rng_call_chains(funcs[caller])
        assert not chains, (
            f"{caller}() 內直接消費 RNG（{chains}）—— 重播端任何額外消費都會讓 "
            f"rk / np_rng 與 run_loop 失去對齊"
        )
    # 事後改動 plan 一樣會偏離 run_loop，即使呼叫了共用 helper
    for caller in ("replay_plans", "replay_schedule"):
        replaces = [n for n in ast.walk(funcs[caller])
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "_replace"]
        assert not replaces, f"{caller}() 對回傳值呼叫 _replace —— 重播會悄悄偏離 run_loop"
    print("✓ replay_shares_plan_step_with_run_loop")


def test_rng_scan_would_catch_an_extra_consumption():
    """掃描器的自我證明：把一次額外 RNG 消費貼進重播函式，必須被抓到。

    「沒有 hit」有兩種解釋——重播乾淨，或掃描器根本沒在看。這條排除後者。
    """
    src = RUN_PY.read_text() + (
        "\n\ndef _contaminated_replay():\n"
        "    k = jax.random.split(rk, 2)\n"
        "    n = np.random.RandomState(0)\n"
        "    return k, n\n"
    )
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_contaminated_replay")
    assert set(_rng_call_chains(fn)) == {"jax.random.split", "np.random.RandomState"}
    print("✓ rng_scan_would_catch_an_extra_consumption")


# ─────────────────────────────────────────────────────────────────────────────
# 4. spec §5.1 —— replay 原理上碰不到的 init stream
# ─────────────────────────────────────────────────────────────────────────────

def test_initialize_draws_init_t_from_the_same_generator_as_init_xy(at_repo_root):
    """spec §5.1：**單一** `RandomState(seed)` 連抽 init_xy → init_t。

    這與 Kolmogorov **相反**（那邊是兩個各自建構的 `RandomState`，故 init_t 不是
    init_xy 的續抽）。兩個 case 各自都是承重的，任何一邊被「統一」都會改動 params。

    `replay_schedule` 不重播 init（迴圈 stream 由 seed 另行建構），所以這條只能
    用行為級測試釘住。先做鑑別力自檢：規格形式與最可能被寫成的壞形式數值上確實不同。
    """
    from pi_lnn_jax.pipeline.cylinder.assembly import TrainingContext
    from pi_lnn_jax.pipeline.cylinder.config import resolve_inputs
    from pi_lnn_jax.pipeline.cylinder.run import initialize

    seed, st0, T_total = 42, 0.05, 19.9
    one_gen = np.random.RandomState(seed)
    expect_xy = one_gen.uniform(0, 1, (8, 2))
    expect_t = one_gen.uniform(st0, st0 + T_total, (8,))
    two_gens = np.random.RandomState(seed).uniform(st0, st0 + T_total, (8,))
    assert not np.allclose(expect_t, two_gens), (
        "鑑別力自檢失敗：單一生成器續抽與兩個生成器各自首抽竟相同，本測試無鑑別力"
    )

    captured: dict = {}

    class _RecordingModel:
        def init(self, key, sv, sp, re_norm, st, init_xy, init_t):
            del key, sv, sp, re_norm, st
            captured["init_xy"] = np.asarray(init_xy)
            captured["init_t"] = np.asarray(init_t)
            return {}

    class _NoopTx:
        def init(self, params):
            del params
            return {}

    config = resolve_inputs(["--config", "configs/exp_cyl_cexp002_notm.toml",
                             "--seed", str(seed), "--allow-cpu"]).config
    ctx = TrainingContext(**dict.fromkeys(TrainingContext._fields))._replace(
        config=config, model=_RecordingModel(), tx=_NoopTx(),
        st0=st0, T_total=T_total, n_sensor_q=4,
        data_loss_fn=lambda p, idx: 0.0,      # probe 走 jax.grad，空 params → 空 grads
        get_subtree=lambda g: g,              # fallback 分支：回傳原 grads
        gn_ref_path=("temporal_encoder",),
        opt_info={"name": "x", "lr_schedule": "y", "fallback_to": None},
    )
    state = initialize(ctx)

    assert np.allclose(captured["init_xy"], expect_xy.astype(np.float32), atol=0, rtol=1e-6)
    assert np.allclose(captured["init_t"], expect_t.astype(np.float32), atol=0, rtol=1e-6), (
        "init_t 不是 init_xy 的續抽 —— 單一 RandomState 被拆成兩個了（spec §5.1）"
    )
    assert state.rk is None, "initialize 不得開迴圈 stream（那是 run_loop 的 §5.2）"
    print("✓ initialize_draws_init_t_from_the_same_generator_as_init_xy")


def test_loop_streams_use_seed_plus_1_and_plus_7():
    """spec §5.2：`PRNGKey(seed + 1)` 與 `RandomState(seed + 7)`。

    偏移量本身是規格（不是 `seed`、也不是 Kolmogorov 的 7919）；寫錯不會 crash，
    只會讓整條軌跡靜靜換一組亂數。
    """
    import jax

    from pi_lnn_jax.pipeline.cylinder.run import _open_loop_streams

    rk, np_rng = _open_loop_streams(42)
    assert np.array_equal(np.asarray(rk), np.asarray(jax.random.PRNGKey(43)))
    assert not np.array_equal(np.asarray(rk), np.asarray(jax.random.PRNGKey(42)))
    assert np.array_equal(np_rng.uniform(size=5),
                          np.random.RandomState(49).uniform(size=5))
    print("✓ loop_streams_use_seed_plus_1_and_plus_7")

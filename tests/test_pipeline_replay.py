"""Ledger replay —— 對 golden fixture 的逐步比對。

這是 bit-identical 契約在本機唯一可執行的證據。fixture 由 lab-server 錄製
(Task 3)，本機不跑 training，只做純比對。

失敗訊息刻意報出「第一個分岔的 step 與欄位」——「跑完 N 步比 params」
只能告訴你壞了，這裡要告訴你哪一步壞的。

`_plan_step` 的 data 衍生輸入（時間範圍、per-Re T/K、CRP 內插表）由 fixture
尾列的 `data_meta` 提供，不在本檔手寫。手寫值等於「猜到綠為止」，而 f2
（multi-Re + CRP + sensor mini-batch + dropout 全開）要猜的量根本不可能靠手寫
可信地填出來——那正是本機 oracle 最大的洞。

一個已知的覆蓋缺口（**不是被繞過，是被標記**）：RAR 步的 `cx/cy/ct` 依 current
params 的 residual 選點，host-side 重播拿不到那一步的 params，**原理上不可重播**。
replay 對那些步回 `RAR_SENTINEL`，由 `test_replay_matches_golden[f3_rar]` 明確
斷言「哨兵恰好落在 6/9/12/15/18」，而不是塞一個看起來對的 digest 讓 oracle
安靜失效。
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest

import _acceptance_strictness
from _paths import REPO_ROOT

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "ledger"
FIXTURE_NAMES = ["f1_single_re", "f2_multi_re_crp", "f3_rar", "f4_prod_scale"]

# f3 的 CLI：--rar_freq 3 --rar_warmup 5 → 20 步中 s>5 且 s%3==0
F3_RAR_STEPS = {6, 9, 12, 15, 18}

#: `_plan_step` 允許從 ctx 讀到的欄位。前四項是 data 衍生量（由尾列 data_meta
#: 記錄、replay 端還原）；`config` 與 `crp_re_norm_scale` 是 config 衍生量，
#: replay 端一律從尾列 argv 重解，刻意不進 data_meta。
PLAN_STEP_CTX_FIELDS = {
    "re_batches", "re_t_min_host", "re_t_max_host", "crp_interp",
    "config", "n_sensor_query", "crp_re_norm_scale",
}


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


@pytest.fixture
def at_repo_root(monkeypatch):
    """replay 會用尾列 argv 的相對 --config 路徑重新 load_config。"""
    monkeypatch.chdir(REPO_ROOT)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_is_self_consistent(name):
    """先確認 fixture 本身健全——壞掉的 oracle 比沒有 oracle 更危險。"""
    rows = _load(name)
    steps = [r for r in rows if r["step"] > 0]
    tail = [r for r in rows if r["step"] == -1]
    assert steps, f"{name}: 無 step 資料列"
    assert len(tail) == 1 and tail[0].get("params_digest"), f"{name}: 缺 params_digest 尾列"
    assert [r["step"] for r in steps] == sorted(r["step"] for r in steps)
    assert tail[0].get("argv"), f"{name}: 尾列缺 argv —— replay 無法還原錄製時 config"
    print(f"✓ fixture_is_self_consistent[{name}] ({len(steps)} 步)")


def test_first_divergence_reports_step_and_field():
    g = [{"step": 1, "cx": "aaa"}, {"step": 2, "cx": "bbb"}]
    a = [{"step": 1, "cx": "aaa"}, {"step": 2, "cx": "ZZZ"}]
    msg = first_divergence(g, a)
    assert msg is not None and "step 2" in msg and "cx" in msg
    assert first_divergence(g, g) is None
    print("✓ first_divergence_reports_step_and_field")


def test_f3_differs_from_f1_exactly_at_rar_steps():
    """f1 與 f3 同 config、同 seed，只差 CLI 的 RAR 旗標。

    Why 這條不需要重播也值得測：它直接從兩份 recorded truth 證明
    「RAR 觸發時換路徑、不觸發時 `rng_collo` 仍與 RAR-off 完全對齊」——
    也就是 §5.4 第 5 點那個無條件 split 確實存在。若哪天有人把
    `rng_collo` 的 split 挪進 else 分支，這條會紅。
    """
    f1 = {r["step"]: r for r in _load("f1_single_re") if r["step"] > 0}
    f3 = {r["step"]: r for r in _load("f3_rar") if r["step"] > 0}
    assert f1.keys() == f3.keys()
    differing = {s for s in f1 if any(f1[s][k] != f3[s][k] for k in ("cx", "cy", "ct"))}
    assert differing == F3_RAR_STEPS, (
        f"f1/f3 collocation 分岔的步數應恰為 {sorted(F3_RAR_STEPS)}，實得 {sorted(differing)}"
    )
    # 非 collocation 欄位（排程/觸發）在兩份 fixture 應完全一致
    for s in f1:
        for k in f1[s]:
            if k in ("cx", "cy", "ct"):
                continue
            assert f1[s][k] == f3[s][k], f"step {s} 欄位 {k} 在 f1/f3 不一致"
    print(f"✓ f3_differs_from_f1_exactly_at_rar_steps {sorted(F3_RAR_STEPS)}")


# ─────────────────────────────────────────────────────────────────────────────
# 重播：resolve_inputs(argv) → build_context（有資料時）或 data_meta（沒有時）
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_replay_matches_golden(name, at_repo_root, capsys):
    """新 pipeline 重播 host-side 決策序列，必須與 golden 逐欄位相同。

    只重播排程/取樣（不呼叫 step_fn、不做 forward/gradient），因此不算
    training，可在本機 CPU 跑。這正好覆蓋最高風險面：RNG 消費時序。

    `_plan_step` 的 data 衍生輸入全部來自 fixture 尾列的 `data_meta`，本檔不
    手寫任何一個值；錄製當時的資料檔在本機時則直接走 `build_context`。
    尾列缺 `data_meta`（fixture 錄製於該欄位存在之前）→ skip，並在理由裡直說
    要重錄——**skip 不是 pass**。

    `build_context` 與 `data_meta` fallback 驗證強度不同（見 `replay_schedule`
    docstring）：前者重新從資料檔算出 `_load_datasets` 等衍生量，後者只是把
    錄製當時的數字讀回來。本機通常無錄製當時的資料檔，三份 fixture 預期都走
    fallback——把 `replay_schedule` 印出的 ctx source 收進最終斷言訊息，讓
    「這次 green 到底驗證了多強」在測試輸出裡一目瞭然，不必臆測。
    """
    from pi_lnn_jax.pipeline.kolmogorov.run import (
        RAR_SENTINEL,
        REPLAY_CTX_SOURCE_BUILD_CONTEXT,
        REPLAY_CTX_SOURCE_DATA_META_FALLBACK,
        ReplayError,
        replay_schedule,
    )
    golden = _load(name)
    try:
        actual = replay_schedule(golden)
    except ReplayError as e:
        if "data_meta" not in str(e):
            raise
        pytest.skip(
            f"{name}: fixture 尾列缺 'data_meta'，且本機無錄製當時的資料檔 —— "
            f"需在 lab-server 以現版 train_kolmogorov.py 重錄此 fixture 後本測試才生效（{e}）"
        )
    captured_out = capsys.readouterr().out
    source_lines = [
        line.rsplit(": ", 1)[1] for line in captured_out.splitlines()
        if line.startswith("[replay_schedule] ctx source:")
    ]
    assert source_lines, f"{name}: replay_schedule 未印出 ctx source，觀測性回報遺失"
    ctx_source = source_lines[-1]
    assert ctx_source in (REPLAY_CTX_SOURCE_BUILD_CONTEXT, REPLAY_CTX_SOURCE_DATA_META_FALLBACK), (
        f"{name}: 未知的 ctx source {ctx_source!r}"
    )
    _acceptance_strictness.require_build_context(
        ctx_source, REPLAY_CTX_SOURCE_BUILD_CONTEXT, name)
    golden_steps = [r for r in golden if r["step"] > 0]
    # RAR 步原理上不可 host-side 重播（見 module docstring）——兩邊一起濾掉再比。
    rar_steps = {r["step"] for r in actual if r["cx"] == RAR_SENTINEL}
    golden_steps = [r for r in golden_steps if r["step"] not in rar_steps]
    actual = [r for r in actual if r["step"] not in rar_steps]
    msg = first_divergence(golden_steps, actual)
    assert msg is None, f"{name}: {msg}"
    if name == "f3_rar":
        assert rar_steps == F3_RAR_STEPS, f"RAR 哨兵步數 {sorted(rar_steps)}"
    else:
        assert not rar_steps, f"{name} 不該有 RAR 步，實得 {sorted(rar_steps)}"
    print(
        f"✓ replay_matches_golden[{name}] (比對 {len(actual)} 步，RAR 未驗 {len(rar_steps)} 步，"
        f"ctx_source={ctx_source})"
    )


def test_replay_rejects_ledger_without_argv():
    """尾列沒有 provenance 就必須 fail-fast，不得在 replay 端猜 config。"""
    from pi_lnn_jax.pipeline.kolmogorov.run import ReplayError, replay_schedule
    with pytest.raises(ReplayError, match="argv"):
        replay_schedule([{"step": 1, "cx": "a"}, {"step": -1, "params_digest": "x"}])
    with pytest.raises(ReplayError, match="尾列"):
        replay_schedule([{"step": 1, "cx": "a"}])
    print("✓ replay_rejects_ledger_without_argv")


def test_replay_rejects_ledger_without_data_meta(at_repo_root, tmp_path):
    """資料檔不可解析、尾列又缺 data_meta → fail-fast，不得退回猜值。

    Why 不能靠「本機沒有資料檔」造情境：那件事只在開發機成立。lab-server 上
    `data/sensors/*.json` 與 `data/dns/*.npy` 確實存在，`build_context` 會成功、
    fallback 根本不會被觸發，同一條測試在兩台機器上給出相反結果——**一個隨機器
    翻面的測試比沒有測試更糟**：它會被在「答案剛好是錯的」那台機器上動手「修好」，
    真正的不變量反而失傳。（job 4781 就是這樣紅的；cylinder 的對應測試早已改成
    下面這種寫法，Kolmogorov 這側漏掉了。）

    改用一份自造 config，其 sensor/DNS 路徑指向 `tmp_path/data/` 下**從未建立**
    的檔案——不是「剛好本機沒有」，是「這個路徑從未被賦予意義」，任何機器上都
    保證不存在。路徑刻意含 `/data/` 且副檔名為 `.json`/`.npy`，好讓
    `_missing_recorded_dataset` 判定它確實是「錄製當時的資料檔不在本機」而放行
    fallback（否則會被當成 resolve 規則的 regression 而直接拋出）。
    """
    from pi_lnn_jax.pipeline.kolmogorov.run import ReplayError, replay_schedule

    missing_dir = tmp_path / "data"
    missing_dir.mkdir()                                   # 目錄在，檔案刻意不建立
    base = (REPO_ROOT / "configs" / "_ledger_single_re.toml").read_text()
    body, _, _ = base.partition("[data]")
    cfg_path = tmp_path / "replay_probe_missing_data.toml"
    cfg_path.write_text(
        body
        + "[data]\n"
        + "re_values = [1000.0]\n"
        + f'sensor_jsons = ["{missing_dir}/never_created_sensors.json"]\n'
        + f'dns_paths = ["{missing_dir}/never_created_dns.npy"]\n'
        + "re_norm_scale = 1000000.0\n"
        + 'observed_sensor_channels = ["u", "v"]\n'
    )
    tail = {
        "step": -1, "params_digest": "x",
        "argv": ["--config", str(cfg_path), "--steps", "2"],
    }

    with pytest.raises(ReplayError, match="data_meta"):
        replay_schedule([{"step": 1, "cx": "a"}, tail])
    print("✓ replay_rejects_ledger_without_data_meta")


def test_data_meta_round_trips_plan_step_inputs(at_repo_root):
    """`_data_meta` 寫下的東西，`_ctx_from_data_meta` 必須原樣還原。

    Why 這條不依賴 fixture：fixture 尚待重錄，但「錄製/還原」這一對函式現在
    就必須是對的，否則要等重錄回來才發現壞掉。順帶驗 JSON 可序列化——
    `Ledger.dump` 沒有 `default=`，非 JSON 型別會拖到錄製尾聲才炸。
    """
    import jax.numpy as jnp

    from pi_lnn_jax.pipeline.kolmogorov.assembly import (
        ReBatch,
        TrainingContext,
        _build_crp_interp,
    )
    from pi_lnn_jax.pipeline.kolmogorov.config import resolve_inputs
    from pi_lnn_jax.pipeline.kolmogorov.run import _ctx_from_data_meta, _data_meta

    def _rb(T, K):
        fields = dict.fromkeys(ReBatch._fields)
        fields["sensor_vals"] = jnp.zeros((T, K, 2))
        return ReBatch(**fields)

    datasets = [
        {"re_norm": 0.9, "norm_stats": {"u_mean": 0.4, "u_std": 1.1, "v_mean": -0.6,
                                        "v_std": 2.1, "p_mean": 0.7, "p_std": 3.1}},
        {"re_norm": 0.5, "norm_stats": {"u_mean": 0.1, "u_std": 1.5, "v_mean": -0.2,
                                        "v_std": 2.5, "p_mean": 0.3, "p_std": 3.5}},
    ]
    ctx = TrainingContext(**dict.fromkeys(TrainingContext._fields))._replace(
        datasets=datasets,
        re_batches=[_rb(7, 3), _rb(11, 5)],
        re_t_min_host=[0.0, 0.25],
        re_t_max_host=[5.0, 4.75],
        crp_interp=_build_crp_interp(datasets),
    )
    # 錄製端最終會走 json.dump；先過一次同樣的序列化，型別不合會在這裡炸
    meta = json.loads(json.dumps(_data_meta(ctx)))
    config = resolve_inputs(["--config", "configs/_ledger_single_re.toml"]).config
    rebuilt = _ctx_from_data_meta(config, {"data_meta": meta}, FileNotFoundError("no data"))

    assert len(rebuilt.datasets) == len(ctx.datasets)
    assert rebuilt.re_t_min_host == ctx.re_t_min_host
    assert rebuilt.re_t_max_host == ctx.re_t_max_host
    assert [tuple(rb.sensor_vals.shape[:2]) for rb in rebuilt.re_batches] == [(7, 3), (11, 5)]
    assert set(rebuilt.crp_interp) == set(ctx.crp_interp)
    for k, v in ctx.crp_interp.items():
        assert np.allclose(np.asarray(rebuilt.crp_interp[k]), np.asarray(v)), f"crp_interp[{k}]"
    # config 衍生量走 argv 重解，不從 data_meta 取
    assert rebuilt.crp_re_norm_scale == 1000000.0
    print("✓ data_meta_round_trips_plan_step_inputs")


def test_data_meta_is_sufficient_to_reproduce_plan_step(at_repo_root, monkeypatch, capsys):
    """同一份 ctx，經 data_meta 繞一圈重建後，`replay_schedule` 必須產出同一串排程。

    這是 `data_meta` 唯一要成立的主張：「replay 端不需要原始資料檔」。用 f2 的
    argv（multi-Re + CRP + sensor mini-batch + dropout 同時開）驅動，四個條件式
    RNG 消費點一次全走到——正是手寫 ctx 填不出來的那一組。

    合成 ctx 的數字只是**輸入**，不當 oracle：兩次重播共用同一份輸入，比的是
    「經過 data_meta 之後有沒有走樣」。fixture 重錄前，這是 f2 分支唯一的本機證據。

    本測試用 monkeypatch 精確控制走哪條路（A 強制 build_context 成功、B 強制
    FileNotFoundError），因此可反過來拿它驗證 `replay_schedule` 的路徑回報本身
    是否準確——這點 `test_replay_matches_golden` 做不到，那裡走哪條路是機器
    現況決定的，不是測試控制的。
    """
    import jax.numpy as jnp

    import pi_lnn_jax.pipeline.kolmogorov.run as run_mod
    from pi_lnn_jax.pipeline.kolmogorov.assembly import (
        ReBatch,
        TrainingContext,
        _build_crp_interp,
    )
    from pi_lnn_jax.pipeline.kolmogorov.config import resolve_inputs

    f2_argv = ["--config", "configs/_ledger_f2.toml", "--steps", "20", "--seed", "42",
               "--arch", "liquid", "--multi_re", "--sensor_dropout_rate", "0.2"]
    config = resolve_inputs(list(f2_argv)).config
    assert (config.loss.use_continuous_re_physics
            and config.curriculum.n_sensor_query_requested > 0)

    datasets = [
        {"re_norm": rn, "norm_stats": {"u_mean": rn, "u_std": 1.0 + rn, "v_mean": -rn,
                                       "v_std": 2.0 + rn, "p_mean": 0.5 * rn, "p_std": 3.0}}
        for rn in (0.9, 0.5, 0.7)
    ]

    def _rb(T, K):
        fields = dict.fromkeys(ReBatch._fields)
        fields["sensor_vals"] = jnp.zeros((T, K, 2))
        return ReBatch(**fields)

    ctx_real = TrainingContext(**dict.fromkeys(TrainingContext._fields))._replace(
        config=config,
        datasets=datasets,
        re_batches=[_rb(20, 10), _rb(20, 10), _rb(20, 10)],
        re_t_min_host=[0.0, 0.1, 0.2],
        re_t_max_host=[5.0, 4.5, 4.0],
        crp_interp=_build_crp_interp(datasets),
        crp_re_norm_scale=1000000.0,
        n_sensor_query=config.curriculum.n_sensor_query_requested,
    )
    tail = {"step": -1, "params_digest": "x", "argv": f2_argv,
            "data_meta": json.loads(json.dumps(run_mod._data_meta(ctx_real)))}
    ledger_rows = [tail]

    def _last_ctx_source() -> str:
        lines = [
            line.rsplit(": ", 1)[1] for line in capsys.readouterr().out.splitlines()
            if line.startswith("[replay_schedule] ctx source:")
        ]
        assert lines, "replay_schedule 未印出 ctx source"
        return lines[-1]

    # A: 資料檔在本機的路徑 —— build_context 直接給出 ctx_real
    monkeypatch.setattr(run_mod, "build_context", lambda _inputs: ctx_real)
    from_ctx = run_mod.replay_schedule(ledger_rows)
    assert _last_ctx_source() == run_mod.REPLAY_CTX_SOURCE_BUILD_CONTEXT, (
        "A 路徑（資料檔在本機）應回報 build_context，路徑回報本身失準"
    )
    # B: 資料檔不在本機 —— 走 data_meta 重建
    def _no_data(_inputs):
        raise FileNotFoundError("sensor JSON not found: /data/sensors/absent.json")

    monkeypatch.setattr(run_mod, "build_context", _no_data)
    from_meta = run_mod.replay_schedule(ledger_rows)
    assert _last_ctx_source() == run_mod.REPLAY_CTX_SOURCE_DATA_META_FALLBACK, (
        "B 路徑（資料檔不在本機）應回報 data_meta fallback，路徑回報本身失準"
    )

    assert len(from_ctx) == 20
    # 四個條件式消費點確實都走到了（否則這條測的只是 f1 那條窄路）
    assert len({r["re_idx"] for r in from_ctx}) > 1, "multi-Re 分支沒走到"
    assert all(r["re_norm_p"] is not None for r in from_ctx), "CRP 分支沒走到"
    assert all(r["sensor_idx"] is not None for r in from_ctx), "sensor mini-batch 分支沒走到"
    assert all(r["dropout_mask"] is not None for r in from_ctx), "dropout 分支沒走到"
    msg = first_divergence(from_ctx, from_meta)
    assert msg is None, f"data_meta 繞一圈後排程走樣: {msg}"
    print(f"✓ data_meta_is_sufficient_to_reproduce_plan_step ({len(from_ctx)} 步全等)")


def test_plan_step_ctx_reads_are_covered_by_data_meta():
    """反漂移哨兵：`_plan_step` 對 ctx 的讀取面必須與 data_meta 的收錄範圍一致。

    Why: `data_meta` 存在的理由就是「replay 端不必猜 `_plan_step` 要什麼」。
    若哪天 `_plan_step` 多讀一個 ctx 欄位而沒人動 `_data_meta`，本機 replay
    會拿到 None 然後炸在一個看不出原因的地方（或更糟，拿到能跑的預設值）。
    這條逼作者當場決定：新欄位是 data 衍生（要錄）還是 config 衍生（重解）。
    """
    import ast
    src = (REPO_ROOT / "pi_lnn_jax" / "pipeline" / "kolmogorov" / "run.py").read_text()
    fn = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "_plan_step"
    )
    reads = {
        n.attr for n in ast.walk(fn)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name) and n.value.id == "ctx"
    }
    assert reads == PLAN_STEP_CTX_FIELDS, (
        f"_plan_step 讀取的 ctx 欄位變了：多={sorted(reads - PLAN_STEP_CTX_FIELDS)} "
        f"少={sorted(PLAN_STEP_CTX_FIELDS - reads)} —— 請同步 _data_meta / "
        f"_ctx_from_data_meta 與本檔的 PLAN_STEP_CTX_FIELDS"
    )
    print(f"✓ plan_step_ctx_reads_are_covered_by_data_meta ({len(reads)} 欄位)")


def _is_jax_random_call(call) -> bool:
    """True 若 `call.func` 是以 `jax.random` 為根的 `ast.Attribute` 鏈。

    涵蓋 `jax.random.split(...)`、`jax.random.uniform(...)` 等任意深度的
    屬性鏈，只要根是 `jax` 且第一層屬性是 `random`。
    """
    import ast
    node = call.func
    attrs = []
    while isinstance(node, ast.Attribute):
        attrs.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return False
    attrs.append(node.id)
    attrs.reverse()
    return len(attrs) >= 2 and attrs[0] == "jax" and attrs[1] == "random"


def _plan_step_result_names(func) -> set:
    """回傳函式中被指派為 `_plan_step(...)` 呼叫結果的變數名集合。"""
    import ast
    names = set()
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_plan_step"
        ):
            names.add(node.targets[0].id)
    return names


def _has_replace_on_names(func, names: set) -> bool:
    """True 若函式內對 `names` 中任一變數呼叫過 `._replace(...)`。"""
    import ast
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_replace"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in names
        ):
            return True
    return False


def test_replay_schedule_shares_plan_step_with_run_loop():
    """反漂移哨兵：`replay_schedule` 與 `run_loop` 必須呼叫同一個 `_plan_step`。

    Why 用測試而不是 code review 記得：兩份排程程式碼不會 crash，只會讓
    replay 安靜地不再測真正在跑的東西（`4792b93` 治的病）。

    Why 只驗證「呼叫了 `_plan_step`」不夠：caller 可以呼叫它、然後在其後
    分岔 —— 例如 `plan = plan._replace(...)` 事後改動排程結果，或在
    `replay_schedule` 內部另外消費一次 `jax.random.*`（RNG 不對齊）。
    這兩種漂移都不會被「呼叫了 helper」這條斷言抓到，故在此對
    `replay_schedule` 額外收斂：它不得有自己的 RNG 消費，也不得改動
    `_plan_step` 的回傳值。
    """
    import ast
    src = (REPO_ROOT / "pi_lnn_jax" / "pipeline" / "kolmogorov" / "run.py").read_text()
    tree = ast.parse(src)
    callers = {}
    funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("run_loop", "replay_schedule"):
            callers[node.name] = {
                c.func.id for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            }
            funcs[node.name] = node
    assert set(callers) == {"run_loop", "replay_schedule"}
    for fn in ("_plan_step", "_ledger_fields", "_open_loop_streams"):
        for caller, calls in callers.items():
            assert fn in calls, f"{caller}() 沒有呼叫 {fn}() —— 排程程式碼疑似被複製一份"

    replay_fn = funcs["replay_schedule"]
    jax_random_calls = [
        c for c in ast.walk(replay_fn)
        if isinstance(c, ast.Call) and _is_jax_random_call(c)
    ]
    assert not jax_random_calls, (
        "replay_schedule() 內直接呼叫 jax.random.* —— replay 不得消費自己的 RNG，"
        "任何額外消費都會讓 rng_collo/rng_re/rng_crp 與 run_loop 失去對齊"
    )
    plan_names = _plan_step_result_names(replay_fn)
    assert plan_names, "replay_schedule() 找不到 `_plan_step(...)` 的指派目標"
    assert not _has_replace_on_names(replay_fn, plan_names), (
        "replay_schedule() 對 _plan_step() 的回傳值呼叫 _replace —— "
        "即使呼叫了共用 helper，事後改動 plan 仍會讓 replay 悄悄偏離 run_loop"
    )
    print("✓ replay_schedule_shares_plan_step_with_run_loop")


def test_env_ledger_flag_is_off_by_default():
    """production 路徑零 overhead：ledger 預設關閉。"""
    from pi_lnn_jax.pipeline._ledger import ledger_enabled
    if os.environ.get("PILNN_RNG_LEDGER"):
        pytest.skip("環境已開啟 PILNN_RNG_LEDGER")
    assert ledger_enabled() is False
    print("✓ env_ledger_flag_is_off_by_default")

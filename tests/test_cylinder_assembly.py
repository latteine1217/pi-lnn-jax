"""Structural tests for cylinder 的建構期 assembly layer。

不跑 forward pass（本機禁跑 training）。守的是「建構期 vs 執行期」那條邊界：
assembly 只組裝依賴，任何 RNG／取樣／state 初始化都屬執行期。邊界腐化不會
crash，只會讓 bit-identical 契約無聲失效——而 cylinder 有論文證據綁著
（07b_cylinder_feasibility.tex、appendix/C_cylinder.tex），所以用測試釘住
而不是靠 code review 記得。

與 tests/test_pipeline_assembly.py（Kolmogorov 版）同形；掃描對象換成
`pi_lnn_jax/pipeline/cylinder/assembly.py`。兩檔刻意不合併成參數化：
兩個 case 的「不得出現什麼」清單不同（Kolmogorov 另有 build_model 哨兵；
cylinder 反而**必須**繞過 model_factory，見 spec §6）。

Spec: knowledge/superpowers/specs/2026-07-30-pipeline-wave2-cylinder-design.md §5.1、§7-B
"""
from __future__ import annotations

import ast

import _boundary_scan

import pytest
from _paths import REPO_ROOT

from pi_lnn_jax.pipeline.cylinder.assembly import TrainingContext

ASSEMBLY = REPO_ROOT / "pi_lnn_jax/pipeline/cylinder/assembly.py"

# spec §5.1/§5.2：RNG 消費一律不得出現在建構期
#: 禁止的屬性前綴取自共用件，不在此重抄。
#: 先前兩個 adopter 各留一份 2 條目的私有副本，而共用件有 3 條目——
#: 實測 `import numpy; numpy.random.rand()` 被共用件抓到、被兩份私有副本漏掉，
#: 而這個守衛正是在保護 bit-identical 的 RNG 契約。共用件存在卻沒被用，
#: 它帶的修正就到不了呼叫端。
FORBIDDEN_ATTRS = _boundary_scan.FORBIDDEN_ATTR_PREFIXES

#: 建構期禁止出現的 RNG import 形式（含 alias 繞過）。
#: 這份清單同時被 test_rng_scan_detects_aliased_imports 拿來當**正向**樣本，
#: 證明掃描器真的會對每一種寫法開火，而不是「剛好沒東西可抓」。
RNG_SMELLS = _boundary_scan.RNG_SMELLS






def test_assembly_contains_no_rng_usage():
    """建構期不得消耗任何隨機源。

    cylinder 的三個 RNG 消費點（spec §5.1/§5.2）全部屬執行期：
      - `np.random.RandomState(seed)` 連抽 init_xy / init_t（**單一**生成器）
      - `model.init(jax.random.PRNGKey(seed), …)`
      - 迴圈的 `PRNGKey(seed+1)` 與 `RandomState(seed+7)`
    """
    hits = _boundary_scan.rng_hits(ASSEMBLY.read_text())
    assert not hits, (
        "cylinder/assembly.py 出現 runtime decision（RNG）——邊界已失敗，必須移回執行期:\n"
        + "\n".join(f"  line {ln}: {name}" for ln, name in hits)
    )
    print("✓ assembly_contains_no_rng_usage")


@pytest.mark.parametrize("smell", RNG_SMELLS)
def test_rng_scan_detects_aliased_imports(smell):
    """掃描器的自我證明：把 RNG 用法貼進 assembly 原始碼，必須被抓到。

    Why: 「沒有 hit」有兩種解釋——邊界乾淨，或掃描器根本沒在看。這條把後者排除，
    特別是 alias 形式（`from jax import random as jr`）：那正是繞過屬性鏈檢查、
    最容易在重構中被寫出來的一種。
    """
    contaminated = ASSEMBLY.read_text() + f"\n\ndef _contaminated():\n    {smell}\n"
    hits = _boundary_scan.rng_hits(contaminated)
    assert hits, f"掃描器沒抓到 {smell!r} —— 邊界測試等於沒有"


def test_assembly_does_not_init_model_params():
    """`model.init` 消耗主 RNG（spec §5.1），`tx.init(params)` 需要 params——
    兩者都屬執行期，不得出現在建構期。"""
    src = ASSEMBLY.read_text()
    assert ".init(" not in src, (
        "cylinder/assembly.py 呼叫了 .init(——model.init 消耗主 RNG、"
        "tx.init 需要 params，兩者都屬執行期"
    )
    print("✓ assembly_does_not_init_model_params")


def test_training_context_carries_no_params():
    """TrainingContext 不得持有 params / opt_state——可變狀態全部屬執行期。"""
    fields = set(TrainingContext._fields)
    assert "params" not in fields, "TrainingContext 不得持有 params（spec §5.1）"
    assert "opt_state" not in fields, "TrainingContext 不得持有 opt_state"
    # weighting / AL state 同理：gradnorm_init / al_init 都在執行期
    assert "gn_state" not in fields and "al_state" not in fields
    assert "task_weights" not in fields
    print("✓ training_context_carries_no_params")


@pytest.mark.parametrize("field", [
    # 輸入 + 原始資料
        "config", "npz",
    # sensor 張量與正規化統計
    "sv_TKC", "sp", "st", "obs_mean", "obs_std", "re_norm",
    # 幾何與域
    "Lx", "Ly", "bcen", "br", "geom", "u_inf",
    # 時間軸衍生純量
    "st0", "T_total", "tm_end",
    # 形狀 / 取樣上界
    "K", "T", "n_sensor_q", "t_q_full_np",
    # 建構好的可呼叫物與 optimizer
    "model", "loss_fn", "data_loss_fn", "get_subtree", "gn_ref_path",
    "grad_norm_fn", "tx", "opt_info", "step_fn",
])
def test_training_context_has_expected_field(field):
    """欄位齊全度——少一個欄位代表某段建構期程式碼沒搬進來。"""
    assert field in TrainingContext._fields, f"TrainingContext 缺欄位 {field!r}"


def test_gradnorm_ref_path_is_construction_time_constant():
    """spec §6【不修 3】：ref path 硬寫 `("temporal_encoder",)`，與 Kolmogorov 的
    `trunk_out` 不同，兩者都是各自的既有行為。

    path 常數屬建構期（由 ctx 提供）；印出「解析到哪個子樹」的 probe 要
    `jax.grad(...)` 吃真實 params，屬執行期——上面的 no-RNG / no-.init 兩條
    已經擋住 probe 被搬進來，這條再釘住常數值本身不被「統一」掉。
    """
    src = ASSEMBLY.read_text()
    tree = ast.parse(src)
    consts = [n for n in ast.walk(tree)
              if isinstance(n, ast.Assign)
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "REF_PATH"]
    assert len(consts) == 1, "REF_PATH 應在 build_context 內恰好定義一次"
    assert ast.unparse(consts[0].value) == "('temporal_encoder',)"
    print("✓ gradnorm_ref_path_is_construction_time_constant")


def test_assembly_preserves_the_two_documented_bypasses():
    """spec §6【不修 1、2】：兩個「繞過專案慣例」的既有行為必須原樣保留。

    1. `np.load(相對 CWD 路徑)` 繞過 `_resolve_data_path`：改走專案解析會把
       「找不到就炸」變成「往別處找」。
    2. 直建 `LiquidOperator(**CFG, …, torch_style_init=True)` 繞過
       `model_factory.build_model`（factory 的預設不同）。

    這兩條與 Kolmogorov 的哨兵**方向相反**（那邊禁止長回本地 build_model），
    所以兩個測試檔不合併。

    負面條件用 AST 而非子字串：兩個名字都會出現在本檔的「為何不修」註解裡，
    子字串檢查會把說明文字當成違規。
    """
    src = ASSEMBLY.read_text()
    tree = ast.parse(src)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    modules = {n.module for n in ast.walk(tree)
               if isinstance(n, ast.ImportFrom) and n.module}

    assert "np.load(npz_path)" in src, "資料載入不得改走 _resolve_data_path"
    assert "_resolve_data_path" not in used
    assert not any("model_factory" in m for m in modules), (
        "cylinder 不得收斂到 model_factory.build_model（factory 預設不同 → 行為變更）"
    )
    assert "build_model" not in used
    assert "LiquidOperator(" in src and "torch_style_init=True" in src
    print("✓ assembly_preserves_the_two_documented_bypasses")


# ─────────────────────────────────────────────────────────────────────────────
# 模型建構規格（spec §8.3 第 6 項）
#
# 缺口：本機測試對 params 為零覆蓋——審查者翻掉一個承重的初始化旗標，整套測試
# 不動。`CFG` 的十五個值與四個具名建構參數**只存在於 assembly.py 這幾行**：
# 不受 config schema 驗證、不在任何 TOML 裡、無預設值可回退。
#
# 為何不直接比對 fixture 尾列的 params_digest（spec 原本的要求）：
# 實測 params 只依賴 seed + CFG + 建構旗標（與 T/K/T_total 無關），
# 故本機重建沒有可填錯的自由度；但本機（arm64）算出 7ee61263e0cf0d5a，
# fixture（lab-server，x86_64）錄的是 b4549f5078513bbc。jax/jaxlib 同為 0.10.1
# 且共用同一份 uv.lock，版本漂移已排除 → 跨架構是存活的假設，與 §7.1 的
# float32 `cos` 殘差同源。digest 的**值**因此無法在本機當判準，
# 於是這裡改釘三樣平台無關的東西，並由第四條證明前三條守的是真的東西。
# ─────────────────────────────────────────────────────────────────────────────

#: `LiquidOperator(**CFG, …)` 那一行的非-CFG 具名參數。值是 AST unparse 後的原始碼字串
#: ——比對原始碼而非執行結果，才抓得到「改成一個剛好等值的變數」這種漂移。
EXPECTED_CTOR_KWARGS = {
    "relpos_bias_mode": "'radial'",
    "periodic_domain": "False",
    "use_sdf_features": "False",
    "T_total": "float(st[-1])",
    "torch_style_init": "True",
}

#: CEXP-002 的模型規格。改動任一項都會改變 params，故等同改論文數字。
EXPECTED_CFG = {
    "sensor_value_dim": 2, "d_model": 256, "d_time": 16,
    "num_spatial_encoder_layers": 1, "num_temporal_cfc_layers": 1,
    "num_token_attention_layers": 2, "token_attention_heads": 4,
    "num_query_mlp_layers": 1, "query_mlp_hidden_dim": 256, "operator_rank": 256,
    "decoder_attention_heads": 4, "use_temporal_anchor": True,
    "temporal_anchor_harmonics": 2, "domain_length": 1.0, "fourier_embed_dim": 128,
    # 2026-09-03 起顯式（值與 models.py 當時的 dataclass 預設相同）。CFG 未指定的
    # 鍵會落到那個**會變的**預設，該次變更讓 params 悄悄 +4.2%。釘在這裡之後，
    # 改 dataclass 預設不再動到 cylinder，而改 CFG 一定會讓本測試報紅。
    "use_rwf": True, "cfc_input_dependent_tau": True, "cfc_tau_mod_scale": 0.5,
}

#: 參數總數是結構指紋：與浮點值無關，故跨架構成立——digest 不成立的地方它成立。
#:
#: 2026-09-03 由 3_139_146 改為 3_272_010（+132,864, +4.2%）。**不是放寬，是模型
#: 真的變了**：`models.py` 的 dataclass 預設對齊 schema（`use_rwf` 與
#: `cfc_input_dependent_tau` 皆 False→True），而當時 cylinder 走 `assembly.CFG`、
#: CFG 不指定這兩鍵，所以吃到新預設。（同日起 CFG 已顯式寫死三鍵，這條路徑
#: 因此關閉——見上方 EXPECTED_CFG。）實測歸因：
#:     cfc_input_dependent_tau  +131,328  (98.8%)   ← tau modulation 子網路
#:     use_rwf                    +1,536   (1.2%)   ← rwf_g 向量
#:     cfc_tau_mod_scale              +0            ← 純數值，同批 2.0→0.5
#: ⚠️ 佔 98.8% 的那個鍵**沒有任何實驗證據支持**（Kolmogorov 兩設定三 scale 皆無
#: 效果、KE 軸更差；cylinder 從未驗過）。它開著是裁決結果——見
#: `knowledge/codebase/technical-debt.md` TD-4。cylinder 的既有數字全部產生於
#: 舊結構，**要重跑才能沿用**。
EXPECTED_N_PARAMS = 3_272_010


def _liquid_operator_call(tree: ast.AST) -> ast.Call:
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "LiquidOperator"]
    assert len(calls) == 1, f"assembly 應恰有一處 LiquidOperator(…)，實得 {len(calls)}"
    return calls[0]


def test_model_ctor_kwargs_are_pinned():
    """四個具名建構參數逐字釘住。

    既有的 `"torch_style_init=True" in src` 是子字串比對——註解裡寫一次就滿足。
    這條改用 AST 取實際呼叫的關鍵字，且四個都涵蓋（原本只有一個）。
    """
    call = _liquid_operator_call(ast.parse(ASSEMBLY.read_text()))
    got = {k.arg: ast.unparse(k.value) for k in call.keywords if k.arg is not None}
    assert got == EXPECTED_CTOR_KWARGS, (
        "LiquidOperator 的建構參數已變動——這會改變 params，等同改論文數字：\n"
        f"  預期 {EXPECTED_CTOR_KWARGS}\n  實得 {got}")
    assert any(k.arg is None for k in call.keywords), "CFG 必須以 **CFG 展開"
    print("✓ model_ctor_kwargs_are_pinned")


def test_cfg_values_are_pinned():
    """`CFG` 是硬寫常數，不經 config schema 驗證——沒有測試就沒有任何守衛。"""
    from pi_lnn_jax.pipeline.cylinder.assembly import CFG

    assert CFG == EXPECTED_CFG, (
        f"CFG 已變動：\n  多/改 {set(CFG.items()) - set(EXPECTED_CFG.items())}\n"
        f"  缺/舊 {set(EXPECTED_CFG.items()) - set(CFG.items())}")
    print("✓ cfg_values_are_pinned")


def _init_params(**over):
    """依 assembly 的規格建模並初始化，回傳 (digest, n_params)。

    形狀刻意取極小值：實測 params 與 T/K/T_total 無關（只依賴 seed + CFG + 旗標），
    小形狀不減損覆蓋卻讓這條測試從數秒降到可接受。
    """
    import jax
    import jax.numpy as jnp

    from pi_lnn_jax.models import LiquidOperator
    from pi_lnn_jax.pipeline._ledger import params_digest
    from pi_lnn_jax.pipeline.cylinder.assembly import CFG

    ctor = dict(relpos_bias_mode="radial", periodic_domain=False,
                use_sdf_features=False, T_total=20.0, torch_style_init=True)
    cfg = {**CFG, **{k: v for k, v in over.items() if k in CFG}}
    ctor.update({k: v for k, v in over.items() if k not in CFG})
    model = LiquidOperator(**cfg, **ctor)
    params = model.init(
        jax.random.PRNGKey(42),
        jnp.zeros((8, 10, 2), jnp.float32), jnp.zeros((10, 2), jnp.float32),
        jnp.float32(0.5), jnp.linspace(0, 1, 8, dtype=jnp.float32),
        jnp.zeros((8, 2), jnp.float32), jnp.zeros((8,), jnp.float32))
    return params_digest(params), sum(
        p.size for p in jax.tree_util.tree_leaves(params))


def test_model_param_count_is_pinned():
    """結構指紋。digest 跨架構不成立，參數總數成立——models.py 動到層數、
    寬度或多接一個子模組，這條會紅。"""
    _, n = _init_params()

    assert n == EXPECTED_N_PARAMS, (
        f"參數量 {n:,} ≠ 預期 {EXPECTED_N_PARAMS:,}——模型結構已變")
    print("✓ model_param_count_is_pinned")


@pytest.mark.parametrize("override", [
    {"torch_style_init": False},
    {"periodic_domain": True},
    {"use_sdf_features": True},
    {"d_model": 128},
    {"operator_rank": 128},
])
def test_pinned_construction_is_load_bearing(override):
    """自證：上面兩條釘住的東西，改了確實會改變 params。

    否則「CFG 沒變」只是在守一組不影響任何結果的裝飾字。同機比對，
    不涉及跨架構，故本機成立。

    刻意不含 `relpos_bias_mode`：實測只有 `"vector"` 會分支
    （models.py:845），`'radial'` 與其他值給出完全相同的 digest。
    它仍被 `test_model_ctor_kwargs_are_pinned` 釘住（改成 `"vector"` 會被抓），
    但把它列進這裡會是一條假的自證。
    """
    base, _ = _init_params()
    got, _ = _init_params(**override)

    assert got != base, (
        f"{override} 未改變 params digest——它不是承重的，"
        "釘住它的測試因此是空守衛（或這條自證寫錯了）")


# ─────────────────────────────────────────────────────────────────────────────
# 邊界守衛的兩個已知繞道（spec §8.3 第 5 項）
# ─────────────────────────────────────────────────────────────────────────────

def test_assembly_does_not_reach_rng_through_another_module():
    """繞道一：不寫 `jax.random`，改匯入／呼叫別的模組裡會抽的 helper。

    既有掃描器只看屬性鏈與 import 形式，
    `from pi_lnn_jax.curriculum import rar_init` 完全合法地通過。
    denylist 由原始碼推導（非手寫），故新增取樣 helper 時自動納入。
    """
    denylist = _boundary_scan.rng_consuming_functions(REPO_ROOT / "pi_lnn_jax")
    assert denylist, "denylist 推導不出任何函式——掃描器沒在看"
    hits = _boundary_scan.indirect_rng_uses(ast.parse(ASSEMBLY.read_text()), denylist)

    assert not hits, (
        "建構期經由其他模組的 helper 消耗 RNG：\n"
        + "\n".join(f"  line {ln}: {w}" for ln, w in hits))
    print("✓ assembly_does_not_reach_rng_through_another_module")


def test_rng_helper_closure_is_flat():
    """上一條只做一層的前提：傳遞閉包不增加任何函式。前提失效時這裡先紅。"""
    root = REPO_ROOT / "pi_lnn_jax"
    direct = _boundary_scan.rng_consuming_functions(root)

    assert _boundary_scan.rng_helper_call_closure(root) == direct, (
        "有函式只是「呼叫」RNG helper 而自己不碰 RNG——"
        "單層 denylist 已不完備，indirect_rng_uses 需改吃閉包")


@pytest.mark.parametrize("smell", [
    "from pi_lnn_jax.curriculum import rar_init",
    "x = rar_init(seed=0)",
    "x = sample_wall_bc(k, 1)",
])
def test_indirect_rng_scan_detects_helper_bypass(smell):
    """掃描器自證：把繞道寫法貼進原始碼，必須被抓到。"""
    denylist = _boundary_scan.rng_consuming_functions(REPO_ROOT / "pi_lnn_jax")
    contaminated = ASSEMBLY.read_text() + f"\n\ndef _contaminated():\n    {smell}\n"

    assert _boundary_scan.indirect_rng_uses(ast.parse(contaminated), denylist), (
        f"掃描器沒抓到 {smell!r} —— 這條守衛等於沒有")


def test_assembly_never_touches_dot_init_even_via_a_local_alias():
    """繞道二：`init_fn = model.init` 再 `init_fn(key, …)` 避開 `".init(" in src`。

    改用 AST 找 `.init` 屬性**存取**：綁成區域名也得先存取一次，躲不掉。
    """
    uses = _boundary_scan.init_attribute_uses(ast.parse(ASSEMBLY.read_text()))

    assert not uses, (
        "建構期出現 .init —— model.init 消耗主 RNG、tx.init 需要 params，"
        "兩者都屬執行期:\n" + "\n".join(f"  line {ln}: {w}" for ln, w in uses))
    print("✓ assembly_never_touches_dot_init_even_via_a_local_alias")


@pytest.mark.parametrize("smell", [
    "params = model.init(key, x)",
    "init_fn = model.init",
    "opt_state = tx.init(params)",
])
def test_init_scan_detects_alias_binding(smell):
    """掃描器自證：三種寫法都必須被抓到，包含只綁不呼叫的那一種。"""
    contaminated = ASSEMBLY.read_text() + f"\n\ndef _contaminated():\n    {smell}\n"

    assert _boundary_scan.init_attribute_uses(ast.parse(contaminated)), (
        f"掃描器沒抓到 {smell!r} —— `.init` 守衛可被繞過")

"""Structural tests for the construction-time assembly layer.

不跑 forward pass（本機禁跑 training）。這些測試守的是「建構期 vs 執行期」
那條邊界：assembly 只組裝依賴，任何 RNG／取樣／排程都屬於 run.py。
邊界腐化不會 crash，只會讓 bit-identical 契約無聲失效，所以用測試釘住
而不是靠 code review 記得。

Spec: knowledge/superpowers/specs/2026-07-29-training-pipeline-design.md §4.2
"""
from __future__ import annotations

import ast

import _boundary_scan

import pytest
from _paths import REPO_ROOT

from pi_lnn_jax.pipeline.kolmogorov.assembly import TrainingContext

ASSEMBLY = REPO_ROOT / "pi_lnn_jax/pipeline/kolmogorov/assembly.py"

# spec §4.2：runtime decision 一律不得出現在建構期
#: 禁止的屬性前綴取自共用件，不在此重抄。
#: 先前兩個 adopter 各留一份 2 條目的私有副本，而共用件有 3 條目——
#: 實測 `import numpy; numpy.random.rand()` 被共用件抓到、被兩份私有副本漏掉，
#: 而這個守衛正是在保護 bit-identical 的 RNG 契約。共用件存在卻沒被用，
#: 它帶的修正就到不了呼叫端。
FORBIDDEN_ATTRS = _boundary_scan.FORBIDDEN_ATTR_PREFIXES




def test_assembly_contains_no_rng_usage():
    """建構期不得消耗任何隨機源。

    用 AST 而非 grep：字串或註解裡出現 'jax.random' 不該算違規，
    真正的屬性存取才算。

    同時檢查 import 陳述，捕捉 alias 繞過：
    - import jax.random as jr
    - from jax import random
    - import numpy.random as npr
    """
    hits = _boundary_scan.rng_hits(ASSEMBLY.read_text(), FORBIDDEN_ATTRS)

    assert not hits, (
        "assembly.py 出現 runtime decision（RNG）——邊界已失敗，必須移回 run.py:\n"
        + "\n".join(f"  line {ln}: {name}" for ln, name in hits)
    )
    print("✓ assembly_contains_no_rng_usage")


def test_assembly_does_not_init_model_params():
    """model.init 消耗主 RNG（spec §5.1 第 1 項），屬執行期，必須留在 run.initialize。"""
    src = ASSEMBLY.read_text()
    assert ".init(" not in src, (
        "assembly.py 呼叫了 .init(——model.init 消耗主 RNG，屬 run.initialize"
    )
    print("✓ assembly_does_not_init_model_params")


def test_assembly_does_not_reimplement_build_model():
    """build_model 已由 4792b93 收斂到 model_factory；不得長回本地副本。"""
    src = ASSEMBLY.read_text()
    assert "def build_model" not in src and "def _build_model" not in src
    assert "model_factory" in src, "assembly 應改為呼叫 pi_lnn_jax.model_factory"
    print("✓ assembly_does_not_reimplement_build_model")


def test_training_context_carries_no_params():
    """TrainingContext 不得持有 params——可變狀態全部屬於 run.TrainingState。"""
    fields = set(TrainingContext._fields)
    assert "params" not in fields, "TrainingContext 不得持有 params（spec §4.2）"
    assert "opt_state" not in fields, "TrainingContext 不得持有 opt_state"
    print("✓ training_context_carries_no_params")


@pytest.mark.parametrize("field", [
        "config", "model", "model_name", "datasets", "re_batches",
    "re_t_min_host", "re_t_max_host", "crp_interp", "crp_re_norm_scale",
    "dns_u", "dns_v", "dns_t",
    "ns_fn", "poisson_fn", "ns_fn_baseline",
    "loss_fn", "grad_norm_fn", "gn_ref_path",
    "tx", "opt_info", "step_fn", "ckpt_mgr",
    "artifacts_dir", "ckpt_dir",
        "use_poisson", "T_total", "n_sensor_query",
])
def test_training_context_has_expected_field(field):
    """欄位齊全度——少一個欄位代表某段建構期程式碼沒搬進來。"""
    assert field in TrainingContext._fields, f"TrainingContext 缺欄位 {field!r}"


def test_artifacts_paths_derived_only_in_assembly():
    """`artifacts_dir` / `ckpt_dir` 只准在 assembly 推導一次。

    Why 用測試守：兩處各自算 `Path(eff["artifacts_dir"]).resolve() / "checkpoints"`
    今天值相同，所以壞掉時不會 crash，只會安靜地把 ckpt 寫到另一個地方——
    「artifacts_dir 未對齊」是本專案記錄有案的 eval 失敗模式，而
    `build_model` 雙份副本已經燒過一次同樣的形狀。
    """
    offenders = []
    for path in (REPO_ROOT / "pi_lnn_jax/pipeline/kolmogorov/run.py",
                 REPO_ROOT / "train_kolmogorov.py"):
        src = path.read_text()
        for lineno, line in enumerate(src.splitlines(), start=1):
            if '"checkpoints"' in line or "'checkpoints'" in line:
                offenders.append(f"  {path}:{lineno}: {line.strip()}")
            elif 'eff["artifacts_dir"]' in line:
                offenders.append(f"  {path}:{lineno}: {line.strip()}")
    assert not offenders, (
        "產物路徑在 assembly 之外被重新推導；請改讀 ctx.artifacts_dir / ctx.ckpt_dir:\n"
        + "\n".join(offenders)
    )
    print("✓ artifacts_paths_derived_only_in_assembly")


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


@pytest.mark.parametrize("smell", _boundary_scan.RNG_SMELLS)
def test_rng_scan_detects_aliased_imports(smell):
    """掃描器自證：把繞道寫法貼進原始碼，必須被抓到。

    先前只有 cylinder 有這份正樣本；Kolmogorov 的屬性鏈／import 掃描因此
    沒有任何東西證明它會開火——`_boundary_scan` 的 module docstring 點名的
    正是這個差距。清單現在共用，兩案覆蓋同一組繞道形式。
    """
    contaminated = ASSEMBLY.read_text() + f"\n\ndef _contaminated():\n    {smell}\n"
    assert _boundary_scan.rng_hits(contaminated, FORBIDDEN_ATTRS), (
        f"掃描器沒抓到 {smell!r} —— 邊界測試等於沒有")

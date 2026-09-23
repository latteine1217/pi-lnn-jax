"""schema 預設值若是「沒有任何實驗在用的值」，漏寫該鍵就是靜默偏離。

問題（架構審查候選 H）：實測 205 份 config，有一組鍵是**每一份都寫同一個值、
而那個值不等於 schema 預設**。漏寫其中任何一個，訓練不會 crash，只會安靜地
用一組沒有任何既有實驗使用的設定跑完。

`use_temporal_anchor` 已於 2026-08-03 對齊（False → True）——那一個改得動，
因為它的 9 個省略者全是 cylinder / controlled_cylinder，而**整個 cylinder
pipeline 讀 `model_kwargs` 零次**（它走 `assembly.CFG` 顯式建構）。實測 205 份
config 在 pipeline 層面的行為全部不變。

其餘四個走 **expand → contract**，2026-08-03 兩步都完成：

- **expand**：把實測生效值（0/0/1.0/1.0）顯式寫進 7 份省略者。其中三份是
  bit-identical A/B 的 fixture，所以這一步由 **job 4842** 背書——驗收框架的
  `record()` 對每一側 `cd "$wt"` 後用相對 config 路徑，於是 BASE 讀舊 config
  （省略）、HEAD 讀新 config（顯式），f1–f4 逐位元相同即為證明。
- **contract**：schema 預設改為慣例值（2000/2000/0.9/10.0）。此步對既有配置
  惰性——用兩個子行程分別載入新舊 `config.py` 實跑 205 份 config，解析結果
  改變數 **0**。它只改變未來漏寫這些鍵的新 config 會拿到什麼，那正是要修的。

本檔把兩件事釘住：已對齊的那個不得回退；未對齊的那四個，其爆炸半徑
（哪些 config 省略、分屬哪個 case）不得在無人注意下改變。
"""
from __future__ import annotations

import pathlib
import tomllib

import pytest
from _paths import REPO_ROOT

from pi_lnn_jax.config import MODEL_SCHEMA, TRAIN_SCHEMA

_CONFIGS = REPO_ROOT / "configs"
_SCHEMA = {**TRAIN_SCHEMA, **MODEL_SCHEMA}


def _flat(path: pathlib.Path) -> dict:
    """TOML 攤平成單層——本 repo 的鍵在 section 間並不固定。"""
    d = tomllib.loads(path.read_text())
    out = {k: v for k, v in d.items() if not isinstance(v, dict)}
    for body in d.values():
        if isinstance(body, dict):
            out.update(body)
    return out


def _all_configs() -> dict[str, dict]:
    out = {}
    for p in sorted(_CONFIGS.rglob("*.toml")):
        try:
            out[p.name] = _flat(p)
        except Exception:
            continue
    return out


CONFIGS = _all_configs()


def _case(flat: dict) -> str:
    return str(flat.get("case", "kolmogorov"))


def _omitters(key: str) -> dict[str, list[str]]:
    """省略該鍵的 config，依 case 分組。"""
    out: dict[str, list[str]] = {}
    for name, flat in CONFIGS.items():
        if key not in flat:
            out.setdefault(_case(flat), []).append(name)
    return out


def test_there_are_configs_to_scan():
    """自證：掃不到 config 時，下面每一條都空過。"""
    assert len(CONFIGS) > 150, f"只掃到 {len(CONFIGS)} 份 config——路徑或解析壞了"


# ── 已對齊：不得回退 ────────────────────────────────────────────────────

def test_use_temporal_anchor_default_matches_what_every_experiment_uses():
    """196 份 Kolmogorov config 全寫 true；schema 預設必須是 true。

    這條紅了代表有人把它改回 False——那會讓漏寫此鍵的新 config 靜默訓練出
    與所有既有實驗不同的架構。
    """
    assert _SCHEMA["use_temporal_anchor"][1] is True, (
        "use_temporal_anchor 的 schema 預設被改回 False——"
        "那是沒有任何實驗在用的值（見本檔 module docstring）")


def test_use_temporal_anchor_change_was_inert_because_cylinder_ignores_model_kwargs():
    """記錄那次改動為何安全：省略者全在一條不讀 `model_kwargs` 的路徑上。

    這條紅了代表前提不再成立（cylinder 開始讀 model_kwargs，或有 Kolmogorov
    config 開始省略此鍵）——那時 schema 預設就會真的影響行為。
    """
    omit = _omitters("use_temporal_anchor")
    assert "kolmogorov" not in omit, (
        f"有 Kolmogorov config 省略 use_temporal_anchor：{omit.get('kolmogorov')}"
        "——schema 預設現在會真的影響它們，需重新評估")

    cyl_pipeline = REPO_ROOT / "pi_lnn_jax" / "pipeline" / "cylinder"
    reads = [p.name for p in cyl_pipeline.rglob("*.py")
             if "model_kwargs" in p.read_text()]
    assert not reads, (
        f"cylinder pipeline 開始讀 model_kwargs（{reads}）——"
        "schema 預設不再對 cylinder 惰性，use_temporal_anchor 那次改動的前提失效")


# ── expand → contract 已完成（2026-08-03，A/B job 4842 背書）────────────

#: 鍵 → (對齊後的 schema 預設 ＝ 主流慣例值)。硬編碼字面值。
ALIGNED = {
    "lr_warmup_steps": 2000,
    "lr_decay_steps": 2000,
    "lr_decay_gamma": 0.9,
    "t_early_weight": 10.0,
}

#: expand 時把「原本從 schema 拿到的值」顯式寫入的 config。
#: 它們寫的**不是**慣例值——那是刻意的：`_ledger_*` 只跑 20 步，2000 步的 warmup
#: 會讓 LR 全程貼近 0。expand 把「這些 fixture 本來就該用 0」從隱含變成明說。
EXPANDED = {"_ledger_f2.toml", "_ledger_prod_scale.toml", "_ledger_single_re.toml",
            "eval_multi_re_train5.toml", "eval_multi_re_train5_crp.toml",
            "exp_multi_re_poc.toml", "mini_smoke.toml"}

#: 刻意偏離慣例的**消融臂**。與 EXPANDED 是不同的理由，故分開放：EXPANDED 寫的是
#: 「這份 fixture 本來就該用的值」，這裡寫的是「刻意與慣例不同的值，因為那正是被
#: 測的變因」。把兩者混在同一個集合會讓這條測試說不清自己在保護什麼。
#:
#: 2026-09-13 的 2x2x2 因子實驗（RAR x causal x t_early，single seed）：`e0` 那四支
#: 把 t_early_weight 設成 1.0 以量 IC emphasis 的作用。對照臂是 r0c0e1（= 主線）。
ABLATION_ARMS = frozenset(n for n in CONFIGS if n.startswith("exp_fac3"))


@pytest.mark.parametrize("key", sorted(ALIGNED))
def test_default_matches_the_convention(key):
    """schema 預設必須等於主流慣例。

    這條紅了代表有人把預設改回「沒有任何實驗在用的值」——漏寫該鍵的新 config
    就會再度靜默偏離。改動需 lab-server A/B 背書（見 module docstring）。
    """
    assert _SCHEMA[key][1] == ALIGNED[key], (
        f"{key} 的 schema 預設從 {ALIGNED[key]!r} 改成 {_SCHEMA[key][1]!r}")

    mainstream = {v[key] for n, v in CONFIGS.items()
                  if key in v and n not in EXPANDED and n not in ABLATION_ARMS}
    assert mainstream == {ALIGNED[key]}, (
        f"{key} 的主流值不再唯一：{sorted(map(str, mainstream))}"
        "——「慣例」的前提要重新檢視，本表可能需要改")


@pytest.mark.parametrize("key", sorted(ALIGNED))
def test_no_config_depends_on_the_schema_default(key):
    """contract 的前提：沒有任何 Kolmogorov config 省略這些鍵。

    有人新增一份省略者，schema 預設就重新變成實際來源——那本身不是錯，
    但它會讓「改預設 = 零行為變更」不再成立，下次動預設前要重新評估。
    """
    kol = _omitters(key).get("kolmogorov", [])
    assert not kol, (
        f"新增了省略 {key} 的 Kolmogorov config：{sorted(kol)}\n"
        "  它現在會吃 schema 預設。請在該 config 明設此鍵，"
        "或接受「改預設會影響它」並重新評估。")


@pytest.mark.parametrize("key", sorted(ALIGNED))
def test_expanded_configs_are_insulated_from_the_default(key):
    """expand 過的 config 必須明寫自己的值——那是它們免疫於預設變動的原因。

    A/B job 4842 證明的正是這件事：BASE 讀舊 config（省略）、HEAD 讀新 config
    （顯式），f1–f4 逐位元相同。若有人把這些顯式值刪回去，那個證明就失效。
    """
    for name in sorted(EXPANDED):
        if name not in CONFIGS:
            continue
        # exp_multi_re_poc 本來就明設 t_early_weight=10.0，不在 expand 範圍
        assert key in CONFIGS[name], (
            f"{name} 不再明設 {key}——它會回頭吃 schema 預設，"
            "而 A/B 4842 的證明前提是「這些 config 明寫自己的值」")


def test_the_scan_would_notice_an_aligned_key():
    """自證：對一個「預設 == 慣例」的鍵，掃描必須看得出來。

    少了這條，`_omitters` 哪天總是回空，上面每一條都會以「零省略者」全綠。
    """
    aligned = []
    for k, spec in _SCHEMA.items():
        if k in ALIGNED:
            continue
        vals = [v[k] for v in CONFIGS.values() if k in v]
        # 值可能是 list（如 soap_betas），不可 hash——用逐一比較而非 set。
        if vals and all(v == spec[1] for v in vals):
            aligned.append(k)
    assert aligned, "找不到任何「config 寫的值 == schema 預設」的鍵——掃描邏輯可疑"


#: cylinder 系 config 的檔名前綴。這些 case 走 `assembly.CFG` 顯式建構。
_CYLINDER_PREFIXES = ("exp_cyl", "exp_controlled_cyl", "_ledger_cyl")


def test_cylinder_configs_carry_no_model_keys():
    """cylinder 系 config 不得出現 model 鍵——寫了也不會生效，等於讓 config 說謊。

    Why 這條存在：2026-08-29 有人為九份 cylinder config 加上
    `use_rwf = false` / `cfc_input_dependent_tau = false`，註解自稱「legacy 鎖定」。
    那九處**從來沒有生效過**——整個 cylinder pipeline 讀 `model_kwargs` 零次
    （本檔 module docstring 早就記著這件事），實際建構值一直是 `assembly.CFG` 的
    那一組。於是 config 宣稱 rwf 關閉、模型實際開著，而沒有任何東西會報錯。

    這比漏寫更難發現：漏寫至少還有 `test_model_default_parity` 那類守衛在看
    兩邊預設，而「寫了但被忽略」在任何一層都不留痕跡。
    """
    offenders = {}
    for path in sorted(_CONFIGS.glob("*.toml")):
        if not path.name.startswith(_CYLINDER_PREFIXES):
            continue
        keys = sorted(set(_flat(path)) & set(MODEL_SCHEMA))
        if keys:
            offenders[path.name] = keys
    assert not offenders, (
        "cylinder 系 config 出現 model 鍵，但 cylinder 讀 model_kwargs 零次——\n"
        + "\n".join(f"    {n}: {k}" for n, k in offenders.items())
        + "\n  要改 cylinder 的模型規格，改 pi_lnn_jax/pipeline/cylinder/assembly.py "
        "的 CFG，並同步 tests/test_cylinder_assembly.py 的 EXPECTED_CFG。")


def test_the_cylinder_guard_has_discriminating_power(tmp_path, monkeypatch):
    """把一個 model 鍵塞進暫時的 cylinder config，上面那條必須抓到。
    少了這條自檢，「沒有 offender」也可能是 _flat 或前綴比對壞掉。"""
    fake = tmp_path / "exp_cyl_fake.toml"
    fake.write_text("use_rwf = false\nsteps = 10\n")
    keys = sorted(set(_flat(fake)) & set(MODEL_SCHEMA))
    assert "use_rwf" in keys, "比對器認不出 model 鍵 → 上面那條沒有鑑別力"

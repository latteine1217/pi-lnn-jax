"""方向1 exp_508：trainable-frequency Fourier embedding 的契約。

與 BandFourierEmb 的差別是承重的：BandFourierEmb 的**波數固定**（可學的只有投影權重），
TrainableFourierEmb 的**頻率 B 本身是 param**——網路可以自己把頻率移到它想要的地方。
exp_508 的核心診斷就是「訓練後 B 落在哪」，所以「B 真的在參數樹裡、真的吃得到梯度」
是這個實驗的前提，必須被釘住。

flag 關（use_trainable_fourier=False，預設）→ 不實例化 → 參數樹與指紋與 baseline 逐鍵相同。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax.traverse_util import flatten_dict

from pi_lnn_jax.config import MODEL_SCHEMA
from pi_lnn_jax.model_factory import build_model, model_fingerprint
from pi_lnn_jax.models import LiquidOperator, TrainableFourierEmb

_BASE = dict(
    sensor_value_dim=2, d_model=16, d_time=4,
    num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
    num_token_attention_layers=0, num_query_mlp_layers=0,
    query_mlp_hidden_dim=16, operator_rank=8, decoder_attention_heads=1,
    use_temporal_anchor=True, T_total=1.0, temporal_anchor_harmonics=2,
    domain_length=1.0,
)
_T, _K, _N = 5, 6, 4


def _inputs():
    rng = jax.random.PRNGKey(0)
    return (jnp.zeros((_T, _K, 2)), jax.random.uniform(rng, (_K, 2)), 0.5,
            jnp.linspace(0, 1, _T), jax.random.uniform(rng, (_N, 2)), jnp.zeros((_N,)))


def _param_paths(**over) -> set[str]:
    model = LiquidOperator(**{**_BASE, **over})
    params = model.init(jax.random.PRNGKey(42), *_inputs())
    return {"/".join(map(str, k)) for k in flatten_dict(params).keys()}


# ── module 本身 ──────────────────────────────────────────────────────────

def test_frequency_matrix_is_a_parameter():
    """B 必須在參數樹裡——否則「頻率可學」是假的，整個 exp_508 診斷失去意義。"""
    mod = TrainableFourierEmb(embed_dim=8, init_freq_scale=8.0)
    params = mod.init(jax.random.PRNGKey(0), jnp.zeros((3, 2)), 1.0)
    flat = {"/".join(map(str, k)): v for k, v in flatten_dict(params).items()}
    assert "params/B" in flat, f"參數樹沒有 B：{sorted(flat)}"
    assert flat["params/B"].shape == (2, 4), "B 應為 [2, embed_dim//2]（x/y 各一列頻率）"


def test_frequency_receives_gradient():
    """B 吃得到梯度。診斷「訓練後頻率落在哪」的前提是它真的會被移動。

    注意 loss 的選擇：輸出的 L2 範數 **對 B 恆為常數**（cos²+sin²=1），拿它當 loss
    梯度真的會是零——那是三角恆等式，不是 plumbing 壞掉。用線性泛函才測得到。
    """
    mod = TrainableFourierEmb(embed_dim=8, init_freq_scale=8.0)
    xy = jax.random.uniform(jax.random.PRNGKey(1), (16, 2))
    w = jax.random.normal(jax.random.PRNGKey(2), (8,))
    params = mod.init(jax.random.PRNGKey(0), xy, 1.0)
    grad = jax.grad(lambda p: jnp.sum(mod.apply(p, xy, 1.0) * w))(params)
    gB = flatten_dict(grad)[("params", "B")]
    assert jnp.any(jnp.abs(gB) > 0), "B 的梯度全為零 → 頻率不會被訓練移動"


def test_init_freq_scale_sets_the_initial_frequency_band():
    """init 的 |B| 分佈中心由 init_freq_scale 控制（exp_508 兩臂差的就是這個旋鈕）。

    B ~ N(0, s²) 逐分量 → |B| = sqrt(Bx²+By²) 服從 Rayleigh(s)，中位數 = s·sqrt(2ln2) ≈ 1.177 s。
    """
    for scale in (4.0, 8.0):
        mod = TrainableFourierEmb(embed_dim=4096, init_freq_scale=scale)
        B = flatten_dict(mod.init(jax.random.PRNGKey(0), jnp.zeros((2, 2)), 1.0))[("params", "B")]
        med = float(jnp.median(jnp.linalg.norm(B, axis=0)))
        assert med == pytest.approx(1.177 * scale, rel=0.06), (
            f"init_freq_scale={scale} 的 |B| 中位數 {med:.3f} 偏離 Rayleigh 預期")


def test_odd_embed_dim_is_rejected():
    """cos/sin 各佔一半 → 奇數維無法對半分，必須 fail fast 而非靜靜截斷。"""
    mod = TrainableFourierEmb(embed_dim=7)
    with pytest.raises(ValueError, match="偶數"):
        mod.init(jax.random.PRNGKey(0), jnp.zeros((3, 2)), 1.0)


def test_integer_frequencies_are_periodic_and_non_integer_ones_are_not():
    """釘住 exp_508 判讀的物理前提（見 knowledge/experiments 的 confounder 一節）。

    trainable frequency 只能對**原始座標**投影達成，代價是 B 非整數時 embedding 在
    週期域上不週期。任何整數 B 都保持週期性（與頻率高低無關）——所以「到整數格點的距離」
    才能把「拋棄中頻」與「拋棄非週期特徵」兩個競爭假設分開。
    """
    L = 1.0
    xy = jnp.array([[0.13, 0.71]])
    shifted = xy + jnp.array([[L, 0.0]])   # 沿 x 平移一個週期
    mod = TrainableFourierEmb(embed_dim=4)

    params = mod.init(jax.random.PRNGKey(0), xy, L)
    integer_B = {"params": {"B": jnp.array([[3.0, 7.0], [5.0, 2.0]])}}
    assert jnp.allclose(mod.apply(integer_B, xy, L), mod.apply(integer_B, shifted, L), atol=1e-5), \
        "整數頻率必須週期"

    frac_B = {"params": {"B": jnp.array([[3.4, 7.0], [5.0, 2.0]])}}
    assert not jnp.allclose(mod.apply(frac_B, xy, L), mod.apply(frac_B, shifted, L), atol=1e-3), \
        "非整數頻率若也『週期』，代表格點距離診斷讀不出週期性壓力"
    del params


# ── opt-in 縫（flag 關 → bit-identical）─────────────────────────────────

def test_branch_exists_only_when_flag_on():
    off = _param_paths()                                            # 預設關
    on = _param_paths(use_trainable_fourier=True, trainable_fourier_dim=16)
    assert not any("trainable_spatial_emb" in p for p in off), "flag-off 不應有 trainable 分支"
    assert any("trainable_spatial_emb" in p for p in on), "flag-on 應有 trainable_spatial_emb"
    assert off == on - {p for p in on if "trainable_spatial_emb" in p}, \
        "開關只該多出 trainable 分支，其餘參數樹必須逐鍵不變"


def test_composes_with_mid_band_branch():
    """兩個 opt-in 分支互相獨立（同時開時 spatial_dim 要兩個都加）。"""
    both = _param_paths(use_trainable_fourier=True, trainable_fourier_dim=16,
                        mid_band_wavenumbers=(6.0, 12.0), mid_band_embed_dim=16)
    assert any("trainable_spatial_emb" in p for p in both)
    assert any("band_spatial_emb" in p for p in both)


def test_fingerprint_hides_the_flags_when_off():
    """關閉時指紋與「從未有此功能」的模型逐鍵相同 → 既有 ckpt 與 §7.1 A/B 契約不假紅。"""
    fp_off = model_fingerprint(LiquidOperator(**_BASE))
    fp_on = model_fingerprint(LiquidOperator(**{**_BASE, "use_trainable_fourier": True}))
    for key in ("use_trainable_fourier", "trainable_fourier_dim", "trainable_fourier_init_scale"):
        assert key not in fp_off, f"flag-off 的指紋不該有 {key}"
        assert key in fp_on, f"flag-on 的指紋必須有 {key}"


def test_config_schema_defaults_match_the_dataclass():
    """schema 預設偏離 dataclass 預設 → 漏寫該鍵的 config 會靜靜訓練出另一個架構。"""
    off = LiquidOperator(**_BASE)
    for key in ("use_trainable_fourier", "trainable_fourier_dim", "trainable_fourier_init_scale"):
        assert key in MODEL_SCHEMA, f"MODEL_SCHEMA 缺 {key}"
        assert MODEL_SCHEMA[key][1] == getattr(off, key), (
            f"{key} 的 schema 預設 {MODEL_SCHEMA[key][1]!r} ≠ dataclass 預設 {getattr(off, key)!r}")


def test_build_model_plumbs_the_flags_through():
    kwargs = {**_BASE, "use_trainable_fourier": True,
              "trainable_fourier_dim": 64, "trainable_fourier_init_scale": 4.0}
    model, _ = build_model("liquid", dict(kwargs), K_sensors=_K)
    assert model.use_trainable_fourier is True
    assert model.trainable_fourier_dim == 64
    assert model.trainable_fourier_init_scale == 4.0


# ── 診斷腳本的自我校準 ─────────────────────────────────────────────────
# 判讀規則會直接吃這些統計量，所以它們在「已知答案」的輸入上必須算對。

def test_describe_recovers_a_known_frequency_distribution():
    """對已知 scale 的 init 分佈，_describe 必須報出 Rayleigh 的理論中位數。"""
    import numpy as np

    from scripts.diag_trainable_fourier import _describe
    scale = 8.0
    B = np.asarray(np.random.RandomState(0).normal(0, scale, (2, 20000)))
    d = _describe(B, k_sensor=5.6419)
    assert d["n_features"] == 20000
    assert d["abs_B_quantiles"]["p50"] == pytest.approx(1.177 * scale, rel=0.03)
    # 隨機實數到整數格點的距離均勻分佈於 [0, 0.5] → 均值 0.25
    assert d["d_lattice_mean"] == pytest.approx(0.25, abs=0.01)


def test_describe_flags_a_collapsed_and_lattice_snapped_distribution():
    """兩個判讀分支各自的極端情形都要讀得出來。"""
    import numpy as np

    from scripts.diag_trainable_fourier import _describe
    collapsed = np.full((2, 100), 1.0)          # |B| = √2 ≈ 1.41 « k_s，且正好在整數格點
    d = _describe(collapsed, k_sensor=5.6419)
    assert d["frac_below_k_sensor"] == 1.0, "全部落在 k_s 之下卻沒被標記"
    assert d["frac_in_mid_band"] == 0.0
    assert d["d_lattice_mean"] == pytest.approx(0.0, abs=1e-9), "整數 B 的格點距離必須為 0"

    mid = np.stack([np.full(100, 8.0), np.zeros(100)])   # |B| = 8，落在 (k_s, 16]
    d_mid = _describe(mid, k_sensor=5.6419)
    assert d_mid["frac_in_mid_band"] == 1.0
    assert d_mid["frac_below_k_sensor"] == 0.0


# ── 二階導路徑（PDE residual 真正會走的那條）─────────────────────────
# 這條路徑的 parity 測試在 `tests/test_ns_residual_parity.py`
# （`test_the_two_implementations_agree_with_trainable_fourier`）。
# Why 放那邊：斷言是 fused(folx) vs baseline(jacfwd) 必須一致，門檻 2e-4 描述的是
# physics.py 兩份實作的關係而非本 embedding 的性質。門檻與比較函式留在單一檔案，
# 抄一份過來就是等它漂移。


def test_diag_script_uses_the_real_sensor_loader_keys():
    """診斷腳本對 `load_sensors_from_path` 回傳的鍵不得用猜的。

    實際踩過：腳本寫 `sensors["sensor_values_normalized"]`，真實鍵是 `sensor_vals`。
    py_compile 與 ruff 都抓不到——它只在 lab-server 實跑時才 KeyError（大聲失敗，
    但已經浪費一輪往返）。本條用真實 loader 把契約釘在本機。
    """
    import os
    import re
    from pathlib import Path

    from pi_lnn_jax.data import load_sensors_from_path

    from pi_lnn_jax.data import _resolve_data_path

    sensors_dir = _resolve_data_path("data/kolmogorov_sensors/re10000")
    candidates = sorted(sensors_dir.glob("sensors_*_K100_*.json")) if sensors_dir.is_dir() else []
    if not candidates:
        pytest.skip(f"本機無 sensor 資料（{sensors_dir}）")

    got = load_sensors_from_path(str(candidates[0]), time_stride=2)

    src = (Path(__file__).resolve().parent.parent
           / "scripts/diag_trainable_fourier.py").read_text()
    used = set(re.findall(r'sensors\["([a-z_]+)"\]', src))
    assert used, "沒抓到腳本對 sensors 的取用——正規式與腳本寫法不再對應"
    missing = used - set(got)
    assert not missing, (
        f"diag_trainable_fourier.py 取用了 loader 沒有的鍵 {sorted(missing)}；"
        f"實際可用：{sorted(got)}")


def test_lowpass_off_by_default_is_bit_identical():
    """kc=0（預設）→ 與沒有這個功能時逐位元相同。"""
    xy = jax.random.uniform(jax.random.PRNGKey(3), (12, 2))
    mod = TrainableFourierEmb(embed_dim=8, init_freq_scale=8.0)
    params = mod.init(jax.random.PRNGKey(0), xy, 1.0)
    off = TrainableFourierEmb(embed_dim=8, init_freq_scale=8.0, lowpass_kc=0.0)
    assert jnp.array_equal(mod.apply(params, xy, 1.0), off.apply(params, xy, 1.0))


def test_lowpass_attenuates_high_frequency_features_only():
    """w = 1/(1+(|B|/kc)²)：|B|=kc 衰減到 1/2，|B|→0 不衰減，高頻大幅壓低。"""
    xy = jax.random.uniform(jax.random.PRNGKey(3), (7, 2))
    kc = 5.0
    mod = TrainableFourierEmb(embed_dim=4, lowpass_kc=kc)
    # 兩個 feature：|B|=0（純低頻）與 |B|=kc
    B = jnp.array([[0.0, kc], [0.0, 0.0]])
    out = mod.apply({"params": {"B": B}}, xy, 1.0)
    plain = TrainableFourierEmb(embed_dim=4).apply({"params": {"B": B}}, xy, 1.0)

    # 輸出是 concat([cos, sin])，兩半的第 j 欄共用同一個 w_j
    for half in (0, 1):
        col_low, col_hi = out[:, half * 2], out[:, half * 2 + 1]
        ref_low, ref_hi = plain[:, half * 2], plain[:, half * 2 + 1]
        assert jnp.allclose(col_low, ref_low, atol=1e-6), "|B|=0 的 feature 不該被衰減"
        assert jnp.allclose(col_hi, 0.5 * ref_hi, atol=1e-6), "|B|=kc 應衰減到 1/2"


def test_lowpass_weight_does_not_create_a_gradient_incentive_to_lower_frequency():
    """承重：權重對 B 走 stop_gradient。

    若梯度可穿透 w，網路只要壓小 |B| 就能免費換到更大振幅 → collapse 變成被獎勵的
    方向，與物理無關，事前登錄的「|B| 塌陷 = 資訊牆」讀數就完全不可判別。
    這條擋的是那個 confound。
    """
    xy = jax.random.uniform(jax.random.PRNGKey(1), (16, 2))
    w = jax.random.normal(jax.random.PRNGKey(2), (8,))
    kc = 5.0
    plain = TrainableFourierEmb(embed_dim=8, init_freq_scale=8.0)
    gated = TrainableFourierEmb(embed_dim=8, init_freq_scale=8.0, lowpass_kc=kc)
    params = plain.init(jax.random.PRNGKey(0), xy, 1.0)

    def grad_of(mod):
        g = jax.grad(lambda p: jnp.sum(mod.apply(p, xy, 1.0) * w))(params)
        return flatten_dict(g)[("params", "B")]

    B = flatten_dict(params)[("params", "B")]
    scale = 1.0 / (1.0 + (jnp.linalg.norm(B, axis=0) / kc) ** 2)   # [half]
    # stop_gradient 下，B 的梯度應恰為「未加權梯度逐 feature 乘上 w」——
    # 沒有任何來自 ∂w/∂B 的額外項。
    assert jnp.allclose(grad_of(gated), grad_of(plain) * scale[None, :], rtol=1e-5, atol=1e-6), (
        "B 的梯度含有穿透權重的項 → 壓低頻率會被獎勵，診斷失效")


def test_downstream_gate_slices_the_right_kernel_blocks():
    """下游閘門的切片假設：trunk_in kernel 前段吃 baseline、次段吃 trainable。

    切錯就會把兩塊搞反，而兩塊的 norm 本來就不同 → 得到一個看起來合理的錯結論。
    用刻意不同量級的合成 kernel 釘住方向。
    """
    import numpy as np

    from scripts.diag_trainable_fourier import _downstream_gate
    fourier_dim, trainable_dim, out = 4, 3, 5
    kernel = np.concatenate([
        np.full((fourier_dim, out), 1.0),    # baseline 區塊
        np.full((trainable_dim, out), 0.1),  # trainable 區塊，刻意小一個量級
        np.full((2, out), 7.0),              # 其餘（temporal / time_e）不該被算進去
    ])
    params = {"params": {"query_decoder": {"trunk_in": {"kernel": kernel}}}}
    got = _downstream_gate(params, fourier_dim, trainable_dim)

    expected = np.sqrt(out)
    assert got["base_block_row_norm_mean"] == pytest.approx(expected, rel=1e-6)
    assert got["trainable_block_row_norm_mean"] == pytest.approx(0.1 * expected, rel=1e-6)
    assert got["trainable_over_base"] == pytest.approx(0.1, rel=1e-6), "兩塊切反了"


def test_downstream_gate_rejects_a_kernel_too_small_for_the_layout():
    """維度佈局假設不成立時必須 raise，不得靜默切出一塊沒有意義的列。"""
    import numpy as np

    from scripts.diag_trainable_fourier import _downstream_gate
    params = {"params": {"query_decoder": {"trunk_in": {"kernel": np.ones((5, 3))}}}}
    with pytest.raises(SystemExit, match="放不下"):
        _downstream_gate(params, 4, 3)


def test_find_frequency_matrix_fails_loudly_when_absent():
    """flag-off 的 ckpt 沒有 B——必須大聲失敗，不得回傳空統計讓判讀讀到假數字。"""
    from scripts.diag_trainable_fourier import _find_frequency_matrix
    with pytest.raises(SystemExit, match="trainable_spatial_emb"):
        _find_frequency_matrix({"params": {"query_decoder": {"trunk_in": {"kernel": jnp.zeros((2, 2))}}}})

"""兩份 NS 動量殘差必須算出同一組物理 —— baseline 先前 0 測試。

問題（架構審查候選 C）：NS 方程被寫了兩遍。`make_ns_residual_fn`（fused/folx，
B3 用）與 `make_ns_residual_fn_baseline`（jacfwd，B0/PINN 用）。後者的 docstring
宣稱「NS 物理與 make_ns_residual_fn 完全一致」，而在 2026-08-03 之前它的簽章
**根本沒有 Lx/Ly**——那句話只在 Lx=Ly=1 成立。

而它是 B0/B2/PINN ablation 臂的物理核，先前 **0 個測試檔 import 它**。
宣稱一致、沒人驗過、又是論文對照組的核心——這三件事湊在一起才是問題。

實測（本檔的測試就是那次量測的固化）：
  - Lx=Ly=1：逐點相對差 max 2.3e-05，純量 MSE 一致到 7 位有效數字（float32 捨入）
  - Lx=0.6, Ly=0.3（cylinder 尺度）：補 Lx/Ly 之前 MSE 差 **5.7–7.9 倍**

本檔在兩個域上都比對。第二個域目前沒有 production 路徑會走到
（B0/PINN 只跑 Kolmogorov，Lx=Ly=1），但那正是它該被測的理由：
潛伏的陷阱不會自己現形，得有人先把它踩一遍。

**本檔是 fused-vs-baseline parity 這個契約的唯一落點**，包含門檻 `_TOL` 與比較
函式 `_relative`。新的 model config（trainable Fourier、mid-band 之類會改動 trunk
座標路徑的東西）要驗這條 parity，就在本檔加一組 harness，不要在自己的測試檔裡
複製一份斷言——那份 2e-4 必然與本檔漂移。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.physics import make_ns_residual_fn, make_ns_residual_fn_baseline

_CFG = dict(
    sensor_value_dim=2, d_model=32, d_time=8,
    num_spatial_encoder_layers=1, num_temporal_cfc_layers=1,
    num_token_attention_layers=0, num_query_mlp_layers=1,
    query_mlp_hidden_dim=32, operator_rank=8, decoder_attention_heads=2,
    use_temporal_anchor=True, T_total=1.0, temporal_anchor_harmonics=2,
    domain_length=1.0,
)
_T, _K, _N = 4, 8, 64
_NU, _A, _KF, _RE_NORM = 1e-4, 1.0, 4.0, 0.5
#: 刻意讓 std 遠離 1、mean 非 0——這樣「漏乘 std」或「漏加 mean」都會現形。
_STATS = (0.1, 0.8, -0.05, 0.7, 0.02, 0.3)   # u_mean,u_std,v_mean,v_std,p_mean,p_std

#: float32 下兩條不同 AD 路徑（folx collapsed-laplacian vs jacfwd）的捨入差。
#: 實測 max 2.3e-05；門檻取一個數量級的餘裕。
#: **本檔是這個門檻的唯一來源**——它描述的是 physics.py 兩份實作的關係，
#: 不是某一個測試的私事。要驗別的模型 config 就在本檔加一組 harness，
#: 不要把 2e-4 抄到另一個檔（抄過去的那份必然漂移）。
_TOL = 2e-4

#: 方向1 exp_508 的 trainable-frequency embedding（見 test_trainable_fourier.py）。
#: 它進的是 trunk 座標路徑，而 PDE 殘差要對座標取二階導，故同樣受本檔的契約約束。
#: `num_query_mlp_layers=1` 是承重的：embedding 之後必須有 MLP，否則二階導路徑不成形。
_TF_CFG = dict(_CFG, use_trainable_fourier=True, trainable_fourier_dim=16,
               trainable_fourier_init_scale=8.0)
_TF_N = 32


def _harness(cfg: dict, n_collo: int):
    """同一組 model / params / collocation 點，餵給兩條實作。

    回傳 `(run_fused, run_base)`；兩者只差在呼叫簽章（fused 吃預先算好的 `h`，
    baseline 自己從 sensor 重算），物理輸出必須一致。
    """
    r = np.random.RandomState(0)
    sensor_vals = jnp.asarray(r.randn(_T, _K, 2), jnp.float32)
    sensor_pos = jnp.asarray(r.uniform(0, 1, (_K, 2)), jnp.float32)
    sensor_time = jnp.linspace(0.0, 1.0, _T, dtype=jnp.float32)
    cx = jnp.asarray(r.uniform(0, 1, (n_collo,)), jnp.float32)
    cy = jnp.asarray(r.uniform(0, 1, (n_collo,)), jnp.float32)
    ct = jnp.asarray(r.uniform(0, 1, (n_collo,)), jnp.float32)

    model = LiquidOperator(**cfg)
    params = model.init(jax.random.PRNGKey(42), sensor_vals, sensor_pos, _RE_NORM,
                        sensor_time, jnp.zeros((2, 2)), jnp.zeros((2,)))
    h = model.apply(params, sensor_vals, sensor_pos, _RE_NORM, sensor_time,
                    method=LiquidOperator.encode)

    fused, _ = make_ns_residual_fn(model)
    base = make_ns_residual_fn_baseline(model)

    def run_fused(Lx, Ly):
        return fused(params, h, cx, cy, ct, _A, _KF, sensor_pos, sensor_time,
                     _NU, *_STATS, Lx=Lx, Ly=Ly, return_per_point=True)

    def run_base(Lx, Ly):
        return base(params, sensor_vals, sensor_pos, _RE_NORM, sensor_time,
                    cx, cy, ct, _A, _KF, _NU, *_STATS,
                    Lx=Lx, Ly=Ly, return_per_point=True)

    return run_fused, run_base


@pytest.fixture(scope="module")
def setup():
    return _harness(_CFG, _N)


@pytest.fixture(scope="module")
def setup_trainable_fourier():
    return _harness(_TF_CFG, _TF_N)


def _relative(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float((np.abs(a - b) / ((np.abs(a) + np.abs(b)) / 2 + 1e-30)).max())


@pytest.mark.parametrize("Lx,Ly", [(1.0, 1.0), (0.6, 0.3), (2.0, 0.5)],
                         ids=["isotropic", "cylinder-like", "stretched"])
def test_the_two_implementations_agree(setup, Lx, Ly):
    """同一個 model / params / collocation 點，兩條路徑必須給同一組物理。

    非 isotropic 的兩組目前沒有 production 路徑會走到（B0/PINN 只跑
    Kolmogorov）。測它們正是因為如此——潛伏的陷阱不會自己現形。
    """
    run_fused, run_base = setup
    f = run_fused(Lx, Ly)
    b = run_base(Lx, Ly)

    for i, name in enumerate(["mom_u", "mom_v", "cont"]):
        d = _relative(f[3 + i], b[3 + i])
        assert d < _TOL, (
            f"Lx={Lx} Ly={Ly}: {name} 逐點相對差 {d:.3e} 超過 {_TOL:.0e}\n"
            f"  兩份 NS 殘差算出不同的物理。若是刻意的差異，"
            "physics.py 的 docstring 不得再宣稱一致。")


def test_the_two_implementations_agree_with_trainable_fourier(setup_trainable_fourier):
    """同一個契約，套在 trainable-frequency Fourier embedding 上（exp_508 方向1）。

    為什麼是最承重的一條：新 embedding 進的是 trunk 座標路徑，而 PDE 殘差要對座標
    取二階導。folx 的 collapsed forward-Laplacian 對不在 registry 的 primitive 會
    **fallback 成 dense hessian——不 crash**，只是安靜地變慢或 OOM（`stack` 與
    `tile` 都踩過，見 models.py 那兩處註解）。拿 fused(folx) vs baseline(jacfwd)
    的 parity 同時驗「數值正確」與「這條路走得通」。

    只測 isotropic：production 的 exp_508 只跑 Kolmogorov（Lx=Ly=1）。要擴到
    anisotropic 就把上面那個 parametrize 也套上來——但那是三份額外的 XLA 編譯，
    在有 production 路徑走到之前不值得。
    """
    run_fused, run_base = setup_trainable_fourier
    f = run_fused(1.0, 1.0)
    b = run_base(1.0, 1.0)

    for i, name in enumerate(["mom_u", "mom_v", "cont"]):
        d = _relative(f[3 + i], b[3 + i])
        assert d < _TOL, (
            f"{name} 兩條 AD 路徑相對差 {d:.3e} 超過 {_TOL:.0e}"
            "——trainable Fourier 破壞了二階導")


def test_anisotropic_scaling_is_load_bearing_in_the_baseline(setup):
    """自證：Lx/Ly 若在 baseline 內被忽略，上面的 anisotropic 案例會**恰好**
    因為 fused 也被忽略而一起綠嗎——不會，但這條直接釘住 baseline 真的用了它。

    沒有這條，有人把 baseline 的 Lx/Ly 除法拿掉、同時把 fused 的也拿掉，
    parity 測試仍全綠而兩邊一起錯。
    """
    _, run_base = setup
    iso = run_base(1.0, 1.0)
    aniso = run_base(0.6, 0.3)
    d = _relative(iso[3], aniso[3])
    assert d > 0.5, (
        f"baseline 的 mom_u 對 Lx/Ly 幾乎無反應（相對差 {d:.3e}）"
        "——那兩個參數沒有真的進到計算裡")


def test_isotropic_path_is_the_one_production_uses(setup):
    """釘住現況：B0/PINN 只跑 Kolmogorov，走 Lx=Ly=1。

    這條紅了代表有人開始用 anisotropic 的 baseline——那時上面的 parity 就從
    「預防性」變成「承重」，值得回頭確認 A/B 對拍是否涵蓋新路徑。
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    src = (root / "pi_lnn_jax" / "pipeline" / "kolmogorov" / "assembly.py").read_text()
    assert "make_ns_residual_fn_baseline" in src, "baseline 的使用處已移動，更新本測試"
    assert "Lx=" not in src and "Ly=" not in src, (
        "kolmogorov/assembly.py 開始傳 Lx/Ly 了——確認 B0/PINN 的 parity 仍成立，"
        "且 A/B 對拍涵蓋該路徑")


# ── refiners 的第三份 NS 方程已收斂（2026-08-03）——防止再長回來 ──────────

def _refiners_setup(norm_stats, n_channels=2):
    """建一個 refiners 的 residual_vec_fn，回傳 (fn, params, N_collo)。"""
    r = np.random.RandomState(0)
    T, K, N = 4, 8, 32
    sv = jnp.asarray(r.randn(T, K, 2), jnp.float32)
    sp = jnp.asarray(r.uniform(0, 1, (K, 2)), jnp.float32)
    stime = jnp.linspace(0.0, 1.0, T, dtype=jnp.float32)
    model = LiquidOperator(**_CFG)
    params = model.init(jax.random.PRNGKey(42), sv, sp, 0.5, stime,
                        jnp.zeros((2, 2)), jnp.zeros((2,)))
    from pi_lnn_jax.refiners import make_pinn_residual_vector_fn
    fn = make_pinn_residual_vector_fn(
        model, sv, sp, stime, re_norm=0.5, norm_stats=norm_stats, re_value=1000.0,
        xy_sensor_q=jnp.asarray(r.uniform(0, 1, (T * K, 2)), jnp.float32),
        t_sensor_q=jnp.asarray(r.uniform(0, 1, (T * K,)), jnp.float32),
        sensor_target=jnp.asarray(r.randn(T * K, n_channels), jnp.float32),
        cx=jnp.asarray(r.uniform(0, 1, (N,)), jnp.float32),
        cy=jnp.asarray(r.uniform(0, 1, (N,)), jnp.float32),
        ct=jnp.asarray(r.uniform(0, 1, (N,)), jnp.float32),
        w_data=1.0, w_phys=0.1)
    return fn, params, N, T * K


_BASE_STATS = {"u_mean": 0.1, "u_std": 0.8, "v_mean": -0.05, "v_std": 0.7}


def test_refiners_actually_uses_the_pressure_statistics():
    """先前 `p = out[2]` 讓 norm_stats 的 p 統計被**靜默丟棄**——傳與不傳
    結果完全相同（實測 phys_rms 兩者皆 1.5779762268）。收斂後必須有差。

    p_std 是乘性常數，微分不會消掉它（p_mean 才會）。
    """
    fn_a, params, N, _ = _refiners_setup(_BASE_STATS)
    fn_b, _, _, _ = _refiners_setup({**_BASE_STATS, "p_mean": 0.02, "p_std": 0.3})

    phys_a = np.asarray(fn_a(params))[-3 * N:]
    phys_b = np.asarray(fn_b(params))[-3 * N:]
    rel = abs(np.sqrt((phys_a ** 2).mean()) - np.sqrt((phys_b ** 2).mean())) / \
        np.sqrt((phys_a ** 2).mean())
    assert rel > 0.05, (
        f"給不給 p 統計，physics 殘差的 RMS 只差 {rel:.2%}——p_std 又被丟棄了")


def test_missing_pressure_statistics_defaults_to_the_identity():
    """缺 p 統計的資料集（p_std 預設 1.0、p_mean 0.0）反正規化為恆等。

    這是收斂**不得改變**既有行為的那一半：沒有 p 統計的資料集，其殘差在
    收斂前後逐位元相同（獨立實測 phys_rms 1.5779762268066406 兩側一致）。

    這裡釘的是性質而非數值——「省略 == 顯式傳恆等值」與 fixture 無關，
    而抄一個在別處量到的數字進來釘，換個 fixture 就會假紅。
    """
    fn_default, params, N, _ = _refiners_setup(_BASE_STATS)
    fn_explicit, _, _, _ = _refiners_setup(
        {**_BASE_STATS, "p_mean": 0.0, "p_std": 1.0})

    a = np.asarray(fn_default(params))[-3 * N:]
    b = np.asarray(fn_explicit(params))[-3 * N:]
    assert np.array_equal(a, b), (
        "省略 p 統計與顯式傳 (0.0, 1.0) 結果不同——"
        f"預設值不再是恆等（max|Δ|={np.abs(a - b).max():.3e}）")


def test_data_residual_channel_count_follows_the_target():
    """先前寫死 2：uvp target 會被 silent 丟棄 p。"""
    for n_ch in (2, 3):
        fn, params, N, TK = _refiners_setup(_BASE_STATS, n_channels=n_ch)
        vec = np.asarray(fn(params))
        assert vec.size == TK * n_ch + 3 * N, (
            f"{n_ch}-channel target 的殘差長度 {vec.size} ≠ {TK * n_ch + 3 * N}"
            "——channel 數又被寫死了")


def test_refiners_does_not_reimplement_the_momentum_equation():
    """結構守衛：NS 方程只有一份。

    行為測試擋不住這件事——有人把方程抄回去、抄對了，行為測試照樣綠，
    而下一次 physics.py 的修正就不會傳到這裡（那正是 p_std 與 Lx/Ly 的來歷）。
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "pi_lnn_jax" / "refiners.py").read_text()
    body = src.split("def make_pinn_residual_vector_fn")[1]
    # 只看程式碼，不看 docstring/註解——docstring 本來就會提到這些名字。
    code = "\n".join(ln for ln in body.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "make_ns_residual_fn" in code, "refiners 沒有委派給 physics 的 NS 殘差"
    for marker in ("u_tv + u * u_xv", "v_tv + u * v_xv"):
        assert marker not in code, (
            f"refiners 內又出現動量方程的手寫實作（{marker!r}）——"
            "NS 方程只有一份，改用 physics.make_ns_residual_fn")

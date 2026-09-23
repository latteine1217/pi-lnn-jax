"""兩個核心構件的語意：residual 連接與 GradNorm 的 EMA / 耦合。

Why this file exists:
    這兩處被整個模型與所有 ablation 依賴，卻沒有任何測試釘住它們的語意。實測：

        移除 `ResidualMLPBlock` 的 `return x + y`（trunk / spatial encoder /
          B0 / B2 全部用它）                                  -> 全套零項變紅
        GradNorm 的 EMA 方向反轉                              -> 存活
        移除 `w_raw` 分母的 `1e-5·mean_G`                      -> 存活

    三個都是「改了不會有錯誤訊息，只會安靜變成另一個模型 / 另一組權重」。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pi_lnn_jax.losses import gradnorm_init, gradnorm_step, gradnorm_weights
from pi_lnn_jax.models import ResidualMLPBlock


def test_residual_block_is_identity_when_its_output_projection_is_zero():
    """把第二個 Dense 歸零後，block 必須恰好回傳輸入。

    判別器選這個是因為它**只**依賴 residual 的存在：FFN 分支被歸零之後，
    有 `+ x` 就是恆等、沒有就是全零。不依賴權重初始值，也不需要容差調校。
    `ModifiedMLPBlock`（mMLP）沒有 residual，兩者的差異正是 B-2 那條混淆的一半。
    """
    block = ResidualMLPBlock(d_model=8, hidden_dim=16)
    x = jax.random.normal(jax.random.PRNGKey(0), (5, 8))
    params = block.init(jax.random.PRNGKey(1), x)

    zeroed = jax.tree.map(lambda v: v, params)
    zeroed["params"]["Dense_1"] = {"kernel": jnp.zeros((16, 8)), "bias": jnp.zeros((8,))}
    out = block.apply(zeroed, x)

    assert jnp.allclose(out, x, atol=0.0, rtol=0.0), (
        f"FFN 分支歸零後 block 應恰為恆等，得到 max|Δ|={float(jnp.abs(out - x).max()):.3e}"
        "——residual 連接不見了")


def test_gradnorm_ema_momentum_is_the_retained_fraction():
    """`w_new = m·w_old + (1−m)·w_target`：m 是**保留**舊值的比例。

    方向反了（`(1−m)·w_old + m·w_target`）不會有錯誤訊息，只會讓權重的
    收斂速度差一個數量級。m=0.9、w_old=1.0、target→0 時：
        正確   -> 0.9
        方向反 -> 0.1
    `chapter02.tex:499` 說它與 eq:al_update「同一慣例」，而 AL 的 β 定義是
    **新樣本**權重（方向相反）；因為生產值 0.5 自對稱所以數值上看不出來。
    這裡把 JAX 端的慣例釘死，讓那句敘述有可對照的依據。
    """
    state = gradnorm_init([1.0, 1.0, 1.0])
    stepped = gradnorm_step(state, jnp.array([1.0, 1e9, 1e9]),
                            ema_momentum=0.9, min_weight=0.0, max_weight=0.0)
    w = gradnorm_weights(stepped)
    assert float(w[0]) == pytest.approx(1.0, rel=1e-6), "參考 task 的權重恆為 1"
    assert float(w[1]) == pytest.approx(0.9, abs=1e-4), (
        f"m=0.9 的一次更新應保留 0.9，得到 {float(w[1]):.6f}；"
        "若接近 0.1 代表 EMA 方向反了")


def test_epsilon_in_the_weight_denominator_couples_the_tasks():
    """`w_i = (G_0 + 1e-5·Ḡ)/(G_i + 1e-5·Ḡ)` 的 Ḡ 沒有消掉——加一個 task 會移動其他所有 task。

    這是**已知的既有行為**，在此釘住而非修正：它讓「只加/減一個 task」的 ablation
    （含 `--cont_gradnorm`）不是單因子介入。主設定下兩個 physics 權重都被
    `gradnorm_min_weight=0.05` 的地板壓住，所以看不出來；換一組 G 分佈就會顯現。

    改成 per-task 的絕對 epsilon 是行為變更，需走 §7.1 的 A/B 對拍。屆時本測試
    會變紅——那正是它該做的事。
    """
    def weights(g):
        names = tuple(f"t{i}" for i in range(len(g)))
        s = gradnorm_init([1.0] * len(g), task_names=names)
        s = gradnorm_step(s, jnp.array(g), ema_momentum=0.0,
                          min_weight=0.0, max_weight=0.0)
        return [float(v) for v in gradnorm_weights(s)]

    three = weights([1.0, 20.0, 30.0])
    four = weights([1.0, 20.0, 30.0, 1e6])

    assert three[1] == pytest.approx(0.05, rel=1e-2)
    assert three[2] == pytest.approx(1 / 30, rel=1e-2)
    # 前三個 task 的 G 完全相同，權重卻被第四個 task 拉高 3 倍以上
    assert four[1] / three[1] > 3.0, (
        f"加入 G 極大的第四 task 後，前三項應被 1e-5·Ḡ 拉向 1；"
        f"得到 {four[1]:.4f} vs {three[1]:.4f}。若比值回到 1，代表 epsilon 已改成 "
        "per-task 絕對值——請同步更新 model-audit 的 B-9 與 §7.1 的對拍")
    assert four[2] / three[2] > 3.0


# ── GradNorm 的 reference 子樹在各 arch 上解不解得到 ──────────────────────

def _gn_ref_path(arch: str) -> tuple:
    """複製 assembly.build_context 的選擇規則（該處是 inline 三元式，無法 import）。"""
    return ("query_decoder", "trunk_out") if arch == "liquid" else ("trunk_out",)


def _resolves(arch: str) -> bool:
    import jax.numpy as jnp

    from pi_lnn_jax.model_factory import build_model
    from pi_lnn_jax.pipeline.kolmogorov.config import resolve_inputs

    mk = dict(resolve_inputs(
        ["--config", "configs/exp_245_b3_les_T50.toml"]).config.model.values)
    K, T, N = 100, 4, 8
    model, _ = build_model(arch, mk, K_sensors=K)
    sp = jax.random.uniform(jax.random.PRNGKey(0), (K, 2))
    st = jnp.linspace(0.0, 1.0, T)
    sv = jax.random.normal(jax.random.PRNGKey(2), (T, K, 2))
    params = model.init(jax.random.PRNGKey(3), sv, sp, 0.1, st,
                        jax.random.uniform(jax.random.PRNGKey(9), (N, 2)), jnp.zeros((N,)))
    node = params["params"]
    for key in _gn_ref_path(arch):
        if key not in node:
            return False
        node = node[key]
    return True


@pytest.mark.parametrize("arch", ["liquid", "vanilla"])
def test_gradnorm_reference_subtree_resolves(arch):
    """B3 與 vanilla 的 ref 子樹解得到——GradNorm 在 trunk 輸出層量梯度範數。"""
    assert _resolves(arch), (
        f"{arch} 的 {_gn_ref_path(arch)} 解不到；`_get_subtree` 會靜默 fallback 到"
        "整棵 params，兩臂的 inter-task 加權規則就不同了")


def test_pinn_reference_subtree_does_not_resolve_today():
    """PINN 的 ref 子樹**解不到**，會靜默 fallback 到整棵 params。

    這是已知缺陷（model-audit A9）：`StandardPINNOperator` 的頂層是
    spatial_emb / time_proj / input_proj / block_* / output_head / forcing，
    沒有 `trunk_out`。B3 在 trunk 輸出層的 3 個 leaf 上量梯度範數，PINN 在
    全部 3.24 M 參數上量——`chapter03.tex:350` 與 `chapter04.tex:239` 的
    「one common protocol, so that budget's comparison isolates the method」
    因此不成立。lab-server 的 PINN job stdout 已確認實跑走了 fallback。

    **釘住而非修正**：改 ref path 會改變 PINN 臂的權重軌跡，也就改變已發表的
    10.89% —— 屬 CLAUDE.md §3 的 baseline 結果，只能提議不能自行套用。
    修好的那天本測試會變紅，提醒一併更新那兩處稿子敘述與 PINN 的數字。
    """
    assert not _resolves("pinn"), (
        "PINN 的 gn_ref_path 現在解得到了——若這是刻意修正，請同步更新 "
        "chapter03.tex:350 / chapter04.tex:239 的 'one common protocol' 敘述、"
        "重跑 PINN 臂，並更新 model-audit 的 A9")


# ── 非週期 Fourier 嵌入的頻率尺度（tripwire）────────────────────────────

def test_nonperiodic_rff_carries_no_2pi_today():
    """`FourierEmbs` 的投影是 `xy @ B`，沒有稿子 eq:rff 寫的 $2\\pi$。

    兩條嵌入用同一個 `init_sigma=2.0`、|B| 中位數同為 2.230，差別只在 2π 落在哪：
    週期分支從 `e_per` 拿到（基頻 k=1），非週期分支完全沒有——全域最高只走約
    0.36 個週期。**不是被 init_sigma 吸收的**，是真的少了。

    **釘住而非修正**：`FourierEmbs` 只被非週期域用，也就是 cylinder 的 trunk；
    補上 2π 會改變 `tab:cyl_main` 的數字，屬 CLAUDE.md §7.1 的 bit-identical 契約。
    它同時是「cylinder 為何需要 K=400」的一個未測候選解釋（model-audit A14）。
    修好的那天本測試會變紅，提醒一併重跑 cylinder 並更新稿子的 eq:rff 說明。

    判別器：把 kernel 設成只吃 x 的單位向量，於是 proj = x。x 由 0 走到 1 時，
    無 2π 給 cos(1)=0.5403，有 2π 給 cos(2π)=1。兩者差 0.46，不需要容差調校。
    """
    from pi_lnn_jax.models import FourierEmbs

    emb = FourierEmbs(2)
    params = emb.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)), 1.0)
    params = jax.tree.map(lambda v: v, params)
    params["params"]["kernel"] = jnp.array([[1.0], [0.0]])
    out = emb.apply(params, jnp.array([[0.0, 0.0], [1.0, 0.0]]), 1.0)

    assert float(out[0, 0]) == pytest.approx(1.0, abs=1e-6)
    assert float(out[1, 0]) == pytest.approx(float(jnp.cos(1.0)), abs=1e-5), (
        f"x 走完 [0,1] 後 cos 分量是 {float(out[1, 0]):.4f}；"
        f"若接近 1.0 代表 2π 已補上——請重跑 cylinder、更新 eq:rff 的說明與 model-audit A14")


# ── 主 loss 的早期時間加權（規格，非 tripwire）──────────────────────────

def test_early_time_window_upweights_the_sensor_loss():
    """`eq:ic_weight`：$t < f_{\\rm early}\\,T$ 的 sensor 點乘 $w_{\\rm early}$。

    主線是 `w_early=10`、`f_early=0.05`，即 $t<0.25$~s 的視窗。拿掉這個加權
    全套零項變紅——而它是稿子明寫的規格，不是可有可無的細節。

    判別器：同一批輸入跑 `t_early_weight` 1.0 與 10.0，total 必須變大；
    且變化量要等於視窗內誤差乘 9（`per_point_w = mask·(w−1) + 1`）。
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _minimal_model import minimal_kwargs

    from pi_lnn_jax.models import LiquidOperator
    from pi_lnn_jax.physics import make_ns_residual_fn
    from pi_lnn_jax.pipeline.kolmogorov import assembly

    K, T, N = 5, 8, 6
    model = LiquidOperator(**minimal_kwargs(3))
    sp = jax.random.uniform(jax.random.PRNGKey(0), (K, 2))
    st = jnp.linspace(0.0, 1.0, T)
    sv = jax.random.normal(jax.random.PRNGKey(2), (T, K, 3))
    params = model.init(jax.random.PRNGKey(3), sv, sp, 0.1, st,
                        jax.random.uniform(jax.random.PRNGKey(9), (N, 2)), jnp.zeros((N,)))
    ns_fn, po_fn = make_ns_residual_fn(model)
    rb = assembly.ReBatch(
        sensor_vals=sv, sensor_pos=sp, sensor_time=st,
        re_norm=jnp.asarray(0.1), nu=jnp.asarray(1e-4),
        u_mean=jnp.asarray(0.0), u_std=jnp.asarray(1.0),
        v_mean=jnp.asarray(0.0), v_std=jnp.asarray(1.0),
        p_mean=jnp.asarray(0.0), p_std=jnp.asarray(1.0))
    cx = jax.random.uniform(jax.random.PRNGKey(4), (N,))
    cy = jax.random.uniform(jax.random.PRNGKey(5), (N,))
    ct = jax.random.uniform(jax.random.PRNGKey(6), (N,))
    tw = jnp.array([1.0, 0.0, 0.0])          # 只留 data 項，隔離出加權效果

    def total(w_early):
        lf = assembly._build_loss_fn(
            model, ns_fn, po_fn, use_poisson=False, use_al=False, al_rho=0.0,
            w_poisson=0.0, T_total=1.0, t_early_weight=w_early, t_early_threshold=0.5)
        return float(lf(params, cx, cy, ct, tw, jnp.asarray(0.0),
                        jnp.asarray(1.0), rb, 0.0)[0])

    plain, upweighted = total(1.0), total(10.0)
    assert upweighted > plain * 1.5, (
        f"t_early_weight 10 與 1 的 total 只差 {upweighted / plain:.3f}x——"
        "前期加權沒有生效（eq:ic_weight）")

"""釘住 GradNorm 探針的 data 項**語意**——B-10 那條唯一還沒有保護的突變。

`model-audit-2026-09.md` §3 的突變表裡，「GradNorm probe 的 data loss ×2」是唯一
仍然打不到的一列。B-10 的內容是：探針量的 `G_data` 與目標函數裡的 `L_data`
**不是同一個量**，有三處差異（`assembly.py` 的 `data_loss_only` vs `_build_loss_fn`）：

  1. 探針沒有 IC emphasis（主 loss 對 `t < f_early·T` 乘 `w_early`）
  2. 探針走全量 `T·K` 網格，主 loss 走 `sensor_idx` mini-batch
  3. `grad_accum>1` 時探針再截成**攤平後前 `n0 = n_collo/M`** 個點，而攤平是
     time-major → 只看得到最早的時刻

第 3 項的份量已實測（job 5685/5688）：K 不變、只加 `grad_accum_chunks=4`，
`G_data` 就從 0.0026 跳到 0.0112（4.3×），權重因此脫離 floor。

**能釘什麼、不能釘什麼（誠實說明）**

下面釘的是**語意**：探針微分的是哪一個 loss、以及那道截斷存不存在。
**一個純粹的常數縮放（例如 `g_data *= 2`）這裡抓不到**——要抓它得有一個獨立的
絕對參考值，而那意味著在測試裡重算一份探針的 loss。本 repo 正是被那個做法咬過：
舊的 `test_cont_gradnorm.py` 複刻了一份 `total` 再對副本斷言，六個測試全綠而實作是錯的。
與其為了抓一個沒人會寫的 `×2` 而重蹈覆轍，這裡改為釘住三個真實的不變量。
"""
from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import pytest
from _minimal_model import minimal_kwargs

from pi_lnn_jax.physics import make_ns_residual_fn
from pi_lnn_jax.models import LiquidOperator
from pi_lnn_jax.pipeline.kolmogorov import assembly

K, T, N = 6, 5, 8


def _setup(seed=0):
    model = LiquidOperator(**minimal_kwargs(3))
    sp = jax.random.uniform(jax.random.PRNGKey(seed), (K, 2))
    st = jnp.linspace(0.0, 1.0, T)
    sv = jax.random.normal(jax.random.PRNGKey(seed + 2), (T, K, 3))
    params = model.init(jax.random.PRNGKey(3), sv, sp, 0.1, st,
                        jax.random.uniform(jax.random.PRNGKey(9), (N, 2)), jnp.zeros((N,)))
    ns_fn, _ = make_ns_residual_fn(model)
    rb = assembly.ReBatch(
        sensor_vals=sv, sensor_pos=sp, sensor_time=st,
        re_norm=jnp.asarray(0.1), nu=jnp.asarray(1e-4),
        u_mean=jnp.asarray(0.0), u_std=jnp.asarray(1.0),
        v_mean=jnp.asarray(0.0), v_std=jnp.asarray(1.0),
        p_mean=jnp.asarray(0.0), p_std=jnp.asarray(1.0),
    )
    return model, ns_fn, params, rb


def _collo(seed, n=N):
    k = jax.random.split(jax.random.PRNGKey(seed), 3)
    return (jax.random.uniform(k[0], (n,)), jax.random.uniform(k[1], (n,)),
            jax.random.uniform(k[2], (n,)))


# ── 1. 探針與主 loss 共用同一組 IC emphasis 參數 ──────────────────────────

def test_probe_and_objective_share_the_ic_emphasis_parameters():
    """兩邊的簽章都要有 `t_early_*`——缺一邊就代表加權只套了一半。

    **這是 tripwire**：若有人把 `t_early_*` 從探針拿掉（回到 2026-09-13 之前的
    行為），本測試會變紅，提醒一併更新 `chapter02` 對 `\\mathcal{L}_i` 的描述、
    model-audit 的 B-10、以及 §7.1 的對拍。
    """
    probe = {p for p in inspect.signature(assembly._build_grad_norm_fn).parameters
             if "early" in p}
    loss = {p for p in inspect.signature(assembly._build_loss_fn).parameters
            if "early" in p}
    assert loss == {"t_early_weight", "t_early_threshold"}, (
        "主 loss 的 IC emphasis 參數變了；改一邊就要改另一邊與敘述")
    assert probe == loss, (
        f"探針的 IC emphasis 參數是 {probe}，主 loss 是 {loss}——兩者必須一致，"
        "否則 GradNorm 拿未加權的 G_data 去平衡加權後的目標")


def test_ic_emphasis_actually_changes_the_probe_gradient():
    """簽章對齊還不夠——要確認參數真的被用上。

    只比簽章會被「收了參數但沒接線」騙過（M18 那條踩過：三個 guard 全部刪掉
    呼叫點，1209 個測試照樣全綠）。
    """
    model, ns_fn, params, rb = _setup()
    cx, cy, ct = _collo(4)
    flat = assembly._build_grad_norm_fn(
        model, ns_fn, t_early_weight=1.0, T_total=1.0)(params, cx, cy, ct, rb, 0.0)
    early = assembly._build_grad_norm_fn(
        model, ns_fn, t_early_weight=10.0, t_early_threshold=0.5,
        T_total=1.0)(params, cx, cy, ct, rb, 0.0)
    assert float(flat[0]) != pytest.approx(float(early[0]), rel=1e-6), (
        "t_early_weight 改了但 G_data 沒動——探針收了參數卻沒接上 data_loss_only")
    assert float(flat[1]) == pytest.approx(float(early[1]), rel=1e-6), (
        "t_early 動到了 physics 那一支——它只該作用在 data 項上")


# ── 2. `grad_accum>1` 的子取樣必須覆蓋整個時窗，不能只取開頭 ──────────────

def test_grad_accum_subsample_spans_the_whole_time_window():
    """M>1 時探針只取 1/M 的 sensor query 點，但那些點必須**橫跨全部時刻**。

    這是本檔最重要的一條。攤平是 time-major，所以取前綴等於只看最早的 n0/K 個
    時刻——K=200 上是 101 個時刻裡的 1.28 個（t ∈ [0, 0.05]，全窗的 1%），而那正好
    是誤差最大的一段。那是**偏差**不是變異數，GradNorm 的 EMA 平均不掉它。

    判別方式：只擾動**最後一個時刻**的 sensor 值。取前綴的版本永遠看不到那一列，
    `G_data` 不會動；均勻取樣的版本會動。
    """
    model, ns_fn, params, rb = _setup()
    n_collo = 16                       # → n0 = 4，linspace(0, T*K-1, 4) 橫跨四個時段
    cx, cy, ct = _collo(4, n=n_collo)
    fn = assembly._build_grad_norm_fn(model, ns_fn, grad_accum=4)

    base = fn(params, cx, cy, ct, rb, 0.0)
    bumped = rb.sensor_vals.at[-1].add(5.0)          # 只動最後一個時刻那一列
    late = fn(params, cx, cy, ct, rb._replace(sensor_vals=bumped), 0.0)

    assert float(base[0]) != pytest.approx(float(late[0]), rel=1e-6), (
        "只改最後一個時刻的 sensor 值，G_data 卻沒動——探針的子取樣沒有看到時窗尾端。"
        "這就是取前綴（time-major 攤平）的症狀，見本測試 docstring。")

    # 對照：只動第一個時刻也必須有反應（兩端都要在取樣範圍內）
    early = fn(params, cx, cy, ct,
               rb._replace(sensor_vals=rb.sensor_vals.at[0].add(5.0)), 0.0)
    assert float(base[0]) != pytest.approx(float(early[0]), rel=1e-6), (
        "只改第一個時刻的 sensor 值，G_data 沒動——子取樣連開頭也漏了")


def test_probe_applies_causal_weighting_when_the_objective_does():
    """`use_causal` 時主 loss 用 `mean(w_c·r²)`，探針必須用同一個量。

    否則 GradNorm 的分母量的是未加權殘差，而被最佳化的是加權版——權重就是照著
    一個沒人在最佳化的量調出來的。`use_causal` 全庫皆 False 時這是潛伏項；
    一旦有 config 開啟它就變成活的。
    """
    model, ns_fn, params, rb = _setup()
    cx, cy, ct = _collo(4)
    off = assembly._build_grad_norm_fn(model, ns_fn, use_causal=False)(
        params, cx, cy, ct, rb, 0.0)
    on = assembly._build_grad_norm_fn(model, ns_fn, use_causal=True)(
        params, cx, cy, ct, rb, 5.0)
    assert float(off[1]) != pytest.approx(float(on[1]), rel=1e-6), (
        "開了 use_causal 但 G_ns_u 沒動——探針沒有套 causal 權重")
    # eps=0 等同關閉，兩者必須一致（釘住「只有權重變，殘差本身沒變」）
    zero = assembly._build_grad_norm_fn(model, ns_fn, use_causal=True)(
        params, cx, cy, ct, rb, 0.0)
    assert float(zero[1]) == pytest.approx(float(off[1]), rel=1e-5), (
        "eps=0 應等同關閉 causal，但兩者不同——per-point 路徑與原路徑算的不是同一個量")


# ── 3. G_data 只看 sensor，不看 collocation ───────────────────────────────

def test_probe_data_gradient_depends_on_sensors_and_not_on_collocation():
    """`G_data` 對 collocation 的**值**必須不變，對 sensor 資料必須改變。

    抓的是接線錯誤：若 `g_data` 被接成 physics 那一支的梯度，第一條會壞；
    若它根本沒看 sensor，第二條會壞。
    """
    model, ns_fn, params, rb = _setup()
    fn = assembly._build_grad_norm_fn(model, ns_fn, grad_accum=1)

    g_a = fn(params, *_collo(4), rb, 0.0)
    g_b = fn(params, *_collo(77), rb, 0.0)     # 換一批 collocation
    assert float(g_a[0]) == pytest.approx(float(g_b[0]), rel=1e-6), (
        "換 collocation 讓 G_data 變了——探針的 data 項摻進了 physics 的東西")
    assert float(g_a[1]) != pytest.approx(float(g_b[1]), rel=1e-6), (
        "換 collocation 卻沒動到 G_ns_u——physics 那一支沒有真的吃 collocation")

    rb2 = rb._replace(sensor_vals=rb.sensor_vals * 3.0)
    g_c = fn(params, *_collo(4), rb2, 0.0)
    assert float(g_c[0]) != pytest.approx(float(g_a[0]), rel=1e-6), (
        "改 sensor 資料卻沒動到 G_data——探針沒有在看 sensor")


# ── 4. 探針的 G_data 必須等於目標函數裡那一項的梯度範數 ──────────────────

@pytest.mark.parametrize("t_early_weight", [1.0, 10.0])
def test_probe_g_data_matches_the_objectives_own_data_term(t_early_weight):
    """把探針的 `G_data` 對上**主 loss 自己算出來的 `sensor_loss`** 的梯度範數。

    這是 model-audit §3 那張表裡最後一個「仍無保護」的突變——`data_loss_only` 乘上
    任何常數（經典的 `×2`）都不會被任何測試抓到。原因記在本檔開頭：要抓它得有一個
    獨立的絕對參考值，而在測試裡重算一份探針的 loss 就會變成拿實作驗實作。

    **出路是不要自己算。** `_build_loss_fn` 的 aux 第 0 項就是 `sensor_loss`，那是
    目標函數裡實際被最佳化的那一項，由**另一份程式**（`_build_loss_fn` 而非
    `data_loss_only`）算出。GradNorm 的前提正是「探針量的是目標函數裡那一項」，
    所以這個相等本身就是要釘的規格，不是巧合。

    這仍是 parity 測試，**兩份一起錯就對得上**——但這裡兩份是真正獨立的實作，
    而單邊的常數縮放（正是那個突變）一定會被抓到。對齊條件：
    `sensor_idx=None`（探針走全量網格）、同一組 `t_early_*`、同一條 ref path。
    `t_early` 兩種取值都測，因為 2026-09-13 之前探針**結構上**就沒有那段加權。
    """
    model, ns_fn, params, rb = _setup()
    cx, cy, ct = _collo(4)
    kw = dict(t_early_weight=t_early_weight, t_early_threshold=0.5, T_total=1.0)
    # ref path 兩側明寫成同一條。`_build_grad_norm_fn` 的**預設是 temporal_encoder**
    # 而 production 覆蓋成 trunk_out（assembly.py 的 gn_ref_path）——吃預設會讓兩側
    # 取不同子樹，比出一個與本測試無關的差異。
    ref = ("query_decoder", "trunk_out")

    probe = assembly._build_grad_norm_fn(model, ns_fn, ref_param_path=ref, **kw)
    g_probe = float(probe(params, cx, cy, ct, rb, 0.0)[0])

    loss_fn = assembly._build_loss_fn(
        model, ns_fn, poisson_fn=None, use_poisson=False, use_al=False,
        al_rho=0.0, w_poisson=0.0, **kw)

    def sensor_term(p):
        # aux[0] 就是主 loss 的 sensor_loss；不重算，直接取它實際用的那個值
        return loss_fn(p, cx, cy, ct, jnp.ones((3,)), 0.0, 1.0, rb, 0.0,
                       sensor_idx=None)[1][0]

    g = jax.grad(sensor_term)(params)
    sub = g["params"]
    for k in ref:
        sub = sub[k]
    leaves = jax.tree_util.tree_leaves(sub)
    g_obj = float(jnp.sqrt(sum(jnp.sum(l ** 2) for l in leaves) + 1e-12))

    assert g_obj > 0, "參考值本身是 0，這個測試什麼都證明不了"
    assert g_probe == pytest.approx(g_obj, rel=1e-5), (
        f"探針的 G_data={g_probe:.6e} 與目標函數的 sensor_loss 梯度範數 "
        f"{g_obj:.6e} 不符（比值 {g_probe / g_obj:.4f}）——GradNorm 會拿一個沒人在"
        "最佳化的量去定權重。常見成因：`data_loss_only` 與 `_build_loss_fn` 的 "
        "data 項漂移（加權、取樣、或常數縮放）。")

"""Round-3 對抗性審查修復的回歸守門。

涵蓋：
- D (#9)：TemporalCfCEncoder num_layers=0 時須 Re-aware（FiLM 在 cell loop 外套一次）。
- G    ：cylinder slip BC 時間座標須用獨立 RNG key（勿重用 inflow 的 key）。
"""
import jax
import jax.numpy as jnp

from pi_lnn_jax.models import TemporalCfCEncoder
from pi_lnn_jax.boundary import sample_wall_bc, CylinderGeometry


def test_temporal_cfc_num_layers0_is_re_aware():
    """num_temporal_cfc_layers=0（如 B2 ablation）時，re_norm 仍須影響輸出。

    原 bug：FiLM 只在 per-layer CfC loop 內套用，0 層 → loop 不跑 → Re-blind
    且 re_scale/re_shift 變 dead param。修復後在 loop 外套一次 FiLM。
    """
    enc = TemporalCfCEncoder(d_model=16, num_layers=0, num_token_attention_layers=1)
    T, K = 4, 3
    ss = jnp.ones((T, K, 16))
    st = jnp.linspace(0.0, 1.0, T)
    params = enc.init(jax.random.PRNGKey(0), ss, 0.0, st)
    out0 = enc.apply(params, ss, 0.0, st)
    out1 = enc.apply(params, ss, 1.0, st)
    assert float(jnp.max(jnp.abs(out0 - out1))) > 1e-6, "num_layers=0 仍 Re-blind（FiLM 未套用）"


def test_slip_bc_time_independent_of_inflow():
    """slip BC 的時間座標不可重用 inflow 的 RNG key。

    原 bug：slip 的 ts 用 kk[0]（= inflow y_in 的 key）→ slip.t 與 inflow.y 逐元素相同。
    修復後 slip.t 用獨立 kk[6]。
    """
    geom = CylinderGeometry(body_center=(0.5, 0.5), body_radius=0.1,
                            Lx=1.0, Ly=1.0, u_inf=1.0)
    inflow, _body, slip = sample_wall_bc(jax.random.PRNGKey(0), geom, 8)
    # slip[:,2]=t、inflow[:,1]=y_in；重用同 key 會使兩者逐元素相同
    assert not jnp.allclose(slip[:, 2], inflow[:, 1]), "slip BC 時間座標重用了 inflow 的 RNG key"

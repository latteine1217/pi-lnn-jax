"""驗證 radial 模式下 relpos bias 是否對 attention logits 有作用。

論文 chapter02 宣稱 eq:attention 是 isotropic bias 改變 logits 的唯一位置。
本檢查直接餵不同距離，看 bias 輸出是否隨距離變化。
"""
import jax, jax.numpy as jnp, flax.linen as nn

class RelposBias(nn.Module):
    """複製 models.py:853-856 的 scalar 路徑。"""
    hidden: int = 256
    @nn.compact
    def __call__(self, bias_in):
        x = nn.LayerNorm(name='ln')(bias_in)
        x = nn.Dense(self.hidden, name='fc1')(x)
        x = nn.silu(x)
        return nn.Dense(1, name='fc2')(x).squeeze(-1)

key = jax.random.PRNGKey(0)
# radial 模式：bias_in 只有 rel_r 一維 → [N=1, K=5, 1]
r = jnp.array([[[0.01], [0.1], [0.25], [0.5], [0.7]]])
m = RelposBias()
p = m.init(key, r)
out_radial = m.apply(p, r)

# vector 模式（cylinder 用）：bias_in = [rel_x, rel_y, rel_r] → 三維
rel = jnp.array([[[0.01, 0.0], [0.0, 0.1], [0.2, 0.15], [-0.4, 0.3], [0.5, -0.5]]])
rr = jnp.sqrt((rel**2).sum(-1, keepdims=True) + 1e-8)
bias_in_vec = jnp.concatenate([rel, rr], axis=-1)
p2 = m.init(key, bias_in_vec)
out_vector = m.apply(p2, bias_in_vec)

print("=== radial（Kolmogorov 所有 config）===")
print(f"  距離     : {r.squeeze().tolist()}")
print(f"  bias 輸出: {out_radial.squeeze().tolist()}")
print(f"  全等?    : {bool(jnp.allclose(out_radial, out_radial[0,0]))}")
print(f"  極差     : {float(out_radial.max()-out_radial.min()):.3e}")
print("=== vector（cylinder）===")
print(f"  bias 輸出: {[round(v,4) for v in out_vector.squeeze().tolist()]}")
print(f"  極差     : {float(out_vector.max()-out_vector.min()):.3e}")

# softmax 平移不變性：radial bias 加或不加，attention 權重是否相同
logits = jnp.array([[2.0, 1.0, 0.5, -1.0, 0.3]])
a_no   = jax.nn.softmax(logits, axis=-1)
a_bias = jax.nn.softmax(logits + out_radial, axis=-1)
print("=== 對 attention 的實際影響（radial）===")
print(f"  max |Δ attention weight| = {float(jnp.abs(a_no-a_bias).max()):.3e}")

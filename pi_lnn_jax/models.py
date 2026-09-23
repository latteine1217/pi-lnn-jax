"""Flax port of pi-lnn LiquidOperator (CfC + DeepONet + cross-attention).

對照 pi-lnn 原始檔：
  - src/pi_con/encodings.py  → encodings 區段
  - src/pi_con/blocks.py     → CfCStep / ResidualMLPBlock / TokenSelfAttentionBlock
  - src/pi_con/encoders.py   → SpatialSetEncoder / TemporalCfCEncoder
  - src/pi_con/decoder.py    → DeepONetCfCDecoder
  - src/pi_con/operator.py   → LiquidOperator

設計重點：
  1) CfCStep 用 `nn.scan` 包成單步單元，避免 Python time loop 的 trace 展開。
  2) 全程 functional：所有 params 透過 `apply({'params': ...}, ...)` 注入。
  3) 移除 hard body BC、modified MLP、bidirectional CfC、locality decay、
     fusion_temperature learnable 參數 — POC 只跑 Kolmogorov periodic case，
     EXP-030 沒用這些開關（exp_030 config 全 false）。
"""
from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn


# ─────────────────────────────────────────────────────────────────────────────
# Kernel init policy
# ─────────────────────────────────────────────────────────────────────────────

def torch_linear_kernel_init(key, shape, dtype=jnp.float32):
    """PyTorch nn.Linear 預設 kernel init：Kaiming-uniform U(±1/sqrt(fan_in))。

    Why: Flax nn.Dense 預設 lecun_normal 的 std 比 PyTorch nn.Linear 大 √3≈1.73×/層；
         數十層未指定 init 的 Dense 累積後使 LiquidOperator init 輸出 std 偏大 ~9×，
         在 cylinder 欠定問題造成均勻 over-energy。此 init 對齊 PyTorch reference。
    Flax kernel shape=(fan_in, fan_out)，故 fan_in=shape[0]。
    Note: bias 仍維持 Flax 預設 zeros。PyTorch nn.Linear 的 bias 是 U(±1/sqrt(fan_in))，
          但 bias 為 additive 不 compound，非 ~9× magnitude 主因，本修復刻意不改。
    """
    fan_in = shape[0]
    bound = 1.0 / jnp.sqrt(jnp.asarray(fan_in, dtype))
    return jax.random.uniform(key, shape, dtype, -bound, bound)


class RWFDense(nn.Module):
    """Random Weight Factorization Dense（Wang et al. 2023；對齊 jaxpi weight-factorization）。

    kernel 重參數化為 W_eff = rwf_g ⊙ V，rwf_g 為 per-output scalar：
      - rwf_g: init = exp(rwf_mean + N(0,1)·rwf_stddev)，shape (out,)
      - kernel(=V): init = base kernel_init / rwf_g，shape (in, out)
      → init 等效權重 rwf_g⊙V == base kernel_init（且對 rwf_mean/rwf_stddev 不變）。
    forward: y = x @ (rwf_g * kernel) + bias。訓練時 rwf_g 與 V 各自自由更新，
             乘法重參數化改變 loss landscape = RWF 的作用。

    use_rwf=False 時退化為標準 nn.Dense：param name path 'kernel'/'bias' 與 nn.Dense
    完全相同 → 同一把 rng 下數值一致，對既有 checkpoint / 結果零影響。
    """
    features: int
    use_rwf: bool = False
    rwf_mean: float = 1.0
    rwf_stddev: float = 0.1
    use_bias: bool = True
    kernel_init: Callable = nn.linear.default_kernel_init
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_features = x.shape[-1]
        if self.use_rwf:
            # rwf_g 先建（自有 key）；V 的 init 捕捉「已物化的 g」令 V = w / g
            # → effective kernel g⊙V = w（base init），對 mean/std 不變。
            g = self.param(
                'rwf_g',
                lambda key, shape: jnp.exp(
                    self.rwf_mean + jax.random.normal(key, shape) * self.rwf_stddev
                ),
                (self.features,),
            )

            def _v_init(key, shape, dtype=jnp.float32):
                return self.kernel_init(key, shape, dtype) / g

            kernel = g * self.param('kernel', _v_init, (in_features, self.features))
        else:
            kernel = self.param('kernel', self.kernel_init, (in_features, self.features))
        y = x @ kernel
        if self.use_bias:
            y = y + self.param('bias', self.bias_init, (self.features,))
        return y


# ─────────────────────────────────────────────────────────────────────────────
# Encodings (pure functions)
# ─────────────────────────────────────────────────────────────────────────────

def periodic_fourier_encode(z: jnp.ndarray, domain_length: float, n_harmonics: int) -> jnp.ndarray:
    """對 2D 座標 (x, y) 做確定性多諧波週期 Fourier 編碼。

    對齊 pi_con/encodings.py:periodic_fourier_encode。
    Returns: [N, 4 * n_harmonics]，排列 [sin_x_k1, cos_x_k1, sin_y_k1, cos_y_k1, ...]
    """
    x = z[:, 0:1]
    y = z[:, 1:2]
    ks = jnp.arange(1, n_harmonics + 1, dtype=z.dtype)
    cx = (2.0 * jnp.pi / domain_length) * ks * x
    cy = (2.0 * jnp.pi / domain_length) * ks * y
    # concatenate+expand_dims 取代 jnp.stack：stack 不在 folx registry（fallback full hessian），
    # concatenate_p/reshape 在 registry → 走稀疏 jacobian。語義等價。
    parts = [jnp.sin(cx), jnp.cos(cx), jnp.sin(cy), jnp.cos(cy)]
    return jnp.concatenate([p[:, :, None] for p in parts], axis=2).reshape(z.shape[0], -1)


def band_fourier_encode(z: jnp.ndarray, domain_length: float, wavenumbers) -> jnp.ndarray:
    """對 2D 座標 (x, y) 在**指定波數** wavenumbers 做確定性週期 Fourier 編碼。

    periodic_fourier_encode 的推廣：波數不必連續 1..n，可 target mid-band [5.64,16]（方向1）。
    Returns: [N, 4 * len(wavenumbers)]，排列 [sin_x_k, cos_x_k, sin_y_k, cos_y_k, ...]。
    folx-safe：與 periodic_fourier_encode 同走 concatenate+reshape（避 stack 的 dense-hessian fallback）。
    """
    x = z[:, 0:1]
    y = z[:, 1:2]
    ks = jnp.asarray(wavenumbers, dtype=z.dtype)
    cx = (2.0 * jnp.pi / domain_length) * ks * x
    cy = (2.0 * jnp.pi / domain_length) * ks * y
    parts = [jnp.sin(cx), jnp.cos(cx), jnp.sin(cy), jnp.cos(cy)]
    return jnp.concatenate([p[:, :, None] for p in parts], axis=2).reshape(z.shape[0], -1)


def _repeat3(x: jnp.ndarray) -> jnp.ndarray:
    """沿 axis 0 block 複製 3 份（c=0/1/2 對齊 trunk 的 [3N] block 排列）。

    取代 jnp.tile(x, (3,1,...))：tile_p 不在 folx forward-Laplacian registry，
    會 fallback dense hessian（記憶體 ∝ N×K，大 K 時 OOM）；broadcast_in_dim+reshape
    在 registry，folx 走稀疏 jacobian 路徑。語義等價（block 複製）。
    """
    return jnp.broadcast_to(x[None], (3,) + x.shape).reshape((3 * x.shape[0],) + x.shape[1:])


def temporal_phase_anchor(t: jnp.ndarray, T_total: float, n_harmonics: int = 2) -> jnp.ndarray:
    """絕對時間的確定性 temporal-phase-anchor。
    對齊 pi_con/encodings.py:temporal_phase_anchor。
    """
    ns = jnp.arange(1, n_harmonics + 1, dtype=t.dtype)
    angles = (2.0 * jnp.pi / T_total) * ns * t
    # concatenate 取代 jnp.stack（folx registry，避免 full hessian fallback）
    return jnp.concatenate(
        [jnp.sin(angles)[:, :, None], jnp.cos(angles)[:, :, None]], axis=2
    ).reshape(t.shape[0], -1)


class LearnableFourierEmb(nn.Module):
    """PeriodEmbs(k=1) + 可學習投影。對齊 pi_con/encodings.py:LearnableFourierEmb 單頻段路徑。
    POC 不實作多頻段（init_sigma_bands）— EXP-030 未啟用。
    """
    embed_dim: int
    init_sigma: float = 2.0

    @nn.compact
    def __call__(self, xy: jnp.ndarray, domain_length: float) -> jnp.ndarray:
        if self.embed_dim % 2 != 0:
            raise ValueError(f"embed_dim 必須為偶數，收到 {self.embed_dim}")
        half = self.embed_dim // 2
        c = 2.0 * jnp.pi / domain_length
        x, y = xy[:, 0:1], xy[:, 1:2]
        period_enc = jnp.concatenate(
            [jnp.sin(c * x), jnp.cos(c * x), jnp.sin(c * y), jnp.cos(c * y)],
            axis=-1,
        )  # [N, 4]
        # PyTorch nn.Linear(4, half, bias=False)；對齊 init: normal(std=init_sigma)
        kernel = self.param(
            'kernel',
            lambda key, shape: jax.random.normal(key, shape) * self.init_sigma,
            (4, half),
        )
        proj = period_enc @ kernel
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)


class BandFourierEmb(nn.Module):
    """band-targeted 版 LearnableFourierEmb（方向1）：在**指定 wavenumbers** 做週期編碼 + 可學投影。

    給 trunk mid-band [5.64,16] 的 representation 容量。結構與 LearnableFourierEmb 一致
    （同 param name 'kernel'、同 init、同 cos/sin 輸出），只把單頻段 (k=1) 的 4-dim period
    編碼換成 band_fourier_encode 的 4·len(wavenumbers)-dim 編碼。
    wavenumbers=(1.0,) 時逐元素退化為 LearnableFourierEmb（單頻段）。
    """
    embed_dim: int
    wavenumbers: tuple = (1.0,)
    init_sigma: float = 2.0

    @nn.compact
    def __call__(self, xy: jnp.ndarray, domain_length: float) -> jnp.ndarray:
        if self.embed_dim % 2 != 0:
            raise ValueError(f"embed_dim 必須為偶數，收到 {self.embed_dim}")
        half = self.embed_dim // 2
        band_enc = band_fourier_encode(xy, domain_length, self.wavenumbers)  # [N, 4·n_bands]
        n_feat = 4 * len(self.wavenumbers)
        # 頻率正規化 init（curriculum 靜態版）：σ_k ∝ (k_min/k)² → 高頻（二階導 ∝ k²）起始更弱，
        # 避免 PDE 殘差被高頻二階導主宰而發散；低頻保留全強度。單一波數時 factor=1（等價 LearnableFourierEmb）。
        ks = jnp.asarray(self.wavenumbers, dtype=jnp.float32)
        row_scale = jnp.repeat((ks.min() / ks) ** 2, 4)[:, None]  # [4·n_bands, 1]
        kernel = self.param(
            'kernel',
            lambda key, shape: jax.random.normal(key, shape) * self.init_sigma * row_scale,
            (n_feat, half),
        )
        proj = band_enc @ kernel
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)


class TrainableFourierEmb(nn.Module):
    """trainable-frequency 版座標 embedding（方向1 exp_508）：**頻率 B 本身是 param**。

    與 BandFourierEmb 的差別是承重的：BandFourierEmb 的波數固定（可學的只有投影權重，
    網路只能對既定頻段加權），這裡 B 吃梯度 → 網路可以自己把頻率移到它要的位置。
    exp_508 的核心觀測量就是「訓練後 |B| 落在哪」。

    代價（已登錄，非疏漏）：trainable frequency 只能對**原始座標**投影達成，B 非整數時
    embedding 在週期域上不週期。把可學映射作用在已週期化的特徵上就是 LearnableFourierEmb /
    BandFourierEmb 本身，那條路已跑過。判讀時用「到整數格點的距離」把「拋棄中頻」與
    「拋棄非週期特徵」兩個競爭假設分開——見
    knowledge/experiments/kolmogorov-midband-identifiability-2026-08.md。
    """
    embed_dim: int
    init_freq_scale: float = 8.0
    #: 低通振幅權重的轉角波數；0 = 關（逐位元等同沒有這個功能）。
    #: w_j = 1/(1+(|B_j|/kc)²)，對 cos/sin 兩半同權。用途見 __call__ 內的註解。
    lowpass_kc: float = 0.0

    @nn.compact
    def __call__(self, xy: jnp.ndarray, domain_length: float) -> jnp.ndarray:
        if self.embed_dim % 2 != 0:
            raise ValueError(f"embed_dim 必須為偶數，收到 {self.embed_dim}")
        half = self.embed_dim // 2
        # B[:, j] = 第 j 個 feature 的 2D 波向量 (kx, ky)；|B[:, j]| 即該 feature 的頻率大小。
        B = self.param(
            'B',
            lambda key, shape: jax.random.normal(key, shape) * self.init_freq_scale,
            (2, half),
        )
        proj = (2.0 * jnp.pi / domain_length) * (xy @ B)   # [N, half]
        feats = jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)
        if self.lowpass_kc > 0.0:
            # 低通振幅權重：壓低高 |B| feature 的起始貢獻。動機是 exp_508 的 |B| 尾巴
            # 達 29.8，而 PDE 殘差的二階導 ∝ k²（exp_505 正是這樣崩的）。
            #
            # **stop_gradient 是承重的，不是最佳化**：若梯度能穿透 w，網路只要壓小 |B|
            # 就能免費換到更大振幅 → 「|B| 塌陷」變成被獎勵的方向，與物理無關，
            # 事前登錄「|B| 塌陷 = 資訊牆」的讀數就再也無法判別。加上實測 B 幾乎不動
            # （exp_508/509），這等價於靜態權重，但不引入那個人工誘因。
            mag = jnp.linalg.norm(jax.lax.stop_gradient(B), axis=0)      # [half]
            w = 1.0 / (1.0 + (mag / self.lowpass_kc) ** 2)               # [half]
            feats = feats * jnp.concatenate([w, w])[None, :]
        return feats


class FourierEmbs(nn.Module):
    """真 RFF（非週期）：raw xy → 高斯隨機投影 + cos/sin。對齊 pi_con/encodings.py:FourierEmbs。

    與 LearnableFourierEmb 的關鍵差別：不預先 sin/cos 週期化，故能區分 x=0 與 x=L
    （非週期域如 cylinder wake 必要：來流 x→0 與出口 x→L 物理意義截然不同）。
    """
    embed_dim: int
    init_sigma: float = 2.0

    @nn.compact
    def __call__(self, xy: jnp.ndarray, domain_length: float | None = None) -> jnp.ndarray:
        del domain_length  # RFF 有效頻率由 init_sigma 決定，與 domain 大小無關
        if self.embed_dim % 2 != 0:
            raise ValueError(f"embed_dim 必須為偶數，收到 {self.embed_dim}")
        half = self.embed_dim // 2
        # PyTorch nn.Linear(input_dim, half, bias=False)；init: normal(std=init_sigma)
        kernel = self.param(
            'kernel',
            lambda key, shape: jax.random.normal(key, shape) * self.init_sigma,
            (xy.shape[-1], half),
        )
        proj = xy @ kernel
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# CfC step + helpers
# ─────────────────────────────────────────────────────────────────────────────

class CfCStep(nn.Module):
    """Single-step CfC for `nn.scan`. 對齊 pi_con/blocks.py:CfCCell。

    signature: (h, (x, dt)) → (new_h, new_h)
    其中 carry=h，y=new_h（讓 scan 一併輸出整段 h 序列）。

    log_tau_a: linspace(log_tau_min, log_tau_max, hidden_size)
    init: ff1/ff2 用 Flax default (lecun_normal)；time_b 用 xavier_uniform + zero bias，
          對齊 PyTorch 原版 (`nn.init.xavier_uniform_(time_b.weight); zeros(time_b.bias)`)。

    input_dependent_tau（液態時間常數，LiquidNN 精髓）：
        False（預設）：τ = exp(log_τ_a)，per-channel 可訓練靜態參數。
        True：         log τ = log_τ_a + tau_mod_scale · tanh(time_a(x,h))
                       time_a 為小 linear layer（zero-init → 啟動時 τ 等同 static 路徑）。
        對齊 pi_con/blocks.py:CfCCell.input_dependent_tau=True 路徑。
    """
    hidden_size: int
    log_tau_min: float = -1.0
    log_tau_max: float = 1.0
    input_dependent_tau: bool = False
    tau_mod_scale: float = 2.0
    torch_style_init: bool = False

    @property
    def _kinit(self):
        return torch_linear_kernel_init if self.torch_style_init else nn.linear.default_kernel_init

    @nn.compact
    def __call__(self, h: jnp.ndarray, x_dt: tuple[jnp.ndarray, jnp.ndarray]):
        x_t, dt_t = x_dt
        xh = jnp.concatenate([x_t, h], axis=-1)

        ff1 = nn.Dense(self.hidden_size, name='ff1', kernel_init=self._kinit)
        ff2 = nn.Dense(self.hidden_size, name='ff2', kernel_init=self._kinit)
        time_b = nn.Dense(
            self.hidden_size,
            name='time_b',
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )

        log_tau_a = self.param(
            'log_tau_a',
            lambda key: jnp.linspace(self.log_tau_min, self.log_tau_max, self.hidden_size),
        )

        f1 = jnp.tanh(ff1(xh))
        f2 = jnp.tanh(ff2(xh))

        if self.input_dependent_tau:
            # liquid time-constant：per-channel τ 由 (x,h) 動態調制。
            # time_a zero-init → tanh(0)=0 → 啟動時 log_tau = log_tau_a，數值與 static 一致。
            time_a = nn.Dense(
                self.hidden_size,
                name='time_a',
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )
            log_tau = log_tau_a + self.tau_mod_scale * jnp.tanh(time_a(xh))
            tau_a = jnp.exp(log_tau)
        else:
            tau_a = jnp.exp(log_tau_a)

        t_b = time_b(xh)

        # dt_t 是 scalar（per-step）或 [K]（per-sensor）。 與 PyTorch unsqueeze(-1) 對齊。
        if jnp.ndim(dt_t) > 0:
            dt_t_b = dt_t[..., None]
        else:
            dt_t_b = dt_t
        gate = jax.nn.sigmoid(-tau_a * dt_t_b + t_b)
        new_h = gate * f1 + (1.0 - gate) * f2
        return new_h, new_h


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResidualMLPBlock(nn.Module):
    """LayerNorm → Dense → SiLU → Dense → residual add。
    對齊 pi_con/blocks.py:ResidualMLPBlock (activation='silu')。
    """
    d_model: int
    hidden_dim: int
    activation: str = 'silu'
    torch_style_init: bool = False
    # RWF（僅 decoder trunk 路徑會開）：use_rwf=False 時保持 nn.Dense（auto-name 'Dense_0/1' 不變）
    use_rwf: bool = False
    rwf_mean: float = 1.0
    rwf_stddev: float = 0.1

    @property
    def _kinit(self):
        return torch_linear_kernel_init if self.torch_style_init else nn.linear.default_kernel_init

    def _dense(self, features: int):
        if self.use_rwf:
            return RWFDense(features, kernel_init=self._kinit, use_rwf=True,
                            rwf_mean=self.rwf_mean, rwf_stddev=self.rwf_stddev)
        return nn.Dense(features, kernel_init=self._kinit)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = nn.LayerNorm()(x)
        y = self._dense(self.hidden_dim)(y)
        if self.activation == 'silu':
            y = nn.silu(y)
        elif self.activation == 'tanh':
            y = jnp.tanh(y)
        elif self.activation == 'gelu':
            y = nn.gelu(y)
        else:
            raise ValueError(f"未支援的 activation: {self.activation}")
        y = self._dense(self.d_model)(y)
        return x + y


class ForcingPrior(nn.Module):
    """Kolmogorov forcing (A, k_f)：fixed 或 learnable per-flag。
    對齊 pi_con/forcing.py:ForcingPrior。

    f_x = A · sin(2π · k_f · y_norm)，f_y = 0
      - A 透過 log(A) 參數化（保 A > 0）
      - k_f 透過 sigmoid(raw) · (k_max - k_min) + k_min 鎖在物理 band，避免 k_f→0 或 Nyquist

    使用：model.forcing() → (A, k_f) scalar tuple
    """
    A_init: float = 0.1
    k_f_init: float = 2.0
    learn_A: bool = False
    learn_k_f: bool = False
    k_f_min: float = 1.0
    k_f_max: float = 8.0

    @nn.compact
    def __call__(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.A_init <= 0:
            raise ValueError(f"A_init 必須 > 0，收到 {self.A_init}")
        if not (self.k_f_min < self.k_f_init < self.k_f_max):
            raise ValueError(
                f"k_f_init={self.k_f_init} 必須嚴格落在 ({self.k_f_min}, {self.k_f_max}) 之間"
            )
        # A
        if self.learn_A:
            log_A = self.param(
                'log_A',
                lambda key: jnp.array([math.log(self.A_init)], dtype=jnp.float32),
            )
            A = jnp.exp(log_A)[0]
        else:
            A = jnp.asarray(self.A_init, dtype=jnp.float32)
        # k_f
        if self.learn_k_f:
            x = (self.k_f_init - self.k_f_min) / (self.k_f_max - self.k_f_min)
            raw_init = math.log(x / (1.0 - x))
            raw_k_f = self.param(
                'raw_k_f',
                lambda key: jnp.array([raw_init], dtype=jnp.float32),
            )
            k_f = jax.nn.sigmoid(raw_k_f[0]) * (self.k_f_max - self.k_f_min) + self.k_f_min
        else:
            k_f = jnp.asarray(self.k_f_init, dtype=jnp.float32)
        return A, k_f


class ModifiedMLPBlock(nn.Module):
    """Wang 2021 modified MLP block — gating between U/V via current activation。
    對齊 pi_con/blocks.py:ModifiedMLPBlock。

    output: z_l = (1 - H_l) ⊙ U + H_l ⊙ V，其中 H_l = SiLU(Dense(LN(z))) 為當層 activation。
    Why: PINN spectral bias 的 architectural mitigation。動態 mix raw input feature (U)
         與 nonlinear transform (V)，緩解 NTK eigenvalue spectrum 不均（mid-k 學得慢）。
    """
    d_model: int
    hidden_dim: int
    torch_style_init: bool = False
    # RWF（僅 decoder trunk 路徑會開）：use_rwf=False 時保持 nn.Dense
    use_rwf: bool = False
    rwf_mean: float = 1.0
    rwf_stddev: float = 0.1

    @property
    def _kinit(self):
        return torch_linear_kernel_init if self.torch_style_init else nn.linear.default_kernel_init

    @nn.compact
    def __call__(self, z: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
        if self.d_model != self.hidden_dim:
            raise ValueError(
                f"ModifiedMLPBlock 要求 d_model==hidden_dim 才能 gate U/V; "
                f"收到 d_model={self.d_model}, hidden_dim={self.hidden_dim}"
            )
        h = nn.LayerNorm()(z)
        if self.use_rwf:
            h = RWFDense(self.hidden_dim, kernel_init=self._kinit, use_rwf=True,
                         rwf_mean=self.rwf_mean, rwf_stddev=self.rwf_stddev)(h)
        else:
            h = nn.Dense(self.hidden_dim, kernel_init=self._kinit)(h)
        h = nn.silu(h)
        return (1.0 - h) * U + h * V


class TokenSelfAttentionBlock(nn.Module):
    """Token-level self attention + FFN，pre-norm 結構。
    對齊 pi_con/blocks.py:TokenSelfAttentionBlock（PyTorch nn.MultiheadAttention with batch_first）。
    """
    d_model: int
    num_heads: int = 4
    torch_style_init: bool = False

    @property
    def _kinit(self):
        return torch_linear_kernel_init if self.torch_style_init else nn.linear.default_kernel_init

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model={self.d_model} 必須能被 num_heads={self.num_heads} 整除")
        y = nn.LayerNorm()(x)
        # Flax MultiHeadDotProductAttention 預期 input [..., L, d_model]
        # 對 [T, K, d_model] 視作 batch=T 對 K token 做 self-attention
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
        )(y, y, y)
        x = x + attn_out
        y2 = nn.LayerNorm()(x)
        ff = nn.Dense(2 * self.d_model, kernel_init=self._kinit)(y2)
        ff = nn.silu(ff)
        ff = nn.Dense(self.d_model, kernel_init=self._kinit)(ff)
        return x + ff


# ─────────────────────────────────────────────────────────────────────────────
# SpatialSetEncoder
# ─────────────────────────────────────────────────────────────────────────────

class SpatialSetEncoder(nn.Module):
    """sensor_vals + pos_enc → tokens。
    對齊 pi_con/encoders.py:SpatialSetEncoder。

    固定走可學習 Fourier 投影（fourier_embed_dim 維）：週期域用 LearnableFourierEmb、
    非週期域用 FourierEmbs（真 RFF）。
    """
    d_model: int
    num_layers: int
    sensor_value_dim: int
    domain_length: float = 1.0
    fourier_embed_dim: int = 128
    periodic_domain: bool = True
    torch_style_init: bool = False

    @property
    def _kinit(self):
        return torch_linear_kernel_init if self.torch_style_init else nn.linear.default_kernel_init

    def setup(self):
        # 固定可學習 Fourier 投影：週期域用 LearnableFourierEmb；非週期域用 FourierEmbs（真 RFF）。
        # periodic_domain 必須接到 spatial embedding（非僅 relpos wrap），否則非週期 cylinder
        # 來流 x→0 與出口 x→L 被週期編碼折疊，模型無法表達上游 freestream。
        self.spatial_emb = (LearnableFourierEmb(self.fourier_embed_dim)
                            if self.periodic_domain
                            else FourierEmbs(self.fourier_embed_dim))
        self.spatial_dim = self.fourier_embed_dim

    def encode_pos(self, sensor_pos: jnp.ndarray) -> jnp.ndarray:
        return self.spatial_emb(sensor_pos, self.domain_length)

    @nn.compact
    def __call__(self, sensor_vals: jnp.ndarray, pos_enc: jnp.ndarray) -> jnp.ndarray:
        """sensor_vals: [T, K, C]; pos_enc: [K, spatial_dim]
        Returns: tokens [T, K, d_model]
        """
        T = sensor_vals.shape[0]
        # broadcast pos_enc → [T, K, spatial_dim]
        pos_enc_b = jnp.broadcast_to(pos_enc[None, :, :], (T,) + pos_enc.shape)
        base = jnp.concatenate([pos_enc_b, sensor_vals], axis=-1)  # [T, K, spatial_dim+C]

        base = nn.LayerNorm(name='base_norm')(base)
        # token_in: LN → Dense(2d) → SiLU → Dense(d)
        tokens = nn.LayerNorm(name='token_in_ln')(base)
        tokens = nn.Dense(2 * self.d_model, name='token_in_fc1', kernel_init=self._kinit)(tokens)
        tokens = nn.silu(tokens)
        tokens = nn.Dense(self.d_model, name='token_in_fc2', kernel_init=self._kinit)(tokens)

        # blocks
        for i in range(max(self.num_layers, 1)):
            tokens = ResidualMLPBlock(
                d_model=self.d_model,
                hidden_dim=2 * self.d_model,
                torch_style_init=self.torch_style_init,
                name=f'block_{i}',
            )(tokens)

        # out_proj: LN → Dense(2d) → SiLU → Dense(d)
        out = nn.LayerNorm(name='out_proj_ln')(tokens)
        out = nn.Dense(2 * self.d_model, name='out_proj_fc1', kernel_init=self._kinit)(out)
        out = nn.silu(out)
        out = nn.Dense(self.d_model, name='out_proj_fc2', kernel_init=self._kinit)(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# TemporalCfCEncoder（含 lax.scan over T）
# ─────────────────────────────────────────────────────────────────────────────

class TemporalCfCEncoder(nn.Module):
    """以 CfC 演化 sensor token 序列。
    對齊 pi_con/encoders.py:TemporalCfCEncoder。

    nn.scan transform：把 CfCStep 包成 sequence scan，
    variable_broadcast='params' 表示 cell 的 params 在所有 timestep 共享。

    注意：Re-FiLM modulation 預設在 per-layer CfC loop 內套用。num_layers=0（如 B2
    cross-attn-only ablation）時 cell loop 不執行，改在 token-attended 序列上套一次 FiLM，
    確保仍 Re-aware（且 re_scale/re_shift 不淪為 dead param）。
    """
    d_model: int
    num_layers: int
    num_token_attention_layers: int = 1
    token_attention_heads: int = 4
    cfc_log_tau_min: float = -1.0
    cfc_log_tau_max: float = 1.0
    cfc_input_dependent_tau: bool = False
    cfc_tau_mod_scale: float = 2.0
    torch_style_init: bool = False

    @property
    def _kinit(self):
        return torch_linear_kernel_init if self.torch_style_init else nn.linear.default_kernel_init

    def setup(self):
        # Re FiLM modulation: attended = attended * (1 + scale(Re)) + shift(Re)
        # Why: Re 控制 NS 黏性／慣性比，作用在時空尺度上以「縮放」為主、平移為輔；
        #      純加法 bias 等同假設不同 Re 只是特徵平移，無法調控 CfC 時間常數。
        # Init policy:
        #   re_scale: kernel=zeros + bias=zeros → init 時對任何 Re 都輸出 0；
        #             配合 (1 + scale) 保證 init 等同 identity，FiLM 必須「學會」乘法縮放。
        #   re_shift: kernel=lecun_normal + bias=zeros（沿用 Flax Dense 預設），
        #             對齊 PyTorch reference 的 re_proj 加法行為。
        self.re_scale = nn.Dense(
            self.d_model, name='re_scale',
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )
        self.re_shift = nn.Dense(
            self.d_model, name='re_shift',
            kernel_init=self._kinit,
            bias_init=nn.initializers.zeros,
        )
        self.token_blocks = [
            TokenSelfAttentionBlock(
                d_model=self.d_model,
                num_heads=self.token_attention_heads,
                torch_style_init=self.torch_style_init,
                name=f'token_block_{i}',
            )
            for i in range(max(self.num_token_attention_layers, 0))
        ]
        # nn.scan: CfCStep 對時間軸 scan，params 跨 timestep broadcast（共享）
        # Forward scan
        ScannedCfCFwd = nn.scan(
            CfCStep,
            variable_broadcast='params',
            split_rngs={'params': False},
            in_axes=0,
            out_axes=0,
            reverse=False,
        )
        self.cells = [
            ScannedCfCFwd(
                hidden_size=self.d_model,
                log_tau_min=self.cfc_log_tau_min,
                log_tau_max=self.cfc_log_tau_max,
                input_dependent_tau=self.cfc_input_dependent_tau,
                tau_mod_scale=self.cfc_tau_mod_scale,
                torch_style_init=self.torch_style_init,
                name=f'cell_{i}',
            )
            for i in range(self.num_layers)
        ]

    def __call__(
        self,
        spatial_states: jnp.ndarray,  # [T, K, d_model]
        re_norm: float,
        sensor_time: jnp.ndarray,     # [T]
    ) -> jnp.ndarray:
        # dts[0] = sensor_time[0]; dts[t>0] = sensor_time[t] - sensor_time[t-1]
        dts = jnp.concatenate([sensor_time[:1], jnp.diff(sensor_time)])  # [T]

        re_t = jnp.array([[re_norm]], dtype=spatial_states.dtype)
        re_scale_v = self.re_scale(re_t).reshape(1, 1, -1)  # [1, 1, d_model]
        re_shift_v = self.re_shift(re_t).reshape(1, 1, -1)  # [1, 1, d_model]

        seq = spatial_states
        # 先序列疊所有 token attention blocks（num_token_attention_layers 真實生效）
        # 修 Bug：原先邏輯把 token_block 綁 num_layers (CfC count)，導致
        # num_token_attention_layers=2 + num_layers=1 時 token_block_1 從未被 call →
        # Flax 不 init 它的 params，少 ~525k 參數量（與 pi-lnn 不一致的 root cause）。
        for block in self.token_blocks:
            seq = block(seq)
        if not self.cells:
            # num_layers=0：cell loop 不跑 → FiLM 不會被套用，模型會 Re-blind 且 re_scale/
            # re_shift 變 dead param。在此對 token-attended 序列套一次 FiLM，保證 Re-aware。
            seq = seq * (1.0 + re_scale_v) + re_shift_v
        for layer_idx, cell in enumerate(self.cells):
            attended = seq
            # FiLM modulation: (1 + scale) 在 init zero → identity；訓練可學乘法縮放
            attended = attended * (1.0 + re_scale_v) + re_shift_v  # [T, K, d_model]

            # nn.scan CfC：carry=h [K, d_model]，input=(x_t [K, d_model], dt_t scalar)
            K = attended.shape[1]
            h_init = jnp.zeros((K, self.d_model), dtype=attended.dtype)
            _, fwd_out = cell(h_init, (attended, dts))
            outputs = fwd_out
            # 層間殘差：第二層起加上前一層輸出
            seq = outputs + seq if layer_idx > 0 else outputs
        return seq


# ─────────────────────────────────────────────────────────────────────────────
# DeepONetCfCDecoder（含 cross-attention）
# ─────────────────────────────────────────────────────────────────────────────

class DeepONetCfCDecoder(nn.Module):
    """以 CfC token states 作 branch、query 作 trunk 的 DeepONet 解碼器。
    對齊 pi_con/decoder.py:DeepONetCfCDecoder（完整 ablation 開關支援）。

    開關:
      - use_modified_mlp: 用 Wang 2021 mMLP gating U/V（PINN spectral bias 緩解）
      - use_locality_decay: cross-attention scores 加 -α·r 距離懲罰（log-space）
      - disable_cross_attention: B1 ablation，cross-attention 改為 mean-pool over K
      - fusion_temperature_init: 控制 trunk·branch 內積的溫度（learnable param）
      - component_scale / component_bias: 各 component (u,v,p) 的線性 affine（learnable）
    """
    d_model: int
    d_time: int
    domain_length: float = 1.0
    use_temporal_anchor: bool = False
    T_total: float = 5.0
    temporal_anchor_harmonics: int = 2
    num_query_mlp_layers: int = 0
    query_mlp_hidden_dim: int = 256
    output_head_gain: float = 1.0
    operator_rank: int = 64
    fourier_embed_dim: int = 128
    decoder_attention_heads: int = 1
    # ── 方向1: opt-in band-targeted mid-band embedding（空 tuple → 關 → bit-identical）──
    mid_band_wavenumbers: tuple = ()
    mid_band_embed_dim: int = 128
    mid_band_init_sigma: float = 2.0
    # ── 方向1 exp_508: opt-in trainable-frequency embedding（False → 關 → bit-identical）──
    use_trainable_fourier: bool = False
    trainable_fourier_dim: int = 128
    trainable_fourier_init_scale: float = 8.0
    trainable_fourier_lowpass_kc: float = 0.0
    # ── 進階開關 ──
    use_modified_mlp: bool = False
    # Wang 2021 modified DeepONet 的跨分支對齊（arXiv 2110.01654 eq. 3.23/3.26/3.27）：
    # 原文 U 取自 branch 輸入、V 取自 trunk 輸入，兩條網路每層共用同一組 U/V。
    # 既有 use_modified_mlp 只做到 trunk 單邊（U/V 皆由座標產生），缺的就是這一半。
    # 兩旗標正交，皆需 use_modified_mlp=True。
    mmlp_branch_u: bool = False      # U 改由 branch tokens（sensor 側）產生
    mmlp_gate_branch: bool = False   # branch context 也走 gated block，共用同一組 U/V
    use_locality_decay: bool = False
    disable_cross_attention: bool = False
    fusion_temperature_init: float = 0.0  # 0.0 → 用預設 1/sqrt(rank)
    # ── 幾何泛化開關（cross-attention relpos bias）──
    # relpos_bias_mode: "radial"(預設,各向同性,對齊 pi-lnn) | "vector"(各向異性,bias 吃 rel 向量)
    # periodic_domain: True(預設,Kolmogorov 週期域 wrap) | False(有界/幾何域,不 wrap)
    relpos_bias_mode: str = "radial"
    # relpos bias 的正規化方式："layernorm"(預設,現行行為) | "none"
    # ⚠ 已知失效組合：radial + no-SDF 時 bias_in 特徵維 = 1，而 LayerNorm 對長度 1 的軸
    #   輸出恆為 0（mean=x, var=0）→ rel_bias 退化成跨 (n,k) 的常數 → softmax 前加常數
    #   是 no-op，整條 bias 不生效（ln/fc1/fc2 成死參數）。398 份 config 中 333 份吃這個
    #   組合，含論文 B3 主結果 exp_245_b3_les_T50。預設保留 "layernorm"：既有 ckpt 與
    #   已發表 run 全在此狀態下產生（Never Break Userspace）。"none" 是修好的對照臂。
    relpos_bias_norm: str = "layernorm"
    # bias 輸出層 zero-init：讓 bias 從 0 起步再長出來。cylinder 上 relpos_bias_norm="none"
    # 有 1/5 seed 在 step 2 發散，機制假設是 fc1 的輸入由恆定 0 變成 O(1) 的 rel_r，
    # bias 一開始就非零而把 attention scores 推到極端（見 knowledge/experiments/
    # cylinder-advective-attention-negative-2026-07-29.md 的 Follow-up）。
    relpos_bias_zero_init: bool = False
    periodic_domain: bool = True
    # #3 SDF features：把「到固體的解析 signed distance」餵進 trunk + attention bias。
    # 解析圓形 SDF = |xy − center| − r（C∞ 可微，對 fof 二階導友善；對齊 pi-lnn「φ 須對 xy 可微」要求）。
    # 預設 use_sdf_features=False → 無 body（Kolmogorov），行為不變。
    use_sdf_features: bool = False
    body_center_x: float = 0.5
    body_center_y: float = 0.5
    body_radius: float = 0.0
    # cross-attention 類型："scalar"(預設,dot-product+scalar bias) | "vector"(PTv2 grouped vector attn)
    attention_kind: str = "scalar"
    torch_style_init: bool = False
    # RWF（Random Weight Factorization）：僅作用於 trunk 座標 MLP（trunk_in/blocks/out + U/V proj）。
    # 預設 False → trunk Dense 維持標準 nn.Dense，對既有結果零影響。
    use_rwf: bool = False
    rwf_mean: float = 1.0
    rwf_stddev: float = 0.1

    @property
    def _kinit(self):
        return torch_linear_kernel_init if self.torch_style_init else nn.linear.default_kernel_init

    @property
    def _rwf_kwargs(self) -> dict:
        """傳給 trunk RWFDense / *MLPBlock 的 RWF 開關。"""
        return dict(use_rwf=self.use_rwf, rwf_mean=self.rwf_mean, rwf_stddev=self.rwf_stddev)

    def _relpos_bias_dim(self) -> int:
        """bias_in 的特徵維。=1 時 LayerNorm 會把整條 bias 塌成常數（見 relpos_bias_norm）。"""
        d = 1                                             # rel_r
        if self.relpos_bias_mode == "vector":
            d += 2                                        # rel
        if self.use_sdf_features:
            d += 2                                        # sdf_q, sdf_s
        return d

    def _trunk_dense(self, features: int, name: str, **kw):
        """trunk 座標路徑用 RWFDense（explicit name → use_rwf=False 時數值同 nn.Dense）。"""
        return RWFDense(features, name=name, kernel_init=self._kinit, **self._rwf_kwargs, **kw)

    def setup(self):
        # spatial_emb 與 SpatialSetEncoder 對稱：週期域 LearnableFourierEmb；非週期域 FourierEmbs（RFF）。
        self.spatial_emb = (LearnableFourierEmb(self.fourier_embed_dim)
                            if self.periodic_domain
                            else FourierEmbs(self.fourier_embed_dim))
        # 方向1: 兩個獨立的 opt-in 座標 embedding 分支，各自 concat 到 spatial_emb 後面。
        # 關閉時不實例化 → 零額外 param、spatial_dim 不變（bit-identical）。
        extra_spatial_dim = 0
        if self.mid_band_wavenumbers:
            self.band_spatial_emb = BandFourierEmb(
                embed_dim=self.mid_band_embed_dim,
                wavenumbers=self.mid_band_wavenumbers,
                init_sigma=self.mid_band_init_sigma,
            )
            extra_spatial_dim += self.mid_band_embed_dim
        if self.use_trainable_fourier:
            self.trainable_spatial_emb = TrainableFourierEmb(
                embed_dim=self.trainable_fourier_dim,
                init_freq_scale=self.trainable_fourier_init_scale,
                lowpass_kc=self.trainable_fourier_lowpass_kc,
            )
            extra_spatial_dim += self.trainable_fourier_dim
        self.spatial_dim = self.fourier_embed_dim + extra_spatial_dim
        self.temporal_dim = (
            2 * self.temporal_anchor_harmonics if self.use_temporal_anchor else 0
        )
        # 組件：trunk path
        self.time_proj = nn.Dense(self.d_time, name='time_proj', kernel_init=self._kinit)
        # component_emb: 3 個 component (u, v, p) 各 8 維
        self.component_emb = self.param(
            'component_emb',
            lambda key, shape: jax.random.normal(key, shape) * 0.1,
            (3, 8),
        )
        self.trunk_in = self._trunk_dense(self.query_mlp_hidden_dim, name='trunk_in')
        if (self.mmlp_branch_u or self.mmlp_gate_branch) and not self.use_modified_mlp:
            raise ValueError(
                "mmlp_branch_u / mmlp_gate_branch 需要 use_modified_mlp=True；"
                "沒有 mMLP 路徑時 U/V gating 不存在，靜默忽略等於讓 config 說謊"
            )
        # modified MLP 路徑：U/V projections（layer-independent）+ ModifiedMLPBlock
        if self.use_modified_mlp:
            # U 是取代不是並存：留著另一條會是孤兒參數。RWF 只用於 trunk 座標 MLP，
            # 故 branch 側投影走一般 Dense。
            if self.mmlp_branch_u:
                self.branch_U_proj = nn.Dense(
                    self.query_mlp_hidden_dim, name='branch_U_proj', kernel_init=self._kinit)
            else:
                self.trunk_U_proj = self._trunk_dense(self.query_mlp_hidden_dim, name='trunk_U_proj')
            self.trunk_V_proj = self._trunk_dense(self.query_mlp_hidden_dim, name='trunk_V_proj')
            self.trunk_blocks = [
                ModifiedMLPBlock(
                    d_model=self.query_mlp_hidden_dim,
                    hidden_dim=self.query_mlp_hidden_dim,
                    torch_style_init=self.torch_style_init,
                    name=f'trunk_block_{i}',
                    **self._rwf_kwargs,
                )
                for i in range(self.num_query_mlp_layers)
            ]
            if self.mmlp_gate_branch:
                self.branch_gate_blocks = [
                    ModifiedMLPBlock(
                        d_model=self.query_mlp_hidden_dim,
                        hidden_dim=self.query_mlp_hidden_dim,
                        torch_style_init=self.torch_style_init,
                        name=f'branch_gate_block_{i}',
                    )
                    for i in range(self.num_query_mlp_layers)
                ]
        else:
            self.trunk_blocks = [
                ResidualMLPBlock(
                    d_model=self.query_mlp_hidden_dim,
                    hidden_dim=self.query_mlp_hidden_dim,
                    torch_style_init=self.torch_style_init,
                    name=f'trunk_block_{i}',
                    **self._rwf_kwargs,
                )
                for i in range(self.num_query_mlp_layers)
            ]
        # cross-attention
        if self.query_mlp_hidden_dim % self.decoder_attention_heads != 0:
            raise ValueError(
                f"query_mlp_hidden_dim={self.query_mlp_hidden_dim} 須被 "
                f"decoder_attention_heads={self.decoder_attention_heads} 整除"
            )
        self.attn_head_dim = self.query_mlp_hidden_dim // self.decoder_attention_heads
        self.branch_norm = nn.LayerNorm(name='branch_norm')
        self.branch_token_proj = nn.Dense(self.query_mlp_hidden_dim, name='branch_token_proj', kernel_init=self._kinit)
        # 真 MHA：q/k/v 各輸出 H*D（= hidden），reshape 後每組 D 維即獨立 head 投影。
        # Dense(H*D) 等效於 H 組 Dense(D) 並排——gradient 在各 D 列間獨立更新。
        self.branch_query_proj = nn.Dense(self.query_mlp_hidden_dim, name='branch_query_proj', kernel_init=self._kinit)
        self.branch_key_proj = nn.Dense(self.query_mlp_hidden_dim, name='branch_key_proj', kernel_init=self._kinit)
        self.branch_value_proj = nn.Dense(self.query_mlp_hidden_dim, name='branch_value_proj', kernel_init=self._kinit)
        # relpos_bias: [LN] → Dense(hidden) → SiLU → Dense(1)（scalar attention 用）
        if self.relpos_bias_norm not in ("layernorm", "none"):
            raise ValueError(
                f"relpos_bias_norm 必須是 'layernorm' 或 'none'，收到 {self.relpos_bias_norm!r}"
            )
        if self.relpos_bias_norm == "layernorm":
            self.relpos_bias_ln = nn.LayerNorm(name='relpos_bias_ln')
        self.relpos_bias_fc1 = nn.Dense(self.query_mlp_hidden_dim, name='relpos_bias_fc1', kernel_init=self._kinit)
        self.relpos_bias_fc2 = nn.Dense(
            1, name='relpos_bias_fc2',
            kernel_init=nn.initializers.zeros if self.relpos_bias_zero_init else self._kinit)
        # vector attention（PTv2）：pos encoding（幾何特徵→hidden）+ weight MLP（relation→逐通道權重）
        if self.attention_kind == "vector":
            self.relpos_vec_proj = nn.Dense(self.query_mlp_hidden_dim, name='relpos_vec_proj', kernel_init=self._kinit)
            self.attn_w_fc1 = nn.Dense(self.query_mlp_hidden_dim, name='attn_w_fc1', kernel_init=self._kinit)
            self.attn_w_fc2 = nn.Dense(self.query_mlp_hidden_dim, name='attn_w_fc2', kernel_init=self._kinit)
        # branch_context: LN → Dense → SiLU → Dense (residual)
        self.branch_context_ln = nn.LayerNorm(name='branch_context_ln')
        self.branch_context_fc1 = nn.Dense(self.query_mlp_hidden_dim, name='branch_context_fc1', kernel_init=self._kinit)
        self.branch_context_fc2 = nn.Dense(self.query_mlp_hidden_dim, name='branch_context_fc2', kernel_init=self._kinit)
        # 輸出：trunk_out / branch_proj → 3 * rank
        # PyTorch xavier_normal_(gain=output_head_gain)；Flax xavier_normal 接受 in_axis/out_axis 但無 gain，
        # 用 lambda 包一層 scale 對齊 PyTorch gain 語意。
        def _xavier_normal_with_gain(key, shape, dtype=jnp.float32):
            base = nn.initializers.xavier_normal()(key, shape, dtype)
            return base * self.output_head_gain
        self.trunk_out = RWFDense(
            3 * self.operator_rank,
            name='trunk_out',
            kernel_init=_xavier_normal_with_gain,
            bias_init=nn.initializers.zeros,
            **self._rwf_kwargs,
        )
        self.branch_proj = nn.Dense(
            3 * self.operator_rank,
            name='branch_proj',
            kernel_init=_xavier_normal_with_gain,
            bias_init=nn.initializers.zeros,
        )
        # learnable fusion_temperature (log space, shape (1,) 對齊 pi-lnn)
        if self.fusion_temperature_init > 0:
            temp_init = self.fusion_temperature_init
        else:
            temp_init = 1.0 / math.sqrt(self.operator_rank)
        self.log_fusion_temperature = self.param(
            'log_fusion_temperature',
            lambda key: jnp.array([math.log(temp_init)], dtype=jnp.float32),
        )
        # learnable component_scale (3,) / component_bias (3,) — affine per (u, v, p)
        self.component_scale = self.param(
            'component_scale',
            lambda key: jnp.ones((3,), dtype=jnp.float32),
        )
        self.component_bias = self.param(
            'component_bias',
            lambda key: jnp.zeros((3,), dtype=jnp.float32),
        )
        # locality decay：α = exp(log_locality_decay)，softmax 前 score -= α·r
        if self.use_locality_decay:
            self.log_locality_decay = self.param(
                'log_locality_decay',
                lambda key: jnp.array([-2.0], dtype=jnp.float32),  # α ≈ 0.135 中性 init
            )

    def __call__(
        self,
        xy: jnp.ndarray,           # [N, 2]
        t_q: jnp.ndarray,          # [N]
        h_states: jnp.ndarray,     # [T, K, d_model]
        sensor_time: jnp.ndarray,  # [T]
        sensor_pos: jnp.ndarray,   # [K, 2]
    ) -> jnp.ndarray:
        """回傳 [N, 3] = (u, v, p)。對應 pi_con DeepONetCfCDecoder.forward_uvp。"""
        # ── c-independent ──
        # idx for each query t_q: searchsorted(sensor_time, t_q, side='right') - 1
        # 用比較求和等價替換 searchsorted（sensor_time 已排序）：sum(st<=t)−1。
        # Why: searchsorted 內部 binary-search scan 無 jax.experimental.jet rule；
        #      比較求和為純 elementwise，讓 Taylor-mode（jet）可穿透整個 decoder。
        idx = jnp.sum(sensor_time[None, :] <= t_q[:, None], axis=1).astype(jnp.int32) - 1
        idx = jnp.clip(idx, 0, h_states.shape[0] - 1)
        dt_to_query = jnp.clip(t_q - sensor_time[idx], min=0.0)  # [N]

        pos_enc = self.spatial_emb(xy, self.domain_length)
        # 方向1: opt-in concat band-targeted mid-band embedding（flag 關時此分支不存在）
        if self.mid_band_wavenumbers:
            pos_enc = jnp.concatenate(
                [pos_enc, self.band_spatial_emb(xy, self.domain_length)], axis=-1)
        # 方向1 exp_508: opt-in concat trainable-frequency embedding（同上，關時不存在）
        if self.use_trainable_fourier:
            pos_enc = jnp.concatenate(
                [pos_enc, self.trainable_spatial_emb(xy, self.domain_length)], axis=-1)
        time_e = self.time_proj(dt_to_query[:, None])  # [N, d_time]

        # #3 解析 SDF（到圓形 body 的 signed distance；可微 → fof 二階導正確）
        if self.use_sdf_features:
            cx, cy, r = self.body_center_x, self.body_center_y, self.body_radius
            sdf_q = jnp.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2 + 1e-8) - r   # [N]
            sdf_s = jnp.sqrt(
                (sensor_pos[:, 0] - cx) ** 2 + (sensor_pos[:, 1] - cy) ** 2 + 1e-8
            ) - r                                                                       # [K]
        else:
            sdf_q = sdf_s = None

        # branch tokens → key/value。三個投影提升到 gather 之前：對 T 個 sensor 時刻各算
        # 一次，再取 query 對應的那個，而不是對 N 個 query（N≫T）各算一次。前向逐位元相同，
        # 參數梯度差 ~1e-6——gather 的 adjoint 是 scatter-add，把線性算子搬到 gather 另一側
        # 會重組梯度的加總順序，而浮點加法不結合。因此**永遠過不了 §7.1 對拍**，那是這個
        # 寫法的固有性質，不是可修的 bug。
        branch_tokens_all = self.branch_token_proj(h_states)           # [T, K, hidden]
        k_proj = self.branch_key_proj(branch_tokens_all)[idx]          # [N, K, hidden]
        v_proj = self.branch_value_proj(branch_tokens_all)[idx]        # [N, K, hidden]

        # rel coord：query→sensor 位移
        rel = xy[:, None, :] - sensor_pos[None, :, :]                  # [N, K, 2]
        # #2 週期 wrap：僅 periodic_domain=True（Kolmogorov 環面）。有界/幾何域關掉避免假鄰接。
        # 註：此 jnp.round 不在 folx registry（stderr 會有 "round not in registry"），但實測
        # （job 4551 A/B）對 s/step 無影響——stop_gradient 迴避 fallback 後速度與 warning 皆
        # 不變，該 fallback 對這顆 [N,K,2] round 的成本可忽略。故不為此改動（保持單行清晰）。
        if self.periodic_domain:
            rel = rel - jnp.round(rel / self.domain_length) * self.domain_length
        rel_r = jnp.sqrt(jnp.sum(rel ** 2, axis=-1, keepdims=True) + 1e-8)  # [N, K, 1]
        N_q, K_s = xy.shape[0], sensor_pos.shape[0]
        # #1 relpos bias 輸入：radial(各向同性,現行) 或 vector(含方向 → 各向異性)
        bias_parts = []
        if self.relpos_bias_mode == "vector":
            bias_parts.append(rel)                                    # [N, K, 2] 方向
        bias_parts.append(rel_r)                                      # [N, K, 1] 半徑
        # #3 SDF：query/sensor 各自到 body 的距離 → attention 學「貼壁 / 尾流」相依聚合（遮蔽 proxy）
        if self.use_sdf_features:
            bias_parts.append(jnp.broadcast_to(sdf_q[:, None, None], (N_q, K_s, 1)))
            bias_parts.append(jnp.broadcast_to(sdf_s[None, :, None], (N_q, K_s, 1)))
        bias_in = jnp.concatenate(bias_parts, axis=-1) if len(bias_parts) > 1 else bias_parts[0]
        if self.attention_kind == "scalar":
            rel_b = self.relpos_bias_ln(bias_in) if self.relpos_bias_norm == "layernorm" else bias_in
            rel_b = self.relpos_bias_fc1(rel_b)
            rel_b = nn.silu(rel_b)
            rel_bias = self.relpos_bias_fc2(rel_b).squeeze(-1)         # [N, K]
        else:
            rel_bias = None                                           # vector mode 用 bias_in 當 pos

        # ── c-conditional：批次化 c=0,1,2 ──
        N = xy.shape[0]
        base_inputs = [pos_enc]
        if self.use_temporal_anchor:
            base_inputs.append(
                temporal_phase_anchor(t_q[:, None], self.T_total, self.temporal_anchor_harmonics)
            )
        base_inputs.append(time_e)
        # #3 SDF 進 trunk：讓預測場值能依「到 body 的距離」變化（可微 → fof 二階導正確）
        if self.use_sdf_features:
            base_inputs.append(sdf_q[:, None])                        # [N, 1]
        base_feat = jnp.concatenate(base_inputs, axis=-1)             # [N, spatial+temporal+d_time(+1)]

        # 對齊 PyTorch: emb_c [3, 8] → [3, N, 8] → trunk_in_3 [3, N, base+8]
        base_feat_3 = jnp.broadcast_to(base_feat[None], (3,) + base_feat.shape)
        emb_c_3 = jnp.broadcast_to(self.component_emb[:, None, :], (3, N, 8))
        trunk_in_3 = jnp.concatenate([base_feat_3, emb_c_3], axis=-1).reshape(3 * N, -1)

        # trunk MLP：兩條路徑視 use_modified_mlp 選擇
        trunk_feat = nn.silu(self.trunk_in(trunk_in_3))
        if self.use_modified_mlp:
            # mMLP: 預算 U/V (layer-independent)，每層 block 在 U/V gate
            if self.mmlp_branch_u:
                # 原文 eq. 3.23 的 U = φ(W_u·u + b_u)，u 為 branch 輸入。取 attention
                # **之前**的 branch tokens 做 mean-pool over K：U 的定義因此與
                # disable_cross_attention 正交。用 attention 後的 branch_ctx 會有循環
                # 依賴（attention 的 query 來自 trunk_feat），兩種 flag 組合下 U 的
                # 語意就不一致，變因會髒。
                U_emb = _repeat3(nn.silu(self.branch_U_proj(
                    jnp.mean(branch_tokens_all[idx], axis=1))))
            else:
                U_emb = nn.silu(self.trunk_U_proj(trunk_in_3))
            V_emb = nn.silu(self.trunk_V_proj(trunk_in_3))
            for block in self.trunk_blocks:
                trunk_feat = block(trunk_feat, U_emb, V_emb)
        else:
            for block in self.trunk_blocks:
                trunk_feat = block(trunk_feat)
        trunk_basis = self.trunk_out(trunk_feat).reshape(3 * N, 3, self.operator_rank)

        # 對齊 PyTorch: branch_query = LN(trunk_feat) → branch_query_proj → q
        branch_query = self.branch_norm(trunk_feat)                    # [3N, hidden]
        q = self.branch_query_proj(branch_query)                       # [3N, hidden]

        # c-independent k/v 對齊到 [3N, K, hidden]（tile c=0/1/2 段）
        k_3 = _repeat3(k_proj)                              # [3N, K, hidden]
        v_3 = _repeat3(v_proj)                              # [3N, K, hidden]
        rel_r_3 = _repeat3(rel_r)                           # [3N, K, 1]

        if self.disable_cross_attention:
            # B1 ablation: cross-attention → mean-pool over K sensor tokens
            branch_ctx = jnp.mean(v_3, axis=1)                          # [3N, hidden]
        elif self.attention_kind == "vector":
            # PTv2 grouped vector attention：pos encoding（幾何特徵）+ MLP(q−k+pos) 逐通道權重。
            # 不同 hidden 通道可對不同 sensor 賦不同權重（scalar dot-product 做不到）。
            pos = self.relpos_vec_proj(bias_in)                        # [N, K, hidden]
            pos_3 = _repeat3(pos)                           # [3N, K, hidden]
            relation = q[:, None, :] - k_3                             # [3N, K, hidden]
            gamma = self.attn_w_fc2(nn.silu(self.attn_w_fc1(relation + pos_3)))  # [3N, K, hidden]
            if self.use_locality_decay:
                decay_rate = jnp.exp(self.log_locality_decay)
                gamma = gamma - decay_rate * rel_r_3                   # broadcast hidden
            attn = jax.nn.softmax(gamma, axis=1)                       # softmax over K（逐通道）
            _ctx = jnp.sum(attn * (v_3 + pos_3), axis=1)              # [3N, hidden]
            branch_ctx = _ctx
        else:
            # scalar multi-head dot-product cross-attention（現行）
            rel_bias_3 = _repeat3(rel_bias)                    # [3N, K]
            H = self.decoder_attention_heads
            D = self.attn_head_dim
            K = k_3.shape[1]
            q_h = q.reshape(3 * N, H, D)
            k_h = k_3.reshape(3 * N, K, H, D)
            v_h = v_3.reshape(3 * N, K, H, D)
            scores = jnp.einsum("nhd,nkhd->nhk", q_h, k_h) / math.sqrt(D)
            scores = scores + rel_bias_3[:, None, :]                    # broadcast H
            if self.use_locality_decay:
                decay_rate = jnp.exp(self.log_locality_decay)           # α > 0
                scores = scores - decay_rate * rel_r_3.squeeze(-1)[:, None, :]
            attn = jax.nn.softmax(scores, axis=-1)
            branch_ctx_h = jnp.einsum("nhk,nkhd->nhd", attn, v_h)
            _ctx = branch_ctx_h.reshape(3 * N, H * D)
            branch_ctx = _ctx

        # branch_context residual
        bc = self.branch_context_ln(branch_ctx)
        bc = self.branch_context_fc1(bc)
        bc = nn.silu(bc)
        bc = self.branch_context_fc2(bc)
        branch_ctx = branch_ctx + bc
        # 原文 eq. 3.27：branch 側也用同一組 U/V 逐層 gate。串在既有 residual FFN
        # 之後而非取代它——off→on 是純新增，變因單一；本架構的 branch 路徑本來就不是
        # 原文的純 MLP stack（前有 CfC encoder 與 cross-attention），取代 FFN 並不會
        # 更接近原文，只會多一個混淆變因。U_emb/V_emb 必然已定義：setup 的 fail fast
        # 保證 mmlp_gate_branch ⇒ use_modified_mlp。
        if self.mmlp_gate_branch:
            for block in self.branch_gate_blocks:
                branch_ctx = block(branch_ctx, U_emb, V_emb)
        branch_basis = self.branch_proj(branch_ctx).reshape(3 * N, 3, self.operator_rank)

        # gather per-component basis 並 fusion
        c_flat = jnp.repeat(jnp.arange(3), N)                          # [3N]
        trunk_sel = jnp.take_along_axis(
            trunk_basis, c_flat[:, None, None].repeat(self.operator_rank, axis=2), axis=1
        ).squeeze(1)                                                   # [3N, rank]
        branch_sel = jnp.take_along_axis(
            branch_basis, c_flat[:, None, None].repeat(self.operator_rank, axis=2), axis=1
        ).squeeze(1)                                                   # [3N, rank]
        # learnable fusion temperature
        fusion_temperature = jnp.exp(self.log_fusion_temperature).astype(trunk_sel.dtype)  # shape (1,)
        out = jnp.sum(trunk_sel * branch_sel, axis=1) * fusion_temperature                 # [3N]
        # learnable per-component affine
        out = out * self.component_scale[c_flat] + self.component_bias[c_flat]             # [3N]
        # [3N] → [3, N] → [N, 3]
        uvp = out.reshape(3, N).T
        return uvp


# ─────────────────────────────────────────────────────────────────────────────
# LiquidOperator（combine all）
# ─────────────────────────────────────────────────────────────────────────────

class LiquidOperator(nn.Module):
    """完整 pi-lnn 模型：spatial encoder → temporal CfC → DeepONet decoder。
    對齊 pi_con/operator.py:LiquidOperator。

    forward signature:
      sensor_vals [T, K, C], sensor_pos [K, 2], re_norm scalar, sensor_time [T],
      xy [N, 2], t_q [N]  →  [N, 3] (u, v, p)
    """
    sensor_value_dim: int
    d_model: int
    d_time: int
    num_spatial_encoder_layers: int
    num_temporal_cfc_layers: int
    domain_length: float = 1.0
    # 2026-08-03 False→True，與 config.MODEL_SCHEMA 對齊（架構候選 H）：196 份
    # Kolmogorov config 全寫 true，舊預設是**沒有任何實驗在用的值**。
    # VanillaDeepONetOperator / StandardPINNOperator 本來就預設 True——
    # 三個 operator 裡只有這個是 False。實測 6 個省略此鍵的建構點全在
    # tests/（無 production），開啟後那 14 個測試全數通過。
    use_temporal_anchor: bool = True
    T_total: float = 5.0
    temporal_anchor_harmonics: int = 2
    num_token_attention_layers: int = 1
    token_attention_heads: int = 4
    num_query_mlp_layers: int = 0
    query_mlp_hidden_dim: int = 256
    output_head_gain: float = 1.0
    operator_rank: int = 64
    fourier_embed_dim: int = 128
    decoder_attention_heads: int = 1
    # 方向1: opt-in band-targeted mid-band embedding（空 tuple → 關 → bit-identical）
    mid_band_wavenumbers: tuple = ()
    mid_band_embed_dim: int = 128
    mid_band_init_sigma: float = 2.0
    # 方向1 exp_508: opt-in trainable-frequency embedding（False → 關 → bit-identical）
    use_trainable_fourier: bool = False
    trainable_fourier_dim: int = 128
    trainable_fourier_init_scale: float = 8.0
    trainable_fourier_lowpass_kc: float = 0.0
    # ── 進階開關（對齊 pi_con/operator.py）──
    cfc_log_tau_min: float = -1.0
    cfc_log_tau_max: float = 1.0
    # liquid time-constant（LiquidNN 精髓）：對齊 pi_con/operator.py cfc_input_dependent_tau
    # 2026-09-03 起預設 True，與 config.py schema 一致（見下方 use_rwf 的同批說明）。
    # scale 同時由 2.0 改為 0.5：2.0 是 τ 動態範圍暴衝 ~400× 的已知崩壞值
    # （config.py:114 一帶），打開 tau 而留著 2.0 等於預設踩進去。
    cfc_input_dependent_tau: bool = True
    cfc_tau_mod_scale: float = 0.5
    use_modified_mlp: bool = False
    use_locality_decay: bool = False
    disable_cross_attention: bool = False
    fusion_temperature_init: float = 0.0  # 0 → 1/sqrt(rank) default
    # Wang 2021 跨分支 mMLP（見 DeepONetCfCDecoder 同名欄位）
    mmlp_branch_u: bool = False
    mmlp_gate_branch: bool = False
    # 幾何泛化（cross-attention relpos bias）：預設對齊 Kolmogorov 現行行為
    relpos_bias_mode: str = "radial"      # "radial"(各向同性) | "vector"(各向異性)
    # 見 DeepONetCfCDecoder 同名欄位：預設維持既有的 no-op 行為
    relpos_bias_norm: str = "layernorm"    # "layernorm"(預設) | "none"(修好的對照臂)
    relpos_bias_zero_init: bool = False
    periodic_domain: bool = True          # False = 有界/幾何域，關閉 rel 週期 wrap
    # #3 SDF features（解析圓形 body distance → trunk + attention bias）
    use_sdf_features: bool = False
    body_center_x: float = 0.5
    body_center_y: float = 0.5
    body_radius: float = 0.0
    attention_kind: str = "scalar"        # "scalar"(預設) | "vector"(PTv2 grouped vector attn)
    # ── ForcingPrior 開關（對齊 pi_con/operator.py:create_picon_model）──
    learn_forcing_A: bool = False
    learn_forcing_k_f: bool = False
    forcing_A_init: float = 0.1
    forcing_k_f_init: float = 2.0
    forcing_k_f_min: float = 1.0
    forcing_k_f_max: float = 8.0
    # 線性（Ekman）阻尼係數。PDE residual 的 -alpha*u 項，非 forcing 而是 sink。
    # 固定值不可學：它是生成參考資料時的已知設定，不是待辨識量。
    # 放在 model 而非 config data 段的理由：data 段的 kolmogorov_A/k_f 是 inert 死鍵
    # （config.py:294），真正被讀的物理常數住在 model；且 model dataclass 欄位會自動
    # 進 model_fingerprint，可擋住「阻尼資料訓的 ckpt 被無阻尼 residual 拿去 eval」。
    drag_alpha: float = 0.0
    # ── Wave 4 perf: remat (gradient checkpointing) on decoder ──
    # Why: 二階 autograd 對 cross-attention + MLP path 會建大 activation graph；
    #      remat 用「forward 重算」換 memory peak，特別重要對 GPU fp64。
    #      M3 CPU 已驗 memory peak ↓ 1.77×（A+B）；remat 進一步壓低。
    #      Trade-off: jit compile 略慢、step wall 略升（重算 forward）。
    use_decoder_remat: bool = False
    # PyTorch nn.Linear 風格 kernel init（Kaiming-uniform）opt-in。
    # 預設 False → 完全沿用 Flax lecun_normal（保護 Kolmogorov 主結果）。
    # True → 未顯式指定 kernel_init 的 Dense 改用 torch_linear_kernel_init（cylinder over-energy 修復）。
    torch_style_init: bool = False
    # ── RWF（Random Weight Factorization, Wang 2023）──
    # 僅作用於 decoder trunk 座標 MLP。2026-09-03 起預設 True，與 config.py schema
    # 對齊（Kolmogorov n=5 判別：-0.114 pp, t=-3.92）。⚠️ 這批證據全在 Kolmogorov
    # 量的，cylinder 從未驗過這三個鍵——本次是使用者明示要求兩案一致，不是證據外推。
    # cylinder 已於同日在 assembly.CFG 顯式寫死三鍵，不再吃這裡的預設。
    use_rwf: bool = True
    rwf_mean: float = 1.0
    rwf_stddev: float = 0.1

    def setup(self):
        self.spatial_encoder = SpatialSetEncoder(
            d_model=self.d_model,
            num_layers=self.num_spatial_encoder_layers,
            sensor_value_dim=self.sensor_value_dim,
            domain_length=self.domain_length,
            fourier_embed_dim=self.fourier_embed_dim,
            periodic_domain=self.periodic_domain,
            torch_style_init=self.torch_style_init,
        )
        self.temporal_encoder = TemporalCfCEncoder(
            d_model=self.d_model,
            num_layers=self.num_temporal_cfc_layers,
            num_token_attention_layers=self.num_token_attention_layers,
            token_attention_heads=self.token_attention_heads,
            cfc_log_tau_min=self.cfc_log_tau_min,
            cfc_log_tau_max=self.cfc_log_tau_max,
            cfc_input_dependent_tau=self.cfc_input_dependent_tau,
            cfc_tau_mod_scale=self.cfc_tau_mod_scale,
            torch_style_init=self.torch_style_init,
        )
        # decoder class：若啟 use_decoder_remat，套 nn.remat lifted transform
        # 對二階 autograd 用 forward-recompute 換 activation memory（peak ↓ 30-40%）
        decoder_cls = (
            nn.remat(DeepONetCfCDecoder)
            if self.use_decoder_remat else
            DeepONetCfCDecoder
        )
        self.query_decoder = decoder_cls(
            d_model=self.d_model,
            d_time=self.d_time,
            domain_length=self.domain_length,
            use_temporal_anchor=self.use_temporal_anchor,
            T_total=self.T_total,
            temporal_anchor_harmonics=self.temporal_anchor_harmonics,
            num_query_mlp_layers=self.num_query_mlp_layers,
            query_mlp_hidden_dim=self.query_mlp_hidden_dim,
            output_head_gain=self.output_head_gain,
            operator_rank=self.operator_rank,
            fourier_embed_dim=self.fourier_embed_dim,
            decoder_attention_heads=self.decoder_attention_heads,
            mid_band_wavenumbers=self.mid_band_wavenumbers,
            mid_band_embed_dim=self.mid_band_embed_dim,
            mid_band_init_sigma=self.mid_band_init_sigma,
            use_trainable_fourier=self.use_trainable_fourier,
            trainable_fourier_dim=self.trainable_fourier_dim,
            trainable_fourier_init_scale=self.trainable_fourier_init_scale,
            trainable_fourier_lowpass_kc=self.trainable_fourier_lowpass_kc,
            use_modified_mlp=self.use_modified_mlp,
            use_locality_decay=self.use_locality_decay,
            disable_cross_attention=self.disable_cross_attention,
            fusion_temperature_init=self.fusion_temperature_init,
            relpos_bias_mode=self.relpos_bias_mode,
            relpos_bias_norm=self.relpos_bias_norm,
            relpos_bias_zero_init=self.relpos_bias_zero_init,
            periodic_domain=self.periodic_domain,
            use_sdf_features=self.use_sdf_features,
            mmlp_branch_u=self.mmlp_branch_u,
            mmlp_gate_branch=self.mmlp_gate_branch,
            body_center_x=self.body_center_x,
            body_center_y=self.body_center_y,
            body_radius=self.body_radius,
            attention_kind=self.attention_kind,
            torch_style_init=self.torch_style_init,
            use_rwf=self.use_rwf,
            rwf_mean=self.rwf_mean,
            rwf_stddev=self.rwf_stddev,
        )
        # ForcingPrior submodule（params 自動納入 model state_dict）
        self.forcing = ForcingPrior(
            A_init=self.forcing_A_init,
            k_f_init=self.forcing_k_f_init,
            learn_A=self.learn_forcing_A,
            learn_k_f=self.learn_forcing_k_f,
            k_f_min=self.forcing_k_f_min,
            k_f_max=self.forcing_k_f_max,
        )

    def get_forcing(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """返回當前 (A, k_f)。用 model.apply(params, method=LiquidOperator.get_forcing) 呼叫。"""
        return self.forcing()

    def encode(
        self,
        sensor_vals: jnp.ndarray,
        sensor_pos: jnp.ndarray,
        re_norm: float,
        sensor_time: jnp.ndarray,
    ) -> jnp.ndarray:
        pos_enc = self.spatial_encoder.encode_pos(sensor_pos)
        spatial_states = self.spatial_encoder(sensor_vals, pos_enc)  # [T, K, d_model]
        h_states = self.temporal_encoder(spatial_states, re_norm, sensor_time)
        return h_states

    def decode_query(
        self,
        xy: jnp.ndarray,
        t_q: jnp.ndarray,
        h_states: jnp.ndarray,
        sensor_time: jnp.ndarray,
        sensor_pos: jnp.ndarray,
    ) -> jnp.ndarray:
        """Decoder-only forward (encode 已預算)。

        Why perf: physics PDE residual 對 (x, y, t) 求 grad 時需要重複 forward；
                  若用 __call__ 每次都重跑 encode (lax.scan over T)，浪費 ~98% flop。
                  此 method 接 pre-computed h_states，grad chain 只走 decoder，
                  與 encode 的 grad 透過外層 loss_fn 的單一 encode call 彙整。
        """
        return self.query_decoder(xy, t_q, h_states, sensor_time, sensor_pos)

    def __call__(
        self,
        sensor_vals: jnp.ndarray,
        sensor_pos: jnp.ndarray,
        re_norm: float,
        sensor_time: jnp.ndarray,
        xy: jnp.ndarray,
        t_q: jnp.ndarray,
    ) -> jnp.ndarray:
        h_states = self.encode(sensor_vals, sensor_pos, re_norm, sensor_time)
        # Trigger forcing submodule init（discard 結果）。
        # Why: Flax setup 內宣告的 submodule 必須被 __call__ 一次才會註冊 params。
        #      此 dummy call 在 init pass 時建立 forcing params；jit dead code elimination 會去掉。
        _ = self.forcing()
        return self.query_decoder(xy, t_q, h_states, sensor_time, sensor_pos)


# ─────────────────────────────────────────────────────────────────────────────
# B0: VanillaDeepONetOperator — 純 DeepONet baseline (ablation)
# ─────────────────────────────────────────────────────────────────────────────

class VanillaDeepONetOperator(nn.Module):
    """B0 architectural ablation: pure DeepONet。對齊 pi_con/vanilla_deeponet.py。

    Branch: sensor_at_t_q [N_q, K*C] → MLP → branch_basis [N_q, 3, rank]
    Trunk:  (x, y, t, c) → 同 LiquidOperator decoder 的 Fourier encoding → MLP → trunk_basis [N_q, 3, rank]
    Readout: u_c(x,y,t) = trunk_basis[c] · branch_basis[c]（純 inner product，無 cross-attention）

    與 LiquidOperator 對比 (B3): 移除 CfC（temporal encoder）+ cross-attention，
    保留同樣的 Fourier feature encoding for fair comparison。
    """
    K_sensors: int                       # 編譯期常數，因為要 reshape [N, K*C]
    sensor_value_dim: int = 2
    d_time: int = 8
    domain_length: float = 1.0
    use_temporal_anchor: bool = True
    T_total: float = 5.0
    temporal_anchor_harmonics: int = 2
    num_branch_layers: int = 3
    num_trunk_layers: int = 3
    hidden_dim: int = 64
    operator_rank: int = 64
    output_head_gain: float = 1.0
    fourier_embed_dim: int = 128
    # ForcingPrior 開關（與 LiquidOperator 對齊）
    learn_forcing_A: bool = False
    learn_forcing_k_f: bool = False
    forcing_A_init: float = 0.1
    forcing_k_f_init: float = 2.0
    forcing_k_f_min: float = 1.0
    forcing_k_f_max: float = 8.0
    # 線性（Ekman）阻尼係數。PDE residual 的 -alpha*u 項，非 forcing 而是 sink。
    # 固定值不可學：它是生成參考資料時的已知設定，不是待辨識量。
    # 放在 model 而非 config data 段的理由：data 段的 kolmogorov_A/k_f 是 inert 死鍵
    # （config.py:294），真正被讀的物理常數住在 model；且 model dataclass 欄位會自動
    # 進 model_fingerprint，可擋住「阻尼資料訓的 ckpt 被無阻尼 residual 拿去 eval」。
    drag_alpha: float = 0.0

    def setup(self):
        # spatial_emb 與 LiquidOperator decoder 對稱
        self.spatial_emb = LearnableFourierEmb(self.fourier_embed_dim)
        self.spatial_dim_attr = self.fourier_embed_dim
        self.temporal_dim_attr = (
            2 * self.temporal_anchor_harmonics if self.use_temporal_anchor else 0
        )
        # Branch path
        self.branch_in = nn.Dense(self.hidden_dim, name='branch_in')
        self.branch_blocks = [
            ResidualMLPBlock(d_model=self.hidden_dim, hidden_dim=self.hidden_dim, name=f'branch_block_{i}')
            for i in range(self.num_branch_layers)
        ]
        # Output head with xavier_normal * gain
        def _xavier_normal_with_gain(key, shape, dtype=jnp.float32):
            base = nn.initializers.xavier_normal()(key, shape, dtype)
            return base * self.output_head_gain
        self.branch_out = nn.Dense(
            3 * self.operator_rank, name='branch_out',
            kernel_init=_xavier_normal_with_gain,
            bias_init=nn.initializers.zeros,
        )
        # Trunk path
        self.time_proj = nn.Dense(self.d_time, name='time_proj')
        self.component_emb = self.param(
            'component_emb',
            lambda key, shape: jax.random.normal(key, shape) * 0.1,
            (3, 8),
        )
        self.trunk_in = nn.Dense(self.hidden_dim, name='trunk_in')
        self.trunk_blocks = [
            ResidualMLPBlock(d_model=self.hidden_dim, hidden_dim=self.hidden_dim, name=f'trunk_block_{i}')
            for i in range(self.num_trunk_layers)
        ]
        self.trunk_out = nn.Dense(
            3 * self.operator_rank, name='trunk_out',
            kernel_init=_xavier_normal_with_gain,
            bias_init=nn.initializers.zeros,
        )
        # Fusion + per-component affine (與 LiquidOperator 一致)
        temp_init = 1.0 / math.sqrt(self.operator_rank)
        self.log_fusion_temperature = self.param(
            'log_fusion_temperature',
            lambda key: jnp.array([math.log(temp_init)], dtype=jnp.float32),
        )
        self.component_scale = self.param(
            'component_scale',
            lambda key: jnp.ones((3,), dtype=jnp.float32),
        )
        self.component_bias = self.param(
            'component_bias',
            lambda key: jnp.zeros((3,), dtype=jnp.float32),
        )
        # ForcingPrior（與 LiquidOperator 對齊）
        self.forcing = ForcingPrior(
            A_init=self.forcing_A_init,
            k_f_init=self.forcing_k_f_init,
            learn_A=self.learn_forcing_A,
            learn_k_f=self.learn_forcing_k_f,
            k_f_min=self.forcing_k_f_min,
            k_f_max=self.forcing_k_f_max,
        )

    def get_forcing(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.forcing()

    def __call__(
        self,
        sensor_vals: jnp.ndarray,    # [T_sensor, K, C]
        sensor_pos: jnp.ndarray,      # [K, 2]   (B0 不用，僅接口相容)
        re_norm: float,                # (B0 不用)
        sensor_time: jnp.ndarray,     # [T_sensor]
        xy: jnp.ndarray,               # [N_q, 2]
        t_q: jnp.ndarray,              # [N_q]
    ) -> jnp.ndarray:
        """Returns [N_q, 3] (u, v, p)."""
        N = xy.shape[0]
        # Branch: sensor at nearest sensor_time ≤ t_q
        idx = jnp.searchsorted(sensor_time, t_q, side='right') - 1
        idx = jnp.clip(idx, 0, sensor_vals.shape[0] - 1)
        sensor_at_tq = sensor_vals[idx]                      # [N, K, C]
        branch_input = sensor_at_tq.reshape(N, -1)            # [N, K*C]
        x = nn.silu(self.branch_in(branch_input))
        for block in self.branch_blocks:
            x = block(x)
        branch_basis = self.branch_out(x).reshape(N, 3, self.operator_rank)  # [N, 3, rank]

        # Trunk: 對 c=0/1/2 批次化
        pos_enc = self.spatial_emb(xy, self.domain_length)
        time_e = self.time_proj(t_q[:, None])
        base_inputs = [pos_enc]
        if self.use_temporal_anchor:
            base_inputs.append(
                temporal_phase_anchor(t_q[:, None], self.T_total, self.temporal_anchor_harmonics)
            )
        base_inputs.append(time_e)
        base_feat = jnp.concatenate(base_inputs, axis=-1)     # [N, spatial+temporal+d_time]
        base_feat_3 = jnp.broadcast_to(base_feat[None], (3,) + base_feat.shape)
        emb_c_3 = jnp.broadcast_to(self.component_emb[:, None, :], (3, N, 8))
        trunk_in_3 = jnp.concatenate([base_feat_3, emb_c_3], axis=-1).reshape(3 * N, -1)
        x = nn.silu(self.trunk_in(trunk_in_3))
        for block in self.trunk_blocks:
            x = block(x)
        trunk_basis = self.trunk_out(x).reshape(3 * N, 3, self.operator_rank)

        # Readout: pure inner product per component
        c_flat = jnp.repeat(jnp.arange(3), N)                 # [3N]
        trunk_sel = jnp.take_along_axis(
            trunk_basis, c_flat[:, None, None].repeat(self.operator_rank, axis=2), axis=1,
        ).squeeze(1)                                          # [3N, rank]
        # branch_basis [N, 3, rank] → tile to [3N, 3, rank] → gather → [3N, rank]
        branch_basis_3 = _repeat3(branch_basis)
        branch_sel = jnp.take_along_axis(
            branch_basis_3, c_flat[:, None, None].repeat(self.operator_rank, axis=2), axis=1,
        ).squeeze(1)
        fusion_temperature = jnp.exp(self.log_fusion_temperature).astype(trunk_sel.dtype)
        out = jnp.sum(trunk_sel * branch_sel, axis=1) * fusion_temperature
        out = out * self.component_scale[c_flat] + self.component_bias[c_flat]
        uvp = out.reshape(3, N).T

        # Trigger forcing init (與 LiquidOperator 一致)
        _ = self.forcing()
        return uvp


# ─────────────────────────────────────────────────────────────────────────────
# B2: StandardPINNOperator — pure (x, y, t) → MLP → (u, v, p) baseline
# ─────────────────────────────────────────────────────────────────────────────

class StandardPINNOperator(nn.Module):
    """B2 architectural ablation: standard single-instance PINN (Wang 2021 style)。
    對齊 pi_con/standard_pinn.py。

    Architecture: (x, y, t) → Fourier + temporal_anchor + time_proj → concat
                  → Linear → ResidualMLPBlock × N → Linear → (u, v, p)
    NO operator framework (sensor 不進 model input，僅在 loss 出現)
    NO temporal recurrence (no CfC)
    NO cross-attention
    NO c conditioning（output 一次給 3 component）

    與 LiquidOperator/B0 接口相容（接受 sensor_vals 等但 ignored）。
    """
    d_time: int = 8
    domain_length: float = 1.0
    use_temporal_anchor: bool = True
    T_total: float = 5.0
    temporal_anchor_harmonics: int = 2
    num_layers: int = 6
    hidden_dim: int = 64       # POC mini: 64 (full 512 留 GPU)
    output_head_gain: float = 1.0
    fourier_embed_dim: int = 128
    # ForcingPrior 開關（與其他 operator 對齊）
    learn_forcing_A: bool = False
    learn_forcing_k_f: bool = False
    forcing_A_init: float = 0.1
    forcing_k_f_init: float = 2.0
    forcing_k_f_min: float = 1.0
    forcing_k_f_max: float = 8.0
    # 線性（Ekman）阻尼係數。PDE residual 的 -alpha*u 項，非 forcing 而是 sink。
    # 固定值不可學：它是生成參考資料時的已知設定，不是待辨識量。
    # 放在 model 而非 config data 段的理由：data 段的 kolmogorov_A/k_f 是 inert 死鍵
    # （config.py:294），真正被讀的物理常數住在 model；且 model dataclass 欄位會自動
    # 進 model_fingerprint，可擋住「阻尼資料訓的 ckpt 被無阻尼 residual 拿去 eval」。
    drag_alpha: float = 0.0

    def setup(self):
        self.spatial_emb = LearnableFourierEmb(self.fourier_embed_dim)
        self.spatial_dim_attr = self.fourier_embed_dim
        self.temporal_dim_attr = (
            2 * self.temporal_anchor_harmonics if self.use_temporal_anchor else 0
        )
        self.time_proj = nn.Dense(self.d_time, name='time_proj')
        self.input_proj = nn.Dense(self.hidden_dim, name='input_proj')
        self.blocks = [
            ResidualMLPBlock(d_model=self.hidden_dim, hidden_dim=self.hidden_dim, name=f'block_{i}')
            for i in range(self.num_layers)
        ]
        # Output: 直接 (u, v, p) 三輸出
        def _xavier_normal_with_gain(key, shape, dtype=jnp.float32):
            base = nn.initializers.xavier_normal()(key, shape, dtype)
            return base * self.output_head_gain
        self.output_head = nn.Dense(
            3, name='output_head',
            kernel_init=_xavier_normal_with_gain,
            bias_init=nn.initializers.zeros,
        )
        # ForcingPrior（與其他 operator 對齊）
        self.forcing = ForcingPrior(
            A_init=self.forcing_A_init,
            k_f_init=self.forcing_k_f_init,
            learn_A=self.learn_forcing_A,
            learn_k_f=self.learn_forcing_k_f,
            k_f_min=self.forcing_k_f_min,
            k_f_max=self.forcing_k_f_max,
        )

    def get_forcing(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.forcing()

    def __call__(
        self,
        sensor_vals: jnp.ndarray,   # ignored
        sensor_pos: jnp.ndarray,     # ignored
        re_norm: float,               # ignored
        sensor_time: jnp.ndarray,    # ignored
        xy: jnp.ndarray,              # [N, 2]
        t_q: jnp.ndarray,             # [N]
    ) -> jnp.ndarray:
        """Returns [N, 3] = (u, v, p)。Sensor inputs ignored（Standard PINN convention）。"""
        pos_enc = self.spatial_emb(xy, self.domain_length)
        time_e = self.time_proj(t_q[:, None])
        parts = [pos_enc]
        if self.use_temporal_anchor:
            parts.append(
                temporal_phase_anchor(t_q[:, None], self.T_total, self.temporal_anchor_harmonics)
            )
        parts.append(time_e)
        x = jnp.concatenate(parts, axis=-1)             # [N, spatial+temporal+d_time]
        x = nn.silu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        uvp = self.output_head(x)                        # [N, 3]
        # Trigger forcing init
        _ = self.forcing()
        return uvp

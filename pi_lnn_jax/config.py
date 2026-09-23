"""Shared TOML validation and effective-configuration resolution primitives.

對齊 pi_con/config.py:DEFAULT_LNN_ARGS 與 validate_config，
但只保留 POC 已實作的 model + training keys。

What:
  - load_config(path): validate the historical TOML surface.
  - resolve_config(policy, argv): merge ordered layers and return a typed
    case-specific effective configuration plus provenance.

Why fail-fast:
  - unknown key → warn (允許 pi-lnn 既有 TOML 含 POC 未實作 keys)
  - known key with wrong type → raise
  - out-of-range value → raise
"""
from __future__ import annotations

import tomllib
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, Mapping, Protocol, Sequence, TypeVar


# ─────────────────────────────────────────────────────────────────────────────
# Schema: 對應每個 key 的 (type, default, validator) 三元組
# ─────────────────────────────────────────────────────────────────────────────

def _positive(name: str):
    def _v(x):
        if x <= 0:
            raise ValueError(f"{name} 必須 > 0，收到 {x}")
    return _v


def _positive_even(name: str):
    """正偶數（fourier_embed_dim 用：LearnableFourierEmb/FourierEmbs 需 embed_dim//2）。"""
    def _v(x):
        if x <= 0 or x % 2 != 0:
            raise ValueError(f"{name} 必須為正偶數，收到 {x}")
    return _v


def _greater_than_one(name: str):
    """嚴格 > 1（re_norm_scale 用：re_norm=log(Re)/log(scale)，scale≤1 會使 log≤0 → inf/負）。"""
    def _v(x):
        if x <= 1.0:
            raise ValueError(f"{name} 必須 > 1（log(scale) 須為正），收到 {x}")
    return _v


def _non_negative(name: str):
    def _v(x):
        if x < 0:
            raise ValueError(f"{name} 必須 >= 0，收到 {x}")
    return _v


def _in_range(name: str, lo: float, hi: float):
    def _v(x):
        if not (lo <= x <= hi):
            raise ValueError(f"{name} 必須在 [{lo}, {hi}]，收到 {x}")
    return _v


def _in_range_half_open(name: str, lo: float, hi: float):
    """半開區間 [lo, hi)：EMA 係數 beta ∈ [0, 1) 使用。"""
    def _v(x):
        if not (lo <= x < hi):
            raise ValueError(f"{name} 必須在 [{lo}, {hi})，收到 {x}")
    return _v


def _in_range_open(name: str, lo: float, hi: float):
    """開區間 (lo, hi)：兩端皆排除（如 t_early_threshold ∈ (0,1)）。"""
    def _v(x):
        if not (lo < x < hi):
            raise ValueError(f"{name} 必須在 ({lo}, {hi})，收到 {x}")
    return _v


def _one_of(name: str, choices: tuple):
    """枚舉驗證：value 必須屬於 choices 集合。"""
    def _v(x):
        if x not in choices:
            raise ValueError(f"{name} 必須為 {choices} 之一，收到 {x!r}")
    return _v


# Model keys → LiquidOperator constructor args
MODEL_SCHEMA = {
    # 必要
    "d_model":                       (int,   64,   _positive("d_model")),
    "d_time":                        (int,   8,    _positive("d_time")),
    "num_spatial_encoder_layers":        (int,   1,    _non_negative("num_spatial_encoder_layers")),
    "num_temporal_cfc_layers":       (int,   1,    _non_negative("num_temporal_cfc_layers")),
    # 預設可選
    "num_token_attention_layers":    (int,   1,    _non_negative("num_token_attention_layers")),
    "token_attention_heads":         (int,   4,    _positive("token_attention_heads")),
    "num_query_mlp_layers":          (int,   0,    _non_negative("num_query_mlp_layers")),
    "query_mlp_hidden_dim":          (int,   64,   _positive("query_mlp_hidden_dim")),
    "operator_rank":                 (int,   64,   _positive("operator_rank")),
    "output_head_gain":              (float, 1.0,  _positive("output_head_gain")),
    "decoder_attention_heads":       (int,   1,    _positive("decoder_attention_heads")),
    "use_temporal_anchor":           (bool,  True,  None),   # 2026-08-03: False→True
    # ↑ 全部 196 份 Kolmogorov config 都寫 true；舊預設 False 是**沒有任何實驗在用的值**，
    #   漏寫這個鍵就會靜靜訓練出與所有既有實驗不同的架構（不 crash，只是換一組數字）。
    #   實測改動前後 205 份 config 的解析值**全部不變**：9 個省略者全是 cylinder(4) /
    #   controlled_cylinder(5)，而 cylinder 走 assembly.CFG 顯式建構（該處寫死 True），
    #   吃不到本 schema 預設。tests/test_schema_default_traps.py 釘住這個爆炸半徑。
    "T_total":                       (float, 5.0,  _positive("T_total")),
    "temporal_anchor_harmonics":     (int,   2,    _positive("temporal_anchor_harmonics")),
    "domain_length":                 (float, 1.0,  _positive("domain_length")),
    # 可學習 Fourier 投影維度（固定啟用；須為正偶數）。取代舊確定性週期編碼路徑。
    "fourier_embed_dim":             (int,   128,  _positive_even("fourier_embed_dim")),
    # 方向1 opt-in band-targeted mid-band embedding（預設 () = 關 = bit-identical；
    # 設 list 波數即開，build_model 轉 tuple）。embed_dim 須正偶數（BandFourierEmb 需 //2）。
    "mid_band_wavenumbers":          (list,  (),   None),
    "mid_band_embed_dim":            (int,   128,  _positive_even("mid_band_embed_dim")),
    "mid_band_init_sigma":           (float, 2.0,  _positive("mid_band_init_sigma")),
    # 方向1 exp_508 opt-in trainable-frequency embedding（預設 False = 關 = bit-identical）。
    # dim 須正偶數（cos/sin 對半分）；init_scale 是 |B| 初始頻率尺度（Rayleigh 中位數 ≈ 1.177×）。
    "use_trainable_fourier":         (bool,  False, None),
    "trainable_fourier_dim":         (int,   128,  _positive_even("trainable_fourier_dim")),
    "trainable_fourier_init_scale":  (float, 8.0,  _positive("trainable_fourier_init_scale")),
    # 低通振幅權重 w=1/(1+(|B|/kc)^2)；0 = 關。權重對 B 走 stop_gradient（見 models.py）。
    "trainable_fourier_lowpass_kc":  (float, 0.0,  _non_negative("trainable_fourier_lowpass_kc")),
    # 進階開關
    "cfc_log_tau_min":               (float, -1.0, None),
    "cfc_log_tau_max":               (float, 1.0,  None),
    # liquid time-constant（LiquidNN 精髓）。實測 exp_521-524：input-varying τ 在 tau_mod_scale=0.5
    # 略優 baseline，但 2.0（舊預設）讓 τ 動態範圍暴衝 ~400× 全面崩壞 +27~58%。故預設 τ on + scale 0.5。
    "cfc_input_dependent_tau":       (bool,  True,  None),
    "cfc_tau_mod_scale":             (float, 0.5,  _positive("cfc_tau_mod_scale")),
    # 幾何泛化（cross-attention relpos bias + SDF features）
    "relpos_bias_mode":              (str,   "radial", _one_of("relpos_bias_mode", ("radial", "vector"))),
    "periodic_domain":              (bool,  True, None),
    "use_sdf_features":              (bool,  False, None),
    "body_center_x":                 (float, 0.5,  None),
    "body_center_y":                 (float, 0.5,  None),
    "body_radius":                   (float, 0.0,  _non_negative("body_radius")),
    "attention_kind":                (str,   "scalar", _one_of("attention_kind", ("scalar", "vector"))),
    # relpos bias 的正規化：預設 "layernorm" 保既有行為（radial+no-SDF 下該 bias 是 no-op）。
    # "none" 讓它真的生效——比較「方向資訊有沒有用」時必須以 "none" 為基準，否則量到的是
    # 「把壞掉的 bias 修好」。見 knowledge/experiments/cylinder-advective-attention-negative-2026-07-29.md
    "relpos_bias_norm":              (str,   "layernorm", _one_of("relpos_bias_norm", ("layernorm", "none"))),
    "relpos_bias_zero_init":         (bool,  False, None),
    # RWF（Random Weight Factorization, Wang 2023）：僅作用於 decoder trunk 座標 MLP
    # 2026-08-29 起預設 True：n=5 判別顯示 rwf on 在主指標上顯著較好
    # （-0.114 pp, t=-3.92，五 seed 範圍不重疊；knowledge/experiments/
    # kolmogorov-rwf-multiseed-2026-08-29.md）。舊註記「exp_521 rwf-only 中性偏微負、
    # 無淨益」出自單 seed × 錯誤 eval stride 的疊加（TD-4），已推翻。
    "use_rwf":                       (bool,  True,  None),
    "rwf_mean":                      (float, 1.0,  None),
    "rwf_stddev":                    (float, 0.1,  _positive("rwf_stddev")),
    "use_modified_mlp":              (bool,  False, None),
    # Wang 2021 modified DeepONet 的跨分支 U/V gating（arXiv 2110.01654 §3.4）。
    # 皆需 use_modified_mlp=true，否則模型建構時 fail fast。
    "mmlp_branch_u":                 (bool,  False, None),
    "mmlp_gate_branch":              (bool,  False, None),
    "use_locality_decay":            (bool,  False, None),
    "disable_cross_attention":       (bool,  False, None),
    "fusion_temperature_init":       (float, 0.0,  _non_negative("fusion_temperature_init")),
    # ForcingPrior
    "learn_forcing_A":               (bool,  False, None),
    "learn_forcing_k_f":             (bool,  False, None),
    "forcing_A_init":                (float, 0.1,  _positive("forcing_A_init")),
    # 線性阻尼 alpha（PDE residual 的 -alpha*u）。0 = 既有無阻尼行為。
    "drag_alpha":                    (float, 0.0,  _non_negative("drag_alpha")),
    "forcing_k_f_init":              (float, 2.0,  _positive("forcing_k_f_init")),
    "forcing_k_f_min":               (float, 1.0,  _positive("forcing_k_f_min")),
    "forcing_k_f_max":               (float, 8.0,  _positive("forcing_k_f_max")),
    # sensor_value_dim 從 observed_sensor_channels 長度推
    "sensor_value_dim":              (int,   2,    _positive("sensor_value_dim")),
}


# Training keys → train script args
TRAIN_SCHEMA = {
    "iterations":                    (int,   5000, _positive("iterations")),
    "learning_rate":                 (float, 3e-3, _positive("learning_rate")),
    "seed":                          (int,   42,   None),
    "num_physics_points":            (int,   64,   _positive("num_physics_points")),
    # physics collocation 的時間上界。0.0（預設）= 沿用 sensor 資料時窗上界，
    # 即既有行為（collocation 只落在有資料的區間）。
    # >0 = 明確指定，讓 collocation 延伸到資料時窗之外：該區間只有 PDE residual
    # 約束、沒有 data loss，即「physics-only 時間外推」。必須 ≥ 資料時窗上界，
    # 否則 build_context fail-fast（縮小 physics 時窗請用 time marching，不是這個鍵）。
    # 注意：外推區的 query 會讓 decoder 的 dt_to_query 遠離訓練分佈，且
    # temporal_phase_anchor 的週期是 model.T_total —— 用本鍵時 T_total 必須
    # 一併設成外推後的總時窗，否則 t 與 t−T_total 在相位錨上完全混疊。
    "physics_t_max":                 (float, 0.0,  _non_negative("physics_t_max")),
    # 訓練時隨機截斷 decoder 看得到的 sensor 序列尾段：每步抽 frac ~ U(此值, 1.0)，
    # decoder 的 branch 狀態只能用到第 round(frac·(T−1)) 幀，但 data loss 仍監督**全部**
    # 幀。0.0（預設）= 關 = 既有行為。
    # Why: 既有訓練裡被監督的 query 其 `dt_to_query` 恆為 0（query 時刻就是 sensor 幀），
    # 大 dt 只出現在 collocation 上、而 PDE residual 選不出軌跡 —— 模型因此從未被告知
    # 「隔一段時間之後場該長什麼樣」。截斷讓截點之後的幀變成 dt>0 且**有真值**的監督點。
    "sensor_cut_min_frac":           (float, 0.0,  _in_range("sensor_cut_min_frac", 0.0, 1.0)),
    # Branch 自迴歸：用模型自己的預測把 sensor 序列延伸到 physics_t_max，
    # 讓外推段的 branch 重新擁有隨時間演化的狀態（否則 h_states 凍結在最後一幀）。
    # dt = pseudo 幀間距（0.0 = 關）；rounds = 自迴歸輪數（每輪重新 encode，
    # 後段 pseudo 幀基於前段；1 = 全部用凍結狀態一次解完）。pseudo 觀測走 stop_gradient。
    "autoreg_pseudo_dt":             (float, 0.0,  _non_negative("autoreg_pseudo_dt")),
    "autoreg_rounds":                (int,   1,    _positive("autoreg_rounds")),
    # gradient accumulation：把 collocation + sensor-query 切 M 塊、lax.scan 逐塊累積梯度，
    # 峰值記憶體 ∝ 1/M（大 K cross-attention OOM 解）。預設 1=全量。n_collo/n_sensor_q 須可整除 M。
    # 注意：AL 的 C² 在 M>1 下是 per-chunk 近似（chunk-mean 方差），非精確全量。
    "grad_accum_chunks":             (int,   1,    _positive("grad_accum_chunks")),
    # RAR（residual-adaptive refinement）：每 rar_freq 步改用「pool 中殘差最大的點」
    # 當 collocation。原本只有 CLI 旗標，於是開了 RAR 的 run 在 config 上看不出來——
    # 那正是本 repo 反覆踩到的 provenance 洞。預設與 pipeline policy 的相容預設逐字相同
    # （0 / 500 / 512），故本次新增不改變任何既有 run 的行為。
    # ⚠️ rar_pool_size 必須 >= round(n_collo × 0.8)，否則 lax.top_k 會拋錯；
    #    預設 512 配主線 n_collo=1024 不相容，由 assert_rar_pool_is_large_enough 擋下。
    "rar_freq":                      (int,   0,    _non_negative("rar_freq")),
    "rar_warmup":                    (int,   500,  _non_negative("rar_warmup")),
    "rar_pool_size":                 (int,   512,  _positive("rar_pool_size")),
    # RAR 每步裡補回來的均勻點比例。這才是「RAR 多激進」的旋鈕——rar_freq 控制的是
    # 工作週期（多少比例的步數用 RAR），本鍵控制每個 RAR 步裡有多少點不是對抗選出的。
    # 原本寫死在 run.py:494；0.2 是寫死時的值，故新增不改變行為。
    "rar_exploration_ratio":         (float, 0.2,  _in_range("rar_exploration_ratio", 0.0, 1.0)),
    # sensor mini-batch：每 step decode 的 sensor query 點數（0 = full T*K）。
    # 對齊 pi-lnn sample_sensor_batch；encode 仍用全量（preserve Wave 4 encode-once）。
    "num_sensor_query_points":       (int,   0,    _non_negative("num_sensor_query_points")),
    "physics_loss_weight":           (float, 0.01, _non_negative("physics_loss_weight")),
    # 2026-08-04：從 POC_NOT_YET_KEYS 移入。它們的接線**早就完整**——
    #   typed 欄位 kolmogorov/config.py:67-68、policy 預設 :247-248、TOML 映射 :318-319——
    #   只是同時掛在「尚未實作」清單裡，於是每次使用都收到一句假警告。
    #   實測：設 physics_loss_warmup_steps=5000 會警告「POC 尚未實作」，
    #   而 config.loss.physics_warmup_steps 確實是 5000。警告主動勸阻了一個能用的功能。
    "physics_loss_warmup_steps":     (int,   0, _non_negative("physics_loss_warmup_steps")),
    "physics_loss_ramp_steps":       (int,   0, _non_negative("physics_loss_ramp_steps")),
    "data_loss_weight":              (float, 1.0,  _non_negative("data_loss_weight")),
    # data loss 的 per-channel 權重（順序 u,v,p）；空 list = 等權（與既有行為一致）。
    "sensor_channel_weights":        (list,  [],   None),
    "poisson_loss_weight":           (float, 0.0,  _non_negative("poisson_loss_weight")),
    "gauge_loss_weight":             (float, 0.0,  _non_negative("gauge_loss_weight")),
    "max_grad_norm":                 (float, 1.0,  _non_negative("max_grad_norm")),
    "checkpoint_period":             (int,   100,  _positive("checkpoint_period")),
    "artifacts_dir":                 (str,   "artifacts", None),
    # GradNorm + AL flags
    "use_gradnorm":                  (bool,  False, None),
    "gradnorm_freq":                 (int,   1000, _positive("gradnorm_freq")),
    "gradnorm_min_weight":           (float, 0.05, _non_negative("gradnorm_min_weight")),
    "gradnorm_max_weight":           (float, 0.0,  _non_negative("gradnorm_max_weight")),
    # GradNorm fine-grained（對齊 pi-lnn）
    # ⚠ KolmogorovPolicy 的相容 fallback 是 [1.0, 0.01, 0.01]（非本值）；
    # 未設此 key 的歷史 config 以該 fallback 訓練。統一屬行為變更，
    # 見 knowledge/codebase/technical-debt.md「config 預設雙源」。
    #
    # ⚠ 0.057 不是中性預設值，是**在目標指標上選出來的常數**。來歷：pi-lnn EXP-070
    # 的靜態 physics_loss_weight，即 ns weight ablation 上 KE 最好的那一格
    # （0.057→6.30% vs 0.100→13.06% vs 0.300→14.57%），且落在該處標記的相變邊緣
    # ns∈[0.057,0.10]。它後來被當成自適應機制的 init 繼承，再以修 crash 的 commit
    # d307151 進本 schema，JAX 側原本無任何記錄。gradnorm_min_weight 預設 0.05 就在
    # 它下方 12%，而 freq=1000 × 20k 步只有 20 次更新（0.9^20=0.122，init 到最後仍佔
    # 最終值 12%）。新 config 沿用此值等於沿用那個選擇——要調請明寫並說明依據。
    # 完整追溯見 knowledge/codebase/model-audit-2026-09.md §8.1。
    "gradnorm_init_weights":         (list,  [1.0, 0.057, 0.057], None),
    "gradnorm_ema_momentum":         (float, 0.9,  _in_range_half_open("gradnorm_ema_momentum", 0.0, 1.0)),
    "al_rho":                        (float, 1.0,  _non_negative("al_rho")),
    "al_lambda_clip":                (float, 10.0, _positive("al_lambda_clip")),
    "al_update_freq":                (int,   100,  _positive("al_update_freq")),
    "al_constraint_mode":            (str,   "mse", _one_of("al_constraint_mode", ("mse", "signed_mean"))),
    # 連續-Re 物理正則（spec 2026-06-13）：每步額外抽 re_norm' 強制 NS 殘差
    "use_continuous_re_physics":     (bool,  False, None),
    "continuous_re_physics_weight":  (float, 1.0,  _non_negative("continuous_re_physics_weight")),
    # AL warmup：前 N 步 lambda 保持 0，只讓 ρ·C²/2 penalty 暖機；N 步後才開始 dual update。
    # 預設 5000：Re=10000 T=20s 實驗顯示 warmup 讓 u/v err 改善 18~28%（vs 無 warmup），
    # 且訓練初期 C 過大若立即更新 lambda 會造成過度約束（v1 λ→0.371, v2 λ→0.176）。
    # ⚠ KolmogorovPolicy 的相容 fallback 是 0（非本值）——60 個未設
    # 此 key 的歷史 config（含 exp_245/ksweep/pv campaign）實際以 al_warmup=0 訓練。
    # 統一預設屬行為變更，見 knowledge/codebase/technical-debt.md「config 預設雙源」。
    "al_warmup_steps":               (int,   5000, _non_negative("al_warmup_steps")),
    # LR schedule（train_kolmogorov.py build_optimizer 使用）
    "lr_warmup_steps":               (int,   2000,    _non_negative("lr_warmup_steps")),
    "lr_decay_steps":                (int,   2000,    _non_negative("lr_decay_steps")),
    "lr_decay_gamma":                (float, 0.9,  _positive("lr_decay_gamma")),
    "min_learning_rate":             (float, 1e-6, _non_negative("min_learning_rate")),
    # SOAP betas（pi-lnn 用 [0.9, 0.999]；SOAP_JAX 預設 0.95/0.95）
    "soap_betas":                    (list,  [],   None),
    # t_early upweighting（前期時間段 data loss 加權，對齊 pi-lnn t_early_weight）
    "t_early_weight":                (float, 10.0,  _positive("t_early_weight")),
    # ⚠ KolmogorovPolicy 的相容 fallback 是 0.05（非本值 0.1）；
    # 見 knowledge/codebase/technical-debt.md「config 預設雙源」。
    "t_early_threshold":             (float, 0.1,  _in_range_open("t_early_threshold", 0.0, 1.0)),
    # time-marching 時間課程（cylinder CEXP-002；t_max 從 start 線性展開到 sensor_time[-1]）
    "time_marching":                 (bool,  False, None),
    "time_marching_start":           (float, 0.5,  _non_negative("time_marching_start")),
    "time_marching_warmup":          (float, 0.3,  _non_negative("time_marching_warmup")),
    # Causal weighting (Wang 2022) — 預設關閉，不改既有 baseline 行為
    "use_causal_weighting":          (bool,  False, None),
    "causal_eps":                    (float, 1.0,  _non_negative("causal_eps")),
    # LRA inter-task weighting removed 2026-08-03; gradnorm is the sole method.
    # ckpt.TrainState.lra_state stays (always None) only for ckpt-format compat — do not re-add lra_* keys.
    # ── Case selector + cylinder wall-case keys (spec 2026-06-10) ──
    # 預設 "kolmogorov" 維持既有行為；"cylinder" 啟用 wall BC + body mask + KE eval。
    "case":                          (str,   "kolmogorov", _one_of("case", ("kolmogorov", "cylinder", "controlled_cylinder"))),
    "bc_weight":                     (float, 0.1,  _non_negative("bc_weight")),
    "bc_body_weight":                (float, 2.0,  _non_negative("bc_body_weight")),
    "bc_n":                          (int,   128,  _positive("bc_n")),
    "use_physics_denormalization":   (bool,  False, None),
    # ── controlled_cylinder case (moving no-slip; marginal 2D physics) ──
    # r=0.148 → physics 是弱/有偏約束；PDE residual weight 依此 scale 打折（預設 0.3）。
    # amp/phase/freq/axis 由 dump 反推寫進 npz，訓練時從 npz 讀，不進 config。
    "physics_weight_scale":          (float, 0.3,  _non_negative("physics_weight_scale")),
    # moving-BC ablation：縮放 npz 內「實測」振幅。1.0=實測 moving；0.0=靜態 body
    # （amp=0 時 OscillatingGeometry 退化回靜態，見 test_amp_zero_reduces_to_static）。
    # 這是明示的 ablation 旋鈕，不是竄改資料。
    "controlled_amp_scale":          (float, 1.0,  _non_negative("controlled_amp_scale")),
}


# Data keys
DATA_SCHEMA = {
    "sensor_jsons":                  (list,  [],   None),
    "sensor_npzs":                   (list,  [],   None),
    "dns_paths":                     (list,  [],   None),
    "re_values":                     (list,  [1000.0], None),
    # Wave 5: per-Re time_strides 對齊不同 Re 的 sensor 採樣率到同一 T
    # 預設 None → 全部用 stride=2（單 Re path 原行為）
    "time_strides":                  (list,  [],   None),
    # re_norm = log(Re)/log(re_norm_scale) 的上界 Re_max。預設 1e4 保持單-Re 既有
    # 行為；多-Re 訓練到高 Re（如 1e6）時設成該上界，讓 FiLM 條件落在 [0,1]。
    "re_norm_scale":                 (float, 10000.0, _greater_than_one("re_norm_scale")),
    "observed_sensor_channels":      (list,  ["u", "v"], None),
    # DEPRECATED（inert）：實際 forcing 由 model 的 forcing_A_init/forcing_k_f_init 決定，
    # 此二 key 不被任何 code 讀取（值與 model 預設 0.1/2.0 巧合一致，故歷來無數值影響）。
    # 保留僅因既有 config 大量設了它們；勿移除以免 validation fail。新 config 請改設
    # model_kwargs.forcing_A_init / forcing_k_f_init。
    "kolmogorov_k_f":                (float, 2.0,  _positive("kolmogorov_k_f")),
    "kolmogorov_A":                  (float, 0.1,  _positive("kolmogorov_A")),
    # Cylinder wall-case data (spec 2026-06-10)
    "cylinder_data_npz":             (str,   "",   None),
    "cylinder_u_inf":                (float, 0.0,  _non_negative("cylinder_u_inf")),
    # controlled_cylinder npz（dump_controlled_cylinder_v2.py 產出；含反推 osc_* 與 base_center/body_radius）
    "controlled_data_npz":           (str,   "",   None),
}


# Keys that pi-lnn TOML 有但 POC 還沒實作 → 不 raise, 但 warn 一次


def flatten_toml(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten top-level values and sections in TOML insertion order."""
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flat[nested_key] = nested_value
        else:
            flat[key] = value
    return flat


def _validate_one(name: str, value: Any, schema: dict) -> Any:
    """Type check + validator；回傳 coerced value。"""
    expected_type, default, validator = schema[name]
    # bool 不能用 isinstance(value, int) 判斷（True is instance of int）
    if expected_type is bool and not isinstance(value, bool):
        raise TypeError(f"{name}: expected bool, got {type(value).__name__}")
    if expected_type is int and (not isinstance(value, int) or isinstance(value, bool)):
        raise TypeError(f"{name}: expected int, got {type(value).__name__}")
    if expected_type is float and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise TypeError(f"{name}: expected float, got {type(value).__name__}")
    if expected_type is str and not isinstance(value, str):
        raise TypeError(f"{name}: expected str, got {type(value).__name__}")
    if expected_type is list and not isinstance(value, list):
        raise TypeError(f"{name}: expected list, got {type(value).__name__}")
    # coerce int → float if expected float
    if expected_type is float and isinstance(value, int):
        value = float(value)
    if validator is not None:
        # validator 慣例：合法則 return None，非法則 raise。
        # 防呆 guard：回傳非 None（如經典 `lambda v: (cond, "msg")` 寫法）代表
        # 該 validator 是 silent no-op —— 直接 raise，避免驗證被默默跳過。
        result = validator(value)
        if result is not None:
            raise RuntimeError(
                f"{name}: validator 必須以 raise 回報錯誤、合法時 return None，"
                f"不可回傳值（收到 {result!r}）；請改用會 raise 的 closure"
                f"（_one_of / _in_range / _non_negative 等）"
            )
    return value


def load_config(path: str | Path) -> dict:
    """Load TOML config，依 schema 拆 model/train/data。

    Returns dict:
      {
        "model_kwargs": {...},      # 傳給 LiquidOperator
        "train_kwargs": {...},      # 覆蓋 CLI args
        "data_kwargs":  {...},
        "raw":          {...},      # 原始 TOML（供 debug）
        "unknown_keys": [...],      # 未認識的 key (warn)
      }
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TOML config 不存在: {path}")
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    # 平展 [train] section + 頂層 key；後出現的 section 仍覆蓋先出現者。
    flat = flatten_toml(raw)

    model_kwargs, train_kwargs, data_kwargs = {}, {}, {}
    unknown_keys = []

    for k, v in flat.items():
        if k in MODEL_SCHEMA:
            model_kwargs[k] = _validate_one(k, v, MODEL_SCHEMA)
        elif k in TRAIN_SCHEMA:
            train_kwargs[k] = _validate_one(k, v, TRAIN_SCHEMA)
        elif k in DATA_SCHEMA:
            data_kwargs[k] = _validate_one(k, v, DATA_SCHEMA)
        else:
            unknown_keys.append(k)

    # sensor_value_dim 從 observed_sensor_channels 推（若 user 沒明示）
    if "sensor_value_dim" not in model_kwargs:
        chs = data_kwargs.get("observed_sensor_channels", ["u", "v"])
        model_kwargs["sensor_value_dim"] = len(chs)

    # 應用 model defaults（補 missing 必要 key）
    # 注意：只對 model_kwargs 補預設（LiquidOperator(**model_kwargs) 需全引數）；
    # train/data_kwargs 刻意「不在此補」，由 train script 的 CLI→TOML→schema-default
    # merge 處理（見 test_causal_config.test_causal_eps_default_when_absent 合約）。
    for k, (_, default, _) in MODEL_SCHEMA.items():
        if k not in model_kwargs:
            model_kwargs[k] = default

    if unknown_keys:
        warnings.warn(
            f"TOML {path.name}: 未認識 keys（將被忽略）: {unknown_keys}",
            UserWarning, stacklevel=2,
        )

    return {
        "model_kwargs": model_kwargs,
        "train_kwargs": train_kwargs,
        "data_kwargs": data_kwargs,
        "raw": raw,
        "unknown_keys": unknown_keys,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Effective configuration resolution
# ─────────────────────────────────────────────────────────────────────────────

TConfig = TypeVar("TConfig")


class ConfigResolutionError(Exception):
    """A configuration error independent of its CLI presentation."""


class UnknownOverrideError(ConfigResolutionError):
    """A programmatic override named a field not owned by the case policy."""


@dataclass(frozen=True)
class CliGuardError(ConfigResolutionError):
    """A policy guard that the outer CLI adapter renders as an exit."""

    message: str
    exit_code: int


@dataclass(frozen=True)
class LayerValue:
    """One value contributed by one ordered configuration layer."""

    layer: str
    value: Any


@dataclass(frozen=True)
class FieldProvenance:
    """Complete shadow chain for one canonical effective-config field."""

    path: str
    chain: tuple[LayerValue, ...]

    @property
    def source(self) -> str:
        return self.chain[-1].layer

    @property
    def overwritten(self) -> tuple[LayerValue, ...]:
        return self.chain[:-1]


@dataclass(frozen=True)
class ConfigProvenance:
    """Sidecar lineage; numerical consumers never need to unwrap it."""

    fields: Mapping[str, FieldProvenance]
    ignored_toml_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))

    def for_field(self, path: str) -> FieldProvenance:
        return self.fields[path]


@dataclass(frozen=True)
class ResolutionResult(Generic[TConfig]):
    """A clean typed config plus diagnostic provenance sidecar."""

    config: TConfig
    provenance: ConfigProvenance


@dataclass(frozen=True)
class ModelConfig:
    """Validated constructor options behind a read-only typed seam."""

    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(deepcopy(dict(self.values))))

    @property
    def T_total(self) -> float:
        return float(self.values["T_total"])

    def to_kwargs(self) -> dict[str, Any]:
        """Return an isolated dict only at the existing model-factory seam."""
        return deepcopy(dict(self.values))


@dataclass(frozen=True)
class ConfigurationLayer:
    """An ordered mapping of canonical fields to candidate values."""

    name: str
    values: Mapping[str, Any]
    reject_unknown: bool = False


class ConfigPolicy(Protocol[TConfig]):
    """Case-local adapter consumed by the shared resolution engine."""

    known_fields: frozenset[str]

    def parse_cli(self, argv: list[str] | None) -> Any: ...

    def pre_load_guard(self, cli: Any) -> None: ...

    def layers(self, cli: Any, loaded: dict) -> Sequence[ConfigurationLayer]: ...

    def build(self, values: Mapping[str, Any]) -> TConfig: ...

    def guard(self, cli: Any, config: TConfig) -> None: ...


def merge_config_layers(
    layers: Sequence[ConfigurationLayer],
    *,
    known_fields: frozenset[str],
) -> tuple[Mapping[str, Any], Mapping[str, FieldProvenance]]:
    """Resolve ordered layers while retaining each overwritten observation.

    Values are defensively copied at the seam.  This preserves historical
    ``list``/``tuple`` types without sharing mutable TOML or argparse state.
    """
    values: dict[str, Any] = {}
    chains: dict[str, list[LayerValue]] = {}
    for layer in layers:
        unknown = sorted(set(layer.values) - known_fields)
        if unknown and layer.reject_unknown:
            raise UnknownOverrideError(
                f"{layer.name} contains unknown fields: {unknown}"
            )
        for path, value in layer.values.items():
            if path not in known_fields:
                continue
            copied = deepcopy(value)
            values[path] = copied
            chains.setdefault(path, []).append(
                LayerValue(layer.name, deepcopy(copied))
            )
    missing = sorted(known_fields - values.keys())
    if missing:
        raise ConfigResolutionError(f"policy did not resolve fields: {missing}")
    provenance = {
        path: FieldProvenance(path=path, chain=tuple(chain))
        for path, chain in chains.items()
    }
    return MappingProxyType(values), MappingProxyType(provenance)


def resolve_config(
    argv: list[str] | None,
    policy: ConfigPolicy[TConfig],
) -> ResolutionResult[TConfig]:
    """CLI + TOML + policy defaults -> one case-specific effective config.

    The policy currently supplies the three compatibility layers.  A future
    explicit-override adapter can append a strict ``ConfigurationLayer``
    without changing this engine or any downstream consumer.
    """
    cli = policy.parse_cli(argv)
    policy.pre_load_guard(cli)
    loaded = load_config(cli.config)
    values, field_provenance = merge_config_layers(
        policy.layers(cli, loaded), known_fields=policy.known_fields,
    )
    config = policy.build(values)
    policy.guard(cli, config)
    return ResolutionResult(
        config=config,
        provenance=ConfigProvenance(
            fields=field_provenance,
            ignored_toml_keys=tuple(loaded["unknown_keys"]),
        ),
    )

"""POC: pi-lnn (CfC + DeepONet + cross-attention) ported to JAX/Flax.

What: 完整保留 pi-lnn 的 LiquidOperator 架構（spatial encoder → temporal CfC →
       DeepONet decoder with cross-attention），改寫為 Flax `nn.Module`，
       時間遞迴用 `lax.scan` 取代 PyTorch Python loop。

Why: 量化 PyTorch → JAX 遷移在 Re=1000 baseline (pi-lnn EXP-030) 的數值差異，
     回答 D-stage 決策問題「能否在 ±2% 內重現 KE rel-err」。
"""

from .models import (
    CfCStep,
    SpatialSetEncoder,
    TemporalCfCEncoder,
    DeepONetCfCDecoder,
    LiquidOperator,
    VanillaDeepONetOperator,
    StandardPINNOperator,
    ForcingPrior,
)
from .physics import make_ns_residual_fn
from .losses import (
    GradNormState, ALState,
    gradnorm_init, gradnorm_step, gradnorm_weights,
    al_constraint_value, al_init, al_update,
)
from .curriculum import (
    physics_weight_at_step, time_marching_t_max,
    rar_init, rar_sample, RARState,
)
from .config import load_config
from .evaluate import (
    reconstruct_field, compute_metrics, evaluate_against_dns, evaluate_time_series,
    compute_divergence_l2, compute_vorticity, compute_energy_spectrum,
)
from .ckpt import CheckpointManager, TrainState
from .optimizers import (
    build_optimizer, is_soap_available, is_schedule_free_available,
)
from .refiners import (
    lbfgs_refine, lm_refine, make_pinn_residual_vector_fn,
)
# 訓練流水線的集中化接線層——兩個對等的 case（kolmogorov / cylinder）。
# 要理解整條訓練鏈，從 pi_lnn_jax/pipeline/__init__.py 開始讀：它是路標，
# 會把你導到該 case 自己的鏈圖（每個節點標明實作在哪個檔）。
#
# 下面這六個名字是 **Kolmogorov 的**（wave 1 只有一案時留下的相容別名，等同
# `pi_lnn_jax.pipeline.kolmogorov` 的同名匯出）。兩案的這六個名字同形不同物
# （TrainingContext / TrainResult 欄位集不同、resolve_inputs 解析不同旗標），
# 故 cylinder **不在**此處提供同名別名，只能從 `pi_lnn_jax.pipeline.cylinder`
# 匯入。新程式碼請一律走 case-qualified 路徑：
#     from pi_lnn_jax.pipeline.kolmogorov import resolve_inputs, build_context, run_training
#     from pi_lnn_jax.pipeline.cylinder   import resolve_inputs, build_context, run_training
from .pipeline import (
    KolmogorovEffectiveConfig, TrainResult, TrainingContext, build_context,
    resolve_inputs, run_training,
)

__all__ = [
    # Models
    "CfCStep", "SpatialSetEncoder", "TemporalCfCEncoder", "DeepONetCfCDecoder",
    "LiquidOperator", "VanillaDeepONetOperator", "StandardPINNOperator", "ForcingPrior",
    # Physics
    "make_ns_residual_fn",
    # Losses
    "GradNormState", "ALState", "gradnorm_init", "gradnorm_step", "gradnorm_weights", "al_constraint_value", "al_init", "al_update", # Curriculum
    "physics_weight_at_step", "time_marching_t_max",
    "rar_init", "rar_sample", "RARState",
    # Config
    "load_config",
    # Evaluate
    "reconstruct_field", "compute_metrics", "evaluate_against_dns", "evaluate_time_series",
    "compute_divergence_l2", "compute_vorticity", "compute_energy_spectrum",
    # Checkpoint
    "CheckpointManager", "TrainState",
    # Optimizers
    "build_optimizer", "is_soap_available", "is_schedule_free_available",
    # Refiners (Wave 5)
    "lbfgs_refine", "lm_refine", "make_pinn_residual_vector_fn",
    # Pipeline —— **Kolmogorov 的**相容別名；cylinder 見 pi_lnn_jax.pipeline.cylinder
    "KolmogorovEffectiveConfig", "TrainResult", "TrainingContext",
    "build_context", "resolve_inputs", "run_training",
]

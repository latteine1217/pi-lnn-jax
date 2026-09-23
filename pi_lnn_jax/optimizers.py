"""統一 optimizer factory（JAX/Optax 版）— pi-lnn PyTorch → JAX 遷移 POC 一部分。

What:
    `build_optimizer(name, ...)` 回傳 `(optax.GradientTransformation, info_dict)`，
    支援三種 optimizer：
      - "adam"          → 純 optax.adam（永遠 available）
      - "soap"          → soap_jax.soap（GPU dep，macOS 可能未裝 → graceful fallback）
      - "schedule_free" → 包在 adam/soap 外層的 ScheduleFree Polyak averaging

Why:
    對齊 pi-lnn config schema（`lr_schedule = "soap" | "adam"`，
    `use_schedule_free = bool`），讓 train_kolmogorov.py 能用同一份
    optimizer-construction code path 跑 baseline / ablation。

設計重點:
    1. **SOAP 不可用時 NEVER silent fallback** —— 印顯眼 warning + warnings.warn，
       並在 info_dict['fallback_to']='adam' 明示，避免重蹈 EXP-082 silent regression。
    2. **schedule_free 與 LR schedule 不重疊** —— schedule_free 內部自帶 warmup，
       不在外面再套 warmup_exponential_decay_schedule，避免 double-warmup。
       純 adam / soap 才用 optax LR schedule。
    3. **conditional import** —— soap-jax / optax_schedule_free 都 lazy import；
       未安裝環境下 module import 本身仍成功，is_*_available() 回 False。
"""

from __future__ import annotations

import sys
import warnings
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax


# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
# What: SOAP_JAX 與 optax_schedule_free 都試圖 import；失敗時把 flag 設成 False。
# Why: M3 macOS 沒裝 soap-jax（依賴 jax-cuda12 wheel）也不能讓本 module crash。

try:
    from soap_jax import soap as _soap_jax_soap  # type: ignore[import-not-found]
    SOAP_AVAILABLE = True
except ImportError:
    _soap_jax_soap = None  # type: ignore[assignment]
    SOAP_AVAILABLE = False


# ScheduleFree 來源優先序：
#   1. optax.contrib.schedule_free（optax 0.2.4+ 已 upstream，推薦）
#   2. optax_schedule_free.schedule_free（舊版套件）
# 兩者 API 都吃 (base_optimizer, learning_rate, ...) 並回 GradientTransformation。
_schedule_free_fn: Optional[Callable[..., optax.GradientTransformation]] = None
_schedule_free_source: Optional[str] = None

if hasattr(optax, "contrib") and hasattr(optax.contrib, "schedule_free"):
    _schedule_free_fn = optax.contrib.schedule_free
    _schedule_free_source = "optax.contrib.schedule_free"
else:
    try:
        from optax_schedule_free import schedule_free as _ext_schedule_free  # type: ignore[import-not-found]
        _schedule_free_fn = _ext_schedule_free
        _schedule_free_source = "optax_schedule_free"
    except ImportError:
        _schedule_free_fn = None
        _schedule_free_source = None

SCHEDULE_FREE_AVAILABLE = _schedule_free_fn is not None


# ---------------------------------------------------------------------------
# Public availability queries
# ---------------------------------------------------------------------------

def is_soap_available() -> bool:
    """回傳 soap_jax 套件是否成功 import。

    What/Why: 提供給 caller 在 build optimizer 前做 capability check，
    避免「請求 soap 但被 silent fallback 到 adam」的混淆。
    """
    return SOAP_AVAILABLE


def is_schedule_free_available() -> bool:
    """回傳 schedule_free（optax.contrib 或 optax_schedule_free）是否可用。"""
    return SCHEDULE_FREE_AVAILABLE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_lr_schedule(
    learning_rate: float,
    warmup_steps: int,
    decay_steps: int,
    decay_rate: float,
    min_learning_rate: float,
) -> tuple[Any, str]:
    """組合 LR schedule。

    Returns:
        (schedule_or_constant, description)
        - 若 warmup_steps=0 且 decay_steps=0: 回 float（constant LR）+ "constant"
        - 若僅 warmup_steps>0: linear warmup → constant + "warmup_then_constant"
        - 若僅 decay_steps>0: 從第 0 步就 exponential decay + "exponential_decay"
        - 若皆>0: warmup_exponential_decay_schedule + "warmup_exponential_decay"

    Note:
        optax 慣例上 schedule 是 `Callable[[step], lr]`；若回 float，optax 也接受。
    """
    has_warmup = warmup_steps > 0
    has_decay = decay_steps > 0 and decay_rate != 1.0

    if not has_warmup and not has_decay:
        return float(learning_rate), "constant"

    # init_value 必須 > 0：optax.contrib.schedule_free 內部用
    #   averaging_weight = (lr / running_max_lr)^weight_lr_power
    # 在 lr=0 時觸發 0/0 NaN（pi-lnn PyTorch ScheduleFree 有保護，optax 沒）。
    # 用 min_learning_rate (預設 1e-6) 當底；對 warmup 行為實質影響可忽略。
    safe_init = max(float(min_learning_rate), 1e-12)

    if has_warmup and not has_decay:
        # Linear warmup then hold at peak.
        warmup = optax.linear_schedule(
            init_value=safe_init,
            end_value=learning_rate,
            transition_steps=warmup_steps,
        )
        hold = optax.constant_schedule(learning_rate)
        schedule = optax.join_schedules([warmup, hold], boundaries=[warmup_steps])
        return schedule, f"warmup_{warmup_steps}_then_constant"

    if has_decay and not has_warmup:
        schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=decay_steps,
            decay_rate=decay_rate,
            end_value=min_learning_rate,
        )
        return schedule, f"exponential_decay_{decay_steps}_rate_{decay_rate}"

    # 兩者皆 enabled：用 optax 內建組合器。
    schedule = optax.warmup_exponential_decay_schedule(
        init_value=safe_init,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        transition_steps=decay_steps,
        decay_rate=decay_rate,
        end_value=min_learning_rate,
    )
    return schedule, f"warmup_{warmup_steps}_exponential_decay_{decay_steps}"


def _build_inner_optimizer(
    name: str,
    learning_rate_or_schedule: Any,
    soap_precondition_frequency: int,
    soap_b1: float = 0.95,
    soap_b2: float = 0.95,
    adam_b1: float = 0.9,
) -> tuple[optax.GradientTransformation, str, bool]:
    """建構 inner optimizer（不含 grad clip / weight decay / schedule_free wrap）。

    Returns:
        (optimizer, resolved_name, was_fallback)
        - resolved_name: 實際 build 出來的 optimizer 名（可能是 "adam" 若 soap fallback）
        - was_fallback: True 表示請求 soap 但已 fallback 至 adam
    """
    if name == "adam":
        return optax.adam(learning_rate=learning_rate_or_schedule, b1=adam_b1), "adam", False

    if name == "soap":
        if SOAP_AVAILABLE and _soap_jax_soap is not None:
            optimizer = _soap_jax_soap(
                learning_rate=learning_rate_or_schedule,
                b1=float(soap_b1),
                b2=float(soap_b2),
                precondition_frequency=int(soap_precondition_frequency),
            )
            return optimizer, f"soap_pf{soap_precondition_frequency}_b{soap_b1}/{soap_b2}", False
        else:
            # ====== LOUD FALLBACK WARNING ======
            # 此路徑風險級別與 EXP-082 silent resume regression 相當：
            # 使用者以為跑了 SOAP，實際跑了 adam。必須顯眼通知。
            warning_msg = (
                "[optimizers.build_optimizer] WARNING: requested 'soap' but "
                "soap_jax is NOT installed in this environment. "
                "Falling back to 'adam'. "
                "If you wanted SOAP, install soap-jax (https://github.com/haydn-jones/SOAP_JAX) "
                "or set name='adam' explicitly to silence this warning. "
                "Check info_dict['fallback_to'] == 'adam' in your training script."
            )
            print(f"\n{'!' * 80}\n{warning_msg}\n{'!' * 80}\n", file=sys.stderr, flush=True)
            warnings.warn(warning_msg, RuntimeWarning, stacklevel=3)
            return optax.adam(learning_rate=learning_rate_or_schedule), "adam_fallback_from_soap", True

    raise ValueError(
        f"build_optimizer: unsupported inner optimizer name='{name}'. "
        "Expected one of ['adam', 'soap']."
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_optimizer(
    name: str,
    learning_rate: float,
    base_optimizer: str = "adam",
    soap_precondition_frequency: int = 10,
    soap_b1: float = 0.95,
    soap_b2: float = 0.95,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 0,
    min_learning_rate: float = 1e-6,
    decay_steps: int = 0,
    decay_rate: float = 1.0,
) -> tuple[optax.GradientTransformation, dict]:
    """統一 optimizer 建構函式。

    Args:
        name: "adam" / "soap" / "schedule_free"
            - "adam" / "soap" → 直接 build 為 inner optimizer
            - "schedule_free" → 用 `base_optimizer` 當 inner，外層包 schedule_free
        learning_rate: peak LR
        base_optimizer: schedule_free 模式下的 inner optimizer 名稱
        soap_precondition_frequency: SOAP 每幾步更新一次 preconditioner
        max_grad_norm: global grad norm clip；<= 0 則不 clip
        warmup_steps: linear warmup 步數；0 = no warmup
        min_learning_rate: exponential decay 下限
        decay_steps: exponential decay transition_steps；0 = no decay
        decay_rate: < 1 才實際 decay；1.0 = no decay

    Returns:
        (optimizer, info_dict)
        info_dict 包含:
            - 'name': str — 實際 build 的 optimizer 名稱
            - 'available': bool — 請求的 optimizer 是否 available
            - 'fallback_to': str | None — 若 fallback，記錄實際使用名稱
            - 'lr_schedule': str — schedule 描述
            - 'soap_available': bool
            - 'schedule_free_available': bool
            - 'schedule_free_source': str | None
    """
    # 1. 組 LR schedule（schedule_free 與 inner adam/soap 都吃同一個）
    schedule, schedule_desc = _build_lr_schedule(
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        min_learning_rate=min_learning_rate,
    )

    info: dict[str, Any] = {
        "name": name,
        "available": True,
        "fallback_to": None,
        "lr_schedule": schedule_desc,
        "soap_available": SOAP_AVAILABLE,
        "schedule_free_available": SCHEDULE_FREE_AVAILABLE,
        "schedule_free_source": _schedule_free_source,
    }

    # 2. schedule_free 路徑
    if name == "schedule_free":
        if not SCHEDULE_FREE_AVAILABLE:
            warning_msg = (
                "[optimizers.build_optimizer] WARNING: requested 'schedule_free' but "
                "neither optax.contrib.schedule_free nor optax_schedule_free is available. "
                "Upgrade optax to >= 0.2.4 or install optax-schedule-free. "
                "Falling back to plain '{base_optimizer}'."
            ).format(base_optimizer=base_optimizer)
            print(f"\n{'!' * 80}\n{warning_msg}\n{'!' * 80}\n", file=sys.stderr, flush=True)
            warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)

            # Fallback：直接 build base optimizer（不 wrap）
            inner, inner_name, was_soap_fallback = _build_inner_optimizer(
                base_optimizer, schedule, soap_precondition_frequency, soap_b1, soap_b2
            )
            info["available"] = False
            info["fallback_to"] = inner_name
            info["name"] = f"{inner_name}_fallback_from_schedule_free"
            chain = _wrap_chain(inner, max_grad_norm)
            return chain, info

        # schedule_free 自帶 momentum（via x/y/z interpolation）；Optax 要求關閉 base
        # optimizer 的 first-moment momentum（adam b1=0、soap b1=0），否則 double momentum。
        inner, inner_name, was_soap_fallback = _build_inner_optimizer(
            base_optimizer,
            learning_rate_or_schedule=schedule,
            soap_precondition_frequency=soap_precondition_frequency,
            soap_b1=0.0,
            soap_b2=soap_b2,
            adam_b1=0.0,
        )
        assert _schedule_free_fn is not None
        wrapped = _schedule_free_fn(base_optimizer=inner, learning_rate=schedule)
        info["name"] = f"schedule_free({inner_name})"
        if was_soap_fallback:
            info["fallback_to"] = "schedule_free(adam_fallback_from_soap)"
            info["available"] = False
        chain = _wrap_chain(wrapped, max_grad_norm)
        return chain, info

    # 3. 純 adam / soap 路徑
    inner, inner_name, was_soap_fallback = _build_inner_optimizer(
        name, schedule, soap_precondition_frequency, soap_b1, soap_b2
    )
    if was_soap_fallback:
        info["available"] = False
        info["fallback_to"] = "adam"
    info["name"] = inner_name
    chain = _wrap_chain(inner, max_grad_norm)
    return chain, info


def _wrap_chain(
    inner: optax.GradientTransformation,
    max_grad_norm: float,
) -> optax.GradientTransformation:
    """組裝 optax.chain(clip → inner)。

    2026-08-04：移除 weight_decay 參數與 add_decayed_weights 分支。它是
    `POC_NOT_YET_KEYS` 的下游死碼——0 份 config 設它、無 CLI 旗標、
    build_optimizer 的兩個 production 呼叫端都不傳，於是那條 chain
    從未被組裝過。"""
    transforms: list[optax.GradientTransformation] = []
    if max_grad_norm is not None and max_grad_norm > 0:
        transforms.append(optax.clip_by_global_norm(max_grad_norm))
    transforms.append(inner)
    if len(transforms) == 1:
        return transforms[0]
    return optax.chain(*transforms)


def accumulate_grads(value_and_grad_fn: Callable, params: Any, chunked_inputs: tuple):
    """Gradient accumulation over input chunks（記憶體峰值 ∝ 單塊，非全量）。

    大 K / 大 batch 下 cross-attention 等中介張量 ∝ query 數，會 OOM。把 query 維度切 M 塊、
    用 `lax.scan` 逐塊 backward（每塊 activation graph 跑完即釋放），最後對梯度取平均——
    等價於全量 mean-loss 的梯度，但峰值記憶體只佔一塊。port 自 main(49527fe)。

    Args:
      value_and_grad_fn: `(params, *chunk) -> ((loss, aux), grads)`，即
          `jax.value_and_grad(loss_fn, has_aux=True)`（**須 has_aux=True**）。loss 須為 mean
          型才與全量精確等價。
      params:        模型參數 pytree。
      chunked_inputs: tuple/list，每個元素 leading 軸為 M（被 scan 逐塊取用）；各塊須等大。

    Returns:
      (grads, loss, aux)，皆對 M 塊取平均。grads pytree 與 params 同構。

    Note:
      - scan 逐塊執行 → 中介 activation 不會同時保留 M 份（記憶體省在此）。
      - **aux 須為 mean-compatible**（per-chunk 取平均）：max / min / count 型 aux 會得到
        錯誤語意。AL 的 C² 在 chunk 下是 per-chunk 近似（呼叫端須註明）。
    """
    if not isinstance(chunked_inputs, (tuple, list)):
        raise TypeError(
            "chunked_inputs 須為 tuple/list of 已切塊輸入（各元素 leading 軸=M）；"
            f"收到 {type(chunked_inputs).__name__}。單一 array 會被 *chunk 誤展開。")

    def _body(_, chunk):
        (loss, aux), grads = value_and_grad_fn(params, *chunk)
        return None, (grads, loss, aux)

    _, (grads_s, loss_s, aux_s) = jax.lax.scan(_body, None, chunked_inputs)

    def _mean(x):
        return jnp.mean(x, axis=0)
    return (jax.tree_util.tree_map(_mean, grads_s), _mean(loss_s),
            jax.tree_util.tree_map(_mean, aux_s))


__all__ = [
    "build_optimizer",
    "is_soap_available",
    "is_schedule_free_available",
    "SOAP_AVAILABLE",
    "SCHEDULE_FREE_AVAILABLE",
    "accumulate_grads",
]

"""Optimizer factory tests — pi-lnn PyTorch → JAX 遷移 POC 一部分。

Why:
    驗證 `pi_lnn_jax.optimizers.build_optimizer` 在三種 optimizer
    （adam / soap / schedule_free）與 SOAP 不可用環境下的行為。

Coverage:
    1. adam 永遠 available
    2. is_soap_available() 不 raise
    3. SOAP not available → graceful fallback to adam + 顯眼 warning
    4. schedule_free 可查可 build
    5. toy quadratic loss 50 步 adam 應降 ≥ 95%

判定 PASS:
    所有 test 通過 + final summary 印「N/N tests PASS」。
"""
from __future__ import annotations

import io
import sys
import warnings

import jax
import jax.numpy as jnp
import optax

from pi_lnn_jax.optimizers import (
    build_optimizer,
    is_soap_available,
    is_schedule_free_available,
    SOAP_AVAILABLE,
    SCHEDULE_FREE_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_optax_transform(obj) -> bool:
    """Optax GradientTransformation/GradientTransformationExtraArgs 都有 init/update。"""
    return hasattr(obj, "init") and hasattr(obj, "update") and callable(obj.init)


def _toy_init_params(n: int = 5):
    """Toy params: 5 個 scalars 初始化為 1.0；target=0 → loss=sum(x^2)。"""
    return {"x": jnp.ones((n,), dtype=jnp.float32)}


def _toy_loss(params):
    return jnp.sum(params["x"] ** 2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_adam_always_available():
    """build_optimizer('adam', 1e-3) 必須永遠成功且回 valid optimizer。"""
    opt, info = build_optimizer("adam", learning_rate=1e-3)
    assert _is_optax_transform(opt), f"expected optax transform, got {type(opt)}"
    assert info["name"] == "adam", f"info name = {info['name']}"
    assert info["available"] is True
    assert info["fallback_to"] is None
    assert info["lr_schedule"] == "constant"

    # smoke: 能 init 並執行 update
    params = _toy_init_params()
    state = opt.init(params)
    grads = jax.grad(_toy_loss)(params)
    updates, new_state = opt.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)
    assert jnp.all(jnp.isfinite(new_params["x"]))
    print(f"✓ test_adam_always_available: info = {info}")


def test_soap_availability_query():
    """is_soap_available() 不 raise，回 bool。SOAP_AVAILABLE 也一致。"""
    result = is_soap_available()
    assert isinstance(result, bool), f"expected bool, got {type(result)}"
    assert result is SOAP_AVAILABLE
    print(f"✓ test_soap_availability_query: SOAP_AVAILABLE = {result}")


def test_soap_fallback():
    """SOAP 不可用時，build_optimizer('soap', ...) 應 fallback 到 adam，
    info['fallback_to']='adam'，且 print 顯眼 warning 到 stderr / warnings.warn。

    若 SOAP 可用（Linux GPU env），此 test 改驗 SOAP 正常 build（no fallback）。
    """
    # 捕捉 stderr 與 warnings
    captured_stderr = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = captured_stderr

    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            opt, info = build_optimizer("soap", learning_rate=1e-3)
    finally:
        sys.stderr = old_stderr

    stderr_text = captured_stderr.getvalue()

    assert _is_optax_transform(opt), f"expected optax transform, got {type(opt)}"

    if SOAP_AVAILABLE:
        # Happy path: SOAP 真的可用
        assert info["available"] is True
        assert info["fallback_to"] is None
        assert "soap" in info["name"]
        print(f"✓ test_soap_fallback (SOAP available): info = {info}")
    else:
        # Fallback path
        assert info["available"] is False, f"expected available=False, got {info}"
        assert info["fallback_to"] == "adam", f"expected fallback_to='adam', got {info['fallback_to']}"
        assert info["name"] == "adam_fallback_from_soap", f"got name={info['name']}"

        # 必須有顯眼 stderr 訊息
        assert "WARNING" in stderr_text or "soap_jax" in stderr_text, (
            f"expected loud stderr warning, got:\n{stderr_text!r}"
        )
        assert "!" * 10 in stderr_text, (
            f"expected '!!!' banner in stderr, got:\n{stderr_text!r}"
        )

        # 必須有 RuntimeWarning
        runtime_warns = [w for w in wlist if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warns) >= 1, (
            f"expected RuntimeWarning, got {len(runtime_warns)} of {len(wlist)} total"
        )

        # smoke: fallback adam 仍可運作
        params = _toy_init_params()
        state = opt.init(params)
        grads = jax.grad(_toy_loss)(params)
        updates, _ = opt.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        assert jnp.all(jnp.isfinite(new_params["x"]))

        print("✓ test_soap_fallback (SOAP unavailable → adam):")
        print(f"    info = {info}")
        print(f"    stderr_excerpt = {stderr_text.strip().splitlines()[0][:80] if stderr_text.strip() else '(empty)'}")


def test_schedule_free_query_and_build():
    """is_schedule_free_available() 回 bool；若 available，
    build_optimizer('schedule_free', ...) 應回 valid wrapped optimizer。
    """
    sf_avail = is_schedule_free_available()
    assert isinstance(sf_avail, bool)
    assert sf_avail is SCHEDULE_FREE_AVAILABLE

    if not sf_avail:
        # 不 available 的環境：build 應 fallback 但仍 return valid optimizer
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            opt, info = build_optimizer("schedule_free", learning_rate=1e-3)
        assert _is_optax_transform(opt)
        assert info["available"] is False
        assert info["fallback_to"] is not None
        print(f"✓ test_schedule_free_query_and_build (SF unavailable): info = {info}")
        return

    # 正常 build schedule_free + adam（最常見）
    opt, info = build_optimizer(
        "schedule_free",
        learning_rate=1e-3,
        base_optimizer="adam",
        warmup_steps=10,
    )
    assert _is_optax_transform(opt)
    assert info["available"] is True
    assert "schedule_free" in info["name"]
    assert "adam" in info["name"]
    assert info["schedule_free_source"] is not None

    # smoke: init + update
    params = _toy_init_params()
    state = opt.init(params)
    grads = jax.grad(_toy_loss)(params)
    updates, _ = opt.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)
    assert jnp.all(jnp.isfinite(new_params["x"])), "schedule_free adam produced non-finite update"

    print(f"✓ test_schedule_free_query_and_build (SF available): info = {info}")


def test_optimizer_works_on_toy_loss():
    """用 build_optimizer('adam') 對 toy quadratic 跑 50 步，verify loss 下降 ≥ 95%。

    Why: 確保整條 (chain=clip+wd+adam) 在實際 optimization loop 上不會破壞收斂。
    """
    opt, info = build_optimizer(
        "adam",
        learning_rate=0.1,
        max_grad_norm=10.0,   # 上限，但 toy grad 不會觸發 clip
    )

    params = _toy_init_params(n=10)
    state = opt.init(params)
    initial_loss = float(_toy_loss(params))

    loss_history = [initial_loss]
    for step in range(50):
        loss_val, grads = jax.value_and_grad(_toy_loss)(params)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        loss_history.append(float(loss_val))

    final_loss = float(_toy_loss(params))
    relative_reduction = (initial_loss - final_loss) / initial_loss

    assert jnp.all(jnp.isfinite(params["x"])), "params diverged"
    assert relative_reduction >= 0.95, (
        f"expected ≥95% loss reduction, got {relative_reduction*100:.2f}% "
        f"(initial={initial_loss:.4f} → final={final_loss:.6f})"
    )

    print("✓ test_optimizer_works_on_toy_loss:")
    print(f"    initial loss = {initial_loss:.6f}")
    print(f"    final loss   = {final_loss:.6e}")
    print(f"    reduction    = {relative_reduction*100:.2f}%")
    print(f"    info         = {info}")

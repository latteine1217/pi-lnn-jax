"""cylinder 訓練迴圈的 NaN fail-fast —— job 4745 的回歸測試。

job 4745（`exp_cyl_cexp002_biasfix.toml`, seed 43）在 step 2 就 NaN，舊行為只印一行
到 stderr 然後 `break`：迴圈之後的 ledger 尾列與 `finalize` 的 DNS 重建 eval 照跑，
印出 `ke_pred/ke_ref = nan`，程序以 rc=0 結束，Slurm 記成 COMPLETED——壞掉的 run
因此混進 multi-seed 聚合。本檔釘住的是「偵測到就當場以非零 exit code 終止」。

不跑真 training：迴圈的終止行為只取決於 `step_fn` 回傳的 total 是否有限，故用
假 step_fn 與最小 ctx（`_ctx_from_data_meta` 的 blank-context 手法）頂掉模型與資料，
可在本機 CPU 秒級跑完。

兩個測試各自帶**鑑別力自檢**：同一組假件在 total 有限時必須跑完、必須落下 ledger
尾列、必須呼叫 finalize。少了這一步，「沒有產物」可能只是假件根本沒走到那裡。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from pi_lnn_jax.boundary import CylinderGeometry
from pi_lnn_jax.pipeline._ledger import reset_ledger
from pi_lnn_jax.pipeline.cylinder import run as run_mod
from pi_lnn_jax.pipeline.cylinder.assembly import TrainingContext
from pi_lnn_jax.pipeline.cylinder.config import resolve_inputs

REPO_ROOT = Path(__file__).resolve().parent.parent

_T, _K = 4, 8          # 假時間軸／sensor 數：只要 T*K ≥ n_sensor_q 即可
_STEPS = 4
_NAN_AT = 2            # 對齊 job 4745：NaN 出現在迴圈中途，不是第 0 步


class _FakeStepFn:
    """假 step_fn：第 `nan_at` 步回 NaN total，其餘回有限值；params/opt_state 原樣回傳。

    記錄呼叫次數，讓「偵測到就停」可以被斷言（而不是只看有沒有拋 SystemExit）。
    """

    def __init__(self, nan_at: int | None):
        self.nan_at = nan_at
        self.calls = 0

    def __call__(self, params, opt_state, *args):
        s = self.calls
        self.calls += 1
        total = np.nan if s == self.nan_at else 1.0
        aux = tuple(jnp.asarray(0.1, jnp.float32) for _ in range(5))
        return params, opt_state, jnp.asarray(total, jnp.float32), aux


def _fake_ctx(artifacts_dir: Path, step_fn) -> TrainingContext:
    """真 config（走生產 `resolve_inputs`）+ 最小資料衍生量 + 假 step_fn。

    只填 `run_loop` / `_plan_step` 會讀的欄位，其餘留 None：走到別的欄位就該炸，
    不該拿到一個看起來合理的預設值（同 `_ctx_from_data_meta` 的紀律）。
    """
    config = resolve_inputs([
        "--config", str(REPO_ROOT / "configs/exp_cyl_cexp002_notm.toml"),
        "--steps", str(_STEPS), "--seed", "42", "--allow-cpu",
    ]).config
    config = replace(config, run=replace(config.run, artifacts_dir=str(artifacts_dir)))
    st = np.linspace(0.0, 1.0, _T, dtype=np.float32)
    blank = TrainingContext(**dict.fromkeys(TrainingContext._fields))
    return blank._replace(
        config=config,
        st=st,   # ledger 尾列的 `_data_meta` 讀它（`_plan_step` 不讀）
        st0=float(st[0]), T_total=1.0, tm_end=float(st[-1]),
        K=_K, T=_T, t_q_full_np=np.repeat(st, _K),
        n_sensor_q=min(config.run.n_sensor_query_requested, _T * _K),
        geom=CylinderGeometry(body_center=(0.2, 0.5), body_radius=0.05,
                              Lx=1.0, Ly=1.0, u_inf=1.0),
        step_fn=step_fn,
        grad_norm_fn=None,   # gradnorm_freq=25 > steps，本檔的 4 步內不觸發
    )


def _fake_state() -> run_mod.TrainingState:
    """假 state：`run_loop` 只把 params/opt_state 轉手給 step_fn，內容不影響控制流。"""
    return run_mod.TrainingState(
        params={}, opt_state=None, rk=None, gn_state=None,
        al_state=SimpleNamespace(lambda_=jnp.asarray(0.0, jnp.float32)),
        task_weights=jnp.ones((4,), jnp.float32),
    )


def test_run_loop_exits_with_code_2_on_nan(tmp_path, capsys):
    """NaN → `SystemExit(2)`，訊息與退出碼對齊 `kolmogorov/run.py`。"""
    step_fn = _FakeStepFn(nan_at=_NAN_AT)
    with pytest.raises(SystemExit) as excinfo:
        run_mod.run_loop(_fake_ctx(tmp_path, step_fn), _fake_state())

    assert excinfo.value.code == 2, (
        f"NaN 後的 exit code 是 {excinfo.value.code!r}；rc=0 會讓 Slurm 記成 COMPLETED"
    )
    out = capsys.readouterr().out
    assert f"[FATAL] loss NaN/Inf at step {_NAN_AT} — abort" in out, (
        f"訊息未對齊 kolmogorov/run.py，實得：\n{out}"
    )
    assert step_fn.calls == _NAN_AT + 1, (
        f"偵測到 NaN 後又跑了 {step_fn.calls - _NAN_AT - 1} 步——應當場終止"
    )
    print(f"✓ run_loop_exits_with_code_2_on_nan (step {_NAN_AT}, {step_fn.calls} 次 step_fn)")


def test_finite_loss_runs_to_completion(tmp_path):
    """鑑別力自檢：同一組假件在 total 恆有限時跑完全部步數且不終止。

    少了這條，上一條的 SystemExit 可能來自假件本身而非 NaN 判斷。
    """
    step_fn = _FakeStepFn(nan_at=None)
    state = run_mod.run_loop(_fake_ctx(tmp_path, step_fn), _fake_state())

    assert step_fn.calls == _STEPS + 1, (   # 迴圈是 range(steps + 1)，含第 0 步
        f"有限 loss 下只跑了 {step_fn.calls} 步，預期 {_STEPS + 1}"
    )
    assert state is not None
    print(f"✓ finite_loss_runs_to_completion ({step_fn.calls} 步)")


@pytest.mark.parametrize("nan_at,expect_ledger", [(_NAN_AT, False), (None, True)])
def test_nan_abort_produces_no_ledger_tail_and_no_eval(
    nan_at, expect_ledger, tmp_path, monkeypatch,
):
    """NaN 後不得續走 ledger 尾列與 eval；`nan_at=None` 那組是同一組假件的鑑別力自檢。

    `finalize` 是 cylinder 唯一的 eval 段（DNS grid 重建 → KE 報表），故「eval 產物」
    在本案等同「finalize 被呼叫過」加上「ledger 尾列落檔」——cylinder 本身不落其他
    artifact（見 `run.py` module docstring）。
    """
    monkeypatch.setenv("PILNN_RNG_LEDGER", "1")
    reset_ledger()
    try:
        state = _fake_state()
        finalize_calls = []
        monkeypatch.setattr(run_mod, "initialize", lambda ctx: state)
        monkeypatch.setattr(run_mod, "finalize",
                            lambda ctx, st: finalize_calls.append(st) or "result")
        ctx = _fake_ctx(tmp_path, _FakeStepFn(nan_at=nan_at))

        if expect_ledger:
            assert run_mod.run_training(ctx) == "result"
        else:
            with pytest.raises(SystemExit) as excinfo:
                run_mod.run_training(ctx)
            assert excinfo.value.code == 2

        ledger_path = tmp_path / "rng_ledger.json"
        assert ledger_path.exists() is expect_ledger, (
            f"nan_at={nan_at}：ledger 尾列落檔與預期不符（{ledger_path}）"
        )
        assert bool(finalize_calls) is expect_ledger, (
            f"nan_at={nan_at}：finalize 呼叫與預期不符——NaN 後不得再做 eval"
        )
    finally:
        reset_ledger()
    print(f"✓ nan_abort_produces_no_ledger_tail_and_no_eval (nan_at={nan_at})")

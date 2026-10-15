"""`eval_pressure_limit.py` 端到端：合成的小 checkpoint 跑完整條路徑。

What:
    造一個最小 LiquidOperator、存成 orbax ckpt、配一份合成 sensor／DNS／config／
    summary.json，再用**既有的** `evaluate_time_series` 算出 uv_rel_err 當作
    `--expect-metrics`。腳本的重建路徑（腳本內自寫的 encode-once 迴圈）必須在閘門
    容差內重現這個數字——這就是 lab-server 上 uv 指紋閘門要背書的那件事，在本機
    先證一次。

    合成 DNS 的 `p` 欄位刻意填 NaN：腳本若讀了它，輸出不可能全是有限值。

Why 不 mock model:
    閘門背書的是「腳本的 decode 迴圈 = 評估器的 decode 迴圈」，mock 掉 model 就
    什麼都沒驗。最小模型的編譯成本（~10 s）是這個背書的價格。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _minimal_model import minimal_kwargs
from _paths import REPO_ROOT

from pi_lnn_jax.ckpt import CheckpointManager, TrainState, reference_params_for
from pi_lnn_jax.config import load_config
from pi_lnn_jax.evaluate import evaluate_time_series
from pi_lnn_jax.evaluation_protocol import load_for_evaluation, resolve_protocol
from pi_lnn_jax.metric_artifact import MetricUnavailable
from pi_lnn_jax.model_factory import build_model, model_fingerprint
from scripts import aggregate_pressure_limit as agg
from scripts import eval_pressure_limit as epl

K, N, N_TIME, STEP = 5, 16, 11, 7


def _write_case(root: Path) -> dict:
    """合成 run：sensor（11 幀，stride 2 後 6 幀）、DNS（16²、同一條時間軸）、config、ckpt。"""
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, N_TIME).astype(np.float32)

    np.savez(root / "sensors_dns_values.npz", time=t,
             u=rng.normal(size=(K, N_TIME)).astype(np.float32),
             v=rng.normal(size=(K, N_TIME)).astype(np.float32))
    sensor_json = root / "sensors.json"
    sensor_json.write_text(json.dumps({
        "K": K, "selected_coordinates": rng.uniform(size=(K, 2)).tolist(),
        "dns_values_npz": "sensors_dns_values.npz", "channels": ["u", "v"]}))

    dns_path = root / "dns.npy"
    np.save(dns_path, {
        "u": rng.normal(size=(N_TIME, N, N)), "v": rng.normal(size=(N_TIME, N, N)),
        "p": np.full((N_TIME, N, N), np.nan),           # 讀了就會汙染輸出
        "time": t.astype(np.float64), "config": {"L": 1.0}}, allow_pickle=True)

    run_dir = root / "run"
    ckpt_dir = run_dir / "checkpoints"
    config = root / "exp.toml"
    body = "\n".join(f"{k} = {json.dumps(v)}" for k, v in minimal_kwargs(2).items())
    config.write_text(
        "[train]\n"
        f'sensor_jsons = ["{sensor_json}"]\n'
        f'dns_paths = ["{dns_path}"]\n'
        "re_values = [10000.0]\n"
        "seed = 7\n"
        f'artifacts_dir = "{run_dir}"\n' + body + "\n")

    cfg = load_config(config)
    protocol = resolve_protocol(mode="follow_training", training_time_strides=[])
    aligned = load_for_evaluation(sensor_json, dns_path, protocol=protocol,
                                  viscosity=1e-4, with_pressure=False)
    sv, sp, st = (jnp.asarray(aligned.sensor_vals_normalized), jnp.asarray(aligned.sensor_pos),
                  jnp.asarray(aligned.sensor_time))
    model, _ = build_model("liquid", cfg["model_kwargs"], K_sensors=K)
    params = reference_params_for(model, sv, sp, st)
    # 隨機擾動，讓輸出不是初始化那組近常數場（uv 指紋才有鑑別力）
    leaves, tree = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(1), len(leaves))
    params = jax.tree_util.tree_unflatten(
        tree, [x + 0.3 * jax.random.normal(k, x.shape, x.dtype) for x, k in zip(leaves, keys)])

    mgr = CheckpointManager(directory=ckpt_dir, max_to_keep=3, save_interval_steps=1)
    mgr.save(STEP, TrainState(params=params, opt_state=(), step=jnp.asarray(STEP, jnp.int32),
                              rng_key=jax.random.PRNGKey(0)), force=True)
    (run_dir / "summary.json").write_text(json.dumps({
        "ckpt_dir": str(ckpt_dir), "arch": "liquid", "config_path": str(config),
        "use_al": True, "model_construction": model_fingerprint(model)}))

    # 既有評估器的 uv（re_norm = log(1e4)/log(1e4) = 1，同腳本由 schema default 推得）
    res = evaluate_time_series(model, params, sv, sp, 1.0, st, aligned.norm_stats,
                               aligned.dns_u_eval, aligned.dns_v_eval, aligned.dns_t_eval,
                               verbose=False)
    expect = {"ckpt_dir": str(ckpt_dir), "ckpt_step": STEP, "config": str(config),
              "metrics_mean": {"uv_rel_err": res["metrics_mean"]["uv_rel_err"]},
              "metrics_per_t": [{"t": row["t"]} for row in res["metrics_per_t"]]}
    expect_path = root / "expect_metrics.json"
    expect_path.write_text(json.dumps(expect))
    return {"config": config, "run_dir": run_dir, "expect": expect, "expect_path": expect_path,
            "n_frames": len(aligned.dns_t_eval)}


def _argv(case, out: Path, expect_path: Path | None = None, arm: str = "K5_al") -> list[str]:
    return ["--arch", "liquid", "--protocol", "follow_training",
            "--artifacts-dir", str(case["run_dir"]), "--config", str(case["config"]),
            "--expect-metrics", str(expect_path or case["expect_path"]),
            "--arm", arm, "--out", str(out)]


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    return _write_case(tmp_path_factory.mktemp("pressure_limit"))


@pytest.fixture(scope="module")
def output(case):
    out = case["run_dir"].parent / "out" / "run.json"
    assert epl.main(_argv(case, out)) == 0
    return json.loads(out.read_text())


def test_end_to_end_output_structure(case, output):
    n = case["n_frames"]
    assert output["producer"] == epl.PRODUCER and output["schema"] == epl.SCHEMA
    assert len(output["times"]) == n
    for k in epl.METRIC_KEYS:
        m = output["metrics"][k]
        assert len(m["per_frame"]) == n and all(math.isfinite(x) for x in m["per_frame"])
        assert m["time_mean"] == pytest.approx(float(np.mean(m["per_frame"])), rel=1e-12)
    gate = output["uv_gate"]
    assert gate["passed"] and gate["abs_diff"] <= epl.UV_GATE_ATOL
    assert output["run"]["ckpt_step"] == STEP and output["run"]["seed"] == 7
    assert output["fingerprint_verified"] is True          # summary 帶 model_construction
    assert output["norm_stats_verified"] is False           # summary 無 norm_stats → 警告路徑
    assert output["norm_stats_eval"]["p_mean"] == 0.0 and output["norm_stats_eval"]["p_std"] == 1.0
    assert output["pressure_reference"]["dns_p_field_used"] is False
    script = Path(epl.__file__)
    assert output["script"]["sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert output["repo"]["root"] == str(REPO_ROOT)
    assert output["repo"]["git_sha"] != ""


def test_script_reconstruction_matches_the_evaluator(case, output):
    """腳本自寫的 decode 迴圈與 evaluate_time_series 走同一個 forward。

    同一台機器、同一個分塊大小（K≤128 → 4096，16² 只有一塊）下應逐位元相同；
    容差留給 float 累加，遠小於閘門的 2e-4。
    """
    assert output["uv_gate"]["abs_diff"] < 1e-6


def test_uv_gate_fails_loudly_on_mismatch(case, tmp_path):
    bad = copy.deepcopy(case["expect"])
    bad["metrics_mean"]["uv_rel_err"] += 1e-3
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    out = tmp_path / "never.json"
    with pytest.raises(SystemExit, match="uv 指紋不符"):
        epl.main(_argv(case, out, p))
    assert not out.exists()


@pytest.mark.parametrize("mutate,exc,match", [
    # 別的 run 的 metrics.json：相鄰 seed 的 uv 可以只差 7e-6，數值擋不住，要靠 provenance
    (lambda d: d.update(ckpt_dir="/elsewhere/checkpoints"), ValueError, "ckpt_dir"),
    (lambda d: d.update(config="/elsewhere/exp.toml"), ValueError, "config"),
    # ablate_no_al_s42 的既有評估早於 uv_rel_err：沒有指紋就不跑
    (lambda d: d["metrics_mean"].pop("uv_rel_err"), MetricUnavailable, "uv_rel_err"),
    (lambda d: d["metrics_per_t"].pop(), ValueError, "評估時刻"),
])
def test_expectation_must_describe_this_run(case, tmp_path, mutate, exc, match):
    bad = copy.deepcopy(case["expect"])
    mutate(bad)
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(exc, match=match):
        epl.main(_argv(case, tmp_path / "never.json", p))


# ── 聚合 ──────────────────────────────────────────────────────────────────────

def _variant(output: dict, tmp_path: Path, name: str, **run) -> Path:
    d = copy.deepcopy(output)
    d["run"].update(run)
    for k in epl.METRIC_KEYS:
        d["metrics"][k]["time_mean"] += 0.01 * run.get("seed", 0)
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(d))
    return p


def test_aggregate_mean_and_sample_sd(output, tmp_path):
    ins = [_variant(output, tmp_path, f"s{s}", seed=s) for s in (1, 2, 3)]
    solo = copy.deepcopy(output)
    solo["arm"] = "K5_noal"
    solo["run"]["use_al"] = False
    (tmp_path / "solo.json").write_text(json.dumps(solo))
    out = tmp_path / "agg.json"
    assert agg.main([*map(str, ins), str(tmp_path / "solo.json"), "--out", str(out)]) == 0
    arms = json.loads(out.read_text())["arms"]
    vals = [output["metrics"]["grad_p_rel_l2"]["time_mean"] + 0.01 * s for s in (1, 2, 3)]
    m = arms["K5_al"]["metrics"]["grad_p_rel_l2"]
    assert arms["K5_al"]["n"] == 3
    assert m["mean"] == pytest.approx(np.mean(vals)) and m["sd"] == pytest.approx(np.std(vals, ddof=1))
    assert arms["K5_noal"]["n"] == 1 and arms["K5_noal"]["metrics"]["uv_rel_err"]["sd"] is None


@pytest.mark.parametrize("second,match", [
    ({"seed": 1}, "重複 seed"),
    ({"seed": 2, "K": 200}, "不同的 K"),
    ({"seed": 2, "use_al": False}, "不同的 use_al"),
])
def test_aggregate_refuses_to_pool_unlike_runs(output, tmp_path, second, match):
    a = _variant(output, tmp_path, "a", seed=1)
    b = _variant(output, tmp_path, "b", **second)
    with pytest.raises(ValueError, match=match):
        agg.main([str(a), str(b), "--out", str(tmp_path / "agg.json")])

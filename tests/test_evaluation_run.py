"""Evaluation-run recorder —— 透過介面驗行為，不驗實作細節。

問題（ADR-0002 的邏輯完成）：`evaluate_field_series` 是 metric artifact 的唯一寫入
路徑，但環繞它的生命週期（組 source provenance、把已解析的 evaluation protocol
內嵌進 provenance、依凍結慣例命名、落盤）先前被 7 支 producer 各手抄一遍。
`EvaluationRunRecorder` 把這段收成一個 deep module。

本檔驗的是 recorder 的**介面**（它的測試面就是介面本身）：

  1. 正常 `record()` 寫出的 artifact，其 provenance 帶 producer + code revision +
     evaluation protocol（recorder 給的那個保證）。
  2. 依 identity 組出的凍結檔名，逐字對上各 producer 現有的名（re+method、
     method-only、含 modes、裸單筆）。
  3. 回傳的 projection == `evaluate_field_series` 回傳的 projection。

汙染探針（承 `test_ns_residual_parity` / `test_attention_kind_forward` 的作法）：

  - 漏傳 `protocol` 或 `inputs` 必須大聲失敗，不得靜默漏蓋——那正是「靜默
    provenance 洞」。
  - 餵錯 reference（把 u_ref/v_ref 對調）必須被 metric 察覺（projection 改變），
    否則這個 seam 對「拿錯真值」是瞎的。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pi_lnn_jax.evaluation_protocol import (
    EvaluationProtocol,
    ProtocolMode,
    resolve_protocol,
)
from pi_lnn_jax.evaluation_run import EvaluationRunRecorder, RunArtifactIdentity
from pi_lnn_jax.metric_artifact import (
    EvaluationContext,
    SourceProvenance,
    evaluate_field_series,
    freeze_provenance_details,
)

_T, _N = 3, 16


@pytest.fixture(scope="module")
def fields():
    """三幀 16×16 週期方域速度場；pred 與 ref 刻意不同、u 與 v 也不同——這樣
    「拿錯 reference」與「對調通道」都會現形。"""
    r = np.random.RandomState(0)
    u_ref = r.standard_normal((_T, _N, _N))
    v_ref = r.standard_normal((_T, _N, _N)) * 0.7 + 0.2
    u_pred = u_ref + 0.05 * r.standard_normal((_T, _N, _N))
    v_pred = v_ref + 0.05 * r.standard_normal((_T, _N, _N))
    return u_pred, v_pred, u_ref, v_ref


@pytest.fixture(scope="module")
def context():
    return EvaluationContext(
        case="test", times=tuple(float(t) for t in range(_T)),
        periodic=True, domain_length=1.0, viscosity=None,
        pressure_evaluated=False, grid_shape=(_N, _N),
    )


@pytest.fixture(scope="module")
def protocol():
    # 用真的 resolver 產一份協定，讓 provenance 內嵌的形狀與 production 一致。
    return resolve_protocol(mode=ProtocolMode.FOLLOW_TRAINING, training_time_strides=[2])


def _record(recorder, fields, context, protocol, identity, **overrides):
    u_pred, v_pred, u_ref, v_ref = fields
    kwargs = dict(
        context=context, protocol=protocol, identity=identity,
        inputs=(("sensor", "s.json"), ("dns", "d.h5")),
        details_extras={"method": "demo", "re_value": 10000.0},
    )
    kwargs.update(overrides)
    return recorder.record(u_pred, v_pred, u_ref, v_ref, **kwargs)


def _written_artifact(tmp_path, name):
    payload = json.loads((tmp_path / name).read_text())
    return payload


# ── 行為：provenance 帶 producer / code revision / evaluation protocol ──


def test_record_writes_artifact_with_guaranteed_provenance(tmp_path, fields, context, protocol):
    out_path = tmp_path / "demo.json"
    rec = EvaluationRunRecorder("scripts/demo.py", out_path)
    _record(rec, fields, context, protocol,
            RunArtifactIdentity(re=10000.0, method="gappy_pod"))

    payload = _written_artifact(tmp_path, "demo_re10000_gappy_pod.metric_artifact.json")
    prov = payload["provenance"]
    assert prov["producer"] == "scripts/demo.py"
    # code revision 由 recorder 自建構時抓；SourceProvenance 已要求非空。
    assert isinstance(prov["code_revision"], str) and prov["code_revision"].strip()
    assert prov["code_revision"] == rec.code_revision
    assert prov["code_dirty"] == rec.code_dirty
    # recorder 的那道保證：evaluation protocol 一定在 provenance details 裡。
    assert prov["details"]["evaluation_protocol"] == protocol.to_provenance()
    # inputs 原樣落盤。
    assert prov["inputs"] == {"sensor": "s.json", "dns": "d.h5"}


# ── 命名：identity 逐字重現各 producer 的凍結名 ──


@pytest.mark.parametrize(
    "stem, identity, expected",
    [
        # evaluate_baselines / cost_accuracy(interp)：re + method
        ("base", RunArtifactIdentity(re=10000.0, method="gappy_pod"),
         "base_re10000_gappy_pod.metric_artifact.json"),
        # classical_baselines_fair：單-Re，不帶 _re
        ("fair", RunArtifactIdentity(method="rbf_multiquadric"),
         "fair_rbf_multiquadric.metric_artifact.json"),
        # cost_accuracy(gappy)：re + method + modes
        ("cost", RunArtifactIdentity(re=10000.0, method="gappy_pod", modes=25),
         "cost_re10000_gappy_pod_m25.metric_artifact.json"),
        # eval_gappy_cross_re / train_baseline_shred：re only
        ("shred", RunArtifactIdentity(re=10000.0),
         "shred_re10000.metric_artifact.json"),
        # exp245 單筆：裸名，無 stem
        ("anything", RunArtifactIdentity(),
         "metric_artifact.json"),
    ],
)
def test_frozen_filenames(tmp_path, fields, context, protocol, stem, identity, expected):
    rec = EvaluationRunRecorder("scripts/demo.py", tmp_path / f"{stem}.json")
    _record(rec, fields, context, protocol, identity)
    assert (tmp_path / expected).exists(), f"預期凍結名 {expected} 未產出"


# ── 行為：回傳的 projection 與 seam 的 projection 一致 ──


def test_returned_projection_equals_seam(tmp_path, fields, context, protocol):
    rec = EvaluationRunRecorder("scripts/demo.py", tmp_path / "demo.json")
    returned = _record(rec, fields, context, protocol,
                       RunArtifactIdentity(re=10000.0, method="demo"))

    u_pred, v_pred, u_ref, v_ref = fields
    # projection 只由場 + context 決定，與 provenance 無關——故任意合法 provenance 可比。
    _, expected = evaluate_field_series(
        u_pred, v_pred, u_ref, v_ref, context=context,
        provenance=SourceProvenance(
            producer="x", code_revision="deadbeef", code_dirty=False,
            details=freeze_provenance_details({"evaluation_protocol": protocol.to_provenance()}),
        ),
    )
    # 逐字比較。用 canonical JSON dump 而非 dict `==`：projection 的 gamma_k 診斷
    # 列含 NaN（近零能量 shell），而 NaN != NaN 會讓兩份結構相同的 dict 判為不等。
    assert json.dumps(returned, sort_keys=True) == json.dumps(expected, sort_keys=True)


# ── 行為：run_measurement 透傳到寫入 seam（以 cost measurement 落盤）──


def test_run_measurement_travels_to_artifact(tmp_path, fields, context, protocol):
    rec = EvaluationRunRecorder("scripts/demo.py", tmp_path / "cost.json")
    _record(rec, fields, context, protocol,
            RunArtifactIdentity(re=10000.0, method="interp_linear"),
            run_measurements={"recon_s_per_field": 0.125})

    payload = _written_artifact(tmp_path, "cost_re10000_interp_linear.metric_artifact.json")
    run_summaries = [
        s for s in payload["summaries"]
        if s["definition_id"] == "cost.reconstruction.seconds_per_field.v1"
    ]
    assert len(run_summaries) == 1 and run_summaries[0]["value"] == 0.125


# ── 汙染探針：漏傳 protocol / inputs 必須大聲失敗 ──


def test_missing_protocol_raises(tmp_path, fields, context):
    rec = EvaluationRunRecorder("scripts/demo.py", tmp_path / "demo.json")
    u_pred, v_pred, u_ref, v_ref = fields
    # 完全不傳 protocol → 必填 kwarg 缺席。
    with pytest.raises(TypeError):
        rec.record(u_pred, v_pred, u_ref, v_ref, context=context,
                   identity=RunArtifactIdentity(re=10000.0),
                   inputs=(("dns", "d.h5"),), details_extras={})
    # 顯式傳 None → 大聲的 ValueError，不靜默漏蓋。
    with pytest.raises(ValueError):
        rec.record(u_pred, v_pred, u_ref, v_ref, context=context, protocol=None,
                   identity=RunArtifactIdentity(re=10000.0),
                   inputs=(("dns", "d.h5"),), details_extras={})


def test_missing_inputs_raises(tmp_path, fields, context, protocol):
    rec = EvaluationRunRecorder("scripts/demo.py", tmp_path / "demo.json")
    u_pred, v_pred, u_ref, v_ref = fields
    with pytest.raises(TypeError):  # inputs 為必填 kwarg
        rec.record(u_pred, v_pred, u_ref, v_ref, context=context, protocol=protocol,
                   identity=RunArtifactIdentity(re=10000.0), details_extras={})
    with pytest.raises(ValueError):  # 顯式 None
        rec.record(u_pred, v_pred, u_ref, v_ref, context=context, protocol=protocol,
                   identity=RunArtifactIdentity(re=10000.0), inputs=None, details_extras={})


def test_details_extras_cannot_override_protocol(tmp_path, fields, context, protocol):
    """caller 若試圖用 details_extras 提供 evaluation_protocol，保證會被靜默覆蓋——
    recorder 拒收保留鍵，讓那個保證無法被繞過。"""
    rec = EvaluationRunRecorder("scripts/demo.py", tmp_path / "demo.json")
    with pytest.raises(ValueError):
        _record(rec, fields, context, protocol,
                RunArtifactIdentity(re=10000.0, method="demo"),
                details_extras={"evaluation_protocol": {"forged": True}})


# ── 汙染探針：拿錯 reference 必須被 metric 察覺 ──


def test_swapped_reference_changes_projection(tmp_path, fields, context, protocol):
    u_pred, v_pred, u_ref, v_ref = fields
    rec = EvaluationRunRecorder("scripts/demo.py", tmp_path / "demo.json")

    correct = rec.record(u_pred, v_pred, u_ref, v_ref, context=context, protocol=protocol,
                         identity=RunArtifactIdentity(re=10000.0, method="correct"),
                         inputs=(("dns", "d.h5"),), details_extras={})
    # 把 u_ref / v_ref 對調 = 拿錯真值。
    swapped = rec.record(u_pred, v_pred, v_ref, u_ref, context=context, protocol=protocol,
                         identity=RunArtifactIdentity(re=10000.0, method="swapped"),
                         inputs=(("dns", "d.h5"),), details_extras={})

    assert correct["metrics_mean"]["uv_rel_err"] != swapped["metrics_mean"]["uv_rel_err"]

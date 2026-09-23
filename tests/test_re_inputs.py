"""Tests for pi_lnn_jax.data.resolve_re_inputs / parse_re_set。

What: 驗證 (Re, sensor_json, dns_path) 三件組的解析與長度驗證，以及 direct 與
      config 兩條輸入路徑的互斥語意。

Why: 這段解析原本在 cost_accuracy / evaluate_baselines / train_baseline_shred
     逐行重複三份，而 eval_gappy_cross_re 的第四份**完全沒做長度驗證**——直接
     `zip(re_values, sensor_jsons, dns_paths)`，config 裡三個 list 不等長時
     zip 會靜默截短，少評的那幾個 Re 不會有任何訊息。test_length_mismatch_raises
     就是釘死「寧可 fail 也不 silent truncation」。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
from _paths import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pi_lnn_jax.data import parse_re_set, resolve_re_inputs  # noqa: E402


def test_direct_triplet():
    got = resolve_re_inputs(
        re_values="1000,10000",
        sensor_jsons="a.json,b.json",
        dns_paths="a.npy,b.npy",
    )
    assert got == [(1000.0, "a.json", "a.npy"), (10000.0, "b.json", "b.npy")]


def test_direct_tolerates_whitespace_and_trailing_comma():
    got = resolve_re_inputs(
        re_values=" 1000 , ",
        sensor_jsons="a.json,",
        dns_paths=" a.npy",
    )
    assert got == [(1000.0, "a.json", "a.npy")]


@pytest.mark.parametrize(
    "re_values, sensor_jsons, dns_paths",
    [
        ("1000,10000", "a.json", "a.npy,b.npy"),   # sensor 少一個
        ("1000", "a.json,b.json", "a.npy,b.npy"),  # re 少一個
        ("1000,10000", "a.json,b.json", "a.npy"),  # dns 少一個
    ],
)
def test_length_mismatch_raises(re_values, sensor_jsons, dns_paths):
    """三者不等長必須 raise——zip 的靜默截短是本專案明令禁止的失敗模式。"""
    with pytest.raises(ValueError, match="長度需相等"):
        resolve_re_inputs(
            re_values=re_values, sensor_jsons=sensor_jsons, dns_paths=dns_paths,
        )


def test_empty_input_raises():
    with pytest.raises(ValueError, match="未提供輸入"):
        resolve_re_inputs()


def test_config_path(tmp_path=None):
    toml = """
[data_kwargs]
re_values = [1000.0, 10000.0]
sensor_jsons = ["a.json", "b.json"]
dns_paths = ["a.npy", "b.npy"]
"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.toml"
        p.write_text(toml)
        got = resolve_re_inputs(config=str(p))
    assert got == [(1000.0, "a.json", "a.npy"), (10000.0, "b.json", "b.npy")]


def test_config_length_mismatch_raises():
    """config 內三個 list 不等長 — 這正是原 eval_gappy_cross_re 會靜默吃掉的情況。"""
    toml = """
[data_kwargs]
re_values = [1000.0, 10000.0, 1000000.0]
sensor_jsons = ["a.json", "b.json"]
dns_paths = ["a.npy", "b.npy"]
"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.toml"
        p.write_text(toml)
        with pytest.raises(ValueError, match="長度需相等"):
            resolve_re_inputs(config=str(p))


def test_parse_re_set():
    assert parse_re_set("1000, 10000") == {1000.0, 10000.0}
    assert parse_re_set("") == set()
    assert parse_re_set("  ") == set()

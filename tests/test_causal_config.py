"""驗證 TRAIN_SCHEMA 接受 causal weighting keys 並做型別/範圍驗證。

腳本式：uv run python tests/test_causal_config.py
"""
import tempfile
from pathlib import Path

from pi_lnn_jax.config import load_config


def _write(toml_text):
    p = Path(tempfile.mkdtemp()) / "c.toml"
    p.write_text(toml_text)
    return p


def test_causal_keys_loaded():
    cfg = load_config(_write(
        "[train]\nuse_causal_weighting = true\ncausal_eps = 2.0\n"
    ))
    assert cfg["train_kwargs"]["use_causal_weighting"] is True
    assert cfg["train_kwargs"]["causal_eps"] == 2.0


def test_causal_eps_default_when_absent():
    # 未提供時不應出現在 train_kwargs（由 train_full merge 補預設）
    cfg = load_config(_write("[train]\niterations = 10\n"))
    assert "causal_eps" not in cfg["train_kwargs"]
    assert "use_causal_weighting" not in cfg["unknown_keys"]


def test_causal_eps_negative_rejected():
    raised = False
    try:
        load_config(_write("[train]\ncausal_eps = -1.0\n"))
    except ValueError:
        raised = True
    assert raised, "負的 causal_eps 應觸發 ValueError"

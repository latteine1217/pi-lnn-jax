"""回歸守門：import pi_lnn_jax 不得（直接或間接）拉入 torch。"""
from __future__ import annotations

import os
import subprocess
import sys

from _paths import REPO_ROOT


def test_pi_lnn_jax_does_not_require_torch():
    code = (
        "import sys, pi_lnn_jax; "
        "leaked = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.')); "
        "assert not leaked, leaked"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    # returncode != 0 = torch 洩漏（assert 觸發）或 import pi_lnn_jax 本身失敗；不預設根因
    assert r.returncode == 0, f"subprocess 非 0 退出（torch 洩漏或 import 失敗）；stderr:\n{r.stderr}"

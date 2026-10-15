"""W&B 紀錄層（pi_lnn_jax/tracking.py）與其設定鍵。

釘住三件事：
1. 預設關閉，且關閉時訓練路徑根本不 import wandb（驗收 job 與本機不需裝它）。
2. `wandb` / `wandb_project` / `wandb_weights_every` 在兩案都走 TOML < CLI 的既有優先序。
3. 開啟時真的寫出 offline run（不連網），且 step 單調、final 能接在 refine 之後。
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import time

import jax.numpy as jnp
import pytest

from pi_lnn_jax import tracking
from pi_lnn_jax.pipeline.cylinder.config import resolve_inputs as resolve_cylinder
from pi_lnn_jax.pipeline.kolmogorov.config import resolve_inputs as resolve_kolmogorov


def _toml(tmp_path, body: str):
    path = tmp_path / "c.toml"
    path.write_text(textwrap.dedent(body))
    return path


_KOLMO = """
    [train]
    iterations = 100
"""

_CYL = """
    [train]
    case = "cylinder"
    iterations = 20
    [data]
    cylinder_data_npz = "missing.npz"
"""


# ── 1. 預設關閉、不 import wandb ──

def test_training_modules_do_not_import_wandb():
    code = (
        "import sys, pi_lnn_jax.pipeline.kolmogorov.run, pi_lnn_jax.pipeline.cylinder.run;"
        "print('wandb' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "False"


def test_tracker_absent_by_default():
    assert tracking.get_tracker() is None


# ── 2. 設定鍵 ──

def test_kolmogorov_wandb_defaults(tmp_path):
    run = resolve_kolmogorov(["--config", str(_toml(tmp_path, _KOLMO))]).config.run
    assert (run.wandb, run.wandb_project, run.wandb_weights_every) == (False, "pi-con", 500)


def test_kolmogorov_wandb_from_toml_and_cli(tmp_path):
    body = _KOLMO + 'wandb = true\nwandb_project = "other"\nwandb_weights_every = 200\n'
    result = resolve_kolmogorov(["--config", str(_toml(tmp_path, body))])
    run = result.config.run
    assert (run.wandb, run.wandb_project, run.wandb_weights_every) == (True, "other", 200)
    assert result.provenance.for_field("run.wandb").source == "toml"

    cli = resolve_kolmogorov(["--config", str(_toml(tmp_path, _KOLMO)), "--wandb"])
    assert cli.config.run.wandb is True
    assert cli.provenance.for_field("run.wandb").source == "cli"


def test_cylinder_wandb_defaults_and_overrides(tmp_path):
    base = resolve_cylinder(["--config", str(_toml(tmp_path, _CYL)), "--allow-cpu"]).config.run
    assert (base.wandb, base.wandb_project, base.wandb_weights_every) == (False, "pi-con", 500)

    body = _CYL.replace("iterations = 20", "iterations = 20\nwandb_weights_every = 50")
    run = resolve_cylinder(["--config", str(_toml(tmp_path, body)), "--allow-cpu",
                            "--wandb"]).config.run
    assert (run.wandb, run.wandb_weights_every) == (True, 50)


def test_weights_every_must_be_positive(tmp_path):
    body = _KOLMO + "wandb_weights_every = 0\n"
    with pytest.raises(ValueError, match="wandb_weights_every"):
        resolve_kolmogorov(["--config", str(_toml(tmp_path, body))])


# ── 3. 詞彙與純函式 ──

def test_unknown_loss_term_fails_fast():
    with pytest.raises(ValueError, match="ns_u"):
        tracking._check_terms({"data": 1.0, "ns_u": 2.0})


def test_gradnorm_task_order_is_a_prefix_of_terms():
    # run.py 以 TERMS[:n_tasks] 命名 GradNorm 權重；順序錯了權重就會掛到別的 loss 項上
    assert tracking.TERMS[:4] == ("data", "mom_u", "mom_v", "cont")


def test_module_name_takes_top_level_module():
    import jax

    tree = {"params": {"enc": {"Dense_0": {"kernel": jnp.ones((2, 2))}}}}
    (path, _), = jax.tree_util.tree_flatten_with_path(tree)[0]
    assert tracking.module_name(path) == "enc"


def test_training_run_file_missing_is_loud(tmp_path):
    assert not tracking.has_training_run(tmp_path)
    with pytest.raises(FileNotFoundError, match="run.json"):
        tracking.training_run_file(tmp_path)


# ── 4. offline：訓練 run → 評估接回同一 run ──

def test_train_then_attach_eval_offline(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    import json

    monkeypatch.delenv("WANDB_MODE", raising=False)   # 未設時必須預設 offline
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    config = resolve_kolmogorov(["--config", str(_toml(tmp_path, _KOLMO))]).config
    artdir = tmp_path / "exp_x"
    params = {"params": {"enc": {"k": jnp.arange(6.0)}, "dec": {"k": jnp.ones(3)}}}

    tracker = tracking.start(config, case="kolmogorov", name=artdir.name, project="t",
                             wandb_dir=artdir)
    try:
        tracker.log_params(0, params)
        tracker.log_train(25, total=jnp.float32(1.5), losses={"data": 1.0, "cont": 0.5},
                          weights={"data": 1.0}, curriculum={"t_max": 2.0},
                          sec_per_step=0.1, al_lambda=0.3)
        tracker.log_gradnorms(25, {"data": 2.0})
        tracker.log_probe(100, [{k: 0.5 for k in tracking.PROBE_KEYS}])
        tracker.summary({"n_params": 9})
    finally:
        tracking.finish()
    run_id = json.loads((artdir / "wandb" / "run.json").read_text())["id"]

    # offline 片段目錄名只精確到秒；同一秒內 resume 會寫進同一個目錄。實際流程裡評估在
    # 訓練結束後數分鐘才開始，這裡模擬那個間隔。
    time.sleep(1.1)
    ev = tracking.attach_eval(artdir)
    try:
        assert ev.last_step == 100                        # eval 接在訓練最後一列
        with pytest.raises(ValueError, match="長度不一"):
            ev.log_eval(metrics={}, per_t={"t": [0, 1], "e": [1]}, x="t", curves=())
        ev.log_eval(metrics={"uv_rel_err": 0.3}, per_t={"t": [0.0, 1.0], "uv_rel_err": [0.2, 0.4]},
                    x="t", curves=("uv_rel_err",))
    finally:
        tracking.finish()

    assert len(list((artdir / "wandb").glob(f"offline-run-*-{run_id}"))) == 2
    state = json.loads((artdir / "wandb" / "run.json").read_text())
    # 上傳時 resume 片段的 summary 會取代訓練片段的，所以訓練的值必須一路帶著走
    assert state["summary"]["n_params"] == 9 and state["summary"]["loss/total"] == 1.5
    assert state["summary"]["eval/uv_rel_err"] == 0.3
    assert (state["last_step"], state["exit_code"]) == (100, 0)


def test_attach_eval_refuses_unfinished_or_failed_training(tmp_path):
    import json

    (tmp_path / "wandb").mkdir()
    for state in ({"id": "x", "project": "t"},                       # 被 Slurm 砍掉：沒 finish
                  {"id": "x", "project": "t", "last_step": 5, "exit_code": 1, "summary": {}}):
        (tmp_path / "wandb" / "run.json").write_text(json.dumps(state))
        with pytest.raises(RuntimeError, match="沒有正常結束"):
            tracking.attach_eval(tmp_path)


def test_missing_wandb_points_to_setup_script(tmp_path, monkeypatch):
    # sbatch 設 UV_NO_SYNC=1（TD-27），落後的 .venv 不會在 job 內補裝；開了 W&B 就要當場講清楚
    monkeypatch.setitem(sys.modules, "wandb", None)
    config = resolve_kolmogorov(["--config", str(_toml(tmp_path, _KOLMO))]).config
    with pytest.raises(ImportError, match="setup_gpu_venv.sh"):
        tracking.start(config, case="kolmogorov", name="x", project="t", wandb_dir=tmp_path)
    assert tracking.get_tracker() is None

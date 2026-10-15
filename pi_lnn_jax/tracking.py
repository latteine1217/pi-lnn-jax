"""Weights & Biases 訓練紀錄（兩案共用）。

What: 把訓練與評估已經算出來的量鏡像到 W&B。由 `run.wandb` 開關，預設關閉；關閉時
      `get_tracker()` 回 None，呼叫端一律以 `if tracker is not None:` 包住——與
      `pipeline/_ledger.py` 同一模式。

面板分組 —— 依「看圖時要回答的問題」分，兩案共用同一套名字：

    loss/<項>              有沒有在收斂？           每個 log 點
    balance/weight_<項>    各項怎麼被平衡？         每個 log 點
    balance/gradnorm_<項>  （GradNorm 權重、梯度範數、AL 乘子）  GradNorm 觸發時
    balance/al_lambda
    curriculum/*           課程進行到哪？           每個 log 點
    probe/*                訓練中單一時刻對 DNS 的快速檢查（**不是**論文口徑）
    eval/*                 正式評估（Kolmogorov：evaluate_exp245 整段時間序列；
                           cylinder：finalize 全部 eval 時刻）。只寫在 run 的最後一步
    params/*               每個模組的權重 norm 與直方圖   每 wandb_weights_every 步
    perf/sec_per_step      跑多快？
    refine/loss            只在 refinement 啟用時出現

`<項>` 一律取自 `TERMS`：同一個 loss 項在 loss/、weight_、gradnorm_ 三處同名，
cylinder 的 ns_u/ns_v 也記成 mom_u/mom_v。

Why 放在訓練路徑之外的副作用層：`pipeline/` 受 bit-identical 契約約束（CLAUDE.md §7.1）。
    本模組只讀 host 端的既有數值與 params，不 split RNG、不改 state、不寫
    `summary.json`；wandb 只在開啟時才 import，本機與驗收 job 都不需要裝它。

連線：lab-server compute node 無對外網路（job 6029 實測），未設 `WANDB_MODE` 時預設
      offline，run 寫到 `<artifacts_dir>/wandb/`（Kolmogorov）或 `$WANDB_DIR/wandb/`
      （cylinder 不落 artifact，見 test_cylinder_ledger 的 artifacts_dir 守衛），由 head node
      `scripts/slurm/wandb_sync.sh` 上傳。正式評估以 resume 片段接回同一個 run
      （`attach_eval`），上傳時兩片段合併。本模組不讀也不存 API key。
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import numpy as np

#: loss 項的唯一詞彙。GradNorm 的 task 順序是它的前綴（data, mom_u, mom_v[, cont]）。
TERMS = ("data", "mom_u", "mom_v", "cont", "poisson", "bc")

#: probe/ 記的指標：headline（uv_rel_err，見 metric suite 2026-08）加上最常看的幾項。
#: DNS 自身的參考值（ke_dns_mean、div_dns_l2）、索引（t_idx、k_cut）不是模型表現，不記。
PROBE_KEYS = ("uv_rel_err", "ke_rel_err", "omega_rel_err", "div_ratio",
              "low_band_rel_err", "spectrum_rel_err")

#: Kolmogorov 正式評估的 eval/ 指標（取自 evaluate_exp245 的 metrics_mean）。
#: 不收 div_dns_l2（DNS 參考值）、div_pred_l2（div_ratio 已含）、ke_pw_*（與 ke_t_* 同值）。
KOLMOGOROV_EVAL_KEYS = (
    "uv_rel_err", "u_rel_err", "v_rel_err", "ke_rel_err", "omega_rel_err",
    "enstrophy_rel_err", "spectrum_rel_err", "low_band_rel_err",
    "band_rel_err_low", "band_rel_err_mid", "gamma_low", "gamma_mid",
    "div_ratio", "kf_amp_ratio",
)
#: 正式評估中另畫「指標 vs 時間」曲線的三項
KOLMOGOROV_EVAL_CURVES = ("uv_rel_err", "ke_rel_err", "omega_rel_err")

_RUN_FILE = "run.json"   # <wandb_dir>/wandb/run.json：訓練 run 的 id，給 attach_eval 接回


class Tracker:
    """一個 wandb run 的薄包裝；所有值在這裡轉成 host 端 Python/NumPy 型別。"""

    def __init__(self, run, *, last_step: int = 0, summary: Mapping[str, Any] | None = None,
                 state_file: Path | None = None) -> None:
        self._run = run
        #: 已記過的最大 step。wandb 丟棄比它小的 step；refine 與 eval 以此接在最後。
        self.last_step = int(last_step)
        #: summary 的 scalar 鏡像（每個鍵的最後值），finish 時連同 last_step 寫進 state_file，
        #: 供 attach_eval 接回。不讀 wandb 的內部檔：offline 片段 finish 後不保證有
        #: files/wandb-summary.json（2026-09-28 本機實測沒有）。
        self._summary: dict[str, Any] = dict(summary or {})
        self._state_file = state_file
        for k, v in self._summary.items():
            self._run.summary[k] = v

    def _log(self, payload: Mapping[str, Any], step: int) -> None:
        self._run.log(dict(payload), step=int(step))
        self.last_step = max(self.last_step, int(step))
        self._summary.update({k: v for k, v in payload.items()
                              if isinstance(v, (int, float, bool))})

    def log_train(self, step: int, *, total: Any, losses: Mapping[str, Any],
                  weights: Mapping[str, Any], curriculum: Mapping[str, Any],
                  sec_per_step: float, al_lambda: Any = None,
                  al_constraint: Any = None) -> None:
        """一個 log 點：loss/、balance/weight_、curriculum/、perf/。

        `al_constraint` 只在它與 loss/cont 不同時傳（signed_mean 模式）；mse 模式下兩者恆等。
        """
        _check_terms(losses, weights)
        payload = {"loss/total": _to_host(total),
                   **{f"loss/{k}": _to_host(v) for k, v in losses.items()}}
        payload.update({f"balance/weight_{k}": _to_host(v) for k, v in weights.items()})
        payload.update({f"curriculum/{k}": _to_host(v) for k, v in curriculum.items()})
        payload["perf/sec_per_step"] = float(sec_per_step)
        if al_lambda is not None:
            payload["balance/al_lambda"] = _to_host(al_lambda)
        if al_constraint is not None:
            payload["balance/al_constraint"] = _to_host(al_constraint)
        self._log(payload, step)

    def log_gradnorms(self, step: int, norms: Mapping[str, Any]) -> None:
        _check_terms(norms)
        self._log({f"balance/gradnorm_{k}": _to_host(v) for k, v in norms.items()}, step)

    def log_probe(self, step: int, results: list[Mapping[str, Any]]) -> None:
        """`evaluate_against_dns` 的逐 time-slice 結果 → probe/（多個 slice 時取平均）。"""
        self._log({f"probe/{k}": float(np.mean([_to_host(r[k]) for r in results]))
                   for k in PROBE_KEYS}, step)

    def log_eval(self, *, metrics: Mapping[str, Any], per_t: Mapping[str, list],
                 x: str, curves: tuple[str, ...]) -> None:
        """正式評估：scalar 進 eval/ 與 summary，逐時刻序列成表並畫 `curves` 的曲線。

        記在 `last_step`——評估對應的是訓練最後的參數，放在最後一列才不會畫成訓練中途的點。
        """
        import wandb

        scalars = {f"eval/{k}": _to_host(v) for k, v in metrics.items()}
        names = list(per_t)
        n = len(per_t[x])
        if any(len(col) != n for col in per_t.values()):
            raise ValueError(f"per_t 欄位長度不一：{ {k: len(v) for k, v in per_t.items()} }")
        table = wandb.Table(columns=names,
                            data=[[_to_host(per_t[c][i]) for c in names] for i in range(n)])
        plots = {f"eval/{k}_vs_{x}": wandb.plot.line(table, x, k, title=f"{k} vs {x}")
                 for k in curves}
        self._log({**scalars, **plots, "eval/per_t": table}, self.last_step)
        self.summary(scalars)

    def log_params(self, step: int, params) -> None:
        """每個頂層模組一條 L2 norm 與一張直方圖，外加全體 norm。"""
        import wandb

        groups: dict[str, list[np.ndarray]] = {}
        for path, leaf in jax.tree_util.tree_flatten_with_path(jax.device_get(params))[0]:
            groups.setdefault(module_name(path), []).append(
                np.asarray(leaf, dtype=np.float64).ravel())
        payload: dict[str, Any] = {}
        sq_total = 0.0
        for mod, arrs in sorted(groups.items()):
            flat = np.concatenate(arrs)
            sq = float(flat @ flat)
            sq_total += sq
            payload[f"params/norm_{mod}"] = sq ** 0.5
            payload[f"params/hist_{mod}"] = wandb.Histogram(flat)
        payload["params/norm_total"] = sq_total ** 0.5
        self._log(payload, step)

    def log_refine(self, step: int, loss: Any) -> None:
        self._log({"refine/loss": _to_host(loss)}, step)

    def summary(self, metrics: Mapping[str, Any]) -> None:
        for k, v in metrics.items():
            self._run.summary[k] = self._summary[k] = _to_host(v)

    def save_file(self, path: Path) -> None:
        """附上產物檔；不存在就略過——它們是否存在由產生端負責。"""
        if Path(path).exists():
            self._run.save(str(path), base_path=str(Path(path).parent), policy="now")

    def finish(self, exit_code: int = 0) -> None:
        self._run.finish(exit_code=exit_code)
        if self._state_file is not None:
            state = json.loads(self._state_file.read_text())
            state.update(last_step=self.last_step, exit_code=exit_code, summary=self._summary)
            self._state_file.write_text(json.dumps(state, indent=1))


_TRACKER: Tracker | None = None


def get_tracker() -> Tracker | None:
    """啟用時回傳 process-wide Tracker，否則 None。"""
    return _TRACKER


def start(config, *, case: str, name: str, project: str,
          wandb_dir: str | Path | None = None,
          extra: Mapping[str, Any] | None = None) -> Tracker:
    """開一個訓練 run 並設為 process-wide tracker。

    `name` 是實驗識別（Kolmogorov 用 artifacts_dir 目錄名、cylinder 用 config 檔名）；
    run 名稱另附 Slurm job id。resume 訓練會開新 run、以同一 `group` 串起來，避免 offline
    模式下對同一 run id 續寫的 step 衝突。`wandb_dir=None` 時由 wandb 依 `WANDB_DIR`
    （未設則 cwd）決定落點；有給時另寫 run.json 供 `attach_eval` 接回。
    """
    global _TRACKER
    wandb = _import_wandb()

    if wandb_dir is not None:
        Path(wandb_dir).mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID")
    run = wandb.init(
        mode=_mode(),
        project=project,
        dir=None if wandb_dir is None else str(wandb_dir),
        name=name + (f"-{job_id}" if job_id else ""),
        group=name,
        job_type=case,
        tags=[case],
        config={**config_to_dict(config), "_meta": {
            "case": case, **_git_state(), "slurm_job_id": job_id,
            **(dict(extra) if extra else {}),
        }},
    )
    state_file = None
    if wandb_dir is not None:
        state_file = Path(wandb_dir) / "wandb" / _RUN_FILE
        state_file.write_text(json.dumps({"id": run.id, "project": project}))
    _TRACKER = Tracker(run, state_file=state_file)
    return _TRACKER


def attach_eval(wandb_dir: str | Path) -> Tracker:
    """以 resume 片段接回 `wandb_dir` 裡訓練開的 run，設為 process-wide tracker。

    上傳時 resume 片段的 summary 會**整份取代**訓練片段的（2026-09-28 實測），所以把訓練
    結束時寫進 run.json 的 summary 抄過來；`last_step` 設成訓練最後一步，eval 記在同一列。
    """
    global _TRACKER
    wandb = _import_wandb()

    state_file = training_run_file(wandb_dir)
    state = json.loads(state_file.read_text())
    if state.get("exit_code") != 0:
        raise RuntimeError(
            f"{state_file}：訓練 run 沒有正常結束（exit_code={state.get('exit_code')}；"
            "缺值表示 finish 未執行，例如被 Slurm 砍掉），不該對它做正式評估。")
    run = wandb.init(mode=_mode(), project=state["project"], id=state["id"],
                     resume="allow", dir=str(wandb_dir))
    _TRACKER = Tracker(run, last_step=state["last_step"], summary=state["summary"],
                       state_file=state_file)
    return _TRACKER


def has_training_run(wandb_dir: str | Path) -> bool:
    return (Path(wandb_dir) / "wandb" / _RUN_FILE).exists()


def training_run_file(wandb_dir: str | Path) -> Path:
    """訓練 run 的 run.json；不存在就大聲失敗（訓練端沒開 W&B 時不該要求接回）。"""
    run_file = Path(wandb_dir) / "wandb" / _RUN_FILE
    if not run_file.exists():
        raise FileNotFoundError(
            f"{run_file} 不存在：訓練端沒有開 W&B（run.wandb），無從接回。")
    return run_file


def finish(exit_code: int = 0) -> None:
    global _TRACKER
    if _TRACKER is not None:
        _TRACKER.finish(exit_code)
    _TRACKER = None


def config_to_dict(config) -> dict:
    """typed EffectiveConfig → 巢狀 dict（ModelConfig 走其唯一出口 `to_kwargs()`）。"""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(config):
        group = getattr(config, f.name)
        if dataclasses.is_dataclass(group) and not hasattr(group, "to_kwargs"):
            out[f.name] = dataclasses.asdict(group)
        else:
            out[f.name] = group.to_kwargs()
    return out


def module_name(path) -> str:
    """flax params 路徑 → 頂層模組名（'params' 根之下的第一段）。"""
    parts = [str(getattr(k, "key", getattr(k, "idx", k))) for k in path]
    if parts and parts[0] == "params":
        parts = parts[1:]
    return parts[0] if parts else "root"


def _check_terms(*groups: Mapping[str, Any]) -> None:
    unknown = sorted({k for g in groups for k in g} - set(TERMS))
    if unknown:
        raise ValueError(f"未登記的 loss 項 {unknown}；詞彙見 tracking.TERMS")


def _import_wandb():
    """run 的兩個入口在這裡 import，缺套件時在訓練開始前就指出修法。

    sbatch 設 UV_NO_SYNC=1（TD-27），所以 `.venv` 落後 uv.lock 時不會在 job 內補裝；
    沒重跑 setup_gpu_venv.sh 的 worktree 只在開了 W&B 時才會走到這裡。
    """
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "run.wandb 已開啟但 .venv 沒有 wandb：這個 worktree 在拉到 ea8f8d0 之後沒重跑 "
            "scripts/slurm/setup_gpu_venv.sh（在 head node 跑，compute node 離線裝不了）。"
        ) from exc
    return wandb


def _mode() -> str:
    # compute node 無對外 DNS，online init 會卡到逾時後讓訓練 crash。未明設時一律 offline。
    return os.environ.get("WANDB_MODE", "offline")


def _to_host(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    arr = np.asarray(v)
    return arr.item() if arr.ndim == 0 else arr.tolist()


def _git_state() -> dict[str, Any]:
    """HEAD sha 與工作區是否有未 commit 改動——只記 sha 會把 patch 過的跑誤認成該 commit。"""
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True,
                cwd=Path(__file__).resolve().parent, check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_sha": git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status)}

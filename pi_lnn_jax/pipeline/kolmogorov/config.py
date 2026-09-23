"""Kolmogorov configuration policy for the shared resolution engine."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Mapping

from pi_lnn_jax.config import (
    DATA_SCHEMA,
    MODEL_SCHEMA,
    ConfigurationLayer,
    ModelConfig,
    ResolutionResult,
    flatten_toml,
    resolve_config,
)

ARCH_CHOICES = ("liquid", "vanilla", "pinn")
OPT_CHOICES = ("adam", "soap", "schedule_free")


@dataclass(frozen=True)
class KolmogorovDataConfig:
    sensor_jsons: list
    sensor_npzs: list
    dns_paths: list
    re_values: list
    time_strides: list
    re_norm_scale: float
    observed_sensor_channels: list
    kolmogorov_k_f: float
    kolmogorov_A: float
    cylinder_data_npz: str
    cylinder_u_inf: float
    controlled_data_npz: str


@dataclass(frozen=True)
class KolmogorovRunConfig:
    config_path: str
    arch: str
    optimizer: str
    base_optimizer: str
    soap_precondition_frequency: int
    soap_b1: float
    soap_b2: float
    resume_step: str | None
    steps: int
    seed: int
    artifacts_dir: str
    learning_rate: float
    max_grad_norm: float
    save_every: int
    log_every: int
    eval_every: int
    multi_re: bool


@dataclass(frozen=True)
class KolmogorovLossConfig:
    data_weight: float
    physics_weight: float
    poisson_weight: float
    gauge_weight: float
    sensor_channel_weights: list
    physics_warmup_steps: int
    physics_ramp_steps: int
    use_gradnorm: bool
    gradnorm_freq: int
    gradnorm_min: float
    gradnorm_max: float
    gradnorm_init_weights: list
    gradnorm_ema_momentum: float
    use_al: bool
    cont_gradnorm: bool
    al_rho: float
    al_lambda_clip: float
    al_update_freq: int
    al_constraint_mode: str
    al_warmup_steps: int
    use_continuous_re_physics: bool
    continuous_re_physics_weight: float
    use_causal: bool
    causal_eps: float
    t_early_weight: float
    t_early_threshold: float


@dataclass(frozen=True)
class KolmogorovCurriculumConfig:
    n_sensor_query_requested: int
    n_collo_start: int
    n_collo_end: int
    n_collo_ramp: int
    #: physics collocation 的時間上界；0.0 = 沿用資料時窗（見 TRAIN_SCHEMA 註解）
    physics_t_max: float
    #: 訓練時 decoder 可見 sensor 序列的最短比例；0.0 = 不截斷（見 TRAIN_SCHEMA 註解）
    sensor_cut_min_frac: float
    #: branch 自迴歸的 pseudo 幀間距與輪數；dt=0.0 = 關（見 TRAIN_SCHEMA 註解）
    autoreg_pseudo_dt: float
    autoreg_rounds: int
    grad_accum_chunks: int
    use_time_marching: bool
    tm_start_frac: float
    tm_warmup_steps: int
    tm_ramp_steps: int
    sensor_dropout_rate: float
    rar_freq: int
    rar_warmup: int
    rar_pool_size: int
    rar_exploration_ratio: float


@dataclass(frozen=True)
class KolmogorovOptimizerSchedule:
    warmup_steps: int
    decay_steps: int
    decay_gamma: float
    min_learning_rate: float


@dataclass(frozen=True)
class KolmogorovRefinementConfig:
    optimizer: str
    steps: int
    rtol: float
    atol: float
    n_collo: int | None
    lbfgs_t_subsample: int
    lbfgs_max_iter: int
    lbfgs_history: int
    gn_lr: float
    gn_cg_iters: int
    gn_damping: float
    gn_log_every: int


@dataclass(frozen=True)
class KolmogorovEffectiveConfig:
    model: ModelConfig
    data: KolmogorovDataConfig
    run: KolmogorovRunConfig
    loss: KolmogorovLossConfig
    curriculum: KolmogorovCurriculumConfig
    schedule: KolmogorovOptimizerSchedule
    refinement: KolmogorovRefinementConfig


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_kolmogorov.py",
        description="Production train script for pi-lnn JAX/Flax POC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True, help="TOML config 路徑")
    p.add_argument("--steps", type=int, default=None, help="覆蓋 TOML iterations")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--artifacts_dir", type=str, default=None)
    p.add_argument("--resume_step", type=str, default=None,
                   help="ckpt step 號 ('latest' 或 int)；None=從 scratch")
    p.add_argument("--arch", choices=ARCH_CHOICES, default="liquid",
                   help="liquid=B3 / vanilla=B0 / pinn=B2")
    p.add_argument("--optimizer", choices=OPT_CHOICES, default="adam")
    p.add_argument("--base_optimizer", choices=["adam", "soap"], default="adam",
                   help="schedule_free 模式下的 inner optimizer (pi-lnn EXP-245 用 soap)")
    p.add_argument("--soap_precondition_frequency", type=int, default=10,
                   help="SOAP preconditioner 更新頻率（pi-lnn 用 2）")
    p.add_argument("--soap_b1", type=float, default=None,
                   help="SOAP beta1；None=從 TOML soap_betas[0] 或預設 0.95（pi-lnn 用 0.9）")
    p.add_argument("--soap_b2", type=float, default=None,
                   help="SOAP beta2；None=從 TOML soap_betas[1] 或預設 0.95（pi-lnn 用 0.999）")
    p.add_argument("--use_gradnorm", action="store_true", help="啟用 GradNorm 動態權重")
    p.add_argument("--cont_gradnorm", action="store_true",
                   help="continuity 改由 GradNorm 加權（第四個 task）而非 AL；同時關閉 AL")
    p.add_argument("--use_al", action="store_true",
                   help="啟用 Augmented Lagrangian for continuity")
    p.add_argument("--ablate_no_al", action="store_true",
                   help="ablation：關閉 AL continuity 約束（continuity 完全無約束，div 對照用）")
    p.add_argument("--w_poisson", type=float, default=None,
                   help="pressure-Poisson loss weight (0=disabled)")
    p.add_argument("--use_time_marching", action="store_true")
    p.add_argument("--use_causal_weighting", action="store_true",
                   help="啟用 Wang2022 causal weighting（時間因果加權 physics 殘差）")
    p.add_argument("--causal_eps", type=float, default=None,
                   help="causal weighting 強度 eps（>=0；預設讀 TOML causal_eps）")
    p.add_argument("--tm_start_frac", type=float, default=0.5,
                   help="time marching 初始 t_max 比例")
    p.add_argument("--tm_warmup_steps", type=int, default=500)
    p.add_argument("--tm_ramp_steps", type=int, default=2500)
    p.add_argument("--n_collo_start", type=int, default=None,
                   help="curriculum start point count (default = TOML num_physics_points)")
    p.add_argument("--n_collo_end", type=int, default=None,
                   help="curriculum end point count")
    p.add_argument("--n_collo_ramp", type=int, default=1000,
                   help="n_collo 線性 ramp 步數")
    p.add_argument("--n_sensor_query", type=int, default=None,
                   help="sensor mini-batch：每 step decode 的 query 點數（0/None=full T*K）。"
                        "對齊 pi-lnn sample_sensor_batch；encode 仍用全量（Wave 4 encode-once）。")
    p.add_argument("--sensor_dropout_rate", type=float, default=0.0,
                   help="實驗2 train-time sensor dropout 比例（0=停用，行為 bit-identical）。"
                        "denoising 語意：每 step 隨機 zero-mask round(rate*K) 個 sensor 的『輸入』，"
                        "data-loss target 仍用未 mask 的真值（強迫從剩餘 sensor 推斷缺失位置）。")
    p.add_argument("--rar_freq", type=int, default=0,
                   help="RAR sample 頻率（步）；0=disabled")
    p.add_argument("--rar_warmup", type=int, default=500,
                   help="RAR 開始前的 warmup 步數")
    p.add_argument("--rar_pool_size", type=int, default=512)
    p.add_argument("--eval_every", type=int, default=0,
                   help="0=訓練結束才 eval；>0 每 N 步 eval 一次")
    p.add_argument("--save_every", type=int, default=None,
                   help="覆蓋 TOML checkpoint_period")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--multi_re", action="store_true",
                   help="從 TOML data_kwargs.re_values 載入多 Re datasets")
    p.add_argument("--refine_optimizer", choices=["none", "lbfgs", "lm", "gn"], default="none",
                   help="主訓練結束後啟動 refinement phase: lbfgs=optimistix L-BFGS scalar minimise; "
                        "lm=Levenberg-Marquardt nonlinear least-squares")
    p.add_argument("--refine_steps", type=int, default=100,
                   help="refinement phase max optimistix iterations")
    p.add_argument("--refine_rtol", type=float, default=1e-7)
    p.add_argument("--refine_atol", type=float, default=1e-7)
    p.add_argument("--n_collo_refine", type=int, default=None,
                   help="LBFGS/LM refine phase 用的 collocation point 數；None=沿用 n_collo_end。")
    p.add_argument("--lbfgs_t_subsample", type=int, default=10,
                   help="LBFGS refine: sample T 個時間段做 sensor mini-batch；"
                        "對齊 pi-lnn 用 mini-batch sample。預設 10 → sensor query 大小降至 10×K=1000")
    p.add_argument("--lbfgs_max_iter", type=int, default=20,
                   help="LBFGS per outer step max inner iter (對齊 pi-lnn torch.optim.LBFGS max_iter=20)")
    p.add_argument("--lbfgs_history", type=int, default=10,
                   help="LBFGS history length (對齊 pi-lnn history_size=10)")
    p.add_argument("--gn_lr", type=float, default=1.0,
                   help="GN step size（純二階 lr=1.0；不穩定時降 0.1-0.5）")
    p.add_argument("--gn_cg_iters", type=int, default=20,
                   help="每步 CG inner iterations")
    p.add_argument("--gn_damping", type=float, default=1e-3,
                   help="Levenberg-Marquardt damping λ（G + λI）")
    p.add_argument("--gn_log_every", type=int, default=50,
                   help="GN refine 每 N 步印 loss")
    return p


def _provided_options(raw: list[str]) -> frozenset[str]:
    return frozenset(token.split("=", 1)[0] for token in raw if token.startswith("--"))


class KolmogorovPolicy:
    _defaults = {
        "run.config_path": None,
        "run.arch": "liquid", "run.optimizer": "adam", "run.base_optimizer": "adam",
        "run.soap_precondition_frequency": 10, "run.soap_b1": 0.95, "run.soap_b2": 0.95,
        "run.resume_step": None, "run.steps": 5000, "run.seed": 42,
        "run.artifacts_dir": "artifacts/run", "run.learning_rate": 3e-3,
        "run.max_grad_norm": 1.0, "run.save_every": 100, "run.log_every": 25,
        "run.eval_every": 0, "run.multi_re": False,
        "loss.data_weight": 1.0, "loss.physics_weight": 0.01,
        "loss.poisson_weight": 0.0, "loss.gauge_weight": 0.0,
        "loss.sensor_channel_weights": [], "loss.physics_warmup_steps": 0,
        "loss.physics_ramp_steps": 0, "loss.use_gradnorm": False,
        "loss.gradnorm_freq": 1000, "loss.gradnorm_min": 0.05,
        "loss.gradnorm_max": 0.0, "loss.gradnorm_init_weights": [1.0, 0.01, 0.01],
        "loss.gradnorm_ema_momentum": 0.9, "loss.use_al": True,
        "loss.cont_gradnorm": False,
        "loss.al_rho": 1.0, "loss.al_lambda_clip": 10.0,
        "loss.al_update_freq": 100, "loss.al_constraint_mode": "mse",
        "loss.al_warmup_steps": 0, "loss.use_continuous_re_physics": False,
        "loss.continuous_re_physics_weight": 1.0, "loss.use_causal": False,
        "loss.causal_eps": 1.0, "loss.t_early_weight": 1.0,
        "loss.t_early_threshold": 0.05,
        "curriculum.n_sensor_query_requested": 0, "curriculum.n_collo_start": 32,
        "curriculum.n_collo_end": 32, "curriculum.n_collo_ramp": 1000,
        "curriculum.physics_t_max": 0.0,
        "curriculum.sensor_cut_min_frac": 0.0,
        "curriculum.autoreg_pseudo_dt": 0.0, "curriculum.autoreg_rounds": 1,
        "curriculum.grad_accum_chunks": 1, "curriculum.use_time_marching": False,
        "curriculum.tm_start_frac": 0.5, "curriculum.tm_warmup_steps": 500,
        "curriculum.tm_ramp_steps": 2500, "curriculum.sensor_dropout_rate": 0.0,
        "curriculum.rar_freq": 0, "curriculum.rar_warmup": 500,
        "curriculum.rar_pool_size": 512,
        "curriculum.rar_exploration_ratio": 0.2,
        "schedule.warmup_steps": 0, "schedule.decay_steps": 0,
        "schedule.decay_gamma": 1.0, "schedule.min_learning_rate": 1e-6,
        "refinement.optimizer": "none", "refinement.steps": 100,
        "refinement.rtol": 1e-7, "refinement.atol": 1e-7,
        "refinement.n_collo": None, "refinement.lbfgs_t_subsample": 10,
        "refinement.lbfgs_max_iter": 20, "refinement.lbfgs_history": 10,
        "refinement.gn_lr": 1.0, "refinement.gn_cg_iters": 20,
        "refinement.gn_damping": 1e-3, "refinement.gn_log_every": 50,
    }
    _defaults.update({f"model.{key}": spec[1] for key, spec in MODEL_SCHEMA.items()})
    _defaults.update({f"data.{key}": spec[1] for key, spec in DATA_SCHEMA.items()})
    known_fields = frozenset(_defaults)

    def parse_cli(self, argv: list[str] | None) -> argparse.Namespace:
        raw = list(sys.argv[1:] if argv is None else argv)
        cli = _parser().parse_args(raw)
        cli._provided = _provided_options(raw)
        return cli

    def layers(self, cli: argparse.Namespace, loaded: dict) -> tuple[ConfigurationLayer, ...]:
        tk, dk = loaded["train_kwargs"], loaded["data_kwargs"]
        flat = flatten_toml(loaded["raw"])
        toml: dict[str, Any] = {}
        model_keys = set(flat) & set(MODEL_SCHEMA)
        for key in model_keys:
            toml[f"model.{key}"] = loaded["model_kwargs"][key]
        if "sensor_value_dim" not in model_keys and "observed_sensor_channels" in dk:
            toml["model.sensor_value_dim"] = loaded["model_kwargs"]["sensor_value_dim"]
        for key, value in dk.items():
            toml[f"data.{key}"] = value
        direct_train = {
            "iterations": "run.steps", "seed": "run.seed",
            "artifacts_dir": "run.artifacts_dir", "learning_rate": "run.learning_rate",
            "max_grad_norm": "run.max_grad_norm", "checkpoint_period": "run.save_every",
            "data_loss_weight": "loss.data_weight", "physics_loss_weight": "loss.physics_weight",
            "poisson_loss_weight": "loss.poisson_weight", "gauge_loss_weight": "loss.gauge_weight",
            "sensor_channel_weights": "loss.sensor_channel_weights",
            "num_sensor_query_points": "curriculum.n_sensor_query_requested",
            "num_physics_points": "curriculum.n_collo_start",
            "physics_t_max": "curriculum.physics_t_max",
            "sensor_cut_min_frac": "curriculum.sensor_cut_min_frac",
            "autoreg_pseudo_dt": "curriculum.autoreg_pseudo_dt",
            "autoreg_rounds": "curriculum.autoreg_rounds",
            "grad_accum_chunks": "curriculum.grad_accum_chunks",
            "rar_freq": "curriculum.rar_freq", "rar_warmup": "curriculum.rar_warmup",
            "rar_pool_size": "curriculum.rar_pool_size",
            "rar_exploration_ratio": "curriculum.rar_exploration_ratio",
            "use_gradnorm": "loss.use_gradnorm", "gradnorm_freq": "loss.gradnorm_freq",
            "gradnorm_min_weight": "loss.gradnorm_min", "gradnorm_max_weight": "loss.gradnorm_max",
            "al_rho": "loss.al_rho", "al_lambda_clip": "loss.al_lambda_clip",
            "use_continuous_re_physics": "loss.use_continuous_re_physics",
            "continuous_re_physics_weight": "loss.continuous_re_physics_weight",
            "use_causal_weighting": "loss.use_causal", "causal_eps": "loss.causal_eps",
        }
        for key, path in direct_train.items():
            if key in tk:
                toml[path] = tk[key]
        if "num_physics_points" in tk:
            toml["curriculum.n_collo_end"] = tk["num_physics_points"]
        raw_train = {
            "physics_loss_warmup_steps": "loss.physics_warmup_steps",
            "physics_loss_ramp_steps": "loss.physics_ramp_steps",
            "lr_warmup_steps": "schedule.warmup_steps", "lr_decay_steps": "schedule.decay_steps",
            "lr_decay_gamma": "schedule.decay_gamma", "min_learning_rate": "schedule.min_learning_rate",
            "gradnorm_init_weights": "loss.gradnorm_init_weights",
            "gradnorm_ema_momentum": "loss.gradnorm_ema_momentum",
            "al_update_freq": "loss.al_update_freq", "al_constraint_mode": "loss.al_constraint_mode",
            "al_warmup_steps": "loss.al_warmup_steps", "t_early_weight": "loss.t_early_weight",
            "t_early_threshold": "loss.t_early_threshold",
        }
        for key, path in raw_train.items():
            if key in flat:
                toml[path] = flat[key]
        if flat.get("soap_betas"):
            toml["run.soap_b1"] = float(flat["soap_betas"][0])
            toml["run.soap_b2"] = float(flat["soap_betas"][1])

        cli_values: dict[str, Any] = {"run.config_path": cli.config}
        option_map = {
            "--steps": ("run.steps", cli.steps), "--seed": ("run.seed", cli.seed),
            "--artifacts_dir": ("run.artifacts_dir", cli.artifacts_dir),
            "--resume_step": ("run.resume_step", cli.resume_step), "--arch": ("run.arch", cli.arch),
            "--optimizer": ("run.optimizer", cli.optimizer),
            "--base_optimizer": ("run.base_optimizer", cli.base_optimizer),
            "--soap_precondition_frequency": ("run.soap_precondition_frequency", cli.soap_precondition_frequency),
            "--soap_b1": ("run.soap_b1", cli.soap_b1), "--soap_b2": ("run.soap_b2", cli.soap_b2),
            "--w_poisson": ("loss.poisson_weight", cli.w_poisson),
            "--causal_eps": ("loss.causal_eps", cli.causal_eps),
            "--tm_start_frac": ("curriculum.tm_start_frac", cli.tm_start_frac),
            "--tm_warmup_steps": ("curriculum.tm_warmup_steps", cli.tm_warmup_steps),
            "--tm_ramp_steps": ("curriculum.tm_ramp_steps", cli.tm_ramp_steps),
            "--n_collo_start": ("curriculum.n_collo_start", cli.n_collo_start),
            "--n_collo_end": ("curriculum.n_collo_end", cli.n_collo_end),
            "--n_collo_ramp": ("curriculum.n_collo_ramp", cli.n_collo_ramp),
            "--n_sensor_query": ("curriculum.n_sensor_query_requested", cli.n_sensor_query),
            "--sensor_dropout_rate": ("curriculum.sensor_dropout_rate", cli.sensor_dropout_rate),
            "--rar_freq": ("curriculum.rar_freq", cli.rar_freq),
            "--rar_warmup": ("curriculum.rar_warmup", cli.rar_warmup),
            "--rar_pool_size": ("curriculum.rar_pool_size", cli.rar_pool_size),
            "--eval_every": ("run.eval_every", cli.eval_every),
            "--save_every": ("run.save_every", cli.save_every),
            "--log_every": ("run.log_every", cli.log_every),
            "--refine_optimizer": ("refinement.optimizer", cli.refine_optimizer),
            "--refine_steps": ("refinement.steps", cli.refine_steps),
            "--refine_rtol": ("refinement.rtol", cli.refine_rtol),
            "--refine_atol": ("refinement.atol", cli.refine_atol),
            "--n_collo_refine": ("refinement.n_collo", cli.n_collo_refine),
            "--lbfgs_t_subsample": ("refinement.lbfgs_t_subsample", cli.lbfgs_t_subsample),
            "--lbfgs_max_iter": ("refinement.lbfgs_max_iter", cli.lbfgs_max_iter),
            "--lbfgs_history": ("refinement.lbfgs_history", cli.lbfgs_history),
            "--gn_lr": ("refinement.gn_lr", cli.gn_lr),
            "--gn_cg_iters": ("refinement.gn_cg_iters", cli.gn_cg_iters),
            "--gn_damping": ("refinement.gn_damping", cli.gn_damping),
            "--gn_log_every": ("refinement.gn_log_every", cli.gn_log_every),
        }
        for option, (path, value) in option_map.items():
            if option in cli._provided:
                cli_values[path] = value
        if cli.use_gradnorm:
            cli_values["loss.use_gradnorm"] = True
        if cli.ablate_no_al:
            cli_values["loss.use_al"] = False
        if cli.cont_gradnorm:
            # continuity 走 GradNorm 第四個 task；AL 同時關閉，兩者互斥
            cli_values["loss.cont_gradnorm"] = True
            cli_values["loss.use_al"] = False
        if cli.use_time_marching:
            cli_values["curriculum.use_time_marching"] = True
        if cli.use_causal_weighting:
            cli_values["loss.use_causal"] = True
        if cli.multi_re:
            cli_values["run.multi_re"] = True
        return (
            ConfigurationLayer("schema_default", self._defaults),
            ConfigurationLayer("toml", toml),
            ConfigurationLayer("cli", cli_values),
        )

    def pre_load_guard(self, cli: argparse.Namespace) -> None:
        del cli

    def build(self, v: Mapping[str, Any]) -> KolmogorovEffectiveConfig:
        model = ModelConfig({key: v[f"model.{key}"] for key in MODEL_SCHEMA})
        data = KolmogorovDataConfig(**{key: v[f"data.{key}"] for key in DATA_SCHEMA})
        return KolmogorovEffectiveConfig(
            model=model,
            data=data,
            run=KolmogorovRunConfig(
                config_path=v["run.config_path"], arch=v["run.arch"], optimizer=v["run.optimizer"],
                base_optimizer=v["run.base_optimizer"],
                soap_precondition_frequency=v["run.soap_precondition_frequency"],
                soap_b1=v["run.soap_b1"], soap_b2=v["run.soap_b2"], resume_step=v["run.resume_step"],
                steps=v["run.steps"], seed=v["run.seed"], artifacts_dir=v["run.artifacts_dir"],
                learning_rate=v["run.learning_rate"], max_grad_norm=v["run.max_grad_norm"],
                save_every=v["run.save_every"], log_every=max(1, v["run.log_every"]),
                eval_every=max(0, v["run.eval_every"]), multi_re=v["run.multi_re"],
            ),
            loss=KolmogorovLossConfig(**{
                field: v[f"loss.{field}"] for field in KolmogorovLossConfig.__dataclass_fields__
            }),
            curriculum=KolmogorovCurriculumConfig(**{
                field: v[f"curriculum.{field}"] for field in KolmogorovCurriculumConfig.__dataclass_fields__
            }),
            schedule=KolmogorovOptimizerSchedule(**{
                field: v[f"schedule.{field}"] for field in KolmogorovOptimizerSchedule.__dataclass_fields__
            }),
            refinement=KolmogorovRefinementConfig(**{
                field: v[f"refinement.{field}"] for field in KolmogorovRefinementConfig.__dataclass_fields__
            }),
        )

    def guard(self, cli: argparse.Namespace, config: KolmogorovEffectiveConfig) -> None:
        del cli
        # cont_gradnorm 的語意就是「continuity 當第四個 GradNorm task」。少了 GradNorm，
        # task_weights 退化成全 1，continuity 會以權重 1.0（gradnorm_min_weight 的 20 倍）
        # 進 loss，而那個臂裡根本沒有 GradNorm——看起來會成功，測的卻是別的東西。
        # data_loss_weight 只在 refinement 路徑被讀（run.py:1033/1087/1119）；主訓練
        # loss 用的是 task_weights[0]，而 GradNorm 令 w_computed[0] = w_raw[0]/w_raw[0]
        # 恆為 1。所以 refinement 關閉時設它是**逐位元無效**的介入——schema 收下、
        # provenance 還會記錄，於是「調了 data 權重」的 ablation 會得到與對照組完全
        # 相同的結果卻被當成有做。大聲擋下來。
        if (config.loss.data_weight != 1.0
                and config.refinement.optimizer == "none"):
            raise ValueError(
                f"loss.data_loss_weight={config.loss.data_weight} 但 refinement 未啟用："
                "主訓練 loss 不讀這個鍵（用的是 GradNorm 的 task_weights[0]，恆為 1.0），"
                "設它不會有任何效果。要調 data 相對權重請改 gradnorm_init_weights。"
            )
        if config.loss.cont_gradnorm and not config.loss.use_gradnorm:
            raise ValueError(
                "loss.cont_gradnorm=True 需要 loss.use_gradnorm=True："
                "continuity 是 GradNorm 的第四個 task，沒有 GradNorm 時它會拿到固定權重 1.0。"
            )


def resolve_inputs(argv: list[str] | None = None) -> ResolutionResult[KolmogorovEffectiveConfig]:
    return resolve_config(argv, KolmogorovPolicy())

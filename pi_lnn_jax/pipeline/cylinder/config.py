"""Cylinder configuration policy for the shared resolution engine."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Mapping

import jax

from pi_lnn_jax.config import (
    CliGuardError,
    ConfigurationLayer,
    ResolutionResult,
    resolve_config,
)


@dataclass(frozen=True)
class CylinderRunConfig:
    config_path: str
    backend: str
    case: str
    is_controlled: bool
    physics_weight_scale: float
    controlled_amp_scale: float
    seed: int
    steps: int
    n_collo: int
    n_sensor_query_requested: int
    learning_rate: float
    optimizer: str
    base_optimizer: str
    #: LR 排程與 SOAP 超參。預設**逐一等於 `optimizers.build_optimizer` 的預設值**，
    #: 亦即本案在加入這些欄位之前實際吃到的值——既有 config 不設它們時行為逐位元不變
    #: （§7.1 契約）。設了才會偏離；`chapter02` 描述的協定要明寫（見 A8）。
    lr_warmup_steps: int
    lr_decay_steps: int
    lr_decay_gamma: float
    min_learning_rate: float
    soap_b1: float
    soap_b2: float
    soap_precondition_frequency: int
    use_physics_denormalization: bool
    artifacts_dir: str


@dataclass(frozen=True)
class CylinderLossConfig:
    bc_weight: float
    bc_body_weight: float
    bc_n: int
    gradnorm_freq: int
    gradnorm_ema_momentum: float
    gradnorm_min: float
    gradnorm_max: float
    al_rho: float
    al_lambda_clip: float
    al_update_freq: int
    t_early_weight: float
    t_early_threshold: float


@dataclass(frozen=True)
class CylinderCurriculumConfig:
    use_time_marching: bool
    time_marching_start: float
    time_marching_warmup_fraction: float


@dataclass(frozen=True)
class CylinderDataConfig:
    npz_key: str
    npz_path: str | None
    u_inf: float


@dataclass(frozen=True)
class CylinderEffectiveConfig:
    run: CylinderRunConfig
    loss: CylinderLossConfig
    curriculum: CylinderCurriculumConfig
    data: CylinderDataConfig

    @property
    def time_marching_warmup_steps(self) -> int:
        return int(self.curriculum.time_marching_warmup_fraction * self.run.steps)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_cylinder.py",
        description="Cylinder CEXP-002 trainer (GradNorm + SOAP)",
    )
    parser.add_argument("--config", required=True, help="TOML config（load_config）")
    parser.add_argument("--steps", type=int, default=None, help="覆蓋 iterations")
    parser.add_argument("--n_collo", type=int, default=None, help="覆蓋 collocation 點數")
    parser.add_argument("--seed", type=int, default=None, help="覆蓋 seed")
    parser.add_argument("--allow-cpu", dest="allow_cpu", action="store_true",
                        help="預設要求 gpu backend；本旗標允許 CPU（僅 smoke）")
    parser.add_argument("--resume", default=None,
                        help="CEXP-002 為 1-shot，禁止 resume；傳入即報錯")
    # 與 Kolmogorov 端同樣走 CLI（共用 TRAIN_SCHEMA 沒有這個鍵，而該檔在 §7.1 契約內，
    # 不為此加鍵）。預設 None = 不覆蓋，既有呼叫端行為不變。
    parser.add_argument("--soap_precondition_frequency", type=int, default=None,
                        help="SOAP preconditioner 更新頻率；不給則沿用 config/policy 的值")
    return parser


class CylinderPolicy:
    _defaults = {
        "run.config_path": None, "run.backend": None, "run.case": "kolmogorov",
        "run.is_controlled": False, "run.physics_weight_scale": 1.0,
        "run.controlled_amp_scale": 1.0, "run.seed": 42, "run.steps": 10000,
        "run.n_collo": 1024, "run.n_sensor_query_requested": 2000,
        "run.learning_rate": 1e-3, "run.optimizer": "schedule_free",
        "run.base_optimizer": "soap", "run.use_physics_denormalization": True,
        # build_optimizer 的預設值，逐字複製；改這裡等於改所有既有 cylinder run 的行為
        "run.lr_warmup_steps": 0, "run.lr_decay_steps": 0,
        "run.lr_decay_gamma": 1.0, "run.min_learning_rate": 1e-6,
        "run.soap_b1": 0.95, "run.soap_b2": 0.95,
        "run.soap_precondition_frequency": 10,
        "run.artifacts_dir": "artifacts",
        "loss.bc_weight": 0.1, "loss.bc_body_weight": 2.0, "loss.bc_n": 128,
        "loss.gradnorm_freq": 25, "loss.gradnorm_ema_momentum": 0.9,
        "loss.gradnorm_min": 0.05, "loss.gradnorm_max": 0.0,
        "loss.al_rho": 1.0, "loss.al_lambda_clip": 10.0,
        "loss.al_update_freq": 25, "loss.t_early_weight": 1.0,
        "loss.t_early_threshold": 0.05,
        "curriculum.use_time_marching": False,
        "curriculum.time_marching_start": 0.5,
        "curriculum.time_marching_warmup_fraction": 0.3,
        "data.npz_key": "cylinder_data_npz", "data.npz_path": None,
        "data.u_inf": 0.0,
    }
    known_fields = frozenset(_defaults)

    def parse_cli(self, argv: list[str] | None) -> argparse.Namespace:
        raw = list(sys.argv[1:] if argv is None else argv)
        return _parser().parse_args(raw)

    def pre_load_guard(self, cli: argparse.Namespace) -> None:
        if cli.resume is not None:
            raise CliGuardError("[FATAL] CEXP-002 為 1-shot 實驗，禁止 resume", 4)
        backend = jax.default_backend()
        cli._backend = backend
        if backend != "gpu" and not cli.allow_cpu:
            raise CliGuardError(f"[FATAL] backend={backend}; --allow-cpu to force", 5)

    def layers(self, cli: argparse.Namespace, loaded: dict) -> tuple[ConfigurationLayer, ...]:
        tk, dk = loaded["train_kwargs"], loaded["data_kwargs"]
        case = str(tk.get("case", "kolmogorov"))
        is_controlled = case == "controlled_cylinder"
        npz_key = "controlled_data_npz" if is_controlled else "cylinder_data_npz"
        toml: dict[str, Any] = {
            "run.case": case,
            "run.is_controlled": is_controlled,
            "run.physics_weight_scale": (
                float(tk.get("physics_weight_scale", 0.3)) if is_controlled else 1.0
            ),
            "data.npz_key": npz_key,
            "data.npz_path": dk.get(npz_key),
        }
        train_map = {
            "controlled_amp_scale": "run.controlled_amp_scale",
            "seed": "run.seed", "iterations": "run.steps",
            "num_physics_points": "run.n_collo",
            "bc_weight": "loss.bc_weight", "bc_body_weight": "loss.bc_body_weight",
            "bc_n": "loss.bc_n", "gradnorm_freq": "loss.gradnorm_freq",
            "gradnorm_ema_momentum": "loss.gradnorm_ema_momentum",
            "gradnorm_min_weight": "loss.gradnorm_min",
            "gradnorm_max_weight": "loss.gradnorm_max", "al_rho": "loss.al_rho",
            "al_lambda_clip": "loss.al_lambda_clip", "al_update_freq": "loss.al_update_freq",
            "learning_rate": "run.learning_rate",
            "lr_warmup_steps": "run.lr_warmup_steps",
            "lr_decay_steps": "run.lr_decay_steps",
            "lr_decay_gamma": "run.lr_decay_gamma",
            "min_learning_rate": "run.min_learning_rate",
            # soap_b1 / soap_b2 / soap_precondition_frequency **不可**列在這裡：
            # 它們不在共用 TRAIN_SCHEMA 裡，load_config 只把 schema 鍵放進
            # train_kwargs，所以 `if key in tk` 恆為 False——列了就是死碼，還會讓
            # 下一個人在 TOML 裡寫 soap_b1 然後被當 unknown key 靜默忽略。
            # betas 走下方的 soap_betas 特例；pf 走 --soap_precondition_frequency 旗標。
            "artifacts_dir": "run.artifacts_dir",
            "use_physics_denormalization": "run.use_physics_denormalization",
            "t_early_weight": "loss.t_early_weight",
            "t_early_threshold": "loss.t_early_threshold",
            "time_marching": "curriculum.use_time_marching",
            "time_marching_start": "curriculum.time_marching_start",
            "time_marching_warmup": "curriculum.time_marching_warmup_fraction",
        }
        for key, path in train_map.items():
            if key in tk:
                toml[path] = tk[key]
        if "soap_betas" in tk:
            _b = list(tk["soap_betas"])
            if len(_b) != 2:
                raise ValueError(f"soap_betas 須為兩個元素 [b1, b2]，收到 {_b}")
            toml["run.soap_b1"], toml["run.soap_b2"] = float(_b[0]), float(_b[1])
        if "num_sensor_query_points" in tk:
            toml["run.n_sensor_query_requested"] = int(tk["num_sensor_query_points"]) or 2000
        if "gradnorm_freq" in tk and "al_update_freq" not in tk:
            toml["loss.al_update_freq"] = int(tk["gradnorm_freq"])
        if "cylinder_u_inf" in dk:
            toml["data.u_inf"] = float(dk["cylinder_u_inf"])
        # optimizer/base_optimizer remain intentionally unreachable through TOML:
        # load_config warns and ignores those unknown keys, preserving TD-14.
        if getattr(cli, "soap_precondition_frequency", None) is not None:
            toml["run.soap_precondition_frequency"] = int(cli.soap_precondition_frequency)
        cli_values: dict[str, Any] = {
            "run.config_path": cli.config,
            "run.backend": cli._backend,
        }
        if cli.steps is not None:
            cli_values["run.steps"] = cli.steps
        if cli.n_collo is not None:
            cli_values["run.n_collo"] = cli.n_collo
        if cli.seed is not None:
            cli_values["run.seed"] = cli.seed
        return (
            ConfigurationLayer("schema_default", self._defaults),
            ConfigurationLayer("toml", toml),
            ConfigurationLayer("cli", cli_values),
        )

    def build(self, v: Mapping[str, Any]) -> CylinderEffectiveConfig:
        return CylinderEffectiveConfig(
            run=CylinderRunConfig(**{
                field: v[f"run.{field}"] for field in CylinderRunConfig.__dataclass_fields__
            }),
            loss=CylinderLossConfig(**{
                field: v[f"loss.{field}"] for field in CylinderLossConfig.__dataclass_fields__
            }),
            curriculum=CylinderCurriculumConfig(**{
                field: v[f"curriculum.{field}"]
                for field in CylinderCurriculumConfig.__dataclass_fields__
            }),
            data=CylinderDataConfig(**{
                field: v[f"data.{field}"] for field in CylinderDataConfig.__dataclass_fields__
            }),
        )

    def guard(self, cli: argparse.Namespace, config: CylinderEffectiveConfig) -> None:
        del cli, config


def resolve_inputs(argv: list[str] | None = None) -> ResolutionResult[CylinderEffectiveConfig]:
    try:
        return resolve_config(argv, CylinderPolicy())
    except CliGuardError as error:
        print(error.message, file=sys.stderr)
        raise SystemExit(error.exit_code) from error

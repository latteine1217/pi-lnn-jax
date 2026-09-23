"""Toy ablation for continuity AL formulations.

Usage:
  uv run python -m bench.al_ablation_toy --steps 200 --out artifacts/al_toy.json

這個 toy 不替代 full PI-CON 訓練；它只把三種 constraint handling 放到同一個
可重跑、快速的衝突問題上，方便檢查 multiplier trajectory 與輸出格式。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _constraint_residual(theta: float) -> np.ndarray:
    targets = np.array([0.8, 0.9, 1.0, 1.1], dtype=np.float64)
    return theta - targets


def _metrics(theta: float) -> dict[str, float]:
    r = _constraint_residual(theta)
    return {
        "final_theta": float(theta),
        "final_objective": float(theta ** 2),
        "final_constraint_mse": float(np.mean(r ** 2)),
        "final_constraint_signed_mean": float(np.mean(r)),
    }


def _finite_grad(fn, theta: float) -> float:
    eps = 1e-5
    return float((fn(theta + eps) - fn(theta - eps)) / (2.0 * eps))


def _run_variant(variant: str, steps: int, lr: float, rho: float) -> dict:
    theta = 0.0
    lam = 0.0
    history = []

    for step in range(1, steps + 1):
        def loss(th: float) -> float:
            r = _constraint_residual(th)
            c_mse = float(np.mean(r ** 2))
            c_signed = float(np.mean(r))
            objective = th ** 2
            if variant == "fixed_penalty":
                return objective + rho * c_mse
            if variant == "mse_al_style":
                return objective + lam * c_mse + 0.5 * rho * c_mse ** 2
            if variant == "signed_mean_al":
                return objective + lam * c_signed + 0.5 * rho * c_signed ** 2
            raise ValueError(f"unknown variant: {variant}")

        theta -= lr * _finite_grad(loss, theta)
        r_now = _constraint_residual(theta)
        c_mse_now = float(np.mean(r_now ** 2))
        c_signed_now = float(np.mean(r_now))

        if variant == "mse_al_style":
            lam = float(np.clip(lam + rho * c_mse_now, 0.0, 10.0))
        elif variant == "signed_mean_al":
            lam = float(np.clip(lam + rho * c_signed_now, -10.0, 10.0))

        if step == 1 or step == steps or step % max(1, steps // 5) == 0:
            history.append({
                "step": step,
                "theta": float(theta),
                "constraint_mse": c_mse_now,
                "constraint_signed_mean": c_signed_now,
                "lambda": lam,
            })

    row = {
        "variant": variant,
        "steps": int(steps),
        "lr": float(lr),
        "rho": float(rho),
        "final_lambda": float(lam),
        "history": history,
    }
    row.update(_metrics(theta))
    return row


def run_ablation(steps: int = 200, lr: float = 0.05, rho: float = 1.0, seed: int = 0) -> list[dict]:
    """Run deterministic toy ablation and return JSON-serializable rows."""
    np.random.seed(seed)
    variants = ["fixed_penalty", "mse_al_style", "signed_mean_al"]
    return [_run_variant(v, steps=steps, lr=lr, rho=rho) for v in variants]


def main() -> int:
    parser = argparse.ArgumentParser(description="Toy AL formulation ablation")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    rows = run_ablation(steps=args.steps, lr=args.lr, rho=args.rho, seed=args.seed)
    text = json.dumps(rows, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"[out] {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

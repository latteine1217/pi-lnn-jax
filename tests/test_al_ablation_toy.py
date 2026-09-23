"""Toy AL ablation tests。

目的不是替代 full train，而是提供快速、可重跑的機制檢查：
fixed penalty、現行 MSE-AL-style、signed aggregated AL 三者需輸出同一組
constraint diagnostics，避免論文/實作討論停在概念層。
"""
from __future__ import annotations


def test_toy_al_ablation_reports_all_variants():
    from bench.al_ablation_toy import run_ablation

    rows = run_ablation(steps=40, lr=0.05, rho=1.0, seed=0)
    names = {row["variant"] for row in rows}

    assert names == {"fixed_penalty", "mse_al_style", "signed_mean_al"}
    for row in rows:
        assert row["steps"] == 40
        assert row["final_objective"] >= 0.0
        assert "final_constraint_mse" in row
        assert "final_constraint_signed_mean" in row
        assert "final_lambda" in row
        assert isinstance(row["history"], list)
        assert row["history"], "history 不可為空，否則無法審查收斂軌跡"

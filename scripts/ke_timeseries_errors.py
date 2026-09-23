#!/usr/bin/env python
"""對既有的 ke_timeseries.npz 離線計算 scalar E(t) 時序誤差，無需重跑 model。

npz 由 evaluate_exp245.py / evaluate_multi_re.py 產出，含 key: t / ke_pred / ke_dns。
指標見 pi_lnn_jax.evaluate.energy_timeseries_errors（relative L2(0,T) / L∞ / final-time）。

用法:
    uv run python scripts/ke_timeseries_errors.py path/to/ke_timeseries.npz [--json out.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pi_lnn_jax.metric_artifact import energy_timeseries_errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", type=Path, help="ke_timeseries.npz 路徑")
    ap.add_argument("--json", type=Path, default=None, help="可選：把結果另存成 JSON")
    args = ap.parse_args()

    if not args.npz.is_file():
        ap.error(f"找不到 npz: {args.npz}")
    data = np.load(args.npz)
    missing = [k for k in ("ke_pred", "ke_dns") if k not in data]
    if missing:
        ap.error(f"npz 缺少 key {missing}（實有 {list(data.keys())}）")

    errs = energy_timeseries_errors(data["ke_pred"], data["ke_dns"])

    print(f"=== E(t) series errors: {args.npz} ===")
    # 2026-08-01 改名：本函式回傳的空間平均版由 `ke_t_mape` 改為
    # `ke_t_mape_spatialmean`（純鍵名，公式未動），headline `ke_t_mape` 已改指
    # pointwise 定義且不由本函式產出。標籤跟著改，避免與 headline 混淆。
    print(f"  MAPE (spatial-mean) = {errs['ke_t_mape_spatialmean']:.4f}  "
          f"({errs['ke_t_mape_spatialmean'] * 100:.2f}%)")
    print(f"  rel-L2(0,T)    = {errs['ke_t_rel_l2']:.4f}  ({errs['ke_t_rel_l2'] * 100:.2f}%)")
    print(f"  rel-Linf       = {errs['ke_t_rel_linf']:.4f}  ({errs['ke_t_rel_linf'] * 100:.2f}%)")
    print(f"  final-time     = {errs['ke_t_final_rel']:.4f}  ({errs['ke_t_final_rel'] * 100:.2f}%)")

    if args.json is not None:
        args.json.write_text(json.dumps(errs, indent=2))
        print(f"[out] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

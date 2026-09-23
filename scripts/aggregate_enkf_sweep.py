#!/usr/bin/env python
"""把 EnKF 敏感度掃描的九份 artifact 彙整成一張 (N_e × loc_radius) 表。

主指標是 `uv_rel_err`（2026-08 指標重設計：KE 降級為 QoI，MAPE 用 pointwise）。
`ke_t_mape.spatialmean.v1` 也讀出來，但**只為與改定義之前已發表的數字對照**——
它先把整場塌成純量再比，空間分布錯誤會被抵消，不得當 headline。

值一律經 `result_table.read_metric_value` 取得（語意檢查過），不自行 json.load
挑鍵：投影鍵漂移會在那裡大聲失敗，手挑則安靜給錯的量。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pi_lnn_jax.metric_artifact import MetricUnavailable
from pi_lnn_jax.result_table import read_metric_value

#: 讀 recorder 寫出的 **canonical artifact**（帶 schema_version、語意完整），
#: 不讀同目錄下扁平化的 compatibility projection——後者沒有 ke_t_errors 結構，
#: KE 家族的語意層無從裁決 pointwise / spatialmean，read_metric_value 會 KeyError。
_ARTIFACT_GLOB = "enkf_ne*_loc*_enkf.metric_artifact.json"

#: 欄位順序即閱讀順序：primary 在最左。
#: 用 **summary** definition id 而非 base id——同一個 base（如
#: velocity.vector.rel_l2.v1）在 canonical artifact 裡對應 time_mean 與 time_p90
#: 多個 summary，用 base id 讀會因「多個匹配」而失敗（那個失敗是對的：
#: 它逼呼叫端說清楚要哪一種聚合，而不是靜默取第一個）。
COLUMNS = [
    ("uv", "velocity.vector.rel_l2.time_mean.v1"),
    ("uv_p90", "velocity.vector.rel_l2.time_p90.v1"),
    ("u", "velocity.u.rel_l2.time_mean.v1"),
    ("v", "velocity.v.rel_l2.time_mean.v1"),
    ("omega", "vorticity.rel_l2.time_mean.v1"),
    ("KE_pw_MAPE", "kinetic_energy.pointwise.mape.time_mean.v2"),
    ("KE_legacy_sm", "ke_t_mape.spatialmean.time_mean.v1"),
]

_CELL = re.compile(r"enkf_ne(\d+)_loc([0-9.]+)_enkf\.metric_artifact\.json$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="掃描產物目錄")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    d = Path(a.dir)
    if not d.is_dir():
        raise FileNotFoundError(f"產物目錄不存在：{d}")
    files = sorted(p for p in d.glob(_ARTIFACT_GLOB) if _CELL.search(p.name))
    if not files:
        raise FileNotFoundError(f"{d} 下沒有 {_ARTIFACT_GLOB} —— 掃描可能全數失敗")

    rows = []
    for p in files:
        m = _CELL.search(p.name)
        proj = json.loads(p.read_text(encoding="utf-8"))
        row = {"Ne": int(m.group(1)), "loc": float(m.group(2)), "file": p.name}
        for label, defn in COLUMNS:
            try:
                row[label] = read_metric_value(proj, defn)
            except MetricUnavailable:
                # 缺席是 availability，不是漂移——落 None 而非中止，但要看得見
                row[label] = None
        rows.append(row)
    rows.sort(key=lambda r: (r["Ne"], r["loc"]))

    hdr = f"{'Ne':>4} {'loc':>5} " + " ".join(f"{c:>15}" for c, _ in COLUMNS)
    print(hdr); print("-" * len(hdr))
    for r in rows:
        cells = " ".join(
            (f"{r[c]*100:14.2f}%" if r[c] is not None else f"{'n/a':>15}") for c, _ in COLUMNS)
        print(f"{r['Ne']:>4} {r['loc']:>5} {cells}")

    best = min((r for r in rows if r["uv"] is not None), key=lambda r: r["uv"], default=None)
    if best:
        print(f"\n[primary] uv_rel_err 最低：Ne={best['Ne']} loc={best['loc']} "
              f"→ {best['uv']*100:.2f}%")
    got = {(r["Ne"], r["loc"]) for r in rows}
    print(f"[coverage] {len(rows)} 格；Ne={sorted({r['Ne'] for r in rows})} "
          f"loc={sorted({r['loc'] for r in rows})}")
    missing = [(n, l) for n in sorted({r['Ne'] for r in rows})
               for l in sorted({r['loc'] for r in rows}) if (n, l) not in got]
    if missing:
        # 掃描途中執行時，缺格只是「還沒跑到」；掃描結束後才等同「發散或失敗」。
        # 這裡不猜是哪一種——寫死其中一個解讀會讓中途檢視誤報成失敗。
        print(f"[WARN] 缺格（尚未跑到，或該組態發散未產出 artifact）：{missing}")
        print("       掃描結束後仍缺的格＝該組態失敗；請對照 job 的 [FAIL] 行判定")

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"[out] {a.json_out}")


if __name__ == "__main__":
    main()

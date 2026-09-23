#!/usr/bin/env python3
"""K–Δt 取樣平面重跑批次的 config 產生器（一次全部吐出，不手抄繼承）。

What:
    從單一 base config 產生 K × Δt × seed 的完整格。每支只改六個鍵：
    sensor json/npz、`time_strides`、`seed`、`num_sensor_query_points`、
    `grad_accum_chunks`、`artifacts_dir`。

Why:
    上一批是手抄基底繼承來的，四個 Δt=0.025 的 config 沿用了 K=200 的
    `grad_accum_chunks=2` 而沒跟著各列調，於是整欄的「取樣密度」與「分塊」共線
    （實測分塊值 −2.25%、取樣值 −2.96%，各佔一半）。同一份 config 也整批繼承了
    `num_sensor_query_points=2000`，在 T×K < 2000 的格被靜默降級為全取。
    兩個缺陷都源於「人照著另一支 config 改」。本腳本把該變的鍵集中在一處，
    並在每個替換點 fail-fast——找不到或找到多個就拋錯，不默默略過。

    本批的兩個設計決定：
      * **M 全格固定 2。** 最吃記憶體的格（K=400/T=201，80,400 個 sensor 時空點）
        實測在 M=2 下跑得完，且 M=2 在 K=400 上同時較準（−2.25%）與較快（2.3×）。
      * **nsq 全格固定 512。** 最小母體是 K=50/Δt=0.5 的 T×K = 550
        （T 為實際載入後的幀數，不是 201//stride）。512 是能讓 20 格都不觸發
        `effective_sensor_query_points` 降級的最大 2 的冪，且被 M=2 整除。

Usage:
    uv run python scripts/gen_sampling_plane_configs.py --preset pilot --tag sp2
    uv run python scripts/gen_sampling_plane_configs.py --preset grid --seeds 3 --dry-run
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

BASE = Path("configs/exp_k200_snap_st2_s1.toml")
SENSOR_TMPL = "data/kolmogorov_sensors/re10000/sensors_qrpivot_K{K}_N256_t0-5_si100_les_n256_T50standalone"
STRIDE_T = {1: 201, 2: 101, 4: 51, 8: 26, 20: 11}     # 實際載入後的幀數，已驗
# nsq 維持主線的 2000。門檻實驗（jobs 5905–5910）實測降到 512 的位移**隨 K 變且號會翻**
# （K=100 −0.33%、K=400 +1.48%，兩者差 +1.81 pp、Welch p=0.045），而那 1.81 pp 是 K 主效應的
# 3.1%——與交互作用 RMS（K 的 2.6%）同量級。換 nsq 等於用一個未知的扭曲換完整矩形，不划算。
#
# ⚠️ 結構性限制，不是可修的缺陷：sensor 母體是 T×K，**正好是要掃的兩軸的乘積**。固定點數則
# 抽樣比例隨兩軸變（2000 時從 2.5% 到 78%）；固定比例則點數變、梯度噪聲跟著變。沒有一個設定
# 對兩軸同時中性，唯一中性的是全格不子取樣（最貴的格 80,400 點，成本未量）。
# 敏感度已框住：4 倍的 count 變動值 ≲1.5%。
NSQ = 2000
GRAD_ACCUM = 2
SEED_ORDER = (42, 1, 2, 3, 4)   # 先跑前 N 個；之後補到 n=5 不必重跑已完成的

PRESETS = {
    # 門檻實驗：兩個 K 都已有 M=2 + nsq=2000 的現成對照，可直接配對比 nsq 的位移
    "pilot": {"K": (100, 400), "stride": (2,), "seed": SEED_ORDER[:3]},
    # grid 的格由母體自動篩（nsq > T×K 的格會被跳過並印出理由），不手維護清單
    "grid":  {"K": (50, 100, 200, 400), "stride": (1, 2, 4, 8, 20), "seed": SEED_ORDER},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), required=True)
    p.add_argument("--tag", default="spg",
                   help="檔名與 artifacts_dir 的批次標記。**不要重用已跑過的 tag**——"
                        "artifacts_dir 會撞進既有目錄、覆寫它的 ckpt 與 summary。"
                        "已使用：sp2 = nsq 門檻實驗（jobs 5905–5910）、spg = 主格。")
    p.add_argument("--seeds", type=int, default=None,
                   help=f"只取 {SEED_ORDER} 的前 N 個。省略則用 preset 的全部。"
                        "先跑 N=3、之後補到 5 時，已完成的那三個 seed 不必重跑。")
    p.add_argument("--dry-run", action="store_true", help="只列出要產生的檔，不寫入")
    return p.parse_args()


def _sub_once(text: str, pattern: str, repl: str, what: str) -> str:
    """替換且只替換一處。命中 0 次或多次都是 base config 漂移了，直接拋錯。"""
    out, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if n != 1:
        raise ValueError(f"{what}：在 base config 命中 {n} 次（預期 1）——"
                         f"base 已漂移，先確認 {BASE} 再跑")
    return out


def build(base: str, K: int, stride: int, seed: int, tag: str) -> tuple[str, str]:
    name = f"exp_{tag}_k{K}_st{stride}_s{seed}"
    sensor = SENSOR_TMPL.format(K=K)
    t = base
    t = _sub_once(t, r'^\s*"data/kolmogorov_sensors/re10000/sensors_qrpivot_K\d+_[^"]*\.json",$',
                  f'  "{sensor}.json",', "sensor json")
    t = _sub_once(t, r'^\s*"data/kolmogorov_sensors/re10000/sensors_qrpivot_K\d+_[^"]*_dns_values\.npz",$',
                  f'  "{sensor}_dns_values.npz",', "sensor npz")
    t = _sub_once(t, r"^time_strides = \[\d+\].*$",
                  f"time_strides = [{stride}]   # T = {STRIDE_T[stride]} 幀（實際載入後，非 201//stride）",
                  "time_strides")
    t = _sub_once(t, r"^seed = \d+$", f"seed = {seed}", "seed")
    t = _sub_once(t, r"^num_sensor_query_points = \d+$",
                  f"num_sensor_query_points = {NSQ}   # 全格固定：最小母體 550（K=50/Δt=0.5），"
                  f"512 是不觸發降級的最大 2 的冪", "num_sensor_query_points")
    t = _sub_once(t, r"^grad_accum_chunks = \d+.*$",
                  f"grad_accum_chunks = {GRAD_ACCUM}   # 全格固定：M 非中性，隨 K 變就是共線",
                  "grad_accum_chunks")
    t = _sub_once(t, r'^artifacts_dir = ".*"$',
                  f'artifacts_dir = "artifacts/kolmogorov/{tag}_k{K}_st{stride}_s{seed}"',
                  "artifacts_dir")
    header = (f"# [SP2] K–Δt 取樣平面重跑：K={K}、Δt={5.0/(STRIDE_T[stride]-1):.3f} s"
              f"（stride={stride}, T={STRIDE_T[stride]}）、seed={seed}\n"
              f"# 全批固定 grad_accum_chunks={GRAD_ACCUM} 與 num_sensor_query_points={NSQ}；\n"
              f"# 由 scripts/gen_sampling_plane_configs.py 產生，不要手改單支。\n")
    return name, header + t.split("\n", 1)[1] if t.startswith("#") else header + t


def main() -> None:
    a = parse_args()
    if not BASE.exists():
        raise FileNotFoundError(f"base config 不存在：{BASE}")
    base = BASE.read_text()
    spec = PRESETS[a.preset]
    seeds = spec["seed"][:a.seeds] if a.seeds else spec["seed"]
    made, skipped = [], []
    for K in spec["K"]:
        for stride in spec["stride"]:
            pop = STRIDE_T[stride] * K
            if NSQ > pop:
                # 跳過而非拋錯：這是設計上就取不到的格（母體小於 mini-batch），
                # 不是設定錯誤。大聲列出來，讓破格的形狀在產生時就可見。
                skipped.append((K, stride, pop))
                continue
            for seed in seeds:
                name, text = build(base, K, stride, seed, a.tag)
                made.append((name, text, pop))
    if skipped:
        print(f"跳過 {len(skipped)} 格（母體 < nsq={NSQ}，抽樣會被降級為全取）：")
        for K, stride, pop in skipped:
            print(f"    K={K:<4} stride={stride:<3} 母體 T×K = {pop}")
    print(f"{a.preset}：{len(made)} 支（{len(made)//max(len(seeds),1)} 格 × {len(seeds)} seed）")
    for name, text, pop in made:
        path = Path("configs") / f"{name}.toml"
        print(f"  {'(dry)' if a.dry_run else '[out]'} {path}   母體 T×K = {pop}")
        if not a.dry_run:
            path.write_text(text)


if __name__ == "__main__":
    main()

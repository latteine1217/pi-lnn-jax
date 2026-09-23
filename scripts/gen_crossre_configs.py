#!/usr/bin/env python3
"""產生 cross-Re × sensor-budget campaign 的 15 個訓練 config。

What:
    Re ∈ {100, 500, 1000, 10000, 10⁶} × K ∈ {10, 50, 100}，FPS (space-filling)
    placement，對應 thesis 的 cross-Re sensor-budget 圖。每個 config 由
    `configs/exp_245_b3_les_T50.toml` 逐字派生，只覆寫六個資料相關欄位：
    sensor_jsons / sensor_npzs / dns_paths / re_values / T_total / artifacts_dir。

Why:
    手寫 15 個 config 會讓架構與優化器參數各自漂移，而漂移不會 crash——只會讓
    「只有 Re 和 K 不同」這個前提悄悄失效，圖上的趨勢就不再歸因於 Re 和 K。
    以文字替換而非重新序列化，是為了保住 base config 的註解（尤其 §JAX-TODO
    那段記錄了哪些 pi-lnn 鍵在 JAX 尚未 wire-up）。

    參數來自 pi-lnn 的 EXP-320~328 / 330~335（PyTorch 側同一組跑）。各 Re 的
    T_total 不同是刻意的：cross-Re DNS 按 eddy-turnover 時間對齊，低 Re 需要
    更長的物理時窗才涵蓋同樣多的 turnover。

Usage:
    uv run python scripts/gen_crossre_configs.py            # 寫入 configs/
    uv run python scripts/gen_crossre_configs.py --dry-run  # 只檢查資料可用性
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from pi_lnn_jax.data import PILNJAX_DATA_ROOT, PILNJAX_ROOT, _resolve_data_path

BASE_CONFIG = "configs/exp_245_b3_les_T50.toml"

# 每個 Re 的格點、matched 時窗與資料路徑（對齊 pi-lnn EXP-320~335）。
# sensor_tag 是 sensor 檔名裡的 Re 標記，與 re_value 的格式不一致（10⁶ 寫作 1e6）。
#
# time_stride 必須逐 Re 明示，不能吃 schema 預設 2（同 exp_302 / EXP-301 的坑）：
# 判準是餵進模型的時間點數要落在 ~101 frames，五條線才是同一個時間解析度。
# cross-Re 的 DNS 已按 eddy-turnover 做過 matched 下採樣（100/101/104 frames），
# 再套預設 stride=2 會只剩 ~50 frames，等於半個時間解析度；只有 Re=10⁴ 的
# 201-frame 原始檔需要 stride=2 才降到 101。
RE_SPECS = [
    dict(re=100.0,     n=128, t_total=55.242, sensor_dir="re100",
         sensor_tag="Re100", time_stride=1, n_frames=100,
         dns="data/dns/cross_re/kolmogorov_dns_Re100_N128_T55p8_matched.npy"),
    dict(re=500.0,     n=128, t_total=17.2,   sensor_dir="re500",
         sensor_tag="Re500", time_stride=1, n_frames=101,
         dns="data/dns/cross_re/kolmogorov_dns_Re500_N128_T17p28_matched.npy"),
    dict(re=1000.0,    n=128, t_total=10.3,   sensor_dir="re1000",
         sensor_tag="Re1000", time_stride=1, n_frames=104,
         dns="data/dns/cross_re/kolmogorov_dns_Re1000_N128_T10p3_matched.npy"),
    dict(re=10000.0,   n=256, t_total=5.0,    sensor_dir="re10000_fps",
         sensor_tag="Re10000", time_stride=2, n_frames=201,
         dns="data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_si100_ds4.npy"),
    dict(re=1000000.0, n=512, t_total=5.0,    sensor_dir="re1e6_fps",
         sensor_tag="Re1e6", time_stride=1, n_frames=101,
         dns="data/dns/kolmogorov_Re1e6_N512_T5_ds4.npy"),
]

K_VALUES = [10, 50, 100]


def _sensor_stem(spec: dict, k: int) -> str:
    return (f"data/kolmogorov_sensors/{spec['sensor_dir']}/"
            f"sensors_spacefill_K{k}_N{spec['n']}_{spec['sensor_tag']}_matched")


def _re_label(re_value: float) -> str:
    """config 檔名用的 Re 標記（1e6 不寫成 1000000，與 pi-lnn 一致）。"""
    return "1e6" if re_value >= 1e6 else str(int(re_value))


def _base_sensor_query_points(base_text: str) -> int:
    """base config 的 sensor mini-batch 大小。讀不到就硬失敗。

    不寫死數字：這個值是本生成器判斷「要不要降級成全取」的門檻，base 改了
    而這裡沒跟上，會讓門檻靜默失準。
    """
    m = re.search(r'^num_sensor_query_points\s*=\s*(\d+)', base_text, flags=re.MULTILINE)
    if m is None:
        raise ValueError("base config 找不到 num_sensor_query_points；base 結構已變")
    return int(m.group(1))


def _sensor_query_points(spec: dict, k: int, base_nsq: int) -> int:
    """本格該用的 sensor mini-batch 大小；population 不足時回 0（= 全取）。

    sensor query 的 population 是「訓練實際餵入的 frames × K」，抽樣走
    replace=False，所以 mini-batch 一旦大於 population 就直接 ValueError。
    K=10 這排在每個 Re 都只有 ~1000 個時空點，撐不起 base 的 2000。

    降級成 0（full T*K）而不是硬塞一個較小的數字：population 已經比 base 的
    抽樣量還小，抽樣本身失去意義，全取才是這格的自然設定。計算量也不會爆——
    K=10 全取 1000 點仍少於 K=50/100 各自抽的 2000 點。
    """
    population = (spec["n_frames"] // spec["time_stride"]) * k
    return 0 if population < base_nsq else base_nsq


def _replace_field(text: str, pattern: str, replacement: str, field: str) -> str:
    """單一欄位替換；沒命中就硬失敗——靜默不替換會產生指向錯資料的 config。"""
    new_text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if n != 1:
        raise ValueError(f"base config 中找不到欄位 {field!r}（命中 {n} 次）；"
                         f"base 結構已變，請更新本生成器")
    return new_text


def render_config(base_text: str, spec: dict, k: int) -> str:
    """把 base config 的六個資料欄位換成本 (Re, K) 組合的值。"""
    stem = _sensor_stem(spec, k)
    label = _re_label(spec["re"])
    frames_fed = spec["n_frames"] // spec["time_stride"]
    base_nsq = _base_sensor_query_points(base_text)
    nsq = _sensor_query_points(spec, k, base_nsq)
    population = frames_fed * k
    header = (
        f"# configs/exp_crossre_re{label}_k{k}.toml — 由 scripts/gen_crossre_configs.py 生成\n"
        f"#\n"
        f"# cross-Re × sensor-budget campaign：Re={spec['re']:g}, K={k}, "
        f"FPS placement, N={spec['n']}。\n"
        f"# 逐字派生自 {BASE_CONFIG}，只覆寫 sensor/dns/re_values/T_total/time_strides/\n"
        f"# artifacts_dir，其餘架構與優化器參數與主 baseline 完全一致（差異只有 Re 和 K）。\n"
        f"# 對應 pi-lnn PyTorch 側的 EXP-320~328 / 330~335。\n"
        f"#\n"
        f"# time_strides = [{spec['time_stride']}] 為明示值，不可省略：DNS 有 "
        f"{spec['n_frames']} frames，\n"
        f"# stride {spec['time_stride']} 後餵入 ~{frames_fed} frames，與其餘 Re 的時間解析度對齊。\n"
        f"# 吃 schema 預設 2 會讓本 case 只剩一半時間點（見 exp_302 的同一個坑）。\n"
        f"# eval 時 --time-stride 必須同樣給 {spec['time_stride']}。\n"
        f"#\n"
        f"# num_sensor_query_points = {nsq}：sensor query population = "
        f"{frames_fed} frames × K={k} = {population}。\n"
        + (f"# 小於 base 的 {base_nsq}，抽樣（replace=False）會直接 ValueError，"
           f"故降為 0 = 全取。\n"
           if nsq == 0 else
           f"# 大於 base 的 {base_nsq}，維持 base 設定。\n")
        + f"# 手改本檔會在下次生成時被覆蓋——要改請改生成器。\n"
    )
    text = base_text
    text = _replace_field(
        text, r'^sensor_jsons = \[\n(?:.*\n)*?\]',
        f'sensor_jsons = [\n  "{stem}.json",\n]', "sensor_jsons")
    text = _replace_field(
        text, r'^sensor_npzs = \[\n(?:.*\n)*?\]',
        f'sensor_npzs = [\n  "{stem}_dns_values.npz",\n]', "sensor_npzs")
    text = _replace_field(
        text, r'^dns_paths = \[.*\]$',
        f'dns_paths = ["{spec["dns"]}"]', "dns_paths")
    # time_strides 不在 base config 裡（主線吃預設 2），故隨 re_values 一併寫入
    text = _replace_field(
        text, r'^re_values = \[.*\]$',
        f're_values = [{spec["re"]}]\ntime_strides = [{spec["time_stride"]}]',
        "re_values")
    text = _replace_field(
        text, r'^T_total = .*$',
        f'T_total = {spec["t_total"]}', "T_total")
    text = _replace_field(
        text, r'^artifacts_dir = .*$',
        f'artifacts_dir = "artifacts/kolmogorov/crossre_re{label}_k{k}"',
        "artifacts_dir")
    text = _replace_field(
        text, r'^num_sensor_query_points = .*$',
        f'num_sensor_query_points = {nsq}', "num_sensor_query_points")
    return header + text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                   help="不寫檔，只列出將產生的 config 與其資料在本機是否存在")
    args = p.parse_args()

    repo = Path(__file__).resolve().parent.parent
    base_path = repo / BASE_CONFIG
    if not base_path.is_file():
        raise FileNotFoundError(f"base config 不存在: {base_path}")
    base_text = base_path.read_text(encoding="utf-8")

    written, missing = [], []
    for spec in RE_SPECS:
        for k in K_VALUES:
            label = _re_label(spec["re"])
            out = repo / "configs" / f"exp_crossre_re{label}_k{k}.toml"
            text = render_config(base_text, spec, k)

            stem = _sensor_stem(spec, k)
            absent = [rel for rel in (f"{stem}.json", f"{stem}_dns_values.npz", spec["dns"])
                      if not _resolve_data_path(rel).is_file()]
            status = "OK" if not absent else f"缺 {len(absent)} 個資料檔"
            if absent:
                missing.append((out.name, absent))
            print(f"  {out.name:<34s} Re={spec['re']:>9g} K={k:<4d} "
                  f"T={spec['t_total']:<7g} {status}")

            if not args.dry_run:
                out.write_text(text, encoding="utf-8")
                written.append(out)

    print(f"\n[data root] {PILNJAX_ROOT}  (大檔 fallback: {PILNJAX_DATA_ROOT})")
    if missing:
        print(f"[warn] {len(missing)}/{len(RE_SPECS) * len(K_VALUES)} 個 config 的資料"
              f"不在本機（訓練在 lab-server 執行，需確認該處存在）：")
        for name, absent in missing:
            for rel in absent:
                print(f"    {name}: {rel}")
    if args.dry_run:
        print("\n[dry-run] 未寫檔")
    else:
        print(f"\n[out] 已寫入 {len(written)} 個 config → configs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

<!-- PROVENANCE -->
> **來源**：`generate_kolmogorov_dns_fp64.py` 與 `downsample_dns.py` 於 2026-08-21 自
> pi-lnn `tools/dns_generator/` **逐字複製**（md5 一致），以消除本專案對 pi-lnn repo
> 的執行期依賴。兩支檔案刻意不做任何修改——已生成的 DNS 是 eval 真值，改動求解器
> 會讓新舊軌跡不可比。
>
> `torch` 是可選 import：有它就走 `torch-cuda`，沒有會**靜默**降級到 numpy backend
> （慢數十倍）。`scripts/slurm/{gen_dns_extended,probe_steady_state}.sbatch` 因此要求
> 提交端以 `PY_DNS` 指定直譯器，並檢查 `torch.cuda.is_available()`，不讓它默默慢死。
>
> **不要在 lab-server 的專案 venv 裝 torch**（2026-08-21 實測）：driver 560.35.05
> 只到 CUDA 12.6，torch 的 cu13 wheel 在 compute node 上 `cuda available == False`；
> 更糟的是它帶進的 `nvidia-cudnn-cu13` 會蓋掉 JAX 的 cu12 路徑，整個 worktree 的
> JAX GPU 當場失效（`RET_CHECK dnn_support != nullptr`），十個訓練/eval job 因此全滅，
> 得跑 `scripts/slurm/setup_gpu_venv.sh` 才修得回來。DNS 一律在 home-gpu 生成。

# DNS Generator (vendored copy)

## Source-of-truth

| Location | Role |
|---|---|
| `home-gpu:~/pi-lnn-cfd-baseline/dns/generate_kolmogorov_dns_fp64.py` | **執行版本**（GPU runs 在這跑）|
| `pi-lnn:tools/dns_generator/generate_kolmogorov_dns_fp64.py` | **Vendored copy**（pytest 本地驗證用，不在這直接 edit）|

home-gpu 的 `pi-lnn-cfd-baseline` 是獨立 repo，generator 的 git history 在那邊。pi-lnn 這份是「frozen snapshot」，供：
1. `tests/test_grid_independence_ic_alignment.py` 本地 import 驗證
2. Code review 時不需 SSH 即可審查 generator 改動

## Sync 流程

修改在 home-gpu 上做 → rsync 回 pi-lnn 同步 vendored copy：

```bash
# 從 home-gpu 拉新版本
rsync -avz home-gpu:~/pi-lnn-cfd-baseline/dns/generate_kolmogorov_dns_fp64.py \
    tools/dns_generator/generate_kolmogorov_dns_fp64.py

# 同步後跑 pytest 確認 N-invariance 仍 PASS
uv run pytest tests/test_grid_independence_ic_alignment.py -v
```

## 修改紀錄

- **2026-05-24**: 加 `ic_mode='spectral_seeded'` + `ic_k_cutoff` 兩個參數
  - 新方法 `_make_spectral_ic`: mode-indexed `SeedSequence` 保證不同 N 在 |k|≤k_cutoff 的共同 modes bit-exact 一致
  - `_align_initial_statistics`: spectral_seeded mode 跳過 iterative alignment（保 N-invariance），改用單一 multiplicative KE rescale 到 `target_initial_ke`
  - CLI: 加 `--ic_mode {band_limited_random, spectral_seeded}` 與 `--ic_k_cutoff`
  - **Backward compat**: `ic_mode='band_limited_random'` 為 default，既有所有訓練資料不受影響
  - 詳見 spec: `knowledge/superpowers/specs/2026-05-24-kolmogorov-re10000-grid-independence-design.md`

## Backup

home-gpu 上修改前已備份：
`~/pi-lnn-cfd-baseline/dns/generate_kolmogorov_dns_fp64.py.bak_pre_gi_2026-05-24`

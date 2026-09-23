"""Paper figure: EXP-245 speedup stack — train wall + KE parity.

5 配置（全 20k 實測，單張 RTX 3090 24GB）：
  PyTorch pi-lnn → JAX+fof → +sensor mini-batch → +taylor(jet) → +folx
雙面板：(a) train wall + speedup×；(b) KE rel-err（同定義）顯示加速不傷精度。

末列 folx（collapsed forward-Laplacian, job 3970）是 2026-08-03 起唯一的 AD 路徑——
ror/taylor/fof 三條分支當日從 code 移除，故前四列為演進紀錄、非現行可切換的模式。

TMLR 格式（textwidth 6.5in, serif）。輸出 vector PDF。
"""
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.journal_style import setup_style, save_figure, PALETTE  # noqa: E402
import matplotlib.pyplot as plt

setup_style("iclr")  # serif (Times) single-column，對齊 TMLR body font

# ── 實測數據（全 20k, 單張 RTX 3090 24GB）──
configs = [
    "PyTorch\npi-lnn",
    "JAX\n+fof",
    "JAX +fof\n+mini-batch",
    "JAX +taylor\n+mini-batch",
    "JAX +folx\n+mini-batch",
]
# train wall (min): 9567/5922/3876/3496/3384 s（末列 job 3970）
wall_min = [159.5, 98.7, 64.6, 58.3, 56.4]
speedup  = [1.00, 1.62, 2.47, 2.74, 2.83]     # vs PyTorch
ke_err   = [5.59, 5.58, 5.61, 5.55, 6.04]     # KE rel-err (%) — 同 aggregate 定義 vs DNS
# baseline 灰、演進三版同色、現行 folx 高亮
colors = ["#7f7f7f", PALETTE[0], PALETTE[2], PALETTE[0], PALETTE[1]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.9))
y = list(range(len(configs)))[::-1]  # 由上而下：PyTorch 在最上

# ── (a) train wall + speedup ──
bars = ax1.barh(y, wall_min, color=colors, height=0.62, edgecolor="black", linewidth=0.5)
ax1.set_yticks(y); ax1.set_yticklabels(configs)
ax1.set_xlabel(r"Train wall, 20k iter (min)")
ax1.set_xlim(0, 195)
for yi, w, s in zip(y, wall_min, speedup):
    tag = f"{w:.0f} min" + ("" if s == 1.0 else f"  ({s:.2f}$\\times$)")
    ax1.text(w + 3, yi, tag, va="center", ha="left", fontsize=6.5)
ax1.set_title(r"(a) Training time", fontsize=8)

# ── (b) KE rel-err parity ──
ax2.barh(y, ke_err, color=colors, height=0.62, edgecolor="black", linewidth=0.5)
ax2.set_yticks(y); ax2.set_yticklabels([])  # 共用左圖標籤
ax2.set_xlabel(r"KE rel-err vs DNS (%)")
ax2.set_xlim(0, 8)
# PyTorch 參考線（parity 標尺）
ax2.axvline(ke_err[0], color="#7f7f7f", linestyle="--", linewidth=0.8, zorder=0)
for yi, e in zip(y, ke_err):
    ax2.text(e + 0.15, yi, f"{e:.2f}", va="center", ha="left", fontsize=6.5)
ax2.set_title(r"(b) Accuracy (parity)", fontsize=8)

fig.tight_layout(w_pad=1.2)
save_figure(fig, _REPO_ROOT / "paper/tmlr-format/figures/speedup_stack")
print("saved speedup_stack.pdf/.png")

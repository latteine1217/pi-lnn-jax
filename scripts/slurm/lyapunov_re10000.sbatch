#!/bin/bash
#SBATCH --job-name=jax_lyapunov_re10k
#SBATCH --output=logs/jax_lyapunov_re10k_%j.out
#SBATCH --error=logs/jax_lyapunov_re10k_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# What:
#   Measure the leading Lyapunov exponent lambda_max of the Re=10000 Kolmogorov
#   system with scripts/diag_lyapunov.py (pseudo-spectral solver + TLM Benettin).
# Why:
#   Check the paper's identification of one eddy-turnover time (t_eddy~2) with one
#   "Lyapunov time" (T/t_eddy=2.51). Reports 1/lambda_max and T=5 in Lyapunov times.
# Ladder inside this job:
#   (1) selftest       -> solver correctness gates (fast)
#   (2) --fidelity     -> gate 3: solver tracks DNS at short time (same system)
#   (3) ensemble measure -> lambda_max mean +/- std over n_init attractor frames

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn-jax}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:."
export PILNJAX_DATA_ROOT="${PILNJAX_DATA_ROOT:-$HOME/pi-lnn-jax-main}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在；head node 跑 scripts/slurm/setup_gpu_venv.sh" >&2; exit 3; }
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Locate the DNS file: repo-local first, then PILNJAX_DATA_ROOT/data fallback.
DNS_REL="data/dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T20_dt1p95e4_si128_seed42.npy"
if   [ -f "$DNS_REL" ];               then DNS="$DNS_REL"
elif [ -f "$PILNJAX_DATA_ROOT/$DNS_REL" ];   then DNS="$PILNJAX_DATA_ROOT/$DNS_REL"
else echo "[ERR] DNS not found: $DNS_REL nor $PILNJAX_DATA_ROOT/$DNS_REL" >&2; exit 4; fi
echo "[lyapunov] DNS = $DNS"

echo "[lyapunov] $(date '+%F %T')  job=${SLURM_JOB_ID}  node=${SLURMD_NODENAME:-?}"
uv run python -c "import jax; print(f'JAX {jax.__version__} {jax.devices()}')"
# preflight: cuDNN compile smoke (fail fast on the pinned-cuDNN pitfall)
uv run python -c "import jax, jax.numpy as jnp; from jax import lax; lax.conv_general_dilated(jnp.ones((1,8,8,1)), jnp.ones((3,3,1,1)), (1,1), 'SAME', dimension_numbers=('NHWC','HWIO','NHWC')).block_until_ready(); print('[preflight] cuDNN OK')" \
  || { echo '[ERR] cuDNN preflight 失敗 → head node 跑 setup_gpu_venv.sh' >&2; exit 5; }

mkdir -p artifacts/lyapunov

echo "===================== (1) solver selftest ====================="
uv run python -u scripts/diag_lyapunov.py selftest

echo "===================== (2)+(3) measure (fidelity + ensemble) ====================="
# dt=1e-4 < DNS dt(1.95e-4) for integration accuracy; tau=0.1, horizon=60 (long
# enough for lambda to converge), n_init=5 attractor frames (t>6, steady state).
uv run python -u scripts/diag_lyapunov.py measure \
  --init_from_dns "$DNS" --fidelity \
  --t_ss 6.0 --dt 1e-4 --spinup 0.5 \
  --tau 0.1 --horizon 60.0 --n_init 5 --seed 42 \
  --out artifacts/lyapunov/lyapunov_re10000.json

echo "[$(date '+%F %T')] lyapunov measurement 完成"

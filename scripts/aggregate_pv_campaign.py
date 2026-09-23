#!/usr/bin/env python3
"""聚合 placement-variance campaign 15 runs → §6/§7 論文數字。

KE MAPE 由 metric-artifact semantic reader 取得 spatial-mean E(t) 定義。
分組：LES(σ_training)、oracle、random(σ_placement)。輸出 mean±sd、6.2× 比值、Δ、Welch p、Cohen d。
"""
import json, os, math

from pi_lnn_jax.metric_artifact import LegacyInterpretation, MetricUnavailable
from pi_lnn_jax.result_table import (
    KE_T_MAPE_SPATIALMEAN_V1,
    metric_definition_id,
    read_metric_value,
)

BASE = os.path.expanduser("~/pi-lnn-jax/artifacts/kolmogorov")
GROUPS = {
    "LES":    [f"pv_les_s{s}"    for s in (42, 1, 2, 3, 4)],
    "oracle": [f"pv_oracle_s{s}" for s in (42, 1, 2, 3, 4)],
    "random": [f"pv_random_p{s}" for s in (42, 1, 2, 3, 4)],
}
# GAP-1（spacefill FPS 5-seed，jobs 4471-4475）：run 齊全時自動納入；缺 run 時明確列出
# 並跳過該組（不影響原三組數字重現）。判讀準則見
# knowledge/experiments/kolmogorov-placement-uniformity-2026-07-06.md。
OPTIONAL_GROUPS = {
    "spacefill": [f"pv_spacefill_s{s}" for s in (42, 1, 2, 3, 4)],
    # CVT 5-training-seed（固定 CVT-init42 placement，seed {42,1,2,3,4}）：進論文負面數字，
    # 與 spacefill/oracle/les 同協議可蘋果對蘋果。scout 已知 s42=5.61（見
    # knowledge/experiments/kolmogorov-cvt-placement-negative-2026-07-12.md）。
    "cvt": [f"pv_cvt_s{s}" for s in (42, 1, 2, 3, 4)],
}
FIELDS = ["u_rel_err", "v_rel_err", "omega_rel_err", "ke_rel_err", "low_band_rel_err", "div_pred_l2"]


def ke_t_mape(d):
    # read_metric_value 對 KE-MAPE 家族內部委派 read_metric_summary（雙語意規則不變），
    # 直接回 float（非 MetricSummary），故不再取 .value。
    return read_metric_value(
        d,
        KE_T_MAPE_SPATIALMEAN_V1,
        legacy_interpretation=LegacyInterpretation(
            definition_id=KE_T_MAPE_SPATIALMEAN_V1,
            basis="placement-variance campaign used spatial-mean E(t)",
        ),
    )


def load(run):
    f = os.path.join(BASE, run, "final_eval", "metrics.json")
    d = json.load(open(f))
    row = {"run": run, "ke_t_mape": ke_t_mape(d) * 100,
           "T_eval": len(d["metrics_per_t"]), "ckpt_step": d.get("ckpt_step")}
    for k in FIELDS:
        # 走 semantic reader（storage key 由寫側 schema 派生，不手抄）：值在 → 同一個
        # float；genuine null → None（映 nan，維持舊 .get 對 null 的下游 nan 需求）；
        # storage key 漂移／漏寫 → MetricUnavailable raise。舊碼是 print warning 後照樣
        # 續跑並靜默填 nan，會讓漂移的 run 混進 §6/§7 聚合——這裡改成大聲失敗。
        try:
            v = read_metric_value(d, metric_definition_id(k))
        except MetricUnavailable as exc:
            raise MetricUnavailable(f"{run}: {exc}  ({f})") from exc
        row[k] = (float("nan") if v is None else v) * 100
    return row


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):  # sample std, ddof=1
    m = mean(xs); return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def welch(a, b):
    ma, mb, va, vb, na, nb = mean(a), mean(b), sd(a) ** 2, sd(b) ** 2, len(a), len(b)
    t = (ma - mb) / math.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    # Cohen's d（pooled sd）
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    d = (mb - ma) / sp
    return t, df, d


def p_from_t(t, df):
    # 雙尾 p，用 scipy 若可用，否則 Student-t 近似（正規化不完全 β）
    try:
        from scipy import stats
        return 2 * stats.t.sf(abs(t), df)
    except Exception:
        # regularized incomplete beta 近似（連分數）
        x = df / (df + t * t)
        def betacf(a, b, x):
            MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
            qab, qap, qam = a + b, a + 1, a - 1
            c = 1.0; dd = 1 - qab * x / qap
            if abs(dd) < FPMIN: dd = FPMIN
            dd = 1 / dd; h = dd
            for mm in range(1, MAXIT):
                m2 = 2 * mm
                aa = mm * (b - mm) * x / ((qam + m2) * (a + m2))
                dd = 1 + aa * dd
                if abs(dd) < FPMIN: dd = FPMIN
                c = 1 + aa / c
                if abs(c) < FPMIN: c = FPMIN
                dd = 1 / dd; h *= dd * c
                aa = -(a + mm) * (qab + mm) * x / ((a + m2) * (qap + m2))
                dd = 1 + aa * dd
                if abs(dd) < FPMIN: dd = FPMIN
                c = 1 + aa / c
                if abs(c) < FPMIN: c = FPMIN
                dd = 1 / dd; delta = dd * c; h *= delta
                if abs(delta - 1) < EPS: break
            return h
        a, b = df / 2, 0.5
        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        bt = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta)
        ib = bt * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - (
            math.exp(b * math.log(1 - x) + a * math.log(x) - lbeta) * betacf(b, a, 1 - x) / b)
        return ib  # = P(T>|t|)*2 for two-sided since x maps symmetric tail


rows = {g: [load(r) for r in runs] for g, runs in GROUPS.items()}
for g, runs in OPTIONAL_GROUPS.items():
    missing = [r for r in runs
               if not os.path.exists(os.path.join(BASE, r, "final_eval", "metrics.json"))]
    if missing:
        print(f"[optional group '{g}'] 缺 {len(missing)}/{len(runs)} run（{missing}）→ 本次跳過")
    else:
        rows[g] = [load(r) for r in runs]

# 跨 run 協定一致性：T_eval 與 ckpt_step 必須全 campaign 相同（混入舊 ckpt / 不同
# stride 的 final_eval 會讓聚合數字無效）
_all = [r for rs in rows.values() for r in rs]
_tset = {r["T_eval"] for r in _all}
assert len(_tset) == 1, f"T_eval 不一致：{ {r['run']: r['T_eval'] for r in _all} }"
_sset = {r["ckpt_step"] for r in _all if r["ckpt_step"] is not None}
assert len(_sset) <= 1, f"ckpt_step 不一致：{ {r['run']: r['ckpt_step'] for r in _all} }"
assert all(not math.isnan(r["ke_t_mape"]) for r in _all), "headline ke_t_mape 出現 NaN"

print("=" * 78)
print("每 run ke_t_mape (KE MAPE %)")
print("=" * 78)
for g, rs in rows.items():
    for r in rs:
        print(f"  {r['run']:<16} ke_t_mape={r['ke_t_mape']:.3f}  "
              f"u={r['u_rel_err']:.2f} v={r['v_rel_err']:.2f} ω={r['omega_rel_err']:.2f} "
              f"ke_field={r['ke_rel_err']:.2f} div={r['div_pred_l2']:.3f}")
    print()

print("=" * 78)
print("分組聚合 (ke_t_mape, n=5)")
print("=" * 78)
agg = {}
for g, rs in rows.items():
    xs = [r["ke_t_mape"] for r in rs]
    agg[g] = (mean(xs), sd(xs), xs)
    print(f"  {g:<8} mean={mean(xs):.3f}  sd={sd(xs):.3f}   values={[round(x,3) for x in xs]}")

les_m, les_sd, les_x = agg["LES"]
ora_m, ora_sd, _ = agg["oracle"]
ran_m, ran_sd, ran_x = agg["random"]

print()
print("=" * 78)
print("§7 placement-variance 導出量")
print("=" * 78)
print(f"  σ_training (LES 組)   = {les_sd:.3f} pp")
print(f"  σ_placement (random)  = {ran_sd:.3f} pp")
print(f"  ratio σ_place/σ_train = {ran_sd/les_sd:.2f}×")
print(f"  Δ = random - LES      = {ran_m - les_m:.3f} pp")
t, df, d = welch(les_x, ran_x)
p = p_from_t(t, df)
print(f"  Welch t={t:.3f}  df={df:.2f}  p={p:.2e}  Cohen d={d:.2f}")

print()
print("=" * 78)
print("§6 tab:main_metrics 各欄 (LES / oracle, mean±sd, n=5)")
print("=" * 78)
for g in ("LES", "oracle"):
    rs = rows[g]
    print(f"  --- {g} ---")
    print(f"    KE MAPE : {mean([r['ke_t_mape'] for r in rs]):.2f} ± {sd([r['ke_t_mape'] for r in rs]):.2f}")
    for k in ("u_rel_err", "v_rel_err", "omega_rel_err"):
        print(f"    {k:<14}: {mean([r[k] for r in rs]):.2f} ± {sd([r[k] for r in rs]):.2f}")

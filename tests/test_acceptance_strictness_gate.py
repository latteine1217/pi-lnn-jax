"""驗收編排的接線測試（wave2 spec §8.3 第 5 項 + 架構深化候選 2）。

門檻的環境變數名與呼叫慣例散在三處：`tests/_acceptance_strictness.py`（定義）、
`scripts/slurm/_verify_common.sh`（武裝）、兩支驗收模板（呼叫並收返回碼）。
任何一處對不上都**不會有錯誤訊息**——sbatch 照跑、測試照過，門檻只是靜靜地
不存在。這正是本輪審查一再指出的失效方式，所以用測試釘住而不是靠 code review。

編排收攏之後（`_verify_common.sh`），釘的位置也跟著移動：武裝在共用層驗一次，
各模板只驗「有沒有用正確的 case 與該案的 replay 測試呼叫它」。
"""
from __future__ import annotations

import pytest

import _acceptance_strictness
from _paths import REPO_ROOT

_COMMON = REPO_ROOT / "scripts" / "slurm" / "_verify_common.sh"

#: (驗收模板, 該案的 case 名, 該案的 replay 測試檔)
GATED_TEMPLATES = [
    ("scripts/slurm/verify_cpu_ab.sbatch.tmpl", "kolmogorov", "tests/test_pipeline_replay.py"),
    ("scripts/slurm/verify_cyl_cpu_ab.sbatch.tmpl", "cylinder", "tests/test_cylinder_replay.py"),
]


def test_shared_orchestration_arms_the_gate_with_the_real_variable_names():
    """共用層是唯一武裝門檻的地方——變數名打錯，門檻會靜靜地不存在。"""
    src = _COMMON.read_text()

    assert f"{_acceptance_strictness.ENV_VAR}=1" in src, (
        f"_verify_common.sh 未以正確的變數名武裝門檻（應含 {_acceptance_strictness.ENV_VAR}=1）")
    assert _acceptance_strictness.CASE_ENV_VAR in src, (
        f"_verify_common.sh 未把 case 傳給 {_acceptance_strictness.CASE_ENV_VAR}"
        "——沒有它，跨案 fixture 缺資料會被誤判為弱路徑")
    assert "tests/test_init_digest.py" in src, (
        "初始化 digest 的比對未納入門檻——它的兩條 skip 路徑在開發機上都是常態，"
        "不在驗收 job 上釘住就可能從寫好之後再也沒執行過（job 4781 的教訓）")


@pytest.mark.parametrize("tmpl,case,test_file", GATED_TEMPLATES,
                         ids=[t[0].split("/")[-1] for t in GATED_TEMPLATES])
def test_template_invokes_the_gate_for_its_own_case(tmpl, case, test_file):
    src = (REPO_ROOT / tmpl).read_text()

    assert "source scripts/slurm/_verify_common.sh" in src, f"{tmpl} 未載入共用編排"
    assert f'vc_weak_path_gate "$HEAD_WT" {case} {test_file}' in src, (
        f"{tmpl} 未以正確的 case（{case}）與 replay 測試（{test_file}）呼叫門檻")
    assert "RS=$?" in src and "RS |" in src, (
        f"{tmpl} 取了門檻的返回碼卻未納入最終 exit code——那等於只是印出來")


@pytest.mark.parametrize("tmpl,_case,_test", GATED_TEMPLATES,
                         ids=[t[0].split("/")[-1] for t in GATED_TEMPLATES])
def test_template_does_not_reimplement_shared_orchestration(tmpl, _case, _test):
    """收攏之後，模板不該再自己寫一份共用段落。

    Why 釘這條：重複回流是最容易發生的退步——下一個人趕時間，直接把某段複製
    回模板改一改，兩份就再度開始漂移，而沒有任何東西會報錯。
    """
    src = (REPO_ROOT / tmpl).read_text()
    reimplemented = [
        marker for marker in (
            "ab_compare.py determinism",     # 應走 vc_determinism
            "拒絕刪除非預期路徑",             # 應走 vc_cleanup
            "uv 不在 PATH",                   # 應走 vc_preflight
            "必須提供 EXPECT_BASE",           # 應走 vc_require_expect_base
        ) if marker in src
    ]

    assert not reimplemented, (
        f"{tmpl} 自己重寫了已收進 _verify_common.sh 的段落：{reimplemented}")

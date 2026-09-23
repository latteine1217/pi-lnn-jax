# shellcheck shell=bash
# 驗收 sbatch 的共用編排（由 verify_*.sbatch.tmpl `source`）。
#
# What: 兩支 fresh-run 對拍模板中**逐字相同**的那些段落：前置檢查、環境、
#       輸出目錄清空、determinism control、弱路徑門檻、對拍呼叫。
#
# Why: 判定層（scripts/_common/ab_compare.py）早已是深模組——介面是「回傳問題
#      清單」，實作含四類比對面與自證，22 條測試背書。摩擦不在那裡，在**編排**：
#      兩支模板曾有 34% 逐行重複，於是每新增一段驗收面都得手動套兩三次。
#      2026-08-01 那一輪就套了三次（弱路徑門檻、ledger-off、分案），
#      靠的是「我記得要改兩個地方」——沒有任何東西擋住漏改。
#
# 刻意**不**收進來的：`record()` 與 fixture→config 對照。兩案的錄製方式本就不同
#      （一案有 --artifacts_dir、ledger 直接落點；另一案沒有，ledger 固定落在
#      artifacts/ 再搬走），把它硬塞進共用層只會長出 `if case ==` 分支——
#      那正是 pipeline 兩案刻意不共用 run_loop 的同一個理由。
#
# 判定一律走 ab_compare.py，**不得**寫回 heredoc：那裡的程式碼只在 job 內執行、
# 本機測試永遠碰不到，而那正是要拿它來背書 bit-identical 的地方。

vc_require_expect_base () {
    # Why 由呼叫端提供而非在此推導：要斷言的正是「BASE worktree 真的是基準分支」。
    # 從 worktree 自己讀出來再拿去比它自己，等於沒有斷言。
    [ -n "${EXPECT_BASE:-}" ] || {
        echo "[ERR] 必須提供 EXPECT_BASE（預期的基準 commit）" >&2; exit 6; }
}

vc_common_env () {
    export PYTHONUNBUFFERED=1
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    export UV_OFFLINE=1          # compute node 無網路；違反此假設要立即 fail
    # `uv run` 預設會先同步依賴。在「compute node 離線 + .venv 已備妥」的前提下
    # （AGENTS.md §6），那個同步永遠只能失敗或無事可做——TD-27。實測 job 5515：
    # 有人把 pytest-xdist 加進 dev 依賴後，段 (0) determinism 在訓練還沒開始前
    # 就 rc=1/1，錯誤是 uv 抓不到 wheel，與對拍邏輯無關。放在這裡而非三支
    # template 各寫一次：三支都 source 本檔並呼叫 vc_common_env。
    export UV_NO_SYNC=1
    export JAX_PLATFORMS=cpu     # GPU 上同碼兩跑 digest 就不同（job 4729）
    export PILNN_RNG_LEDGER=1
}

vc_preflight () {                # $1=HEAD_WT  $2=BASE_WT
    command -v uv >/dev/null 2>&1 || { echo "[ERR] uv 不在 PATH" >&2; exit 2; }
    [ -d "$2" ] || { echo "[ERR] BASE worktree 不存在：$2" >&2; exit 3; }
    [ -x "$1/.venv/bin/python" ] || { echo "[ERR] HEAD .venv 缺" >&2; exit 4; }
    [ -x "$2/.venv/bin/python" ] || { echo "[ERR] BASE .venv 缺（head node 跑 uv sync）" >&2; exit 5; }
}

vc_cleanup () {                  # $1=HEAD_WT  $2=BASE_WT  其餘=artifacts 子目錄名
    local head="$1" base="$2"; shift 2
    local wt sub target
    # 先驗子目錄**名字本身**，再組路徑。
    # Why 不能只比對組好的路徑：把 "$sub" 放進 case 的 pattern 會讓守衛恆真
    # （pattern 由待驗的值自己構成），`../PRECIOUS` 這種逃逸照樣通過——
    # 原版用字面 alternatives 才擋得住。名字限定為單層識別字，逃逸就無從構造。
    for sub in "$@"; do
        case "$sub" in
            *[!A-Za-z0-9_]*|"") echo "[ERR] 子目錄名不合法（僅允許 [A-Za-z0-9_]）：$sub" >&2; exit 9 ;;
        esac
    done
    for wt in "$head" "$base"; do
        for sub in "$@"; do
            target="$wt/artifacts/$sub"
            rm -rf "$target"
        done
    done
    echo "[cleanup] 兩個 worktree 的 artifacts/{$(printf '%s,' "$@" | sed 's/,$//')} 已清空"
    # Why 必須清：orbax 對已存在的 step 直接丟 StepAlreadyExistsError，同一個
    #      artifacts_dir 錄第二次必炸；且 record 是 && 串接，跑掛時搬移不會發生，
    #      前一次 job 的舊檔會留在落點被當成本次結果——兩份相同的舊檔會回報
    #      「相同」，得到一份假的成功紀錄。
}

vc_determinism () {              # $1=HEAD_WT  $2=ledger_a  $3=ledger_b
    ( cd "$1" && PYTHONPATH=. uv run python scripts/_common/ab_compare.py determinism \
        --a "$2" --b "$3" )
}

vc_weak_path_gate () {           # $1=HEAD_WT  $2=case  $3=replay 測試檔
    # replay 有兩條建 ctx 的路；初始化 digest 的比對有兩條 skip 路徑。
    # 兩者在開發機上走弱路徑都是常態，於是「強路徑到底有沒有被執行過」是**機率
    # 事件**。驗收 job 上資料必然在位，故把弱路徑升級為失敗。
    # CASE 區分「不在範圍」（跨案 fixture 缺資料）與「前提沒滿足」。
    ( cd "$1" && PILNN_ACCEPTANCE_STRICT=1 PILNN_ACCEPTANCE_CASE="$2" PYTHONPATH=. \
        uv run python -m pytest "$3" tests/test_init_digest.py -q )
}

vc_ab_compare () {               # $1=HEAD_WT $2=BASE_WT $3=names $4=evidence 其餘=比對面旗標
    local head="$1" base="$2" names="$3" evidence="$4"; shift 4
    ( cd "$head" && PYTHONPATH=. uv run python scripts/_common/ab_compare.py ab \
        --base-root "$base" --head-root "$head" --names "$names" \
        --base-sha "$BASE_SHA" --head-sha "$HEAD_SHA" \
        --expect-base "$EXPECT_BASE" --expect-head "${EXPECT_HEAD:-}" \
        ${evidence:+--evidence "$evidence"} \
        --job-id "${SLURM_JOB_ID:-}" --node "${SLURMD_NODENAME:-}" \
        --recorded-utc "$(date -u '+%FT%TZ')" "$@" )
}

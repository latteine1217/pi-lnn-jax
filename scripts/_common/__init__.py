"""scripts/_common — 只有 scripts/ 需要、不屬於 pi_lnn_jax 函式庫的共用件。

分層原則（見 scripts/CLAUDE.md）：
  - 會與訓練漂移就出錯的東西 → `pi_lnn_jax/`（model factory、ckpt restore、
    config 三元組、資料載入）。訓練端與 eval 端必須是同一個物件。
  - 只有腳本會用到的東西 → 本套件（論文繪圖風格、cylinder placement 共用件）。
    放進函式庫只會讓 pi_lnn_jax 被一次性實驗工具汙染。

匯入方式（scripts/ 內的腳本）：

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _common.cylinder_sensors import cylinder_geometry

`pi_lnn_jax` 已是 editable-installed，直接 `import pi_lnn_jax` 即可，不需要
path 引導；本套件住在 scripts/ 內，才需要上面那一行。
"""

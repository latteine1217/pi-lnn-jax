"""建構期／執行期邊界掃描的共用件（`test_{pipeline,cylinder}_assembly.py` 共用）。

兩案的「不得出現什麼」清單刻意不同（Kolmogorov 禁止長回本地 build_model；
cylinder 反而**必須**繞過 model_factory，見 wave2 spec §6），所以兩個測試檔不合併。
但**掃描器本身**兩案相同——各留一份就是「兩處必須一致卻無機制維持一致」，
而那正是本輪審查點名的味道：實測 Kolmogorov 版比 cylinder 版少了 alias 自證。

本模組只放掃描，不放清單。
"""
from __future__ import annotations

import ast
import pathlib

#: 屬性鏈前綴命中即視為 RNG 消費。
#: 掃描器自證用的正樣本：每一種都是實際可行的繞道寫法。
#: 兩案共用——先前只有 cylinder 有這份清單，Kolmogorov 的屬性鏈掃描因此
#: 沒有正樣本背書（見 module docstring 點名的那個差距）。
RNG_SMELLS = (
    "x = jax.random.split(k, 2)",
    "x = np.random.RandomState(0)",
    "x = numpy.random.rand(3)",     # 未 alias 的 numpy——先前兩案的私有清單都漏了它
    "import jax.random",
    "import jax.random as jr",
    "import numpy.random as npr",
    "from jax.random import split",
    "from jax import random",
    "from jax import random as jr",
    "from numpy import random",
)


FORBIDDEN_ATTR_PREFIXES = (
    ("jax", "random"),
    ("np", "random"),
    ("numpy", "random"),
)


def attr_chain(node: ast.AST) -> tuple[str, ...]:
    """把 `a.b.c` 攤成 `('a','b','c')`。"""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def init_attribute_uses(tree: ast.AST) -> list[tuple[int, str]]:
    """所有 `.init` 屬性**存取**（不限於直接呼叫）。

    Why 不用 `".init(" in src`：`init_fn = model.init` 再 `init_fn(key, …)`
    完全避開那個子字串，卻照樣消耗主 RNG。屬性存取才是真正的訊號——
    綁成區域名也得先存取一次。
    """
    return [(n.lineno, ".".join(attr_chain(n)))
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr == "init"]


def rng_consuming_functions(pkg_root: pathlib.Path,
                            skip_parts: tuple[str, ...] = ("pipeline",)) -> set[str]:
    """掃出套件內**自己直接消耗 RNG** 的函式名。

    Why 由原始碼推導而非手寫清單：手寫的會漂移——新增一個取樣 helper 而忘了
    加進清單，守衛就安靜失效。這裡每次執行都重新推導。

    只做一層：實測本套件的傳遞閉包不增加任何函式（呼叫 RNG helper 的函式
    自己也都直接碰 RNG），故一層即為完整答案；哪天不再成立，
    `test_*_rng_helper_closure_is_flat` 會紅。
    """
    out: set[str] = set()
    for path in sorted(pkg_root.rglob("*.py")):
        if any(p in skip_parts for p in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:                      # 掃描器不該因單一壞檔而整個失效
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(isinstance(n, ast.Attribute)
                   and any(attr_chain(n)[:len(b)] == b for b in FORBIDDEN_ATTR_PREFIXES)
                   for n in ast.walk(fn)):
                out.add(fn.name)
    return out


def rng_helper_call_closure(pkg_root: pathlib.Path,
                            skip_parts: tuple[str, ...] = ("pipeline",)) -> set[str]:
    """`rng_consuming_functions` 的傳遞閉包（呼叫了清單內函式的函式也算）。"""
    direct = rng_consuming_functions(pkg_root, skip_parts)
    calls: dict[str, set[str]] = {}
    for path in sorted(pkg_root.rglob("*.py")):
        if any(p in skip_parts for p in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                calls.setdefault(fn.name, set()).update(
                    c.func.id for c in ast.walk(fn)
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Name))
    closure = set(direct)
    while True:
        grown = {f for f, cs in calls.items() if cs & closure} - closure
        if not grown:
            return closure
        closure |= grown


def indirect_rng_uses(tree: ast.AST, denylist: set[str]) -> list[tuple[int, str]]:
    """匯入或呼叫了 denylist 內的函式——即「不寫 jax.random，改叫別人去抽」。"""
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            hits += [(node.lineno, f"from {node.module} import {a.name}")
                     for a in node.names if a.name in denylist]
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in denylist):
            hits.append((node.lineno, f"{node.func.id}(…)"))
    return hits


def rng_hits(src: str, forbidden=None) -> list[tuple[int, str]]:
    """回傳 src 內所有 RNG 消費／RNG import 的位置（兩案共用）。

    用 AST 而非 grep：字串或註解裡出現 'jax.random' 不該算違規，
    真正的屬性存取與 import 陳述才算。
    """
    tree = ast.parse(src)
    hits: list[tuple[int, str]] = []

    # 屬性鏈檢查
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = attr_chain(node)
            for bad in (forbidden or FORBIDDEN_ATTR_PREFIXES):
                if chain[:len(bad)] == bad:
                    hits.append((node.lineno, ".".join(chain)))

    # import 層檢查——捕捉 alias import 繞過
    known_array_libs = {"jax", "numpy", "np"}

    for node in ast.walk(tree):
        # ast.Import：import jax.random, import numpy.random as npr 等
        if isinstance(node, ast.Import):
            for alias in node.names:
                # alias.name 是完整名稱，如 "jax.random"
                if "random" in alias.name.split("."):
                    import_str = f"import {alias.name}"
                    if alias.asname:
                        import_str += f" as {alias.asname}"
                    hits.append((node.lineno, import_str))

        # ast.ImportFrom：from X import Y
        if isinstance(node, ast.ImportFrom):
            # 情況 1：from jax.random import ...（module 路徑包含 "random"）
            if node.module and "random" in node.module.split("."):
                hits.append((node.lineno, f"from {node.module} import ..."))
            # 情況 2：from jax import random 或 from numpy import random
            elif node.module:
                top_lib = node.module.split(".")[0]
                if top_lib in known_array_libs:
                    for alias in node.names:
                        if alias.name == "random":
                            import_str = f"from {node.module} import {alias.name}"
                            if alias.asname:
                                import_str += f" as {alias.asname}"
                            hits.append((node.lineno, import_str))
    return hits

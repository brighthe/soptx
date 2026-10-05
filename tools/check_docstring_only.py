"""核对 Python 改动只涉及 docstring 与注释.

把基准版本与工作区中同一文件各自解析为 AST, 去掉所有裸字符串语句 (docstring
以及不在首行、不起作用的字符串表达式) 后比较 ``ast.dump``. 注释不进入 AST, 因此
只改 docstring 与注释的文件两侧必然相同; 任何代码改动 (包括参与运算的字符串常量、
默认参数、装饰器) 都会使两侧不同. 把误放在第二条语句的说明字符串挪成 docstring
也视为只改 docstring.

用法::

    python tools/check_docstring_only.py                  # 对比 HEAD 与工作区中改动的 .py
    python tools/check_docstring_only.py --base main      # 对比 main
    python tools/check_docstring_only.py src/soptx/sparse # 只看指定路径
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STATEMENT_LISTS = ("body", "orelse", "finalbody")


def is_bare_string(statement: ast.stmt) -> bool:
    """判断语句是否为不起作用的裸字符串表达式 (含 docstring)."""
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def strip_docstrings(tree: ast.Module) -> ast.Module:
    """原地删除所有语句块中的裸字符串表达式, 返回同一棵树.

    Parameters
    ----------
    tree : ast.Module
        待处理的语法树.

    Returns
    -------
    ast.Module
        删除 docstring 与其余裸字符串语句后的语法树.
    """
    for node in ast.walk(tree):
        for field in STATEMENT_LISTS:
            statements = getattr(node, field, None)
            if isinstance(statements, list):
                statements[:] = [s for s in statements if not is_bare_string(s)]
    return tree


def first_difference(old: ast.Module, new: ast.Module) -> int:
    """返回两棵已去 docstring 的树中第一条不同的顶层语句在新文件中的行号.

    Parameters
    ----------
    old, new : ast.Module
        基准版本与工作区版本的语法树.

    Returns
    -------
    int
        新文件中的行号; 差异出现在末尾多出或缺少的语句时返回最后一条语句的行号,
        新文件为空时返回 1.
    """
    for old_node, new_node in zip(old.body, new.body):
        if ast.dump(old_node) != ast.dump(new_node):
            return new_node.lineno
    if new.body:
        return new.body[min(len(old.body), len(new.body)) - 1].lineno
    return 1


def changed_files(base: str, paths: list[str]) -> list[str]:
    """列出相对基准版本有改动的 .py 文件 (仓库相对路径).

    Parameters
    ----------
    base : str
        基准 git 版本.
    paths : list of str
        限定的路径; 为空时不限定.

    Returns
    -------
    list of str
        有改动的 .py 文件, 含已跟踪的新增与删除文件; 未跟踪的新文件不在其中.
    """
    command = ["git", "diff", "--name-only", base, "--"]
    command += paths if paths else ["."]
    output = subprocess.run(
        command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in output.splitlines() if line.endswith(".py")]


def read_base(base: str, relative: str) -> str | None:
    """读取基准版本中的文件内容; 文件在基准版本中不存在时返回 None."""
    result = subprocess.run(
        ["git", "show", f"{base}:{relative}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else None


def main() -> int:
    """比较改动文件去掉 docstring 后的 AST, 有代码改动时返回 1."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="HEAD", help="基准 git 版本, 默认 HEAD.")
    parser.add_argument("paths", nargs="*", help="限定检查的路径.")
    arguments = parser.parse_args()

    errors: list[str] = []
    files = changed_files(arguments.base, arguments.paths)
    for relative in files:
        path = REPOSITORY_ROOT / relative
        old_text = read_base(arguments.base, relative)
        if old_text is None:
            errors.append(f"{relative}: 基准版本中不存在 (新增文件)")
            continue
        if not path.exists():
            errors.append(f"{relative}: 工作区中已删除")
            continue
        old = strip_docstrings(ast.parse(old_text, filename=relative))
        new = strip_docstrings(ast.parse(path.read_text(encoding="utf-8"), filename=relative))
        if ast.dump(old) != ast.dump(new):
            errors.append(f"{relative}:{first_difference(old, new)}: 代码有改动")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"{len(files)} 个改动的 .py 文件只涉及 docstring 与注释.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

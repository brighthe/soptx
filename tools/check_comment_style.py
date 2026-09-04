"""注释与 docstring 风格棘轮门禁.

看住两条约定: 注释与 docstring 使用英文半角标点; ``src/`` 下公开 API 必须有
docstring. 存量违规以棘轮基线常数记录, 数量超过基线即失败; 清理存量后应同步
下调基线常数, 使其只降不升.
"""

from __future__ import annotations

import argparse
import ast
import io
from pathlib import Path
import sys
import tokenize

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("src", "tools", "tests", "examples", "experiments")
SKIP_PARTS = {"__pycache__", "legacy", "old"}
FULLWIDTH_PUNCTUATION = "，。；：！？（）【】“”‘’"

# 棘轮基线: 2026-08-28 的存量违规数. 只允许下调, 不允许上调.
FULLWIDTH_BASELINE = 645
MISSING_DOCSTRING_BASELINE = 314


def iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        base = REPOSITORY_ROOT / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if SKIP_PARTS.isdisjoint(path.parts):
                files.append(path)
    return files


def docstring_expressions(tree: ast.Module) -> dict[int, ast.Expr]:
    """返回节点 id 到其 docstring 表达式的映射."""
    found: dict[int, ast.Expr] = {}
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                found[id(node)] = body[0]
    return found


def collect_violations() -> tuple[list[str], list[str]]:
    fullwidth: list[str] = []
    missing: list[str] = []
    for path in iter_python_files():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        try:
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type == tokenize.COMMENT and any(
                    c in FULLWIDTH_PUNCTUATION for c in token.string
                ):
                    fullwidth.append(f"{relative}:{token.start[0]} [comment]")
        except tokenize.TokenizeError:
            pass
        try:
            tree = ast.parse(text, filename=relative)
        except SyntaxError:
            continue
        documented = docstring_expressions(tree)
        for expression in documented.values():
            for offset, line in enumerate(expression.value.value.splitlines()):
                if any(c in FULLWIDTH_PUNCTUATION for c in line):
                    fullwidth.append(
                        f"{relative}:{expression.lineno + offset} [docstring]"
                    )
        if not relative.startswith("src/"):
            continue
        if id(tree) not in documented:
            missing.append(f"{relative}:1 <module>")
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                if not node.name.startswith("_") and id(node) not in documented:
                    missing.append(f"{relative}:{node.lineno} {node.name}")
    return fullwidth, missing


def report(
    label: str, hits: list[str], baseline: int, list_all: bool
) -> bool:
    count = len(hits)
    if count > baseline:
        print(
            f"{label}: {count} 处违规, 超过棘轮基线 {baseline}; "
            "请修正新增违规, 不要上调基线.",
            file=sys.stderr,
        )
        if not list_all:
            print("  (用 --list 查看全部位置)", file=sys.stderr)
        return False
    if count < baseline:
        print(
            f"{label}: {count} 处存量违规, 低于基线 {baseline}; "
            "请把脚本中的基线常数下调至当前值."
        )
    else:
        print(f"{label}: {count} 处存量违规, 与棘轮基线持平.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="列出全部违规位置."
    )
    arguments = parser.parse_args()
    fullwidth, missing = collect_violations()
    if arguments.list:
        for hit in fullwidth:
            print(f"fullwidth  {hit}")
        for hit in missing:
            print(f"docstring  {hit}")
    ok_fullwidth = report(
        "全角标点(注释/docstring)", fullwidth, FULLWIDTH_BASELINE, arguments.list
    )
    ok_missing = report(
        "src/ 公开 API 缺 docstring",
        missing,
        MISSING_DOCSTRING_BASELINE,
        arguments.list,
    )
    return 0 if ok_fullwidth and ok_missing else 1


if __name__ == "__main__":
    raise SystemExit(main())

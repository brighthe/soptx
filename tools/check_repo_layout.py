"""仓库布局门禁.

看住两条约定: ``docs/known-issues/`` 恰好只有两份文档; 每个 ``examples/``
主题目录具备 ``README.md`` 与 ``results_analysis.md`` 两文档结构.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

KNOWN_ISSUES_DIR = REPOSITORY_ROOT / "docs" / "known-issues"
KNOWN_ISSUES_EXPECTED = {"README.md", "fealpy-patches.md"}

EXAMPLES_DIR = REPOSITORY_ROOT / "examples"
REQUIRED_DOCS = ("README.md", "results_analysis.md")
SKIP_DIRS = {"__pycache__"}

# 历史欠账白名单: 这些目录建立时就缺两文档. 补齐文档后必须删除对应条目,
# 白名单只减不增.
MISSING_DOCS_TODO = {"gpu_elasticity", "topopt_platform"}


def check_known_issues() -> list[str]:
    errors: list[str] = []
    actual = {
        path.name for path in KNOWN_ISSUES_DIR.iterdir() if path.is_file()
    }
    for unexpected in sorted(actual - KNOWN_ISSUES_EXPECTED):
        errors.append(
            f"docs/known-issues/{unexpected}: 该目录只允许 README.md 与 "
            "fealpy-patches.md 两份文档, 新内容并入对应文档."
        )
    for absent in sorted(KNOWN_ISSUES_EXPECTED - actual):
        errors.append(f"docs/known-issues/{absent}: 必需文档缺失.")
    return errors


def check_examples_docs() -> list[str]:
    errors: list[str] = []
    for topic in sorted(EXAMPLES_DIR.iterdir()):
        if not topic.is_dir() or topic.name in SKIP_DIRS:
            continue
        missing = [name for name in REQUIRED_DOCS if not (topic / name).is_file()]
        if topic.name in MISSING_DOCS_TODO:
            if not missing:
                errors.append(
                    f"examples/{topic.name}: 两文档已补齐, 请从 "
                    "MISSING_DOCS_TODO 白名单删除该条目."
                )
            continue
        for name in missing:
            errors.append(f"examples/{topic.name}/{name}: 主题目录必需文档缺失.")
    return errors


def main() -> int:
    errors = check_known_issues() + check_examples_docs()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("SOPTX repository layout check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

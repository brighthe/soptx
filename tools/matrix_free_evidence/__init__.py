"""Matrix-Free 线弹性基线的证据流水线

``run.py`` 产出单次运行的原始 JSON, ``validate.py`` 按算例表批量驱动并施加门禁,
``sync_results.py`` 只接受 clean-revision 的原始 JSON 并生成精简证据 JSON。
``contract.py`` 是这三者共用的数值契约: 每个阈值、缺省值与支持范围在那里定义一次,
收紧门禁不会只生效在流水线的一侧。

本包住在 ``tools/`` 而不是示例目录里, 因为它同时是 fealpy fork 的 merge 前门禁
(见 ``docs/known-issues/README.md``), 不只服务于那一个示例。

依赖方向是单向的, 而且方向与直觉相反: **本包不导入示例的任何模块**——制造解已下沉到
``soptx.problems.elasticity``, 阈值就在本包里, 示例目录只剩两个 demo 脚本, 它们反过来导入
本包的 ``contract`` 以保证 demo 里印出的 PASS/FAIL 与正式门禁同源。``layout.py``
仍然知道示例目录的*路径*, 但那只用来定位产物, 不构成 import 依赖。
"""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"

for _directory in (SOURCE_ROOT, REPOSITORY_ROOT):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

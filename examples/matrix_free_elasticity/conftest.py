"""Make the flat example modules importable from ``tests/``.

The example deliberately keeps a flat module layout instead of a package, so
pytest needs both this directory (for ``import cg``) and the repository root
(for ``import soptx``) on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLE_DIR.parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"

for directory in (EXAMPLE_DIR, SOURCE_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from __future__ import annotations

from pathlib import Path
import sys


EXAMPLE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLE_DIR.parents[1]

for path in (EXAMPLE_DIR, REPOSITORY_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

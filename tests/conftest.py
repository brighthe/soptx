"""Put the repository root on ``sys.path`` for the Matrix-Free evidence tests.

Those tests import ``tools.matrix_free_evidence``, which is not an installed
package — it is repository tooling.  ``python -m pytest`` from the repository
root would put the root on ``sys.path`` by itself, but a bare ``pytest`` or an
invocation from elsewhere would not, so make it explicit here.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

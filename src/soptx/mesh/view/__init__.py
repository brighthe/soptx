# 移植自 brighthe/fealpy ``fealpy/mesh/view/__init__.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Public view objects.

This package exposes the root-cell-anchored :class:`MeshView` and the
sector-bound :class:`EntityView`.  ``Mesh`` here is a private transitional
alias to ``MeshView`` retained only for legacy in-repository import sites; the
public top-level ``fealpy.mesh.Mesh`` is the multi-block aggregate.
"""

from .entity_view import EntityView
from .mesh_view import MeshView

# Private compatibility alias.  New code must import ``MeshView`` directly or
# use the top-level ``fealpy.mesh.Mesh`` aggregate for multi-block ownership.
Mesh = MeshView

__all__ = ["EntityView", "MeshView"]

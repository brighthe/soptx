# 移植自 brighthe/fealpy ``fealpy/mesh/schema/__init__.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Public immutable entity Schemas and descriptor/resolver APIs.

Construct concrete Schema values from :mod:`fealpy.mesh.schema.classic` names,
compare them by value, and serialize their identities through
``SchemaDescriptor``.  Geometry interpolation is selected by the Schema value;
independent finite-element reference bases use its ``lagrange_basis_function``
interface without changing geometry identity.
"""

from .classic import *
from .descriptor import *
from .entity_schema import *
from .local_entity import *
from .polygon import *
from .registry import *

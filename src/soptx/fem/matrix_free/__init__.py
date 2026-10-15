"""Matrix-Free 算子: EA 懒装配算子门面.

依赖方向为 operator → (soptx.fem.analyzers).
"""

from .operator import ElasticityEAOperator

__all__ = [
    "ElasticityEAOperator",
]

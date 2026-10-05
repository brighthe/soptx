# 移植自 brighthe/fealpy ``fealpy/mesh/mesh_base.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""网格类别标记: 基于元类 ``__instancecheck__`` 的 ``isinstance`` 判断.

``SimplexMesh``、``TensorMesh`` 等不是可实例化的网格类, 只用于判断一个 ``Mesh``
是否为单纯形网格、张量积网格等.
"""

from .view import Mesh

__all__ = [
    "Mesh",
    "HomogeneousMesh",
    "SimplexMesh",
    "TensorMesh",
    "StructuredMesh",
]

# 已弃用, 将在后续版本中移除
class _HomogeneousMeshMeta(type):
    """齐次网格: 任何 ``Mesh`` 实例都判为真."""
    def __instancecheck__(self, instance):
        return isinstance(instance, Mesh)

HomogeneousMesh = _HomogeneousMeshMeta("HomogeneousMesh", (Mesh,), {})


class _SimplexMeshMeta(type):
    """单纯形网格: ``Mesh`` 实例且 ``is_simplex_mesh()`` 为真."""
    def __instancecheck__(self, instance):
        return isinstance(instance, Mesh) and instance.is_simplex_mesh()

SimplexMesh = _SimplexMeshMeta("SimplexMesh", (Mesh,), {})


class _TensorMeshMeta(type):
    """张量积网格: ``Mesh`` 实例且 ``is_tensor_mesh()`` 为真."""
    def __instancecheck__(self, instance):
        return isinstance(instance, Mesh) and instance.is_tensor_mesh()

TensorMesh = _TensorMeshMeta("TensorMesh", (Mesh,), {})


class _StructuredMeshMeta(type):
    """结构网格: 目前任何 ``Mesh`` 实例都判为真."""
    def __instancecheck__(self, instance):
        return isinstance(instance, Mesh)
    # TODO: 有了结构网格实现后再修改判定

StructuredMesh = _StructuredMeshMeta("StructuredMesh", (Mesh,), {})

# -*- coding: utf-8 -*-
"""全装配层级 (full assembly, FA).

装配期就把 {K_e} 求和成一个全局稀疏矩阵 A, 之后每次作用都是一次 SpMV. 常驻内存
最大, 但拿到的是显式矩阵, 直接解法与代数多重网格这类需要读矩阵元的算法只有在这
一层级可用.

与 MFEM 的 ``AssemblyLevel::FULL`` 对应.

Notes
-----
``operator`` 返回的是稀疏矩阵本身而不是层级对象: FA 的下游 (直接解法, ``to_dense``,
矩阵加法, 对称消元) 消费的就是矩阵, 层级对象只在装配这一侧有意义. 其余层级没有可
交出去的矩阵, ``operator`` 返回层级对象自身.
"""

from typing import Any, Optional, Union

from fealpy.sparse import COOTensor, CSRTensor
from fealpy.typing import TensorLike

from soptx.fem.bilinear_form import BilinearForm
from soptx.solvers.base import operator_diagonal

from .base import AssemblyLevelExtension
from .registry import register_level


@register_level('fa')
class FullAssembly(AssemblyLevelExtension):
    """FA 层级: 常驻全局稀疏矩阵, 作用时走一次 SpMV.

    Parameters
    ----------
    space : 该双线性型所在的函数空间.
    matrix : 装配好的全局稀疏矩阵.
    pattern : 装配用的 CSR 拓扑骨架, 交回给调用方以便下一次装配复用; 可为 None.
    """

    level = 'fa'

    def __init__(self,
                space,
                matrix: Union[CSRTensor, COOTensor],
                pattern: Optional[Any] = None,
            ) -> None:
        super().__init__(spaces=(space, ), shape=tuple(matrix.shape))
        self._matrix = matrix
        self._pattern = pattern

    @classmethod
    def build(cls,
            space,
            integrator,
            pattern: Optional[Any] = None,
            **kwargs: Any,
        ) -> "FullAssembly":
        """从积分子装配出全局稀疏矩阵.

        Parameters
        ----------
        space : 该双线性型所在的函数空间.
        integrator : 积分子, 由 ``BilinearForm`` 负责逐单元积分与散加.
        pattern : 已有的 CSR 拓扑骨架, 为 None 时由 ``BilinearForm`` 首次装配时建好.
        """
        bform = BilinearForm(space, pattern=pattern)
        bform.add_integrator(integrator)

        matrix = bform.assembly(format='csr')

        return cls(space=space, matrix=matrix, pattern=bform.pattern)

    @property
    def matrix(self) -> Union[CSRTensor, COOTensor]:
        """常驻的全局稀疏矩阵"""
        return self._matrix

    @property
    def pattern(self) -> Optional[Any]:
        """本次装配用的 CSR 拓扑骨架, 供下一次装配复用"""
        return self._pattern

    @property
    def operator(self) -> Union[CSRTensor, COOTensor]:
        """交给调用方当作算子 A 使用的对象: FA 下是稀疏矩阵本身"""
        return self._matrix

    def __matmul__(self, x: TensorLike) -> TensorLike:
        """算子作用 y = A x, 一次 SpMV"""
        return self._matrix @ x

    def diagonal(self) -> TensorLike:
        """取算子对角: 直接从稀疏矩阵读.

        Returns
        -------
        diag : (gdof, ) 的 L 向量; 分布式下矩阵本身就是 rank 本地的, 故同样未归约.
        """
        return operator_diagonal(self._matrix)

    def update(self, matrix: Union[CSRTensor, COOTensor]) -> None:
        """替换常驻的全局矩阵, 拓扑骨架不变"""
        self._matrix = matrix

    def persistent_bytes(self) -> int:
        """常驻内存字节数: 非零元加索引数组"""
        matrix = self._matrix
        if hasattr(matrix, 'crow'):
            arrays = (matrix.crow, matrix.col, matrix.values)
        else:
            arrays = (matrix.indices, matrix.values)

        return sum(int(a.nbytes) for a in arrays if a is not None)

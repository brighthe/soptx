# -*- coding: utf-8 -*-
"""常量积分子 (const integrator).

把一份已经算好的局部张量包装成积分子, 使它能被 ``Form`` 的装配流程接手.

``Form`` 在装配时只向积分子要两样东西 (见 FEALPy ``Form._assembly_kernel``):
``assembly(space)`` 给出的 (NC, ldof) 局部张量, 与 ``to_global_dof(space)`` 给出的
单元-自由度映射. 它不关心这两份数据是现场积出来的还是外部给定的. 本类就利用这一
点: 构造时收下什么, 两个方法就原样返回什么, 不做任何积分, ``space`` 参数进来即被
忽略.

两个典型用法
------------
外部数组进 Form
    矩阵自由算子作用完得到的 E 向量 ``y_E`` 不是任何积分子算出来的, 包成本类即可
    交给 ``soptx.fem.LinearForm`` 散加成 L 向量.
真积分子的固化
    把一个真积分子算一次, 结果冻结下来, 之后不再重复积分. EA 层级常驻的单元矩阵
    {K_e} 就是这样来的.

与 FEALPy ``ConstIntegrator`` 的关系
------------------------------------
语义一致, 实现独立: 本类只继承 FEALPy 的 ``Integrator`` 基类 (``Form.__lshift__``
与 ``GroupIntegrator`` 都按该基类做 isinstance 判断), 不继承 FEALPy 的
``ConstIntegrator``. 一处刻意不同: FEALPy 那份的构造函数仍按旧签名调
``super().__init__('assembly', False, False)``, 字符串落进了现签名的 ``keep_data``
形参, 本类直接写 ``keep_data=False``.
"""

from typing import Any, Generic, Optional, TypeVar

from fealpy.fem.integrator import Integrator
from fealpy.typing import Index, TensorLike

_GT = TypeVar('_GT')
_OpIndex = Optional[Index]


class ConstIntegrator(Integrator, Generic[_GT]):
    """以给定值充当积分结果的积分子.

    Parameters
    ----------
    value : (NC, ldof) 或带额外维度的局部张量, 即 ``assembly`` 的返回值. 第 0 维必须
        是实体维.
    to_gdof : 单元-自由度映射, 即 ``to_global_dof`` 的返回值; 混合型可以是映射的元组.
        只在调用 ``to_global_dof`` 时才需要, 因此允许为 None. 默认为 None.

    Attributes
    ----------
    value : 构造时收下的局部张量.
    to_gdof : 构造时收下的单元-自由度映射.

    Notes
    -----
    积分区域对本类无意义: 值已经给定, 再选实体子集也不会改变它. ``set_region`` 因此
    是空操作, 调用它说明上游把本类当成了真积分子, 属于用法错误, 直接报错而不是像
    FEALPy 那样只记一条警告.

    ``indices`` 子集在本类里退化成对给定值直接切片: 第 0 维是实体维, 按 ``Form`` 的
    分块装配 (``UniformSplitter``) 约定, 局部张量与自由度映射按同一组下标取子集.
    """

    def __init__(self, value: TensorLike, to_gdof: Optional[_GT] = None) -> None:
        super().__init__(keep_data=False)
        self.value = value
        self.to_gdof = to_gdof
        self._region = slice(None)

    def set_region(self, region: Any, /):
        """积分区域对本类无意义, 直接报错.

        Raises
        ------
        RuntimeError
            总是抛出: 值已给定, 设置积分区域不会改变任何结果.
        """
        raise RuntimeError(
            f"{self.__class__.__name__} 的值已经给定, 设置积分区域不会改变结果; "
            "需要子集请在构造前对 value 与 to_gdof 自行切片"
        )

    def to_global_dof(self, space, /, indices: _OpIndex = None) -> _GT:
        """返回构造时收下的单元-自由度映射.

        Parameters
        ----------
        space : 函数空间, 本类忽略.
        indices : 实体子集下标, None 表示全部. 默认为 None.

        Returns
        -------
        构造时给的映射, 给了 ``indices`` 时取其第 0 维的子集.

        Raises
        ------
        RuntimeError
            构造时未给 ``to_gdof``.
        """
        if self.to_gdof is None:
            raise RuntimeError(
                f"{self.__class__.__name__} 构造时未给 to_gdof, 无法给出单元-自由度映射"
            )

        if indices is None:
            return self.to_gdof

        if isinstance(self.to_gdof, (tuple, list)):
            return self.to_gdof.__class__(tg[indices] for tg in self.to_gdof)

        return self.to_gdof[indices]

    def assembly(self, space, /, indices: _OpIndex = None) -> TensorLike:
        """返回构造时收下的局部张量.

        Parameters
        ----------
        space : 函数空间, 本类忽略.
        indices : 实体子集下标, None 表示全部. 默认为 None.

        Returns
        -------
        构造时给的局部张量, 给了 ``indices`` 时取其第 0 维的子集.
        """
        if indices is None:
            return self.value

        return self.value[indices]

    def __repr__(self) -> str:
        shape = tuple(self.value.shape)
        mapped = 'None' if self.to_gdof is None else 'given'

        return f"{self.__class__.__name__}(value_shape={shape}, to_gdof={mapped})"

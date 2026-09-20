# -*- coding: utf-8 -*-
"""装配层级扩展的公共基类.

一个装配层级 (assembly level) 回答的是同一个离散算子 A 以什么形式常驻: FA 存全局
稀疏矩阵, EA 存单元矩阵 {K_e}, PA 存每积分点的几何与材料数据, UA 什么都不存. 四者
表示不同, 但对外只暴露同一组操作: 作用 (``@``), 取对角 (``diagonal``), 随设计变量
更新 (``update``).

这一分层取自 MFEM 的 ``AssemblyLevel`` 与 ``*BilinearFormExtension``: 层级是双线性
型的属性而不是物理的属性, 因此本基类里不出现任何弹性力学的词汇, 也不假定空间是
``TensorFunctionSpace``.

三条契约
--------
作用域
    所有方法都是 rank 本地的, 不含任何跨 rank 通信. 输入输出都是 L 向量.
可加性
    ``diagonal()`` 返回未归约的 L 向量, 交界自由度上只含本 rank 的贡献, 与右端项
    同一个约定, 由调用方决定何时归约.
纯度
    ``__matmul__`` 与 ``diagonal`` 不修改对象状态, 只有 ``update`` 会.
"""

from typing import Tuple

from fealpy.typing import TensorLike


class AssemblyLevelExtension:
    """装配层级扩展的抽象基类.

    Attributes
    ----------
    level : 层级标识, 取 'fa' / 'ea' / 'pa' / 'ua' 之一, 由子类给出.

    Notes
    -----
    ``shape`` 显式存成 ``(vgdof, ugdof)`` 而不是从单一空间推出, 是为了矩形型
    (试探空间与检验空间不同) 和混合型 (如 Hu-Zhang) 留出位置; 同样地 ``_spaces``
    存成元组而非单个空间. 本轮只有方阵单空间的用法, 但接口按宽的写.
    """

    level: str = ''

    @classmethod
    def build(cls, space, integrator, pattern=None, **kwargs) -> "AssemblyLevelExtension":
        """从函数空间与积分子构造该层级, 由注册表统一调用.

        Parameters
        ----------
        space : 该双线性型所在的函数空间.
        integrator : 积分子, 各层级自行决定把它化成什么常驻形式.
        pattern : CSR 拓扑骨架, 只有 FA 用得上, 其余层级接受并忽略.
        """
        raise NotImplementedError

    def __init__(self, spaces: Tuple, shape: Tuple[int, int]) -> None:
        self._spaces = tuple(spaces)
        self._shape = (int(shape[0]), int(shape[1]))

    @property
    def spaces(self) -> Tuple:
        """参与该双线性型的函数空间, 单空间时长度为 1"""
        return self._spaces

    @property
    def shape(self) -> Tuple[int, int]:
        """算子形状 (vgdof, ugdof)"""
        return self._shape

    @property
    def operator(self) -> "AssemblyLevelExtension":
        """交给调用方当作算子 A 使用的对象.

        默认是层级对象自身. FA 覆盖它返回装配好的稀疏矩阵: 那一层的下游消费的是
        矩阵元本身 (直接解法, 对称消元, 矩阵加法), 层级对象只在装配这一侧有意义.
        """
        return self

    def __matmul__(self, x: TensorLike) -> TensorLike:
        """算子作用 y = A x, 输入输出均为 L 向量"""
        raise NotImplementedError

    def diagonal(self) -> TensorLike:
        """取算子对角, 返回未归约的 L 向量"""
        raise NotImplementedError

    def update(self, coef) -> None:
        """随设计变量更新算子的常驻数据, 不重建拓扑"""
        raise NotImplementedError

    def persistent_bytes(self) -> int:
        """常驻内存字节数, 用于层级之间的存储代价比较"""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(level={self.level!r}, shape={self._shape})"

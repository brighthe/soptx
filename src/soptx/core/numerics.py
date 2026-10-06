"""迭代求解器与证据工具共用的数值缺省值.

验证流水线的两端都需要这些常数: 求解器本身, 以及无需构造求解器就要复现一次运行的
收敛判据的证据工具. 证据工具必须能在没有 MPI runtime 的机器上运行, 所以本模块放在
:mod:`soptx.core` (第 0 层, 刻意不导入有限元与 mpi4py), 而不是 ``soptx.fem`` 下:
在那里导入任何模块都会经包的 ``__init__`` 拉起整个有限元栈.

本模块刻意不从 ``soptx.core.__init__`` 再导出: 使用方按完整路径导入, 让依赖在每个
调用点都可见.

表达 *验收门槛* 而非求解器缺省值的数字不属于这里, 应放在定义该门槛的示例或研究中.
"""

from __future__ import annotations


#: :mod:`soptx.solvers.overlap` 中 Krylov 求解器的迭代上限.
DEFAULT_MAX_ITERATIONS = 1000

#: 相对残差容差.
DEFAULT_RTOL = 1.0e-10

#: 绝对残差容差.
DEFAULT_ATOL = 1.0e-12

#: CG 内部两次重算真残差之间的迭代步数.
RESIDUAL_REFRESH = 20

#: 范数出现在分母时使用的下界.
NORM_FLOOR = 1.0e-30


__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_RTOL",
    "NORM_FLOOR",
    "RESIDUAL_REFRESH",
]

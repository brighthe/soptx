"""线弹性有限元分析器的构造工厂 (只服务线弹性, 与算子层级无关).

``build_serial_analyzer`` 与 ``build_distributed_analyzer`` 共用同一份参数装配
``_analyzer_arguments``; ``operator_level`` 取 ``"fa"`` (全局稀疏矩阵全装配) 或
``"ea"`` (无矩阵单元装配), 两个层级都由本模块产出, 因此本模块不隶属于任何单一算子层级.

``preconditioner_level`` 是与 ``operator_level`` 平行的第二根轴, 决定预条件子在哪个
层级上取算子; 为 ``None`` (默认) 时预条件子绑主算子本身. 两根轴可以取不同的值 ——
主算子走 matrix-free 省内存, 预条件子仍可落在 ``"fa"`` 上供需要显式矩阵的后端使用.

本模块导入时不依赖 ``mpi4py``: 仅在构造分布式分析器时延迟导入分布式实现,
确保无 MPI 环境下依然可以安全导入本模块执行单机串行分析.

EA 算子门面见 ``soptx.fem.matrix_free.operator``, 一步正向求解见
``soptx.fem.matrix_free.solve``.
"""

from __future__ import annotations

from typing import Any

from fealpy.functionspace import TensorFunctionSpace

from soptx.fem.analyzers.lagrange_fem_analyzer import LagrangeFEMAnalyzer


def _analyzer_arguments(
    space: TensorFunctionSpace,
    pde: Any,
    material: Any,
    degree: int = 1,
    operator_level: str = "ea",
    assembly_method: str = "standard",
    preconditioner_level: str | None = None,
) -> dict[str, Any]:
    """将物理问题、材料本构与有限元空间组合为标准分析器构造参数字典."""
    return {
        "disp_mesh": space.mesh,
        "pde": pde,
        "material": material,
        "space_degree": degree,
        "integration_order": degree + 3,
        "operator_level": operator_level,
        "preconditioner_level": preconditioner_level,
        "assembly_method": assembly_method,
        "tensor_space": space,
    }


def build_serial_analyzer(
    space: TensorFunctionSpace,
    pde: Any,
    material: Any,
    degree: int = 1,
    operator_level: str = "ea",
    assembly_method: str = "standard",
    preconditioner_level: str | None = None,
) -> LagrangeFEMAnalyzer:
    """构造不含跨进程通信的串行拉格朗日有限元分析器.

    供单 Rank 下的 EA (无矩阵单元装配) 或 FA (全矩阵装配) 求解与基线对比使用,
    直接建立在未经分布式切分的全局空间上.

    参数:
        space (TensorFunctionSpace): 全局向量有限元空间.
        pde (Any): 物理模型或制造解对象 (提供几何区域、体力荷载与边界条件).
        material (Any): 弹性力学本构材料模型 (提供材料常数与计算设备 device).
        degree (int, 可选): 有限元插值多项式阶数. 默认值为 1.
        operator_level (str, 可选): 算子装配级别, 可选 "ea" (无矩阵单元装配) 或 "fa" (全局稀疏矩阵全装配). 默认值为 "ea".
        assembly_method (str, 可选): 单元矩阵的收缩顺序, 可选 ``"standard"``、``"voigt"`` 或 ``"fast"``. 它只改变
            中间张量的规模与峰值内存, 不改变单元矩阵的数值. 默认值为 ``"standard"``.
        preconditioner_level (str | None, 可选): 预条件子取算子的层级, 取值同 ``operator_level``;
            为 ``None`` 时预条件子绑主算子本身. 默认值为 ``None``.

    返回:
        LagrangeFEMAnalyzer: 初始化的串行有限元分析器实例.
    """
    return LagrangeFEMAnalyzer(
        solve_method="scipy",
        **_analyzer_arguments(
            space, pde, material, degree, operator_level, assembly_method,
            preconditioner_level,
        ),
    )


def build_distributed_analyzer(
    space: TensorFunctionSpace,
    pde: Any,
    material: Any,
    degree: int = 1,
    operator_level: str = "ea",
    assembly_method: str = "standard",
    preconditioner_level: str | None = None,
    *,
    dof_comm: Any,
) -> Any:
    """构造重叠副本布局下的分布式线弹性分析器 (支持单 Rank 与多 Rank).

    本函数内部对 ``soptx.fem.analyzers.distributed_analyzer.DistributedElasticityAnalyzer``
    采用延迟导入 (Lazy Import), 确保在未安装 MPI 运行时的环境中仍能正常导入本模块的串行能力.

    参数:
        space (TensorFunctionSpace): 局部子域上的向量有限元空间.
        pde (Any): 物理模型或制造解对象.
        material (Any): 弹性力学本构材料模型.
        degree (int, 可选): 有限元插值多项式阶数. 默认值为 1.
        operator_level (str, 可选): 算子装配级别 ("ea" 或 "fa"). 默认值为 "ea".
        assembly_method (str, 可选): 单元矩阵的收缩顺序, 含义同 ``build_serial_analyzer``. 默认值为 ``"standard"``.
        preconditioner_level (str | None, 可选): 预条件子取算子的层级, 含义同 ``build_serial_analyzer``.
            多 Rank 下不能取 ``"fa"`` —— 对称消元没有重叠归约的插入点. 默认值为 ``None``.
        dof_comm (EntityMPI): 自由度跨进程通信器 (关键字参数).

    返回:
        DistributedElasticityAnalyzer: 初始化的分布式有限元分析器实例.
    """
    from soptx.fem.analyzers.distributed_analyzer import (
        DistributedElasticityAnalyzer,
    )

    return DistributedElasticityAnalyzer(
        dof_comm=dof_comm,
        **_analyzer_arguments(
            space, pde, material, degree, operator_level, assembly_method,
            preconditioner_level,
        ),
    )


__all__ = [
    "build_distributed_analyzer",
    "build_serial_analyzer",
]

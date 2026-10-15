"""线弹性有限元分析器的构造工厂 (只服务线弹性, 与算子层级无关).

``build_serial_analyzer`` 的参数装配由 ``_analyzer_arguments`` 完成;
``operator_level`` 取 ``"fa"`` (全局稀疏矩阵全装配) 或
``"ea"`` (无矩阵单元装配), 两个层级都由本模块产出, 因此本模块不隶属于任何单一算子层级.

``preconditioner_level`` 是与 ``operator_level`` 平行的第二根轴, 决定预条件子在哪个
层级上取算子; 为 ``None`` (默认) 时预条件子绑主算子本身. 两根轴可以取不同的值 ——
主算子走 matrix-free 省内存, 预条件子仍可落在 ``"fa"`` 上供需要显式矩阵的后端使用.

EA 算子门面见 ``soptx.fem.matrix_free.operator``.
"""

from __future__ import annotations

from typing import Any

from soptx.functionspace import TensorFunctionSpace

from soptx.fem.analyzers.lagrange_fem_analyzer import LagrangeFEMAnalyzer


def _analyzer_arguments(
    space: TensorFunctionSpace,
    pde: Any,
    material: Any,
    degree: int = 1,
    operator_level: str = "ea",
    assembly_method: str = "standard",
    preconditioner_level: str | None = None,
    integration_order: int | None = None,
) -> dict[str, Any]:
    """将物理问题、材料本构与有限元空间组合为标准分析器构造参数字典.

    ``integration_order`` 为 ``None`` 时取 ``degree + 3``.
    """
    return {
        "disp_mesh": space.mesh,
        "pde": pde,
        "material": material,
        "space_degree": degree,
        "integration_order": degree + 3 if integration_order is None else integration_order,
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
    *,
    integration_order: int | None = None,
) -> LagrangeFEMAnalyzer:
    """构造串行拉格朗日有限元分析器.

    供 EA (无矩阵单元装配) 或 FA (全矩阵装配) 求解与基线对比使用.

    Args:
        space (TensorFunctionSpace): 全局向量有限元空间.
        pde (Any): 物理模型或制造解对象 (提供几何区域、体力荷载与边界条件).
        material (Any): 弹性力学本构材料模型 (提供材料常数与计算设备 device).
        degree (int, optional): 有限元插值多项式阶数. 默认值为 1.
        operator_level (str, optional): 算子装配级别, 可选 "ea" (无矩阵单元装配) 或 "fa" (全局稀疏矩阵全装配). 默认值为 "ea".
        assembly_method (str, optional): 单元矩阵的收缩顺序, 可选 ``"standard"``、``"voigt"`` 或 ``"fast"``. 它只改变
            中间张量的规模与峰值内存, 不改变单元矩阵的数值. 默认值为 ``"standard"``.
        preconditioner_level (str | None, optional): 预条件子取算子的层级, 取值同 ``operator_level``;
            为 ``None`` 时预条件子绑主算子本身. 默认值为 ``None``.
        integration_order (int | None, optional): 积分参数 q (关键字参数); 张量积单元为每方向点数.
            为 ``None`` 时取 ``degree + 3``. 默认值为 ``None``.

    Returns:
        LagrangeFEMAnalyzer: 初始化的串行有限元分析器实例.
    """
    return LagrangeFEMAnalyzer(
        solve_method="scipy",
        **_analyzer_arguments(
            space, pde, material, degree, operator_level, assembly_method,
            preconditioner_level, integration_order,
        ),
    )


__all__ = [
    "build_serial_analyzer",
]

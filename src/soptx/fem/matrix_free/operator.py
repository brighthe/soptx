"""线弹性 Matrix-Free EA 刚度算子的懒装配 (Lazy-Assembly) 门面模块.

``ElasticityEAOperator`` 封装 EA (无矩阵单元装配) 模式下的装配生命周期与算子代数接口:
首次访问时才触发装配并缓存、Dirichlet 自由度掩码推导, 以及可直接传入 Krylov
求解器的 ``@`` 矩阵乘接口. 串行与分布式由 ``dof_comm`` 是否为 ``None`` 决定,
底层分析器统一由 ``soptx.fem.analyzers.builders`` 构造.

分析器构造工厂见 ``soptx.fem.analyzers.builders``, 一步正向求解见 ``solve``.
"""

from __future__ import annotations

from typing import Any, Optional

from fealpy.functionspace import TensorFunctionSpace
from fealpy.typing import TensorLike

from soptx.fem.analyzers.builders import (
    build_distributed_analyzer,
    build_serial_analyzer,
)
from soptx.fem.analyzers.lagrange_fem_analyzer import LagrangeFEMAnalyzer


class ElasticityEAOperator:
    """线弹性 Matrix-Free EA 刚度算子与边界条件的懒装配 (Lazy-Assembly) 缓存门面.

    本类封装了线弹性分析器的装配生命周期与算子代数接口, 承担三大职责:
        1. 懒加载与结果缓存: 首次访问刚度算子或载荷向量时才触发装配, 之后自动复用, 避免重复装配开销;
        2. Dirichlet 自由度掩码推导: 自动提取并缓存边界自由度索引掩码 ``boundary_dofs``;
        3. 算子矩阵乘接口 (``@``): 实现 ``__matmul__``, 使其实例可作为线性算子直接传入 Krylov 求解器.

    串行与分布式统一:
        - 当 ``dof_comm=None`` 时, 底层自动调用 ``LagrangeFEMAnalyzer`` 串行分析器;
        - 当提供 ``dof_comm`` 时, 底层自动调用 ``DistributedElasticityAnalyzer`` 分布式分析器.
    """

    def __init__(
        self,
        space: TensorFunctionSpace,
        pde: Any,
        material: Any,
        degree: int = 1,
        dof_comm: Any = None,
        assembly_method: str = "standard",
    ) -> None:
        """初始化 EA 算子门面.

        参数:
            space (TensorFunctionSpace): 局部或全局交错布局的向量有限元空间.
            pde (Any): 物理工程问题或制造解对象.
            material (Any): 弹性力学本构材料模型.
            degree (int, 可选): 有限元基函数多项式次数. 默认值为 1.
            dof_comm (EntityMPI | None, 可选): 自由度跨进程通信器. 串行传入 None. 默认值为 None.
            assembly_method (str, 可选): 单元矩阵的收缩顺序, 可选 ``"standard"``、``"voigt"`` 或 ``"fast"``.
                它只改变中间张量的规模与峰值内存, 不改变单元矩阵的数值. 默认值为 ``"standard"``.
        """
        self.space = space
        self.pde = pde
        self.degree = degree
        self.dof_comm = dof_comm
        self.assembly_method = assembly_method

        self.analyzer: LagrangeFEMAnalyzer
        if dof_comm is None:
            self.analyzer = build_serial_analyzer(
                space, pde, material, degree, "ea", assembly_method
            )
        else:
            self.analyzer = build_distributed_analyzer(
                space, pde, material, degree, "ea", assembly_method,
                dof_comm=dof_comm,
            )

        self._system_operator: Any = None
        self._load_vector: Optional[TensorLike] = None
        self._prescribed: Optional[TensorLike] = None
        self._boundary_dofs: Optional[TensorLike] = None

    def assemble(self) -> tuple[Any, TensorLike]:
        """执行 EA 单元刚度矩阵与体力向量装配, 并施加 Dirichlet 边界条件对角投影.

        计算流程说明:
            1. 刚度算子装配 (EA 多态分发): 调用 ``assemble_stiff_matrix()`` 预计算并缓存各单元的小稠密
               刚度矩阵集 :math:`\\{\\mathbf{K}_e\\}`, 不拼接全局大矩阵;
            2. 体力荷载装配 (全通用实现): 调用 ``assemble_body_force_vector()`` 组装一维稠密载荷向量 :math:`\\mathbf{f}` (与 FA 共享);
            3. 边界条件施加 (EA 投影多态): 调用 ``apply_bc()`` 构建无矩阵对角投影算子 :math:`\\mathbf{A} = \\boldsymbol{\\Pi}_I \\mathbf{K} \\boldsymbol{\\Pi}_I + \\boldsymbol{\\Pi}_D`
               (多进程下自动包含 ``OverlapOperator`` 重叠同步包装);
            4. 边界几何信息提取 (全通用实现): 提取指定位移值 ``prescribed_solution`` 与边界自由度掩码 ``boundary_dofs``.

        返回:
            tuple[Any, TensorLike]: 包含以下两项的二元组:
                - operator (ConstrainedOperator): 施加边界条件后的系统刚度算子.
                - load (TensorLike): 施加边界条件修正后的等效右端载荷向量.
        """
        # 1. 刚度算子: 走 EA 多态路径, 预计算单元矩阵集并封装为无矩阵 BilinearForm
        stiff_matrix = self.analyzer.assemble_stiff_matrix()
        # 2. 体力荷载: 走通用线性型积分, 组装稠密右端项载荷向量
        body_force = self.analyzer.assemble_body_force_vector()
        # 3. 边界条件: 走 EA 对角投影算子包装 (分布式下自动提升为 OverlapOperator)
        operator, load = self.analyzer.apply_bc(stiff_matrix, body_force)

        self._system_operator = operator
        self._load_vector = load
        self._prescribed = self.analyzer.prescribed_solution
        self._boundary_dofs = self.space.is_boundary_dof(
            threshold=self.pde.is_dirichlet_boundary(),
            method="interp",
        )
        return operator, load

    def _ensure_assembled(self) -> None:
        """内部辅助方法: 确保刚度算子与载荷向量已装配."""
        if self._system_operator is None:
            self.assemble()

    @property
    def system_operator(self) -> Any:
        """施加 Dirichlet 边界条件后的系统刚度算子 (懒装配属性)."""
        self._ensure_assembled()
        return self._system_operator

    @property
    def load_vector(self) -> TensorLike:
        """施加 Dirichlet 边界条件后的等效右端载荷向量 (懒装配属性)."""
        self._ensure_assembled()
        assert self._load_vector is not None
        return self._load_vector

    @property
    def prescribed_solution(self) -> TensorLike:
        """Dirichlet 边界上的指定位移真解向量 (懒装配属性)."""
        self._ensure_assembled()
        assert self._prescribed is not None
        return self._prescribed

    @property
    def boundary_dofs(self) -> TensorLike:
        """Dirichlet 边界自由度的布尔掩码向量 (懒装配属性)."""
        self._ensure_assembled()
        assert self._boundary_dofs is not None
        return self._boundary_dofs

    def __matmul__(self, vector: TensorLike) -> TensorLike:
        """执行系统刚度算子与向量的矩阵-向量乘积运算 (A @ v)."""
        return self.system_operator @ vector


__all__ = [
    "ElasticityEAOperator",
]

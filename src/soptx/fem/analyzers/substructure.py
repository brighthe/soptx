"""完整接口子结构静力分析器.

本模块只负责编排一次给定密度下的有限元分析阶段: 局部刚度装配, LocalReduction,
完整接口显式装配与求解, 全场位移恢复及单元能量计算. 过滤, OC 更新, 收敛判据和
结果输出属于 topology/experiment 层, 不在此实现.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from fealpy.backend import backend_manager as bm

from soptx.fem.substructure.assembler import GlobalAssembler, InterfaceSystem
from soptx.fem.substructure.mesh import (
    SubstructureMesh,
    SubstructurePrototype,
    build_substructures,
)
from soptx.fem.substructure.problem_adapter import (
    InterfaceConditions,
    project_problem_conditions_to_interface_system,
)
from soptx.fem.substructure.reductions import (
    ExactSchurReduction,
    LocalReduction,
    LocalReductionBatchResult,
)
from soptx.fem.substructure.solve import solve_interface_system
from soptx.fem.substructure.traces import FullTraceBasis


@dataclass(frozen=True)
class FullInterfaceAnalysisResult:
    """一次完整接口子结构分析的不可变结果快照."""

    density: Any
    displacement: Any
    interface_displacement: Any
    force: Any
    interface_force: Any
    fixed_dofs: Any
    interface_fixed_dofs: Any
    cell_energy: Any
    compliance: float
    interface_system: InterfaceSystem
    local_reduction: LocalReductionBatchResult

    def as_state(self) -> Dict[str, Any]:
        """转换为 topology analysis stage 常用的状态字典."""
        return {
            "density": self.density,
            "displacement": self.displacement,
            "interface_displacement": self.interface_displacement,
            "force": self.force,
            "interface_force": self.interface_force,
            "fixed_dofs": self.fixed_dofs,
            "interface_fixed_dofs": self.interface_fixed_dofs,
            "cell_energy": self.cell_energy,
            "compliance": self.compliance,
            "interface_system": self.interface_system,
            "local_reduction": self.local_reduction,
        }


class FullInterfaceSubstructureAnalyzer:
    """保留全部接口自由度的子结构静力分析阶段.

    局部缩聚策略通过 ``LocalReduction`` 注入; 缺省为 ``ExactSchurReduction``.
    ``FullTraceBasis`` 在此明确表示接口迹不降阶, 全局系统由
    ``GlobalAssembler.assemble_interface_system`` 对完整 Schur 补散加得到.
    """

    def __init__(
        self,
        assembler: GlobalAssembler,
        problem: Any,
        *,
        prototype: Optional[SubstructurePrototype] = None,
        sub_meshes: Optional[Sequence[SubstructureMesh]] = None,
        local_reduction: Optional[LocalReduction] = None,
        solver: str = "scipy",
        assembly_chunk_size: Optional[int] = None,
    ) -> None:
        """初始化完整接口子结构分析器.

        参数:
            assembler: 规则同构子结构的全局装配器.
            problem: 通过 ``loads()`` 提供物理载荷并提供 Dirichlet 边界契约的
                问题对象.
            prototype: 共享参考子结构. 与 ``sub_meshes`` 同时省略时自动构造.
            sub_meshes: 全部子结构. 与 ``prototype`` 同时省略时自动构造.
            local_reduction: 完整接口局部缩聚策略; 缺省为 Exact Schur.
            solver: 接口稀疏系统求解器.
            assembly_chunk_size: 完整接口显式散加的子结构批大小.
        """
        if (prototype is None) != (sub_meshes is None):
            raise ValueError(
                "prototype 与 sub_meshes 必须同时提供或同时省略."
            )
        if prototype is None:
            prototype, built_meshes, _ = build_substructures(assembler)
            sub_meshes = built_meshes

        assert prototype is not None
        assert sub_meshes is not None
        meshes = tuple(sub_meshes)
        if not meshes:
            raise ValueError("sub_meshes 不能为空.")
        expected_count = 1
        for count in assembler.n_sub:
            expected_count *= int(count)
        if len(meshes) != expected_count:
            raise ValueError(
                "sub_meshes 数量必须等于 assembler.n_sub 的乘积."
            )
        if tuple(prototype.n_fine) != tuple(assembler.n_fine):
            raise ValueError(
                "prototype.n_fine 必须与 assembler.n_fine 一致."
            )
        if int(problem.dimension) != int(assembler.dim):
            raise ValueError(
                "problem.dimension 必须与 assembler.dim 一致."
            )
        if assembly_chunk_size is not None and assembly_chunk_size <= 0:
            raise ValueError("assembly_chunk_size 必须为正整数或 None.")

        self._assembler = assembler
        self._problem = problem
        self._prototype = prototype
        self._sub_meshes = meshes
        self._trace_basis = FullTraceBasis.from_prototype(prototype)
        self._local_reduction = local_reduction or ExactSchurReduction(
            prototype.i_dofs,
            prototype.b_dofs,
        )
        self._solver = solver
        self._assembly_chunk_size = assembly_chunk_size
        self._last_result: Optional[FullInterfaceAnalysisResult] = None

    @property
    def assembler(self) -> GlobalAssembler:
        """返回全局子结构装配器."""
        return self._assembler

    @property
    def pde(self) -> Any:
        """返回边界条件和载荷问题对象."""
        return self._problem

    @property
    def disp_mesh(self) -> Any:
        """返回与完整恢复位移一致的全尺度网格."""
        return self._assembler.full_mesh

    @property
    def material(self) -> Any:
        """返回装配器使用的全尺度线弹性材料."""
        return self._assembler.material

    @property
    def prototype(self) -> SubstructurePrototype:
        """返回共享参考子结构."""
        return self._prototype

    @property
    def sub_meshes(self) -> Sequence[SubstructureMesh]:
        """返回不可变的子结构序列."""
        return self._sub_meshes

    @property
    def trace_basis(self) -> FullTraceBasis:
        """返回恒等 ``full_trace`` 接口迹基."""
        return self._trace_basis

    @property
    def local_reduction(self) -> LocalReduction:
        """返回当前局部缩聚策略."""
        return self._local_reduction

    @property
    def last_result(self) -> Optional[FullInterfaceAnalysisResult]:
        """返回最近一次成功分析的结果快照."""
        return self._last_result

    def _normalize_density(self, density: Any) -> Any:
        """校验并展平全局单元密度场."""
        value = bm.asarray(density, dtype=bm.float64)
        actual_count = 1
        for count in value.shape:
            actual_count *= int(count)
        expected_count = 1
        for count in self._assembler.total_fine:
            expected_count *= int(count)
        if actual_count != expected_count:
            raise ValueError(
                f"density 必须包含 {expected_count} 个全局单元值; "
                f"当前形状为 {tuple(value.shape)}."
            )
        return bm.reshape(value, (-1,))

    def _cell_energy(self, displacement: Any) -> Any:
        """由完整恢复位移计算并合并全局单位刚度二次型."""
        positions = self._assembler.substructure_positions(self._sub_meshes)
        sub_global_dofs = bm.stack(
            [
                self._assembler.get_substructure_global_dofs(position, mesh)
                for position, mesh in zip(positions, self._sub_meshes)
            ],
            axis=0,
        )
        u_local = displacement[sub_global_dofs]
        u_element = u_local[:, self._prototype.cell2dof]
        unit_stiffness = self._prototype.KE_unit[0]
        energy_sub_cell = bm.sum(
            (u_element @ unit_stiffness) * u_element,
            axis=-1,
        )
        energy_sub_grid = self._prototype.cell_to_grid_field(energy_sub_cell)
        energy_global_grid = self._assembler.merge_substructure_cell_field(
            energy_sub_grid
        )
        return bm.reshape(energy_global_grid, (-1,))

    def analyze(self, density: Any) -> FullInterfaceAnalysisResult:
        """完成一次完整接口子结构静力分析."""
        rho = self._normalize_density(density)
        rho_sub_grid = self._assembler.split_global_cell_field(rho)
        rho_sub_cell = self._prototype.grid_to_cell_field(rho_sub_grid)
        local_stiffness = self._prototype.assemble_local_stiffness_batch(
            rho_sub_cell
        )
        reduction = self._local_reduction.reduce_many(
            local_stiffness,
            rho_sub_grid,
        )
        if reduction.recovery is None:
            raise RuntimeError(
                "完整接口位移恢复要求 LocalReduction 返回 recovery."
            )
        if int(reduction.stiffness.shape[-1]) != self._trace_basis.n_boundary_dofs:
            raise ValueError("LocalReduction 刚度的接口自由度数与 full_trace 不一致.")
        if int(reduction.recovery.shape[-1]) != self._trace_basis.n_boundary_dofs:
            raise ValueError("LocalReduction 恢复矩阵的接口自由度数与 full_trace 不一致.")

        system = self._assembler.assemble_interface_system(
            list(self._sub_meshes),
            reduction,
            chunk_size=self._assembly_chunk_size,
        )
        conditions: InterfaceConditions = (
            project_problem_conditions_to_interface_system(
                self._problem,
                self._assembler,
                system,
            )
        )
        interface_displacement = solve_interface_system(
            system,
            conditions.interface_force,
            conditions.interface_fixed_dofs,
            solver=self._solver,
        )
        displacement = self._assembler.recover_full_displacement(
            list(self._sub_meshes),
            reduction,
            system,
            interface_displacement,
        )
        cell_energy = self._cell_energy(displacement)
        compliance = float(bm.dot(conditions.full_force, displacement))

        result = FullInterfaceAnalysisResult(
            density=rho,
            displacement=displacement,
            interface_displacement=interface_displacement,
            force=conditions.full_force,
            interface_force=conditions.interface_force,
            fixed_dofs=conditions.full_fixed_dofs,
            interface_fixed_dofs=conditions.interface_fixed_dofs,
            cell_energy=cell_energy,
            compliance=compliance,
            interface_system=system,
            local_reduction=reduction,
        )
        self._last_result = result
        return result

    def solve_state(self, rho_val: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """以 topology analysis-stage 字典形式返回一次分析结果."""
        if rho_val is None:
            raise ValueError("完整接口变密度分析必须提供 rho_val.")
        return self.analyze(rho_val).as_state()

    def compute_compliance_sensitivity(
        self,
        state: FullInterfaceAnalysisResult | Dict[str, Any],
    ) -> Any:
        """计算 modified SIMP 柔度对物理密度的解析导数."""
        if isinstance(state, FullInterfaceAnalysisResult):
            density = state.density
            cell_energy = state.cell_energy
        else:
            density = state["density"]
            cell_energy = state["cell_energy"]
        penalty = float(self._prototype.penal)
        rho_min = float(self._prototype.rho_min)
        derivative = penalty * (1.0 - rho_min) * density ** (penalty - 1.0)
        return -derivative * cell_energy


__all__ = [
    "FullInterfaceAnalysisResult",
    "FullInterfaceSubstructureAnalyzer",
]

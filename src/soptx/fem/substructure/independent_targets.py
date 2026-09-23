"""子结构独立条目标签与刚体约束补全.

本模块实现 ``linear_corner`` 迹空间下的两类监督标签。它直接保留矩阵条目,
再由刚体约束消去一组 pivot 自由度; 因此既不是刚体正交补投影, 也不是
Cholesky 参数化.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch
from fealpy.backend import backend_manager as bm

from .mesh import SubstructurePrototype
from .traces import FullTraceBasis, LinearCornerTraceBasis


def _pivot_rows(matrix: np.ndarray) -> np.ndarray:
    """用确定性的行主元 Gram-Schmidt 选出满秩方阵."""
    work = np.asarray(matrix, dtype=np.float64).copy()
    n_row, n_col = work.shape
    selected: list[int] = []
    available = np.ones(n_row, dtype=bool)
    for _ in range(n_col):
        norms = np.linalg.norm(work, axis=1)
        norms[~available] = -np.inf
        pivot = int(np.argmax(norms))
        scale = float(norms[pivot])
        if not np.isfinite(scale) or scale <= 100.0 * np.finfo(float).eps:
            raise ValueError("刚体约束矩阵无法选出满秩 pivot 自由度.")
        direction = work[pivot] / scale
        work -= np.outer(work @ direction, direction)
        available[pivot] = False
        selected.append(pivot)
    return np.asarray(selected, dtype=np.int64)


def _selectors(
    n_trace: int,
    pivot_indices: np.ndarray,
    free_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """构造把 pivot/free 列散回完整迹空间的选择矩阵."""
    free = np.zeros((len(free_indices), n_trace), dtype=np.float64)
    pivot = np.zeros((len(pivot_indices), n_trace), dtype=np.float64)
    free[np.arange(len(free_indices)), free_indices] = 1.0
    pivot[np.arange(len(pivot_indices)), pivot_indices] = 1.0
    return free, pivot


def _torch_constant(value: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    """把 codec 常量移到输入张量的 dtype 与 device."""
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


class ShapeIndependentCodec:
    """提取并补全内部形函数的独立矩阵条目.

    对满足 ``B Q = Phi`` 的内部延拓 ``B``, 网络只预测非 pivot 列
    ``B_F``. pivot 列由
    ``B_P Q_P = Phi - B_F Q_F`` 唯一恢复.
    """

    def __init__(
        self,
        rigid_basis: Any,
        rigid_interior: Any,
        pivot_indices: Optional[Sequence[int]] = None,
    ) -> None:
        q = np.asarray(rigid_basis, dtype=np.float64)
        phi = np.asarray(rigid_interior, dtype=np.float64)
        if q.ndim != 2 or phi.ndim != 2 or phi.shape[1] != q.shape[1]:
            raise ValueError("rigid_basis 与 rigid_interior 的刚体模态数必须一致.")

        pivots = (
            _pivot_rows(q)
            if pivot_indices is None
            else np.asarray(pivot_indices, dtype=np.int64)
        )
        if pivots.shape != (q.shape[1],) or len(np.unique(pivots)) != len(pivots):
            raise ValueError("pivot_indices 必须包含恰好 n_rigid 个互异索引.")
        if np.any(pivots < 0) or np.any(pivots >= q.shape[0]):
            raise ValueError("pivot_indices 超出迹自由度范围.")

        mask = np.ones(q.shape[0], dtype=bool)
        mask[pivots] = False
        free = np.flatnonzero(mask).astype(np.int64)
        q_pivot = q[pivots]
        if np.linalg.matrix_rank(q_pivot) != q.shape[1]:
            raise ValueError("pivot_indices 对应的刚体约束子矩阵不满秩.")

        self.rigid_basis = q
        self.rigid_interior = phi
        self.pivot_indices = pivots
        self.free_indices = free
        self._q_free = q[free]
        self._q_pivot_inverse = np.linalg.inv(q_pivot)
        self._free_selector, self._pivot_selector = _selectors(
            q.shape[0], pivots, free
        )
        self.n_internal = int(phi.shape[0])
        self.n_trace = int(q.shape[0])
        self.n_rigid = int(q.shape[1])
        self.n_output = self.n_internal * len(free)

    def encode(self, recovery: Any) -> Any:
        """提取非 pivot 列并展平为 ``(..., n_output)``."""
        if tuple(recovery.shape[-2:]) != (self.n_internal, self.n_trace):
            raise ValueError(
                "recovery 的末两维应为 "
                f"({self.n_internal}, {self.n_trace}); 当前为 {tuple(recovery.shape)}."
            )
        if isinstance(recovery, torch.Tensor):
            index = torch.as_tensor(
                self.free_indices, dtype=torch.long, device=recovery.device
            )
            independent = recovery.index_select(-1, index)
        else:
            independent = np.take(np.asarray(recovery), self.free_indices, axis=-1)
        return independent.reshape(recovery.shape[:-2] + (self.n_output,))

    def decode(self, values: Any) -> Any:
        """由独立条目可微地补全 ``B Q = Phi``."""
        if values.shape[-1] != self.n_output:
            raise ValueError(
                f"独立形函数条目数应为 {self.n_output}; 当前为 {values.shape[-1]}."
            )
        free_shape = values.shape[:-1] + (
            self.n_internal,
            len(self.free_indices),
        )
        free_values = values.reshape(free_shape)
        if isinstance(values, torch.Tensor):
            q_free = _torch_constant(self._q_free, values)
            q_pivot_inverse = _torch_constant(self._q_pivot_inverse, values)
            phi = _torch_constant(self.rigid_interior, values)
            free_selector = _torch_constant(self._free_selector, values)
            pivot_selector = _torch_constant(self._pivot_selector, values)
        else:
            q_free = self._q_free
            q_pivot_inverse = self._q_pivot_inverse
            phi = self.rigid_interior
            free_selector = self._free_selector
            pivot_selector = self._pivot_selector
        pivot_values = (phi - free_values @ q_free) @ q_pivot_inverse
        return free_values @ free_selector + pivot_values @ pivot_selector


class StiffnessIndependentCodec:
    """提取并补全对称缩聚刚度的独立矩阵条目.

    网络预测非 pivot 自由度子块 ``K_FF`` 的下三角条目. 补全使用刚体
    零空间 ``K Q = 0``, 但不对 ``K_FF`` 作 Cholesky 分解, 因而不会
    额外保证半正定性.
    """

    def __init__(
        self,
        rigid_basis: Any,
        pivot_indices: Optional[Sequence[int]] = None,
    ) -> None:
        q = np.asarray(rigid_basis, dtype=np.float64)
        if q.ndim != 2:
            raise ValueError("rigid_basis 必须是二维矩阵.")
        pivots = (
            _pivot_rows(q)
            if pivot_indices is None
            else np.asarray(pivot_indices, dtype=np.int64)
        )
        if pivots.shape != (q.shape[1],) or len(np.unique(pivots)) != len(pivots):
            raise ValueError("pivot_indices 必须包含恰好 n_rigid 个互异索引.")
        if np.any(pivots < 0) or np.any(pivots >= q.shape[0]):
            raise ValueError("pivot_indices 超出迹自由度范围.")

        mask = np.ones(q.shape[0], dtype=bool)
        mask[pivots] = False
        free = np.flatnonzero(mask).astype(np.int64)
        q_pivot = q[pivots]
        if np.linalg.matrix_rank(q_pivot) != q.shape[1]:
            raise ValueError("pivot_indices 对应的刚体约束子矩阵不满秩.")

        n_free = len(free)
        tri_i, tri_j = np.tril_indices(n_free)
        # 以二维索引表重构对称块, 避免存储四阶规模的稠密基.
        symmetric_indices = np.empty((n_free, n_free), dtype=np.int64)
        symmetric_indices[tri_i, tri_j] = np.arange(len(tri_i))
        symmetric_indices[tri_j, tri_i] = np.arange(len(tri_i))

        selector = np.zeros((n_free, q.shape[0]), dtype=np.float64)
        selector[np.arange(n_free), free] = 1.0
        selector[:, pivots] = -(q[free] @ np.linalg.inv(q_pivot))

        self.rigid_basis = q
        self.pivot_indices = pivots
        self.free_indices = free
        self._tri_i = tri_i
        self._tri_j = tri_j
        self._symmetric_indices = symmetric_indices
        self._constraint_selector = selector
        self.n_trace = int(q.shape[0])
        self.n_rigid = int(q.shape[1])
        self.n_free = n_free
        self.n_output = n_free * (n_free + 1) // 2

    def encode(self, stiffness: Any) -> Any:
        """提取 ``K_FF`` 的下三角条目."""
        if tuple(stiffness.shape[-2:]) != (self.n_trace, self.n_trace):
            raise ValueError(
                "stiffness 的末两维应为 "
                f"({self.n_trace}, {self.n_trace}); 当前为 {tuple(stiffness.shape)}."
            )
        if isinstance(stiffness, torch.Tensor):
            free = torch.as_tensor(
                self.free_indices, dtype=torch.long, device=stiffness.device
            )
            tri_i = torch.as_tensor(
                self._tri_i, dtype=torch.long, device=stiffness.device
            )
            tri_j = torch.as_tensor(
                self._tri_j, dtype=torch.long, device=stiffness.device
            )
            reduced = stiffness.index_select(-2, free).index_select(-1, free)
            return reduced[..., tri_i, tri_j]
        reduced = np.take(
            np.take(np.asarray(stiffness), self.free_indices, axis=-2),
            self.free_indices,
            axis=-1,
        )
        return reduced[..., self._tri_i, self._tri_j]

    def decode(self, values: Any) -> Any:
        """由独立条目可微地重构对称且满足 ``K Q = 0`` 的刚度."""
        if values.shape[-1] != self.n_output:
            raise ValueError(
                f"独立刚度条目数应为 {self.n_output}; 当前为 {values.shape[-1]}."
            )
        if isinstance(values, torch.Tensor):
            indices = torch.as_tensor(
                self._symmetric_indices, dtype=torch.long, device=values.device
            )
            selector = _torch_constant(self._constraint_selector, values)
            reduced = values[..., indices]
            return selector.transpose(-1, -2) @ reduced @ selector
        reduced = np.asarray(values)[..., self._symmetric_indices]
        selector = self._constraint_selector
        return np.swapaxes(selector, -1, -2) @ reduced @ selector


class IndependentTargetProvider:
    """生成二维或三维子结构在所选接口空间下的精确独立条目标签.

    输入是归一化杨氏模量 ``E / E_base``, 而非再经过 SIMP 插值的设计密度.
    默认原型令 ``penal=1``、``rho_min=0``, 因此局部单元刚度对输入线性缩放.
    二维原型沿用 ``SubstructurePrototype`` 的平面应力假设, 三维原型使用三维
    各向同性线弹性假设.
    """

    def __init__(
        self,
        *,
        cell_size: Sequence[float] = (1.0, 1.0, 1.0),
        n_fine: Sequence[int] = (5, 5, 5),
        nu: float = 0.3,
        chunk_size: Optional[int] = None,
        trace_kind: str = "linear_corner",
    ) -> None:
        """初始化子结构及独立条目补全器.

        Parameters
        ----------
        cell_size : sequence of float
            子结构各方向尺寸.
        n_fine : sequence of int
            各方向细单元数量.
        nu : float
            固定泊松比.
        chunk_size : int or None
            局部刚度装配批量大小.
        trace_kind : str
            linear_corner 或 full_trace, 默认保留角点接口.
        """
        trace_types = {
            "linear_corner": LinearCornerTraceBasis,
            "full_trace": FullTraceBasis,
        }
        if trace_kind not in trace_types:
            raise ValueError("trace_kind 必须为 linear_corner 或 full_trace")
        self.trace_kind = trace_kind
        self.prototype = SubstructurePrototype(
            cell_size=cell_size,
            n_fine=n_fine,
            E_base=1.0,
            nu=nu,
            penal=1.0,
            rho_min=0.0,
        )
        self.trace = trace_types[trace_kind].from_prototype(self.prototype)
        rigid, _, rigid_interior = self.prototype.trace_interface_bases(
            self.trace
        )
        rigid_numpy = np.asarray(bm.to_numpy(rigid), dtype=np.float64)
        interior_numpy = np.asarray(
            bm.to_numpy(rigid_interior), dtype=np.float64
        )
        pivots = _pivot_rows(rigid_numpy)
        self.shape_codec = ShapeIndependentCodec(
            rigid_numpy, interior_numpy, pivot_indices=pivots
        )
        self.stiffness_codec = StiffnessIndependentCodec(
            rigid_numpy, pivot_indices=pivots
        )
        self.codecs = {
            "shape": self.shape_codec,
            "stiffness": self.stiffness_codec,
        }
        self._trace_matrix = np.asarray(
            bm.to_numpy(self.trace.matrix), dtype=np.float64
        )
        self._i_dofs = np.asarray(
            bm.to_numpy(self.prototype.i_dofs), dtype=np.int64
        )
        self._b_dofs = np.asarray(
            bm.to_numpy(self.prototype.b_dofs), dtype=np.int64
        )
        self.chunk_size = chunk_size
        self.nu = float(nu)

    def __call__(self, normalized_modulus: Any) -> dict[str, np.ndarray]:
        """返回形函数与刚度的独立条目标签."""
        modulus = np.asarray(normalized_modulus, dtype=np.float64)
        expected = self.prototype.n_cells
        if modulus.ndim != 2 or modulus.shape[1] != expected:
            raise ValueError(
                f"normalized_modulus 应具有形状 (batch, {expected}); "
                f"当前为 {tuple(modulus.shape)}."
            )
        if not np.all(np.isfinite(modulus)) or np.any(modulus <= 0.0):
            raise ValueError("normalized_modulus 必须全部有限且严格为正.")
        if np.any(modulus > 1.0):
            raise ValueError("normalized_modulus 不得大于 1.")

        local_backend = self.prototype.assemble_local_stiffness_batch(
            bm.asarray(modulus, dtype=bm.float64),
            chunk_size=self.chunk_size,
        )
        local = np.asarray(bm.to_numpy(local_backend), dtype=np.float64)
        i_dofs = self._i_dofs
        b_dofs = self._b_dofs
        trace = self._trace_matrix

        K_ii = local[..., i_dofs[:, None], i_dofs]
        K_ib = local[..., i_dofs[:, None], b_dofs]
        K_bb = local[..., b_dofs[:, None], b_dofs]
        K_ib_trace = K_ib @ trace
        K_bb_trace = np.swapaxes(trace, -1, -2) @ K_bb @ trace

        recovery = -np.linalg.solve(K_ii, K_ib_trace)
        stiffness = (
            K_bb_trace
            + np.swapaxes(K_ib_trace, -1, -2) @ recovery
        )
        targets = {
            "shape": np.asarray(self.shape_codec.encode(recovery)),
            "stiffness": np.asarray(self.stiffness_codec.encode(stiffness)),
        }
        for name, exact in (("shape", recovery), ("stiffness", stiffness)):
            restored = self.codecs[name].decode(targets[name])
            error = np.linalg.norm(restored - exact, axis=(-2, -1))
            scale = np.linalg.norm(exact, axis=(-2, -1))
            if (
                not np.isfinite(error).all()
                or not np.isfinite(scale).all()
                or np.any(error > 1e-8 * scale + 1e-10)
            ):
                raise ValueError(f"{name} 独立条目补全与精确标签不一致")
        return targets

    def metadata(self) -> dict[str, Any]:
        """返回可直接写入 JSON 的标签构造契约."""
        geometry = (
            "axis_aligned_quadrilateral"
            if self.prototype.dim == 2
            else "axis_aligned_hexahedron"
        )

        return {
            "trace": self.trace_kind,
            "spatial_dimension": int(self.prototype.dim),
            "geometry": geometry,
            "cell_size": list(self.prototype.cell_size),
            "n_fine": list(self.prototype.n_fine),
            "finite_element_degree": int(self.prototype.degree),
            "dof_ordering": "node_major_component_minor",
            "n_cells": int(self.prototype.n_cells),
            "poisson_ratio": self.nu,
            **(
                {"material_hypothesis": "plane_stress"}
                if self.prototype.dim == 2
                else {}
            ),
            "input_quantity": "normalized_young_modulus",
            "input_range": "(0, 1]",
            "n_i": self.shape_codec.n_internal,
            "n_trace": self.shape_codec.n_trace,
            "n_rigid": self.shape_codec.n_rigid,
            "n_shape_targets": self.shape_codec.n_output,
            "n_stiffness_targets": self.stiffness_codec.n_output,
            "rigid_basis": self.shape_codec.rigid_basis.tolist(),
            "rigid_interior": self.shape_codec.rigid_interior.tolist(),
            "roundtrip_tolerance": {"rtol": 1e-8, "atol": 1e-10},
            "pivot_indices": self.shape_codec.pivot_indices.tolist(),
            "free_indices": self.shape_codec.free_indices.tolist(),
            "shape_encoding": (
                "free_matrix_entries_with_rigid_pivot_completion"
            ),
            "stiffness_encoding": (
                "symmetric_free_block_with_rigid_pivot_completion"
            ),
        }

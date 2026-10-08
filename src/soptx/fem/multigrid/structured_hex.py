"""结构化六面体网格上 Q1 位移空间的几何多重网格层次.

粗层遵循 Galerkin 原则 :math:`A_{l+1} = P_0^{\\mathsf T} A_l P_0`, 其中 :math:`P` 为三线性插值,
:math:`P_0` 把受约束 (或无耦合) 细自由度所在的行置零. 这样粗层只作用在自由自由度上,
支承不必落在粗节点上; 与施加了 Dirichlet 条件的细层矩阵
:math:`\\Pi_I K \\Pi_I + \\Pi_D` 相容 (其单位块经 :math:`P_0` 投影后为零).

粗网格各方向单元数取 :math:`\\lceil n / 2 \\rceil`, 奇数边上最后一层粗单元伸出计算域,
伸出部分的子单元系数取零; 效果同在域外补零刚度单元, 但不改动最细层的有限元问题.

第 1 层到第 2 层按单元组合: 嵌套网格上细单元 :math:`e` 的自由度只由其父粗单元
:math:`E` 的节点插值, 于是

.. math::

    K_E = \\sum_{c=1}^{8} s_{e(c)} M_c, \\qquad M_c = P_c^{\\mathsf T} K^0 P_c,

其中 :math:`P_c` 为第 :math:`c` 个子单元位置上的 :math:`24 \\times 24` 局部插值. 挨着支承的
细单元先把受约束的局部自由度置零再组合. 这一步不需要细层的全局矩阵, 也不产生
:math:`K P` 这样的大中间乘积. 第 2 层以下的矩阵已经很小, 直接做稀疏的 Galerkin 投影.

网格编号遵循 :mod:`soptx.mesh.structured_box` 的公开约定: 节点与单元都按 x 最慢、z 最快的
字典序排列, 单元局部顶点顺序取该模块的六面体模板.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from soptx.backend import backend_manager as bm
from soptx.backend import TensorLike
from soptx.mesh.structured_box import _TEMPLATES, _grid_cells
from soptx.solvers import DirectSolver, JacobiSmoother, Multigrid, operator_diagonal
from soptx.sparse import CSRTensor

from ..matrix.csr_pattern import assemble_csr, build_csr_pattern_from_dofmap

# 六面体模板的局部顶点偏移 (8, 3), 顺序即单元的局部顶点顺序
_HEX_OFFSETS = np.asarray(_TEMPLATES['hex'][2][0], dtype=np.int64)
# 子单元位置 (a, b, c), a 最慢; 第 c 个子单元为细单元 (2I + a, 2J + b, 2K + c)
_CHILD_OFFSETS = np.asarray(list(product((0, 1), repeat=3)), dtype=np.int64)
# 粗层对角不超过最大对角的这一倍时, 视为与自由自由度无耦合, 置单位对角
_DECOUPLED_RTOL = 1.0e-14


@dataclass(frozen=True)
class StructuredHexGrid:
    """按 ``create_box_mesh`` 编号约定的结构化六面体网格, 只记形状与范围.

    Parameters
    ----------
    shape : 各方向单元数 ``(nx, ny, nz)``.
    origin : 最小角点坐标.
    spacing : 各方向单元边长.
    """

    shape: Tuple[int, int, int]
    origin: Tuple[float, float, float]
    spacing: Tuple[float, float, float]

    @property
    def num_cells(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    @property
    def num_nodes(self) -> int:
        nx, ny, nz = self.shape
        return (nx + 1) * (ny + 1) * (nz + 1)

    @classmethod
    def from_mesh(cls, mesh: Any, *, rtol: float = 1.0e-10) -> "StructuredHexGrid":
        """由网格反推结构化形状, 并校验其与编号约定逐位一致.

        Parameters
        ----------
        mesh : 三维六面体网格.
        rtol : 节点坐标的相对容差, 以区域最大边长计.

        Returns
        -------
        grid : 对应的结构化网格描述.

        Raises
        ------
        ValueError
            不是三维六面体网格, 节点不构成等距张量积格点, 或节点、单元编号与约定不符.
        """
        node = np.asarray(bm.to_numpy(mesh.entity('node')), dtype=np.float64)
        cell = np.asarray(bm.to_numpy(mesh.entity('cell')))
        if node.ndim != 2 or node.shape[1] != 3 or cell.ndim != 2 or cell.shape[1] != 8:
            raise ValueError('StructuredHexGrid 只接受三维六面体网格')
        lower, upper = node.min(axis=0), node.max(axis=0)
        shape = tuple(int(np.unique(node[:, d]).size) - 1 for d in range(3))
        if (min(shape) < 1 or int(np.prod([n + 1 for n in shape])) != node.shape[0]
                or int(np.prod(shape)) != cell.shape[0]):
            raise ValueError(f'节点或单元数与张量积格点 {shape} 不符, 不是结构化网格')
        grid = cls(shape=shape, origin=tuple(float(v) for v in lower),
                   spacing=tuple(float(v) for v in (upper - lower) / np.asarray(shape)))
        if not np.allclose(node, grid.node_coordinates(), rtol=0.0,
                           atol=rtol * float(np.max(upper - lower))):
            raise ValueError('节点坐标与结构化编号约定不符 (编号顺序或步长不等距)')
        if not np.array_equal(cell, grid.cell_to_node()):
            raise ValueError('单元连接与 create_box_mesh 的编号约定不符')

        return grid

    def coarsen(self) -> "StructuredHexGrid":
        """各方向单元数取 ceil(n / 2), 边长加倍, 最小角点不变."""
        return StructuredHexGrid(shape=tuple(-(-n // 2) for n in self.shape), origin=self.origin,
                                 spacing=tuple(2.0 * h for h in self.spacing))

    def node_coordinates(self) -> np.ndarray:
        """(NN, 3) 的节点坐标, 按编号约定排列."""
        axes = [o + h * np.arange(n + 1) for o, h, n in zip(self.origin, self.spacing, self.shape)]
        grids = np.meshgrid(*axes, indexing='ij')
        return np.stack([g.ravel() for g in grids], axis=1)

    def cell_to_node(self) -> np.ndarray:
        """(NC, 8) 的单元顶点编号, 与 ``create_box_mesh('hex', ...)`` 一致."""
        return np.asarray(bm.to_numpy(_grid_cells(self.shape, _TEMPLATES['hex'][2], None)))


def _tensor_dof(node: np.ndarray, component: int, num_nodes: int, dof_priority: bool) -> np.ndarray:
    """标量节点与分量到张量自由度的编号, 与 ``TensorFunctionSpace`` 一致."""
    return component * num_nodes + node if dof_priority else 3 * node + component


def _prolongation(fine: StructuredHexGrid, coarse: StructuredHexGrid, dof_priority: bool,
                  keep_rows: Optional[np.ndarray] = None) -> CSRTensor:
    """由粗网格到细网格的三线性插值延拓, 向量值 (3 分量).

    Parameters
    ----------
    fine, coarse : 细网格与其 ``coarsen()`` 所得的粗网格.
    dof_priority : 两层共用的张量自由度排布.
    keep_rows : (3 NN_fine, ) 的布尔掩码; 给出时只保留其为 True 的细自由度所在行.

    Returns
    -------
    P : (3 NN_fine, 3 NN_coarse) 的 CSR 矩阵.
    """
    index, weight = [], []
    for n, nc in zip(fine.shape, coarse.shape):
        i = np.arange(n + 1)
        odd = (i % 2) == 1
        index.append(np.stack([i // 2, np.minimum(i // 2 + 1, nc)], axis=1))
        weight.append(np.stack([np.where(odd, 0.5, 1.0), np.where(odd, 0.5, 0.0)], axis=1))
    nyc, nzc = coarse.shape[1] + 1, coarse.shape[2] + 1
    # (nx+1, ny+1, nz+1, 2, 2, 2) 的粗节点编号与权重, 前三轴的展平次序即细节点编号
    coarse_node = ((index[0][:, None, None, :, None, None] * nyc
                    + index[1][None, :, None, None, :, None]) * nzc
                   + index[2][None, None, :, None, None, :])
    w = (weight[0][:, None, None, :, None, None] * weight[1][None, :, None, None, :, None]
         * weight[2][None, None, :, None, None, :])
    fine_node = np.broadcast_to(np.arange(fine.num_nodes).reshape(fine.shape[0] + 1, fine.shape[1] + 1,
                                                                 fine.shape[2] + 1, 1, 1, 1), w.shape)
    nonzero = w > 0.0
    fine_node, coarse_node, w = fine_node[nonzero], coarse_node[nonzero], w[nonzero]

    rows = np.concatenate([_tensor_dof(fine_node, c, fine.num_nodes, dof_priority) for c in range(3)])
    cols = np.concatenate([_tensor_dof(coarse_node, c, coarse.num_nodes, dof_priority) for c in range(3)])
    vals = np.concatenate([w, w, w])
    if keep_rows is not None:
        kept = keep_rows[rows]
        rows, cols, vals = rows[kept], cols[kept], vals[kept]
    matrix = sp.csr_matrix((vals, (rows, cols)), shape=(3 * fine.num_nodes, 3 * coarse.num_nodes))

    return CSRTensor.from_scipy(matrix)


def _child_interpolation(dof_priority: bool) -> np.ndarray:
    """(8, 24, 24) 的子单元局部插值 P_c: 父粗单元的局部自由度到第 c 个子单元的局部自由度."""
    scalar = np.empty((8, 8, 8))
    for c, child in enumerate(_CHILD_OFFSETS):
        position = (child + _HEX_OFFSETS) / 2.0                     # 子单元顶点在父单元中的位置
        scalar[c] = np.prod(1.0 - np.abs(position[:, None, :] - _HEX_OFFSETS[None, :, :]), axis=2)
    identity = np.eye(3)
    if dof_priority:
        return np.stack([np.kron(identity, s) for s in scalar])
    return np.stack([np.kron(s, identity) for s in scalar])


def _diagonal_slots(matrix: CSRTensor) -> np.ndarray:
    """(n, ) 的各行对角元在 CSR values 中的位置; 要求每行恰有一个对角槽位."""
    crow, col = np.asarray(matrix.crow), np.asarray(matrix.col)
    rows = np.repeat(np.arange(crow.size - 1), np.diff(crow))
    slots = np.flatnonzero(col == rows)
    if slots.size != crow.size - 1:
        raise RuntimeError('粗层 CSR 骨架缺少对角槽位')
    return slots


def _decoupled(diag: np.ndarray) -> np.ndarray:
    """对角不超过最大对角 _DECOUPLED_RTOL 倍的自由度, 即与自由自由度无耦合者."""
    return diag <= _DECOUPLED_RTOL * float(np.max(diag))


class StructuredHexHierarchy:
    """结构化六面体网格上 Q1 位移空间的几何多重网格层次.

    只依赖网格与约束的部分 (各层网格, 延拓, 8 个 M_c, 支承单元的修正, 第 2 层 CSR 骨架)
    在构造时建好, 跨优化轮复用; :meth:`update` 只按新单元系数重建粗层算子.

    Parameters
    ----------
    space : 最细层位移空间, 三维 Lagrange p = 1 张量空间, 网格须满足结构化编号约定.
    is_dirichlet : (gdof, ) 的最细层 Dirichlet 自由度掩码.
    reference_matrix : (24, 24) 的实体参考单元刚度 K^0; 等距结构化网格上各单元的
        K_e^0 只差舍入, 取任一单元的即可.
    coarse_max_dofs : 最粗层自由度数上限; 超过即继续粗化. 无论多小都至少粗化一层 (网格还能
        粗化时): 最粗层要直接分解, 而 'ea' 层级的最细层算子没有可分解的矩阵.

    Raises
    ------
    NotImplementedError
        后端不是 numpy (最粗层依赖 CPU 直接法).
    ValueError
        空间不是三维 p = 1 张量空间, 网格不是结构化六面体网格, 或参考矩阵形状不符.

    Notes
    -----
    细层的 Dirichlet 处理须为对称消元 :math:`\\Pi_I K \\Pi_I + \\Pi_D`: 分析器 'fa' 层级的保结构
    消元矩阵与 'ea' 层级的 ``ConstrainedOperator`` 都是如此, 二者可作同一层次的最细层算子.
    """

    def __init__(self, space: Any, is_dirichlet: TensorLike, reference_matrix: TensorLike, *,
                 coarse_max_dofs: int = 20000) -> None:
        if bm.backend_name != 'numpy':
            raise NotImplementedError('StructuredHexHierarchy 目前只支持 numpy 后端')
        if int(getattr(space, 'dof_numel', 0)) != 3 or int(getattr(space, 'p', 0)) != 1:
            raise ValueError('StructuredHexHierarchy 只支持三维 p = 1 的位移张量空间')
        K0 = np.asarray(bm.to_numpy(reference_matrix), dtype=np.float64)
        if K0.shape != (24, 24):
            raise ValueError(f'reference_matrix 须为 (24, 24), 得到 {K0.shape}')
        fixed = np.asarray(bm.to_numpy(is_dirichlet), dtype=bool)

        self.dof_priority = bool(space.dof_priority)
        fine = StructuredHexGrid.from_mesh(space.mesh)
        if fixed.shape != (3 * fine.num_nodes, ):
            raise ValueError(f'is_dirichlet 须为 ({3 * fine.num_nodes}, ), 得到 {fixed.shape}')
        self.grids: List[StructuredHexGrid] = [fine]
        while ((len(self.grids) == 1 or 3 * self.grids[-1].num_nodes > coarse_max_dofs)
               and max(self.grids[-1].shape) > 1):
            self.grids.append(self.grids[-1].coarsen())
        self._operators: Optional[List[CSRTensor]] = None
        self._prolongations: List[CSRTensor] = []
        if len(self.grids) == 1:
            return

        coarse = self.grids[1]
        self._first_prolongation = _prolongation(fine, coarse, self.dof_priority, keep_rows=~fixed)
        P_c = _child_interpolation(self.dof_priority)
        self._child_matrices = np.einsum('cki,kl,clj->cij', P_c, K0, P_c)

        # 挨着支承的细单元: 先置零受约束的局部自由度再组合, 记为对无约束组合的修正
        cell2node = fine.cell_to_node()
        local = np.stack([_tensor_dof(cell2node[:, v], c, fine.num_nodes, self.dof_priority)
                          for v, c in self._local_order()], axis=1)
        local_fixed = fixed[local]
        boundary = np.flatnonzero(local_fixed.any(axis=1))
        ijk = np.stack(np.unravel_index(boundary, fine.shape), axis=1)
        child = ((ijk[:, 0] % 2) * 2 + ijk[:, 1] % 2) * 2 + ijk[:, 2] % 2
        keep = (~local_fixed[boundary]).astype(np.float64)
        masked = K0[None, :, :] * keep[:, :, None] * keep[:, None, :]
        self._boundary_cells = boundary
        self._boundary_parents = np.ravel_multi_index(tuple((ijk // 2).T), coarse.shape)
        self._boundary_corrections = (np.einsum('bki,bkl,blj->bij', P_c[child], masked, P_c[child])
                                      - self._child_matrices[child])

        self._pattern = build_csr_pattern_from_dofmap(coarse.cell_to_node(), 3 * coarse.num_nodes,
                                                      dof_numel=3, dof_priority=self.dof_priority)
        self._coarse_prolongations = [_prolongation(f, c, self.dof_priority)
                                      for f, c in zip(self.grids[1:-1], self.grids[2:])]

    def _local_order(self) -> List[Tuple[int, int]]:
        """单元局部张量自由度依次对应的 (局部顶点, 分量)."""
        if self.dof_priority:
            return [(v, c) for c in range(3) for v in range(8)]
        return [(v, c) for v in range(8) for c in range(3)]

    @property
    def num_levels(self) -> int:
        """层数, 含最细层."""
        return len(self.grids)

    def update(self, coef: TensorLike) -> None:
        """按单元系数重建全部粗层算子.

        Parameters
        ----------
        coef : (NC, ) 的单元系数 s_e, 即细层刚度 K_e = s_e K^0.

        Raises
        ------
        ValueError
            ``coef`` 形状与最细层单元数不符.
        """
        fine = self.grids[0]
        s = np.asarray(bm.to_numpy(coef), dtype=np.float64)
        if s.shape != (fine.num_cells, ):
            raise ValueError(f'coef 须为 ({fine.num_cells}, ), 得到 {s.shape}')
        if self.num_levels == 1:
            self._operators = []
            return

        # 第 2 层: K_E = sum_c s_c M_c, 域外子单元系数为零; 支承单元另加修正
        coarse = self.grids[1]
        padded = np.zeros(tuple(2 * n for n in coarse.shape))
        padded[:fine.shape[0], :fine.shape[1], :fine.shape[2]] = s.reshape(fine.shape)
        nxc, nyc, nzc = coarse.shape
        children = padded.reshape(nxc, 2, nyc, 2, nzc, 2).transpose(0, 2, 4, 1, 3, 5).reshape(-1, 8)
        K_E = np.einsum('ec,cij->eij', children, self._child_matrices)
        np.add.at(K_E, self._boundary_parents,
                  s[self._boundary_cells, None, None] * self._boundary_corrections)
        operator = assemble_csr(K_E, self._pattern)
        del K_E
        values = np.asarray(operator.values)
        slots = _diagonal_slots(operator)
        decoupled = _decoupled(values[slots])
        values[slots[decoupled]] += 1.0

        operators = [operator]
        prolongations = [self._first_prolongation]
        for P in self._coarse_prolongations:
            P0 = P
            if decoupled.any():
                P0 = CSRTensor.from_scipy(sp.diags((~decoupled).astype(np.float64)) @ P.to_scipy())
            operator = P0.T @ operators[-1] @ P0
            decoupled = _decoupled(np.asarray(operator_diagonal(operator)))
            if decoupled.any():
                identity = sp.diags(decoupled.astype(np.float64), format='csr')
                operator = CSRTensor.from_scipy(operator.to_scipy() + identity)
            operators.append(operator)
            prolongations.append(P0)
        self._operators = operators
        self._prolongations = prolongations

    @property
    def operators(self) -> List[CSRTensor]:
        """由细到粗的粗层算子 (不含最细层); 未 update 时报错."""
        if self._operators is None:
            raise RuntimeError('StructuredHexHierarchy 尚未 update, 没有粗层算子')
        return self._operators

    @property
    def prolongations(self) -> List[CSRTensor]:
        """由细到粗的延拓, 第 l 个把第 l + 1 层映到第 l 层 (0 为最细层); 已按约束置零行."""
        return self._prolongations

    def build_multigrid(self, *, omega: Optional[float] = None, sweeps: int = 1,
                        coarse_solver: str = 'scipy') -> Multigrid:
        """以当前粗层算子构造 V 循环多重网格; 最细层算子留空, 由 ``Multigrid.setup(K)`` 填入.

        Parameters
        ----------
        omega, sweeps : 各层加权 Jacobi 光滑子的松弛因子与扫描次数; ``omega`` 缺省时各层按
            本层算子的谱估计自动取值, 见 :class:`~soptx.solvers.JacobiSmoother`.
        coarse_solver : 最粗层直接法后端, ``'scipy'`` 或 ``'mumps'``.

        Returns
        -------
        mg : 尚未 setup 的多重网格.
        """
        operators = [None] + self.operators
        mg = Multigrid(coarse_solver=DirectSolver(coarse_solver))
        mg.add_level(operators[-1])
        for level in range(self.num_levels - 2, -1, -1):
            mg.add_level(operators[level], JacobiSmoother(omega=omega, sweeps=sweeps),
                         self._prolongations[level])

        return mg

"""T1: 粗/细两级网格与子结构自由度映射.

在矩形域上构造均匀四边形细网格 (nx = ncx*L, ny = ncy*L), 粗网格由
ncx*ncy 个子结构 (粗单元) 构成, 每个子结构内部含 L*L 个细单元.

依赖的 FEALPy 结构化约定 (在 __init__ 中逐条 assert 校验, 不静默信任):

1. ``QuadrangleMesh.from_box`` 节点编号 x-major: node(ix, iy) = ix*(ny+1) + iy;
2. p=1 标量 ``cell_to_dof()`` 每行按全局节点编号升序:
   [ll, ll+1, ll+(ny+1), ll+(ny+2)], 其中 ll 为单元左下角节点;
3. ``TensorFunctionSpace(shape=(-1, GD))`` 自由度按节点交错: dof = node*GD + comp.

子结构内部使用统一的"参考局部编号" (所有子结构共享):

- 局部节点 (lix, liy) ∈ [0, L]^2, 编号 lix*(L+1) + liy (字典序);
- 内部节点: 0 < lix < L 且 0 < liy < L; 其余为子结构边界 (接口) 节点;
- 缩聚用局部自由度顺序为 [内部节点字典序; 边界节点字典序], 各节点 GD 个分量交错.
"""

from __future__ import annotations

import numpy as np


class CoarseFineMeshPair:
    """粗/细两级均匀四边形网格对与子结构映射 (2D, p=1, GD=2)."""

    def __init__(self, domain=(0.0, 1.0, 0.0, 1.0), ncx: int = 4, ncy: int = 4,
                 L: int = 5):
        from fealpy.mesh import QuadrangleMesh
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

        self.domain = tuple(float(v) for v in domain)
        self.ncx, self.ncy, self.L = int(ncx), int(ncy), int(L)
        self.nx, self.ny = self.ncx * self.L, self.ncy * self.L
        self.GD = 2

        self.fine_mesh = QuadrangleMesh.from_box(box=list(self.domain),
                                                 nx=self.nx, ny=self.ny)
        self.scalar_space = LagrangeFESpace(self.fine_mesh, p=1, ctype="C")
        self.tensor_space = TensorFunctionSpace(self.scalar_space,
                                                shape=(-1, self.GD))

        nx, ny, L = self.nx, self.ny, self.L
        node = np.asarray(self.fine_mesh.entity("node"))
        self.node = node
        self.n_nodes = (nx + 1) * (ny + 1)
        self.n_dofs = self.GD * self.n_nodes
        assert self.tensor_space.number_of_global_dofs() == self.n_dofs

        # -- 校验约定 1: 节点 x-major 编号 --
        x0, x1, y0, y1 = self.domain
        hx, hy = (x1 - x0) / nx, (y1 - y0) / ny
        ix = np.repeat(np.arange(nx + 1), ny + 1)
        iy = np.tile(np.arange(ny + 1), nx + 1)
        expected = np.stack([x0 + ix * hx, y0 + iy * hy], axis=1)
        assert np.allclose(node, expected), "QuadrangleMesh 节点编号非 x-major"
        self.node_ix, self.node_iy = ix, iy
        self.hx, self.hy = hx, hy

        # -- 校验约定 2: 标量 cell_to_dof 升序模式; 并确定单元网格编号 --
        c2d = np.asarray(self.scalar_space.cell_to_dof())
        cx = np.repeat(np.arange(nx), ny)
        cy = np.tile(np.arange(ny), nx)
        ll = cx * (ny + 1) + cy
        expected_c2d = np.stack([ll, ll + 1, ll + (ny + 1), ll + (ny + 2)], axis=1)
        assert np.array_equal(c2d, expected_c2d), "cell_to_dof 非预期升序模式"
        self.cell_ix, self.cell_iy = cx, cy

        # -- 校验约定 3: tensor dof 节点交错 --
        tc2d = np.asarray(self.tensor_space.cell_to_dof())
        expected_t = np.stack(
            [2 * expected_c2d + k for k in range(self.GD)], axis=2
        ).reshape(-1, 4 * self.GD)
        assert np.array_equal(tc2d, expected_t), "tensor dof 非节点交错"

        # -- 接口 (骨架) 自由度: 位于粗网格线上的细节点 --
        on_skeleton = (ix % L == 0) | (iy % L == 0)
        self.interface_nodes = np.where(on_skeleton)[0]
        self.interface_dofs = self._nodes_to_dofs(self.interface_nodes)
        self.interior_dofs = self._nodes_to_dofs(np.where(~on_skeleton)[0])
        self.n_interface_dofs = self.interface_dofs.size
        # 全局 dof -> 接口方程编号 (-1 表示非接口)
        self.interface_index = np.full(self.n_dofs, -1, dtype=np.int64)
        self.interface_index[self.interface_dofs] = np.arange(
            self.n_interface_dofs, dtype=np.int64)

        # -- 参考子结构局部编号 (所有子结构共享) --
        lix = np.repeat(np.arange(L + 1), L + 1)
        liy = np.tile(np.arange(L + 1), L + 1)
        interior_mask = (lix > 0) & (lix < L) & (liy > 0) & (liy < L)
        self.local_interior_nodes = np.where(interior_mask)[0]
        self.local_boundary_nodes = np.where(~interior_mask)[0]
        self.n_local_interior_nodes = self.local_interior_nodes.size
        self.n_local_boundary_nodes = self.local_boundary_nodes.size
        # [内部; 边界] 顺序的局部节点排列
        self._local_node_perm = np.concatenate(
            [self.local_interior_nodes, self.local_boundary_nodes])
        self._local_lix, self._local_liy = lix, liy

    # ------------------------------------------------------------------
    # 子结构访问 (j = cx * ncy + cy, 与全局单元编号同为 x-major)
    # ------------------------------------------------------------------

    @property
    def n_sub(self) -> int:
        return self.ncx * self.ncy

    def sub_coarse_index(self, j: int):
        return divmod(int(j), self.ncy)

    def sub_cells(self, j: int) -> np.ndarray:
        """子结构 j 内细单元全局编号, 按参考局部单元字典序 (lcx, lcy)."""
        cx, cy = self.sub_coarse_index(j)
        lcx = np.repeat(np.arange(self.L), self.L)
        lcy = np.tile(np.arange(self.L), self.L)
        return (cx * self.L + lcx) * self.ny + (cy * self.L + lcy)

    def sub_nodes(self, j: int) -> np.ndarray:
        """子结构 j 全局节点编号, 按 [内部字典序; 边界字典序] 排列."""
        cx, cy = self.sub_coarse_index(j)
        gnode = ((cx * self.L + self._local_lix) * (self.ny + 1)
                 + (cy * self.L + self._local_liy))
        return gnode[self._local_node_perm]

    def sub_dofs(self, j: int) -> np.ndarray:
        """子结构 j 全局张量自由度, [内部; 边界] 顺序, 节点分量交错."""
        return self._nodes_to_dofs(self.sub_nodes(j))

    def sub_boundary_dofs(self, j: int) -> np.ndarray:
        return self.sub_dofs(j)[self.GD * self.n_local_interior_nodes:]

    def sub_interior_dofs(self, j: int) -> np.ndarray:
        return self.sub_dofs(j)[: self.GD * self.n_local_interior_nodes]

    def sub_node_coords(self, j: int) -> np.ndarray:
        """子结构 j 节点坐标, 与 sub_nodes 同序 (刚体模态检验用)."""
        return self.node[self.sub_nodes(j)]

    # ------------------------------------------------------------------

    def _nodes_to_dofs(self, nodes: np.ndarray) -> np.ndarray:
        return (self.GD * np.asarray(nodes)[:, None]
                + np.arange(self.GD)[None, :]).reshape(-1)

    def dofs_on_line_x(self, x: float, tol: float = 1e-12) -> np.ndarray:
        """竖直线 x=const 上全部张量自由度 (施加 Dirichlet 用)."""
        nodes = np.where(np.abs(self.node[:, 0] - x) < tol)[0]
        return self._nodes_to_dofs(nodes)

    def dof_at_point(self, x: float, y: float, comp: int) -> int:
        """最接近 (x, y) 节点的第 comp 分量自由度 (施加点载荷用)."""
        d2 = (self.node[:, 0] - x) ** 2 + (self.node[:, 1] - y) ** 2
        return int(self.GD * np.argmin(d2) + comp)

    def summary(self) -> dict:
        return {
            "coarse": f"{self.ncx}x{self.ncy}",
            "L": self.L,
            "fine": f"{self.nx}x{self.ny}",
            "n_fine_dofs": self.n_dofs,
            "n_interface_dofs": self.n_interface_dofs,
            "dof_reduction": self.n_dofs / self.n_interface_dofs,
        }

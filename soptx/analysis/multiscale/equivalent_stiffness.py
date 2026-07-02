"""T3: 接口缩聚方程组装 + T5: 单步前向闭环 (含接口求解与细尺度恢复).

各子结构内部自由度互不耦合, 故按接口自由度组装逐子结构 Schur 补

    K^cond = sum_j (A^j)^T K_s^j A^j

与全尺度刚度阵消去全部内部自由度的全局 Schur 补数学等价 (V1 验证).
全局求解自由度 = 接口自由度, 远小于全尺度细网格自由度.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


class InterfaceCondensedSystem:
    """接口缩聚系统: 组装 K^cond, 缩聚载荷, 接口求解, 细尺度恢复."""

    def __init__(self, mesh_pair, predictor, rho: np.ndarray):
        """rho: (NC_fine,) 逐细单元密度 (已插值的相对刚度系数 coef)."""
        self.mesh_pair = mesh_pair
        self.predictor = predictor
        self.rho = np.asarray(rho, dtype=np.float64)
        assert self.rho.shape == (mesh_pair.fine_mesh.number_of_cells(),)

        self.ops: list = []
        self._sub_iface_idx: list[np.ndarray] = []
        n_iface = mesh_pair.n_interface_dofs

        rows, cols, vals = [], [], []
        for j in range(mesh_pair.n_sub):
            rho_local = self.rho[mesh_pair.sub_cells(j)]
            op = predictor.predict(rho_local)
            self.ops.append(op)

            idx = mesh_pair.interface_index[mesh_pair.sub_boundary_dofs(j)]
            assert np.all(idx >= 0), "子结构边界自由度不在全局接口集合内"
            self._sub_iface_idx.append(idx)

            n_b = idx.size
            rows.append(np.repeat(idx, n_b))
            cols.append(np.tile(idx, n_b))
            vals.append(op.K_s.reshape(-1))

        self.K_cond = coo_matrix(
            (np.concatenate(vals),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_iface, n_iface),
        ).tocsr()

    def condense_load(self, F: np.ndarray) -> np.ndarray:
        """全尺度载荷 F (n_dofs,) -> 接口缩聚载荷 F_b (n_iface,).

        F_b = F[接口] - sum_j X_j^T F_i^j (接口分量直接取值, 避免重复计数).
        """
        mp = self.mesh_pair
        F = np.asarray(F, dtype=np.float64)
        F_b = F[mp.interface_dofs].copy()
        for j, op in enumerate(self.ops):
            F_i = F[mp.sub_interior_dofs(j)]
            if np.any(F_i):
                F_b[self._sub_iface_idx[j]] += -op.X.T @ F_i
        return F_b

    def solve_interface(self, F: np.ndarray, fixed_dofs: np.ndarray):
        """接口求解 K^cond U_b = F_b (Dirichlet 行列消去).

        fixed_dofs 为全局张量自由度编号, 必须落在接口集合内
        (均匀两级网格下域边界天然位于粗网格骨架上).
        返回 (U_b, info), info 含接口方程残差与规模参数.
        """
        mp = self.mesh_pair
        fixed_idx = mp.interface_index[np.asarray(fixed_dofs)]
        assert np.all(fixed_idx >= 0), "Dirichlet 自由度必须是接口自由度"

        F_b = self.condense_load(F)
        K_bc = self.K_cond.tolil(copy=True)
        K_bc[fixed_idx, :] = 0.0
        K_bc[:, fixed_idx] = 0.0
        K_bc[fixed_idx, fixed_idx] = 1.0
        K_bc = K_bc.tocsr()
        F_bc = F_b.copy()
        F_bc[fixed_idx] = 0.0

        U_b = spsolve(K_bc, F_bc)
        residual = (np.linalg.norm(K_bc @ U_b - F_bc)
                    / max(np.linalg.norm(F_bc), 1e-300))
        info = {
            "interface_residual": float(residual),
            "n_fine_dofs": mp.n_dofs,
            "n_interface_dofs": mp.n_interface_dofs,
        }
        return U_b, info

    def recover_fine(self, U_b: np.ndarray,
                     F: np.ndarray | None = None) -> np.ndarray:
        """细尺度恢复 u^fine: 接口取 U_b, 内部 u_i = -X u_b (+ K_ii^{-1} F_i)."""
        mp = self.mesh_pair
        u = np.zeros(mp.n_dofs)
        u[mp.interface_dofs] = U_b
        for j, op in enumerate(self.ops):
            u_b = U_b[self._sub_iface_idx[j]]
            F_i = None if F is None else np.asarray(F)[mp.sub_interior_dofs(j)]
            if F_i is not None and not np.any(F_i):
                F_i = None
            u[mp.sub_interior_dofs(j)] = op.recover_interior(u_b, F_i)
        return u

"""T2: 参考子结构模板与静力缩聚 (Schur 补).

多尺度形函数与等效刚度采用子结构静力缩聚形式 (路线①):

    N^j = [-(K_ii^j)^{-1} K_ib^j; I],
    K_s^j = (N^j)^T K^j N^j = K_bb^j - (K_ib^j)^T (K_ii^j)^{-1} K_ib^j.

均匀细网格下所有细单元同构, 单元刚度 K_e = rho_e * KE_ref, 因此局部
子结构刚度 K^j(rho) 只依赖参考单元刚度 KE_ref 与参考局部散射模板
(问题无关性): 预测器只需消费局部密度 rho_local, 不感知全局网格.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve


@dataclass
class SubstructureOperator:
    """单个子结构的缩聚结果 (预测器 predict 的返回类型).

    局部自由度顺序与 CoarseFineMeshPair.sub_dofs 一致: [内部; 边界].
    """

    X: np.ndarray            # (n_i, n_b) = K_ii^{-1} K_ib
    K_s: np.ndarray          # (n_b, n_b) 缩聚 (等效) 刚度
    lu_ii: tuple | None = None  # K_ii 的 LU 分解 (内部载荷恢复用)

    @property
    def N(self) -> np.ndarray:
        """多尺度形函数 (n_i + n_b, n_b), 上块 -X, 下块单位阵."""
        n_b = self.K_s.shape[0]
        return np.vstack([-self.X, np.eye(n_b)])

    def recover_interior(self, u_b: np.ndarray,
                         F_i: np.ndarray | None = None) -> np.ndarray:
        """u_i = -X u_b + K_ii^{-1} F_i."""
        u_i = -self.X @ u_b
        if F_i is not None and self.lu_ii is not None:
            u_i = u_i + lu_solve(self.lu_ii, F_i)
        return u_i

    def condense_load(self, F_i: np.ndarray, F_b: np.ndarray) -> np.ndarray:
        """接口缩聚载荷 (N^j)^T F^j = F_b - X^T F_i."""
        return F_b - self.X.T @ F_i


class SubstructureTemplate:
    """参考子结构: KE_ref + 局部散射模板 + 静力缩聚.

    所有子结构共享同一模板; condense(rho_local) 即精确静力缩聚,
    也是 ExactPredictor 的实现与 PIML 训练标注的来源.
    """

    def __init__(self, mesh_pair, material, q: int = 4):
        from soptx.analysis.integrators.linear_elastic_integrator import (
            LinearElasticIntegrator,
        )

        self.mesh_pair = mesh_pair
        self.material = material
        self.q = int(q)
        L, GD = mesh_pair.L, mesh_pair.GD

        integrator = LinearElasticIntegrator(material=material, q=self.q,
                                             method="standard")
        KE = np.asarray(integrator.assembly(mesh_pair.tensor_space))
        assert np.allclose(KE[0], KE[-1]) and np.allclose(KE[0], KE[KE.shape[0] // 2]), \
            "细单元刚度不同构, 均匀网格假设不成立"
        self.KE_ref = KE[0].copy()          # (2*4, 2*4), coef=1 参考单元刚度

        # 参考局部单元 -> 局部节点 (升序, 与全局 cell_to_dof 同一相对顺序)
        lcx = np.repeat(np.arange(L), L)
        lcy = np.tile(np.arange(L), L)
        ll = lcx * (L + 1) + lcy
        local_c2n = np.stack([ll, ll + 1, ll + (L + 1), ll + (L + 2)], axis=1)

        # 局部节点字典序 -> [内部; 边界] 缩聚顺序 的重编号
        n_nodes_loc = (L + 1) ** 2
        node_perm_inv = np.empty(n_nodes_loc, dtype=np.int64)
        node_perm_inv[mesh_pair._local_node_perm] = np.arange(n_nodes_loc)
        c2n_perm = node_perm_inv[local_c2n]
        self.local_c2d = (GD * c2n_perm[:, :, None]
                          + np.arange(GD)[None, None, :]).reshape(-1, 4 * GD)

        self.n_i = GD * mesh_pair.n_local_interior_nodes
        self.n_b = GD * mesh_pair.n_local_boundary_nodes
        n_loc = self.n_i + self.n_b
        self._rows = np.repeat(self.local_c2d, 4 * GD, axis=1).reshape(-1)
        self._cols = np.tile(self.local_c2d, (1, 4 * GD)).reshape(-1)
        self.n_loc = n_loc

    def local_stiffness(self, rho_local: np.ndarray) -> np.ndarray:
        """K^j(rho_local), (n_loc, n_loc), [内部; 边界] 顺序."""
        rho_local = np.asarray(rho_local, dtype=np.float64)
        assert rho_local.shape == (self.local_c2d.shape[0],)
        vals = (rho_local[:, None, None] * self.KE_ref[None]).reshape(-1)
        K = np.zeros((self.n_loc, self.n_loc))
        np.add.at(K, (self._rows, self._cols), vals)
        return K

    def condense(self, rho_local: np.ndarray) -> SubstructureOperator:
        """静力缩聚: 精确 Schur 补 (ExactPredictor / 训练标注)."""
        K = self.local_stiffness(rho_local)
        n_i = self.n_i
        K_ii, K_ib = K[:n_i, :n_i], K[:n_i, n_i:]
        K_bb = K[n_i:, n_i:]
        lu_ii = lu_factor(K_ii)
        X = lu_solve(lu_ii, K_ib)
        K_s = K_bb - K_ib.T @ X
        return SubstructureOperator(X=X, K_s=K_s, lu_ii=lu_ii)

"""V1 参照: 全尺度细网格装配、全局 Schur 补与直解.

与子结构路径共用同一 LinearElasticIntegrator 配置, 保证对照在同一
离散模型上进行 (机器精度可比).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu, spsolve


def assemble_fullscale_stiffness(mesh_pair, material, rho: np.ndarray,
                                 q: int = 4) -> csr_matrix:
    """全尺度细网格刚度阵 (scipy CSR), coef = rho."""
    from fealpy.fem import BilinearForm
    from soptx.analysis.integrators.linear_elastic_integrator import (
        LinearElasticIntegrator,
    )

    integrator = LinearElasticIntegrator(material=material, q=q,
                                         method="standard")
    integrator.coef = np.asarray(rho, dtype=np.float64)
    bform = BilinearForm(mesh_pair.tensor_space)
    bform.add_integrator(integrator)
    K = bform.assembly(format="coo")
    return K.to_scipy().tocsr()


def fullscale_schur_complement(K: csr_matrix, b_dofs: np.ndarray,
                               chunk: int = 512) -> np.ndarray:
    """全局 Schur 补 S = K_bb - K_bi K_ii^{-1} K_ib (接口自由度 b_dofs).

    K_ii 稀疏 LU 分解后按列块求解, 返回稠密 (n_b, n_b).
    """
    n = K.shape[0]
    mask = np.zeros(n, dtype=bool)
    mask[b_dofs] = True
    i_dofs = np.where(~mask)[0]

    K_ii = K[i_dofs][:, i_dofs].tocsc()
    K_ib = K[i_dofs][:, b_dofs].tocsc()
    S = K[b_dofs][:, b_dofs].toarray()

    lu = splu(K_ii)
    n_b = b_dofs.size
    for start in range(0, n_b, chunk):
        cols = slice(start, min(start + chunk, n_b))
        X = lu.solve(K_ib[:, cols].toarray())
        S[:, cols] -= K_ib.T @ X
    return S


def solve_with_dirichlet(K: csr_matrix, F: np.ndarray,
                         fixed_dofs: np.ndarray) -> np.ndarray:
    """行列消去式 Dirichlet + 直解 (与仓库既有测试同一约定)."""
    K_bc = K.tolil(copy=True)
    fixed = np.asarray(fixed_dofs, dtype=int)
    K_bc[fixed, :] = 0.0
    K_bc[:, fixed] = 0.0
    K_bc[fixed, fixed] = 1.0
    F_bc = np.asarray(F, dtype=np.float64).copy()
    F_bc[fixed] = 0.0
    return spsolve(K_bc.tocsr(), F_bc)

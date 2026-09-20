# -*- coding: utf-8 -*-
"""积分点上几何量与材料量的共用构造.

PA 与 UA 喂给 B 和 D 的输入完全相同, 区别只在何时算: PA 在 ``build`` 里算一次并
常驻, UA 每次 apply 现算. 把这段抽出来共用, 两个层级走的浮点运算就是字面上同一
串, 于是 "UA 与 PA 逐位相同" 是结构上的保证而不是抄写的巧合 -- 任何一方改了这里,
另一方跟着改, 等式不会悄悄破掉.

只认线弹性: 从 ``LinearElasticIntegrator`` 取材料与积分阶. 换方程时另写一份即可,
``DofToQuad`` 与 ``QFunction`` 的接口不用动.
"""

from typing import NamedTuple

from fealpy.backend import backend_manager as bm
from fealpy.mesh import SimplexMesh
from fealpy.functionspace.utils import flatten_indices
from fealpy.typing import TensorLike, _S


class QuadratureGeometry(NamedTuple):
    """B 与 D 所需的全部逐积分点输入.

    Attributes
    ----------
    grad_ref : (NQ, ldof, R) 参考单元上的基函数梯度, 与单元无关.
    jacobi_inverse : (NC, NQ, R, GD) 几何因子 d xi_r / d x_b.
    weighted_measure : (NC, NQ) 积分权重乘 Jacobi 行列式.
    dof_permutation : (ldof, GD) 标量自由度到单元自由度槽位的映射.
    """

    grad_ref: TensorLike
    jacobi_inverse: TensorLike
    weighted_measure: TensorLike
    dof_permutation: TensorLike


def quadrature_geometry(space, integrator) -> QuadratureGeometry:
    """算出积分点上的几何因子、积分权重与自由度重排.

    Parameters
    ----------
    space : 该双线性型所在的张量函数空间.
    integrator : ``LinearElasticIntegrator``, 提供积分阶与单元子集.

    Returns
    -------
    QuadratureGeometry

    Raises
    ------
    TypeError
        积分子不是 ``LinearElasticIntegrator``.
    NotImplementedError
        参考维数与几何维数不等 (嵌入流形网格).

    Notes
    -----
    积分权重按网格类型分两支, 与 ``LinearElasticIntegrator.assembly`` 完全一致:
    单纯形网格的重心坐标求积权重之和为 1 而不是参考单元的测度, 故那里用的是单元
    测度 cm 而非 |J|. 若这里改用 |J|, 三角形会差 2 倍, 四面体差 6 倍.
    """
    from soptx.fem.integrators import LinearElasticIntegrator

    if not isinstance(integrator, LinearElasticIntegrator):
        raise TypeError(
            "PA / UA 层级目前只支持 LinearElasticIntegrator, 得到 "
            f"{type(integrator).__name__}"
        )

    scalar_space = space.scalar_space
    mesh = scalar_space.mesh
    index = integrator.index

    q = scalar_space.p + 3 if integrator.q is None else integrator.q
    qf = mesh.quadrature_formula(q)
    bcs, ws = qf.get_quadrature_points_and_weights()

    # 参考单元上的基函数梯度 (NQ, ldof, R), 与单元无关, 全网格共享一份
    grad_ref = scalar_space.grad_basis(bcs, index=index, variable='u')

    # 几何因子 (NC, NQ, GD, R); entity_view 的 index 用 None 表示全体
    geo_index = None if index is _S else index
    jacobi = mesh.entity_view('cell').jacobi_matrix(bcs, index=geo_index)

    geo_dimension = int(jacobi.shape[-2])
    ref_dimension = int(jacobi.shape[-1])
    if geo_dimension != ref_dimension:
        raise NotImplementedError(
            f"该层级要求 Jacobi 矩阵可逆, 但参考维数 {ref_dimension} 与几何维数 "
            f"{geo_dimension} 不等 (嵌入流形网格). 这类网格上 B 要走伪逆, 与本层级"
            "的存储假设不同, 留到需要时单独实现"
        )

    # inv 后下标变成 (NC, NQ, R, GD), 即 d xi_r / d x_b
    jacobi_inverse = bm.linalg.inv(jacobi)

    if isinstance(mesh, SimplexMesh):
        cell_measure = mesh.entity_measure('cell', index=index)
        weighted_measure = ws[None, :] * cell_measure[:, None]
    else:
        weighted_measure = ws[None, :] * bm.abs(bm.linalg.det(jacobi))

    local_dofs = int(grad_ref.shape[1])
    dof_permutation = flatten_indices(
                            (local_dofs, geo_dimension),
                            (1, 0) if space.dof_priority else (0, 1)
                        )

    return QuadratureGeometry(grad_ref=grad_ref,
                            jacobi_inverse=jacobi_inverse,
                            weighted_measure=weighted_measure,
                            dof_permutation=dof_permutation)

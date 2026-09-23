# -*- coding: utf-8 -*-
"""积分点上的逐点算子 (Q-function).

本模块实现矩阵自由算子分解

    A = P^T G^T B^T D B G P

中的 D 一项: 它在每个积分点上独立作用, 把 B 交出的量 (对位移元是 grad u) 换成与之
共轭的量 (乘上积分权重的应力). 物理只出现在这一层 -- ``ReferenceBasis`` 与
``GeometricFactors`` 只管基函数与几何, ``PartialAssembly`` 只管数据流.

与 ``gradients`` 模块同一分工: ``weighted_stress`` 与 ``weighted_stress_diagonal`` 是
只吃数组的计算核, ``LinearElasticQFunction`` 只持有并更新它们要的数组.

目前只有线弹性一种实现, 故不设抽象基类; 出现第二种方程时, 再按两者的共同部分抽出
接口.

与 libCEED 的 ``CeedQFunction`` 对应. 逐点意味着它不含任何跨积分点的耦合, 因此随
设计变量更新时只改每积分点一个标量, 不需要重新积分. 单元密度 (NC, ) 下 EA 同样不必
重新积分: 单元矩阵对 rho_e 线性, 由实体单元矩阵逐单元缩放即得; 只有逐点密度
(NC, NQ) 使单元矩阵无法由缩放得到, 这时 EA 要重新积分, 而 PA 的更新代价不变.
"""

from typing import Optional, Sequence

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike


def strain_map(geo_dimension: int,
            shear_order: Sequence[str] = ("yz", "xz", "xy"),
            **kwargs,
        ) -> TensorLike:
    """工程应变对位移梯度的常量映射, 形状 (NS, GD, GD).

    ``eps_s = sum_{d, b} map[s, d, b] * d u_d / d x_b``. 正应变行在前, 工程剪应变
    行在后, 与 ``LinearElasticMaterial.strain_matrix`` 的行次序逐行一致 -- 两者必须
    同一套约定, 本构矩阵才对得上.

    Parameters
    ----------
    geo_dimension : 几何维数, 取 2 或 3.
    shear_order : 三维工程剪应变分量的排列次序, 默认与 ``strain_matrix`` 相同.
    **kwargs : 传给 ``bm.tensor`` 的 dtype / device.

    Returns
    -------
    strain_map : (NS, GD, GD) 的常量数组, NS = GD * (GD + 1) / 2.
    """
    if geo_dimension not in (2, 3):
        raise ValueError(f"只支持 GD 为 2 或 3, 得到 {geo_dimension}")

    if geo_dimension == 2:
        shear_pairs = ((0, 1), )
    else:
        index_map = {"xy": (0, 1), "yz": (1, 2), "xz": (2, 0)}
        if len(shear_order) != 3 or set(shear_order) != set(index_map):
            raise ValueError(
                f"shear_order 必须是 ('xy', 'yz', 'xz') 的一个排列, 得到 {shear_order}"
            )
        shear_pairs = tuple(index_map[name] for name in shear_order)

    n_strain = geo_dimension * (geo_dimension + 1) // 2
    rows = []
    for s in range(n_strain):
        entry = [[0.0] * geo_dimension for _ in range(geo_dimension)]
        if s < geo_dimension:
            entry[s][s] = 1.0
        else:
            i, j = shear_pairs[s - geo_dimension]
            entry[i][j] = 1.0
            entry[j][i] = 1.0
        rows.append(entry)

    return bm.tensor(rows, **kwargs)


def weighted_stress(grad_u: TensorLike,
                *,
                weighted_coef: TensorLike,
                elastic_matrix: TensorLike,
                strain_map: TensorLike,
            ) -> TensorLike:
    """逐点作用 D: grad u -> w |J| rho S^T D S grad u.

    每个积分点上是一个 GD^2 x GD^2 的小矩阵乘向量, 全体积分点拼起来即块对角矩阵 D;
    这里不展开该矩阵, 由三次收缩逐点完成.

    Parameters
    ----------
    grad_u : (NC, NQ, GD, GD[, NB]) 的位移物理梯度.
    weighted_coef : (NC, NQ) 的逐点标量 w |J| rho.
    elastic_matrix : (NS, NS) 的本构矩阵.
    strain_map : (NS, GD, GD) 的工程应变映射 S.

    Returns
    -------
    s_Q : 与 ``grad_u`` 同形状, 与 grad u 共轭的量, 即乘了积分权重的对称应力张量.
    """
    # Voigt 工程应变 eps = S grad u: (NC, NQ, NS[, NB])
    strain = bm.einsum('sdb, cqdb... -> cqs...', strain_map, grad_u)

    # Voigt 应力乘逐点标量 w |J| rho D eps: (NC, NQ, NS[, NB])
    stress = bm.einsum('cq, st, cqt... -> cqs...',
                    weighted_coef, elastic_matrix, strain)

    # 加权应力张量 S^T sigma, 与 grad u 共轭: (NC, NQ, GD, GD[, NB])
    s_Q = bm.einsum('sdb, cqs... -> cqdb...', strain_map, stress)

    return s_Q


def weighted_stress_diagonal(basis_gradients: TensorLike,
                            *,
                            weighted_coef: TensorLike,
                            quadratic_form: TensorLike,
                        ) -> TensorLike:
    """由物理基函数梯度算单元矩阵 B^T D B 的对角.

    第 (i, d) 个单元自由度上的对角元是

        sum_q wc[c, q] sum_{b, e} M[d, b, e] gphi[c, q, i, b] gphi[c, q, i, e]

    其中 wc 即 ``weighted_coef``, M 即 ``quadratic_form``. 按分量 d 逐个算, 中间量
    始终不超过 gphi 本身的规模, 循环最多 3 次.

    Parameters
    ----------
    basis_gradients : (NC, NQ, ldof, GD) 的物理基函数梯度.
    weighted_coef : (NC, NQ) 的逐点标量 w |J| rho.
    quadratic_form : (GD, GD, GD) 的二次型常量, 见
        ``LinearElasticQFunction.quadratic_form``.

    Returns
    -------
    diag : (NC, ldof, GD) 的单元矩阵对角.
    """
    blocks = []
    for d in range(quadratic_form.shape[0]):
        projected = bm.einsum('cqib, be -> cqie', basis_gradients, quadratic_form[d])
        blocks.append(bm.einsum('cq, cqie, cqie -> ci',
                            weighted_coef, projected, basis_gradients))

    return bm.stack(blocks, axis=-1)


class LinearElasticQFunction:
    """线弹性逐点算子 D 的常驻数据, 由 ``weighted_stress`` 与
    ``weighted_stress_diagonal`` 按数组取用.

    Parameters
    ----------
    elastic_matrix : (NS, NS) 的本构矩阵, 全域一致.
    weighted_measure : (NC, NQ) 的积分权重乘 Jacobi 行列式.
    coef : 相对密度系数, 可为 None, (NC, ) 或 (NC, NQ), 与
        ``LinearElasticIntegrator`` 的约定相同.
    shear_order : 三维工程剪应变分量的排列次序.

    Notes
    -----
    只支持全域一致的本构矩阵: 各向异性或逐单元变化的材料要把本构矩阵换成带
    (NC, NQ) 前置维的数组, 逐点核的形式不变, 但常驻量会多出一份 O(NC * NQ * NS^2),
    PA 的存储优势随之改变, 因此留到需要时单独做.
    """

    def __init__(self,
                elastic_matrix: TensorLike,
                weighted_measure: TensorLike,
                coef: Optional[TensorLike] = None,
                shear_order: Sequence[str] = ("yz", "xz", "xy"),
            ) -> None:
        if elastic_matrix.ndim != 2:
            raise ValueError(
                "elastic_matrix 必须是 (NS, NS) 的二维数组, 得到 "
                f"shape={tuple(elastic_matrix.shape)}"
            )
        if weighted_measure.ndim != 2:
            raise ValueError(
                "weighted_measure 必须是 (NC, NQ) 的二维数组, 得到 "
                f"shape={tuple(weighted_measure.shape)}"
            )

        n_strain = int(elastic_matrix.shape[0])
        geo_dimension = 2 if n_strain == 3 else 3
        if geo_dimension * (geo_dimension + 1) // 2 != n_strain:
            raise ValueError(f"无法由 NS={n_strain} 推出几何维数")

        self._D = elastic_matrix
        self._weighted_measure = weighted_measure
        self._geo_dim = geo_dimension
        self._strain_map = strain_map(geo_dimension, shear_order,
                                    **bm.context(elastic_matrix))

        # 取对角用的常量: M[d, b, e] = sum_{s, t} map[s, d, b] D[s, t] map[t, d, e],
        # 把 "第 (i, d) 个单元自由度自作用" 的二次型压成 (GD, GD, GD) 的小张量, 免得
        # 在取对角时展开 (NC, NQ, ldof, GD, NS) 的中间量
        self._quadratic_form = bm.einsum('sdb, st, tde -> dbe',
                                        self._strain_map, self._D, self._strain_map)

        self._coef = None
        self._weighted_coef = None
        self.update(coef)

    @property
    def elastic_matrix(self) -> TensorLike:
        """本构矩阵, 形状 (NS, NS)"""
        return self._D

    @property
    def strain_map(self) -> TensorLike:
        """工程应变对位移梯度的常量映射, 形状 (NS, GD, GD)"""
        return self._strain_map

    @property
    def quadratic_form(self) -> TensorLike:
        """取对角用的二次型常量 M, 形状 (GD, GD, GD)"""
        return self._quadratic_form

    @property
    def coef(self) -> Optional[TensorLike]:
        """当前的相对密度系数"""
        return self._coef

    @property
    def weighted_coef(self) -> TensorLike:
        """``weighted_measure * coef``, 即逐积分点的 w |J| rho, 形状 (NC, NQ)"""
        return self._weighted_coef

    @property
    def geo_dimension(self) -> int:
        """几何维数 GD"""
        return self._geo_dim

    def update(self, coef: Optional[TensorLike]) -> None:
        """更新相对密度系数.

        只重算 (NC, NQ) 个标量, 不碰几何因子, 也不重新积分.

        Parameters
        ----------
        coef : None 表示相对密度恒为 1; (NC, ) 为单元密度; (NC, NQ) 为节点密度在积分
            点上的取值.
        """
        weighted_measure = self._weighted_measure
        n_cells, n_quad = weighted_measure.shape

        if coef is None:
            weighted_coef = weighted_measure
        elif tuple(coef.shape) == (n_cells, ):
            weighted_coef = weighted_measure * coef[:, None]
        elif tuple(coef.shape) == (n_cells, n_quad):
            weighted_coef = weighted_measure * coef
        else:
            raise ValueError(
                f"coef 的形状必须是 None, ({n_cells}, ) 或 ({n_cells}, {n_quad}), "
                f"得到 {tuple(coef.shape)}"
            )

        self._coef = coef
        self._weighted_coef = weighted_coef

    def persistent_bytes(self) -> int:
        """常驻内存字节数: 逐点的 weighted_coef 是主项, 本构矩阵与应变映射是 O(1)"""
        total = int(self._weighted_coef.nbytes) + int(self._D.nbytes)
        total += int(self._strain_map.nbytes) + int(self._quadratic_form.nbytes)

        return total

    def __repr__(self) -> str:
        n_cells, n_quad = self._weighted_coef.shape

        return (f"LinearElasticQFunction(n_cells={n_cells}, n_quad={n_quad}, "
                f"geo_dim={self._geo_dim})")

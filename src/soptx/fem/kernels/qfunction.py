# -*- coding: utf-8 -*-
"""积分点上的逐点算子 (Q-function).

本模块实现矩阵自由算子分解

    A = P^T G^T B^T D B G P

中的 D 一项: 它在每个积分点上独立作用, 把 B 交出的量 (对位移元是 grad u) 换成与之
共轭的量 (乘上积分权重的应力). 物理只出现在这一层 -- ``DofToQuad`` 只管几何与基
函数, ``PartialAssembly`` 只管数据流, 换一个方程只需换一个 QFunction.

与 libCEED 的 ``CeedQFunction`` 对应. 逐点意味着它不含任何跨积分点的耦合, 因此随
设计变量更新时只改每积分点一个标量, 不需要重新积分 -- 这是 PA 相对 EA 在拓扑优化
循环里的另一处优势 (EA 每次迭代要重算全部单元矩阵).
"""

from typing import Optional, Sequence

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike


class QFunction:
    """积分点逐点算子的抽象基类.

    Notes
    -----
    梯度类量的下标约定与 ``DofToQuad`` 一致: ``[c, q, d, b]`` 表示 d u_d / d x_b,
    多列右端项的批量维在最后.
    """

    def __call__(self, grad_u: TensorLike) -> TensorLike:
        """逐点作用, 输入输出同形状"""
        raise NotImplementedError

    def diagonal(self, basis_gradients: TensorLike) -> TensorLike:
        """由物理基函数梯度算出单元矩阵的对角.

        Parameters
        ----------
        basis_gradients : (NC, NQ, ldof, GD) 的物理基函数梯度.

        Returns
        -------
        diag : (NC, ldof, GD) 的单元矩阵对角, 下标为 (标量自由度, 分量).
        """
        raise NotImplementedError

    def update(self, coef: Optional[TensorLike]) -> None:
        """随设计变量更新逐点系数, 不重新积分"""
        raise NotImplementedError

    def persistent_bytes(self) -> int:
        """常驻内存字节数"""
        raise NotImplementedError


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


class LinearElasticQFunction(QFunction):
    """线弹性的逐点算子: grad u -> 乘了积分权重的应力.

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
        self._scale = None
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
    def coef(self) -> Optional[TensorLike]:
        """当前的相对密度系数"""
        return self._coef

    @property
    def scale(self) -> TensorLike:
        """逐积分点的标量因子, 形状 (NC, NQ), 即积分权重乘相对密度"""
        return self._scale

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
            scale = weighted_measure
        elif tuple(coef.shape) == (n_cells, ):
            scale = weighted_measure * coef[:, None]
        elif tuple(coef.shape) == (n_cells, n_quad):
            scale = weighted_measure * coef
        else:
            raise ValueError(
                f"coef 的形状必须是 None, ({n_cells}, ) 或 ({n_cells}, {n_quad}), "
                f"得到 {tuple(coef.shape)}"
            )

        self._coef = coef
        self._scale = scale

    def __call__(self, grad_u: TensorLike) -> TensorLike:
        """逐点作用: grad u -> w |J| rho D eps(grad u), 再送回梯度的下标形式.

        Parameters
        ----------
        grad_u : (NC, NQ, GD, GD) 或 (NC, NQ, GD, GD, B) 的物理梯度, 批量维在后.

        Returns
        -------
        s_Q : 与输入同形状, 与 grad u 共轭的量.
        """
        strain = bm.einsum('sdb, cqdb... -> cqs...', self._strain_map, grad_u)
        stress = bm.einsum('st, cqt... -> cqs...', self._D, strain)

        # scale 是 (NC, NQ); 补足到 stress 的秩再相乘, 单列与多列共用一条语句
        scale = bm.reshape(self._scale,
                        tuple(self._scale.shape) + (1, ) * (stress.ndim - 2))
        stress = stress * scale

        return bm.einsum('sdb, cqs... -> cqdb...', self._strain_map, stress)

    def diagonal(self, basis_gradients: TensorLike) -> TensorLike:
        """由物理基函数梯度算单元矩阵对角.

        第 (i, d) 个单元自由度上的对角元是

            sum_q scale[c, q] sum_{b, e} M[d, b, e] gphi[c, q, i, b] gphi[c, q, i, e]

        按分量 d 逐个算, 中间量始终不超过 gphi 本身的规模, 循环最多 3 次.

        Parameters
        ----------
        basis_gradients : (NC, NQ, ldof, GD) 的物理基函数梯度.

        Returns
        -------
        diag : (NC, ldof, GD) 的单元矩阵对角.
        """
        blocks = []
        for d in range(self._geo_dim):
            projected = bm.einsum('cqib, be -> cqie',
                                basis_gradients, self._quadratic_form[d])
            blocks.append(bm.einsum('cq, cqie, cqie -> ci',
                                self._scale, projected, basis_gradients))

        return bm.stack(blocks, axis=-1)

    def persistent_bytes(self) -> int:
        """常驻内存字节数: 逐点标量因子是主项, 本构矩阵与应变映射是 O(1)"""
        total = int(self._scale.nbytes) + int(self._D.nbytes)
        total += int(self._strain_map.nbytes) + int(self._quadratic_form.nbytes)

        return total

    def __repr__(self) -> str:
        n_cells, n_quad = self._scale.shape

        return (f"LinearElasticQFunction(n_cells={n_cells}, n_quad={n_quad}, "
                f"geo_dim={self._geo_dim})")

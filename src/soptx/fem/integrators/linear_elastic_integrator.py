
from typing import NamedTuple, Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S
from fealpy.mesh import HomogeneousMesh, SimplexMesh, StructuredMesh, TensorMesh
from fealpy.functionspace.space import FunctionSpace, Function
from fealpy.functionspace.tensor_space import TensorFunctionSpace
from fealpy.decorator.variantmethod import variantmethod
from fealpy.fem.integrator import (LinearInt, OpInt, CellInt, enable_cache)

from ...materials import LinearElasticMaterial
from soptx.fem.integrators.utils import LinearSymbolicIntegration

from soptx.core import timer


def cell_jacobi_det(mesh: HomogeneousMesh, bcs: TensorLike, index: Index = _S) -> Optional[TensorLike]:
    """积分点上的 Jacobi 行列式绝对值 |det J|, 形状 (NC, NQ); 单纯形网格返回 None.

    单纯形上仿射映射的 |det J| 是常数, 已并入 ``cell_measure``, 各条装配路径按
    ``detJ is None`` 走单纯形分支; 张量网格 (四边形 / 六面体) 上 J 随积分点变化,
    须逐点取行列式. 本函数把原先散在各 ``fetch_*`` 变体里的这段分支收成一处.
    """
    if isinstance(mesh, SimplexMesh):
        return None
    J = mesh.entity_view("cell").jacobi_matrix(bcs, index=index)
    return bm.abs(bm.linalg.det(J))


class IntegrationContext(NamedTuple):
    """单元积分所需的共用上下文: 校验过的句柄加求积数据.

    Attributes
    ----------
    scalar_space : 张量空间背后的标量空间.
    mesh : 空间所在的齐次网格.
    index : 参与装配的单元子集.
    q : 实际使用的积分阶.
    bcs : 积分点的重心坐标.
    ws : 积分权重, 形状 (NQ, ).
    cell_measure : 单元测度, 形状 (NC, ).
    """

    scalar_space: FunctionSpace
    mesh: HomogeneousMesh
    index: Index
    q: int
    bcs: TensorLike
    ws: TensorLike
    cell_measure: TensorLike


class LinearElasticIntegrator(LinearInt, OpInt, CellInt):
    """The linear elastic integrator for function spaces based on homogeneous meshes."""
    def __init__(self, 
                material: LinearElasticMaterial,
                coef: Optional[TensorLike]=None,
                q: Optional[int]=None, 
                *,
                index: Index=_S,
                method: Optional[str]=None
            ) -> None:
        super().__init__()

        self._material = material
        self._coef = coef
        self._q = q
        self._index = index
        
        # 设置默认组装方法
        self.assembly.set(method)

    @property
    def coef(self) -> Optional[TensorLike]:
        """获取当前的材料密度系数"""
        return self._coef

    @coef.setter
    def coef(self, value: Optional[TensorLike]):
        """设置材料密度系数"""
        self._coef = value

    @property
    def material(self) -> LinearElasticMaterial:
        """本积分子使用的材料, 只读"""
        return self._material

    @property
    def q(self) -> Optional[int]:
        """外部指定的积分阶; 为 None 时由 fetch_context 取 p + 3"""
        return self._q

    @property
    def index(self) -> Index:
        """参与装配的单元子集, 默认 _S 表示全体"""
        return self._index

    @enable_cache
    def to_global_dof(self, space: FunctionSpace) -> TensorLike:
        return space.cell_to_dof()[self._index]
    
    ########################################################################################
    # 变体方法
    ########################################################################################

    def quadrature_order(self, space: TensorFunctionSpace) -> int:
        """本积分子在该空间上实际使用的积分阶.

        Parameters
        ----------
        space : 该双线性型所在的张量函数空间.

        Returns
        -------
        q : 积分阶, 构造时未指定则取默认的 ``p + 3``.

        Notes
        -----
        单独开这个方法, 是为了让只需要积分阶的调用方 (矩阵自由层级的 build 阶段,
        它只在参考单元上取积分点) 不必走 ``fetch_context``: 后者附带算一次 O(NC)
        的单元测度, 而 build 阶段的代价必须与网格规模无关.
        """
        return space.scalar_space.p + 3 if self._q is None else self._q

    def fetch_context(self, space: TensorFunctionSpace) -> IntegrationContext:
        """取积分公式、单元测度与常用句柄, 供各条装配路径共用.

        默认积分阶 ``q = p + 3`` 与单元测度的取法只在这里写一次: 各 ``fetch_*``
        变体都从这里取, 于是矩阵自由层级与显式装配用的是字面上同一个积分公式, 改一
        处即可, 不会出现某条路径悄悄换了积分阶而其余路径没跟上.

        Parameters
        ----------
        space : 该双线性型所在的张量函数空间.

        Returns
        -------
        IntegrationContext

        Raises
        ------
        RuntimeError
            空间所在的网格不是齐次网格.

        Notes
        -----
        本方法刻意不加 ``enable_cache``: 它只做 O(NC) 的轻量取数, 而缓存会把 ``bcs``
        / ``ws`` / ``cell_measure`` 钉在积分子上常驻, 改变 PA 等矩阵自由层级的常驻
        内存口径. 需要缓存的是各 ``fetch_*`` 变体的最终结果, 那一层已经加了.
        """
        index = self._index
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The LinearElasticIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        q = self.quadrature_order(space)
        qf = mesh.quadrature_formula(q)
        bcs, ws = qf.get_quadrature_points_and_weights()
        cell_measure = mesh.entity_measure('cell', index=index)

        return IntegrationContext(scalar_space=scalar_space, mesh=mesh,
                            index=index, q=q, bcs=bcs, ws=ws,
                            cell_measure=cell_measure)

    @enable_cache
    def fetch_assembly(self, space: TensorFunctionSpace):
        ctx = self.fetch_context(space)
        scalar_space, mesh, index = ctx.scalar_space, ctx.mesh, ctx.index
        bcs, ws, cm = ctx.bcs, ctx.ws, ctx.cell_measure

        gphi = scalar_space.grad_basis(bcs, index=index, variable='x')
        detJ = cell_jacobi_det(mesh, bcs, index)

        return cm, bcs, ws, gphi, detJ

    @variantmethod('standard')
    def assembly(self, 
                space: TensorFunctionSpace, 
                enable_timing: bool = False
            ) -> TensorLike:
        t = None
        if enable_timing:
            t = timer(f"矩阵组装")
            next(t)

        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)
        cm, bcs, ws, gphi, detJ = self.fetch_assembly(space)

        if enable_timing:
            t.send('缓存部分')

        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        NQ = len(ws)
        D0 = self._material.elastic_matrix()  # (1, 1, NS, NS)
        NS = D0.shape[-1]

        # coef 的四种约定:
        #   None          不考虑相对密度, D 为实体本构矩阵
        #   (NC, )        相对单元密度, D_e = coef_e * D0
        #   (NC, NS, NS)  逐单元本构矩阵, 由调用方直接给出: 泊松比随密度变化时
        #                 D_e 不再是 D0 的标量倍 (见 LagrangeFEMAnalyzer)
        #   (NC, NQ)      相对节点密度, D 在积分点上变化
        # 下面的组装公式只读 D 的 (0,0)/(0,1)/(2,2) 或 (5,5) 元, 即假定各向同性结构
        coef = self._coef

        if coef is None:
            D_mode = 'constant'
            D = D0[0, 0] # (NS, NS)
        elif coef.shape == (NC, ):
            D_mode = 'cell'
            D = bm.einsum('c, kl -> ckl', coef, D0[0, 0])  # (NC, NS, NS)
        elif coef.shape == (NC, NS, NS):
            D_mode = 'cell'
            D = coef                                       # (NC, NS, NS)
        elif coef.shape == (NC, NQ):
            D_mode = 'quadrature'
            D = bm.einsum('cq, cqkl -> cqkl', coef, D0)    # (NC, NQ, NS, NS)
        else:
            raise ValueError(
                f"coef 形状 {tuple(coef.shape)} 不在支持的约定内: None, "
                f"(NC,)={(NC,)}, (NC, NS, NS)={(NC, NS, NS)}, (NC, NQ)={(NC, NQ)}"
            )

        if isinstance(mesh, SimplexMesh):
            A_xx = bm.einsum('q, cqi, cqj, c -> cqij', ws, gphi[..., 0], gphi[..., 0], cm)
            A_yy = bm.einsum('q, cqi, cqj, c -> cqij', ws, gphi[..., 1], gphi[..., 1], cm)
            A_xy = bm.einsum('q, cqi, cqj, c -> cqij', ws, gphi[..., 0], gphi[..., 1], cm)
            A_yx = bm.einsum('q, cqi, cqj, c -> cqij', ws, gphi[..., 1], gphi[..., 0], cm)
        else:
            A_xx = bm.einsum('q, cqi, cqj, cq -> cqij', ws, gphi[..., 0], gphi[..., 0], detJ)
            A_yy = bm.einsum('q, cqi, cqj, cq -> cqij', ws, gphi[..., 1], gphi[..., 1], detJ)
            A_xy = bm.einsum('q, cqi, cqj, cq -> cqij', ws, gphi[..., 0], gphi[..., 1], detJ)
            A_yx = bm.einsum('q, cqi, cqj, cq -> cqij', ws, gphi[..., 1], gphi[..., 0], detJ)

        GD = mesh.geo_dimension()
        if GD == 3:
            if isinstance(mesh, SimplexMesh):
                A_xz = bm.einsum('q, cqi, cqj, c -> cqij', ws, gphi[..., 0], gphi[..., 2], cm)
                A_zx = bm.einsum('q, cqi, cqj, c -> cqij', ws, gphi[..., 2], gphi[..., 0], cm)
                A_yz = bm.einsum('q, cqi, cqj, c -> cqij', ws, gphi[..., 1], gphi[..., 2], cm)
                A_zy = bm.einsum('q, cqi, cqj, c -> cqij', ws, gphi[..., 2], gphi[..., 1], cm)
                A_zz = bm.einsum('q, cqi, cqj, c -> cqij', ws, gphi[..., 2], gphi[..., 2], cm)
            else:
                A_xz = bm.einsum('q, cqi, cqj, cq -> cqij', ws, gphi[..., 0], gphi[..., 2], detJ)
                A_zx = bm.einsum('q, cqi, cqj, cq -> cqij', ws, gphi[..., 2], gphi[..., 0], detJ)
                A_yz = bm.einsum('q, cqi, cqj, cq -> cqij', ws, gphi[..., 1], gphi[..., 2], detJ)
                A_zy = bm.einsum('q, cqi, cqj, cq -> cqij', ws, gphi[..., 2], gphi[..., 1], detJ)
                A_zz = bm.einsum('q, cqi, cqj, cq -> cqij', ws, gphi[..., 2], gphi[..., 2], detJ)

        ldof = scalar_space.number_of_local_dofs()
        KK = bm.zeros((NC, GD * ldof, GD * ldof), dtype=bm.float64, device=mesh.device)

        # 区域内的相对密度恒定都为 1, D 为全局常数矩阵
        if D_mode == 'constant':
            if GD == 2:
                D00 = D[0, 0] # 2D: E/(1-ν²) 或 2μ+λ
                D01 = D[0, 1] # 2D: νE/(1-ν²) 或 λ
                D22 = D[2, 2] # 2D: E/2(1+ν) 或 μ
                KK_11 = D00 * bm.einsum('cqij -> cij', A_xx) + D22 * bm.einsum('cqij -> cij', A_yy)
                KK_22 = D00 * bm.einsum('cqij -> cij', A_yy) + D22 * bm.einsum('cqij -> cij', A_xx)
                KK_12 = D01 * bm.einsum('cqij -> cij', A_xy) + D22 * bm.einsum('cqij -> cij', A_yx)
                KK_21 = D01 * bm.einsum('cqij -> cij', A_yx) + D22 * bm.einsum('cqij -> cij', A_xy)
            else: 
                D00 = D[0, 0]  # 2μ + λ
                D01 = D[0, 1]  # λ
                D55 = D[5, 5]  # μ
                KK_11 = D00 * bm.einsum('cqij -> cij', A_xx) + D55 * bm.einsum('cqij -> cij', A_yy + A_zz)
                KK_22 = D00 * bm.einsum('cqij -> cij', A_yy) + D55 * bm.einsum('cqij -> cij', A_xx + A_zz)
                KK_33 = D00 * bm.einsum('cqij -> cij', A_zz) + D55 * bm.einsum('cqij -> cij', A_xx + A_yy)
                KK_12 = D01 * bm.einsum('cqij -> cij', A_xy) + D55 * bm.einsum('cqij -> cij', A_yx)
                KK_13 = D01 * bm.einsum('cqij -> cij', A_xz) + D55 * bm.einsum('cqij -> cij', A_zx)
                KK_21 = D01 * bm.einsum('cqij -> cij', A_yx) + D55 * bm.einsum('cqij -> cij', A_xy)
                KK_23 = D01 * bm.einsum('cqij -> cij', A_yz) + D55 * bm.einsum('cqij -> cij', A_zy)
                KK_31 = D01 * bm.einsum('cqij -> cij', A_zx) + D55 * bm.einsum('cqij -> cij', A_xz)
                KK_32 = D01 * bm.einsum('cqij -> cij', A_zy) + D55 * bm.einsum('cqij -> cij', A_yz)
        # 单元密度情况 (含逐单元本构矩阵), D 为单元均匀矩阵
        elif D_mode == 'cell':
            if GD == 2:
                D00 = D[:, 0, 0] # 2D: E/(1-ν²) 或 2μ+λ
                D01 = D[:, 0, 1] # 2D: νE/(1-ν²) 或 λ
                D22 = D[:, 2, 2] # 2D: E/2(1+ν) 或 μ
                KK_11 = bm.einsum('c, cqij -> cij', D00, A_xx) + bm.einsum('c, cqij -> cij', D22, A_yy)
                KK_22 = bm.einsum('c, cqij -> cij', D00, A_yy) + bm.einsum('c, cqij -> cij', D22, A_xx)
                KK_12 = bm.einsum('c, cqij -> cij', D01, A_xy) + bm.einsum('c, cqij -> cij', D22, A_yx)
                KK_21 = bm.einsum('c, cqij -> cij', D01, A_yx) + bm.einsum('c, cqij -> cij', D22, A_xy)
            else:
                D00 = D[:, 0, 0] # 2μ + λ
                D01 = D[:, 0, 1] # λ
                D55 = D[:, 5, 5] # μ
                KK_11 = bm.einsum('c, cqij -> cij', D00, A_xx) + bm.einsum('c, cqij -> cij', D55, A_yy + A_zz)
                KK_22 = bm.einsum('c, cqij -> cij', D00, A_yy) + bm.einsum('c, cqij -> cij', D55, A_xx + A_zz)
                KK_33 = bm.einsum('c, cqij -> cij', D00, A_zz) + bm.einsum('c, cqij -> cij', D55, A_xx + A_yy)
                KK_12 = bm.einsum('c, cqij -> cij', D01, A_xy) + bm.einsum('c, cqij -> cij', D55, A_yx)
                KK_13 = bm.einsum('c, cqij -> cij', D01, A_xz) + bm.einsum('c, cqij -> cij', D55, A_zx)
                KK_21 = bm.einsum('c, cqij -> cij', D01, A_yx) + bm.einsum('c, cqij -> cij', D55, A_xy)
                KK_23 = bm.einsum('c, cqij -> cij', D01, A_yz) + bm.einsum('c, cqij -> cij', D55, A_zy)
                KK_31 = bm.einsum('c, cqij -> cij', D01, A_zx) + bm.einsum('c, cqij -> cij', D55, A_xz)
                KK_32 = bm.einsum('c, cqij -> cij', D01, A_zy) + bm.einsum('c, cqij -> cij', D55, A_yz)
        # 节点密度情况, 区域内的相对密度在单元内变化, D 为节点变化矩阵
        elif D_mode == 'quadrature':
            if GD == 2:
                D00 = D[..., 0, 0] # 2D: E/(1-ν²) 或 2μ+λ
                D01 = D[..., 0, 1] # 2D: νE/(1-ν²) 或 λ
                D22 = D[..., 2, 2] # 2D: E/2(1+ν) 或 μ
                KK_11 = bm.einsum('cq, cqij -> cij', D00, A_xx) + bm.einsum('cq, cqij -> cij', D22, A_yy)
                KK_22 = bm.einsum('cq, cqij -> cij', D00, A_yy) + bm.einsum('cq, cqij -> cij', D22, A_xx)
                KK_12 = bm.einsum('cq, cqij -> cij', D01, A_xy) + bm.einsum('cq, cqij -> cij', D22, A_yx)
                KK_21 = bm.einsum('cq, cqij -> cij', D01, A_yx) + bm.einsum('cq, cqij -> cij', D22, A_xy)
            else:
                D00 = D[..., 0, 0] # 2μ + λ
                D01 = D[..., 0, 1] # λ
                D55 = D[..., 5, 5] # μ
                KK_11 = bm.einsum('cq, cqij -> cij', D00, A_xx) + bm.einsum('cq, cqij -> cij', D55, A_yy + A_zz)
                KK_22 = bm.einsum('cq, cqij -> cij', D00, A_yy) + bm.einsum('cq, cqij -> cij', D55, A_xx + A_zz)
                KK_33 = bm.einsum('cq, cqij -> cij', D00, A_zz) + bm.einsum('cq, cqij -> cij', D55, A_xx + A_yy)
                KK_12 = bm.einsum('cq, cqij -> cij', D01, A_xy) + bm.einsum('cq, cqij -> cij', D55, A_yx)
                KK_13 = bm.einsum('cq, cqij -> cij', D01, A_xz) + bm.einsum('cq, cqij -> cij', D55, A_zx)
                KK_21 = bm.einsum('cq, cqij -> cij', D01, A_yx) + bm.einsum('cq, cqij -> cij', D55, A_xy)
                KK_23 = bm.einsum('cq, cqij -> cij', D01, A_yz) + bm.einsum('cq, cqij -> cij', D55, A_zy)
                KK_31 = bm.einsum('cq, cqij -> cij', D01, A_zx) + bm.einsum('cq, cqij -> cij', D55, A_xz)
                KK_32 = bm.einsum('cq, cqij -> cij', D01, A_zy) + bm.einsum('cq, cqij -> cij', D55, A_yz)

        if GD == 2:
            if space.dof_priority:
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(ldof, None), slice(ldof, None)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, None)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(ldof, None), slice(0, ldof)), KK_21)
            else:
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_21)
        else: 
            if space.dof_priority:
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(ldof, 2 * ldof)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(2 * ldof, None)), KK_33)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, 2 * ldof)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(2 * ldof, None)), KK_13)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(0, ldof)), KK_21)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(2 * ldof, None)), KK_23)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(0, ldof)), KK_31)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(ldof, 2 * ldof)), KK_32)
            else:
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(2, KK.shape[2], GD)), KK_33)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(2, KK.shape[2], GD)), KK_13)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_21)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(2, KK.shape[2], GD)), KK_23)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_31)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_32)

        if enable_timing:
            t.send('组装部分')
            t.send(None)

        return KK
    
    @assembly.register('standard_multiresolution')
    def assembly(self, space: TensorFunctionSpace) -> TensorLike:
        index = self._index
        mesh_u = getattr(space, 'mesh', None)
        s_space_u = space.scalar_space
        GD = mesh_u.geo_dimension()
        # TODO 原高阶 q 作为 fallback 保留
        default_q = s_space_u.p+3 if self._q is None else self._q
    
        # 单元密度多分辨率: (NC, n_sub); 节点密度多分辨率: (NC, n_sub, NQ)
        coef = self._coef
        NC, n_sub = coef.shape[0], coef.shape[1]

        if 4 <= n_sub <= 9:
            q = 4
        elif n_sub >= 16:
            # TODO 2 不对, 3 对, 5 对
            q = 3
        else:
            q = default_q
        
        # 计算位移单元积分点处的重心坐标
        qf_e = mesh_u.quadrature_formula(q)
        # bcs_e.shape = ( (NQ_x, GD), (NQ_y, GD) ), ws_e.shape = (NQ, )
        bcs_e, ws_e = qf_e.get_quadrature_points_and_weights()
        NQ = ws_e.shape[0]

        # 把位移单元高斯积分点处的重心坐标映射到子密度单元 (子参考单元) 高斯积分点处的重心坐标 (仍表达在位移单元中)
        from soptx.fem.utils import map_bcs_to_sub_elements
        # bcs_eg.shape = ( (n_sub, NQ_x, GD), (n_sub, NQ_y, GD) ), ws_e.shape = (NQ, )
        bcs_eg = map_bcs_to_sub_elements(bcs_e=bcs_e, n_sub=n_sub)
        bcs_eg_x, bcs_eg_y = bcs_eg[0], bcs_eg[1]

        # 计算子密度单元内高斯积分点处的基函数梯度和 jacobi 矩阵
        LDOF = s_space_u.number_of_local_dofs()
        gphi_eg = bm.zeros((NC, n_sub, NQ, LDOF, GD)) # (NC, n_sub, NQ, LDOF, GD)
        detJ_eg = None

        if isinstance(mesh_u, SimplexMesh):
            cm = mesh_u.entity_measure('cell')
            cm_eg = bm.tile(cm.reshape(NC, 1), (1, n_sub)) # (NC, n_sub)
            
            for s_idx in range(n_sub):
                sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])  # ((NQ_x, GD), (NQ_y, GD))
                gphi_sub = s_space_u.grad_basis(sub_bcs, index=index, variable='x')  # (NC, NQ, LDOF, GD)
                gphi_eg[:, s_idx, :, :, :] = gphi_sub

        else:
            detJ_eg = bm.zeros((NC, n_sub, NQ)) # (NC, n_sub, NQ)
            for s_idx in range(n_sub):
                sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])  # ((NQ_x, GD), (NQ_y, GD))
                gphi_sub = s_space_u.grad_basis(sub_bcs, index=index, variable='x') # (NC, NQ, LDOF, GD)

                J_sub = mesh_u.entity_view('cell').jacobi_matrix(sub_bcs) # (NC, NQ, GD, GD)
                detJ_sub = bm.abs(bm.linalg.det(J_sub)) # (NC, NQ)

                gphi_eg[:, s_idx, :, :, :] = gphi_sub
                detJ_eg[:, s_idx, :] = detJ_sub

        if isinstance(mesh_u, SimplexMesh):
            A_xx_eg = bm.einsum('q, cnqi, cnqj, cn -> cnqij', ws_e, gphi_eg[..., 0], gphi_eg[..., 0], cm_eg)
            A_yy_eg = bm.einsum('q, cnqi, cnqj, cn -> cnqij', ws_e, gphi_eg[..., 1], gphi_eg[..., 1], cm_eg)
            A_xy_eg = bm.einsum('q, cnqi, cnqj, cn -> cnqij', ws_e, gphi_eg[..., 0], gphi_eg[..., 1], cm_eg)
            A_yx_eg = bm.einsum('q, cnqi, cnqj, cn -> cnqij', ws_e, gphi_eg[..., 1], gphi_eg[..., 0], cm_eg)
        else:
            A_xx_eg = bm.einsum('q, cnqi, cnqj, cnq -> cnqij', ws_e, gphi_eg[..., 0], gphi_eg[..., 0], detJ_eg)
            A_yy_eg = bm.einsum('q, cnqi, cnqj, cnq -> cnqij', ws_e, gphi_eg[..., 1], gphi_eg[..., 1], detJ_eg)
            A_xy_eg = bm.einsum('q, cnqi, cnqj, cnq -> cnqij', ws_e, gphi_eg[..., 0], gphi_eg[..., 1], detJ_eg)
            A_yx_eg = bm.einsum('q, cnqi, cnqj, cnq -> cnqij', ws_e, gphi_eg[..., 1], gphi_eg[..., 0], detJ_eg)

        if GD == 3:
            if isinstance(mesh_u, SimplexMesh):
                A_xz_eg = bm.einsum('q, cnqi, cnqj, cn -> cnqij', ws_e, gphi_eg[..., 0], gphi_eg[..., 2], cm_eg)
                A_zx_eg = bm.einsum('q, cnqi, cnqj, cn -> cnqij', ws_e, gphi_eg[..., 2], gphi_eg[..., 0], cm_eg)
                A_yz_eg = bm.einsum('q, cnqi, cnqj, cn -> cnqij', ws_e, gphi_eg[..., 1], gphi_eg[..., 2], cm_eg)
                A_zy_eg = bm.einsum('q, cnqi, cnqj, cn -> cnqij', ws_e, gphi_eg[..., 2], gphi_eg[..., 1], cm_eg)
                A_zz_eg = bm.einsum('q, cnqi, cnqj, cn -> cnqij', ws_e, gphi_eg[..., 2], gphi_eg[..., 2], cm_eg)
            else:
                A_xz_eg = bm.einsum('q, cnqi, cnqj, cnq -> cnqij', ws_e, gphi_eg[..., 0], gphi_eg[..., 2], detJ_eg)
                A_zx_eg = bm.einsum('q, cnqi, cnqj, cnq -> cnqij', ws_e, gphi_eg[..., 2], gphi_eg[..., 0], detJ_eg)
                A_yz_eg = bm.einsum('q, cnqi, cnqj, cnq -> cnqij', ws_e, gphi_eg[..., 1], gphi_eg[..., 2], detJ_eg)
                A_zy_eg = bm.einsum('q, cnqi, cnqj, cnq -> cnqij', ws_e, gphi_eg[..., 2], gphi_eg[..., 1], detJ_eg)
                A_zz_eg = bm.einsum('q, cnqi, cnqj, cnq -> cnqij', ws_e, gphi_eg[..., 2], gphi_eg[..., 2], detJ_eg)

        # 位移单元 → 子密度单元的缩放
        J_g = 1 / n_sub

        # 基础材料的弹性矩阵
        D0 = self._material.elastic_matrix()[0, 0] # 2D: (3, 3); 3D: (6, 6)

        ldof = s_space_u.number_of_local_dofs()
        KK = bm.zeros((NC, GD * ldof, GD * ldof), dtype=bm.float64, device=mesh_u.device)

        # 区域内的相对密度恒定都为 1, D 为全局常数矩阵
        if coef is None:
            raise NotImplementedError("The global uniform density case is not implemented yet.")

        # 单元密度情况
        elif coef.shape == (NC, n_sub):
            if GD == 2:
                D00 = D0[0, 0] # 2D: E/(1-ν²) 或 2μ+λ
                D01 = D0[0, 1] # 2D: νE/(1-ν²) 或 λ
                D22 = D0[2, 2] # 2D: E/2(1+ν) 或 μ
                KK_11 = J_g * bm.einsum('cn, cnqij -> cij', coef * D00, A_xx_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D22, A_yy_eg)
                KK_22 = J_g * bm.einsum('cn, cnqij -> cij', coef * D00, A_yy_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D22, A_xx_eg)
                KK_12 = J_g * bm.einsum('cn, cnqij -> cij', coef * D01, A_xy_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D22, A_yx_eg)
                KK_21 = J_g * bm.einsum('cn, cnqij -> cij', coef * D01, A_yx_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D22, A_xy_eg)
            else: 
                D00 = D0[0, 0] # 2μ + λ
                D01 = D0[0, 1] # λ
                D55 = D0[5, 5] # μ
                KK_11 = J_g * bm.einsum('cn, cnqij -> cij', coef * D00, A_xx_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D55, A_yy_eg + A_zz_eg)
                KK_22 = J_g * bm.einsum('cn, cnqij -> cij', coef * D00, A_yy_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D55, A_xx_eg + A_zz_eg)
                KK_33 = J_g * bm.einsum('cn, cnqij -> cij', coef * D00, A_zz_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D55, A_xx_eg + A_yy_eg)
                KK_12 = J_g * bm.einsum('cn, cnqij -> cij', coef * D01, A_xy_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D55, A_yx_eg)
                KK_13 = J_g * bm.einsum('cn, cnqij -> cij', coef * D01, A_xz_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D55, A_zx_eg)
                KK_21 = J_g * bm.einsum('cn, cnqij -> cij', coef * D01, A_yx_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D55, A_xy_eg)
                KK_23 = J_g * bm.einsum('cn, cnqij -> cij', coef * D01, A_yz_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D55, A_zy_eg)
                KK_31 = J_g * bm.einsum('cn, cnqij -> cij', coef * D01, A_zx_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D55, A_xz_eg)
                KK_32 = J_g * bm.einsum('cn, cnqij -> cij', coef * D01, A_zy_eg) + \
                        J_g * bm.einsum('cn, cnqij -> cij', coef * D55, A_yz_eg)
                    
        # 节点密度情况
        elif coef.shape == (NC, n_sub, NQ):
            if GD == 2:
                D00, D01, D22 = D0[0, 0], D0[0, 1], D0[2, 2]
                KK_11 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D00, A_xx_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D22, A_yy_eg)
                KK_22 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D00, A_yy_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D22, A_xx_eg)
                KK_12 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D01, A_xy_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D22, A_yx_eg)
                KK_21 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D01, A_yx_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D22, A_xy_eg)
            else: 
                D00, D01, D55 = D0[0, 0], D0[0, 1], D0[5, 5]
                KK_11 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D00, A_xx_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D55, A_yy_eg + A_zz_eg)
                KK_22 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D00, A_yy_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D55, A_xx_eg + A_zz_eg)
                KK_33 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D00, A_zz_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D55, A_xx_eg + A_yy_eg)
                KK_12 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D01, A_xy_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D55, A_yx_eg)
                KK_13 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D01, A_xz_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D55, A_zx_eg)
                KK_21 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D01, A_yx_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D55, A_xy_eg)
                KK_23 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D01, A_yz_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D55, A_zy_eg)
                KK_31 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D01, A_zx_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D55, A_xz_eg)
                KK_32 = J_g * bm.einsum('cnq, cnqij -> cij', coef * D01, A_zy_eg) + \
                        J_g * bm.einsum('cnq, cnqij -> cij', coef * D55, A_yz_eg)

        if GD == 2:
            if space.dof_priority:
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(ldof, None), slice(ldof, None)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, None)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(ldof, None), slice(0, ldof)), KK_21)
            else:
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_21)
        else:  
            if space.dof_priority:
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(ldof, 2 * ldof)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(2 * ldof, None)), KK_33)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, 2 * ldof)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(2 * ldof, None)), KK_13)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(0, ldof)), KK_21)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(2 * ldof, None)), KK_23)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(0, ldof)), KK_31)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(ldof, 2 * ldof)), KK_32)
            else:
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(2, KK.shape[2], GD)), KK_33)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(2, KK.shape[2], GD)), KK_13)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_21)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(2, KK.shape[2], GD)), KK_23)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_31)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_32)

        return KK

    @enable_cache
    def fetch_voigt_assembly(self, space: TensorFunctionSpace):
        ctx = self.fetch_context(space)
        scalar_space, mesh, index = ctx.scalar_space, ctx.mesh, ctx.index
        bcs, ws, cm = ctx.bcs, ctx.ws, ctx.cell_measure

        gphi = scalar_space.grad_basis(bcs, index=index, variable='x')
        detJ = cell_jacobi_det(mesh, bcs, index)

        return cm, ws, bcs, gphi, detJ

    @assembly.register('voigt')
    def assembly(self, space: TensorFunctionSpace) -> TensorLike:
        mesh = getattr(space, 'mesh', None)
        cm, ws, bcs, gphi, detJ = self.fetch_voigt_assembly(space)

        NC = mesh.number_of_cells()
        NQ = gphi.shape[1]
        D0 = self._material.elastic_matrix() # 2D: (1, 1, NS, NS)
        B = self._material.strain_matrix(dof_priority=space.dof_priority,
                                                    gphi=gphi) # (NC, NQ, NS, LDOF)

        # 单元密度: (NC, ); 节点密度: (NC, NQ)
        coef = self._coef

        if coef is None:
            D = D0[0, 0] # (NS, NS)

            if isinstance(mesh, SimplexMesh):
                KK = bm.einsum('q, c, cqki, kl, cqlj -> cij', ws, cm, B, D, B)
            else:
                KK = bm.einsum('q, cq, cqki, kl, cqlj -> cij', ws, detJ, B, D, B)
        
        # 单元密度的情况
        elif coef.shape == (NC, ):
            D_base = D0[0, 0] # (NS, NS)
            D = bm.einsum('c, kl -> ckl', coef, D_base) # (NC, NS, NS)
            
            if isinstance(mesh, SimplexMesh):
                KK = bm.einsum('q, c, cqki, ckl, cqlj -> cij', ws, cm, B, D, B)
            else:
                KK = bm.einsum('q, cq, cqki, ckl, cqlj -> cij', ws, detJ, B, D, B)
                    
        # 节点密度的情况
        elif coef.shape == (NC, NQ):
            D = bm.einsum('cq, ijkl -> cqkl', coef, D0) # (NC, NQ, NS, NS)
            
            if isinstance(mesh, SimplexMesh):
                KK = bm.einsum('q, c, cqki, cqkl, cqlj -> cij', ws, cm, B, D, B)
            else:
                KK = bm.einsum('q, cq, cqki, cqkl, cqlj -> cij', ws, detJ, B, D, B)
        
        else:
            raise NotImplementedError

        return KK

    @assembly.register('voigt_multiresolution')
    def assembly(self, space: TensorFunctionSpace) -> TensorLike:
        index = self._index
        mesh_u = getattr(space, 'mesh', None)
        s_space_u = space.scalar_space
        GD = mesh_u.geo_dimension()
        # TODO 原高阶 q 作为 fallback 保留
        default_q = s_space_u.p+3 if self._q is None else self._q
       
        # 单元密度多分辨率: (NC, n_sub); 节点密度多分辨率: (NC, n_sub, NQ)
        coef = self._coef
        NC, n_sub = coef.shape[0], coef.shape[1]

        if 4 <= n_sub <= 9:
            q = 3
        elif n_sub >= 16:
            q = 2
        else:
            q = default_q
        
        # 计算位移单元积分点处的重心坐标
        qf_e = mesh_u.quadrature_formula(q)
        # bcs_e.shape = ( (NQ_x, GD), (NQ_y, GD) ), ws_e.shape = (NQ, )
        bcs_e, ws_e = qf_e.get_quadrature_points_and_weights()
        NQ = ws_e.shape[0]

        # 把位移单元高斯积分点处的重心坐标映射到子密度单元 (子参考单元) 高斯积分点处的重心坐标 (仍表达在位移单元中)
        from soptx.fem.utils import map_bcs_to_sub_elements
        # bcs_eg.shape = ( (n_sub, NQ_x, GD), (n_sub, NQ_y, GD) ), ws_e.shape = (NQ, )
        bcs_eg = map_bcs_to_sub_elements(bcs_e=bcs_e, n_sub=n_sub)
        bcs_eg_x, bcs_eg_y = bcs_eg[0], bcs_eg[1]

        # 计算子密度单元内高斯积分点处的基函数梯度和 jacobi 矩阵
        LDOF = s_space_u.number_of_local_dofs()
        gphi_eg = bm.zeros((NC, n_sub, NQ, LDOF, GD)) # (NC, n_sub, NQ, LDOF, GD)
        detJ_eg = None

        if isinstance(mesh_u, SimplexMesh):
            for s_idx in range(n_sub):
                sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])  # ((NQ_x, GD), (NQ_y, GD))
                gphi_sub = s_space_u.grad_basis(sub_bcs, index=index, variable='x')  # (NC, NQ, LDOF, GD)
                gphi_eg[:, s_idx, :, :, :] = gphi_sub

        else:
            detJ_eg = bm.zeros((NC, n_sub, NQ)) # (NC, n_sub, NQ)
            for s_idx in range(n_sub):
                sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])  # ((NQ_x, GD), (NQ_y, GD))
                gphi_sub = s_space_u.grad_basis(sub_bcs, index=index, variable='x') # (NC, NQ, LDOF, GD)

                J_sub = mesh_u.entity_view('cell').jacobi_matrix(sub_bcs) # (NC, NQ, GD, GD)
                detJ_sub = bm.abs(bm.linalg.det(J_sub)) # (NC, NQ)

                gphi_eg[:, s_idx, :, :, :] = gphi_sub
                detJ_eg[:, s_idx, :] = detJ_sub

        # 计算 B 矩阵
        from soptx.fem.utils import reshape_multiresolution_data, reshape_multiresolution_data_inverse
        gphi_eg_reshaped = reshape_multiresolution_data(mesh=mesh_u, data=gphi_eg) # (NC*n_sub, NQ, NS, TLDOF)
        B_eg_reshaped = self._material.strain_matrix(
                                            dof_priority=space.dof_priority, 
                                            gphi=gphi_eg_reshaped
                                        ) # (NC*n_sub, NQ, NS, TLDOF)
        B_eg = reshape_multiresolution_data_inverse(mesh=mesh_u, data_flat=B_eg_reshaped, n_sub=n_sub) # (NC, n_sub, NQ, NS, TLDOF)

        # 位移单元 → 子密度单元的缩放
        J_g = 1 / n_sub

        # 基础材料的弹性矩阵
        D0 = self._material.elastic_matrix()[0, 0] # (NS, NS)

        if coef is None:
            raise NotImplementedError("The global constant density is not implemented"
                                      " in the multiresolution assembly.")

        # 单元密度
        if coef.shape == (NC, n_sub):

            D_g = bm.einsum('kl, cn -> cnkl', D0, coef) # (NC, n_sub, NS, NS)
            if isinstance(mesh_u, SimplexMesh):
                cm = mesh_u.entity_measure('cell')
                cm_eg = bm.tile(cm.reshape(NC, 1), (1, n_sub)) # (NC, n_sub)
                KK = J_g * bm.einsum('q, cn, cnqki, cnkl, cnqlj -> cij',
                                    ws_e, cm_eg, B_eg, D_g, B_eg)
            else:
                KK = J_g * bm.einsum('q, cnq, cnqki, cnkl, cnqlj -> cij',
                                    ws_e, detJ_eg, B_eg, D_g, B_eg)
                
            return KK

        # 节点密度
        elif coef.shape == (NC, n_sub, NQ):

            D_g = bm.einsum('ijkl, cnq -> cnqkl', D0, coef) # (NC, n_sub, NQ, NS, NS)
            if isinstance(mesh_u, SimplexMesh):
                KK = J_g * bm.einsum('q, cn, cnqki, cnqkl, cnqlj -> cij',
                                    ws_e, cm_eg, B_eg, D_g, B_eg)
            else:
                KK = J_g * bm.einsum('q, cnq, cnqki, cnqkl, cnqlj -> cij',
                                    ws_e, detJ_eg, B_eg, D_g, B_eg)
                
            return KK

    @enable_cache
    def fetch_fast_assembly(self, space: TensorFunctionSpace):
        ctx = self.fetch_context(space)
        scalar_space, mesh, index = ctx.scalar_space, ctx.mesh, ctx.index
        bcs, ws, cm = ctx.bcs, ctx.ws, ctx.cell_measure

        gphi_lambda = scalar_space.grad_basis(bcs, index=index, variable='u')    # (NQ, LDOF, BC)

        if isinstance(mesh, SimplexMesh):
            glambda_x = mesh.grad_lambda()   # (NC, LDOF, GD)
            # 快速装配用的恒等式是 ``grad(phi_i) = sum_k (d phi_i / d lambda_k) grad(lambda_k)``,
            # 因此 ``S`` 的后两轴必须是重心坐标轴 (长度 ``BC = GD + 1``), 与 ``glambda_x``
            # 的重心坐标轴对齐。上面按 ``variable='u'`` 取到的是对物理坐标的导数
            # (末轴长度 ``GD``), 轴长对不上, 收缩会直接报错
            gphi_lambda_b = scalar_space.grad_basis(bcs, index=index, variable='b')  # (NQ, LDOF, BC)
            S = bm.einsum('q, qik, qjl -> ijkl', ws, gphi_lambda_b, gphi_lambda_b)  # (LDOF, LDOF, BC, BC)
            return cm, glambda_x, S
        
        else:
            # 适用于结构/张量积/多边形等非单纯形网格
            cell_view = mesh.entity_view('cell')
            J = cell_view.jacobi_matrix(bcs) # (NC, NQ, GD, GD)
            if not bm.allclose(J[:, 0, ...], J[:, -1, ...]):
                raise ValueError("雅可比矩阵 J 在积分点上不恒定, 无法使用快速组装. 请使用传统组装或检查网格类型")
            J_const = J[:, 0, ...] # (NC, GD, GD)
            invJ = bm.linalg.inv(J_const)
            invJT = invJ.swapaxes(-1, -2) # (NC, GD, GD)
            gphi_u = scalar_space.grad_basis(bcs, index=index, variable='u') # (NQ, LDOF, GD)
            S = bm.einsum('qim, qjn, q -> ijmn', gphi_u, gphi_u, ws) # (LDOF, LDOF, GD, GD)
            return cm, invJT, S

    @assembly.register('fast')
    def assembly(self, 
                space: TensorFunctionSpace,
                enable_timing: bool = False
                ) -> TensorLike:
        t = None
        if enable_timing:
            t = timer(f"矩阵快速组装")
            next(t)
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)
                
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        D0 = self._material.elastic_matrix()  # (1, 1, NS, NS)

        # 不考虑相对密度: None; 相对单元密度: (NC, )
        coef = self._coef
        if coef is None:
            D = D0[0, 0] # (NS, NS)
        elif coef.shape == (NC, ):
            D = bm.einsum('c, kl -> ckl', coef, D0[0, 0])  # (NC, NS, NS)
        else:
            raise NotImplementedError("The fast assembly currently only supports")
        
        if isinstance(mesh, SimplexMesh):
            cm, glambda_x, S = self.fetch_fast_assembly(space)
            if enable_timing:
                t.send('缓存部分')
            A_xx = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 0], glambda_x[..., 0], cm) # (NC, LDOF, LDOF)
            A_yy = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 1], glambda_x[..., 1], cm)
            A_xy = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 0], glambda_x[..., 1], cm)
            A_yx = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 1], glambda_x[..., 0], cm)
            if GD == 3:
                A_zz = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 2], glambda_x[..., 2], cm)
                A_xz = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 0], glambda_x[..., 2], cm)
                A_yz = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 1], glambda_x[..., 2], cm)
                A_zx = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 2], glambda_x[..., 0], cm)
                A_zy = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 2], glambda_x[..., 1], cm)

        else:
            cm, invJT, S = self.fetch_fast_assembly(space)
            if enable_timing:
                t.send('缓存部分')
            A_xx = bm.einsum('ijmn, cm, cn, c -> cij', S, invJT[..., 0, :], invJT[..., 0, :], cm) # (NC, LDOF, LDOF)
            A_yy = bm.einsum('ijmn, cm, cn, c -> cij', S, invJT[..., 1, :], invJT[..., 1, :], cm)
            A_xy = bm.einsum('ijmn, cm, cn, c -> cij', S, invJT[..., 0, :], invJT[..., 1, :], cm)
            A_yx = bm.einsum('ijmn, cm, cn, c -> cij', S, invJT[..., 1, :], invJT[..., 0, :], cm)
            if GD == 3:
                A_zz = bm.einsum('ijmn, cm, cn, c -> cij', S, invJT[..., 2, :], invJT[..., 2, :], cm)
                A_xz = bm.einsum('ijmn, cm, cn, c -> cij', S, invJT[..., 0, :], invJT[..., 2, :], cm)
                A_yz = bm.einsum('ijmn, cm, cn, c -> cij', S, invJT[..., 1, :], invJT[..., 2, :], cm)
                A_zx = bm.einsum('ijmn, cm, cn, c -> cij', S, invJT[..., 2, :], invJT[..., 0, :], cm)
                A_zy = bm.einsum('ijmn, cm, cn, c -> cij', S, invJT[..., 2, :], invJT[..., 1, :], cm)

        ldof = scalar_space.number_of_local_dofs()
        KK = bm.zeros((NC, GD * ldof, GD * ldof), dtype=bm.float64, device=mesh.device)

        # 区域内的相对密度恒定都为 1, D 为全局常数矩阵
        if coef is None:
            if GD == 2:
                D00 = D[0, 0] # 2D: E/(1-ν²) 或 2μ+λ
                D01 = D[0, 1] # 2D: νE/(1-ν²) 或 λ
                D22 = D[2, 2] # 2D: E/2(1+ν) 或 μ
                KK_11 = D00 * A_xx + D22 * A_yy
                KK_22 = D00 * A_yy + D22 * A_xx
                KK_12 = D01 * A_xy + D22 * A_yx
                KK_21 = D01 * A_yx + D22 * A_xy
            else:
                D00 = D[0, 0]  # 2μ + λ
                D01 = D[0, 1]  # λ
                D55 = D[5, 5]  # μ
                KK_11 = D00 * A_xx + D55 * (A_yy + A_zz)
                KK_22 = D00 * A_yy + D55 * (A_xx + A_zz)
                KK_33 = D00 * A_zz + D55 * (A_xx + A_yy)
                KK_12 = D01 * A_xy + D55 * A_yx
                KK_13 = D01 * A_xz + D55 * A_zx
                KK_21 = D01 * A_yx + D55 * A_xy
                KK_23 = D01 * A_yz + D55 * A_zy
                KK_31 = D01 * A_zx + D55 * A_xz
                KK_32 = D01 * A_zy + D55 * A_yz

        # 单元密度情况, D 为单元均匀矩阵
        elif coef.shape == (NC, ):
            if GD == 2:
                D00 = D[:, 0, 0]  # 2D: E/(1-ν²) 或 2μ+λ
                D01 = D[:, 0, 1]  # 2D: νE/(1-ν²) 或 λ
                D22 = D[:, 2, 2]  # 2D: E/2(1+ν) 或 μ
                KK_11 = bm.einsum('c, cij -> cij', D00, A_xx) + bm.einsum('c, cij -> cij', D22, A_yy)
                KK_22 = bm.einsum('c, cij -> cij', D00, A_yy) + bm.einsum('c, cij -> cij', D22, A_xx)
                KK_12 = bm.einsum('c, cij -> cij', D01, A_xy) + bm.einsum('c, cij -> cij', D22, A_yx)
                KK_21 = bm.einsum('c, cij -> cij', D01, A_yx) + bm.einsum('c, cij -> cij', D22, A_xy)
            else:
                D00 = D[:, 0, 0]  # 2μ + λ
                D01 = D[:, 0, 1]  # λ
                D55 = D[:, 5, 5]  # μ
                KK_11 = bm.einsum('c, cij -> cij', D00, A_xx) + bm.einsum('c, cij -> cij', D55, (A_yy + A_zz))
                KK_22 = bm.einsum('c, cij -> cij', D00, A_yy) + bm.einsum('c, cij -> cij', D55, (A_xx + A_zz))
                KK_33 = bm.einsum('c, cij -> cij', D00, A_zz) + bm.einsum('c, cij -> cij', D55, (A_xx + A_yy))
                KK_12 = bm.einsum('c, cij -> cij', D01, A_xy) + bm.einsum('c, cij -> cij', D55, A_yx)
                KK_13 = bm.einsum('c, cij -> cij', D01, A_xz) + bm.einsum('c, cij -> cij', D55, A_zx)
                KK_21 = bm.einsum('c, cij -> cij', D01, A_yx) + bm.einsum('c, cij -> cij', D55, A_xy)
                KK_23 = bm.einsum('c, cij -> cij', D01, A_yz) + bm.einsum('c, cij -> cij', D55, A_zy)
                KK_31 = bm.einsum('c, cij -> cij', D01, A_zx) + bm.einsum('c, cij -> cij', D55, A_xz)
                KK_32 = bm.einsum('c, cij -> cij', D01, A_zy) + bm.einsum('c, cij -> cij', D55, A_yz)

        if GD == 2:
            if space.dof_priority:
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(ldof, None), slice(ldof, None)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, None)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(ldof, None), slice(0, ldof)), KK_21)
            else:
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_21)

        else:
            if space.dof_priority:
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(ldof, 2 * ldof)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(2 * ldof, None)), KK_33)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, 2 * ldof)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(2 * ldof, None)), KK_13)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(0, ldof)), KK_21)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(2 * ldof, None)), KK_23)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(0, ldof)), KK_31)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(ldof, 2 * ldof)), KK_32)
            else:
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(2, KK.shape[2], GD)), KK_33)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(2, KK.shape[2], GD)), KK_13)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_21)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(2, KK.shape[2], GD)), KK_23)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_31)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_32)

        if enable_timing:
            t.send('组装部分')
            t.send(None)

        return KK

    @enable_cache
    def fetch_symbolic_assembly(self, 
                            space: TensorFunctionSpace,
                            enable_timing: bool = False
                        ) -> TensorLike:
        t = None
        if enable_timing:
            t = timer(f"参考单元解析预计算组装(缓存)")
            next(t)

        ctx = self.fetch_context(space)
        scalar_space, mesh = ctx.scalar_space, ctx.mesh
        bcs, cm = ctx.bcs, ctx.cell_measure

        node = mesh.entity('node')
        cell = mesh.entity('cell')
        cell_vertices = node[cell]

        if enable_timing:
            t.send('准备时间')
        
        symbolic_int = LinearSymbolicIntegration(space1=scalar_space, space2=scalar_space)
        kwargs = bm.context(node)

        S = bm.tensor(symbolic_int.gphi_gphi_matrix(), **kwargs)  # (LDOF1, LDOF1, BC, BC)

        if enable_timing:
            t.send('计算 S')

        if isinstance(mesh, SimplexMesh):   
            glambda_x = mesh.grad_lambda()  # (NC, LDOF, GD)
            return cm, bcs, glambda_x, S
        
        elif isinstance(mesh, TensorMesh):
            JG = symbolic_int.compute_mapping(vertices=cell_vertices)  # (NC, GD, GD)

            if enable_timing:
                t.send('计算部分')
                t.send(None)

            return cm, bcs, JG, S
        
    @assembly.register('symbolic')
    def assembly(self, 
                space: TensorFunctionSpace,
                enable_timing: bool = False
            ) -> TensorLike:
        t = None
        if enable_timing:
            t = timer(f"参考单元解析预计算组装")
            next(t)
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)

        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        D0 = self._material.elastic_matrix()  # (1, 1, NS, NS)

        # 不考虑相对密度: None; 相对单元密度: (NC, )
        coef = self._coef
        if coef is None:
            D = D0[0, 0] # (NS, NS)
        elif coef.shape == (NC, ):
            D = bm.einsum('c, kl -> ckl', coef, D0[0, 0])  # (NC, NS, NS)
        else:
            raise NotImplementedError("The fast assembly currently only supports")
        
        if isinstance(mesh, SimplexMesh):
            cm, bcs, glambda_x, S = self.fetch_symbolic_assembly(space)   
            if enable_timing:
                t.send('缓存部分')
            A_xx = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 0], glambda_x[..., 0], cm)
            A_yy = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 1], glambda_x[..., 1], cm)
            A_xy = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 0], glambda_x[..., 1], cm)
            A_yx = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 1], glambda_x[..., 0], cm)
        
        elif isinstance(mesh, TensorMesh):
            if enable_timing:
                t.send('缓存部分')
            cm, bcs, JG, S = self.fetch_symbolic_assembly(space)
            A_xx = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 0], JG[..., 0], cm)
            A_yy = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 1], JG[..., 1], cm)
            A_xy = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 0], JG[..., 1], cm)
            A_yx = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 1], JG[..., 0], cm)   
        
        else:
            raise NotImplementedError("symbolic assembly for general meshes is not implemented yet.")
        
        GD = mesh.geo_dimension()
        if GD == 3:
            if isinstance(mesh, SimplexMesh):
                A_zz = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 2], glambda_x[..., 2], cm)
                A_xz = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 0], glambda_x[..., 2], cm)
                A_yz = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 1], glambda_x[..., 2], cm)
                A_zx = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 2], glambda_x[..., 0], cm)
                A_zy = bm.einsum('ijkl, ck, cl, c -> cij', S, glambda_x[..., 2], glambda_x[..., 1], cm)

            elif isinstance(mesh, TensorMesh):
                A_zz = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 2], JG[..., 2], cm)
                A_xz = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 0], JG[..., 2], cm)
                A_yz = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 1], JG[..., 2], cm)
                A_zx = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 2], JG[..., 0], cm)
                A_zy = bm.einsum('ijmn, cm, cn, c -> cij', S, JG[..., 2], JG[..., 1], cm)

            else:
                raise NotImplementedError("symbolic assembly for general meshes is not implemented yet.")

        ldof = scalar_space.number_of_local_dofs()
        KK = bm.zeros((NC, GD * ldof, GD * ldof), dtype=bm.float64, device=mesh.device)

        # 区域内的相对密度恒定都为 1, D 为全局常数矩阵
        if coef is None:
            if GD == 2:
                D00 = D[0, 0] # 2D: E/(1-ν²) 或 2μ+λ
                D01 = D[0, 1] # 2D: νE/(1-ν²) 或 λ
                D22 = D[2, 2] # 2D: E/2(1+ν) 或 μ
                KK_11 = D00 * A_xx + D22 * A_yy
                KK_22 = D00 * A_yy + D22 * A_xx
                KK_12 = D01 * A_xy + D22 * A_yx
                KK_21 = D01 * A_yx + D22 * A_xy
            else:
                D00 = D[0, 0]  # 2μ + λ
                D01 = D[0, 1]  # λ
                D55 = D[5, 5]  # μ
                KK_11 = D00 * A_xx + D55 * (A_yy + A_zz)
                KK_22 = D00 * A_yy + D55 * (A_xx + A_zz)
                KK_33 = D00 * A_zz + D55 * (A_xx + A_yy)
                KK_12 = D01 * A_xy + D55 * A_yx
                KK_13 = D01 * A_xz + D55 * A_zx
                KK_21 = D01 * A_yx + D55 * A_xy
                KK_23 = D01 * A_yz + D55 * A_zy
                KK_31 = D01 * A_zx + D55 * A_xz
                KK_32 = D01 * A_zy + D55 * A_yz

        # 单元密度情况, D 为单元均匀矩阵
        elif coef.shape == (NC, ):
            if GD == 2:
                D00 = D[:, 0, 0]  # 2D: E/(1-ν²) 或 2μ+λ
                D01 = D[:, 0, 1]  # 2D: νE/(1-ν²) 或 λ
                D22 = D[:, 2, 2]  # 2D: E/2(1+ν) 或 μ
                KK_11 = bm.einsum('c, cij -> cij', D00, A_xx) + bm.einsum('c, cij -> cij', D22, A_yy)
                KK_22 = bm.einsum('c, cij -> cij', D00, A_yy) + bm.einsum('c, cij -> cij', D22, A_xx)
                KK_12 = bm.einsum('c, cij -> cij', D01, A_xy) + bm.einsum('c, cij -> cij', D22, A_yx)
                KK_21 = bm.einsum('c, cij -> cij', D01, A_yx) + bm.einsum('c, cij -> cij', D22, A_xy)
            else:
                D00 = D[:, 0, 0]  # 2μ + λ
                D01 = D[:, 0, 1]  # λ
                D55 = D[:, 5, 5]  # μ
                KK_11 = bm.einsum('c, cij -> cij', D00, A_xx) + bm.einsum('c, cij -> cij', D55, (A_yy + A_zz))
                KK_22 = bm.einsum('c, cij -> cij', D00, A_yy) + bm.einsum('c, cij -> cij', D55, (A_xx + A_zz))
                KK_33 = bm.einsum('c, cij -> cij', D00, A_zz) + bm.einsum('c, cij -> cij', D55, (A_xx + A_yy))
                KK_12 = bm.einsum('c, cij -> cij', D01, A_xy) + bm.einsum('c, cij -> cij', D55, A_yx)
                KK_13 = bm.einsum('c, cij -> cij', D01, A_xz) + bm.einsum('c, cij -> cij', D55, A_zx)
                KK_21 = bm.einsum('c, cij -> cij', D01, A_yx) + bm.einsum('c, cij -> cij', D55, A_xy)
                KK_23 = bm.einsum('c, cij -> cij', D01, A_yz) + bm.einsum('c, cij -> cij', D55, A_zy)
                KK_31 = bm.einsum('c, cij -> cij', D01, A_zx) + bm.einsum('c, cij -> cij', D55, A_xz)
                KK_32 = bm.einsum('c, cij -> cij', D01, A_zy) + bm.einsum('c, cij -> cij', D55, A_yz)

        if GD == 2:
            if space.dof_priority:
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(ldof, None), slice(ldof, None)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, None)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(ldof, None), slice(0, ldof)), KK_21)
            else:
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_21)

        else:
            if space.dof_priority:
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(ldof, 2 * ldof)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(2 * ldof, None)), KK_33)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, 2 * ldof)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(2 * ldof, None)), KK_13)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(0, ldof)), KK_21)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(2 * ldof, None)), KK_23)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(0, ldof)), KK_31)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(ldof, 2 * ldof)), KK_32)
            else:
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_11)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_22)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(2, KK.shape[2], GD)), KK_33)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_12)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(2, KK.shape[2], GD)), KK_13)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_21)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(2, KK.shape[2], GD)), KK_23)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(0, KK.shape[2], GD)), KK_31)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(1, KK.shape[2], GD)), KK_32)

        if enable_timing:
            t.send('组装部分')
            t.send(None)

        return KK

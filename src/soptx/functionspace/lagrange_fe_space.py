# 移植自 brighthe/fealpy ``fealpy/functionspace/lagrange_fe_space.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""标量 Lagrange 有限元空间."""

from typing import Optional, TypeVar, Union, Generic, Callable
from ..typing import TensorLike, Index, _S, Threshold

from ..backend import TensorLike
from ..backend import backend_manager as bm
from ..mesh.view.mesh_view import MeshView
from .space import FunctionSpace
from .dofs import LinearMeshCFEDof, LinearMeshDFEDof
from .function import Function
from ..decorator import barycentric, cartesian


_MT = TypeVar('_MT', bound=MeshView)


class LagrangeFESpace(FunctionSpace, Generic[_MT]):
    """单个 ``MeshView`` 上的标量 Lagrange 有限元空间.

    ``mesh`` 须为 ``MeshView``; ``TriangleMesh``、``TetrahedronMesh`` 等经典网格
    经其 ``MeshView`` 基类得到支持. 本类以 FEALPy 的网格接口作为对网格的约定.

    Parameters
    ----------
    mesh : MeshView
        网格.
    p : int, optional
        空间次数, 默认 1.
    ctype : {'C', 'D'}, optional
        连续 ('C', 默认) 或间断 ('D').

    Raises
    ------
    TypeError
        ``mesh`` 不是 ``MeshView``.
    """

    def __init__(self, mesh: _MT, p: int=1, ctype='C'):
        if not isinstance(mesh, MeshView):
            raise TypeError(
                "LagrangeFESpace expects a MeshView, "
                f"got {type(mesh).__name__}"
            )
        self.mesh = mesh
        self.p = p

        assert ctype in {'C', 'D'}
        self.ctype = ctype # 空间连续性类型

        if ctype == 'C':
            self.dof = LinearMeshCFEDof(mesh, p)
        elif ctype == 'D':
            self.dof = LinearMeshDFEDof(mesh, p)
        else:
            raise ValueError(f"Unknown type: {ctype}")

        self.ftype = mesh.ftype
        self.itype = mesh.itype
        # self.multi_index_matrix = mesh.multi_index_matrix(p,2)

        #TODO:JAX
        self.device = mesh.device
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

    def __str__(self):
        return "Lagrange finite element space on linear mesh!"

    def number_of_local_dofs(self, doftype='cell') -> int:
        """``doftype`` 类实体上的局部自由度个数."""
        return self.dof.number_of_local_dofs(doftype=doftype)

    def number_of_global_dofs(self) -> int:
        """全局自由度个数."""
        return self.dof.number_of_global_dofs()

    def interpolation_points(self) -> TensorLike:
        """自由度对应的插值点坐标."""
        return self.dof.interpolation_points()

    def cell_to_dof(self, index: Index=_S) -> TensorLike:
        """单元到全局自由度的映射."""
        return self.dof.cell_to_dof(index=index)

    def face_to_dof(self, index: Index=_S) -> TensorLike:
        """面到全局自由度的映射."""
        return self.dof.face_to_dof(index=index)

    def edge_to_dof(self, index=_S):
        """边到全局自由度的映射."""
        return self.dof.edge_to_dof(index=index)

    def is_boundary_dof(self, threshold=None, method=None) -> TensorLike:
        """标记边界自由度, 参数含义见 ``LinearMeshCFEDof.is_boundary_dof``.

        Raises
        ------
        RuntimeError
            间断空间没有边界自由度.
        """
        if self.ctype == 'C':
            return self.dof.is_boundary_dof(threshold, method=method)
        else:
            raise RuntimeError("boundary dof is not supported by discontinuous spaces.")

    def geo_dimension(self):
        """几何维数."""
        return self.GD

    def top_dimension(self):
        """拓扑维数."""
        return self.TD

    def project(self, u: Union[Callable[..., TensorLike], TensorLike],) -> TensorLike:
        """把单元上的分片常数值投影到 p=1 连续空间: 每个节点取相邻单元值的平均.

        Parameters
        ----------
        u : TensorLike
            单元值 ``(NC, )``, 或各单元各顶点上的值 ``(NC, NV)``.

        Returns
        -------
        Function
            投影得到的有限元函数.

        Notes
        -----
        只支持 p=1 且全局自由度数等于节点数的空间, 不满足时触发 ``AssertionError``.
        """
        gdof = self.number_of_global_dofs()
        NC = self.mesh.number_of_cells()
        NN = self.mesh.number_of_nodes()
        assert( NN == gdof )
        assert( u.shape[0] == NC )
        assert( self.p == 1 )
        if u.ndim == 1:
            u = u[:, None]
        cell = self.mesh.entity('cell')
        uh = bm.zeros(gdof, dtype=self.ftype, device=self.device)
        nn = bm.zeros(gdof, dtype=self.itype, device=self.device)
        uh = bm.index_add(uh, cell, u) 
        nn = bm.index_add(nn, cell, 1)

        return self.function(uh/nn)

    def interpolate(self, u: Union[Callable[..., TensorLike], TensorLike],) -> TensorLike:
        """把函数插值到空间中.

        Parameters
        ----------
        u : callable
            被插值函数; 无 ``coordtype`` 或为直角坐标时在插值点处求值.

        Returns
        -------
        Function
            插值函数.

        Notes
        -----
        ``coordtype == 'barycentric'`` 的分支按原注释标注结果不对, 不应使用.
        """
        assert callable(u)

        if not hasattr(u, 'coordtype') or u.coordtype == 'cartesian':
            ips = self.interpolation_points()
            uI = u(ips)
        elif u.coordtype == 'barycentric': # TODO: 这个结果是不对的 
            TD = self.TD
            p = self.p
            bcs = self.mesh.multi_index_matrix(p, TD)/p
            val = u(bcs)
            cell2dof = self.cell_to_dof()
            uI = bm.zeros(self.number_of_global_dofs(), dtype=self.ftype)
            uI = bm.index_add(uI, cell2dof, val) 
        return self.function(uI)

    def boundary_interpolate(self,
            gd: Union[Callable, int, float, TensorLike],
            uh: Optional[TensorLike] = None,
            *, threshold: Optional[Threshold]=None, method=None) -> TensorLike:
        """在边界自由度上设置第一类 (Dirichlet) 边界值.

        Parameters
        ----------
        gd : callable or TensorLike
            边界值: 在插值点处求值的函数, 或长度为全局自由度数的张量.
        uh : TensorLike, optional
            自由度数组, 在其上写入边界值; 默认新建全零数组.
        threshold : TensorLike or callable, optional
            边界自由度的筛选条件, 见 ``is_boundary_dof``.
        method : optional
            未使用; 边界自由度一律按 ``method='interp'`` 判定.

        Returns
        -------
        tuple
            ``(uh, isDDof)``: 写入边界值后的有限元函数, 以及边界自由度的布尔掩码.

        Raises
        ------
        TypeError
            ``gd`` 既不是张量也不是函数.
        """
        ipoints = self.interpolation_points() # TODO: 直接获取过滤后的插值点
        isDDof = self.is_boundary_dof(threshold=threshold, method='interp')
        if bm.is_tensor(gd):
            assert len(gd) == self.number_of_global_dofs()
            if uh is None:
                uh = bm.zeros_like(gd)
            uh = bm.set_at(uh, (..., isDDof), gd[isDDof])
        elif callable(gd):
            gd = gd(ipoints[isDDof])
            if uh is None:
                kwargs = bm.context(gd)
                uh = self.array(**kwargs)
            uh = bm.set_at(uh, (..., isDDof), gd)
        else:
            raise TypeError("gd must be a tensor or a callable function")
        
        return self.function(uh), isDDof

    set_dirichlet_bc = boundary_interpolate

    def basis(self, bc: TensorLike, index: Index=_S):
        """单元积分点处的基函数值, 形状 ``(1, NQ, ldof)``."""
        phi = self.mesh.shape_function(bc, self.p, index=index)
        return phi[None, ...] # (NC, NQ, LDOF)

    # 以下两个方法不能是 basis 的别名: basis 调用的 mesh.shape_function 绑定在
    # Entities(-1) 即单元上, 而面上的重心坐标属于面实体 (三角形的边上 2 个分量,
    # 四面体的面上 3 个分量), 单元的 schema 会拒绝它们. 重构前单纯形的
    # shape_function 由 bcs.shape[-1] - 1 推出 TD, 对传入的什么都能适配, 所以
    # 当时用别名可行.
    def face_basis(self, bc: TensorLike, index: Index=_S):
        """面积分点处的面基函数值, 形状 ``(1, NQ_face, ldof_face)``."""
        phi = self.mesh.face_shape_function(bc, self.p, index=index)
        return phi[None, ...] # (1, NQ_face, LDOF_face)

    def edge_basis(self, bc: TensorLike, index: Index=_S):
        """边积分点处的边基函数值, 形状 ``(1, NQ_edge, ldof_edge)``."""
        phi = self.mesh.edge_shape_function(bc, self.p, index=index)
        return phi[None, ...] # (1, NQ_edge, LDOF_edge)

    def grad_basis(self, bc: TensorLike, index: Index=_S, variable='x'):
        """单元积分点处的基函数梯度; ``variable='x'`` 为对直角坐标求导."""
        return self.mesh.grad_shape_function(
            bc,
            self.p,
            index=index,
            variables=variable,
        )

    @barycentric
    def value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        """有限元函数在积分点处的值, 形状 ``(..., NC, NQ)``.

        单纯形的重心坐标可以是张量或长度为 1 的元组; 张量积单元为各方向的元组.
        """
        if isinstance(bc, tuple):
            if len(bc) == 1:
                # 单纯形 bc 被积分基础设施包装为 1-tuple, 从张量形状推断 TD
                TD = bc[0].shape[-1] - 1
            else:
                # 张量积型单元 (四边形/六面体), TD = 积分公式个数
                TD = len(bc)
        else:
            # 单纯形单元 (三角形/四面体), TD = 重心坐标个数 - 1
            TD = bc.shape[-1] - 1
        phi = self.basis(bc, index=index)
        e2dof = self.dof.entity_to_dof(TD, index=index)
        val = bm.einsum('cql, ...cl -> ...cq', phi, uh[..., e2dof])
        return val

    @barycentric
    def grad_value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        """有限元函数在积分点处的梯度, 形状 ``(NC, NQ, GD)``; 重心坐标的形式同 ``value``."""
        if isinstance(bc, tuple):
            if len(bc) == 1:
                # 单纯形 bc 被积分基础设施包装为 1-tuple, 从张量形状推断 TD
                TD = bc[0].shape[-1] - 1
            else:
                # 张量积型单元 (四边形/六面体), TD = 积分公式个数
                TD = len(bc)
        else:
            # 单纯形单元 (三角形/四面体), TD = 重心坐标个数 - 1
            TD = bc.shape[-1] - 1
        gphi = self.grad_basis(bc, index=index)
        e2dof = self.dof.entity_to_dof(TD, index=index)
        val = bm.einsum('cilm, cl -> cim', gphi, uh[e2dof])
        return val

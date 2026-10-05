# 移植自 brighthe/fealpy ``fealpy/functionspace/lagrange_fe_space.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

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
    """Scalar Lagrange finite-element space over one :class:`MeshView`.

    ``mesh`` must be a ``MeshView``; classic mesh types such as
    ``TriangleMesh`` and ``TetrahedronMesh`` are supported through their
    ``MeshView`` base class.  This class keeps the historical FEALPy mesh
    interface as its consumer contract.
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
        return self.dof.number_of_local_dofs(doftype=doftype)

    def number_of_global_dofs(self) -> int:
        return self.dof.number_of_global_dofs()

    def interpolation_points(self) -> TensorLike:
        return self.dof.interpolation_points()

    def cell_to_dof(self, index: Index=_S) -> TensorLike:
        return self.dof.cell_to_dof(index=index)

    def face_to_dof(self, index: Index=_S) -> TensorLike:
        return self.dof.face_to_dof(index=index)

    def edge_to_dof(self, index=_S):
        return self.dof.edge_to_dof(index=index)

    def is_boundary_dof(self, threshold=None, method=None) -> TensorLike:
        if self.ctype == 'C':
            return self.dof.is_boundary_dof(threshold, method=method)
        else:
            raise RuntimeError("boundary dof is not supported by discontinuous spaces.")

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def project(self, u: Union[Callable[..., TensorLike], TensorLike],) -> TensorLike:
        """Project a function to the FE function space.
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
        """Set the first type (Dirichlet) boundary conditions.

        Parameters:
            gd: boundary condition function or value (can be a callable, int, float, TensorLike).
            uh: TensorLike, FE function uh .
            threshold: optional, threshold for determining boundary degrees of freedom (default: None).

        Returns:
            TensorLike: a bool array indicating the boundary degrees of freedom.

        This function sets the Dirichlet boundary conditions for the FE function `uh`. It supports
        different types for the boundary condition `gd`, such as a function, a scalar, or a array.
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
        phi = self.mesh.shape_function(bc, self.p, index=index)
        return phi[None, ...] # (NC, NQ, LDOF)

    # These cannot alias basis: basis calls mesh.shape_function, which is bound
    # to Entities(-1), the cell. Barycentric coordinates on a face belong to the
    # face entity instead -- 2 components on a triangle's edge, 3 on a
    # tetrahedron's face -- so the cell schema rejects them. The pre-refactor
    # simplex shape_function derived TD from bcs.shape[-1] - 1 and adapted to
    # whatever it was given, which is why the alias used to work.
    def face_basis(self, bc: TensorLike, index: Index=_S):
        phi = self.mesh.face_shape_function(bc, self.p, index=index)
        return phi[None, ...] # (1, NQ_face, LDOF_face)

    def edge_basis(self, bc: TensorLike, index: Index=_S):
        phi = self.mesh.edge_shape_function(bc, self.p, index=index)
        return phi[None, ...] # (1, NQ_edge, LDOF_edge)

    def grad_basis(self, bc: TensorLike, index: Index=_S, variable='x'):
        return self.mesh.grad_shape_function(
            bc,
            self.p,
            index=index,
            variables=variable,
        )

    @barycentric
    def value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        if isinstance(bc, tuple):
            if len(bc) == 1:
                # 单纯形 bc 被积分基础设施包装为 1-tuple，从张量形状推断 TD
                TD = bc[0].shape[-1] - 1
            else:
                # 张量积型单元 (四边形/六面体)，TD = 积分公式个数
                TD = len(bc)
        else:
            # 单纯形单元 (三角形/四面体)，TD = 重心坐标个数 - 1
            TD = bc.shape[-1] - 1
        phi = self.basis(bc, index=index)
        e2dof = self.dof.entity_to_dof(TD, index=index)
        val = bm.einsum('cql, ...cl -> ...cq', phi, uh[..., e2dof])
        return val

    @barycentric
    def grad_value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        if isinstance(bc, tuple):
            if len(bc) == 1:
                # 单纯形 bc 被积分基础设施包装为 1-tuple，从张量形状推断 TD
                TD = bc[0].shape[-1] - 1
            else:
                # 张量积型单元 (四边形/六面体)，TD = 积分公式个数
                TD = len(bc)
        else:
            # 单纯形单元 (三角形/四面体)，TD = 重心坐标个数 - 1
            TD = bc.shape[-1] - 1
        gphi = self.grad_basis(bc, index=index)
        e2dof = self.dof.entity_to_dof(TD, index=index)
        val = bm.einsum('cilm, cl -> cim', gphi, uh[e2dof])
        return val

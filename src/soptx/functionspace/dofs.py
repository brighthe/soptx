# 移植自 brighthe/fealpy ``fealpy/functionspace/dofs.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""线性网格 (``MeshView``) 上 Lagrange 空间的标量自由度管理器.

管理器使用 soptx.mesh 的 ``MeshView`` 接口: 调用 ``cell_to_ipoint``、
``face_to_ipoint``、``edge_to_ipoint``、``interpolation_points`` 及相应计数方法,
而不直接访问更底层的 ``EntityView``.
"""
__all__ = ['LinearMeshCFEDof']

from typing import Union, Generic, TypeVar

from ..backend import TensorLike
from ..backend import backend_manager as bm
from ..mesh.view.mesh_view import MeshView


_MT = TypeVar('_MT', bound=MeshView)
Index = Union[int, slice, TensorLike]
_S = slice(None)


class LinearMeshCFEDof(Generic[_MT]):
    """连续 (``ctype='C'``) Lagrange 空间的自由度管理器, 自由度即网格插值点.

    Parameters
    ----------
    mesh : MeshView
        网格.
    p : int
        空间次数.

    Raises
    ------
    TypeError
        ``mesh`` 不是 ``MeshView``.
    """
    def __init__(self, mesh: _MT, p: int):
        if not isinstance(mesh, MeshView):
            raise TypeError(
                "LinearMeshCFEDof expects a MeshView, "
                f"got {type(mesh).__name__}"
            )
        TD = mesh.top_dimension()
        self.mesh = mesh
        self.p = p
        self.multiIndex = mesh.multi_index_matrix(p, TD)
   
    def is_boundary_dof(self, threshold=None, method=None):
        """标记边界自由度.

        Parameters
        ----------
        threshold : TensorLike or callable, optional
            长度为全局自由度数的布尔张量时原样返回; 函数时用于在边界上再筛选.
        method : {None, 'centroid', 'interp'}, optional
            函数筛选的依据: 'centroid' (默认) 按边界面重心, 'interp' 按各自由度的
            插值点.

        Returns
        -------
        TensorLike
            长度为全局自由度数的布尔张量.

        Raises
        ------
        ValueError
            ``threshold`` 为张量但不是合法的布尔掩码, 或 ``method`` 未知.
        """
        TD = self.mesh.top_dimension()
        gdof = self.number_of_global_dofs()
        if bm.is_tensor(threshold):
            index = threshold
            if (index.dtype == bm.bool) and (len(index) == gdof):
                return index
            else:
                raise ValueError(f"Unknown threshold: {threshold}")
        else:
            if (method == 'centroid') | (method is None):
                index = self.mesh.boundary_face_index()
                if callable(threshold):
                    bc = self.mesh.entity_barycenter(TD-1, index=index)
                    flag = threshold(bc)
                    index = index[flag]
                face2dof = self.face_to_dof(index=index) # 只获取指定的面的自由度信息
                isBdDof = bm.zeros(gdof, dtype=bm.bool, device=bm.get_device(self.mesh))
                isBdDof = bm.set_at(isBdDof, face2dof, True)
            elif method == 'interp':
                index = self.mesh.boundary_face_index()
                face2dof = self.face_to_dof(index=index) # 只获取指定的面的自由度信息
                index_dof = face2dof.flatten()
                if callable(threshold):
                    ##TODO: 把 index_dof 传进插值点函数, 避免先算全部插值点
                    ipoint = self.mesh.interpolation_points(p=self.p)[index_dof]
                    flag = threshold(ipoint)
                    index_dof = index_dof[flag]
                isBdDof = bm.zeros(gdof, dtype=bm.bool, device=bm.get_device(self.mesh))
                isBdDof = bm.set_at(isBdDof, index_dof, True)
            else:
                raise ValueError(f"Unknown method: {method}")
        return isBdDof

    def entity_to_dof(self, etype: int, index: Index=_S):
        """按拓扑维数取单元、面或边到自由度的映射."""
        TD = self.mesh.top_dimension()
        if etype == TD:
            return self.cell_to_dof(index)
        elif etype == TD-1:
            return self.face_to_dof(index)
        elif etype == 1:
            return self.edge_to_dof(index)
        else:
            raise ValueError(f"Unknown entity type: {etype}")

    def edge_to_dof(self, index: Index=_S):
        """边到自由度的映射."""
        return self.mesh.edge_to_ipoint(self.p, index=index)

    def face_to_dof(self, index: Index=_S):
        """面到自由度的映射."""
        return self.mesh.face_to_ipoint(self.p, index=index)

    def cell_to_dof(self, index: Index=_S):
        """单元到自由度的映射."""
        return self.mesh.cell_to_ipoint(self.p, index=index)

    def interpolation_points(self, index: Index=_S) -> TensorLike:
        """自由度对应的插值点坐标."""
        return self.mesh.interpolation_points(self.p, index=index)

    def number_of_global_dofs(self) -> int:
        """全局自由度个数."""
        return self.mesh.number_of_global_ipoints(self.p)

    def number_of_local_dofs(self, doftype='cell') -> int:
        """``doftype`` 类实体上的局部自由度个数."""
        return self.mesh.number_of_local_ipoints(self.p, iptype=doftype)
    
class LinearMeshDFEDof(Generic[_MT]):
    """间断 (``ctype='D'``) Lagrange 空间的自由度管理器, 自由度按单元独立编号.

    Parameters
    ----------
    mesh : MeshView
        网格.
    p : int
        空间次数, 0 表示分片常数.

    Raises
    ------
    TypeError
        ``mesh`` 不是 ``MeshView``.
    """
    def __init__(self, mesh: _MT, p: int):
        if not isinstance(mesh, MeshView):
            raise TypeError(
                "LinearMeshDFEDof expects a MeshView, "
                f"got {type(mesh).__name__}"
            )
        TD = mesh.top_dimension()
        self.mesh = mesh
        self.p = p
        if p > 0:
            self.multiIndex = mesh.multi_index_matrix(p, TD)
        else:
            TD = mesh.top_dimension()
            self.multiIndex = bm.array((TD+1)*(0,), dtype=mesh.itype)
        self.cell2dof = self.cell_to_dof()

    def entity_to_dof(self, etype: int, index: Index=_S):
        """取单元到自由度的映射; 间断空间只支持单元.

        Raises
        ------
        ValueError
            ``etype`` 不是单元的拓扑维数.
        """
        TD = self.mesh.top_dimension()
        if etype == TD:
            return self.cell_to_dof(index)
        else:
            raise ValueError(f"Unknown entity type: {etype}")

    def cell_to_dof(self, index : Index=_S) -> TensorLike:
        """单元到自由度的映射: 第 ``c`` 个单元为 ``c*ldof`` 到 ``(c+1)*ldof - 1``."""
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs()
        cell2dof = bm.arange(NC*ldof).reshape(NC, ldof)

        return cell2dof[index]

    def number_of_global_dofs(self):
        """全局自由度个数, 即单元数乘以局部自由度数."""
        NC = self.mesh.number_of_cells()
        ldof = self.number_of_local_dofs()
        gdof = ldof*NC
        
        return gdof
    
    def number_of_local_dofs(self, doftype='cell') -> int:
        """``doftype`` 类实体上的局部自由度个数."""
        return self.mesh.number_of_local_ipoints(self.p, iptype=doftype)

    def interpolation_points(self):
        """各单元插值点坐标逐单元拼接, 形状 ``(NC*ldof, GD)``; p=0 时为单元重心."""
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')
        GD = mesh.geo_dimension()

        if p == 0:
            return mesh.entity_barycenter('cell')

        if p == 1:
            return node[cell].reshape(-1, GD)

        w = self.multiIndex/p
        ipoint = bm.einsum('ij, kj...->ki...', w, node[cell]).reshape(-1, GD)
        
        return ipoint

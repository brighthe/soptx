
from typing import Optional, TypeVar, Union, Generic, Callable
from fealpy.typing import TensorLike, Index, _S, Threshold

from fealpy.backend import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.mesh.mesh_base import Mesh
from fealpy.functionspace import FunctionSpace
from fealpy.functionspace.function import Function
from fealpy.functionspace.functional import symmetry_span_array, symmetry_index
from fealpy.decorator import barycentric, cartesian

import time

def number_of_multiindex(p, d):
    if d == 1:
        return p+1
    elif d == 2:
        return (p+1)*(p+2)//2
    elif d == 3:
        return (p+1)*(p+2)*(p+3)//6

def multiindex_to_number(a):
    d = a.shape[1] - 1
    if d==1:
        return a[:, 1]
    elif d==2:
        a1 = a[:, 1] + a[:, 2]
        a2 = a[:, 2]
        return a1*(1+a1)//2 + a2 
    elif d==3:
        a1 = a[:, 1] + a[:, 2] + a[:, 3]
        a2 = a[:, 2] + a[:, 3]
        a3 = a[:, 3]
        return a1*(1+a1)*(2+a1)//6 + a2*(1+a2)//2 + a3

class TensorDofsOnSubsimplex():
    def __init__(self, dofs : list, subsimplex : list):
        """
        dofs: list of tuple (alpha, I), alpha is the multi-index, I is the
              tensor index.
        """
        self.dof_scalar = bm.array([dof[0] for dof in dofs], dtype=bm.int32)
        self.dof_tensor = bm.array([dof[1] for dof in dofs], dtype=bm.int32)

        self.subsimplex = subsimplex

        self.dof2num = self._get_dof_to_num()

    def __getitem__(self, idx):
        return self.dof_scalar[idx], self.dof_tensor[idx]

    def __len__(self):
        return self.dof_scalar.shape[0]

    def _get_dof_to_num(self):
        alpha = self.dof_scalar
        I     = self.dof_tensor
        ldof  = number_of_multiindex(bm.sum(alpha[0]), alpha.shape[1]-1)
        idx = multiindex_to_number(alpha) + I*ldof

        nummap = bm.zeros((idx.max()+1,), dtype=alpha.dtype)
        nummap[idx] = bm.arange(len(idx), dtype=alpha.dtype)
        return nummap

    def permute_to_order(self, perm):
        alpha = self.dof_scalar.copy()
        alpha[:, self.subsimplex] = alpha[:, self.subsimplex][:, perm]

        I     = self.dof_tensor
        ldof  = number_of_multiindex(bm.sum(alpha[0]), alpha.shape[1]-1)
        idx = multiindex_to_number(alpha) + I*ldof
        return self.dof2num[idx]

class HuZhangFECellDof2d():
    def __init__(self, mesh : Mesh, p: int):
        self.p = p
        self.mesh = mesh
        self.TD = mesh.top_dimension() 

        self._get_simplex()
        self.boundary_dofs, self.internal_dofs = self.dof_classfication()

    def _get_simplex(self):
        TD = self.TD 
        mesh = self.mesh
        
        localnode = bm.array([[0], [1], [2]], dtype=mesh.itype)
        localcell = bm.array([[0, 1, 2]], dtype=mesh.itype)
        self.subsimplex = [localnode, mesh.localEdge, localcell]


        dual = lambda alpha : [i for i in range(self.TD+1) if i not in alpha]
        self.dual_subsimplex = [[dual(f) for f in ssixi] for ssixi in self.subsimplex]

    def dof_classfication(self):
        """
        张量自由度顺序: (-1, NS): gd_priority 
        (σ0_xx, σ0_xy, σ0_yy, 
         σ1_xx, σ1_xy, σ1_yy, 
         ...,
         σn_xx, σn_xy, σn_yy)
        Classify the dofs by the the entities.
        """
        p = self.p
        mesh = self.mesh
        TD = mesh.top_dimension()
        NS = TD*(TD+1)//2
        multiindex = bm.multi_index_matrix(self.p, TD)

        boundary_dofs = [[] for i in range(TD+1)]
        internal_dofs = [[] for i in range(TD+1)]
        for i in range(TD+1):
            fs = self.subsimplex[i] 
            fds = self.dual_subsimplex[i] 
            for j in range(len(fs)):
                flag0 = bm.all(multiindex[:, fs[j]] != 0, axis=-1)
                flag1 = bm.all(multiindex[:, fds[j]] == 0, axis=-1)
                flag =  flag0 & flag1 
                idx = bm.where(flag)[0]
                
                N_c = NS-i*(i+1)//2 # 连续标架的个数

                dof_cotinuous = [(alpha, num) for alpha in multiindex[idx] for num in range(N_c)]
                dof_discontinuous = [(alpha, num) for alpha in multiindex[idx] for num in range(N_c, NS)]

                if len(dof_cotinuous) > 0:
                    boundary_dofs[i].append(TensorDofsOnSubsimplex(dof_cotinuous, fs[j]))
                if len(dof_discontinuous) > 0:
                    internal_dofs[i].append(TensorDofsOnSubsimplex(dof_discontinuous, fs[j]))

        return boundary_dofs, internal_dofs 

    def get_boundary_dof_from_dim(self, d):
        """
        Get the dofs of the entities of dimension d.
        """
        return self.boundary_dofs[d]

    def get_internal_dof_from_dim(self, d):
        """
        Get the dofs of the entities of dimension d.
        """
        return self.internal_dofs[d]

class HuZhangFEDof2d():
    """ 
    @brief: The class of HuZhang finite element space dofs.
    @note: Only support the simplicial mesh, the order of  
            local edge, face of the mesh is the same as the order of subsimplex.
    """
    def __init__(self, mesh: Mesh, p: int):
        self.mesh = mesh
        self.p = p
        self.ftype = mesh.ftype
        self.itype = mesh.itype
        self.device = mesh.device

        self.cell_dofs = HuZhangFECellDof2d(mesh, p)

    def number_of_local_dofs(self) -> int:
        """Get the number of local dofs on cell """
        p = self.p
        TD = self.mesh.top_dimension()
        NS = TD*(TD+1)//2 # 对称矩阵的自由度个数
        
        return NS*(p+1)*(p+2)//2 

    def number_of_internal_local_dofs(self, doftype : str='cell') -> int:
        """Get the number of internal local dofs of the finite element space."""
        p = self.p
        TD = self.mesh.top_dimension()
        NS = TD*(TD+1)//2
        ldof = self.number_of_local_dofs()

        if doftype == 'cell':
            return ldof - NS*3 - 2*(p-1)*3
        
        elif doftype == 'face' or doftype == 'edge':
            return 2*(p-1) 
        
        elif doftype == 'node':
            return NS
        else:
            raise ValueError("Unknown doftype: {}".format(doftype))

    def number_of_global_dofs(self) -> int:
        """Get the number of global dofs of the finite element space."""
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()
        NN = mesh.number_of_nodes()

        cldof = self.number_of_internal_local_dofs('cell')
        eldof = self.number_of_internal_local_dofs('edge')
        nldof = self.number_of_internal_local_dofs('node')
        
        return NC*cldof + NE*eldof + NN*nldof

    def node_to_internal_dof(self) -> TensorLike:
        """Get the index array of the dofs defined on the nodes of the mesh."""
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        nldof = self.number_of_internal_local_dofs('node')

        node2dof = bm.arange(NN*nldof, dtype=self.itype, device=self.device)
        return node2dof.reshape(NN, nldof)

    node_to_dof = node_to_internal_dof

    def edge_to_internal_dof(self) -> TensorLike:
        """得到每条边的 2(p-1) 个牵引迹矩 DOFs"""
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        nldof = self.number_of_internal_local_dofs('node')
        eldof = self.number_of_internal_local_dofs('edge')

        N = NN*nldof
        edge2dof = bm.arange(N, N+NE*eldof, dtype=self.itype, device=self.device)

        return edge2dof.reshape(NE, eldof)

    def edge_to_dof(self, index: Index=_S) -> TensorLike:
        pass

    def cell_to_internal_dof(self) -> TensorLike:
        """Get the index array of the dofs defined on the cells of the mesh."""
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        nldof = self.number_of_internal_local_dofs('node')
        eldof = self.number_of_internal_local_dofs('edge')
        cldof = self.number_of_internal_local_dofs('cell')

        N = NN*nldof + NE*eldof
        cell2dof = bm.arange(N, N+NC*cldof, dtype=self.itype, device=self.device)
        return cell2dof.reshape(NC, cldof)

    def cell_to_dof(self, index: Index=_S) -> TensorLike:
        """
        自由度顺序: 顶点 → 边 → 单元
        Get the cell to dof map of the finite element space.
        """
        p = self.p
        mesh = self.mesh
        ldof = self.number_of_local_dofs()

        if index is _S:
            cell_indices = bm.arange(mesh.number_of_cells(), dtype=self.itype)
        else:
            cell_indices = index
        
        NC = len(cell_indices)

        edge = mesh.entity('edge')

        cell = mesh.entity('cell')[cell_indices]
        c2e  = mesh.cell_to_edge()[cell_indices]

        ndofs = self.cell_dofs.get_boundary_dof_from_dim(0)
        edofs = self.cell_dofs.get_boundary_dof_from_dim(1)

        node2idof = self.node_to_internal_dof()
        edge2idof = self.edge_to_internal_dof()
        cell2idof = self.cell_to_internal_dof()

        c2d = bm.zeros((NC, ldof), dtype=self.itype, device=self.device)
        idx = 0 # 统计自由度的个数

        # 顶点自由度
        for v, dof in enumerate(ndofs):
            n = len(dof)
            c2d[:, idx:idx+n] = node2idof[cell[:, v]]
            idx += n

        # 边自由度
        inverse_perm = [1, 0]
        for e, dof in enumerate(edofs):
            n = len(dof)

            le = bm.sort(mesh.localEdge[e])
            flag = cell[:, le[0]] != edge[c2e[:, e], 0]

            c2d[:, idx:idx+n] = edge2idof[c2e[:, e]]

            inverse_dofidx = dof.permute_to_order(inverse_perm)
            c2d[flag, idx:idx+n] = edge2idof[c2e[flag, e]][:, inverse_dofidx]
            idx += n

        # 单元自由度
        c2d[:, idx:] = cell2idof[cell_indices]

        return c2d

    def interpolation_points(self) -> TensorLike:
            """
            返回参考单元上插值点的重心坐标。

            胡张元的自由度是张量函数对多项式基的矩。这些自由度由函数在 p 次主格点（拉格朗日节点）
            处的值唯一确定。

            此函数返回这些点的坐标，并按照局部自由度的顺序（顶点 -> 边 -> 单元内部）进行排列，
            以确保与 cell_to_dof 和 basis 函数的顺序一致。
            """
            p = self.p
            mesh = self.mesh
            TD = mesh.top_dimension()

            # 1. 生成 p 次标量拉格朗日节点的重心坐标
            # multi_index_matrix 返回的数组包含度 p 在第一列，形如 (p, p1, p2, ...)
            multi_indices_with_p = bm.multi_index_matrix(p, TD)

            # 提取真正的多重指标 (p1, p2, ...)
            scalar_multi_indices = multi_indices_with_p[:, 1:]

            # 计算第一个重心坐标分量 p0 = p - sum(pi)
            p0 = p - bm.sum(scalar_multi_indices, axis=-1)

            # 拼接成完整的重心坐标多重指标 (p0, p1, p2, ...)
            bary_multi_indices = bm.concatenate([p0[:, None], scalar_multi_indices], axis=-1)
            
            # 归一化后即为重心坐标点
            # 对于 p=0 的情况，直接返回单元重心
            if p == 0:
                return bm.array([[1/3, 1/3, 1/3]], dtype=self.ftype, device=self.device)
                
            scalar_points = bary_multi_indices / p  # (ldof_scalar, TD+1)

            # 2. 按照与 basis 和 cell_to_dof 函数一致的顺序，收集所有自由度对应的标量部分的索引
            scalar_phi_indices_list = []

            # 顶点自由度 (边界)
            for dof_obj in self.cell_dofs.get_boundary_dof_from_dim(0):
                scalar_phi_indices_list.append(multiindex_to_number(dof_obj.dof_scalar))

            # 边自由度 (边界)
            for dof_obj in self.cell_dofs.get_boundary_dof_from_dim(1):
                scalar_phi_indices_list.append(multiindex_to_number(dof_obj.dof_scalar))
            
            # 边自由度 (内部)
            for dof_obj in self.cell_dofs.get_internal_dof_from_dim(1):
                scalar_phi_indices_list.append(multiindex_to_number(dof_obj.dof_scalar))
                
            # 单元自由度 (内部)
            for dof_obj in self.cell_dofs.get_internal_dof_from_dim(2):
                scalar_phi_indices_list.append(multiindex_to_number(dof_obj.dof_scalar))
            
            ldof = self.number_of_local_dofs()
            if not scalar_phi_indices_list:
                # 对于 p=0 且没有自由度的特殊情况
                return bm.zeros((ldof, TD + 1), dtype=self.ftype, device=self.device)

            # 将所有标量索引合并为一个长数组
            final_scalar_indices = bm.concatenate(scalar_phi_indices_list)

            # 3. 使用这些标量索引来从 `scalar_points` 中选取并排列插值点
            # multiindex_to_number 函数将多重指标映射到标量基函数的唯一索引，
            # 这些索引正对应 `scalar_points` 数组的行号。
            ipoints = scalar_points[final_scalar_indices]
            
            # 确保最终生成的插值点数量与局部自由度总数一致
            assert ipoints.shape[0] == ldof, \
                f"Shape mismatch: expected {ldof} points, but got {ipoints.shape[0]}"
            
            return ipoints

    def is_boundary_dof(self, threshold=None, method=None):
        """
        获取边界 DOFs 的布尔数组（节点边界 DOF + 边界边牵引迹矩 DOF）

        Parameters
        ----------
        threshold :
            - bm.bool 张量且长度为 gdof: 直接作为结果返回
            - None: 返回所有边界 DOF
            - callable(x) -> bool array: 在边界 DOF 中依据 method 进一步筛选
        method : {'barycenter', 'endpoint', None}
            - None/'barycenter': 通过边的重心判断是否在选定边界上
            - 'endpoint': 通过端点判断（任一端点满足 threshold 即选中该边）

        Returns
        -------
        isBdDof : (gdof,) 的 bool 张量
        """
        mesh = self.mesh
        gdof = self.number_of_global_dofs()
        device = bm.get_device(mesh)

        # 1) 布尔掩码直通（与拉格朗日空间一致）
        if bm.is_tensor(threshold):
            idx = threshold
            if (idx.dtype == bm.bool) and (len(idx) == gdof):
                return idx
            raise ValueError(f"Unknown threshold: {threshold}")

        # 2) 规范化 method
        if method is None:
            method = 'barycenter'
        if method not in ('barycenter', 'endpoint'):
            raise ValueError(f"Unknown method: {method}. Use 'barycenter', 'endpoint', or None.")

        # 3) 基础边界实体
        bd_edge_idx = mesh.boundary_face_index()   # (n_bd_edges,)
        edge = mesh.entity('edge')                 # (NE, 2)

        # 4) 选择边界边子集
        if threshold is None:
            sel_edge_idx = bd_edge_idx
        else:
            if not callable(threshold):
                raise ValueError("threshold must be callable, a bool mask of length gdof, or None")

            if method == 'barycenter':
                # 按边重心筛选
                edge_bc_all = mesh.entity_barycenter('edge')  # (NE, GD)
                edge_bc = edge_bc_all[bd_edge_idx]            # (n_bd_edges, GD)
                flag = threshold(edge_bc)                     # (n_bd_edges,)
                sel_edge_idx = bd_edge_idx[flag]
            else:  # 'endpoint'
                # 任一端点满足阈值则选中该边
                nodes = mesh.entity('node')                   # (NN, GD)
                node_flag = threshold(nodes)                  # (NN,)
                e_bd = edge[bd_edge_idx]                      # (n_bd_edges, 2)
                flag = node_flag[e_bd[:, 0]] | node_flag[e_bd[:, 1]]
                sel_edge_idx = bd_edge_idx[flag]

        # 5) 由选中边得到边界节点集合
        sel_edges = edge[sel_edge_idx]                        # (n_sel_edges, 2)
        if len(sel_edges) > 0:
            sel_nodes = bm.unique(sel_edges.reshape(-1))
        else:
            sel_nodes = bm.array([], dtype=self.itype, device=device)

        # 6) 索引映射：节点 DOF 与边 DOF
        node2idof = self.node_to_internal_dof()               # (NN, n_node_ldof)
        edge2idof = self.edge_to_internal_dof()               # (NE, 2*(p-1))

        # 7) 组装边界 DOF 掩码
        isBdDof = bm.zeros(gdof, dtype=bm.bool, device=device)

        # 节点上的 DOF（p=1 时也存在）
        if len(sel_nodes) > 0:
            idx_node = node2idof[sel_nodes].reshape(-1)
            isBdDof = bm.set_at(isBdDof, idx_node, True)

        # 边界边上的牵引迹矩 DOF（p=1 时此项为空）
        if len(sel_edge_idx) > 0:
            idx_edge = edge2idof[sel_edge_idx].reshape(-1)
            isBdDof = bm.set_at(isBdDof, idx_edge, True)

        return isBdDof

    def is_boundary_edge_dof(self, threshold: Optional[Threshold]=None, method='barycenter'):
        """
        获取边界边上的牵引迹矩 DOFs
        """
        mesh = self.mesh
        gdof = self.number_of_global_dofs()

        # 如果没有提供 threshold，返回全 False
        if threshold is None:
            return bm.zeros(gdof, dtype=bm.bool, device=self.device)

        # 判定哪些边在边界上
        if method == 'barycenter':
            edge_centers = mesh.entity_barycenter('edge')
            is_bd_edge   = threshold(edge_centers)          # (NE,)
        elif method == 'endpoint':
            nodes       = mesh.entity('node')
            is_bd_node  = threshold(nodes)                  # (NN,)
            edge        = mesh.entity('edge')               # (NE, 2)
            is_bd_edge  = is_bd_node[edge[:,0]] | is_bd_node[edge[:,1]]
        else:
            raise ValueError("method must be 'barycenter' or 'endpoint'")

        # 取出这些边上的牵引迹矩 DOF 全局索引，并置位
        isDDof     = bm.zeros(gdof, dtype=bm.bool, device=self.device)
        edge2idof  = self.edge_to_internal_dof()           # (NE, 2*(p-1)) —— 仅牵引迹矩
        bd_edges   = bm.where(is_bd_edge)[0]               # (n_bd_edges,)
        if bd_edges.size > 0:
            idx = edge2idof[bd_edges].reshape(-1)          # (n_bd_edges * 2*(p-1),)
            isDDof = bm.set_at(isDDof, idx, True)

        return isDDof


class HuZhangFESpace2d(FunctionSpace):
    def __init__(self, mesh, p: int=1, ctype='C'):
        self.mesh = mesh
        self.p = p

        cell_type = mesh.entity('cell').shape[1]
        if cell_type != 3:
            raise ValueError("HuZhangFESpace only support the simplicial mesh in 2D, "
                             "but the cell type of the mesh is {}".format(cell_type))

        self.dof = HuZhangFEDof2d(mesh, p)

        self.ftype = mesh.ftype
        self.itype = mesh.itype

        self.device = mesh.device
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

    # def __str__(self):
    #     return "HuZhangFESpace on {} with p={}".format(self.mesh, self.p)
    
    @property
    def NS(self):
        """
        对称矩阵/张量的独立分量数
        
        对于 2D: NS = 3 (σ_xx, σ_xy, σ_yy)
        对于 3D: NS = 6 (σ_xx, σ_xy, σ_xz, σ_yy, σ_yz, σ_zz)
        
        Returns
        -------
        独立分量的数量 = TD*(TD+1)//2
        """
        return self.TD * (self.TD + 1) // 2
    
    @property
    def shape(self):
        """
        自由度的形状，表示排序方式
        (-1, NS): gd_priority, 先排每个位置的所有应力分量
        
        对于 2D: (-1, 3) 表示 (σ0_xx, σ0_xy, σ0_yy, σ1_xx, σ1_xy, σ1_yy, ...)
        对于 3D: (-1, 6) 表示 (σ0_xx, σ0_xy, σ0_xz, σ0_yy, σ0_yz, σ0_zz, ...)
        """
        return (-1, self.NS)

    ## 自由度接口
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
        return self.dof.is_boundary_dof(threshold, method=method)
    
    def is_boundary_edge_dof(self, threshold=None, method=None) -> TensorLike:
        return self.dof.is_boundary_edge_dof(threshold, method=method)

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def dof_frame(self) -> TensorLike:
        mesh = self.mesh

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        nframe = bm.zeros((NN, 2, 2), dtype=mesh.ftype)
        eframe = bm.zeros((NE, 2, 2), dtype=mesh.ftype)
        cframe = bm.zeros((NC, 2, 2), dtype=mesh.ftype)

        node = mesh.entity('node')
        edge = mesh.entity('edge')
        cell = mesh.entity('cell')

        nframe[:, 0] = bm.array([[1, 0]], dtype=mesh.ftype) 
        nframe[:, 1] = bm.array([[0, 1]], dtype=mesh.ftype)
        cframe[:, 0] = bm.array([[1, 0]], dtype=mesh.ftype)
        cframe[:, 1] = bm.array([[0, 1]], dtype=mesh.ftype)

        eframe[:, 0] = mesh.edge_unit_normal()
        eframe[:, 1] = mesh.edge_unit_tangent()
        return nframe, eframe, cframe

    def dof_frame_of_S(self):
        mesh = self.mesh

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        nframe, eframe, cframe = self.dof_frame()
        multiindex = bm.multi_index_matrix(2, 1)
        idx, num = symmetry_index(2, 2)

        nsframe = bm.zeros((NN, 3, 3), dtype=self.ftype)
        for i, alpha in enumerate(multiindex): 
            nsframe[:, i] = symmetry_span_array(nframe, alpha).reshape(NN, -1)[:, idx]

        esframe = bm.zeros((NE, 3, 3), dtype=self.ftype)
        for i, alpha in enumerate(multiindex): 
            esframe[:, i] = symmetry_span_array(eframe, alpha).reshape(NE, -1)[:, idx]

        csframe = bm.zeros((NC, 3, 3), dtype=self.ftype)
        for i, alpha in enumerate(multiindex): 
            csframe[:, i] = symmetry_span_array(cframe, alpha).reshape(NC, -1)[:, idx]
        return nsframe, esframe, csframe

    basis_frame = dof_frame
    basis_frame_of_S = dof_frame_of_S

    def basis(self, bc: TensorLike, index: Index=_S):
        p = self.p
        mesh = self.mesh
        dof = self.dof

        ldof = dof.number_of_local_dofs()

        ndofs = dof.cell_dofs.get_boundary_dof_from_dim(0)
        edofs = dof.cell_dofs.get_boundary_dof_from_dim(1)

        iedofs = dof.cell_dofs.get_internal_dof_from_dim(1)
        icdofs = dof.cell_dofs.get_internal_dof_from_dim(2)

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        cell = mesh.entity('cell')
        c2e = mesh.cell_to_edge()

        nsframe, esframe, csframe = self.basis_frame_of_S()

        phi_s = self.mesh.shape_function(bc, self.p, index=index) # (NQ, LDOF)

        NQ = bc.shape[0]
        phi = bm.zeros((NC, NQ, ldof, 3), dtype=self.ftype)

        # 顶点基函数
        idx = 0
        for v, vdof in enumerate(ndofs):
            N = len(vdof)
            scalar_phi_idx = multiindex_to_number(vdof.dof_scalar)
            scalar_part = phi_s[None, :, scalar_phi_idx, None]
            tensor_part = nsframe[cell[:, v]][:, None, vdof.dof_tensor, :]
            phi[..., idx:idx+N, :] = scalar_part * tensor_part
            idx += N

        # 边界边基函数
        for e, edof in enumerate(edofs):
            N = len(edof)
            scalar_phi_idx = multiindex_to_number(edof.dof_scalar)
            scalar_part = phi_s[None, :, scalar_phi_idx, None]
            tensor_part = esframe[c2e[:, e]][:, None, edof.dof_tensor, :]
            phi[..., idx:idx+N, :] = scalar_part * tensor_part
            idx += N

        # 边内部基函数
        for e, edof in enumerate(iedofs):
            N = len(edof)
            scalar_phi_idx = multiindex_to_number(edof.dof_scalar)
            scalar_part = phi_s[None, :, scalar_phi_idx, None]
            tensor_part = esframe[c2e[:, e]][:, None, edof.dof_tensor, :]
            phi[..., idx:idx+N, :] = scalar_part * tensor_part
            idx += N
        
        # 单元气泡基函数 - 只有当 p >= n+1 时才存在单元内部自由度
        n = mesh.geo_dimension()
        if p >= n + 1:
            scalar_phi_idx = multiindex_to_number(icdofs[0].dof_scalar)
            scalar_part = phi_s[None, :, scalar_phi_idx, None]
            tensor_part = csframe[:, None, icdofs[0].dof_tensor, :]

            phi[..., idx:, :] = scalar_part * tensor_part

        return phi

    @barycentric
    def cell_basis_on_face(self, bc: TensorLike, index: Index=_S) -> TensorLike:
        """
        在面的积分点上计算所属单元的基函数
        
        这个函数专门用于边界积分, 例如计算 <u_D, τ·n>_Γ
        
        Parameters
        ----------
        bc: 边上的重心坐标 (NQ, 2)
        index: 边的索引（边界边的全局索引）
        
        Returns
        -------
        phi: 单元基函数在边上的值 (NF, NQ, LDOF, 3)
            其中 LDOF 是单元的全部局部自由度数
        """
        p = self.p
        mesh = self.mesh
        dof = self.dof
        
        NF = len(index) if isinstance(index, TensorLike) else mesh.number_of_edges()
        NQ = bc.shape[0]
        ldof = dof.number_of_local_dofs()  # 单元的全部局部自由度
        
        # 获取边与单元的拓扑关系
        face2cell = mesh.face_to_cell(index)
        cell_index = face2cell[:, 0]  # 左侧单元（边界边只有左侧单元）
        local_face_idx = face2cell[:, 2]  # 边在单元中的局部编号
        
        # 将边的重心坐标转换为单元的重心坐标
        # bc 是边的重心坐标 (NQ, 2)，需要转换为三角形的重心坐标 (NQ, 3)
        NLF = mesh.number_of_faces_of_cells()  # 三角形有3条边
        cbcs = mesh.update_bcs(bc, 'cell')  # 转换后的单元重心坐标列表
        
        # 初始化结果
        result = bm.zeros((NF, NQ, ldof, 3), dtype=self.ftype, device=self.device)
        
        # 获取所需的自由度信息
        ndofs = dof.cell_dofs.get_boundary_dof_from_dim(0)
        edofs = dof.cell_dofs.get_boundary_dof_from_dim(1)
        iedofs = dof.cell_dofs.get_internal_dof_from_dim(1)
        icdofs = dof.cell_dofs.get_internal_dof_from_dim(2)
        
        cell = mesh.entity('cell')[cell_index]
        c2e = mesh.cell_to_edge()[cell_index]
        
        nsframe, esframe, csframe = self.basis_frame_of_S()
        
        # 对每条边，根据其在单元中的局部位置计算基函数
        for i in range(NLF):
            # 找到局部编号为 i 的边
            tag = bm.where(local_face_idx == i)[0]
            if len(tag) == 0:
                continue
            
            # 在对应的单元重心坐标下计算标量基函数
            phi_s = mesh.shape_function(cbcs[i], p)  # (NQ, ldof_scalar)
            phi_s = phi_s[None, :, :, None]  # (1, NQ, ldof_scalar, 1)
            
            # 当前批次的单元索引
            curr_cells = cell_index[tag]
            curr_cell = cell[tag]
            curr_c2e = c2e[tag]
            
            # 计算基函数的每个部分
            idx = 0
            
            # 节点基函数
            for v, vdof in enumerate(ndofs):
                N = len(vdof)
                scalar_phi_idx = multiindex_to_number(vdof.dof_scalar)
                scalar_part = phi_s[:, :, scalar_phi_idx, :]  # (1, NQ, N, 1)
                tensor_part = nsframe[curr_cell[:, v]][:, None, vdof.dof_tensor, :]  # (NF_i, 1, N, 3)
                result[tag, :, idx:idx+N, :] = scalar_part * tensor_part
                idx += N
            
            # 边界边基函数
            for e, edof in enumerate(edofs):
                N = len(edof)
                scalar_phi_idx = multiindex_to_number(edof.dof_scalar)
                scalar_part = phi_s[:, :, scalar_phi_idx, :]
                tensor_part = esframe[curr_c2e[:, e]][:, None, edof.dof_tensor, :]
                result[tag, :, idx:idx+N, :] = scalar_part * tensor_part
                idx += N
            
            # 边内部基函数
            for e, edof in enumerate(iedofs):
                N = len(edof)
                scalar_phi_idx = multiindex_to_number(edof.dof_scalar)
                scalar_part = phi_s[:, :, scalar_phi_idx, :]
                tensor_part = esframe[curr_c2e[:, e]][:, None, edof.dof_tensor, :]
                result[tag, :, idx:idx+N, :] = scalar_part * tensor_part
                idx += N
            
            # 单元内部基函数（气泡函数）
            n = mesh.geo_dimension()
            if p >= n + 1:
                scalar_phi_idx = multiindex_to_number(icdofs[0].dof_scalar)
                scalar_part = phi_s[:, :, scalar_phi_idx, :]
                tensor_part = csframe[curr_cells][:, None, icdofs[0].dof_tensor, :]
                result[tag, :, idx:, :] = scalar_part * tensor_part
        
        return result

    def div_basis(self, bc: TensorLike): 
        p = self.p
        mesh = self.mesh
        dof = self.dof

        ldof = dof.number_of_local_dofs()

        ndofs = dof.cell_dofs.get_boundary_dof_from_dim(0)
        edofs = dof.cell_dofs.get_boundary_dof_from_dim(1)

        iedofs = dof.cell_dofs.get_internal_dof_from_dim(1)
        icdofs = dof.cell_dofs.get_internal_dof_from_dim(2)

        cell = mesh.entity('cell')
        c2e  = mesh.cell_to_edge()

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        nsframe, esframe, csframe = self.basis_frame_of_S()

        gphi_s = self.mesh.grad_shape_function(bc, self.p) # (NC, ldof, GD)

        NQ = bc.shape[0]
        dphi = bm.zeros((NC, NQ, ldof, 2), dtype=self.ftype)

        # 顶点基函数
        idx = 0
        for v, vdof in enumerate(ndofs):
            N = len(vdof)
            scalar_phi_idx = multiindex_to_number(vdof.dof_scalar)
            grad_scalar = gphi_s[..., scalar_phi_idx, :] # (NC, NQ, N, 2)
            frame = nsframe[cell[:, v]][:, None, vdof.dof_tensor] # (NC, 1, N, 3)
            dphi[..., idx:idx+N, 0] = bm.sum(grad_scalar * frame[..., :2], axis=-1)
            dphi[..., idx:idx+N, 1] = bm.sum(grad_scalar * frame[..., 1:], axis=-1)
            idx += N

        # 边界边基函数
        for e, edof in enumerate(edofs):
            N = len(edof)
            scalar_phi_idx = multiindex_to_number(edof.dof_scalar)
            grad_scalar = gphi_s[..., scalar_phi_idx, :]
            frame = esframe[c2e[:, e]][:, None, edof.dof_tensor]
            dphi[..., idx:idx+N, 0] = bm.sum(grad_scalar * frame[..., :2], axis=-1)
            dphi[..., idx:idx+N, 1] = bm.sum(grad_scalar * frame[..., 1:], axis=-1)
            idx += N

        # 边内部基函数
        for e, edof in enumerate(iedofs):
            N = len(edof)
            scalar_phi_idx = multiindex_to_number(edof.dof_scalar)
            grad_scalar = gphi_s[..., scalar_phi_idx, :]
            frame = esframe[c2e[:, e]][:, None, edof.dof_tensor]
            dphi[..., idx:idx+N, 0] = bm.sum(grad_scalar * frame[..., :2], axis=-1)
            dphi[..., idx:idx+N, 1] = bm.sum(grad_scalar * frame[..., 1:], axis=-1)
            idx += N

        # 单元气泡基函数 - 只有当 p >= n+1 时才存在单元内部自由度
        n = mesh.geo_dimension()
        if p >= n + 1:
            scalar_phi_idx = multiindex_to_number(icdofs[0].dof_scalar)
            grad_scalar = gphi_s[..., scalar_phi_idx, :]
            frame = csframe[:, None, icdofs[0].dof_tensor]

            dphi[..., idx:, 0] = bm.sum(grad_scalar * frame[..., :2], axis=-1)
            dphi[..., idx:, 1] = bm.sum(grad_scalar * frame[..., 1:], axis=-1)

        return dphi

    def hess_basis(self, bc: TensorLike, index: Index=_S, variable='x'):
        return self.mesh.hess_shape_function(bc, self.p, index=index, variables=variable)

    @barycentric
    def value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike: 
        if isinstance(bc, tuple):
            TD = len(bc)
        else :
            TD = bc.shape[-1] - 1
        phi = self.basis(bc, index=index)
        e2dof = self.dof.cell_to_dof()
        val = bm.einsum('cqld, ...cl -> ...cqd', phi, uh[..., e2dof])
        return val

    @barycentric
    def div_value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        if isinstance(bc, tuple):
            TD = len(bc)
        else :
            TD = bc.shape[-1] - 1
        gphi = self.div_basis(bc)
        # TODO 目前只考虑散度值在单元上计算的情形
        e2dof = self.dof.cell_to_dof(index=index)
        val = bm.einsum('cilm, cl -> cim', gphi, uh[e2dof])
        return val
    

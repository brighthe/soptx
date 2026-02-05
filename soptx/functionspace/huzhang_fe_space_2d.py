
from typing import Optional, Union, Callable
from fealpy.typing import TensorLike, Index, _S, Threshold

from fealpy.backend import backend_manager as bm
from fealpy.mesh.mesh_base import Mesh
from fealpy.sparse import COOTensor
from fealpy.sparse.ops import spdiags
from fealpy.functionspace import FunctionSpace
from fealpy.functionspace.functional import symmetry_span_array, symmetry_index
from fealpy.decorator import barycentric

import numpy as np

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
        """Get the dofs of the entities of dimension d."""
        return self.boundary_dofs[d]

    def get_internal_dof_from_dim(self, d):
        """Get the dofs of the entities of dimension d."""
        return self.internal_dofs[d]

class HuZhangFEDof2d():
    """ 
    @brief: The class of HuZhang finite element space dofs.
    @note: Only support the simplicial mesh, the order of  
            local edge, face of the mesh is the same as the order of subsimplex.
    """
    def __init__(self, mesh: Mesh, p: int, corner: dict=None, use_relaxation: bool=False):
        self.mesh = mesh
        self.p = p
        self.ftype = mesh.ftype
        self.itype = mesh.itype
        self.device = mesh.device

        self.use_relaxation = use_relaxation
        self.corner = corner

        # TODO 只有开启松弛且有角点数据时，NCP 才有效，否则为0
        self.NCP = len(corner['coords']) if (use_relaxation and corner is not None) else 0

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
        
        return NC*cldof + NE*eldof + NN*nldof + self.NCP
    
    def node_to_internal_dof(self) -> TensorLike:
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        nldof = self.number_of_internal_local_dofs('node')
        
        node2dof = bm.arange(NN*nldof, dtype=self.itype, device=self.device).reshape(NN, nldof)

        # TODO 根据开关决定是否处理角点自由度重定向
        if self.use_relaxation and self.NCP > 0:
            cornidx = self.corner['idx']
            N = self.NCP
            # 扩展出的自由度索引
            extra_dofs = bm.arange(NN*nldof, NN*nldof+N, dtype=self.itype, device=self.device).reshape(N, 1)
            # 构造角点处的 4 个自由度索引 [d0, d1, d2, new_d3]
            corner2dof = bm.concatenate([node2dof[cornidx], extra_dofs], axis=1)
            return node2dof, corner2dof 
        else:
            # 未松弛模式，不需要 corner2dof，返回 None 占位
            return node2dof, None

    node_to_dof = node_to_internal_dof

    def edge_to_internal_dof(self) -> TensorLike:
        """得到每条边的 2(p-1) 个牵引迹矩 DOFs"""
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        nldof = self.number_of_internal_local_dofs('node')
        eldof = self.number_of_internal_local_dofs('edge')

        N = NN*nldof + self.NCP
        edge2dof = bm.arange(N, N+NE*eldof, dtype=self.itype, device=self.device)

        return edge2dof.reshape(NE, eldof)

    def edge_to_dof(self) -> TensorLike:
        mesh = self.mesh
        edge = mesh.entity('edge')
        edge2idof = self.edge_to_internal_dof()
        node2idof, corner2dof = self.node_to_internal_dof()

        e0dof = node2idof[edge[:, 0], :2]
        e1dof = node2idof[edge[:, 1], :2]

        # TODO 只在开启松弛时执行角点逻辑
        if self.use_relaxation and self.NCP > 0:
            corner = self.corner
            for p in range(self.NCP):
                ce = corner['to_edge'][p]
                # 处理第一条边
                eid = ce[0]
                loc = ce[1]
                if loc == 0:
                    e0dof[eid, :] = corner2dof[p, :2]  # 分配前两个自由度给第一条边
                else:
                    e1dof[eid, :] = corner2dof[p, :2]  # 分配前两个自由度给第一条边
                
                # 处理第二条边
                eid = ce[2]
                loc = ce[3]
                if loc == 0:
                    e0dof[eid, :] = corner2dof[p, 2:]  # 分配后两个自由度给第二条边
                else:
                    e1dof[eid, :] = corner2dof[p, 2:]  # 分配后两个自由度给第二条边
                
        e2d = bm.concatenate([e0dof, edge2idof, e1dof], axis=1)
        
        return e2d

    def cell_to_internal_dof(self) -> TensorLike:
        """Get the index array of the dofs defined on the cells of the mesh."""
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        nldof = self.number_of_internal_local_dofs('node')
        eldof = self.number_of_internal_local_dofs('edge')
        cldof = self.number_of_internal_local_dofs('cell')

        N = NN*nldof + NE*eldof + self.NCP
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
        
        NC = mesh.number_of_cells()
        cell = mesh.entity('cell')
        edge = mesh.entity('edge')
        c2e  = mesh.cell_to_edge()

        ndofs = self.cell_dofs.get_boundary_dof_from_dim(0)
        edofs = self.cell_dofs.get_boundary_dof_from_dim(1)

        node2idof, corner2dof = self.node_to_internal_dof() 
        edge2idof = self.edge_to_internal_dof()
        cell2idof = self.cell_to_internal_dof()

        c2d = bm.zeros((NC, ldof), dtype=self.itype, device=self.device)
        idx = 0 

        # 1. 顶点自由度
        for v, dof in enumerate(ndofs):
            n = len(dof)
            c2d[:, idx:idx+n] = node2idof[cell[:, v]]
            idx += n

        # 2. 边自由度
        inverse_perm = [1, 0]
        for e, dof in enumerate(edofs):
            n = len(dof)
            le = bm.sort(mesh.localEdge[e])
            flag = cell[:, le[0]] != edge[c2e[:, e], 0]
            c2d[:, idx:idx+n] = edge2idof[c2e[:, e]]
            inverse_dofidx = dof.permute_to_order(inverse_perm)
            c2d[flag, idx:idx+n] = edge2idof[c2e[flag, e]][:, inverse_dofidx]
            idx += n

        # 3. 单元自由度
        c2d[:, idx:] = cell2idof

        # TODO 4. 角点自由度覆盖 (仅在松弛模式下)
        if self.use_relaxation and self.NCP > 0:
            corner = self.corner
            local_dof = bm.array([[0, 1, 2], [0, 1, 3]], dtype=bm.int32)
            for p in range(self.NCP):
                cp2dpf = corner2dof[p] # 取出该角点的4个DOFs
                cp2c = corner['to_cell'][p]
                for c in range(2):
                    cid = cp2c[c*2]
                    loc = cp2c[c*2+1]
                    # 将对应位置的3个标准DOF替换为松弛后的DOF组合
                    c2d[cid, loc*3:loc*3+3] = cp2dpf[local_dof[c]]
                    
        return c2d

class HuZhangFESpace2d(FunctionSpace):
    def __init__(self, mesh, p: int=1, ctype='C', use_relaxation: bool=False):
        self.mesh = mesh
        self.p = p

        cell_type = mesh.entity('cell').shape[1]
        if cell_type != 3:
            raise ValueError("HuZhangFESpace only support the simplicial mesh in 2D, "
                             "but the cell type of the mesh is {}".format(cell_type))

        # TODO 保持开关状态
        self.use_relaxation = use_relaxation

        # TODO 仅在需要松弛时计算角点拓扑
        if self.use_relaxation:
            self.corner = self._get_corner_data() 
            self.NCP = len(self.corner['coords'])
        else:
            self.corner = None
            self.NCP = 0

        # TODO 将开关传入 Dof 类
        self.dof = HuZhangFEDof2d(mesh, p, self.corner, use_relaxation=self.use_relaxation)

        self.ftype = mesh.ftype
        self.itype = mesh.itype
        self.device = mesh.device
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

        self.nsframe, self.esframe, self.csframe = self.dof_frame_of_S()
        self.TM = self._transform_matrix()

    def _transform_matrix(self):
        """
        构建基底变换矩阵 TM
        如果 use_relaxation=False, 则返回单位矩阵 (即不变换)
        """
        gdof = self.number_of_global_dofs()

        # 未松弛模式直接返回单位阵
        if not self.use_relaxation or self.NCP == 0:
            TM = spdiags(bm.ones(gdof, dtype=self.ftype), 0, gdof, gdof)

            return TM

        # 松弛模式
        nframe, eframe = self.nsframe, self.esframe
        _, corner2dof = self.dof.node_to_internal_dof()

        TM = bm.zeros((self.NCP, 4, 4), dtype=self.ftype)
        for p in range(self.NCP): 
            nid = self.corner['idx'][p]
            nf = nframe[nid] 

            ef0 = eframe[self.corner['to_edge'][p, 0], :2].copy()
            ef1 = eframe[self.corner['to_edge'][p, 2], :2].copy()
            ef0[1] *= 2
            ef1[1] *= 2

            M = bm.zeros((4, 4), dtype=self.ftype)
            num = bm.array([1, 2, 1], dtype=self.ftype)

            M[:3, :2] = bm.einsum('id, jd, d -> ij', nf, ef0, num)
            M[[0, 1, 3], 2:] = bm.einsum('id, jd, d -> ij', nf, ef1, num)
            TM[p] = bm.linalg.inv(M)

        I = bm.broadcast_to(corner2dof[:, None, :], (self.NCP, 4, 4))
        J = bm.broadcast_to(corner2dof[:, :, None], (self.NCP, 4, 4))

        TM = COOTensor(indices=bm.stack([I.reshape(-1), J.reshape(-1)], axis=0),
                    values=TM.reshape(-1), 
                    spshape=(gdof, gdof))
        TM = TM.tocsr()
        flag = bm.ones((gdof,), dtype=self.itype)
        flag = bm.set_at(flag, corner2dof.reshape(-1), 0.0)
        TM = TM.add(spdiags(flag, 0, gdof, gdof))

        return TM
    
    def _get_corner_data(self): 
        mesh = self.mesh
        node = mesh.entity('node')     # (NN, 2)
        cell = mesh.entity('cell')     # (NC, 3)
        edge = mesh.entity('edge')     # (NE, 2)
        c2e  = mesh.cell_to_edge()     # (NC, 3)

        isbdedge = mesh.boundary_edge_flag()  # (NE,)

        corners = mesh.meshdata['corner']
        NCP = len(corners)

        corner_idx = bm.zeros((NCP,), dtype=bm.int32)
        corner_to_cell = bm.zeros((NCP, 4), dtype=bm.int32)
        corner_to_edge = bm.zeros((NCP, 4), dtype=bm.int32)
        corner_to_midedge = bm.zeros((NCP,), dtype=bm.int32)
        for p, corner in enumerate(corners):
            nid = bm.where(bm.max(bm.abs(node - corner[None, :]), axis=-1) < 1e-12)[0]

            corner_idx[p] = nid

            # ====== 单元 ======
            mask = (cell == nid)              # (NC, 3)
            cids, locs = mask.nonzero()       # array of indices

            # 保证正好取两组
            corner_to_cell[p, 0] = cids[0]
            corner_to_cell[p, 1] = locs[0]
            corner_to_cell[p, 2] = cids[1]
            corner_to_cell[p, 3] = locs[1]

            # ====== 边 ======
            loc = bm.where(isbdedge[c2e[cids[0]]])[0]
            eid = c2e[cids[0], loc]
            corner_to_edge[p, 0] = eid
            corner_to_edge[p, 1] = bm.where(edge[eid[0]] == nid[0])[0]

            loc = bm.where(isbdedge[c2e[cids[1]]])[0]
            eid = c2e[cids[1], loc]
            corner_to_edge[p, 2] = eid
            corner_to_edge[p, 3] = bm.where(edge[eid[0]] == nid[0])[0]

            corner_to_midedge[p] = np.intersect1d(c2e[cids[0]], c2e[cids[1]])
            
        corner = {'coords' : corners, 'idx'    : corner_idx, 'to_cell':
                  corner_to_cell, 'to_edge': corner_to_edge, 'to_midedge':
                  corner_to_midedge}
        
        return corner

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
    def number_of_local_dofs(self) -> int:
        return self.dof.number_of_local_dofs()

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

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD
    
    def boundary_interpolate(self,
                            gd: Union[Callable, int, float, TensorLike],
                            uh: Optional[TensorLike] = None,
                            *, threshold: Optional[Threshold]=None, method=None,
                        ) -> TensorLike:
        if uh is None:
            uh = bm.zeros((self.number_of_global_dofs(),), dtype=self.ftype, device=self.device)

        mesh = self.mesh
        p = self.p

        if 'essential_bc' in mesh.edgedata:
            ebdflag = mesh.edgedata['essential_bc']
        else:
            ebdflag = mesh.boundary_edge_flag()

        e2d = self.dof.edge_to_dof()[ebdflag] # (NEb, Nbasis)
        NEb = e2d.shape[0]

        bcs = bm.multi_index_matrix(p, 1)/p
        points = self.mesh.bc_to_point(bcs)[ebdflag] # (NEb, Nbasis, 2)
        
        # 计算给定函数在边界积分点的值 (牵引力或应力)
        if callable(gd):
            gd_vals = gd(points)     # (NEb, Nbasis, 3) 或 (NEb, Nbasis, 3)
        else:
            gd_vals = bm.broadcast_to(gd, (NEb, len(bcs), gd.shape[-1]))

        # 获取边界标架
        eframe = self.esframe[ebdflag, :2].copy() # (NEb, 2, 3)

        dim_gd = gd_vals.shape[-1]

        # TODO 根据输入类型进行投影
        if dim_gd == 3:
            #* Case A: 输入是应力张量 (Voigt) [xx, xy, yy]
            # voigt 内积权重
            eframe[:, 1] *= 2
            num = bm.array([1, 2, 1], dtype=self.ftype)

            # 计算应力张量内积
            val = bm.einsum('eid, ejd, d -> eij', gd_vals, eframe, num)

        elif dim_gd == 2:
            #* Case B: 输入是牵引力向量 [gn, gt]
            # 获取边界法向 n 和切向 t
            en = mesh.edge_unit_normal()[ebdflag]   # (NEb, 2)
            et = mesh.edge_unit_tangent()[ebdflag]  # (NEb, 2)
            
            en = en[:, None, :]
            et = et[:, None, :]
                        
            # 法向投影 (sigma_nn)
            val_n = bm.sum(gd_vals * en, axis=-1)
            
            # 切向投影 (sigma_nt)
            val_t = bm.sum(gd_vals * et, axis=-1)
            
            val = bm.stack([val_n, 2.0 * val_t], axis=-1)

        else:
            raise ValueError(f"Unknown gd output dimension: {dim_gd.shape[-1]}")
        
        # 赋值
        uh[e2d] = val.reshape(NEb, -1)

        isDDof = bm.zeros((uh.shape[0],), dtype=bm.bool)
        isDDof[e2d] = True

        return self.function(uh), isDDof

    set_dirichlet_bc = boundary_interpolate

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
        nframe[edge] = eframe[:, None]

        # 修改边界点的标架
        isbdege = mesh.boundary_edge_flag()
        nframe[edge[isbdege]] = eframe[isbdege, None]

        # 修改角点的标架 (仅在松弛模式有效)
        # 如果 self.NCP == 0 (未开启松弛)，循环自动跳过
        for p in range(self.NCP):
            eid = self.corner['to_midedge'][p]
            nid = self.corner['idx'][p]
            # 将角点处的节点标架旋转至对齐中间分割边
            nframe[nid] = eframe[eid]
            
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

        if index is _S:
            cell_indices = slice(None)
            NC = mesh.number_of_cells()
        else:
            cell_indices = index
            NC = len(cell_indices) if hasattr(cell_indices, '__len__') else 1

        cell = mesh.entity('cell')[cell_indices]     
        c2e = mesh.cell_to_edge()[cell_indices]      

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

            if csframe.shape[0] == mesh.number_of_cells():
                 current_csframe = csframe[cell_indices]
            else:
                 current_csframe = csframe

            tensor_part = current_csframe[:, None, icdofs[0].dof_tensor, :]
            phi[..., idx:, :] = scalar_part * tensor_part

        return phi

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
        """
        计算有限元函数的值。
        自动处理松弛模式下的系数变换。
        """
        # 1. 系数变换 (仅在松弛模式且存在角点时执行，避免不必要的矩阵乘法)
        if self.use_relaxation and self.NCP > 0:
            uh0 = self.TM @ uh
        else:
            uh0 = uh

        # 2. 计算值
        phi = self.basis(bc, index=index)
        
        # 获取单元到自由度的映射
        e2dof = self.dof.cell_to_dof(index=index)

        val = bm.einsum('cqld, ...cl -> ...cqd', phi, uh0[..., e2dof])

        return val

    @barycentric
    def div_value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        """
        计算有限元函数的散度值
        """
        if isinstance(bc, tuple):
            TD = len(bc)
        else :
            TD = bc.shape[-1] - 1

        # 1. 系数变换
        if self.use_relaxation and self.NCP > 0:
            uh0 = self.TM @ uh
        else:
            uh0 = uh

        # 2. 计算散度基函数值
        gphi = self.div_basis(bc) 
        
        # 如果提供了 index，需要对 gphi 进行切片，因为 div_basis 通常计算所有单元
        if index is not _S:
            gphi = gphi[index]

        # 3. 获取自由度映射
        e2dof = self.dof.cell_to_dof(index=index)
        
        val = bm.einsum('cilm, cl -> cim', gphi, uh0[e2dof])

        return val
    

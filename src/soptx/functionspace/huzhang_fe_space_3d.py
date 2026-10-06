r"""三维四面体网格上的胡张 (Hu-Zhang) 对称应力有限元空间, 不含角点松弛.

应力空间 :math:`\Sigma_h` 由 :math:`P_k` 对称矩阵值函数组成, 满足 :math:`H(\mathrm{div})`
协调. 每个基函数写成 "标量 Lagrange 基 x 对称张量标架" 的乘积, 自由度按子单纯形
(顶点, 边, 面, 单元) 分层; ``i`` 维子单纯形上 6 个对称张量标架分量中有
``i(i+1)/2`` 个跨单元断开. 对称张量分量按 ``[xx, xy, xz, yy, yz, zz]`` 排列.
实现说明见 ``docs/fem/huzhang-mixed-fem-implementation.md``.
"""

from typing import Optional, TypeVar, Union, Generic, Callable
from soptx.typing import TensorLike, Index, _S, Threshold

from soptx.backend import TensorLike
from soptx.backend import backend_manager as bm
from soptx.mesh.mesh_base import Mesh
from .space import FunctionSpace
from .function import Function
from .functional import symmetry_span_array, symmetry_index
from soptx.decorator import barycentric, cartesian


def number_of_multiindex(p, d):
    """``d`` 维单纯形上 ``p`` 次多重指标的个数, 即 :math:`P_p` 的维数.

    Parameters
    ----------
    p : int
        多项式次数.
    d : int
        单纯形维数, 仅支持 1, 2, 3; 其余取值返回 None.

    Returns
    -------
    int
        ``p+1``, ``(p+1)(p+2)/2`` 或 ``(p+1)(p+2)(p+3)/6``.
    """
    if d == 1:
        return p+1
    elif d == 2:
        return (p+1)*(p+2)//2
    elif d == 3:
        return (p+1)*(p+2)*(p+3)//6

def multiindex_to_number(a):
    """把重心多重指标映射为它在 ``bm.multi_index_matrix(p, d)`` 中的行号.

    Parameters
    ----------
    a : TensorLike
        形状 ``(N, d+1)`` 的多重指标, 各行分量和相同, ``d`` 仅支持 1, 2, 3.

    Returns
    -------
    TensorLike
        形状 ``(N,)`` 的行号, 可直接索引 ``mesh.shape_function`` 的最后一维.
    """
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
    """单元内某个子单纯形上的一组张量自由度.

    每个自由度是二元组 ``(alpha, I)``: ``alpha`` 为标量 Lagrange 基的重心多重指标,
    ``I`` 为对称张量标架的分量编号. 对应的基函数为 "标量基 x 标架第 ``I`` 个对称张量".

    Parameters
    ----------
    dofs : list of tuple
        自由度列表 ``[(alpha, I), ...]``.
    subsimplex : TensorLike
        该子单纯形的局部顶点编号, 形状 ``(i+1,)``, ``i`` 为子单纯形维数.

    Attributes
    ----------
    dof_scalar : TensorLike
        形状 ``(n, TD+1)`` 的多重指标.
    dof_tensor : TensorLike
        形状 ``(n,)`` 的标架分量编号.
    dof2num : TensorLike
        由 ``multiindex_to_number(alpha) + I*ldof`` 到组内序号的反查表, 未出现的位置
        填 ``ldof+1``.
    """
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

        nummap = bm.zeros((idx.max()+1,), dtype=alpha.dtype)+ldof+1
        nummap[idx] = bm.arange(len(idx), dtype=alpha.dtype)
        return nummap

    def permute_to_order(self, perm):
        """子单纯形顶点按 ``perm`` 重排后, 各自由度对应到组内的序号.

        先把 ``subsimplex`` 升序排列, 再把每个自由度多重指标在这些位置上的分量按
        ``perm`` 重排, 张量分量编号保持不变, 最后经 ``dof2num`` 反查.
        ``cell_to_dof`` 用它对齐局部与全局的边, 面顶点顺序.

        Parameters
        ----------
        perm : list of int or TensorLike
            升序子单纯形顶点的排列, 边上为 ``[1, 0]``, 面上为 3 个顶点的排列.

        Returns
        -------
        TensorLike
            形状 ``(n,)`` 的组内序号.
        """
        alpha = self.dof_scalar.copy()
        idx = bm.sort(self.subsimplex)
        alpha[:, idx] = alpha[:, idx][:, perm]

        I     = self.dof_tensor
        ldof  = number_of_multiindex(bm.sum(alpha[0]), alpha.shape[1]-1)
        idx = multiindex_to_number(alpha) + I*ldof
        return self.dof2num[idx]

class HuZhangFECellDof3d():
    """四面体单元上胡张元局部自由度按子单纯形的分类.

    多重指标 ``alpha`` 归属于子单纯形 ``f`` 当且仅当 ``alpha`` 在 ``f`` 的顶点上全非零,
    在其余顶点上全为零. 对 ``i`` 维子单纯形, 6 个对称张量标架分量中前
    ``6 - i(i+1)/2`` 个跨单元连续, 其余断开: 顶点 6 个全连续, 边上 5 连续 1 断开,
    面上 3 连续 3 断开, 单元内部全断开.

    Parameters
    ----------
    mesh : Mesh
        三维单纯形网格, 局部边, 面顺序取自 ``mesh.localEdge``, ``mesh.localFace``.
    p : int
        应力空间的多项式次数.

    Attributes
    ----------
    boundary_dofs : list of list of TensorDofsOnSubsimplex
        ``boundary_dofs[i]`` 为各 ``i`` 维子单纯形上跨单元连续的自由度组.
    internal_dofs : list of list of TensorDofsOnSubsimplex
        ``internal_dofs[i]`` 为各 ``i`` 维子单纯形上单元私有的自由度组.
    """
    def __init__(self, mesh : Mesh, p: int):
        self.p = p
        self.mesh = mesh
        self.TD = mesh.top_dimension() 

        self._get_simplex()
        self.boundary_dofs, self.internal_dofs = self._dof_classfication()

    def _get_simplex(self):
        TD = self.TD 
        mesh = self.mesh

        localnode = bm.array([[0], [1], [2], [3]], dtype=mesh.itype)
        localcell = bm.array([[0, 1, 2, 3]], dtype=mesh.itype)
        self.subsimplex = [localnode, mesh.localEdge, mesh.localFace, localcell]

        dual = lambda alpha : [i for i in range(self.TD+1) if i not in alpha]
        self.dual_subsimplex = [[dual(f) for f in ssixi] for ssixi in self.subsimplex]

    def _dof_classfication(self):
        """
        张量自由度顺序: (-1, NS): gd_priority 
        (σ0_xx, σ0_xy, σ0_xz, σ0_yy, σ0_yz, σ0_zz, 
         σ1_xx, σ1_xy, σ1_xz, σ1_yy, σ1_yz, σ1_zz, 
         ...,
         σn_xx, σn_xy, σn_xz, σn_yy, σn_yz, σn_zz)

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

    # 下面的函数暂时不需要
    #def get_subsimplex_of_multiindex(self, alpha):
    #    """
    #    Get the subsimplex of the multi-index alpha.
    #    """
    #    subsimplex = bm.where(alpha != 0)[0]
    #    return subsimplex

    #def get_all_dofs(self):
    #    all_dofs_scalar = []
    #    all_dofs_tensor = []
    #    for dofs in self.boundary_dofs:
    #        all_dofs_scalar += [dof.dof_scalar for dof in dofs]
    #        all_dofs_tensor += [dof.dof_tensor for dof in dofs]
    #    for dofs in self.internal_dofs:
    #        all_dofs_scalar += [dof.dof_scalar for dof in dofs]
    #        all_dofs_tensor += [dof.dof_tensor for dof in dofs]
    #    all_dofs_scalar = bm.concatenate(all_dofs_scalar, axis=0)
    #    all_dofs_tensor = bm.concatenate(all_dofs_tensor, axis=0)
    #    return all_dofs_scalar, all_dofs_tensor 

class HuZhangFEDof3d():
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

        self.cell_dofs = HuZhangFECellDof3d(mesh, p)

    def number_of_local_dofs(self) -> int:
        """
        Get the number of local dofs on cell 
        """
        p = self.p
        TD = self.mesh.top_dimension()
        NS = TD*(TD+1)//2 # 对称矩阵的自由度个数

        return NS * number_of_multiindex(p, TD)

    def number_of_internal_local_dofs(self, doftype : str='cell') -> int:
        """
        Get the number of internal local dofs of the finite element space.
        """
        p = self.p
        TD = self.mesh.top_dimension()
        NS = TD*(TD+1)//2
        ldof = self.number_of_local_dofs()
        if doftype == 'cell':
            nidofs = NS
            eidofs = 5*(p-1)
            fidofs = 3*(p-1)*(p-2)//2
            return ldof - nidofs*4 - eidofs*6 - fidofs*4
        elif doftype == 'face':
            return (p-1)*(p-2)//2*3
        elif doftype == 'edge':
            return 5*(p-1) 
        elif doftype == 'node':
            return NS
        else:
            raise ValueError("Unknown doftype: {}".format(doftype))

    def number_of_global_dofs(self) -> int:
        """Get the number of global dofs of the finite element space."""
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NF = mesh.number_of_faces()
        NE = mesh.number_of_edges()
        NN = mesh.number_of_nodes()

        cldof = self.number_of_internal_local_dofs('cell')
        fldof = self.number_of_internal_local_dofs('face')
        eldof = self.number_of_internal_local_dofs('edge')
        nldof = self.number_of_internal_local_dofs('node')

        return NC*cldof + NF*fldof + NE*eldof + NN*nldof

    def node_to_internal_dof(self) -> TensorLike:
        """Get the index array of the dofs defined on the nodes of the mesh."""
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        nldof = self.number_of_internal_local_dofs('node')

        node2dof = bm.arange(NN*nldof, dtype=self.itype, device=self.device)

        return node2dof.reshape(NN, nldof)

    node_to_dof = node_to_internal_dof

    def edge_to_internal_dof(self) -> TensorLike:
        """Get the index array of the dofs defined on the edges of the mesh."""
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        nldof = self.number_of_internal_local_dofs('node')
        eldof = self.number_of_internal_local_dofs('edge')

        N = NN*nldof
        edge2dof = bm.arange(N, N+NE*eldof, dtype=self.itype, device=self.device)

        return edge2dof.reshape(NE, eldof)

    def edge_to_dof(self, index: Index=_S) -> TensorLike:
        """边到全局自由度的映射; 尚未实现.

        Raises
        ------
        NotImplementedError
            总是抛出.
        """
        raise NotImplementedError("HuZhangFEDof3d 尚未实现 edge_to_dof.")

    def face_to_internal_dof(self) -> TensorLike:
        """Get the index array of the dofs defined on the faces of the mesh."""
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()

        nldof = self.number_of_internal_local_dofs('node')
        eldof = self.number_of_internal_local_dofs('edge')
        fldof = self.number_of_internal_local_dofs('face')

        N = NN*nldof + NE*eldof
        face2dof = bm.arange(N, N+NF*fldof, dtype=self.itype, device=self.device)
        return face2dof.reshape(NF, fldof)

    def face_to_dof(self, index: Index=_S) -> TensorLike:
        """面到全局自由度的映射; 尚未实现.

        Raises
        ------
        NotImplementedError
            总是抛出.
        """
        raise NotImplementedError("HuZhangFEDof3d 尚未实现 face_to_dof.")

    def cell_to_internal_dof(self) -> TensorLike:
        """Get the index array of the dofs defined on the cells of the mesh."""
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()

        nldof = self.number_of_internal_local_dofs('node')
        eldof = self.number_of_internal_local_dofs('edge')
        fldof = self.number_of_internal_local_dofs('face')
        cldof = self.number_of_internal_local_dofs('cell')

        N = NN*nldof + NE*eldof + NF*fldof
        cell2dof = bm.arange(N, N+NC*cldof, dtype=self.itype, device=self.device)
        return cell2dof.reshape(NC, cldof)

    def cell_to_dof(self, index: Index=_S) -> TensorLike:
        """Get the cell to dof map of the finite element space."""
        p = self.p
        mesh = self.mesh
        ldof = self.number_of_local_dofs()

        NC = mesh.number_of_cells()
        cell = mesh.entity('cell')
        face = mesh.entity('face')
        edge = mesh.entity('edge')
        c2e  = mesh.cell_to_edge()
        c2f  = mesh.cell_to_face()

        ndofs = self.cell_dofs.get_boundary_dof_from_dim(0)
        edofs = self.cell_dofs.get_boundary_dof_from_dim(1)
        fdofs = self.cell_dofs.get_boundary_dof_from_dim(2)

        node2idof = self.node_to_internal_dof()
        edge2idof = self.edge_to_internal_dof()
        cell2idof = self.cell_to_internal_dof()
        face2idof = self.face_to_internal_dof()

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

        # 面自由度
        perm2num = lambda a : a[:, 0]*2 + (a[:, 1]>a[:, 2])
        for f, dof in enumerate(fdofs):
            n = len(dof)

            lf = bm.sort(mesh.localFace[f])

            face_glo = face[c2f[:, f]]
            face_loc = cell[:, lf]

            glo = face_glo.copy()
            loc = face_loc.copy()

            face_glo = bm.argsort(face_glo, axis=1)
            face_glo = bm.argsort(face_glo, axis=1)
            face_loc = bm.argsort(face_loc, axis=1)

            face_order = face_loc[bm.arange(NC)[:, None], face_glo]

            # global = local[order]
            pnum = perm2num(face_order)
            for i in range(6):
                flag = pnum == i
                if ~bm.any(flag):
                    continue
                perm = face_order[flag][0]
                permidx = dof.permute_to_order(perm)
                c2d[flag, idx:idx+n] = face2idof[c2f[flag, f]][:, permidx]

            idx += n

        # 单元自由度
        c2d[:, idx:] = cell2idof

        # 同 2d 版本: 之前这里忽略了 index, 与按子集计算的 basis 不匹配
        return c2d[index]

    def is_boundary_dof(self, threshold=None, method=None) -> TensorLike:
        """标记边界自由度; 尚未实现.

        Raises
        ------
        NotImplementedError
            总是抛出.
        """
        raise NotImplementedError("HuZhangFEDof3d 尚未实现 is_boundary_dof.")

def _require_all_cells(index: Index, name: str) -> None:
    """三维胡张元的单元量尚未按子集选取, ``index`` 不是全部单元时报错.

    Raises
    ------
    NotImplementedError
        ``index`` 不是 ``_S``.
    """
    if index is not _S:
        raise NotImplementedError(f"HuZhangFESpace3d.{name} 只支持全部单元, 尚不支持 index 子集.")


class HuZhangFESpace3d(FunctionSpace):
    r"""三维四面体网格上的胡张 (Hu-Zhang) 应力空间 :math:`\Sigma_h`, 不支持角点松弛.

    基函数取值为 Voigt 形式 ``[xx, xy, xz, yy, yz, zz]`` 的对称张量; 全局自由度按
    "顶点段 -> 边段 -> 面段 -> 单元段" 编号.

    Parameters
    ----------
    mesh : Mesh
        四面体网格.
    p : int, optional
        应力多项式次数 ``k``, 默认 1.
    ctype : str, optional
        保留参数, 当前未使用.

    Attributes
    ----------
    dof : HuZhangFEDof3d
        自由度管理对象.
    use_relaxation : bool
        恒为 False; 与二维空间同名, 供分析器统一判断.
    """
    def __init__(self, mesh, p: int=1, ctype='C'):
        self.mesh = mesh
        self.p = p

        self.dof = HuZhangFEDof3d(mesh, p)

        self.ftype = mesh.ftype
        self.itype = mesh.itype

        self.device = mesh.device
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()
        # 三维没有角点松弛; 与二维空间同名的属性供分析器统一判断
        self.use_relaxation = False


    ## 自由度接口
    def number_of_local_dofs(self) -> int:
        """单元上的局部自由度个数 ``(p+1)(p+2)(p+3)``."""
        return self.dof.number_of_local_dofs()

    def number_of_global_dofs(self) -> int:
        """全局自由度个数."""
        return self.dof.number_of_global_dofs()

    def interpolation_points(self) -> TensorLike:
        """自由度对应的插值点坐标; 尚未实现.

        Raises
        ------
        NotImplementedError
            总是抛出.
        """
        raise NotImplementedError("HuZhangFESpace3d 尚未实现 interpolation_points.")

    def cell_to_dof(self, index: Index=_S) -> TensorLike:
        """单元到全局自由度的映射, 形状 ``(NC, ldof)``, 顺序为顶点 -> 边 -> 面 -> 单元."""
        return self.dof.cell_to_dof(index=index)

    def face_to_dof(self, index: Index=_S) -> TensorLike:
        """面到全局自由度的映射; 尚未实现, 抛出 NotImplementedError."""
        return self.dof.face_to_dof(index=index)

    def edge_to_dof(self, index=_S):
        """边到全局自由度的映射; 尚未实现, 抛出 NotImplementedError."""
        return self.dof.edge_to_dof(index=index)

    def is_boundary_dof(self, threshold=None, method=None) -> TensorLike:
        """标记边界自由度; 尚未实现, 抛出 NotImplementedError."""
        return self.dof.is_boundary_dof(threshold, method=method)

    def _traction_writes(self, gd, threshold, tangential_only: bool):
        r"""边界面闭包格点上受牵引约束的自由度及其取值.

        在格点上只有该点的标量 Lagrange 基为 1, 位于此点的局部基函数取值即其标架张量
        :math:`S`. 记外法向 :math:`n`, :math:`P = I - nn^{\mathsf T}`:

        - :math:`Sn = 0`: 不受牵引约束;
        - :math:`PSP = 0`: :math:`S = \mathrm{sym}(n \otimes w)`, :math:`w = 2Sn - (n^{\mathsf T}Sn)n`,
          故 :math:`\sigma : S = t \cdot w` 只依赖牵引 :math:`t = \sigma n`; 标架单位正交时系数为
          :math:`(\sigma : S) / \|S\|_F^2`, 即 ``num_k * (sigma : S_k)``;
        - 其余情况: 标架向量与边界法向既不平行也不正交, 无法只由牵引确定.

        ``tangential_only`` 为 True 时只取 :math:`n^{\mathsf T}Sn = 0` 的切向分量.

        Returns
        -------
        dofs : TensorLike
            受约束的全局自由度编号 (去重, 升序).
        values : TensorLike
            对应取值; 被多个面写入的自由度取平均.

        Raises
        ------
        NotImplementedError
            某个受约束的标架张量含切向-切向分量 (非坐标对齐边界上的顶点等).
        """
        from .huzhang_fe_space_2d import boundary_outward_sign

        mesh = self.mesh
        p = self.p
        flag = mesh.boundary_face_flag() if threshold is None else threshold
        fidx = bm.nonzero(flag)[0] if getattr(flag, 'dtype', None) == bm.bool else bm.asarray(flag)
        if len(fidx) == 0:
            return bm.zeros((0,), dtype=self.itype), bm.zeros((0,), dtype=self.ftype)

        f2c = mesh.face_to_cell()[fidx]
        n_out = mesh.face_unit_normal()[fidx] * boundary_outward_sign(mesh, fidx)[:, None]
        lattice = bm.multi_index_matrix(p, 2) / p                 # 面上的格点, (NL, 3)
        c2d = self.cell_to_dof()
        pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        eye = bm.eye(3, dtype=self.ftype)
        tol = 1e-10

        dofs, values = [], []
        for i in range(4):
            on = f2c[:, 2] == i
            if not bool(bm.any(on)):
                continue
            cids = f2c[on, 0]
            n = n_out[on]                                          # (nf, 3)
            bcs = bm.insert(lattice, i, 0, axis=-1)                 # 单元局部面 i 与顶点 i 相对
            phi = self.basis(bcs)[cids]                            # (nf, NL, ldof, 6)
            points = mesh.bc_to_point(bcs, index=cids)             # (nf, NL, 3)

            t = gd(points) if callable(gd) else bm.broadcast_to(bm.tensor(gd, dtype=self.ftype), points.shape)
            if t.shape[-1] == 6:                                   # Voigt 应力, 取 t = sigma n
                T = bm.zeros(t.shape[:-1] + (3, 3), dtype=self.ftype)
                for k, (a, b) in enumerate(pairs):
                    T[..., a, b] = t[..., k]
                    T[..., b, a] = t[..., k]
                t = bm.einsum('fqab,fb->fqa', T, n)
            elif t.shape[-1] != 3:
                raise ValueError(f"gd 的最后一维须为 3 (牵引) 或 6 (Voigt 应力), 得到 {t.shape[-1]}.")

            S = bm.zeros(phi.shape[:-1] + (3, 3), dtype=self.ftype)
            for k, (a, b) in enumerate(pairs):
                S[..., a, b] = phi[..., k]
                S[..., b, a] = phi[..., k]
            Sn = bm.einsum('fqlab,fb->fqla', S, n)
            nSn = bm.einsum('fqla,fa->fql', Sn, n)
            P = eye - n[:, :, None] * n[:, None, :]
            tt = bm.einsum('fab,fqlbc,fcd->fqlad', P, S, P)
            norm2 = bm.sum(S ** 2, axis=(-2, -1))

            present = norm2 > tol
            constrained = present & (bm.max(bm.abs(Sn), axis=-1) > tol)
            if tangential_only:
                constrained = constrained & (bm.abs(nSn) < tol)
            misaligned = constrained & (bm.max(bm.abs(tt), axis=(-2, -1)) > tol)
            if bool(bm.any(misaligned)):
                raise NotImplementedError(
                    "边界上的标架向量与边界法向既不平行也不正交 (如非坐标对齐边界上的顶点), "
                    "牵引无法只由法向迹确定; 需要顶点标架对齐或角点松弛, 尚未实现."
                )
            w = 2.0 * Sn - nSn[..., None] * n[:, None, None, :]
            val = bm.einsum('fqa,fqla->fql', t, w) / bm.where(present, norm2, 1.0)
            gidx = bm.broadcast_to(c2d[cids][:, None, :], constrained.shape)
            dofs.append(gidx[constrained])
            values.append(val[constrained])

        dofs = bm.concat(dofs) if dofs else bm.zeros((0,), dtype=self.itype)
        values = bm.concat(values) if values else bm.zeros((0,), dtype=self.ftype)
        unique, inverse = bm.unique(dofs, return_inverse=True)
        total = bm.zeros(unique.shape, dtype=self.ftype)
        count = bm.zeros(unique.shape, dtype=self.ftype)
        total = bm.index_add(total, inverse, values)
        count = bm.index_add(count, inverse, bm.ones_like(values))
        return unique, total / count

    def boundary_interpolate(self,
                             gd: Union[Callable, TensorLike],
                             uh: Optional[TensorLike] = None,
                             *, threshold: Optional[Threshold] = None, method=None,
                         ) -> TensorLike:
        r"""在牵引边界上强施加法向迹 :math:`\sigma n = t` (混合法的本质边界).

        取值规则见 ``_traction_writes``; 与二维约定一致, 写入的系数为
        ``num_k * (sigma : S_k)``, 被多个面重复写入的自由度取平均.

        Parameters
        ----------
        gd : Callable or TensorLike
            边界数据. 可调用时以形状 ``(NFb, NL, 3)`` 的面格点坐标调用 (``NL`` 为面上
            ``p`` 次格点数), 返回最后一维为 3 的外法向牵引, 或最后一维为 6 的 Voigt 应力;
            非可调用时为常张量.
        uh : TensorLike, optional
            形状 ``(gdof,)`` 的自由度向量, 缺省为零向量.
        threshold : TensorLike, optional
            边界面的选取 (布尔标记或下标), 缺省取 ``mesh.boundary_face_flag()``.
        method : optional
            保留参数, 当前未使用.

        Returns
        -------
        uh : TensorLike
            写入边界值后的自由度向量.
        isDDof : TensorLike
            形状 ``(gdof,)`` 的布尔数组, 标记被写入的自由度.

        Raises
        ------
        NotImplementedError
            边界标架未与边界法向对齐 (非坐标对齐边界上的顶点等).
        """
        return self._write_traction(gd, uh, threshold, tangential_only=False)

    set_dirichlet_bc = boundary_interpolate

    def set_tangential_traction_bc(self,
                                   gd: Union[Callable, TensorLike],
                                   uh: Optional[TensorLike] = None,
                                   *, threshold: Optional[Threshold] = None,
                               ) -> TensorLike:
        r"""只强施加切向牵引 (对称面), 法向-法向分量保持自由.

        与 ``boundary_interpolate`` 相同, 但只写入 :math:`n^{\mathsf T} S n = 0` 的分量, 即
        :math:`\sigma : \mathrm{sym}(n \otimes b)`, :math:`b \perp n`.

        Returns
        -------
        uh : TensorLike
            写入边界值后的自由度向量.
        isDDof : TensorLike
            形状 ``(gdof,)`` 的布尔数组, 标记被写入的自由度.
        """
        return self._write_traction(gd, uh, threshold, tangential_only=True)

    def _write_traction(self, gd, uh, threshold, tangential_only: bool):
        """把 ``_traction_writes`` 的结果写入自由度向量并给出标记."""
        gdof = self.number_of_global_dofs()
        if uh is None:
            uh = bm.zeros((gdof,), dtype=self.ftype, device=self.device)
        dofs, values = self._traction_writes(gd, threshold, tangential_only)
        isDDof = bm.zeros((gdof,), dtype=bm.bool, device=self.device)
        if len(dofs) > 0:
            uh = bm.set_at(uh, dofs, values)
            isDDof = bm.set_at(isDDof, dofs, True)
        return uh, isDDof

    def geo_dimension(self):
        """几何维数."""
        return self.GD

    def top_dimension(self):
        """拓扑维数."""
        return self.TD

    def dof_frame(self) -> TensorLike:
        """顶点, 边, 面, 单元上的向量标架.

        各标架均为单位正交基, 与 2D 一致. 顶点与单元取笛卡尔基. 面标架为
        ``[n, t, n x t]``, 其中 ``n`` 为 ``face_unit_normal``, ``t`` 为该面第 0 条边的
        ``edge_unit_tangent``. 边标架为 ``[n, t x n, t]``, 其中 ``t`` 为
        ``edge_unit_tangent``, ``n`` 取某个相邻面的单位法向 (按 ``face_to_edge`` 写入,
        同一条边多次写入时以最后一次为准; 任取一个都与 ``t`` 正交). 边界边最后再按边界面
        写一次, 使其 ``n`` 取边界面法向: 牵引强施加要求边界上的标架向量与边界法向平行或
        正交, 见 ``boundary_interpolate``.

        Returns
        -------
        nframe : TensorLike
            形状 ``(NN, 3, 3)``.
        eframe : TensorLike
            形状 ``(NE, 3, 3)``.
        fframe : TensorLike
            形状 ``(NF, 3, 3)``.
        cframe : TensorLike
            形状 ``(NC, 3, 3)``.
        """
        mesh = self.mesh

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()

        nframe = bm.zeros((NN, 3, 3), dtype=mesh.ftype)
        eframe = bm.zeros((NE, 3, 3), dtype=mesh.ftype)
        fframe = bm.zeros((NF, 3, 3), dtype=mesh.ftype)
        cframe = bm.zeros((NC, 3, 3), dtype=mesh.ftype)

        f2e = mesh.face_to_edge()
        et  = mesh.edge_unit_tangent()
        fn  = mesh.face_unit_normal()

        node = mesh.entity('node')
        edge = mesh.entity('edge')
        cell = mesh.entity('cell')

        nframe[:] = bm.eye(3, dtype=mesh.ftype) 
        cframe[:] = bm.eye(3, dtype=mesh.ftype)

        fframe[:, 0] = fn
        fframe[:, 1] = et[f2e[:, 0]]
        fframe[:, 2] = bm.cross(fframe[:, 0], fframe[:, 1])

        eframe[f2e, 0] = fn[:, None] 
        isbdface = mesh.boundary_face_flag()
        eframe[f2e[isbdface], 0] = fn[isbdface, None]
        eframe[:, 1] = bm.cross(et, eframe[:, 0])
        eframe[:, 2] = et 

        return nframe, eframe, fframe, cframe

    def dof_frame_of_S(self):
        r"""由 ``dof_frame`` 张成的对称张量标架, 以 Voigt 分量 ``[xx, xy, xz, yy, yz, zz]`` 表示.

        对向量标架 :math:`(v_0, v_1, v_2)`, 第 ``i`` 个对称张量按
        ``bm.multi_index_matrix(2, 2)`` 的顺序依次为 :math:`v_0 \otimes v_0`,
        :math:`\mathrm{sym}(v_0 \otimes v_1)`, :math:`\mathrm{sym}(v_0 \otimes v_2)`,
        :math:`v_1 \otimes v_1`, :math:`\mathrm{sym}(v_1 \otimes v_2)`, :math:`v_2 \otimes v_2`.

        Returns
        -------
        nsframe : TensorLike
            形状 ``(NN, 6, 6)``, 第二维为张量编号, 第三维为 Voigt 分量.
        esframe : TensorLike
            形状 ``(NE, 6, 6)``.
        fsframe : TensorLike
            形状 ``(NF, 6, 6)``.
        csframe : TensorLike
            形状 ``(NC, 6, 6)``.
        """
        mesh = self.mesh

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()

        nframe, eframe, fframe, cframe = self.dof_frame()
        multiindex = bm.multi_index_matrix(2, 2)
        idx, num = symmetry_index(d=3, r=2)

        nsframe = bm.zeros((NN, 6, 6), dtype=self.ftype)
        for i, alpha in enumerate(multiindex): 
            nsframe[:, i] = symmetry_span_array(nframe, alpha).reshape(NN, -1)[:, idx]

        esframe = bm.zeros((NE, 6, 6), dtype=self.ftype)
        for i, alpha in enumerate(multiindex): 
            esframe[:, i] = symmetry_span_array(eframe, alpha).reshape(NE, -1)[:, idx]

        fsframe = bm.zeros((NF, 6, 6), dtype=self.ftype)
        for i, alpha in enumerate(multiindex): 
            fsframe[:, i] = symmetry_span_array(fframe, alpha).reshape(NF, -1)[:, idx]

        csframe = bm.zeros((NC, 6, 6), dtype=self.ftype)
        for i, alpha in enumerate(multiindex): 
            csframe[:, i] = symmetry_span_array(cframe, alpha).reshape(NC, -1)[:, idx]

        return nsframe, esframe, fsframe, csframe

    basis_frame = dof_frame

    # 基函数直接取自由度标架, 与 2D 一致: 自由度系数按 Voigt 重数解释,
    # 即 ``c_k = num_k * (sigma : S_k)``, ``num = [1, 2, 2, 1, 2, 1]``.
    basis_frame_of_S = dof_frame_of_S

    def basis(self, bc: TensorLike, index: Index=_S):
        """单元积分点处的基函数值, 取值为 Voigt 对称张量 ``[xx, xy, xz, yy, yz, zz]``.

        每个基函数为标量 Lagrange 基与 ``basis_frame_of_S`` 中对称张量的乘积. 局部
        基函数顺序与 ``cell_to_dof`` 一致: 顶点 -> 边上连续分量 -> 面上连续分量 ->
        边上断开分量 -> 面上断开分量 -> 单元泡函数 (仅 ``p >= 4``).

        Parameters
        ----------
        bc : TensorLike
            形状 ``(NQ, 4)`` 的重心坐标.
        index : Index, optional
            只支持全部单元; 单元实体与标架尚未按子集选取.

        Returns
        -------
        TensorLike
            形状 ``(NC, NQ, ldof, 6)``.

        Raises
        ------
        NotImplementedError
            ``index`` 不是全部单元.
        """
        _require_all_cells(index, "basis")
        p = self.p
        mesh = self.mesh
        dof = self.dof

        ldof = dof.number_of_local_dofs()

        ndofs = dof.cell_dofs.get_boundary_dof_from_dim(0)
        edofs = dof.cell_dofs.get_boundary_dof_from_dim(1)
        fdofs = dof.cell_dofs.get_boundary_dof_from_dim(2)

        iedofs = dof.cell_dofs.get_internal_dof_from_dim(1)
        ifdofs = dof.cell_dofs.get_internal_dof_from_dim(2)
        icdofs = dof.cell_dofs.get_internal_dof_from_dim(3)

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        cell = mesh.entity('cell')
        c2e = mesh.cell_to_edge()
        c2f = mesh.cell_to_face()

        nsframe, esframe, fsframe, csframe = self.basis_frame_of_S()

        phi_s = self.mesh.shape_function(bc, self.p, index=index) # (NC, NQ, ldof)

        NQ = bc.shape[0]
        phi = bm.zeros((NC, NQ, ldof, 6), dtype=self.ftype)

        # 顶点基函数
        idx = 0
        for v, vdof in enumerate(ndofs):
            N = len(vdof)
            scalar_phi_idx = multiindex_to_number(vdof.dof_scalar)
            scalar_part = phi_s[None, :, scalar_phi_idx, None]
            tensor_part = nsframe[cell[:, v]][:, None, vdof.dof_tensor, :]
            phi[..., idx:idx+N, :] = scalar_part * tensor_part
            idx += N

        # 边基函数
        for e, edof in enumerate(edofs):
            N = len(edof)
            scalar_phi_idx = multiindex_to_number(edof.dof_scalar)
            scalar_part = phi_s[None, :, scalar_phi_idx, None]
            tensor_part = esframe[c2e[:, e]][:, None, edof.dof_tensor, :]
            phi[..., idx:idx+N, :] = scalar_part * tensor_part
            idx += N

        for f, fdof in enumerate(fdofs):
            N = len(fdof)
            scalar_phi_idx = multiindex_to_number(fdof.dof_scalar)
            scalar_part = phi_s[None, :, scalar_phi_idx, None]
            tensor_part = fsframe[c2f[:, f]][:, None, fdof.dof_tensor, :]
            phi[..., idx:idx+N, :] = scalar_part * tensor_part
            idx += N

        # 单元基函数
        for e, edof in enumerate(iedofs):
            N = len(edof)
            scalar_phi_idx = multiindex_to_number(edof.dof_scalar)
            scalar_part = phi_s[None, :, scalar_phi_idx, None]
            tensor_part = esframe[c2e[:, e]][:, None, edof.dof_tensor, :]
            phi[..., idx:idx+N, :] = scalar_part * tensor_part
            idx += N

        for f, fdof in enumerate(ifdofs):
            N = len(fdof)
            scalar_phi_idx = multiindex_to_number(fdof.dof_scalar)
            scalar_part = phi_s[None, :, scalar_phi_idx, None]
            tensor_part = fsframe[c2f[:, f]][:, None, fdof.dof_tensor, :]
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

    def div_basis(self, bc: TensorLike): 
        r"""单元积分点处基函数的散度, 在全部单元上计算.

        第 ``i`` 个分量为 :math:`\sum_j \partial_j \sigma_{ij}`, 梯度取
        ``grad_shape_function(..., variables='x')`` 的物理导数. 局部基函数顺序同 ``basis``.

        Parameters
        ----------
        bc : TensorLike
            形状 ``(NQ, 4)`` 的重心坐标.

        Returns
        -------
        TensorLike
            形状 ``(NC, NQ, ldof, 3)``.
        """
        p = self.p
        mesh = self.mesh
        dof = self.dof

        ldof = dof.number_of_local_dofs()

        ndofs = dof.cell_dofs.get_boundary_dof_from_dim(0)
        edofs = dof.cell_dofs.get_boundary_dof_from_dim(1)
        fdofs = dof.cell_dofs.get_boundary_dof_from_dim(2)

        iedofs = dof.cell_dofs.get_internal_dof_from_dim(1)
        ifdofs = dof.cell_dofs.get_internal_dof_from_dim(2)
        icdofs = dof.cell_dofs.get_internal_dof_from_dim(3)

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        cell = mesh.entity('cell')
        c2e = mesh.cell_to_edge()
        c2f = mesh.cell_to_face()

        nsframe, esframe, fsframe, csframe = self.basis_frame_of_S() 

        gphi_s = self.mesh.grad_shape_function(bc, self.p, variables='x') # (NC, ldof, GD)

        NQ = bc.shape[0]
        dphi = bm.zeros((NC, NQ, ldof, 3), dtype=self.ftype)

        symidx = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
        # 顶点基函数
        idx = 0
        for v, vdof in enumerate(ndofs):
            N = len(vdof)
            scalar_phi_idx = multiindex_to_number(vdof.dof_scalar)
            grad_scalar = gphi_s[..., scalar_phi_idx, :] # (NC, NQ, N, 2)
            frame = nsframe[cell[:, v]][:, None, vdof.dof_tensor] # (NC, 1, N, 3)
            dphi[..., idx:idx+N, 0] = bm.sum(grad_scalar * frame[..., symidx[0]], axis=-1)
            dphi[..., idx:idx+N, 1] = bm.sum(grad_scalar * frame[..., symidx[1]], axis=-1)
            dphi[..., idx:idx+N, 2] = bm.sum(grad_scalar * frame[..., symidx[2]], axis=-1)
            idx += N

        # 边基函数
        for e, edof in enumerate(edofs):
            N = len(edof)
            scalar_phi_idx = multiindex_to_number(edof.dof_scalar)
            grad_scalar = gphi_s[..., scalar_phi_idx, :]
            frame = esframe[c2e[:, e]][:, None, edof.dof_tensor]
            dphi[..., idx:idx+N, 0] = bm.sum(grad_scalar * frame[..., symidx[0]], axis=-1)
            dphi[..., idx:idx+N, 1] = bm.sum(grad_scalar * frame[..., symidx[1]], axis=-1)
            dphi[..., idx:idx+N, 2] = bm.sum(grad_scalar * frame[..., symidx[2]], axis=-1)
            idx += N

        # 面基函数
        for f, fdof in enumerate(fdofs):
            N = len(fdof)
            scalar_phi_idx = multiindex_to_number(fdof.dof_scalar)
            grad_scalar = gphi_s[..., scalar_phi_idx, :]
            frame = fsframe[c2f[:, f]][:, None, fdof.dof_tensor]
            dphi[..., idx:idx+N, 0] = bm.sum(grad_scalar * frame[..., symidx[0]], axis=-1)
            dphi[..., idx:idx+N, 1] = bm.sum(grad_scalar * frame[..., symidx[1]], axis=-1)
            dphi[..., idx:idx+N, 2] = bm.sum(grad_scalar * frame[..., symidx[2]], axis=-1)
            idx += N

        # 单元基函数
        for e, edof in enumerate(iedofs):
            N = len(edof)
            scalar_phi_idx = multiindex_to_number(edof.dof_scalar)
            grad_scalar = gphi_s[..., scalar_phi_idx, :]
            frame = esframe[c2e[:, e]][:, None, edof.dof_tensor]
            dphi[..., idx:idx+N, 0] = bm.sum(grad_scalar * frame[..., symidx[0]], axis=-1)
            dphi[..., idx:idx+N, 1] = bm.sum(grad_scalar * frame[..., symidx[1]], axis=-1)
            dphi[..., idx:idx+N, 2] = bm.sum(grad_scalar * frame[..., symidx[2]], axis=-1)
            idx += N

        for f, fdof in enumerate(ifdofs):
            N = len(fdof)
            scalar_phi_idx = multiindex_to_number(fdof.dof_scalar)
            grad_scalar = gphi_s[..., scalar_phi_idx, :]
            frame = fsframe[c2f[:, f]][:, None, fdof.dof_tensor]
            dphi[..., idx:idx+N, 0] = bm.sum(grad_scalar * frame[..., symidx[0]], axis=-1)
            dphi[..., idx:idx+N, 1] = bm.sum(grad_scalar * frame[..., symidx[1]], axis=-1)
            dphi[..., idx:idx+N, 2] = bm.sum(grad_scalar * frame[..., symidx[2]], axis=-1)
            idx += N

        # 单元气泡基函数 - 只有当 p >= n+1 时才存在单元内部自由度
        n = mesh.geo_dimension()
        if p >= n + 1:
            scalar_phi_idx = multiindex_to_number(icdofs[0].dof_scalar)
            grad_scalar = gphi_s[..., scalar_phi_idx, :]
            frame = csframe[:, None, icdofs[0].dof_tensor]
            dphi[..., idx:, 0] = bm.sum(grad_scalar * frame[..., symidx[0]], axis=-1)
            dphi[..., idx:, 1] = bm.sum(grad_scalar * frame[..., symidx[1]], axis=-1)
            dphi[..., idx:, 2] = bm.sum(grad_scalar * frame[..., symidx[2]], axis=-1)

        return dphi

    @barycentric
    def value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike: 
        """有限元函数在积分点处的值, 形状 ``(..., NC, NQ, 6)``.

        ``index`` 只支持全部单元, 否则抛出 NotImplementedError.
        """
        _require_all_cells(index, "value")
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
        """有限元函数在积分点处的散度, 形状 ``(NC, NQ, 3)``.

        ``div_basis`` 总在全部单元上计算, 故 ``index`` 只支持全部单元, 否则抛出
        NotImplementedError.
        """
        _require_all_cells(index, "div_value")
        if isinstance(bc, tuple):
            TD = len(bc)
        else :
            TD = bc.shape[-1] - 1
        gphi = self.div_basis(bc)
        # TODO 目前只考虑散度值在单元上计算的情形
        e2dof = self.dof.cell_to_dof(index=index)
        val = bm.einsum('cilm, cl -> cim', gphi, uh[e2dof])

        return val
    

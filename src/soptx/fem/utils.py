"""多分辨率拓扑优化的有限元辅助函数.

包括把位移单元积分点映射到子密度单元, 计算子密度单元积分点处的形函数梯度与子单元
实体刚度, 以及位移单元布局 ``(NC, n_sub, ...)`` 与密度单元布局 ``(NC * n_sub, ...)``
之间的数据重排.
"""

from __future__ import annotations

from typing import Optional
import math

from soptx.backend import backend_manager as bm
from soptx.decorator import  cartesian
from soptx.typing import TensorLike, Tuple
from soptx.functionspace import TensorFunctionSpace, FunctionSpace

def map_bcs_to_sub_elements(bcs_e: Tuple[TensorLike, TensorLike], n_sub: int):
    """将位移单元的积分点的重心坐标映射成各个子密度单元的积分点的重心坐标.

    Parameters
    ----------
    bcs_e : 位移单元积分点的重心坐标, 结构为 ( (NQ, GD), (NQ, GD) ).
    n_sub : 子密度单元的总数.

    Returns
    -------
    bcs_g : 子密度单元积分点的重心坐标, 结构为
        ( (n_sub, NQ_x, GD), (n_sub, NQ_y, GD) ).

    Notes
    -----
    子密度单元的编号先列后行::

        +-------+-------+
        |   1   |   3   |
        +-------+-------+
        |   0   |   2   |
        +-------+-------+
    """
    if not isinstance(bcs_e, tuple):
        raise TypeError(
            "输入参数 'bcs_e' 必须是一个元组 (tuple),"
            "此函数仅适用于张量积网格."
        )

    bcs_xi, bcs_eta = bcs_e
    GD = bcs_xi.shape[1]

    sqrt_n_sub = math.sqrt(n_sub)
    if sqrt_n_sub != int(sqrt_n_sub):
        raise ValueError("子密度单元个数 'n_sub' 必须是一个完全平方数")
    
    n_sub_x = int(sqrt_n_sub)
    n_sub_y = int(sqrt_n_sub)

    p_1d = bm.unique(bcs_xi[:, 0])
    NQ_x = len(p_1d)
    NQ_y = len(p_1d)

    bcs_g_xi = bm.zeros((n_sub, NQ_x, GD))
    bcs_g_eta = bm.zeros((n_sub, NQ_y, GD))

    for i in range(n_sub_y):  
        for j in range(n_sub_x): 
            
            # 先列后行
            sub_element_idx = j * n_sub_y + i
            # # 先行后列
            # sub_element_idx = i * n_sub_x + j

            # 计算当前密度单元的区间范围 (假设父单元为 [0, 1] x [0, 1])
            xi_start, xi_end = j / n_sub_x, (j + 1) / n_sub_x
            eta_start, eta_end = i / n_sub_y, (i + 1) / n_sub_y
            
            # 线性映射
            mapped_xi = xi_start + p_1d * (xi_end - xi_start)
            mapped_eta = eta_start + p_1d * (eta_end - eta_start)
            
            # 构造重心坐标: 对于一维单纯形, 位置t的重心坐标是[1-t, t]
            bcs_g_xi[sub_element_idx, :, 0] = 1.0 - mapped_xi  # xi方向的重心坐标
            bcs_g_xi[sub_element_idx, :, 1] = mapped_xi
            
            bcs_g_eta[sub_element_idx, :, 0] = 1.0 - mapped_eta  # eta方向的重心坐标  
            bcs_g_eta[sub_element_idx, :, 1] = mapped_eta

    
    return (bcs_g_xi, bcs_g_eta)

def calculate_multiresolution_gphi_eg(
                            s_space_u: FunctionSpace,
                            *,
                            q: int,
                            n_sub: int,
                        ) -> TensorLike:
    """
    在多分辨率框架下, 计算父位移单元内部各子密度单元高斯点评估处的形函数梯度

    Note
    ----
    位移自由度仍来自父位移单元 (粗网格), 但应力/应变评估点取自子密度单元 (细网格),
    - 首先在父参考单元上生成高斯点, 然后将这些高斯点映射到各子单元的参考区域中,
    - 并把映射后的点仍用 "父参考单元坐标" 表达, 从而可直接调用父位移空间的 grad_basis.
    - 最终返回的 gphi_eg_reshaped 形状为 (NC*n_sub, NQ, LDOF, GD), 
    - 可直接传入 material.strain_matrix(...) 构造 B 矩阵.

    Parameters
    ----------
    s_space_u: 
        位移单元对应的标量函数空间 (父单元空间), 提供 `grad_basis` 和 `number_of_local_dofs`.
    q:        
        位移单元上用于生成基础高斯点的积分阶次
    n_sub:
        每个位移单元内的子密度单元数量

    Returns
    -------
    gphi_eg_reshaped: 
        展平后的形函数梯度数组, 形状 (NC*n_sub, NQ, LDOF, GD), 用于后续构造 B 矩阵.
    """
    mesh_u = s_space_u.mesh
    
    NC = mesh_u.number_of_cells()
    GD = mesh_u.geo_dimension()

    # 计算位移单元 (父参考单元) 高斯积分点处的重心坐标
    qf_e = mesh_u.quadrature_formula(q)
    bcs_e, ws_e = qf_e.get_quadrature_points_and_weights() # bcs_e - ( (NQ_x, GD), (NQ_y, GD) ), ws_e - (NQ, )
    NQ = ws_e.shape[0]

    # 把位移单元高斯积分点处的重心坐标映射到子密度单元 (子参考单元) 高斯积分点处的重心坐标 (仍表达在位移单元中)
    from soptx.fem.utils import map_bcs_to_sub_elements
    bcs_eg = map_bcs_to_sub_elements(bcs_e=bcs_e, n_sub=n_sub)
    bcs_eg_x, bcs_eg_y = bcs_eg

    # 在各子密度单元高斯积分点处计算位移单元的形函数梯度
    LDOF = s_space_u.number_of_local_dofs()
    gphi_eg = bm.zeros((NC, n_sub, NQ, LDOF, GD))  # (NC, n_sub, NQ, LDOF, GD)

    for s_idx in range(n_sub):
        sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])  # ((NQ_x, GD), (NQ_y, GD))
        gphi_sub = s_space_u.grad_basis(sub_bcs, variable='x')    # (NC, NQ, LDOF, GD)
        gphi_eg[:, s_idx, :, :, :] = gphi_sub

    # 展平为 (NC*n_sub, ...) 的多分辨率布局 
    gphi_eg_reshaped = reshape_multiresolution_data(mesh=mesh_u, data=gphi_eg)  # (NC*n_sub, NQ, LDOF, GD)

    return gphi_eg_reshaped

def multiresolution_sub_element_matrices(space: TensorFunctionSpace,
                                         material,
                                         *,
                                         q: int,
                                         n_sub: int,
                                    ) -> TensorLike:
    """各子密度单元上的实体刚度 K^0_{e,n} = ∫_{子单元 n} B^T D_0 B.

    Parameters
    ----------
    space : 位移张量函数空间 (父位移单元).
    material : 线弹性材料, 提供 ``strain_matrix`` 与实体本构 ``elastic_matrix``.
    q : 父参考单元上生成积分点的积分阶次.
    n_sub : 每个位移单元内的子密度单元数, 须为完全平方数.

    Returns
    -------
    ke0_sub : (NC, n_sub, TLDOF, TLDOF) 的子单元刚度, 对 n 求和即位移单元的 K_e^0.

    Notes
    -----
    只支持二维张量积网格 (``map_bcs_to_sub_elements`` 的限制). 积分点映射到子单元后仍用
    父参考单元坐标表达, 故形函数梯度与 ``|det J|`` 都取父单元的, 子单元面积占父单元的
    1 / n_sub, 由该因子补上. ``strain_matrix`` 对每个 (单元, 积分点) 独立计算, 因此这里
    直接按 (NC * n_sub, ...) 展平调用, 无需 ``reshape_multiresolution_data`` 的重排, 也不读
    ``meshdata``.
    """
    s_space = space.scalar_space
    mesh = space.mesh
    GD = mesh.geo_dimension()
    NC = mesh.number_of_cells()
    LDOF = s_space.number_of_local_dofs()

    qf_e = mesh.quadrature_formula(q)
    bcs_e, ws_e = qf_e.get_quadrature_points_and_weights()  # ws_e: (NQ, )
    bcs_eg_x, bcs_eg_y = map_bcs_to_sub_elements(bcs_e=bcs_e, n_sub=n_sub)
    NQ = ws_e.shape[0]

    gphi_eg = bm.zeros((NC, n_sub, NQ, LDOF, GD))
    detJ_eg = bm.zeros((NC, n_sub, NQ))
    for s_idx in range(n_sub):
        sub_bcs = (bcs_eg_x[s_idx], bcs_eg_y[s_idx])
        gphi_eg[:, s_idx] = s_space.grad_basis(sub_bcs, variable='x')   # (NC, NQ, LDOF, GD)
        J_sub = mesh.entity_view('cell').jacobi_matrix(sub_bcs)         # (NC, NQ, GD, GD)
        detJ_eg[:, s_idx] = bm.abs(bm.linalg.det(J_sub))                # (NC, NQ)

    B_flat = material.strain_matrix(dof_priority=space.dof_priority,
                                    gphi=gphi_eg.reshape(NC * n_sub, NQ, LDOF, GD))  # (NC*n_sub, NQ, NS, TLDOF)
    B_eg = B_flat.reshape(NC, n_sub, *B_flat.shape[1:])                  # (NC, n_sub, NQ, NS, TLDOF)

    J_g = 1.0 / n_sub
    D0 = material.elastic_matrix()[0, 0]  # (NS, NS)

    return J_g * bm.einsum('q, cnq, cnqki, kl, cnqlj -> cnij', ws_e, detJ_eg, B_eg, D0, B_eg)

def reshape_multiresolution_data(mesh, data: TensorLike) -> TensorLike:
    """将多分辨率数据从位移单元布局映射到密度单元布局.

    Parameters
    ----------
    mesh : 位移网格对象.
    data : (NC, n_sub, ...) 的位移单元布局数据.

    Returns
    -------
    data_reordered : (NC * n_sub, ...) 的密度单元布局数据.
    """
    original_shape = data.shape
    NC, n_sub = original_shape[0], original_shape[1]
    extra_dims = original_shape[2:]
    sub_dim = int(bm.sqrt(n_sub))

    nx = mesh.meshdata['nx']
    ny = mesh.meshdata['ny']

    is_full_rect = (NC == nx * ny)

    if is_full_rect:
        # 完整矩形网格: 直接按列优先编号生成 col/row, 无需计算 cell_centers
        cols = bm.arange(NC) // ny
        rows = bm.arange(NC) % ny
        pos_to_local = None  # 完整矩形无需查找表
    else:
        # 非完整区域 (如 L 型): 需要从几何位置计算 cell_positions
        hx = mesh.meshdata['hx']
        hy = mesh.meshdata['hy']
        cell_centers = mesh.entity_barycenter('cell')  # (NC, 2)
        eps = 1e-10
        cols = bm.floor(cell_centers[:, 0] / hx + eps).astype(int)
        rows = bm.floor(cell_centers[:, 1] / hy + eps).astype(int)
        pos_to_local = {}
        for c in range(NC):
            pos_to_local[(int(cols[c]), int(rows[c]))] = c

    reorder_indices = []
    for pos_col in range(nx):
        for sub_row in range(sub_dim):
            for pos_row in range(ny):
                if is_full_rect:
                    c = pos_col * ny + pos_row
                    for sub_col in range(sub_dim):
                        reorder_indices.append(c * n_sub + sub_row * sub_dim + sub_col)
                else:
                    key = (pos_col, pos_row)
                    if key not in pos_to_local:
                        continue
                    c = pos_to_local[key]
                    for sub_col in range(sub_dim):
                        reorder_indices.append(c * n_sub + sub_row * sub_dim + sub_col)

    reorder_indices = bm.array(reorder_indices)

    assert len(reorder_indices) == NC * n_sub, (
        f"重排索引数量 {len(reorder_indices)} 与预期 {NC * n_sub} 不符，"
        f"请检查网格数据是否正确（可能存在浮点精度问题）"
    )

    data_reshaped = data.reshape(NC * n_sub, *extra_dims)
    return data_reshaped[reorder_indices]

def reshape_multiresolution_data_inverse(mesh, data_flat: TensorLike, n_sub: int) -> TensorLike:
    """将多分辨率数据从密度单元布局映射回位移单元布局.

    Parameters
    ----------
    mesh : 位移网格对象.
    data_flat : (NC * n_sub, ...) 的密度单元布局数据.
    n_sub : 每个位移单元的子密度单元数量.

    Returns
    -------
    data_restored : (NC, n_sub, ...) 的位移单元布局数据.
    """
    original_shape = data_flat.shape
    extra_dims = original_shape[1:]
    sub_dim = int(bm.sqrt(n_sub))

    nx = mesh.meshdata['nx']
    ny = mesh.meshdata['ny']

    is_full_rect = (data_flat.shape[0] // n_sub == nx * ny)

    if is_full_rect:
        NC = nx * ny
        pos_to_local = None
    else:
        hx = mesh.meshdata['hx']
        hy = mesh.meshdata['hy']
        cell_centers = mesh.entity_barycenter('cell')  # (NC, 2)
        NC = cell_centers.shape[0]
        eps = 1e-10
        cols = bm.floor(cell_centers[:, 0] / hx + eps).astype(int)
        rows = bm.floor(cell_centers[:, 1] / hy + eps).astype(int)
        pos_to_local = {}
        for c in range(NC):
            pos_to_local[(int(cols[c]), int(rows[c]))] = c

    inverse_indices = bm.zeros(NC * n_sub, dtype=bm.int32)

    idx = 0
    for pos_col in range(nx):
        for sub_row in range(sub_dim):
            for pos_row in range(ny):
                if is_full_rect:
                    c = pos_col * ny + pos_row
                    for sub_col in range(sub_dim):
                        s = sub_row * sub_dim + sub_col
                        inverse_indices[c * n_sub + s] = idx
                        idx += 1
                else:
                    key = (pos_col, pos_row)
                    if key not in pos_to_local:
                        continue
                    c = pos_to_local[key]
                    for sub_col in range(sub_dim):
                        s = sub_row * sub_dim + sub_col
                        inverse_indices[c * n_sub + s] = idx
                        idx += 1

    assert idx == NC * n_sub, (
        f"逆映射索引计数 {idx} 与预期 {NC * n_sub} 不符，"
        f"请检查网格数据是否正确"
    )

    data_restored_flat = data_flat[inverse_indices]
    return data_restored_flat.reshape(NC, n_sub, *extra_dims)

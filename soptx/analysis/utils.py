from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.decorator import  cartesian
from fealpy.typing import TensorLike, Tuple
from fealpy.functionspace import TensorFunctionSpace, Function, FunctionSpace
from fealpy.mesh import TriangleMesh, QuadrangleMesh

from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.model.pde_base import PDEBase 

def _scalar_disp_to_tensor_disp(dof_priority: bool,
                            uh: Function, 
                            uh0: Function, uh1: Function, uh2: Function = None
                        ) -> Function:
    """将标量位移转换为张量位移"""
    if uh2 is None:
        if dof_priority:
            uh[:] = bm.stack((uh0, uh1), axis=-1).T.flatten()
        else:
            uh[:] = bm.stack((uh0, uh1), axis=-1).flatten()
    else:
        if dof_priority:
            uh[:] = bm.stack((uh0, uh1, uh2), axis=-1).T.flatten()
        else:
            uh[:] = bm.stack((uh0, uh1, uh2), axis=-1).flatten()

    return uh

def project_solution_to_finer_mesh(pde: PDEBase,
                                nx: int, ny: int,
                                uh: Function,
                                lfa: LagrangeFEMAnalyzer,
                                source_refinement_level,
                                target_mesh: TriangleMesh
                            ) -> Function:
    # 重新生成源网格
    mesh = pde.init_mesh(nx=nx, ny=ny)
    for _ in range(source_refinement_level):
        mesh.bisect(isMarkedCell=None)

    # 只要当前网格的单元数还少于目标网格，就继续加密和插值
    while mesh.number_of_cells() < target_mesh.number_of_cells():
        sspace = lfa.get_scalar_space_from_mesh(mesh=mesh)
        scell2dof = sspace.cell_to_dof()
        tspace = lfa.get_tensor_space_from_scalar_space(scalar_space=sspace)
        uh_dof_component = _get_displacement_dof_component(uh=uh, space=tspace)

        uh0c2f = uh_dof_component[..., 0][scell2dof]
        uh1c2f = uh_dof_component[..., 1][scell2dof]
        data = {'uh0c2f': uh0c2f, 'uh1c2f': uh1c2f}

        options = mesh.bisect_options(data=data, disp=False)
        mesh.bisect(isMarkedCell=None, options=options)

        # 在新的、更细的网格上构建插值解
        sspace = lfa.get_scalar_space_from_mesh(mesh=mesh)
        tspace = lfa.get_tensor_space_from_scalar_space(scalar_space=sspace)
        uh_new = tspace.function()
        uh0_new = sspace.function()
        uh1_new = sspace.function()
        scell2dof_new = sspace.cell_to_dof()
        uh0_new[scell2dof_new.reshape(-1)] = options['data']['uh0c2f'].reshape(-1)
        uh1_new[scell2dof_new.reshape(-1)] = options['data']['uh1c2f'].reshape(-1)

        uh = _scalar_disp_to_tensor_disp(dof_priority=True, 
                                            uh=uh_new, uh0=uh0_new, uh1=uh1_new)
    
    return uh

def map_bcs_to_sub_elements(bcs_e: Tuple[TensorLike, TensorLike], n_sub: int):
    """将位移单元的积分点的重心坐标映射成各个子密度单元的积分点的重心坐标
    
    Parameters:
    -----------
    bcs_e : 位移单元积分点的重心坐标, 结构为 ( (NQ, GD), (NQ, GD) )
    n_sub : 子密度单元的总数
    子密度的顺序 (先列后行)
    +-------+-------+
    |   1   |   3   |  
    +-------+-------+
    |   0   |   2   |  
    +-------+-------+

    Returns:
    --------
    bcs_g : 子密度单元积分点的重心坐标, 结构为 ( (n_sub, NQ_x, GD), (n_sub, NQ_y, GD) )
    """
    if not isinstance(bcs_e, tuple):
        raise TypeError(
            "输入参数 'bcs_e' 必须是一个元组 (tuple),"
            "此函数仅适用于张量积网格."
        )

    bcs_xi, bcs_eta = bcs_e
    GD = bcs_xi.shape[1]

    sqrt_n_sub = bm.sqrt(n_sub)
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
            
            # 构造重心坐标：对于一维单纯形，位置t的重心坐标是[1-t, t]
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
    在多分辨率框架下，计算父位移单元内部各子密度单元高斯点评估处的形函数梯度

    Note
    ----
    位移自由度仍来自父位移单元 (粗网格)，但应力/应变评估点取自子密度单元 (细网格),
    - 首先在父参考单元上生成高斯点, 然后将这些高斯点映射到各子单元的参考区域中，
    - 并把映射后的点仍用 “父参考单元坐标” 表达，从而可直接调用父位移空间的 grad_basis.
    - 最终返回的 gphi_eg_reshaped 形状为 (NC*n_sub, NQ, LDOF, GD), 
    - 可直接传入 material.strain_displacement_matrix(...) 构造 B 矩阵.

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
        展平后的形函数梯度数组, 形状 (NC*n_sub, NQ, LDOF, GD), 用于后续构造 B 矩阵。
    """
    mesh_u = s_space_u.mesh
    
    NC = mesh_u.number_of_cells()
    GD = mesh_u.geo_dimension()

    # 计算位移单元 (父参考单元) 高斯积分点处的重心坐标
    qf_e = mesh_u.quadrature_formula(q)
    bcs_e, ws_e = qf_e.get_quadrature_points_and_weights() # bcs_e - ( (NQ_x, GD), (NQ_y, GD) ), ws_e - (NQ, )
    NQ = ws_e.shape[0]

    # 把位移单元高斯积分点处的重心坐标映射到子密度单元 (子参考单元) 高斯积分点处的重心坐标 (仍表达在位移单元中)
    from soptx.analysis.utils import map_bcs_to_sub_elements
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
    nx_u, ny_u = mesh_u.meshdata["nx"], mesh_u.meshdata["ny"]
    gphi_eg_reshaped = reshape_multiresolution_data(nx=nx_u, ny=ny_u, data=gphi_eg)  # (NC*n_sub, NQ, LDOF, GD)

    return gphi_eg_reshaped

def reshape_multiresolution_data(nx: int, ny: int, 
                                data: TensorLike,
                                cell_positions: Optional[TensorLike] = None
                            ) -> TensorLike:
    """
    Parameters:
    -----------
    nx, ny   : 位移单元在 x、y 方向的数量（完整矩形网格尺寸，仅用于确定 sub_dim 等）
    data     : (NC, n_sub, ...)
    cell_positions : (NC, 2)，每个实际单元在完整矩形网格中的 (col, row) 坐标，
                     即 col ∈ [0, nx), row ∈ [0, ny)
                     若为 None, 则退化为完整矩形网格
    """
    original_shape = data.shape
    NC, n_sub = original_shape[0], original_shape[1]
    extra_dims = original_shape[2:]
    sub_dim = int(bm.sqrt(n_sub))

    if cell_positions is None:
        # 完整矩形网格：生成默认 (col, row) 坐标，列优先编号
        cols = bm.arange(NC) // ny
        rows = bm.arange(NC) % ny
    else:
        cols = cell_positions[:, 0]  # (NC,)
        rows = cell_positions[:, 1]  # (NC,)

    # 构建从 (col, row) 到实际单元局部索引 c 的查找表
    pos_to_local = {}
    for c in range(NC):
        key = (int(cols[c]), int(rows[c]))
        pos_to_local[key] = c

    # 按密度单元的空间顺序生成重排索引
    reorder_indices = []
    for pos_col in range(nx):
        for sub_row in range(sub_dim):
            for pos_row in range(ny):
                key = (pos_col, pos_row)
                if key not in pos_to_local:
                    continue  # L 型区域中不存在的单元，跳过
                c = pos_to_local[key]
                for sub_col in range(sub_dim):
                    s = sub_row * sub_dim + sub_col
                    data_idx = c * n_sub + s
                    reorder_indices.append(data_idx)

    reorder_indices = bm.array(reorder_indices)

    # 验证重排索引数量与数据规模一致
    assert len(reorder_indices) == NC * n_sub, (
        f"重排索引数量 {len(reorder_indices)} 与预期 {NC * n_sub} 不符，"
        f"请检查 cell_positions 是否正确（可能存在浮点精度问题）"
    )

    # 重塑数据：(NC, n_sub, ...) -> (NC * n_sub, ...)
    data_reshaped = data.reshape(NC * n_sub, *extra_dims)

    # 按照空间位置重排
    data_reordered = data_reshaped[reorder_indices]

    return data_reordered

def reshape_multiresolution_data_inverse(nx: int, ny: int, 
                                         data_flat: TensorLike, 
                                         n_sub: int,
                                         cell_positions: Optional[TensorLike] = None) -> TensorLike:
    """
    将多分辨率数据从密度单元布局映射到位移单元布局

    将 (NC*n_sub, ...) 形状的数据重新排列为 (NC, n_sub, ...) 形状,
    其中数据按照空间位置顺序重新排列。支持非完整矩形域（如 L 型区域）。

    Parameters:
    -----------
    nx, ny         : 位移单元在 x、y 方向的数量（完整矩形网格尺寸）
    data_flat      : (NC*n_sub, ...)
    n_sub          : 每个位移单元的子密度单元数量
    cell_positions : (NC, 2)，每个实际单元在完整矩形网格中的 (col, row) 坐标。
                     若为 None，则退化为完整矩形网格（原有逻辑）。

    Returns:
    --------
    data_restored : (NC, n_sub, ...)
    """
    original_shape = data_flat.shape
    extra_dims = original_shape[1:]
    sub_dim = int(bm.sqrt(n_sub))

    if cell_positions is None:
        NC = nx * ny
        cols = bm.arange(NC) // ny
        rows = bm.arange(NC) % ny
    else:
        NC = cell_positions.shape[0]
        cols = cell_positions[:, 0]
        rows = cell_positions[:, 1]

    # 构建从 (col, row) 到实际单元局部索引 c 的查找表
    pos_to_local = {}
    for c in range(NC):
        key = (int(cols[c]), int(rows[c]))
        pos_to_local[key] = c

    # 创建逆映射索引
    inverse_indices = bm.zeros(NC * n_sub, dtype=bm.int32)

    idx = 0
    for pos_col in range(nx):
        for sub_row in range(sub_dim):
            for pos_row in range(ny):
                key = (pos_col, pos_row)
                if key not in pos_to_local:
                    continue  # 非矩形区域中不存在的单元，跳过
                c = pos_to_local[key]
                for sub_col in range(sub_dim):
                    s = sub_row * sub_dim + sub_col
                    data_idx = c * n_sub + s
                    inverse_indices[data_idx] = idx
                    idx += 1

    # 验证 idx 与数据规模一致
    assert idx == NC * n_sub, (
        f"逆映射索引计数 {idx} 与预期 {NC * n_sub} 不符，"
        f"请检查 cell_positions 是否正确"
    )

    # 按逆映射重排数据
    data_restored_flat = data_flat[inverse_indices]

    # 重塑为 (NC, n_sub, ...)
    data_restored = data_restored_flat.reshape(NC, n_sub, *extra_dims)

    return data_restored

def reshape_multiresolution_data_bcakup(nx: int, ny: int, data: TensorLike) -> TensorLike:
    """
    将多分辨率数据从位移单元布局映射到密度单元布局
    
    将 (NC, n_sub, ...) 形状的数据重新排列为 (NC*n_sub, ...) 形状，
    其中数据按照空间位置顺序重新排列。

    Parameters:
    -----------
    nx: 位移单元在 x 方向的数量
    ny: 位移单元在 y 方向的数量  
    data : (NC, n_sub, ...)

    Returns:
    --------
    data_reordered : (NC * n_sub, ...)
    """
    original_shape = data.shape
    NC, n_sub = original_shape[0], original_shape[1]
    extra_dims = original_shape[2:]

    sub_dim = int(bm.sqrt(n_sub))
    
    # 获取重排索引
    reorder_indices = []

    # 按列优先遍历位移单元，对于每列的位移单元，按子单元行优先排列
    for pos_col in range(nx):  # 对于每一列的位移单元
        for sub_row in range(sub_dim):  # 对于子单元的每一行
            for pos_row in range(ny):  # 对于该列中的每个位移单元
                for sub_col in range(sub_dim):  # 对于子单元的每一列
                    # 计算位移单元索引（按列优先编号）
                    c = pos_col * ny + pos_row
                    # 计算子单元索引
                    s = sub_row * sub_dim + sub_col
                    # 计算在展平数据中的索引
                    data_idx = c * n_sub + s
                    reorder_indices.append(data_idx)
    
    reorder_indices = bm.array(reorder_indices)

    # 重塑数据：(NC, n_sub, ...) -> (NC * n_sub, ...)
    data_reshaped = data.reshape(NC * n_sub, *extra_dims)
    
    # 按照空间位置重排
    data_reordered = data_reshaped[reorder_indices]

    return data_reordered

def reshape_multiresolution_data_inverse_backup(nx: int, ny: int, data_flat: TensorLike, n_sub: int) -> TensorLike:
    """
    将多分辨率数据从密度单元布局映射到位移单元布局

    将 (NC*n_sub, ...) 形状的数据重新排列为 (NC, n_sub, ...) 形状,
    其中数据按照空间位置顺序重新排列

    Parameters:
    -----------
    nx: 位移单元在 x 方向的数量
    ny: 位移单元在 y 方向的数量  
    data_flat : (NC*n_sub, ...)
    n_sub : 每个位移单元的子密度单元数量

    Returns:
    --------
    data_reordered : (NC, n_sub, ...)
    """
    NC = nx * ny
    original_shape = data_flat.shape
    extra_dims = original_shape[1:]
    
    sub_dim = int(bm.sqrt(n_sub))
    
    # 创建逆映射索引
    inverse_indices = bm.zeros(NC * n_sub, dtype=bm.int32)
    
    idx = 0
    for pos_col in range(nx):  # 对于每一列的位移单元
        for sub_row in range(sub_dim):  # 对于子单元的每一行
            for pos_row in range(ny):  # 对于该列中的每个位移单元
                for sub_col in range(sub_dim):  # 对于子单元的每一列
                    # 计算位移单元索引（按列优先编号）
                    c = pos_col * ny + pos_row
                    # 计算子单元索引
                    s = sub_row * sub_dim + sub_col
                    # 计算在原始数据中的索引
                    data_idx = c * n_sub + s
                    inverse_indices[data_idx] = idx
                    idx += 1
    
    # 按逆映射重排数据
    data_restored_flat = data_flat[inverse_indices]
    
    # 重塑为 (NC, n_sub, ...)
    data_restored = data_restored_flat.reshape(NC, n_sub, *extra_dims)
    
    return data_restored
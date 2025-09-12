from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Tuple
from fealpy.functionspace import TensorFunctionSpace, Function
from fealpy.mesh import TriangleMesh

from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.model.pde_base import PDEBase 

def _get_displacement_dof_component(uh: Function, space: TensorFunctionSpace) -> TensorLike:
    """获取位移自由度分量形式"""
    shape = space.shape
    scalar_space = space.scalar_space
    mesh = space.mesh
    gdof = scalar_space.number_of_global_dofs()
    GD = mesh.geo_dimension()

    if shape[1] == -1: # dof_priority
        uh_reshaped = uh.reshape(GD, gdof)  
        return uh_reshaped.T
    
    elif shape[1] == GD: # gd_priority
        return uh.reshape(GD, gdof)

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


def reshape_multiresolution_data(nx: int, ny: int, data: TensorLike) -> TensorLike:
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
    for u_row in range(ny):  
        for sub_row in range(sub_dim):   
            for u_col in range(nx):  
                for sub_col in range(sub_dim):  
                    c = u_row * nx + u_col
                    s = sub_row * sub_dim + sub_col
                    data_idx = c * n_sub + s
                    reorder_indices.append(data_idx)
    
    reorder_indices = bm.array(reorder_indices)

    # 重塑数据：(NC, n_sub, ...) -> (NC * n_sub, ...)
    data_reshaped = data.reshape(NC * n_sub, *extra_dims)
    
    # 按照空间位置重排
    data_reordered = data_reshaped[reorder_indices]

    return data_reordered

def reshape_multiresolution_data_inverse(nx: int, ny: int, data_flat: TensorLike, n_sub: int) -> TensorLike:
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
    sub_dim = int(bm.sqrt(n_sub))

    flat_shape = data_flat.shape
    extra_dims = flat_shape[1:] 

    # 重新计算 reorder_indices（与正向函数完全相同的逻辑）
    reorder_indices = []
    for u_row in range(ny):  
        for sub_row in range(sub_dim):   
            for u_col in range(nx):  
                for sub_col in range(sub_dim):  
                    c = u_row * nx + u_col
                    s = sub_row * sub_dim + sub_col
                    data_idx = c * n_sub + s
                    reorder_indices.append(data_idx)
    
    reorder_indices = bm.array(reorder_indices)
    
    # 创建逆映射索引
    inverse_indices = bm.zeros(NC * n_sub, dtype=bm.int32)
    for new_pos, old_pos in enumerate(reorder_indices):
        inverse_indices[old_pos] = new_pos
    
    # 应用逆映射
    data_reordered = data_flat[inverse_indices]
    
    # 重塑为多分辨率格式：(NC*n_sub, ...) -> (NC, n_sub, ...)
    return data_reordered.reshape(NC, n_sub, *extra_dims)


if __name__ == "__main__":
    # 测试代码
    # from fealpy.mesh import QuadrangleMesh
    # mesh_u = QuadrangleMesh.from_box(box=[0, 1, 0, 1], nx=1, ny=1)
    # q = 2
    # # 计算位移单元积分点处的重心坐标
    # qf_e = mesh_u.quadrature_formula(q)
    # # bcs_e.shape = ( (NQ, GD), (NQ, GD) ), ws_e.shape = (NQ, )
    # bcs_e, ws_e = qf_e.get_quadrature_points_and_weights()

    # n_sub = 4
    # bcs_eg = map_bcs_to_sub_elements(bcs_e=bcs_e, n_sub=n_sub)
    # bcs_eg_x, bcs_eg_y = bcs_eg[0], bcs_eg[1]

    # from fealpy.decorator import  cartesian
    # @cartesian
    # def test_func(points):
    #     x = points[:, 0]
    #     y = points[:, 1]
    #     return x**2 + y**2
    
    # from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
    # s_space = LagrangeFESpace(mesh=mesh_u, p=1)
    # t_space = TensorFunctionSpace(scalar_space=s_space, shape=(-1, 2))

    # ps_e = mesh_u.bc_to_point(bcs_e)
    # f_e = test_func(ps_e[0])
    # J = mesh_u.jacobi_matrix(bcs_e)
    # detJ = bm.abs(bm.linalg.det(J))

    # K1 = bm.einsum('q, q, cq -> c', f_e, ws_e, detJ)

    # NC = 1
    # NQ = ws_e.shape[0]
    # f_eg = bm.zeros((n_sub, NQ)) # (n_sub, NQ)
    # detJ_eg = bm.zeros((NC, n_sub, NQ)) # (NC, n_sub, NQ)
    # for s_idx in range(n_sub):
    #     sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])  # ((q, GD), (q, GD))
    #     # J_q = 1/4
    #     J_sub = mesh_u.jacobi_matrix(sub_bcs) # (NC, NQ, GD, GD)
    #     detJ_sub = bm.abs(bm.linalg.det(J_sub)) # (NC, NQ)
    #     detJ_eg[:, s_idx, :] = detJ_sub

    #     sub_ps = mesh_u.bc_to_point(sub_bcs) # (NC, NQ, GD)
    #     f_eg[s_idx, :] = test_func(sub_ps[0])  # (NQ, )

    # KK1 = bm.einsum('nq, q, cnq -> cn', f_eg, ws_e, detJ_eg / 4)
    # KK1sum = KK1.sum()


    nx, ny = 3, 2
    NC, n_sub = nx*ny,  9
    test_grad = bm.arange(NC * n_sub).reshape(NC, n_sub)
    reshaped = reshape_multiresolution_data(nx=nx, ny=ny, data=test_grad)
    reshaped_test = test_grad.reshape(NC, n_sub)
    reshaped2 = reshape_multiresolution_data_inverse(nx=nx, ny=ny, data_flat=reshaped, n_sub=n_sub)


    print("--------------")
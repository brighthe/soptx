from typing import Optional, Tuple, Union, Dict, Any

from fealpy.backend import backend_manager as bm
from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace import FunctionSpace
from fealpy.typing import TensorLike

def reshape_multiresolution_data_inverse(nx: int, ny: int, data_flat: TensorLike, n_sub: int) -> TensorLike:
    """
    将多分辨率数据从密度单元布局映射到位移单元布局

    将 (NC*n_sub, ...) 形状的数据重新排列为 (NC, n_sub, ...) 形状,
    其中数据按照空间位置顺序重新排列

    Parameters
    ----------
    nx: 位移单元在 x 方向的数量
    ny: 位移单元在 y 方向的数量  
    data_flat : (NC*n_sub, ...)
    n_sub : 每个位移单元的子密度单元数量

    Returns
    -------
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
    from soptx.analysis.utils import reshape_multiresolution_data
    nx_u, ny_u = mesh_u.meshdata["nx"], mesh_u.meshdata["ny"]
    gphi_eg_reshaped = reshape_multiresolution_data(nx=nx_u, ny=ny_u, data=gphi_eg)  # (NC*n_sub, NQ, LDOF, GD)

    return gphi_eg_reshaped

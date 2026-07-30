from typing import Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

def get_gauss_integration_point_mapping_old(nx: int, ny: int, nq_per_dim: int = 3):
    """局部高斯积分点和全局高斯积分点的映射"""
    NC = nx * ny
    NQ = nq_per_dim * nq_per_dim
    
    local_to_global = bm.zeros(NC * NQ, dtype=bm.int32)
    
    global_idx = 0
    
    # 按照transpose后的维度顺序遍历
    for ix in range(nx):                    # 网格x方向
        for iq_x in range(nq_per_dim):      # 积分点x方向（列）
            for iy in range(ny):            # 网格y方向
                for iq_y in range(nq_per_dim):  # 积分点y方向（行）
                    
                    # 单元索引（列优先存储）
                    cell_idx = iy + ix * ny
                    
                    # 单元内积分点索引（行优先）
                    point_idx = iq_x * nq_per_dim + iq_y
                    
                    # 局部索引
                    local_idx = cell_idx * NQ + point_idx
                    
                    # 全局位置global_idx对应局部位置local_idx
                    local_to_global[global_idx] = local_idx
                    global_idx += 1
    
    global_to_local = bm.argsort(local_to_global)
    
    return local_to_global, global_to_local

def get_gauss_integration_point_mapping(nx: int, ny: int, nq_per_dim: int = 3) -> Tuple[Tuple[TensorLike, TensorLike], TensorLike]:
    """局部高斯积分点和全局高斯积分点的映射"""
    NC = nx * ny
    NQ = nq_per_dim * nq_per_dim
    
    # 创建映射关系：全局位置 -> 局部位置
    global_pos_to_local_pos = bm.zeros(NC * NQ, dtype=bm.int32)
    
    global_idx = 0
    
    # 按照transpose后的维度顺序遍历
    for ix in range(nx):                    # 网格x方向
        for iq_x in range(nq_per_dim):      # 积分点x方向（列）
            for iy in range(ny):            # 网格y方向
                for iq_y in range(nq_per_dim):  # 积分点y方向（行）
                    
                    # 单元索引（列优先存储）
                    cell_idx = iy + ix * ny
                    
                    # 单元内积分点索引（行优先）
                    point_idx = iq_x * nq_per_dim + iq_y
                    
                    # 局部索引
                    local_idx = cell_idx * NQ + point_idx
                    
                    # 全局位置 global_idx 对应局部位置local_idx
                    global_pos_to_local_pos[global_idx] = local_idx
                    global_idx += 1
    
    # 构造 local_to_global 用于局部到全局转换
    # global_pos_to_local_pos[i] 是全局位置 i 对应的局部一维索引
    # 我们需要将这个一维索引转换为二维索引 (cell_idx, point_idx)
    cell_indices = global_pos_to_local_pos // NQ   # 单元索引
    point_indices = global_pos_to_local_pos % NQ   # 积分点索引
    local_to_global = (cell_indices, point_indices)

    # 构造 global_to_local 用于全局到局部转换  
    # 我们需要从全局一维数组重排为局部二维数组
    global_to_local = global_pos_to_local_pos.reshape(NC, NQ)
    
    return local_to_global, global_to_local